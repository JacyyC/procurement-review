import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { act } from 'react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App, { RUN_POLL_INTERVAL_MS } from './App'
import {
  FIXTURE_CHECKSUM as CHECKSUM,
  FIXTURE_RUN_ID as RUN_ID,
  TOKEN,
  operatorSession,
  readerSession,
  resumeLessSession,
  reviewRun as run,
} from './runFixtures'
import type {
  AuditResponse,
  CaseListResponse,
  RecoveryInfo,
  ReviewRun,
  SessionInfo,
  TraceResponse,
} from './types'

const cases: CaseListResponse = {
  api_version: 'review-api.v2',
  execution_mode: 'test_only',
  cases: [
    {
      api_version: 'review-api.v2',
      case_id: 'CASE-LOW-001',
      case_version: 'case-low-001.v1',
      label: '低风险完整证据',
      case_type: 'purchase_order',
      scenario: 'low',
      document_version: 'retrieval-fixture-v1',
      dataset_version: 'synthetic-procurement.v1',
      index_version: 'retrieval-index-v1',
      rules_version: '2.0.0',
    },
    {
      api_version: 'review-api.v2',
      case_id: 'CASE-ALL-FAILURE-001',
      case_version: 'case-all-failure-001.v1',
      label: '全部 Worker 失败',
      case_type: 'purchase_order',
      scenario: 'all_failure',
      document_version: 'retrieval-fixture-v1',
      dataset_version: 'synthetic-procurement.v1',
      index_version: 'retrieval-index-v1',
      rules_version: '2.0.0',
    },
    {
      api_version: 'review-api.v2',
      case_id: 'CASE-CANCEL-001',
      case_version: 'case-cancel-001.v1',
      label: '处理中取消',
      case_type: 'purchase_order',
      scenario: 'cancellable',
      document_version: 'retrieval-fixture-v1',
      dataset_version: 'synthetic-procurement.v1',
      index_version: 'retrieval-index-v1',
      rules_version: '2.0.0',
    },
  ],
}

function interruptedRun(
  recovery: Partial<RecoveryInfo> = {},
  overrides: Partial<ReviewRun> = {},
): ReviewRun {
  const base = run('interrupted')
  return {
    ...base,
    version: 5,
    revision: 5,
    ...overrides,
    recovery: { ...(base.recovery as RecoveryInfo), ...recovery },
  }
}

const trace: TraceResponse = {
  api_version: 'review-api.v1',
  trace: {
    schema_version: '1.0',
    trace_schema_version: 'trace.v1',
    trace_id: 'trace-001',
    run_id: RUN_ID,
    complete: false,
    events: [
      {
        schema_version: '1.0',
        event_id: 'event-run-001-0001',
        run_id: RUN_ID,
        trace_id: 'trace-001',
        event_type: 'run_started',
        step: 'review_run',
        occurred_at: '2026-08-31T11:59:59Z',
        input_summary: { case_id: 'CASE-LOW-001' },
        output_summary: {},
        source_versions: { api: 'review-api.v2' },
        duration_ms: 0,
        token_count: 0,
        cost_amount: 0,
        cost_currency: 'CNY',
        redaction_status: 'not_required',
        error: null,
      },
    ],
  },
}

const emptyAudit: AuditResponse = {
  api_version: 'review-api.v2',
  audit_schema_version: 'approval-audit.v2',
  run_id: RUN_ID,
  events: [],
}

function response(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function errorResponse(
  status: number,
  code: string,
  message = 'denied',
  details: Record<string, unknown> = {},
): Response {
  return response(
    { error: { code, message, request_id: 'request-test', details } },
    status,
  )
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((settle) => {
    resolve = settle
  })
  return { promise, resolve }
}

interface RecordedCall {
  url: string
  method: string
  body: unknown
  headers: Headers
  signal: AbortSignal | null | undefined
}

interface HarnessOptions {
  session?: SessionInfo
  initialRuns?: ReviewRun[]
  completed?: ReviewRun
  actionResult?: ReviewRun
  /** Hold the run-list response until the test releases it. */
  holdRunList?: boolean
  /** Make the first confirm/reject/cancel call fail with this API error. */
  actionErrorOnce?: { status: number; code: string; message: string }
  /** Serve a custom approval-audit payload. */
  audit?: AuditResponse
  /** Answer the first POST /resume with this run (202) or this API error. */
  resumeResult?: ReviewRun
  resumeErrorOnce?: {
    status: number
    code: string
    message: string
    details?: Record<string, unknown>
  }
  /** Successive GET /reviews/{id} payloads, consumed one per read. */
  runSequence?: ReviewRun[]
  /** Make the first GET /reviews/{id} fail once, e.g. a storage outage mid-poll. */
  getErrorOnce?: { status: number; code: string; message?: string }
}

function installReviewFetch(options: HarnessOptions = {}) {
  const session = options.session ?? operatorSession
  const calls: RecordedCall[] = []
  const store = new Map<string, ReviewRun>()
  for (const item of options.initialRuns ?? []) store.set(item.run_id, item)
  const heldRunList = deferred<Response>()
  let heldRunListUsed = false
  let credentialValid = true
  let pollCount = 0

  const fetchMock = vi.fn(
    async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input)
      const method = (init?.method ?? 'GET').toUpperCase()
      const headers = new Headers(init?.headers)
      let body: unknown = null
      if (typeof init?.body === 'string') body = JSON.parse(init.body)
      calls.push({ url, method, body, headers, signal: init?.signal })

      if (!url.startsWith('/api/v2/')) {
        // The retired v1 surface must never be reached from the workbench.
        return errorResponse(410, 'api_version_retired', 'review-api.v1 is retired')
      }
      if (
        !credentialValid ||
        !headers.get('Authorization')?.startsWith('Bearer ')
      ) {
        return errorResponse(401, 'unauthenticated', 'Credential is unusable')
      }
      if (url === '/api/v2/session') return response(session)
      if (url === '/api/v2/cases') return response(cases)
      if (url === '/api/v2/reviews' && method === 'GET') {
        if (options.holdRunList && !heldRunListUsed) {
          heldRunListUsed = true
          return heldRunList.promise
        }
        return response({
          api_version: 'review-api.v2',
          runs: [...store.values()],
        })
      }
      if (url === '/api/v2/reviews' && method === 'POST') {
        // The real server always echoes the Case the caller pinned in the body, so the
        // fixture does the same: a response for another Case is a bad payload (FIX-06C).
        const requestedCase = (body as { case_id?: string } | null)?.case_id
        const created = run('processing', requestedCase)
        store.set(created.run_id, created)
        return response(created, 202)
      }
      const target = url.match(/^\/api\/v2\/reviews\/([^/]+)(\/.*)?$/)
      if (target) {
        const runId = target[1]
        const suffix = target[2] ?? ''
        if (suffix === '' && method === 'GET') {
          if (options.getErrorOnce) {
            const failure = options.getErrorOnce
            options.getErrorOnce = undefined
            return errorResponse(
              failure.status,
              failure.code,
              failure.message ?? 'The local store is unavailable.',
            )
          }
          if (options.runSequence && options.runSequence.length > 0) {
            const next = options.runSequence.shift() as ReviewRun
            store.set(runId, next)
            return response(next)
          }
          const current = store.get(runId) ?? run('processing')
          if (current.status === 'processing') {
            pollCount += 1
            if (pollCount >= 1) {
              const completed = options.completed ?? run('pending_confirmation')
              store.set(runId, completed)
              return response(completed)
            }
          }
          return response(current)
        }
        if (suffix === '/resume' && method === 'POST') {
          if (options.resumeErrorOnce) {
            const failure = options.resumeErrorOnce
            options.resumeErrorOnce = undefined
            return errorResponse(
              failure.status,
              failure.code,
              failure.message,
              failure.details,
            )
          }
          const next = options.resumeResult ?? run('processing')
          store.set(runId, next)
          return response(next, 202)
        }
        if (suffix === '/trace') return response(trace)
        if (suffix === '/audit') return response(options.audit ?? emptyAudit)
        if (
          ['/confirm', '/reject', '/cancel'].includes(suffix) &&
          method === 'POST'
        ) {
          if (options.actionErrorOnce) {
            const failure = options.actionErrorOnce
            options.actionErrorOnce = undefined
            return errorResponse(failure.status, failure.code, failure.message)
          }
          const next = options.actionResult ?? run('confirmed')
          store.set(runId, next)
          return response(next)
        }
      }
      throw new Error(`Unhandled fetch: ${method} ${url}`)
    },
  )

  vi.stubGlobal('fetch', fetchMock)
  return {
    fetchMock,
    calls,
    store,
    releaseRunList: () => heldRunList.resolve(
      response({ api_version: 'review-api.v2', runs: [...store.values()] }),
    ),
    expireCredential: () => {
      credentialValid = false
    },
    callsTo: (pattern: RegExp, method = 'POST') =>
      calls.filter((call) => call.method === method && pattern.test(call.url)),
  }
}

async function signIn(user: ReturnType<typeof userEvent.setup>, token = TOKEN) {
  await user.type(screen.getByLabelText('Bearer 凭据'), token)
  await user.click(screen.getByRole('button', { name: '验证凭据' }))
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  window.localStorage.clear()
  window.sessionStorage.clear()
})

describe('Review workbench v2 credential gate', () => {
  it('sends no business request before a credential is submitted', async () => {
    const harness = installReviewFetch()
    render(<App />)

    expect(screen.getByText('需要本地凭据')).toBeInTheDocument()
    // The profile is a fact only the authenticated API can state: before a credential is
    // submitted the header says so instead of claiming the test-only default.
    expect(screen.getByText('MODE UNKNOWN')).toBeInTheDocument()
    expect(screen.getByText('建议不是审批事实')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '运行审查' })).not.toBeInTheDocument()
    expect(harness.fetchMock).not.toHaveBeenCalled()
  })

  it('verifies the credential and shows the server identity with sorted scopes', async () => {
    const harness = installReviewFetch()
    const user = userEvent.setup()
    render(<App />)

    const field = screen.getByLabelText('Bearer 凭据')
    expect(field).toHaveAttribute('type', 'password')
    await user.click(screen.getByRole('button', { name: '显示凭据' }))
    expect(screen.getByLabelText('Bearer 凭据')).toHaveAttribute('type', 'text')
    await user.click(screen.getByRole('button', { name: '隐藏凭据' }))
    expect(screen.getByLabelText('Bearer 凭据')).toHaveAttribute('type', 'password')

    await signIn(user)

    expect(await screen.findByText('已验证身份')).toBeInTheDocument()
    expect(screen.getByTestId('identity-subject')).toHaveTextContent('subject-local-demo')
    expect(screen.getByTestId('identity-tenant')).toHaveTextContent('tenant-local-demo')
    expect(screen.getByTestId('identity-scopes')).toHaveTextContent(
      'audit:read、case:read、review:cancel、review:create、review:decide、review:read、review:resume、trace:read',
    )
    expect(screen.getAllByText(/cred-operator-0001/).length).toBeGreaterThan(0)
    expect(screen.getByLabelText('已授权案例')).toHaveValue('CASE-LOW-001')

    // The raw credential is dropped from the DOM and only the masked indicator stays.
    expect(screen.queryByLabelText('Bearer 凭据')).not.toBeInTheDocument()
    expect(document.body.textContent).not.toContain(TOKEN)
    expect(harness.calls[0].url).toBe('/api/v2/session')
    expect(harness.calls[0].headers.get('Authorization')).toBe(`Bearer ${TOKEN}`)

    // The free-text actor input is gone: the actor is the verified subject.
    expect(screen.queryByLabelText('操作人')).not.toBeInTheDocument()
  })

  it('hides 运行审查 without review:create', async () => {
    const harness = installReviewFetch({ session: readerSession })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)

    expect(await screen.findByText('已验证身份')).toBeInTheDocument()
    expect(screen.getByTestId('identity-subject')).toHaveTextContent(
      'subject-local-demo-reader',
    )
    expect(screen.queryByRole('button', { name: '运行审查' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '重放最近创建请求' })).not.toBeInTheDocument()
    expect(screen.getByText('缺少 review:create 授权，运行审查已隐藏。')).toBeInTheDocument()
    const hidden = screen.getByTestId('hidden-controls')
    expect(hidden).toHaveTextContent('review:create')
    expect(hidden).toHaveTextContent('review:decide')
    expect(hidden).toHaveTextContent('review:cancel')
    expect(hidden).toHaveTextContent('trace:read')
    expect(hidden).toHaveTextContent('audit:read')
    expect(harness.callsTo(/^\/api\/v2\/reviews$/, 'POST')).toHaveLength(0)
  })

  it('hides the decide, cancel, trace and audit controls without their scopes', async () => {
    const harness = installReviewFetch({
      session: readerSession,
      initialRuns: [run('pending_confirmation')],
    })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)

    expect(await screen.findByRole('heading', { name: '建议通过' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '确认建议（本地）' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '拒绝建议（本地）' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '取消' })).not.toBeInTheDocument()
    expect(screen.queryByRole('tab', { name: /Trace/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('tab', { name: /审批审计/ })).not.toBeInTheDocument()
    expect(screen.getByText('缺少 review:decide 授权，确认/拒绝已隐藏。')).toBeInTheDocument()
    expect(screen.getByText('缺少 review:cancel 授权，取消已隐藏。')).toBeInTheDocument()
    expect(screen.getByText(/缺少 trace:read 授权/)).toBeInTheDocument()
    expect(screen.getByText(/缺少 audit:read 授权/)).toBeInTheDocument()
    expect(harness.callsTo(/\/audit$/, 'GET')).toHaveLength(0)
    expect(harness.callsTo(/\/trace$/, 'GET')).toHaveLength(0)
  })

  it('clears every artifact when a later call returns 401', async () => {
    const harness = installReviewFetch({ initialRuns: [run('pending_confirmation')] })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)

    expect(await screen.findByRole('heading', { name: '建议通过' })).toBeInTheDocument()
    expect(screen.getAllByText('CASE-LOW-001').length).toBeGreaterThan(0)
    expect(screen.getByTestId('identity-subject')).toBeInTheDocument()

    harness.expireCredential()
    await user.click(screen.getByRole('button', { name: '刷新当前 Review' }))

    expect(
      await screen.findByText('凭据无效或已失效，请重新输入'),
    ).toBeInTheDocument()
    expect(screen.queryByTestId('identity-subject')).not.toBeInTheDocument()
    expect(screen.queryAllByText('CASE-LOW-001')).toHaveLength(0)
    expect(screen.queryByRole('heading', { name: '建议通过' })).not.toBeInTheDocument()
    expect(screen.getByText('需要本地凭据')).toBeInTheDocument()
    expect(screen.getByText('验证凭据后读取运行列表。')).toBeInTheDocument()
  })

  it('never applies a response that arrives after the identity changed', async () => {
    const harness = installReviewFetch({ holdRunList: true })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)

    expect(await screen.findByText('已验证身份')).toBeInTheDocument()
    await waitFor(() =>
      expect(harness.callsTo(/^\/api\/v2\/reviews$/, 'GET')).toHaveLength(1),
    )

    await user.click(screen.getByRole('button', { name: '身份与权限' })); await user.click(screen.getByRole('button', { name: '退出并清除' }))
    expect(screen.getByText('需要本地凭据')).toBeInTheDocument()

    // The previous identity's late response carries a run; it must be discarded.
    harness.store.set(RUN_ID, run('pending_confirmation'))
    harness.releaseRunList()

    await waitFor(() => expect(screen.getByText('需要本地凭据')).toBeInTheDocument())
    expect(screen.queryAllByText('CASE-LOW-001')).toHaveLength(0)
    expect(screen.queryByRole('heading', { name: '建议通过' })).not.toBeInTheDocument()
  })

  it('writes the credential to no browser storage, cookie or URL', async () => {
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem')
    installReviewFetch()
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)
    await screen.findByText('已验证身份')

    expect(setItemSpy).not.toHaveBeenCalled()
    expect(window.localStorage.length).toBe(0)
    expect(window.sessionStorage.length).toBe(0)
    expect(document.cookie).toBe('')
    expect(window.location.search).toBe('')
    for (let index = 0; index < window.localStorage.length; index += 1) {
      expect(window.localStorage.key(index)).toBeNull()
    }
  })
})

describe('Review workbench v2 review flow', () => {
  it('renders the suggestion, evidence, facts and an empty audit', async () => {
    installReviewFetch()
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)
    await screen.findByText('尚未创建审查')

    await user.click(screen.getByRole('button', { name: '运行审查' }))
    expect(await screen.findByText('三个只读 Worker 正在并行审查')).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: '建议通过' })).toBeInTheDocument()
    expect(screen.getByText('政策阈值证据完整。')).toBeInTheDocument()
    expect(screen.getByText('fixture-competition')).toBeInTheDocument()
    expect(screen.getAllByText('PO-001').length).toBeGreaterThan(0)
    expect(screen.getByText('主体 subject-local-demo')).toBeInTheDocument()
    expect(screen.queryByText('本地模拟回执')).not.toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: /审批审计/ }))
    expect(
      await screen.findByText('尚无人工动作，审计记录与本地回执均为空。'),
    ).toBeInTheDocument()
  })

  it('creates a review with case_version and never echoes a reserved authority field', async () => {
    const harness = installReviewFetch()
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)
    await screen.findByText('尚未创建审查')
    await user.click(screen.getByRole('button', { name: '运行审查' }))
    await screen.findByRole('heading', { name: '建议通过' })

    const creates = harness.callsTo(/^\/api\/v2\/reviews$/, 'POST')
    expect(creates).toHaveLength(1)
    expect(creates[0].body).toEqual({
      api_version: 'review-api.v2',
      case_id: 'CASE-LOW-001',
      case_version: 'case-low-001.v1',
    })
    expect(creates[0].headers.get('Idempotency-Key')).toBeTruthy()
    for (const reserved of [
      'initiator',
      'human_actor',
      'actor',
      'scenario',
      'purchase_order_id',
      'query',
      'tenant_id',
      'subject_id',
    ]) {
      expect(creates[0].body).not.toHaveProperty(reserved)
    }
  })

  it('confirms with the displayed version and checksum, then replays the exact payload', async () => {
    const harness = installReviewFetch({ actionResult: run('confirmed') })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)
    await screen.findByText('尚未创建审查')
    await user.click(screen.getByRole('button', { name: '运行审查' }))
    await screen.findByRole('heading', { name: '建议通过' })

    await user.click(screen.getByRole('button', { name: '确认建议（本地）' }))
    const receipt = await screen.findByRole('heading', { name: '本地模拟回执' })
    const panel = receipt.closest('section')!
    expect(within(panel).getByText('0 · 未执行')).toBeInTheDocument()
    expect(
      screen.getByText('Gate 已关闭；新动作会被拒绝，仅可幂等重放原动作。'),
    ).toBeInTheDocument()

    let confirms = harness.callsTo(/\/confirm$/)
    expect(confirms).toHaveLength(1)
    expect(confirms[0].body).toEqual({
      api_version: 'review-api.v2',
      expected_version: 2,
      expected_summary_checksum: CHECKSUM,
    })
    expect(confirms[0].headers.get('Idempotency-Key')).toBeTruthy()
    expect(confirms[0].body).not.toHaveProperty('human_actor')

    const firstKey = confirms[0].headers.get('Idempotency-Key')
    await user.click(screen.getByRole('button', { name: '重放原动作' }))
    expect(
      await screen.findByText('人工动作已幂等重放：回执与审计记录均未增加。'),
    ).toBeInTheDocument()

    confirms = harness.callsTo(/\/confirm$/)
    expect(confirms).toHaveLength(2)
    expect(confirms[1].headers.get('Idempotency-Key')).toBe(firstKey)
    expect(confirms[1].body).toEqual(confirms[0].body)
  })

  it('offers no replay for a run loaded from the server and shows its key digest', async () => {
    const harness = installReviewFetch({ initialRuns: [run('confirmed')] })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)

    expect(await screen.findByRole('heading', { name: '本地模拟回执' })).toBeInTheDocument()
    // The v2 projection returns only a key digest, so the original key cannot be
    // reconstructed and no payload may be fabricated for the replay.
    expect(screen.getByTitle(`sha256:${'9'.repeat(64)}`)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '重放原动作' })).toBeDisabled()
    expect(
      screen.getByText(
        '幂等重放需要本次会话保留的原始幂等键；服务端只返回幂等键摘要，刷新页面或重新验证凭据后无法重放。',
      ),
    ).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '重放原动作' }))
    expect(harness.callsTo(/\/confirm$/)).toHaveLength(0)
  })

  it('shows the audit trail with key digests instead of raw keys', async () => {
    const auditWithEvents: AuditResponse = {
      api_version: 'review-api.v2',
      audit_schema_version: 'approval-audit.v2',
      run_id: RUN_ID,
      events: [
        {
          schema_version: '1.0',
          audit_schema_version: 'approval-audit.v2',
          audit_id: 'audit-001',
          sequence: 1,
          tenant_id: 'tenant-local-demo',
          case_id: 'CASE-LOW-001',
          run_id: RUN_ID,
          event_class: 'approval',
          action_type: 'confirm',
          actor: 'subject-local-demo',
          authorization_revision: 1,
          previous_status: 'pending_confirmation',
          new_status: 'confirmed',
          previous_revision: 2,
          new_revision: 3,
          result_checksum: CHECKSUM,
          idempotency_digest: `sha256:${'7'.repeat(64)}`,
          receipt_id: 'receipt-001',
          api_version: 'review-api.v2',
          occurred_at: '2026-08-31T12:00:00Z',
          detail_code: null,
          simulation_only: true,
          external_write_performed: false,
        },
      ],
    }
    installReviewFetch({ initialRuns: [run('confirmed')], audit: auditWithEvents })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)
    await screen.findByRole('heading', { name: '本地模拟回执' })

    await user.click(screen.getByRole('tab', { name: /审批审计/ }))
    expect(await screen.findByText('confirm', { exact: true })).toBeInTheDocument()
    expect(screen.getByTitle(`sha256:${'7'.repeat(64)}`)).toBeInTheDocument()
    expect(screen.getByText('幂等键摘要')).toBeInTheDocument()
    expect(screen.getAllByText('subject-local-demo').length).toBeGreaterThan(1)
  })

  it('renders a lifecycle audit event without an action, actor or digest', async () => {
    const lifecycleAudit: AuditResponse = {
      api_version: 'review-api.v2',
      audit_schema_version: 'approval-audit.v2',
      run_id: RUN_ID,
      events: [
        {
          schema_version: '1.0',
          audit_schema_version: 'approval-audit.v2',
          audit_id: 'audit-lifecycle-001',
          sequence: 1,
          tenant_id: 'tenant-local-demo',
          case_id: 'CASE-LOW-001',
          run_id: RUN_ID,
          event_class: 'lifecycle',
          action_type: null,
          actor: null,
          authorization_revision: null,
          previous_status: 'processing',
          new_status: 'interrupted',
          previous_revision: 1,
          new_revision: 2,
          result_checksum: null,
          idempotency_digest: null,
          receipt_id: null,
          api_version: 'review-api.v2',
          occurred_at: '2026-08-31T12:00:00Z',
          detail_code: 'lease_expired',
          simulation_only: true,
          external_write_performed: false,
        },
      ],
    }
    installReviewFetch({
      initialRuns: [interruptedRun()],
      audit: lifecycleAudit,
    })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)
    await screen.findByTestId('interrupted-state')

    await user.click(screen.getByRole('tab', { name: /审批审计/ }))
    expect(await screen.findByText('lifecycle', { exact: true })).toBeInTheDocument()
    // A lifecycle event carries no actor and no idempotency key: the view says so
    // instead of printing "undefined".
    expect(screen.getByText('（系统事件）')).toBeInTheDocument()
    expect(screen.getByText('无（该事件没有幂等键）')).toBeInTheDocument()
  })

  it('cancels without sending a summary checksum', async () => {
    const harness = installReviewFetch({ actionResult: run('cancelled') })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)
    await screen.findByText('尚未创建审查')
    await user.click(screen.getByRole('button', { name: '运行审查' }))
    await screen.findByRole('heading', { name: '建议通过' })

    await user.click(screen.getByRole('button', { name: '取消' }))
    expect(
      await screen.findByText('Review 已取消；没有生成本地模拟回执。'),
    ).toBeInTheDocument()

    const cancels = harness.callsTo(/\/cancel$/)
    expect(cancels).toHaveLength(1)
    expect(cancels[0].body).toEqual({
      api_version: 'review-api.v2',
      expected_version: 2,
    })
  })

  it('never calls the retired /api/v1 surface during the whole flow', async () => {
    const harness = installReviewFetch({ actionResult: run('confirmed') })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)
    await screen.findByText('尚未创建审查')
    await user.click(screen.getByRole('button', { name: '运行审查' }))
    await screen.findByRole('heading', { name: '建议通过' })
    await user.click(screen.getByRole('button', { name: '确认建议（本地）' }))
    await screen.findByRole('heading', { name: '本地模拟回执' })
    await user.click(screen.getByRole('tab', { name: /Trace/ }))
    await screen.findByRole('heading', { name: '运行 Trace' })
    await user.click(screen.getByRole('tab', { name: /审批审计/ }))
    await screen.findByRole('heading', { name: '审批审计' })

    expect(harness.calls.length).toBeGreaterThan(5)
    expect(harness.calls.every((call) => call.url.startsWith('/api/v2/'))).toBe(true)
    expect(harness.calls.some((call) => call.url.includes('/api/v1'))).toBe(false)
  })

  it('shows all worker failures and never presents them as a normal suggestion', async () => {
    const failedWorkers = run('pending_human_review', 'CASE-ALL-FAILURE-001')
    failedWorkers.summary!.recommendation = 'insufficient_information'
    failedWorkers.summary!.highest_risk = 'unknown'
    failedWorkers.summary!.missing_information = ['policy evidence', 'contract evidence']
    failedWorkers.summary!.failure_reasons = ['all worker branches failed']
    failedWorkers.branch_failures = [
      {
        api_version: 'review-api.v1',
        worker: 'policy',
        code: 'model_failure',
        detail: 'model fixture ID is not declared',
      },
      {
        api_version: 'review-api.v1',
        worker: 'contract',
        code: 'model_failure',
        detail: 'model fixture ID is not declared',
      },
      {
        api_version: 'review-api.v1',
        worker: 'procurement_data',
        code: 'model_failure',
        detail: 'model fixture ID is not declared',
      },
    ]
    installReviewFetch({ completed: failedWorkers })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)
    await screen.findByText('尚未创建审查')
    await user.selectOptions(screen.getByLabelText('已授权案例'), 'CASE-ALL-FAILURE-001')
    await user.click(screen.getByRole('button', { name: '运行审查' }))

    expect(await screen.findByRole('heading', { name: '信息不足' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Worker 失败' })).toBeInTheDocument()
    expect(screen.getAllByText('model_failure')).toHaveLength(3)
    expect(screen.queryByText('本地模拟回执')).not.toBeInTheDocument()
  })

  it('renders a closed runtime failure without a receipt', async () => {
    installReviewFetch({ completed: run('failed') })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)
    await screen.findByText('尚未创建审查')
    await user.click(screen.getByRole('button', { name: '运行审查' }))

    expect(
      await screen.findByRole('heading', { name: 'Review 运行失败并已关闭' }),
    ).toBeInTheDocument()
    expect(screen.getByText('review_execution_failed')).toBeInTheDocument()
    expect(screen.queryByText('本地模拟回执')).not.toBeInTheDocument()
  })

  it('surfaces a version conflict without retrying the human decision', async () => {
    const harness = installReviewFetch({
      actionErrorOnce: {
        status: 409,
        code: 'version_conflict',
        message: 'The review version is stale.',
      },
    })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)
    await screen.findByText('尚未创建审查')
    await user.click(screen.getByRole('button', { name: '运行审查' }))
    await screen.findByRole('heading', { name: '建议通过' })

    await user.click(screen.getByRole('button', { name: '确认建议（本地）' }))

    expect(
      await screen.findByText('版本已过期，已重新读取最新审查状态。'),
    ).toBeInTheDocument()
    // The human decision is never auto-retried.
    expect(harness.callsTo(/\/confirm$/)).toHaveLength(1)
  })
})

describe('Review workbench v2 interrupted runs and explicit resume', () => {
  it('renders interrupted as its own state with no confirm, reject or receipt', async () => {
    const harness = installReviewFetch({ initialRuns: [interruptedRun()] })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)

    expect(await screen.findByTestId('interrupted-state')).toBeInTheDocument()
    expect(screen.getAllByText('已中断 · 等待恢复').length).toBeGreaterThan(0)
    expect(
      screen.getByRole('heading', { name: '运行已中断，等待恢复' }),
    ).toBeInTheDocument()
    // No recommendation, so no decision control and no receipt may exist.
    expect(screen.queryByRole('heading', { name: '建议通过' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '确认建议（本地）' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '拒绝建议（本地）' })).not.toBeInTheDocument()
    expect(screen.getByTestId('interrupted-gate-note')).toBeInTheDocument()
    expect(screen.queryByText('本地模拟回执')).not.toBeInTheDocument()

    // The reason, the budget and the adopted branches are all operator-visible.
    expect(screen.getByTestId('recovery-reason')).toHaveTextContent('上次执行租约已失效')
    expect(screen.getByTestId('recovery-budget')).toHaveTextContent('恢复 0 / 2')
    expect(screen.getByTestId('recovery-adopted')).toHaveTextContent('合同条款')
    expect(screen.getByTestId('recovery-adopted')).toHaveTextContent('采购数据')
    expect(screen.getByTestId('recovery-budget-chip')).toHaveTextContent('恢复 0 / 2')

    // Cancelling an interrupted run stays available; nothing is auto-decided.
    expect(screen.getByRole('button', { name: '取消' })).toBeEnabled()
    expect(screen.queryByRole('button', { name: '重放原动作' })).not.toBeInTheDocument()
    expect(harness.callsTo(/\/confirm$/)).toHaveLength(0)
    expect(harness.callsTo(/\/resume$/)).toHaveLength(0)
  })

  it('hides 恢复审查 without review:resume and never calls the endpoint', async () => {
    const harness = installReviewFetch({
      session: resumeLessSession,
      initialRuns: [interruptedRun()],
    })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)

    expect(await screen.findByTestId('interrupted-state')).toBeInTheDocument()
    expect(screen.queryByTestId('resume-button')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '恢复审查' })).not.toBeInTheDocument()
    expect(
      screen.getByText('缺少 review:resume 授权，恢复审查已隐藏。'),
    ).toBeInTheDocument()
    expect(screen.getByTestId('hidden-controls')).toHaveTextContent('review:resume')
    expect(harness.callsTo(/\/resume$/)).toHaveLength(0)
  })

  it('keeps cancel available but disables resume when the budget is exhausted', async () => {
    const harness = installReviewFetch({
      initialRuns: [interruptedRun({ resume_count: 2, resume_limit: 2, resumable: false })],
    })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)

    expect(await screen.findByTestId('recovery-budget')).toHaveTextContent('恢复 2 / 2')
    expect(screen.getByTestId('recovery-reason')).toHaveTextContent('恢复预算已用尽')
    expect(screen.getByTestId('recovery-next-step')).toHaveTextContent(
      '创建同一 Case 的新审查（不复制旧回执）',
    )
    expect(screen.getByRole('button', { name: '恢复审查' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '取消' })).toBeEnabled()

    await user.click(screen.getByRole('button', { name: '恢复审查' }))
    expect(harness.callsTo(/\/resume$/)).toHaveLength(0)
  })

  it('explains an untrusted checkpoint as needing a new review', async () => {
    installReviewFetch({
      initialRuns: [
        interruptedRun(
          { resumable: false, interrupted_reason: 'checkpoint_incompatible' },
          { error_code: 'checkpoint_incompatible' },
        ),
      ],
    })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)

    expect(await screen.findByTestId('recovery-reason')).toHaveTextContent(
      '检查点不可信或版本不符',
    )
    expect(screen.getByTestId('recovery-next-step')).toHaveTextContent(
      '检查点不可信或版本不符，需新建审查。',
    )
    expect(screen.getByRole('button', { name: '恢复审查' })).toBeDisabled()
  })

  it('resumes with the displayed version and a fresh key, then polls to a recommendation', async () => {
    const resumedProcessing = { ...run('processing'), version: 6, revision: 6 }
    const resumedPending: ReviewRun = {
      ...run('pending_confirmation'),
      version: 7,
      revision: 7,
      recovery: {
        ...(run('pending_confirmation').recovery as RecoveryInfo),
        resume_count: 1,
        generation: 2,
      },
    }
    const harness = installReviewFetch({
      initialRuns: [interruptedRun()],
      resumeResult: resumedProcessing,
      runSequence: [resumedProcessing, resumedPending],
    })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)

    const resume = await screen.findByTestId('resume-button')
    expect(resume).toBeEnabled()
    // Nothing fires before the operator clicks.
    expect(harness.callsTo(/\/resume$/)).toHaveLength(0)

    await user.click(resume)

    expect(await screen.findByRole('heading', { name: '建议通过' })).toBeInTheDocument()
    const resumes = harness.callsTo(/\/resume$/)
    expect(resumes).toHaveLength(1)
    expect(resumes[0].body).toEqual({
      api_version: 'review-api.v2',
      expected_version: 5,
    })
    expect(resumes[0].headers.get('Idempotency-Key')).toBeTruthy()
    expect(resumes[0].body).not.toHaveProperty('expected_summary_checksum')

    // The resumed run was polled while processing and now shows its consumed budget.
    expect(
      harness.callsTo(/^\/api\/v2\/reviews\/run-001$/, 'GET').length,
    ).toBeGreaterThanOrEqual(2)
    expect(screen.getByTestId('recovery-budget-chip')).toHaveTextContent('恢复 1 / 2')
  })

  it('keeps polling a resumed run and shows the updated budget when it stops again', async () => {
    const resumedProcessing = { ...run('processing'), version: 6, revision: 6 }
    const interruptedAgain = interruptedRun(
      { resume_count: 1, resume_limit: 2, generation: 2 },
      { version: 7, revision: 7 },
    )
    const harness = installReviewFetch({
      initialRuns: [interruptedRun()],
      resumeResult: resumedProcessing,
      runSequence: [resumedProcessing, interruptedAgain],
    })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)

    await user.click(await screen.findByTestId('resume-button'))

    await waitFor(() =>
      expect(screen.getByTestId('recovery-budget')).toHaveTextContent('恢复 1 / 2'),
    )
    expect(screen.getByTestId('interrupted-state')).toBeInTheDocument()
    expect(screen.getByTestId('recovery-budget-chip')).toHaveTextContent('恢复 1 / 2')
    expect(
      harness.callsTo(/^\/api\/v2\/reviews\/run-001$/, 'GET').length,
    ).toBeGreaterThanOrEqual(2)
    // The budget was consumed but the run is still resumable: no new review needed yet.
    expect(screen.getByRole('button', { name: '恢复审查' })).toBeEnabled()
  })

  it('surfaces a 409 review_not_recoverable with new-review guidance and never retries', async () => {
    const exhausted = {
      // A later state carries a higher revision (verified against the live API).
      ...run('failed'),
      version: 6,
      revision: 6,
      error_code: 'resume_budget_exhausted',
      error_message: 'The review exhausted its resume budget.',
    }
    const harness = installReviewFetch({
      initialRuns: [interruptedRun({ resume_count: 1, resume_limit: 2, generation: 2 })],
      resumeErrorOnce: {
        status: 409,
        code: 'review_not_recoverable',
        message: 'This review exhausted its resume budget.',
        details: { resume_count: 2 },
      },
      runSequence: [exhausted],
    })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)

    await user.click(await screen.findByTestId('resume-button'))

    expect(
      await screen.findByText(/创建同一 Case 的新审查（不复制旧回执）/),
    ).toBeInTheDocument()
    // A rejected resume is never auto-retried, and the server's own move to failed
    // is what the operator sees next.
    expect(harness.callsTo(/\/resume$/)).toHaveLength(1)
    expect(
      await screen.findByRole('heading', { name: 'Review 运行失败并已关闭' }),
    ).toBeInTheDocument()
  })

  it('keeps polling through a transient storage failure instead of freezing at 处理中', async () => {
    const resumedProcessing = { ...run('processing'), version: 6, revision: 6 }
    const harness = installReviewFetch({
      initialRuns: [interruptedRun()],
      resumeResult: resumedProcessing,
      // The storage outage answers one poll with a bare 500 before the Run finishes.
      getErrorOnce: { status: 500, code: 'unknown_error' },
      // Each later read carries a higher revision, exactly as the service does.
      runSequence: [
        resumedProcessing,
        { ...run('pending_confirmation'), version: 7, revision: 7 },
      ],
    })
    const user = userEvent.setup()
    render(<App />)
    await signIn(user)

    await user.click(await screen.findByTestId('resume-button'))

    expect(await screen.findByRole('heading', { name: '建议通过' })).toBeInTheDocument()
    // One failed poll plus the two successful reads: the failure did not stop the watch.
    expect(
      harness.callsTo(/^\/api\/v2\/reviews\/run-001$/, 'GET').length,
    ).toBeGreaterThanOrEqual(3)
  })
})

describe('Review workbench v2 polling timer identity', () => {
  it('keeps one interval across ticks instead of rebuilding it every poll', async () => {
    vi.useFakeTimers()
    try {
      const intervals = vi.spyOn(window, 'setInterval')
      const cleared = vi.spyOn(window, 'clearInterval')
      // The first two ticks keep the Run processing so several reads share one
      // timer; the third finalises it.
      const harness = installReviewFetch({
        initialRuns: [run('processing')],
        runSequence: [
          run('processing'),
          run('processing'),
          run('pending_confirmation'),
        ],
      })
      const advance = (ms: number) =>
        act(async () => {
          await vi.advanceTimersByTimeAsync(ms)
        })
      const pollReads = () =>
        harness.callsTo(/^\/api\/v2\/reviews\/run-001$/, 'GET').length

      render(<App />)
      fireEvent.change(screen.getByLabelText('Bearer 凭据'), {
        target: { value: TOKEN },
      })
      fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
      await advance(0)

      // The workspace auto-selected the processing Run: exactly one timer so far,
      // opened on the documented cadence.
      expect(intervals).toHaveBeenCalledTimes(1)
      expect(intervals).toHaveBeenCalledWith(
        expect.any(Function),
        RUN_POLL_INTERVAL_MS,
      )
      expect(cleared).not.toHaveBeenCalled()

      await advance(RUN_POLL_INTERVAL_MS)
      await advance(RUN_POLL_INTERVAL_MS)
      expect(pollReads()).toBe(2)
      // Two ticks, two reads, still the same timer object: nothing was rebuilt.
      expect(intervals).toHaveBeenCalledTimes(1)
      expect(cleared).not.toHaveBeenCalled()

      await advance(RUN_POLL_INTERVAL_MS)
      expect(pollReads()).toBe(3)
      // Settling the Run stops the watch.  The poll callback and the effect cleanup
      // both address that one timer id — neither of them creates a replacement.
      const timerId = intervals.mock.results[0]?.value
      expect(cleared).toHaveBeenCalled()
      expect(cleared.mock.calls.every(([id]) => id === timerId)).toBe(true)
      expect(intervals).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('watches the run identity rather than the per-tick payload object', async () => {
    vi.useFakeTimers()
    try {
      const intervals = vi.spyOn(window, 'setInterval')
      installReviewFetch({
        initialRuns: [run('processing')],
        runSequence: [run('processing'), run('processing')],
      })
      render(<App />)
      fireEvent.change(screen.getByLabelText('Bearer 凭据'), {
        target: { value: TOKEN },
      })
      fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })

      const first = intervals.mock.results[0]?.value
      for (let tick = 0; tick < 4; tick += 1) {
        await act(async () => {
          await vi.advanceTimersByTimeAsync(RUN_POLL_INTERVAL_MS)
        })
      }
      // Every tick replaced the run object in state; the interval identity did not change.
      expect(intervals).toHaveBeenCalledTimes(1)
      expect(intervals.mock.results[0]?.value).toBe(first)
    } finally {
      vi.useRealTimers()
    }
  })
})
