/**
 * Controllable `fetch` harness for the FIX-01C ownership and contract tests.
 *
 * The point of this harness is *release order*: a test can hold any response and
 * release it after later requests have already settled, which is exactly the race
 * AC-05 talks about.  The held promise deliberately ignores the request's
 * `AbortSignal`, so every ownership guarantee under test has to come from the
 * generation / selection / per-resource guards rather than from cancellation.
 */

import { vi } from 'vitest'
import type {
  AuditResponse,
  CaseListResponse,
  CaseOption,
  ReviewRun,
  SessionInfo,
  TraceResponse,
} from './types'

export interface RecordedCall {
  url: string
  method: string
  body: unknown
  headers: Headers
  signal: AbortSignal | null | undefined
}

export interface Deferred<T> {
  promise: Promise<T>
  resolve: (value: T) => void
}

export function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((settle) => {
    resolve = settle
  })
  return { promise, resolve }
}

export function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

export function errorResponse(
  status: number,
  code: string,
  message = 'denied',
  details: Record<string, unknown> = {},
): Response {
  return jsonResponse(
    { error: { code, message, request_id: 'request-fixture', details } },
    status,
  )
}

export interface HarnessOptions {
  session: SessionInfo
  runs: ReviewRun[]
  /** Granted Cases; `startReview` needs one to enable the create control. */
  cases?: CaseOption[]
  trace?: (runId: string) => TraceResponse
  audit?: (runId: string) => AuditResponse
  createResult?: () => ReviewRun
  actionResult?: () => ReviewRun
  resumeResult?: () => ReviewRun
  /** Answer a business call with this failure once, before the normal route. */
  failOnce?: { match: RegExp; method?: string; status: number; code: string }
  /** Answer one call with a caller-built body, e.g. a malformed success payload. */
  overrideOnce?: { match: RegExp; method?: string; response: () => Response }
  /**
   * Let one call reach the server — so the mutation it represents really commits —
   * and then lose the response on the way back (`TypeError: Failed to fetch`).  This
   * is the "confirmed but the answer never arrived" case of C-AUD-02: the client
   * cannot tell whether the write happened, so it must keep the request context.
   */
  loseResponseOnce?: { match: RegExp; method?: string }
  /** Fail one call before it reaches the server: nothing is committed, nothing known. */
  networkErrorOnce?: { match: RegExp; method?: string; error?: () => Error }
  /** Commit the mutation a create/action/resume call represents. */
  onWrite?: (runId: string, suffix: string) => void
}

export interface Harness {
  calls: RecordedCall[]
  /** Hold the *next* matching response until the returned deferred resolves. */
  hold: (match: (call: RecordedCall) => boolean) => Deferred<void>
  putRun: (run: ReviewRun) => void
  putTrace: (runId: string, value: TraceResponse) => void
  putAudit: (runId: string, value: AuditResponse) => void
  runs: () => ReviewRun[]
  callsTo: (pattern: RegExp, method?: string) => RecordedCall[]
  expireCredential: () => void
  /** Serve a different identity next, e.g. a credential with fewer scopes. */
  setSession: (session: SessionInfo) => void
}

export function installHarness(options: HarnessOptions): Harness {
  const calls: RecordedCall[] = []
  const runStore = new Map<string, ReviewRun>()
  for (const run of options.runs) runStore.set(run.run_id, run)
  const traceStore = new Map<string, TraceResponse>()
  const auditStore = new Map<string, AuditResponse>()
  const gates: { match: (call: RecordedCall) => boolean; release: Deferred<void> }[] = []
  let credentialValid = true
  let currentSession = options.session
  let pendingFailure = options.failOnce ?? null
  let pendingOverride = options.overrideOnce ?? null
  let pendingLostResponse = options.loseResponseOnce ?? null
  let pendingNetworkError = options.networkErrorOnce ?? null

  const cases: CaseListResponse = {
    api_version: 'review-api.v2',
    execution_mode: 'test_only',
    cases: options.cases ?? [],
  }

  function route(call: RecordedCall): Response {
    const { url, method } = call
    if (!url.startsWith('/api/v2/')) {
      // The retired v1 surface must never be reached from the workbench.
      return errorResponse(410, 'api_version_retired', 'review-api.v1 is retired')
    }
    if (!credentialValid || !call.headers.get('Authorization')?.startsWith('Bearer ')) {
      return errorResponse(401, 'unauthenticated', 'Credential is unusable')
    }
    if (pendingFailure && pendingFailure.match.test(url) &&
        (!pendingFailure.method || pendingFailure.method === method)) {
      const failure = pendingFailure
      pendingFailure = null
      return errorResponse(failure.status, failure.code, 'fixture failure')
    }
    if (pendingOverride && pendingOverride.match.test(url) &&
        (!pendingOverride.method || pendingOverride.method === method)) {
      const override = pendingOverride
      pendingOverride = null
      return override.response()
    }
    if (url === '/api/v2/session') return jsonResponse(currentSession)
    if (url === '/api/v2/cases') return jsonResponse(cases)
    if (url === '/api/v2/reviews' && method === 'GET') {
      return jsonResponse({ api_version: 'review-api.v2', runs: [...runStore.values()] })
    }
    if (url === '/api/v2/reviews' && method === 'POST') {
      const created = options.createResult?.() ?? [...runStore.values()][0]
      runStore.set(created.run_id, created)
      return jsonResponse(created, 202)
    }
    const target = url.match(/^\/api\/v2\/reviews\/([^/]+)(\/.*)?$/)
    if (target) {
      const runId = decodeURIComponent(target[1])
      const suffix = target[2] ?? ''
      const current = runStore.get(runId)
      if (suffix === '' && method === 'GET') {
        if (!current) return errorResponse(404, 'resource_not_found')
        return jsonResponse(current)
      }
      if (suffix === '/trace') {
        const value = traceStore.get(runId) ?? options.trace?.(runId)
        if (!value) return errorResponse(404, 'resource_not_found')
        return jsonResponse(value)
      }
      if (suffix === '/audit') {
        const value = auditStore.get(runId) ?? options.audit?.(runId)
        if (!value) return errorResponse(404, 'resource_not_found')
        return jsonResponse(value)
      }
      if (suffix === '/resume' && method === 'POST') {
        options.onWrite?.(runId, suffix)
        const next = options.resumeResult?.() ?? current
        if (!next) return errorResponse(404, 'resource_not_found')
        runStore.set(runId, next)
        return jsonResponse(next, 202)
      }
      if (['/confirm', '/reject', '/cancel'].includes(suffix) && method === 'POST') {
        options.onWrite?.(runId, suffix)
        const next = options.actionResult?.() ?? current
        if (!next) return errorResponse(404, 'resource_not_found')
        runStore.set(runId, next)
        return jsonResponse(next)
      }
    }
    throw new Error(`Unhandled fetch: ${method} ${url}`)
  }

  const fetchMock = vi.fn(
    async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = String(input)
      const method = (init?.method ?? 'GET').toUpperCase()
      const headers = new Headers(init?.headers)
      let body: unknown = null
      if (typeof init?.body === 'string') body = JSON.parse(init.body)
      const call: RecordedCall = {
        url,
        method,
        body,
        headers,
        signal: init?.signal,
      }
      calls.push(call)
      if (pendingNetworkError && pendingNetworkError.match.test(url) &&
          (!pendingNetworkError.method || pendingNetworkError.method === method)) {
        const failure = pendingNetworkError
        pendingNetworkError = null
        throw (failure.error?.() ?? new TypeError('Failed to fetch'))
      }
      // The response is produced when the request *arrives*, exactly as a real server
      // would, so a held response still carries the value it had at that moment —
      // which is what makes "the stale read lands last" a meaningful scenario.
      const response = route(call)
      if (pendingLostResponse && pendingLostResponse.match.test(url) &&
          (!pendingLostResponse.method || pendingLostResponse.method === method)) {
        pendingLostResponse = null
        throw new TypeError('Failed to fetch')
      }
      // One-shot: a later request that happens to match must not be held too, or a
      // test could not exercise "the refresh overtakes the held read".
      const gateIndex = gates.findIndex((gate) => gate.match(call))
      if (gateIndex >= 0) {
        const [gate] = gates.splice(gateIndex, 1)
        // Deliberately no `signal` handling: releasing is the test's decision.
        await gate.release.promise
      }
      return response
    },
  )
  vi.stubGlobal('fetch', fetchMock)

  return {
    calls,
    hold: (match) => {
      const release = deferred<void>()
      gates.push({ match, release })
      return release
    },
    putRun: (run) => {
      runStore.set(run.run_id, run)
    },
    putTrace: (runId, value) => {
      traceStore.set(runId, value)
    },
    putAudit: (runId, value) => {
      auditStore.set(runId, value)
    },
    runs: () => [...runStore.values()],
    callsTo: (pattern, method) =>
      calls.filter((call) => (!method || call.method === method) && pattern.test(call.url)),
    expireCredential: () => {
      credentialValid = false
    },
    setSession: (session) => {
      currentSession = session
    },
  }
}

/** A Run fixture with sane, monotonic revisions. */
export function makeRun(
  runId: string,
  overrides: Partial<ReviewRun> = {},
): ReviewRun {
  return {
    run_id: runId,
    trace_id: `trace-${runId}`,
    status: 'pending_confirmation',
    version: 2,
    revision: 2,
    execution_mode: 'test_only',
    request: {
      schema_version: '1.0',
      case_id: 'CASE-LOW-001',
      case_type: 'purchase_order',
      document_version: 'retrieval-fixture-v1',
      purchase_order_id: 'PO-001',
      dataset_version: 'synthetic-procurement.v1',
      initiator: 'subject-local-demo',
    },
    summary: null,
    summary_checksum: null,
    execution: null,
    branch_failures: [],
    evidence_refs: [],
    business_facts: [],
    receipt: null,
    created_at: '2026-09-01T00:00:00Z',
    updated_at: '2026-09-01T00:00:05Z',
    trace_event_count: 3,
    audit_event_count: 0,
    error_code: null,
    error_message: null,
    recovery: null,
    ...overrides,
  }
}

export function makeTrace(runId: string, eventTypes: string[]): TraceResponse {
  return {
    api_version: 'review-api.v1',
    trace: {
      schema_version: '1.0',
      trace_schema_version: 'trace.v1',
      trace_id: `trace-${runId}`,
      run_id: runId,
      complete: false,
      events: eventTypes.map((eventType, index) => ({
        schema_version: '1.0',
        event_id: `event-${runId}-${String(index + 1).padStart(4, '0')}`,
        run_id: runId,
        trace_id: `trace-${runId}`,
        event_type: eventType,
        step: 'review_run',
        occurred_at: '2026-09-01T00:00:01Z',
        input_summary: {},
        output_summary: {},
        source_versions: { api: 'review-api.v2' },
        duration_ms: 0,
        token_count: 0,
        cost_amount: '0',
        cost_currency: 'CNY',
        redaction_status: 'not_required',
        error: null,
      })),
    },
  }
}

export function makeAudit(runId: string, actors: string[]): AuditResponse {
  return {
    api_version: 'review-api.v2',
    audit_schema_version: 'approval-audit.v2',
    run_id: runId,
    events: actors.map((actor, index) => ({
      schema_version: '1.0',
      audit_schema_version: 'approval-audit.v2',
      audit_id: `audit-${runId}-${index + 1}`,
      sequence: index + 1,
      tenant_id: 'tenant-local-demo',
      case_id: 'CASE-LOW-001',
      run_id: runId,
      event_class: 'approval' as const,
      action_type: 'confirm' as const,
      actor,
      authorization_revision: 1,
      previous_status: 'pending_confirmation' as const,
      new_status: 'confirmed' as const,
      previous_revision: 2,
      new_revision: 3,
      result_checksum: `sha256:${'a'.repeat(64)}`,
      idempotency_digest: `sha256:${'b'.repeat(64)}`,
      receipt_id: `receipt-${runId}`,
      api_version: 'review-api.v2' as const,
      occurred_at: '2026-09-01T00:00:06Z',
      detail_code: null,
      simulation_only: true as const,
      external_write_performed: false as const,
    })),
  }
}

export const OPERATOR_SESSION: SessionInfo = {
  api_version: 'review-api.v2',
  subject_id: 'subject-local-demo',
  tenant_id: 'tenant-local-demo',
  credential_id: 'cred-operator-fixture',
  authorization_revision: 1,
  roles: ['approver', 'operator'],
  scopes: [
    'audit:read',
    'case:read',
    'review:cancel',
    'review:create',
    'review:decide',
    'review:read',
    'review:resume',
    'trace:read',
  ],
  issued_at: '2026-09-01T00:00:00Z',
  expires_at: '2026-09-01T01:00:00Z',
}

export const TOKEN = 'fix01c-local-credential'
