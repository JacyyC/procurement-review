/**
 * FIX-07 S-02 — every asynchronous source of a `ReviewRun` projection must pass
 * through one ordering decision.
 *
 * The 2026-09-17 review reproduced this with the production ordering function: a
 * newer projection set `trace_event_count` to 8, and an *earlier* detail GET that
 * arrived afterwards was still accepted and rolled the page back to 3.  The reason is
 * that only the read paths went through the per-Run read ticket; the write success and
 * the create/action reconciliation called the low-level cache writer directly, and the
 * cache writer itself only compared `revision` — which is equal here, because a Run's
 * progress counts move without its lifecycle revision changing.
 *
 * These tests drive the real `App` against a controllable `fetch`.  Held responses
 * deliberately ignore `AbortSignal`, so only the applied ordering rule can keep the
 * page right.  The first four cases are the counter-examples that must fail before the
 * repair; the rest are the guards that must keep passing after it.
 */

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App, { RUN_POLL_INTERVAL_MS } from './App'
import {
  OPERATOR_SESSION,
  TOKEN,
  installHarness,
  jsonResponse,
  makeAudit,
  makeRun,
  makeTrace,
} from './fix01cHarness'
import type { Harness, RecordedCall } from './fix01cHarness'
import { reviewRun } from './runFixtures'
import type { CaseListResponse, ReviewRun } from './types'

const CASE_LOW = 'CASE-LOW-001'
const CASE_HIGH = 'CASE-HIGH-001'

const CASES: CaseListResponse = {
  api_version: 'review-api.v2',
  execution_mode: 'test_only',
  cases: [
    {
      api_version: 'review-api.v2',
      case_id: CASE_LOW,
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
      case_id: CASE_HIGH,
      case_version: 'case-high-001.v1',
      label: '高风险冲突证据',
      case_type: 'purchase_order',
      scenario: 'high',
      document_version: 'retrieval-fixture-v1',
      dataset_version: 'synthetic-procurement.v1',
      index_version: 'retrieval-index-v1',
      rules_version: '2.0.0',
    },
  ],
}

// --------------------------------------------------------------------- helpers

/** A contract-valid Run payload, renamed to the Run under test. */
function runWith(runId: string, status: ReviewRun['status']): ReviewRun {
  const run = structuredClone(reviewRun(status))
  run.run_id = runId
  run.trace_id = `trace-${runId}`
  if (run.summary) run.summary.run_id = runId
  if (run.receipt) run.receipt = { ...run.receipt, run_id: runId }
  // The execution ledger carries its own copy of the summary, and the contract checks
  // that both name the Run they were served for.
  if (run.execution && typeof run.execution === 'object' && 'summary' in run.execution) {
    ;(run.execution as { summary: { run_id: string } }).summary.run_id = runId
  }
  return run
}

/**
 * A Run whose progress counters are the only thing that moves, so two projections of
 * the same `revision` differ exactly where the reproduced rollback happened.
 */
function progressRun(
  runId: string,
  counts: { revision: number; trace: number; status?: ReviewRun['status'] },
): ReviewRun {
  const run = runWith(runId, counts.status ?? 'pending_confirmation')
  run.revision = counts.revision
  run.version = counts.revision
  run.trace_event_count = counts.trace
  return run
}

function resumableRun(runId: string, trace: number): ReviewRun {
  const run = runWith(runId, 'interrupted')
  run.revision = 3
  run.version = 3
  run.trace_event_count = trace
  return run
}

async function settle() {
  await act(async () => {
    for (let tick = 0; tick < 200; tick += 1) await Promise.resolve()
  })
}

async function signIn() {
  render(<App />)
  fireEvent.change(screen.getByLabelText('Bearer 凭据'), { target: { value: TOKEN } })
  fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
  await settle()
}

function select(runId: string) {
  fireEvent.click(document.querySelector(`[data-run-id="${runId}"]`)!)
}

function refresh() {
  fireEvent.click(screen.getByRole('button', { name: '刷新当前 Review' }))
}

function confirm() {
  fireEvent.click(screen.getByRole('button', { name: '确认建议（本地）' }))
}

function resume() {
  fireEvent.click(screen.getByRole('button', { name: '恢复审查' }))
}

function cancel() {
  fireEvent.click(screen.getByRole('button', { name: '取消' }))
}

function replay() {
  fireEvent.click(screen.getByRole('button', { name: '重放原动作' }))
}

/** The event count the "运行 Trace" tab shows for the active Run. */
function traceTabCount(): number | null {
  const tab = screen.queryByRole('tab', { name: /运行 Trace/ })
  if (!tab) return null
  const match = tab.textContent?.match(/(\d+)/)
  return match ? Number(match[1]) : null
}

function selectedRunId(): string | null {
  return document.querySelector('[aria-current="true"]')?.getAttribute('data-run-id') ?? null
}

const isGetRun = (runId: string) => (call: RecordedCall) =>
  call.method === 'GET' && call.url === `/api/v2/reviews/${runId}`

function pollReads(harness: Harness, runId: string): number {
  return harness.callsTo(new RegExp(`/reviews/${runId}$`), 'GET').length
}

/** Wait until the poll tick has really reached the server, then hand back its gate. */
async function holdNextTick(
  harness: Harness,
  runId: string,
  advance: (ms: number) => Promise<void>,
): Promise<{ release: () => void }> {
  const gate = harness.hold(isGetRun(runId))
  const before = pollReads(harness, runId)
  await advance(RUN_POLL_INTERVAL_MS)
  const deadline = Date.now() + 2000
  while (pollReads(harness, runId) <= before && Date.now() < deadline) {
    await advance(1)
  }
  expect(pollReads(harness, runId)).toBeGreaterThan(before)
  return { release: () => gate.resolve() }
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

// ------------------------------------------------- red counter-examples (S-02)

describe('FIX-07 S-02 · a late detail GET may not roll back a newer projection', () => {
  it('keeps the newer write projection when an older same-revision GET lands after it', async () => {
    const initial = progressRun('run-write', { revision: 2, trace: 3 })
    const written = progressRun('run-write', {
      revision: 2,
      trace: 8,
      status: 'confirmed',
    })
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [initial],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
      actionResult: () => written,
    })
    await signIn()
    expect(traceTabCount()).toBe(3)

    // The human re-selects the Run; that detail read answers from the state at request
    // time (3) and is held while newer work completes.
    const gate = harness.hold(isGetRun('run-write'))
    select('run-write')
    await settle()

    confirm()
    await settle()
    // The write response is the newest projection the client has seen.
    expect(traceTabCount()).toBe(8)

    gate.resolve()
    await settle()

    expect(traceTabCount()).toBe(8)
    expect(harness.callsTo(/\/confirm$/, 'POST')).toHaveLength(1)
  })

  it('keeps the newer action reconciliation when an older same-revision GET lands after it', async () => {
    const initial = progressRun('run-write', { revision: 2, trace: 3 })
    const reconciled = progressRun('run-write', { revision: 2, trace: 8 })
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [initial],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
      failOnce: {
        match: /\/confirm$/,
        method: 'POST',
        status: 503,
        code: 'security_dependency_unavailable',
      },
    })
    await signIn()

    const gate = harness.hold(isGetRun('run-write'))
    select('run-write')
    await settle()

    // The write's outcome is unknown, so the client reconciles with a read — and that
    // read already sees the newer projection.
    harness.putRun(reconciled)
    confirm()
    await settle()
    expect(traceTabCount()).toBe(8)

    gate.resolve()
    await settle()

    expect(traceTabCount()).toBe(8)
  })

  it('keeps the newer resume reconciliation when an older same-revision GET lands after it', async () => {
    const initial = resumableRun('run-resume', 3)
    const reconciled = resumableRun('run-resume', 8)
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [initial],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
      failOnce: {
        match: /\/resume$/,
        method: 'POST',
        status: 503,
        code: 'security_dependency_unavailable',
      },
    })
    await signIn()

    const gate = harness.hold(isGetRun('run-resume'))
    select('run-resume')
    await settle()

    harness.putRun(reconciled)
    resume()
    await settle()
    expect(traceTabCount()).toBe(8)

    gate.resolve()
    await settle()

    expect(traceTabCount()).toBe(8)
  })

  it('keeps the newest projection when an older poll tick lands after an idempotent replay', async () => {
    vi.useFakeTimers()
    const processing = progressRun('run-replay', {
      revision: 1,
      trace: 2,
      status: 'processing',
    })
    const replayed = progressRun('run-replay', {
      revision: 1,
      trace: 8,
      status: 'processing',
    })
    const advance = (ms: number) =>
      act(async () => {
        await vi.advanceTimersByTimeAsync(ms)
      })
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [processing],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
      networkErrorOnce: { match: /\/cancel$/, method: 'POST' },
      actionResult: () => replayed,
    })
    await signIn()
    await advance(0)
    expect(traceTabCount()).toBe(2)

    // The first cancel never reached the server, so its result is unknown and the
    // session keeps the original request for an explicit same-key replay.
    cancel()
    await advance(1)
    expect(screen.getByRole('button', { name: '重放原动作' })).toBeEnabled()

    // One poll tick is answered from the pre-replay state and then held.
    const tick = await holdNextTick(harness, 'run-replay', advance)

    replay()
    await advance(1)
    expect(traceTabCount()).toBe(8)

    tick.release()
    await advance(1)

    expect(traceTabCount()).toBe(8)
    expect(harness.callsTo(/\/cancel$/, 'POST')).toHaveLength(2)
  })
})

// ------------------------------------------------------ ordering guards (AC-06)

describe('FIX-07 S-02 · the projection order is closed-form', () => {
  it('never lets a lower revision overwrite a higher one, however new the request is', async () => {
    const current = progressRun('run-lower', { revision: 3, trace: 8 })
    const stale = progressRun('run-lower', { revision: 1, trace: 3 })
    installHarness({
      session: OPERATOR_SESSION,
      runs: [current],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
      overrideOnce: {
        match: /^\/api\/v2\/reviews\/run-lower$/,
        method: 'GET',
        response: () => jsonResponse(stale),
      },
    })
    await signIn()
    expect(traceTabCount()).toBe(8)

    refresh()
    await settle()

    expect(traceTabCount()).toBe(8)
  })

  it('lets a genuinely newer projection of the same revision replace an older one', async () => {
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [progressRun('run-forward', { revision: 1, trace: 3 })],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
    })
    await signIn()
    expect(traceTabCount()).toBe(3)

    harness.putRun(progressRun('run-forward', { revision: 1, trace: 8 }))
    refresh()
    await settle()
    expect(traceTabCount()).toBe(8)

    harness.putRun(progressRun('run-forward', { revision: 1, trace: 9 }))
    refresh()
    await settle()
    expect(traceTabCount()).toBe(9)
  })

  it('keeps one ordering state per Run instead of one shared state', async () => {
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [
        progressRun('run-a', { revision: 1, trace: 3 }),
        progressRun('run-b', { revision: 1, trace: 5 }),
      ],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
    })
    await signIn()
    expect(selectedRunId()).toBe('run-a')

    harness.putRun(progressRun('run-a', { revision: 1, trace: 8 }))
    // The re-selection of run-a is held, so the only way its newer value can reach the
    // page is through the ordering state that belongs to run-a.
    const gate = harness.hold(isGetRun('run-a'))
    select('run-a')
    await settle()

    // A *newer* request for another Run is applied in between.
    select('run-b')
    await settle()
    expect(traceTabCount()).toBe(5)

    gate.resolve()
    await settle()

    // run-a's own late read is newer than run-a's applied read, so it still lands.
    const finalGate = harness.hold(isGetRun('run-a'))
    select('run-a')
    await settle()
    expect(traceTabCount()).toBe(8)
    finalGate.resolve()
    await settle()
  })
})

// ------------------------------------------------ ownership and binding guards

describe('FIX-07 S-02 · ownership, binding and replay are unchanged', () => {
  it('invalidates every response of the previous identity', async () => {
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [
        progressRun('run-a', { revision: 1, trace: 3 }),
        progressRun('run-b', { revision: 1, trace: 5 }),
      ],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
    })
    await signIn()

    const gate = harness.hold(isGetRun('run-a'))
    select('run-a')
    await settle()

    // The credential stops being valid; the next call ends the session.
    harness.expireCredential()
    select('run-b')
    await settle()
    expect(screen.getByLabelText('Bearer 凭据')).toBeInTheDocument()

    gate.resolve()
    await settle()

    expect(screen.getByLabelText('Bearer 凭据')).toBeInTheDocument()
    expect(document.querySelector('[data-run-id]')).toBeNull()
  })

  it('binds a create response to the Case frozen before the request was sent', async () => {
    const otherCase = makeRun('run-other-case', {
      status: 'processing',
      request: { ...makeRun('run-other-case').request, case_id: CASE_HIGH },
    })
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
      overrideOnce: {
        match: /^\/api\/v2\/reviews$/,
        method: 'POST',
        response: () => jsonResponse(otherCase, 202),
      },
    })
    await signIn()

    // The frozen Case is the selected one (CASE-LOW-001); CASE-HIGH-001 is also
    // authorized, so accepting the response would really be an identity mix-up.
    expect((screen.getByLabelText('已授权案例') as HTMLSelectElement).value).toBe(CASE_LOW)
    fireEvent.click(screen.getByRole('button', { name: '运行审查' }))
    await settle()

    expect(harness.callsTo(/^\/api\/v2\/reviews$/, 'POST')).toHaveLength(1)
    expect(document.querySelector('[data-run-id="run-other-case"]')).toBeNull()
    expect(harness.runs()).toEqual([])
    expect(screen.getByRole('button', { name: '重放最近创建请求' })).toBeEnabled()
  })

  it('does not let a write or its reconciliation steal the Run the human selected', async () => {
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [
        progressRun('run-a', { revision: 2, trace: 3 }),
        progressRun('run-b', { revision: 2, trace: 5 }),
      ],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
      failOnce: {
        match: /\/confirm$/,
        method: 'POST',
        status: 503,
        code: 'security_dependency_unavailable',
      },
    })
    await signIn()
    expect(selectedRunId()).toBe('run-a')

    harness.putRun(progressRun('run-a', { revision: 2, trace: 8 }))
    const gate = harness.hold(isGetRun('run-a'))
    confirm()
    await settle()

    // While run-a's reconciliation read is in flight the human moves to run-b.
    select('run-b')
    await settle()
    expect(selectedRunId()).toBe('run-b')

    gate.resolve()
    await settle()

    expect(selectedRunId()).toBe('run-b')
    expect(traceTabCount()).toBe(5)

    // run-a did receive its newer projection; it simply never took the selection.
    const finalGate = harness.hold(isGetRun('run-a'))
    select('run-a')
    await settle()
    expect(traceTabCount()).toBe(8)
    finalGate.resolve()
    await settle()
  })

  it('never retries a write on its own', async () => {
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [progressRun('run-a', { revision: 2, trace: 3 })],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
      failOnce: {
        match: /\/confirm$/,
        method: 'POST',
        status: 503,
        code: 'security_dependency_unavailable',
      },
    })
    await signIn()

    confirm()
    await settle()
    await settle()

    expect(harness.callsTo(/\/confirm$/, 'POST')).toHaveLength(1)
    // The only follow-up is a read: the unknown outcome is reconciled, never resent.
    expect(harness.callsTo(/^\/api\/v2\/reviews\/run-a$/, 'GET').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: '重放原动作' })).toBeEnabled()
  })
})
