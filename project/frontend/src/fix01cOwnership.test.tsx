/**
 * FIX-01C / V-C-01 — request ownership, selection and the human-action context.
 *
 * Every scenario here releases held responses *out of order* through a harness whose
 * held promise ignores `AbortSignal`, so a pass means the guards themselves are
 * correct rather than that the browser cancelled something.  The scenarios are the
 * ones AC-05 names: same-identity A→B→A, list refresh / poll overtaking a click, an
 * independent late Trace, a late write result, identity change and unmount.
 */

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import type { ReviewRun } from './types'
import {
  OPERATOR_SESSION,
  TOKEN,
  installHarness,
  makeAudit,
  makeRun,
  makeTrace,
} from './fix01cHarness'

const CHECKSUM = `sha256:${'c'.repeat(64)}`

/** A Run that has committed a summary, so the human Gate is actually open. */
function decidable(runId: string, overrides: Partial<ReviewRun> = {}): ReviewRun {
  return makeRun(runId, {
    version: 4,
    revision: 4,
    summary: {
      schema_version: '1.0',
      run_id: runId,
      case_id: 'CASE-LOW-001',
      rules_version: '2.0.0',
      findings: [
        {
          schema_version: '1.0',
          worker: 'policy',
          finding_key: 'policy.required_policy_evidence',
          category: 'policy_compliance',
          risk_level: 'low',
          assertion: 'risk_absent',
          statement: '政策阈值证据完整。',
          status: 'supported',
          evidence_refs: [],
          business_fact_refs: [],
          missing_information: [],
          recommended_action: '保留证据供人工确认。',
        },
      ],
      highest_risk: 'low',
      recommendation: 'recommend_approve',
      human_review_status: 'pending_confirmation',
      conflicts: [],
      missing_information: [],
      failure_reasons: [],
      is_final_decision: false,
    },
    summary_checksum: CHECKSUM,
    ...overrides,
  })
}

function selectedRunId(): string | null {
  return document.querySelector('.run-title code')?.getAttribute('title') ?? null
}

/**
 * Drain the promise chain started by a click.  Vitest's fake timers do not drive
 * Testing Library's `waitFor`, so settling is explicit: N microtask ticks inside one
 * `act`, which is also what makes the release order in these tests deterministic.
 */
async function settle(rounds = 40): Promise<void> {
  await act(async () => {
    for (let index = 0; index < rounds; index += 1) {
      await Promise.resolve()
    }
  })
}

async function signIn(): Promise<void> {
  render(<App />)
  fireEvent.change(screen.getByLabelText('Bearer 凭据'), { target: { value: TOKEN } })
  fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
  await settle()
  expect(screen.getByTestId('identity-subject')).toBeInTheDocument()
}

function selectRunById(runId: string): void {
  const button = document.querySelector(`[data-run-id="${runId}"]`)
  expect(button, `run ${runId} must be listed`).not.toBeNull()
  fireEvent.click(button as Element)
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('FIX-01C request ownership', () => {
  it('keeps the human A→B→A selection when the responses arrive in reverse order', async () => {
    vi.useFakeTimers()
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a'), makeRun('run-b')],
    })
    await signIn()
    expect(selectedRunId()).toBe('run-a')

    // A and B both move on before the human starts clicking, so every held response
    // carries the newer payload.  Two reads for A (check-out and check-back) plus one
    // for B are held; the B response is released *last*, so anything that re-selects
    // on completion would land on B even though the final click was A.
    harness.putRun(makeRun('run-a', { status: 'failed', revision: 9 }))
    harness.putRun(makeRun('run-b', { status: 'cancelled', revision: 9 }))
    const gateA1 = harness.hold((call) => call.url === '/api/v2/reviews/run-a')
    const gateA2 = harness.hold((call) => call.url === '/api/v2/reviews/run-a')
    const gateB = harness.hold((call) => call.url === '/api/v2/reviews/run-b')

    selectRunById('run-a')
    await settle()
    selectRunById('run-b')
    await settle()
    // Back to A: the third click is the human's latest word.
    selectRunById('run-a')
    await settle()
    expect(selectedRunId()).toBe('run-a')

    gateA1.resolve()
    await settle()
    gateA2.resolve()
    await settle()
    gateB.resolve()
    await settle()

    // The human's last click stands, and every late payload still landed on the Run
    // it describes rather than on the selected one.
    expect(selectedRunId()).toBe('run-a')
    const labels = [...document.querySelectorAll('[data-run-id]')].map((node) =>
      node.textContent ?? '',
    )
    expect(labels.some((text) => text.includes('运行失败'))).toBe(true)
    expect(labels.some((text) => text.includes('已取消'))).toBe(true)
  })

  it('does not let an older revision of the selected Run overwrite the newer one', async () => {
    vi.useFakeTimers()
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
    })
    await signIn()
    expect(selectedRunId()).toBe('run-a')

    // The check-out read is held; the Run then moves on to a later revision.
    const gate = harness.hold((call) => call.url === '/api/v2/reviews/run-a')
    selectRunById('run-a')
    await settle()
    harness.putRun(makeRun('run-a', { status: 'failed', revision: 7 }))
    fireEvent.click(screen.getByRole('button', { name: '刷新当前 Review' }))
    await settle()
    expect(screen.getByRole('heading', { name: 'Review 运行失败并已关闭' })).toBeInTheDocument()

    // The stale read finally arrives and must be discarded.
    harness.putRun(makeRun('run-a', { status: 'pending_confirmation', revision: 2 }))
    gate.resolve()
    await settle()

    expect(screen.getByRole('heading', { name: 'Review 运行失败并已关闭' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '建议通过' })).not.toBeInTheDocument()
  })

  it('never lets a poll tick steal a selection the human made while it was in flight', async () => {
    vi.useFakeTimers()
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [
        makeRun('run-a', { status: 'processing', version: 1, revision: 1 }),
        makeRun('run-b'),
      ],
    })
    await signIn()
    expect(selectedRunId()).toBe('run-a')

    // The first poll tick for the still-processing Run is held mid-flight.
    const gate = harness.hold((call) => call.url === '/api/v2/reviews/run-a')
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200)
    })
    expect(harness.callsTo(/^\/api\/v2\/reviews\/run-a$/, 'GET').length).toBeGreaterThan(0)

    selectRunById('run-b')
    await settle()
    expect(selectedRunId()).toBe('run-b')

    gate.resolve()
    await settle()

    // The late tick updated run-a's cache entry; it did not re-select it.
    expect(selectedRunId()).toBe('run-b')
  })

  it('renders only the selected Run Trace, whatever order the Trace reads finish in', async () => {
    vi.useFakeTimers()
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a'), makeRun('run-b')],
      trace: (runId) => makeTrace(runId, [`${runId}_event`]),
      audit: (runId) => makeAudit(runId, [`${runId}-actor`]),
    })
    await signIn()
    expect(selectedRunId()).toBe('run-a')

    const gateA = harness.hold((call) => call.url === '/api/v2/reviews/run-a/trace')
    selectRunById('run-a')
    await settle()
    selectRunById('run-b')
    await settle()

    fireEvent.click(screen.getByRole('tab', { name: /Trace/ }))
    await settle()
    expect(screen.getByText('run-b_event')).toBeInTheDocument()

    await act(async () => {
      gateA.resolve()
      await Promise.resolve()
    })
    await settle()

    // A's late Trace is on screen under B: exactly the cross-object leak AC-05 forbids.
    expect(screen.queryByText('run-a_event')).not.toBeInTheDocument()
    expect(screen.getByText('run-b_event')).toBeInTheDocument()
  })

  it('keeps the newest Trace when an older read for the same Run lands later', async () => {
    vi.useFakeTimers()
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: (runId) => makeTrace(runId, ['run_started']),
      audit: (runId) => makeAudit(runId, []),
    })
    await signIn()

    const gate = harness.hold((call) => call.url === '/api/v2/reviews/run-a/trace')
    selectRunById('run-a')
    await settle()
    harness.putTrace('run-a', makeTrace('run-a', ['run_completed']))
    fireEvent.click(screen.getByRole('button', { name: '刷新当前 Review' }))
    await settle()

    fireEvent.click(screen.getByRole('tab', { name: /Trace/ }))
    await settle()
    expect(screen.getByText('run_completed')).toBeInTheDocument()

    gate.resolve()
    await settle()

    expect(screen.getByText('run_completed')).toBeInTheDocument()
    expect(screen.queryByText('run_started')).not.toBeInTheDocument()
  })

  it('freezes the human action on the Run, version and checksum that were on screen', async () => {
    vi.useFakeTimers()
    const a = decidable('run-a')
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [a, makeRun('run-b')],
      actionResult: () => ({
        ...a,
        status: 'confirmed',
        version: 5,
        revision: 5,
        receipt: {
          api_version: 'review-api.v2',
          receipt_id: 'receipt-run-a',
          run_id: 'run-a',
          case_id: 'CASE-LOW-001',
          human_actor: 'subject-local-demo',
          decision: 'confirm_recommendation',
          reviewed_recommendation: 'recommend_approve',
          recorded_at: '2026-09-01T00:00:06Z',
          idempotency_key_digest: `sha256:${'d'.repeat(64)}`,
          simulation_status: 'local_simulation_recorded',
          simulation_mode: 'local_test_only',
          external_write_performed: false,
        },
      }),
    })
    await signIn()
    expect(selectedRunId()).toBe('run-a')

    const gate = harness.hold((call) => call.url === '/api/v2/reviews/run-a/confirm')
    fireEvent.click(screen.getByRole('button', { name: '确认建议（本地）' }))
    await settle()

    // The human moves to another Run while the decision is still in flight.
    selectRunById('run-b')
    await settle()
    expect(selectedRunId()).toBe('run-b')

    gate.resolve()
    await settle()

    const decisions = harness.callsTo(/\/confirm$/, 'POST')
    expect(decisions).toHaveLength(1)
    // The payload is the one the human actually saw: run-a, version 4, its checksum.
    expect(decisions[0].url).toBe('/api/v2/reviews/run-a/confirm')
    expect(decisions[0].body).toEqual({
      api_version: 'review-api.v2',
      expected_version: 4,
      expected_summary_checksum: CHECKSUM,
    })
    // The late result is cached on run-a and never re-selects or leaks into run-b.
    expect(selectedRunId()).toBe('run-b')
    expect(screen.queryByRole('heading', { name: '本地模拟回执' })).not.toBeInTheDocument()
    expect(screen.queryByText('人工动作仅在本地记录，未写入采购系统。')).not.toBeInTheDocument()
    const aButton = document.querySelector('[data-run-id="run-a"]')
    expect(aButton?.textContent).toContain('已确认建议')
  })

  it('reports a late write failure against the Run it belonged to, not the visible one', async () => {
    vi.useFakeTimers()
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [decidable('run-a'), makeRun('run-b')],
      failOnce: { match: /\/confirm$/, method: 'POST', status: 409, code: 'version_conflict' },
    })
    await signIn()

    const gate = harness.hold((call) => call.url === '/api/v2/reviews/run-a/confirm')
    fireEvent.click(screen.getByRole('button', { name: '确认建议（本地）' }))
    await settle()
    selectRunById('run-b')
    await settle()

    gate.resolve()
    await settle()

    expect(selectedRunId()).toBe('run-b')
    // Another Run's failure notice must not describe the object on screen.
    expect(screen.queryByText(/版本已过期/)).not.toBeInTheDocument()
    expect(harness.callsTo(/\/confirm$/, 'POST')).toHaveLength(1)
  })

  it('aborts in-flight work on unmount and applies nothing that still resolves', async () => {
    vi.useFakeTimers()
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
    })
    await signIn()

    const gate = harness.hold((call) => call.url === '/api/v2/reviews/run-a')
    selectRunById('run-a')
    await settle()
    const held = harness.callsTo(/^\/api\/v2\/reviews\/run-a$/, 'GET')
    expect(held.length).toBeGreaterThan(0)
    expect(held.at(-1)?.signal?.aborted).toBe(false)

    cleanup()
    // Resource cleanup is observable: the request channel was aborted.
    expect(held.at(-1)?.signal?.aborted).toBe(true)

    // ...and correctness does not depend on it: the harness ignores the signal, and
    // releasing the response afterwards must not throw or resurrect the workspace.
    gate.resolve()
    await settle()
    expect(document.querySelector('.run-title code')).toBeNull()
  })

  it('discards an identity superseded while its read was still in flight', async () => {
    vi.useFakeTimers()
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
    })
    await signIn()

    const gate = harness.hold((call) => call.url === '/api/v2/reviews/run-a')
    selectRunById('run-a')
    await settle()

    fireEvent.click(screen.getByRole('button', { name: '身份与权限' })); fireEvent.click(screen.getByRole('button', { name: '退出并清除' }))
    await settle()
    expect(screen.getByText('需要本地凭据')).toBeInTheDocument()

    harness.putRun(decidable('run-a'))
    gate.resolve()
    await settle()

    expect(screen.getByText('需要本地凭据')).toBeInTheDocument()
    expect(document.querySelectorAll('[data-run-id]')).toHaveLength(0)
    expect(screen.queryByRole('heading', { name: '建议通过' })).not.toBeInTheDocument()
  })
})
