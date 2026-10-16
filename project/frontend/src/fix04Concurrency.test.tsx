/**
 * FIX-04A / C-AUD-03 — in-flight write state is keyed by operation, not by one slot.
 *
 * The audit found a single `busy` slot holding one `runId`: starting a second Run's
 * decision overwrote the first, and whichever request finished first cleared the slot
 * for both.  A Run whose write was still in flight therefore looked idle, and the
 * operator could start a second write that the previous state was supposed to block.
 *
 * The observable used here is deliberately a *control*, not the handler: a Run with an
 * in-flight write keeps its 刷新/确认 controls disabled, and another Run's completion
 * neither frees it nor is freed by it.
 */

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { unstable_batchedUpdates } from 'react-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import {
  OPERATOR_SESSION,
  TOKEN,
  installHarness,
  makeAudit,
  makeTrace,
} from './fix01cHarness'
import type { Harness } from './fix01cHarness'
import { reviewRun } from './runFixtures'
import type { ReviewRun } from './types'

function runWith(
  id: string,
  status: ReviewRun['status'],
  patch: Partial<ReviewRun> = {},
): ReviewRun {
  const run = structuredClone(reviewRun(status))
  run.run_id = id
  run.trace_id = `trace-${id}`
  if (run.summary) run.summary.run_id = id
  if (run.receipt) run.receipt.run_id = id
  if (run.execution && typeof run.execution === 'object' && 'summary' in run.execution) {
    ;(run.execution as { summary: { run_id: string } }).summary.run_id = id
  }
  return { ...run, ...patch }
}

async function settle() {
  await act(async () => {
    for (let tick = 0; tick < 100; tick += 1) await Promise.resolve()
  })
}

async function submitCredential() {
  fireEvent.change(screen.getByLabelText('Bearer 凭据'), { target: { value: TOKEN } })
  fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
  await settle()
}

async function signIn() {
  render(<App />)
  await submitCredential()
}

const confirmButton = () =>
  screen.getByRole('button', { name: '确认建议（本地）' }) as HTMLButtonElement
const refreshButton = () =>
  screen.getByRole('button', { name: '刷新当前 Review' }) as HTMLButtonElement
const isBusy = () => refreshButton().disabled

function select(runId: string) {
  fireEvent.click(document.querySelector(`[data-run-id="${runId}"]`)!)
}

/**
 * Activate two controls inside one React batch, so neither handler can observe the
 * other's `disabled` render.  This is the "two clicks before the next render" case:
 * a correct implementation needs a *synchronous* admission check, not just a
 * disabled attribute.  `ReactDOM.unstable_batchedUpdates` is the public way to hold
 * the render back; the product itself never uses it.
 */
async function activateTogether(...nodes: HTMLElement[]) {
  await act(async () => {
    unstable_batchedUpdates(() => {
      for (const node of nodes) node.click()
    })
  })
}

/** Two decidable Runs whose decisions answer with the Run that was written. */
function twoRunHarness(extra: Partial<Parameters<typeof installHarness>[0]> = {}): Harness {
  let writing = 'run-a'
  return installHarness({
    session: OPERATOR_SESSION,
    runs: [runWith('run-a', 'pending_confirmation'), runWith('run-b', 'pending_confirmation')],
    trace: (id) => makeTrace(id, []),
    audit: (id) => makeAudit(id, []),
    onWrite: (runId) => {
      writing = runId
    },
    actionResult: () => runWith(writing, 'confirmed'),
    ...extra,
  })
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('C-AUD-03 · one write slot per operation', () => {
  it('keeps both Runs busy while two decisions are in flight', async () => {
    vi.useFakeTimers()
    const harness = twoRunHarness()
    await signIn()

    const gateA = harness.hold((call) => call.url.endsWith('run-a/confirm'))
    const gateB = harness.hold((call) => call.url.endsWith('run-b/confirm'))
    fireEvent.click(confirmButton())
    await settle()
    select('run-b')
    await settle()
    fireEvent.click(confirmButton())
    await settle()
    select('run-a')
    await settle()

    // Both requests are genuinely in flight; A may not look idle because B started.
    expect(harness.callsTo(/\/confirm$/, 'POST')).toHaveLength(2)
    expect(confirmButton().disabled).toBe(true)
    expect(isBusy()).toBe(true)

    select('run-b')
    await settle()
    expect(confirmButton().disabled).toBe(true)

    gateA.resolve()
    gateB.resolve()
    await settle()
  })

  it('does not free another Run when the first decision completes', async () => {
    vi.useFakeTimers()
    const harness = twoRunHarness()
    await signIn()

    const gateB = harness.hold((call) => call.url.endsWith('run-b/confirm'))
    fireEvent.click(confirmButton())
    await settle()
    select('run-b')
    await settle()
    fireEvent.click(confirmButton())
    await settle()
    select('run-a')
    await settle()
    // A is settled and no longer busy; B still is.
    expect(isBusy()).toBe(false)

    gateB.resolve()
    await settle()
  })

  it('keeps the second Run busy after the first one answers', async () => {
    vi.useFakeTimers()
    const harness = twoRunHarness()
    await signIn()

    const gateA = harness.hold((call) => call.url.endsWith('run-a/confirm'))
    const gateB = harness.hold((call) => call.url.endsWith('run-b/confirm'))
    fireEvent.click(confirmButton())
    await settle()
    select('run-b')
    await settle()
    fireEvent.click(confirmButton())
    await settle()

    gateA.resolve()
    await settle()
    select('run-b')
    await settle()
    // A completing must not clear B's in-flight write.
    expect(isBusy()).toBe(true)

    gateB.resolve()
    await settle()
    expect(isBusy()).toBe(false)
  })

  it('keeps the first Run busy after the second one answers', async () => {
    vi.useFakeTimers()
    const harness = twoRunHarness()
    await signIn()

    const gateA = harness.hold((call) => call.url.endsWith('run-a/confirm'))
    const gateB = harness.hold((call) => call.url.endsWith('run-b/confirm'))
    fireEvent.click(confirmButton())
    await settle()
    select('run-b')
    await settle()
    fireEvent.click(confirmButton())
    await settle()

    gateB.resolve()
    await settle()
    select('run-a')
    await settle()
    expect(isBusy()).toBe(true)

    gateA.resolve()
    await settle()
    expect(isBusy()).toBe(false)
  })

  it('keeps a write busy when a refresh for another Run finishes in between', async () => {
    vi.useFakeTimers()
    const harness = twoRunHarness()
    await signIn()

    const heldRefresh = harness.hold(
      (call) => call.url === '/api/v2/reviews/run-a' && call.method === 'GET',
    )
    fireEvent.click(refreshButton())
    await settle()
    select('run-b')
    await settle()

    const gateB = harness.hold((call) => call.url.endsWith('run-b/confirm'))
    fireEvent.click(confirmButton())
    await settle()

    // The GET finishes while B's write is still in flight: it must not free it.
    heldRefresh.resolve()
    await settle()
    expect(isBusy()).toBe(true)

    gateB.resolve()
    await settle()
    expect(isBusy()).toBe(false)
  })

  it('sends only one write when the same control is clicked twice before a re-render', async () => {
    vi.useFakeTimers()
    const harness = twoRunHarness()
    await signIn()

    await activateTogether(confirmButton(), confirmButton())
    await settle()
    expect(harness.callsTo(/\/confirm$/, 'POST')).toHaveLength(1)
    expect(harness.callsTo(/\/cancel$/, 'POST')).toHaveLength(0)
  })

  it('keeps mutually exclusive actions on one Run from both being sent', async () => {
    vi.useFakeTimers()
    const harness = twoRunHarness()
    await signIn()

    await activateTogether(
      confirmButton(),
      screen.getByRole('button', { name: '取消' }),
    )
    await settle()
    expect(harness.callsTo(/\/confirm$/, 'POST')).toHaveLength(1)
    expect(harness.callsTo(/\/cancel$/, 'POST')).toHaveLength(0)
  })

  it('sends one replay when the replay control is clicked twice', async () => {
    vi.useFakeTimers()
    const harness = twoRunHarness()
    await signIn()

    fireEvent.click(confirmButton())
    await settle()
    const replay = screen.getByRole('button', { name: '重放原动作' }) as HTMLButtonElement
    expect(replay).toBeEnabled()
    await activateTogether(replay, replay)
    await settle()
    // Exactly the original decision plus one explicit replay.
    expect(harness.callsTo(/\/confirm$/, 'POST')).toHaveLength(2)
  })

  it('clears only the failing Run when one of two writes fails', async () => {
    vi.useFakeTimers()
    const harness = twoRunHarness({
      failOnce: { match: /run-b\/confirm/, method: 'POST', status: 409, code: 'version_conflict' },
    })
    await signIn()

    const gateA = harness.hold((call) => call.url.endsWith('run-a/confirm'))
    fireEvent.click(confirmButton())
    await settle()
    select('run-b')
    await settle()
    fireEvent.click(confirmButton())
    await settle()

    select('run-a')
    await settle()
    expect(isBusy()).toBe(true)

    gateA.resolve()
    await settle()
    expect(isBusy()).toBe(false)
  })

  it('does not let a superseded identity free the next session operation', async () => {
    vi.useFakeTimers()
    const harness = twoRunHarness()
    await signIn()

    const gateA = harness.hold((call) => call.url.endsWith('run-a/confirm'))
    fireEvent.click(confirmButton())
    await settle()

    fireEvent.click(screen.getByRole('button', { name: '身份与权限' })); fireEvent.click(screen.getByRole('button', { name: '退出并清除' }))
    await settle()
    await submitCredential()

    select('run-b')
    await settle()
    const gateB = harness.hold((call) => call.url.endsWith('run-b/confirm'))
    fireEvent.click(confirmButton())
    await settle()
    expect(isBusy()).toBe(true)

    // The previous identity's write finally settles.
    gateA.resolve()
    await settle()
    expect(isBusy()).toBe(true)

    gateB.resolve()
    await settle()
    expect(isBusy()).toBe(false)
  })
})
