/**
 * FIX-04A / C-AUD-02 — recovering the *view* must not drop a pending write.
 *
 * The render boundary exists so a broken view never becomes a blank page: its reset
 * clears derived view state and re-issues reads.  It must not, however, throw away the
 * session's record of a write whose outcome is still unknown — that record is the only
 * way the operator can replay the original request with its original idempotency key.
 *
 * The view is made to fail through a test-local mock of the view module, so no fault
 * switch exists in the product build.
 */

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
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

const renderGate = vi.hoisted(() => ({ armed: false }))

vi.mock('./ReviewViews', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./ReviewViews')>()
  return {
    ...actual,
    ResultView: (props: Parameters<typeof actual.ResultView>[0]) => {
      if (renderGate.armed) throw new Error('FIX04-RENDER-CANARY')
      return actual.ResultView(props)
    },
  }
})

function runWith(id: string, status: ReviewRun['status']): ReviewRun {
  const run = structuredClone(reviewRun(status))
  run.run_id = id
  run.trace_id = `trace-${id}`
  if (run.summary) run.summary.run_id = id
  if (run.receipt) run.receipt.run_id = id
  if (run.execution && typeof run.execution === 'object' && 'summary' in run.execution) {
    ;(run.execution as { summary: { run_id: string } }).summary.run_id = id
  }
  return run
}

async function settle() {
  await act(async () => {
    for (let tick = 0; tick < 100; tick += 1) await Promise.resolve()
  })
}

function install(): Harness {
  return installHarness({
    session: OPERATOR_SESSION,
    runs: [runWith('run-a', 'pending_confirmation')],
    actionResult: () => runWith('run-a', 'confirmed'),
    loseResponseOnce: { match: /\/confirm$/, method: 'POST' },
    trace: (id) => makeTrace(id, []),
    audit: (id) => makeAudit(id, []),
  })
}

afterEach(() => {
  renderGate.armed = false
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('C-AUD-02 · view recovery keeps the original request', () => {
  it('keeps the pending request across a boundary reset and never re-sends it', async () => {
    vi.useFakeTimers()
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const harness = install()
    render(<App />)
    fireEvent.change(screen.getByLabelText('Bearer 凭据'), { target: { value: TOKEN } })
    fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
    await settle()

    fireEvent.click(screen.getByRole('button', { name: '确认建议（本地）' }))
    await settle()
    expect(harness.callsTo(/\/confirm$/, 'POST')).toHaveLength(1)
    expect(screen.getByRole('button', { name: '重放原动作' })).toBeEnabled()

    renderGate.armed = true
    fireEvent.click(screen.getByRole('button', { name: '刷新当前 Review' }))
    await settle()
    expect(screen.getByTestId('review-error-boundary')).toBeInTheDocument()

    renderGate.armed = false
    fireEvent.click(screen.getByRole('button', { name: '重新读取视图' }))
    await settle()

    // The reset re-read only: the request context survives and nothing was re-sent.
    expect(screen.queryByTestId('review-error-boundary')).toBeNull()
    expect(harness.callsTo(/\/confirm$/, 'POST')).toHaveLength(1)
    expect(screen.getByRole('button', { name: '重放原动作' })).toBeEnabled()

    fireEvent.click(screen.getByRole('button', { name: '重放原动作' }))
    await settle()
    expect(harness.callsTo(/\/confirm$/, 'POST')).toHaveLength(2)
  })

  it('offers the credential escape hatch from the boundary without a write', async () => {
    vi.useFakeTimers()
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const harness = install()
    render(<App />)
    fireEvent.change(screen.getByLabelText('Bearer 凭据'), { target: { value: TOKEN } })
    fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
    await settle()

    renderGate.armed = true
    fireEvent.click(screen.getByRole('button', { name: '刷新当前 Review' }))
    await settle()
    expect(screen.getByTestId('review-error-boundary')).toBeInTheDocument()

    renderGate.armed = false
    fireEvent.click(screen.getByRole('button', { name: '清除凭据并返回登录' }))
    await settle()
    expect(screen.getByText('需要本地凭据')).toBeInTheDocument()
    expect(harness.callsTo(/\/confirm$/, 'POST')).toHaveLength(0)
    expect(harness.callsTo(/\/cancel$/, 'POST')).toHaveLength(0)
    expect(harness.callsTo(/\/resume$/, 'POST')).toHaveLength(0)
  })
})
