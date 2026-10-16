/**
 * FIX-04A / C-AUD-01 — request ownership inside the workbench.
 *
 * Every case drives the real `App` against a controllable `fetch` whose held
 * responses **ignore `AbortSignal`**, so the only thing that can keep the display
 * correct is the ownership rule (identity generation + selection epoch +
 * per-resource request ticket) rather than cancellation.  Release order is decided
 * by the test, never by a sleep.
 */

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import {
  OPERATOR_SESSION,
  TOKEN,
  deferred,
  errorResponse,
  installHarness,
  jsonResponse,
  makeAudit,
  makeRun,
  makeTrace,
} from './fix01cHarness'
import type { Deferred } from './fix01cHarness'
import type { ReviewRun } from './types'

type Matcher = (url: string, method: string) => boolean

const isTrace: Matcher = (url) => url.endsWith('/trace')
const isAudit: Matcher = (url) => url.endsWith('/audit')

async function settle() {
  await act(async () => {
    for (let tick = 0; tick < 100; tick += 1) await Promise.resolve()
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

function openTab(name: RegExp) {
  fireEvent.click(screen.getByRole('tab', { name }))
}

function refresh() {
  fireEvent.click(screen.getByRole('button', { name: '刷新当前 Review' }))
}

/**
 * Intercept `fetch` *in front of* the fixture harness so one specific call can be
 * answered with a caller-chosen response — immediately, or only when the test
 * releases it.  Rules are one-shot and consumed in registration order, which is how
 * a test decides the exact interleaving instead of sleeping.
 */
function intercept() {
  const base = globalThis.fetch
  const rules: { match: Matcher; answer: () => Response | Promise<Response> }[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const method = (init?.method ?? 'GET').toUpperCase()
      const index = rules.findIndex((rule) => rule.match(url, method))
      if (index >= 0) {
        const [rule] = rules.splice(index, 1)
        return Promise.resolve(rule.answer())
      }
      return base(input, init)
    }),
  )
  return {
    /** Answer the next matching call with this HTTP failure. */
    failNext(match: Matcher, status: number, code: string) {
      rules.push({ match, answer: () => errorResponse(status, code) })
    },
    /** Hold the next matching call until the test releases it with a response. */
    holdNext(match: Matcher): Deferred<Response> {
      const gate = deferred<Response>()
      rules.push({ match, answer: () => gate.promise })
      return gate
    },
    /** Answer the next matching call with this success body. */
    answerNext(match: Matcher, body: unknown, status = 200) {
      rules.push({ match, answer: () => jsonResponse(body, status) })
    },
  }
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('C-AUD-01 · Trace ownership', () => {
  it('keeps the newer Trace when an older read fails last', async () => {
    vi.useFakeTimers()
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: (id) => makeTrace(id, ['run_started']),
      audit: (id) => makeAudit(id, []),
    })
    const gate = intercept()
    await signIn()

    const held = gate.holdNext(isTrace)
    select('run-a')
    await settle()
    harness.putTrace('run-a', makeTrace('run-a', ['run_completed']))
    refresh()
    await settle()
    openTab(/运行 Trace/)
    await settle()
    expect(screen.queryByText('run_completed')).not.toBeNull()

    held.resolve(errorResponse(503, 'security_dependency_unavailable'))
    await settle()
    // The held read was superseded: its 503 may not clear the newer Trace.
    expect(screen.queryByText('run_completed')).not.toBeNull()
  })

  it('keeps the newer Audit when an older read fails last', async () => {
    vi.useFakeTimers()
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: (id) => makeTrace(id, []),
      audit: (id) => makeAudit(id, ['subject-earlier']),
    })
    const gate = intercept()
    await signIn()

    const held = gate.holdNext(isAudit)
    select('run-a')
    await settle()
    harness.putAudit('run-a', makeAudit('run-a', ['subject-newest']))
    refresh()
    await settle()
    openTab(/审批审计/)
    await settle()
    expect(screen.queryByText('subject-newest')).not.toBeNull()

    held.resolve(errorResponse(503, 'security_dependency_unavailable'))
    await settle()
    expect(screen.queryByText('subject-newest')).not.toBeNull()
  })

  it('does not let an older success refill a slot a newer failure emptied', async () => {
    vi.useFakeTimers()
    installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: (id) => makeTrace(id, ['run_started']),
      audit: (id) => makeAudit(id, []),
    })
    const gate = intercept()
    await signIn()

    const held = gate.holdNext(isTrace)
    gate.failNext(isTrace, 503, 'security_dependency_unavailable')
    select('run-a')
    await settle()
    refresh()
    await settle()
    openTab(/运行 Trace/)
    await settle()
    expect(screen.queryByText('run_started')).toBeNull()

    held.resolve(jsonResponse(makeTrace('run-a', ['run_started'])))
    await settle()
    // The superseded success must not resurrect what the newer failure cleared.
    expect(screen.queryByText('run_started')).toBeNull()
  })

  it('does not apply a superseded response while a newer read is still in flight', async () => {
    vi.useFakeTimers()
    // The workspace load answers with an empty Trace so the only values that can
    // appear later are the ones this test releases.
    let traceValue = makeTrace('run-a', [])
    installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: () => traceValue,
      audit: (id) => makeAudit(id, []),
    })
    const gate = intercept()
    await signIn()

    const first = gate.holdNext(isTrace)
    select('run-a')
    await settle()
    traceValue = makeTrace('run-a', ['run_completed'])
    const second = gate.holdNext(isTrace)
    refresh()
    await settle()

    first.resolve(jsonResponse(makeTrace('run-a', ['run_started'])))
    await settle()
    openTab(/运行 Trace/)
    await settle()
    // A newer read is in flight: the older answer is superseded, not merely late.
    expect(screen.queryByText('run_started')).toBeNull()

    second.resolve(jsonResponse(makeTrace('run-a', ['run_completed'])))
    await settle()
    expect(screen.queryByText('run_completed')).not.toBeNull()
  })

  it('applies Trace and Audit independently instead of waiting for the slower one', async () => {
    vi.useFakeTimers()
    installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: (id) => makeTrace(id, ['run_completed']),
      audit: (id) => makeAudit(id, ['subject-demo']),
    })
    const gate = intercept()
    await signIn()

    const heldAudit = gate.holdNext(isAudit)
    select('run-a')
    await settle()
    openTab(/运行 Trace/)
    await settle()
    // The slow Audit read must not hold back an already-settled Trace.
    expect(screen.queryByText('run_completed')).not.toBeNull()

    heldAudit.resolve(jsonResponse(makeAudit('run-a', ['subject-demo'])))
    await settle()
    expect(screen.queryByText('run_completed')).not.toBeNull()
    openTab(/审批审计/)
    await settle()
    expect(screen.queryByText('subject-demo')).not.toBeNull()
  })

  it('does not let a superseded failure replace a newer notice', async () => {
    vi.useFakeTimers()
    installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: (id) => makeTrace(id, ['run_started']),
      audit: (id) => makeAudit(id, []),
    })
    const gate = intercept()
    await signIn()

    const held = gate.holdNext(isTrace)
    select('run-a')
    await settle()
    refresh()
    await settle()
    expect(screen.queryByText('已读取最新状态。')).not.toBeNull()

    // The superseded read fails *after* the newer refresh reported success.
    held.resolve(errorResponse(503, 'security_dependency_unavailable'))
    await settle()
    expect(screen.queryByText('已读取最新状态。')).not.toBeNull()
    expect(screen.queryByText(/security_dependency_unavailable/)).toBeNull()
  })

  it('keeps each Run on its own Trace cache across A → B → A', async () => {
    vi.useFakeTimers()
    installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a'), makeRun('run-b')],
      trace: (id) => makeTrace(id, [`event-${id}`]),
      audit: (id) => makeAudit(id, []),
    })
    const gate = intercept()
    await signIn()

    const heldA = gate.holdNext((url) => url.includes('/reviews/run-a/trace'))
    select('run-a')
    await settle()
    select('run-b')
    await settle()

    heldA.resolve(jsonResponse(makeTrace('run-a', ['event-run-a'])))
    await settle()
    openTab(/运行 Trace/)
    await settle()
    // A's late Trace belongs to A's cache; it may not take over B's panel.
    expect(screen.queryByText('event-run-b')).not.toBeNull()
    expect(screen.queryByText('event-run-a')).toBeNull()

    select('run-a')
    await settle()
    openTab(/运行 Trace/)
    await settle()
    expect(screen.queryByText('event-run-a')).not.toBeNull()
  })

  it('never lets an older revision roll a Run back', async () => {
    vi.useFakeTimers()
    const stale: ReviewRun = { ...makeRun('run-b'), revision: 2, version: 2 }
    const fresh: ReviewRun = { ...makeRun('run-b'), revision: 7, version: 7 }
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a'), stale],
      trace: (id) => makeTrace(id, []),
      audit: (id) => makeAudit(id, []),
    })
    const gate = intercept()
    await signIn()

    const held = gate.holdNext((url) => url === '/api/v2/reviews/run-b')
    select('run-b')
    await settle()
    harness.putRun(fresh)
    refresh()
    await settle()
    expect(screen.getByText('v7')).toBeInTheDocument()

    held.resolve(jsonResponse(stale))
    await settle()
    expect(screen.getByText('v7')).toBeInTheDocument()
  })

  it('ends the session on a genuine 401 from an artifact read', async () => {
    vi.useFakeTimers()
    installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: (id) => makeTrace(id, []),
      audit: (id) => makeAudit(id, []),
    })
    const gate = intercept()
    await signIn()
    gate.failNext(isAudit, 401, 'unauthenticated')
    select('run-a')
    await settle()
    expect(screen.getByText('需要本地凭据')).toBeInTheDocument()
  })

  it('discards every result of a superseded identity', async () => {
    vi.useFakeTimers()
    installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: (id) => makeTrace(id, ['run_completed']),
      audit: (id) => makeAudit(id, []),
    })
    const gate = intercept()
    await signIn()

    const held = gate.holdNext((url) => url === '/api/v2/reviews/run-a')
    select('run-a')
    await settle()
    // Sign out while the read is still in flight, then release it.
    fireEvent.click(screen.getByRole('button', { name: '身份与权限' })); fireEvent.click(screen.getByRole('button', { name: '退出并清除' }))
    await settle()
    held.resolve(jsonResponse(makeRun('run-a')))
    await settle()
    expect(screen.getByText('需要本地凭据')).toBeInTheDocument()
    // Nothing from the previous identity may survive into the credential screen.
    expect(document.querySelector('.run-title')).toBeNull()
    expect(document.querySelector('.review-view')).toBeNull()
  })

  it('does not apply a response released after unmount', async () => {
    vi.useFakeTimers()
    installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: (id) => makeTrace(id, ['run_completed']),
      audit: (id) => makeAudit(id, []),
    })
    const gate = intercept()
    await signIn()
    const held = gate.holdNext(isTrace)
    select('run-a')
    await settle()
    cleanup()
    held.resolve(jsonResponse(makeTrace('run-a', ['run_completed'])))
    await settle()
    expect(document.body.textContent).not.toContain('run_completed')
  })
})
