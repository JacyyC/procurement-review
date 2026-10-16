/**
 * FIX-06C — the two response-ownership gaps the 2026-09-15 review reproduced.
 *
 * R-03: a structurally valid create response that describes a *different* Case was
 * still cached and selected.  The Case is frozen before the request is sent, so the
 * response has to be bound to that same Case; a mismatch is a bad success payload,
 * not a new review.
 *
 * R-04: a late detail read with the *same* `revision` rolled the displayed progress
 * back (3 events after the human had already seen 8).  `revision` guards lifecycle
 * order; the request ticket decides between two reads of the same revision.  Both
 * counter-examples drive the real `App` against a controllable `fetch` whose held
 * responses ignore `AbortSignal`, so only the ownership rule can keep the page right.
 */

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import { createReview } from './api'
import { ReviewAPIError } from './apiErrors'
import {
  OPERATOR_SESSION,
  TOKEN,
  deferred,
  installHarness,
  jsonResponse,
  makeAudit,
  makeRun,
  makeTrace,
} from './fix01cHarness'
import type { Deferred } from './fix01cHarness'
import type { CaseListResponse } from './types'

const CASES: CaseListResponse = {
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
  ],
}

const AUTH = { token: TOKEN, credentialId: 'cred-1' }

type Matcher = (url: string, method: string) => boolean

const isGetRun = (runId: string): Matcher => (url, method) =>
  method === 'GET' && url === `/api/v2/reviews/${runId}`

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

/** The event count the "运行 Trace" tab shows for the active Run. */
function traceTabCount(): number | null {
  const tab = screen.queryByRole('tab', { name: /运行 Trace/ })
  if (!tab) return null
  const match = tab.textContent?.match(/(\d+)/)
  return match ? Number(match[1]) : null
}

/**
 * Hold one matching response until the test releases it.
 *
 * The body is read from the server *now* and delivered only when the test releases
 * it, which is what a slow response really is: the answer was produced from the
 * state at request time and arrives after newer reads already landed.  Rules are
 * one-shot, so the held call is exactly the one the test names.
 */
function holdResponse(match: Matcher) {
  const base = globalThis.fetch
  const held: Deferred<void> = deferred()
  const rules: { match: Matcher; release: Deferred<void> }[] = [{ match, release: held }]
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString()
      const method = (init?.method ?? 'GET').toUpperCase()
      const rule = rules.find((item) => item.match(url, method))
      if (!rule) return base(input, init)
      rules.splice(rules.indexOf(rule), 1)
      const response = await base(input, init)
      await rule.release.promise
      return response
    }),
  )
  return { release: () => held.resolve() }
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('FIX-06C create response binding (R-03)', () => {
  it('rejects a create response that describes another Case', async () => {
    const mismatched = makeRun('run-other-case', {
      status: 'processing',
      request: {
        ...makeRun('run-other-case').request,
        case_id: 'CASE-HIGH-001',
      },
    })
    const base = globalThis.fetch
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const method = (init?.method ?? 'GET').toUpperCase()
        if (method === 'POST') return jsonResponse(mismatched, 202)
        return base(input, init)
      }),
    )

    const error = await createReview(
      'CASE-LOW-001',
      'case-low-001.v1',
      'fix06c-key',
      AUTH,
    ).then(
      () => null,
      (reason: unknown) => reason,
    )

    expect(error).toBeInstanceOf(ReviewAPIError)
    expect((error as ReviewAPIError).code).toBe('invalid_response')
    // The real HTTP status is preserved for the operator and the record.
    expect((error as ReviewAPIError).status).toBe(202)
  })

  it('accepts a create response for the Case that was frozen before sending', async () => {
    const matching = makeRun('run-matching', {
      status: 'processing',
      request: { ...makeRun('run-matching').request, case_id: 'CASE-LOW-001' },
    })
    const base = globalThis.fetch
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const method = (init?.method ?? 'GET').toUpperCase()
        if (method === 'POST') return jsonResponse(matching, 202)
        return base(input, init)
      }),
    )

    const run = await createReview('CASE-LOW-001', 'case-low-001.v1', 'fix06c-ok', AUTH)
    expect(run.run_id).toBe('run-matching')
  })

  it('does not cache, select or adopt a Run whose Case does not match', async () => {
    const mismatched = makeRun('run-wrong-case', {
      status: 'processing',
      request: {
        ...makeRun('run-wrong-case').request,
        case_id: 'CASE-HIGH-001',
      },
    })
    // The server answers this request with a body that describes another Case and
    // commits nothing, so the list reconcile after the failure finds no review: the
    // only way the wrong Run could appear is if the client adopted the bad body.
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [],
      cases: CASES.cases,
      overrideOnce: {
        match: /^\/api\/v2\/reviews$/,
        method: 'POST',
        response: () => jsonResponse(mismatched, 202),
      },
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
    })

    await signIn()
    fireEvent.click(screen.getByRole('button', { name: '运行审查' }))
    await settle()

    // The send really happened, and the bad payload never became a review.
    expect(harness.callsTo(/^\/api\/v2\/reviews$/, 'POST')).toHaveLength(1)
    expect(document.querySelector('[data-run-id="run-wrong-case"]')).toBeNull()
    expect(document.querySelector('[aria-current="true"]')).toBeNull()
    expect(harness.runs()).toEqual([])
    // The request itself is unresolved, so the session keeps it for an explicit
    // same-key replay instead of guessing that the create succeeded.
    expect(screen.getByRole('button', { name: '重放最近创建请求' })).toBeEnabled()
  })

  it('adopts a create response that matches the frozen Case', async () => {
    const matching = makeRun('run-right-case', {
      status: 'processing',
      request: { ...makeRun('run-right-case').request, case_id: 'CASE-LOW-001' },
    })
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [],
      cases: CASES.cases,
      createResult: () => matching,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
    })

    await signIn()
    fireEvent.click(screen.getByRole('button', { name: '运行审查' }))
    await settle()

    expect(harness.callsTo(/^\/api\/v2\/reviews$/, 'POST')).toHaveLength(1)
    expect(document.querySelector('[data-run-id="run-right-case"]')).not.toBeNull()
    expect(document.querySelector('[aria-current="true"]')).not.toBeNull()
  })
})

describe('FIX-06C same-revision detail order (R-04)', () => {
  it('keeps the newer progress when a same-revision read arrives late', async () => {
    const stale = makeRun('run-progress', {
      status: 'processing',
      revision: 1,
      version: 1,
      trace_event_count: 3,
    })
    const fresh = makeRun('run-progress', {
      status: 'processing',
      revision: 1,
      version: 1,
      trace_event_count: 8,
    })
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [stale],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, new Array(3).fill('worker.policy')),
      audit: (runId) => makeAudit(runId, []),
    })
    await signIn()
    const gate = holdResponse(isGetRun('run-progress'))

    // The human selects the Run; that read is held while the page goes on.
    select('run-progress')
    await settle()
    harness.putRun(fresh)
    refresh()
    await settle()
    // The refresh read really landed: the page shows the newer count.
    expect(traceTabCount()).toBe(8)

    // Now the older read is released.  Same revision, older dynamic fields.
    gate.release()
    await settle()

    expect(traceTabCount()).toBe(8)
  })

  it('still applies a read through the newest ticket for the same Run', async () => {
    const first = makeRun('run-ticket', { status: 'processing', revision: 1, version: 1 })
    installHarness({
      session: OPERATOR_SESSION,
      runs: [first],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, new Array(9).fill('worker.policy')),
      audit: (runId) => makeAudit(runId, []),
    })
    await signIn()
    select('run-ticket')
    await settle()

    expect(traceTabCount()).not.toBeNull()
    expect(document.querySelector('[data-run-id="run-ticket"]')).not.toBeNull()
  })

  it('drops a superseded refresh read instead of rolling the Run back', async () => {
    const stale = makeRun('run-refresh', {
      status: 'pending_confirmation',
      revision: 4,
      version: 4,
      trace_event_count: 2,
    })
    const fresh = makeRun('run-refresh', {
      status: 'pending_confirmation',
      revision: 4,
      version: 4,
      trace_event_count: 7,
    })
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [stale],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
    })
    await signIn()
    select('run-refresh')
    await settle()
    expect(traceTabCount()).toBe(2)

    // The manual refresh is held while the human re-selects the same Run, which takes a
    // newer ticket for the same key and lands first.
    const gate = holdResponse(isGetRun('run-refresh'))
    harness.putRun(fresh)
    refresh()
    await settle()
    select('run-refresh')
    await settle()
    expect(traceTabCount()).toBe(7)

    gate.release()
    await settle()

    expect(traceTabCount()).toBe(7)
  })

  it('answers only one read when refresh is clicked twice', async () => {
    const current = makeRun('run-double', {
      status: 'pending_confirmation',
      revision: 3,
      version: 3,
      trace_event_count: 4,
    })
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [current],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
    })
    await signIn()
    select('run-double')
    await settle()

    const gate = holdResponse(isGetRun('run-double'))
    refresh()
    await settle()
    // The first refresh still occupies the Run's refresh slot, so the second click is
    // admitted as a duplicate instead of opening a second in-flight read.
    refresh()
    await settle()
    gate.release()
    await settle()

    expect(harness.callsTo(/\/reviews\/run-double$/)).toHaveLength(2)
    expect(traceTabCount()).toBe(4)
  })

  it('does not let a superseded poll tick overwrite a newer read', async () => {
    const processing = makeRun('run-poll', {
      status: 'processing',
      revision: 1,
      version: 1,
      trace_event_count: 1,
    })
    const newer = makeRun('run-poll', {
      status: 'processing',
      revision: 1,
      version: 1,
      trace_event_count: 6,
    })
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [processing],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
    })
    await signIn()
    select('run-poll')
    await settle()

    // The next poll tick is answered from the older state and then held.
    const gate = holdResponse(isGetRun('run-poll'))
    const deadline = Date.now() + 2000
    while (harness.callsTo(/\/reviews\/run-poll$/).length < 2 && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 20))
    }
    expect(harness.callsTo(/\/reviews\/run-poll$/).length).toBeGreaterThanOrEqual(2)

    harness.putRun(newer)
    refresh()
    await settle()
    expect(traceTabCount()).toBe(6)

    gate.release()
    await act(async () => {
      await Promise.resolve()
    })
    expect(traceTabCount()).toBe(6)
  })

  it('stops watching once a poll tick sees the Run settle', async () => {
    const processing = makeRun('run-settle', {
      status: 'processing',
      revision: 1,
      version: 1,
      trace_event_count: 2,
    })
    const settled = makeRun('run-settle', {
      status: 'pending_confirmation',
      revision: 2,
      version: 2,
      trace_event_count: 5,
    })
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [processing],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
    })
    await signIn()
    select('run-settle')
    await settle()
    expect(traceTabCount()).toBe(2)

    // The Run settles between two ticks; the tick that sees it caches the settled
    // state and stops the watch instead of leaving the page at 处理中.
    harness.putRun(settled)
    await waitFor(() => expect(traceTabCount()).toBe(5), { timeout: 3000 })
  })

  it('keeps the tab order legal when an arrow key targets the tab list itself', async () => {
    const run = makeRun('run-tabs', { status: 'pending_confirmation', revision: 2, version: 2 })
    installHarness({
      session: OPERATOR_SESSION,
      runs: [run],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, ['worker.policy']),
      audit: (runId) => makeAudit(runId, []),
    })
    await signIn()
    select('run-tabs')
    await settle()

    // Roving focus has to stay inside the rendered tabs even when the event's target is
    // the list rather than one of its tabs (no focusable tab to read).
    const list = screen.getByRole('tablist', { name: 'Review 视图' })
    fireEvent.keyDown(list, { key: 'ArrowRight' })
    await settle()

    const tabs = screen.getAllByRole('tab')
    const selected = tabs.filter((tab) => tab.getAttribute('aria-selected') === 'true')
    expect(selected).toHaveLength(1)
    expect(tabs.some((tab) => tab.getAttribute('tabindex') === '0')).toBe(true)
  })

  it('does not let two Runs fight over one another’s ticket', async () => {
    const alpha = makeRun('run-alpha', { status: 'processing', revision: 1, version: 1 })
    const beta = makeRun('run-beta', {
      status: 'processing',
      revision: 1,
      version: 1,
      trace_event_count: 5,
    })
    installHarness({
      session: OPERATOR_SESSION,
      runs: [alpha, beta],
      cases: CASES.cases,
      trace: (runId) => makeTrace(runId, []),
      audit: (runId) => makeAudit(runId, []),
    })
    await signIn()
    const gate = holdResponse(isGetRun('run-alpha'))

    select('run-alpha')
    await settle()
    select('run-beta')
    await settle()
    gate.release()
    await settle()

    // The late alpha read updates (or leaves) alpha's own entry and never steals
    // the selection back from beta.
    const current = document.querySelector('[aria-current="true"]')
    expect(current?.getAttribute('data-run-id')).toBe('run-beta')
  })
})
