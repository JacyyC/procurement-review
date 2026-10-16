/**
 * FIX-04A / C-AUD-02 — the original write request must survive a lost response.
 *
 * The audit found that the workbench only remembered a decision *after* the server
 * answered successfully, in a single slot that selecting another Run cleared.  A
 * confirm that really committed but whose response never arrived therefore left the
 * operator with a receipt they could not replay; A → B → A lost the context too, and
 * create/resume kept no context at all.
 *
 * These cases pin the contract: the request context is frozen **before** the send, it
 * is keyed by the object it belongs to, losing the response does not erase it, and the
 * only thing that may send the original payload again is an explicit replay click that
 * reuses the identical idempotency key and body.
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
import type { CaseOption, ReviewRun } from './types'

const CASE: CaseOption = {
  api_version: 'review-api.v2',
  case_id: 'CASE-LOW-001',
  case_version: 'case-version-001',
  label: '低风险完整证据',
  case_type: 'purchase_order',
  scenario: 'low',
  document_version: 'retrieval-fixture-v1',
  dataset_version: 'synthetic-procurement.v1',
  index_version: 'index-fixture-v1',
  rules_version: '2.0.0',
}

/** A Run of the requested lifecycle status, carrying its own identity. */
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

function confirm() {
  fireEvent.click(screen.getByRole('button', { name: '确认建议（本地）' }))
}

function select(runId: string) {
  fireEvent.click(document.querySelector(`[data-run-id="${runId}"]`)!)
}

function refresh() {
  fireEvent.click(screen.getByRole('button', { name: '刷新当前 Review' }))
}

function replayButton() {
  return screen.getByRole('button', { name: '重放原动作' }) as HTMLButtonElement
}

/** Install a harness whose artifact reads answer with empty, valid payloads. */
function install(options: Parameters<typeof installHarness>[0]): Harness {
  return installHarness({
    trace: (id) => makeTrace(id, []),
    audit: (id) => makeAudit(id, []),
    ...options,
  })
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('C-AUD-02 · a lost response keeps the original request', () => {
  it('offers an explicit replay after a confirm commits but its response is lost', async () => {
    vi.useFakeTimers()
    const harness = install({
      session: OPERATOR_SESSION,
      runs: [runWith('run-a', 'pending_confirmation')],
      actionResult: () => runWith('run-a', 'confirmed'),
      loseResponseOnce: { match: /\/confirm$/, method: 'POST' },
    })
    await signIn()
    confirm()
    await settle()

    const posts = harness.callsTo(/\/confirm$/, 'POST')
    // Losing the response is not a reason to send the write again.
    expect(posts).toHaveLength(1)
    // The GET re-read still shows the real, committed state.
    expect(screen.queryByRole('heading', { name: '本地模拟回执' })).not.toBeNull()
    expect(replayButton()).toBeEnabled()
    expect(harness.callsTo(/\/confirm$/, 'POST')).toHaveLength(1)

    fireEvent.click(replayButton())
    await settle()

    const after = harness.callsTo(/\/confirm$/, 'POST')
    expect(after).toHaveLength(2)
    expect(after[1].headers.get('Idempotency-Key')).toBe(
      posts[0].headers.get('Idempotency-Key'),
    )
    expect(after[1].body).toEqual(posts[0].body)
  })

  it('offers the same-key replay when the confirm never reached the server', async () => {
    vi.useFakeTimers()
    const harness = install({
      session: OPERATOR_SESSION,
      runs: [runWith('run-a', 'pending_confirmation')],
      actionResult: () => runWith('run-a', 'confirmed'),
      networkErrorOnce: { match: /\/confirm$/, method: 'POST' },
    })
    await signIn()
    confirm()
    await settle()

    const posts = harness.callsTo(/\/confirm$/, 'POST')
    expect(posts).toHaveLength(1)
    expect(replayButton()).toBeEnabled()

    fireEvent.click(replayButton())
    await settle()
    const after = harness.callsTo(/\/confirm$/, 'POST')
    expect(after).toHaveLength(2)
    expect(after[1].headers.get('Idempotency-Key')).toBe(
      posts[0].headers.get('Idempotency-Key'),
    )
    expect(after[1].body).toEqual(posts[0].body)
  })

  it('keeps the decision context across A → B → A in the same session', async () => {
    vi.useFakeTimers()
    install({
      session: OPERATOR_SESSION,
      runs: [runWith('run-a', 'pending_confirmation'), runWith('run-b', 'pending_confirmation')],
      actionResult: () => runWith('run-a', 'confirmed'),
    })
    await signIn()
    confirm()
    await settle()
    expect(replayButton()).toBeEnabled()

    select('run-b')
    await settle()
    select('run-a')
    await settle()
    expect(replayButton()).toBeEnabled()
  })

  it('keeps a decision replayable even after the Run moved to a newer version', async () => {
    vi.useFakeTimers()
    const harness = install({
      session: OPERATOR_SESSION,
      runs: [runWith('run-a', 'pending_confirmation')],
      actionResult: () => runWith('run-a', 'confirmed'),
      loseResponseOnce: { match: /\/confirm$/, method: 'POST' },
    })
    await signIn()
    confirm()
    await settle()

    harness.putRun(runWith('run-a', 'confirmed', { version: 9, revision: 9 }))
    refresh()
    await settle()
    expect(screen.getByText('v9')).toBeInTheDocument()
    expect(replayButton()).toBeEnabled()

    fireEvent.click(replayButton())
    await settle()
    const after = harness.callsTo(/\/confirm$/, 'POST')
    expect(after).toHaveLength(2)
    // The replay resends the payload frozen at the original click, not the new state.
    expect(after[1].body).toEqual({
      api_version: 'review-api.v2',
      expected_version: 2,
      expected_summary_checksum: `sha256:${'a'.repeat(64)}`,
    })
    expect(after[1].headers.get('Idempotency-Key')).toBe(
      harness.callsTo(/\/confirm$/, 'POST')[0].headers.get('Idempotency-Key'),
    )
  })

  it('keeps a cancel replayable although a cancellation produces no receipt', async () => {
    vi.useFakeTimers()
    const harness = install({
      session: OPERATOR_SESSION,
      runs: [runWith('run-a', 'pending_confirmation')],
      actionResult: () => runWith('run-a', 'cancelled'),
      loseResponseOnce: { match: /\/cancel$/, method: 'POST' },
    })
    await signIn()
    fireEvent.click(screen.getByRole('button', { name: '取消' }))
    await settle()

    const posts = harness.callsTo(/\/cancel$/, 'POST')
    expect(posts).toHaveLength(1)
    expect(screen.queryByText('本地模拟回执')).toBeNull()
    expect(replayButton()).toBeEnabled()

    fireEvent.click(replayButton())
    await settle()
    const after = harness.callsTo(/\/cancel$/, 'POST')
    expect(after).toHaveLength(2)
    expect(after[1].headers.get('Idempotency-Key')).toBe(
      posts[0].headers.get('Idempotency-Key'),
    )
    expect(after[1].body).toEqual(posts[0].body)
    expect(after[1].body).not.toHaveProperty('expected_summary_checksum')
  })

  it('keeps a resume replayable with its original expected_version and key', async () => {
    vi.useFakeTimers()
    const harness = install({
      session: OPERATOR_SESSION,
      runs: [runWith('run-a', 'interrupted')],
      resumeResult: () => runWith('run-a', 'interrupted'),
      loseResponseOnce: { match: /\/resume$/, method: 'POST' },
    })
    await signIn()
    fireEvent.click(screen.getByTestId('resume-button'))
    await settle()

    const posts = harness.callsTo(/\/resume$/, 'POST')
    expect(posts).toHaveLength(1)
    expect(replayButton()).toBeEnabled()

    fireEvent.click(replayButton())
    await settle()
    const after = harness.callsTo(/\/resume$/, 'POST')
    expect(after).toHaveLength(2)
    expect(after[1].headers.get('Idempotency-Key')).toBe(
      posts[0].headers.get('Idempotency-Key'),
    )
    expect(after[1].body).toEqual(posts[0].body)
    expect(after[1].body).toEqual({
      api_version: 'review-api.v2',
      expected_version: 2,
    })
  })

  it('keeps a create replayable when the response is lost, using the same key and body', async () => {
    vi.useFakeTimers()
    const harness = install({
      session: OPERATOR_SESSION,
      cases: [CASE],
      runs: [],
      createResult: () => runWith('run-new', 'processing'),
      loseResponseOnce: { match: /^\/api\/v2\/reviews$/, method: 'POST' },
    })
    await signIn()
    fireEvent.click(screen.getByRole('button', { name: '运行审查' }))
    await settle()

    const posts = harness.callsTo(/^\/api\/v2\/reviews$/, 'POST')
    expect(posts).toHaveLength(1)
    const replayCreate = screen.getByRole('button', {
      name: '重放最近创建请求',
    }) as HTMLButtonElement
    expect(replayCreate).toBeEnabled()

    fireEvent.click(replayCreate)
    await settle()
    const after = harness.callsTo(/^\/api\/v2\/reviews$/, 'POST')
    expect(after).toHaveLength(2)
    expect(after[1].headers.get('Idempotency-Key')).toBe(
      posts[0].headers.get('Idempotency-Key'),
    )
    expect(after[1].body).toEqual(posts[0].body)
  })

  it('drops the original request context when the credential is replaced', async () => {
    vi.useFakeTimers()
    install({
      session: OPERATOR_SESSION,
      cases: [CASE],
      runs: [runWith('run-a', 'pending_confirmation')],
      actionResult: () => runWith('run-a', 'confirmed'),
    })
    await signIn()
    confirm()
    await settle()
    expect(replayButton()).toBeEnabled()

    fireEvent.click(screen.getByRole('button', { name: '身份与权限' })); fireEvent.click(screen.getByRole('button', { name: '退出并清除' }))
    await settle()
    await submitCredential()

    // The receipt is still readable, but the session no longer holds the key.
    expect(screen.queryByRole('heading', { name: '本地模拟回执' })).not.toBeNull()
    expect(replayButton()).toBeDisabled()
  })
})
