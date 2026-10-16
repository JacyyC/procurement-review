/**
 * FIX-04B / C-AUD-05 — a nested event must belong to the envelope that carries it.
 *
 * The audit found both validators checked the *envelope* identity but only required an
 * event's `run_id`/`trace_id` to be strings.  A payload whose envelope was correct but
 * whose events described another Run was accepted, and the component probe showed the
 * other Run's content rendered under the requested Run's heading.
 *
 * The endpoint facts these cases rely on (recorded in the FIX-04 evidence):
 * `GET /reviews/{id}/audit` loads `SELECT * FROM review_audit_event WHERE run_id = :run_id`
 * and every writer inserts the audited Run as `run_id`, so a Run-scoped audit event is
 * never `null` and never another Run.  `GET /reviews/{id}/trace` wraps the Run's own
 * `trace_events`, whose `trace_id` is the Run's `trace_id`.
 */

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import { validateAudit, validateTrace } from './apiContract'
import {
  OPERATOR_SESSION,
  TOKEN,
  installHarness,
  makeAudit,
  makeRun,
  makeTrace,
} from './fix01cHarness'

async function settle() {
  await act(async () => {
    for (let tick = 0; tick < 100; tick += 1) await Promise.resolve()
  })
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('C-AUD-05 · Trace envelope and events agree', () => {
  it('accepts a Trace whose envelope and events agree', () => {
    expect(() => validateTrace(makeTrace('run-a', ['run_completed']), 'trace', 'run-a')).not.toThrow()
  })

  it('accepts a legal empty event array', () => {
    expect(() => validateTrace(makeTrace('run-a', []), 'trace', 'run-a')).not.toThrow()
  })

  it('rejects an event from another Run inside a correct envelope', () => {
    const payload = makeTrace('run-a', ['run_completed'])
    payload.trace.events[0].run_id = 'run-b'
    expect(() => validateTrace(payload, 'trace', 'run-a')).toThrow()
  })

  it('rejects an event that names another Trace inside a correct envelope', () => {
    const payload = makeTrace('run-a', ['run_completed'])
    payload.trace.events[0].trace_id = 'trace-run-b'
    expect(() => validateTrace(payload, 'trace', 'run-a')).toThrow()
  })

  it('rejects a mismatched event even when no Run identity is passed in', () => {
    const payload = makeTrace('run-a', ['run_completed'])
    payload.trace.events[0].run_id = 'run-b'
    expect(() => validateTrace(payload, 'trace')).toThrow()
  })

  it('rejects an envelope whose trace_id is not the Run trace it claims', () => {
    const payload = makeTrace('run-a', ['run_completed'])
    payload.trace.trace_id = 'trace-other'
    expect(() => validateTrace(payload, 'trace', 'run-a', 'trace-run-a')).toThrow()
  })

  it('rejects a whole payload when only one event is wrong', () => {
    const payload = makeTrace('run-a', ['run_started', 'run_completed'])
    payload.trace.events[1].run_id = 'run-b'
    expect(() => validateTrace(payload, 'trace', 'run-a')).toThrow()
    // The good event is not salvaged and no identifier is rewritten.
    expect(payload.trace.events[0].run_id).toBe('run-a')
  })
})

describe('C-AUD-05 · Run-scoped Audit events belong to that Run', () => {
  it('accepts a Run’s own audit events and a legal empty ledger', () => {
    expect(() => validateAudit(makeAudit('run-a', ['subject-demo']), 'audit', 'run-a')).not.toThrow()
    expect(() => validateAudit(makeAudit('run-a', []), 'audit', 'run-a')).not.toThrow()
  })

  it('rejects an Audit event from another Run inside a correct envelope', () => {
    const payload = makeAudit('run-a', ['subject-demo'])
    payload.events[0].run_id = 'run-b'
    expect(() => validateAudit(payload, 'audit', 'run-a')).toThrow()
  })

  it('rejects a null run_id in a Run-scoped ledger', () => {
    // The real endpoint selects the ledger by `run_id`, so `null` is not a legal
    // member of it: accepting it would let an unanchored event render as this Run's.
    const payload = makeAudit('run-a', ['subject-demo'])
    ;(payload.events[0] as { run_id: string | null }).run_id = null
    expect(() => validateAudit(payload, 'audit', 'run-a')).toThrow()
  })

  it('rejects a mismatched event even when no Run identity is passed in', () => {
    const payload = makeAudit('run-a', ['subject-demo'])
    payload.events[0].run_id = 'run-b'
    expect(() => validateAudit(payload, 'audit')).toThrow()
  })
})

describe('C-AUD-05 · the view never shows another object’s content', () => {
  it('does not render another Run’s Trace event under the requested Run', async () => {
    vi.useFakeTimers()
    const foreign = makeTrace('run-b', ['foreign_run_event'])
    installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: (id) => ({
        ...makeTrace(id, []),
        trace: { ...makeTrace(id, []).trace, events: foreign.trace.events },
      }),
      audit: (id) => makeAudit(id, []),
    })
    render(<App />)
    fireEvent.change(screen.getByLabelText('Bearer 凭据'), { target: { value: TOKEN } })
    fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
    await settle()
    fireEvent.click(screen.getByRole('tab', { name: /运行 Trace/ }))
    await settle()
    expect(screen.queryByText('foreign_run_event')).toBeNull()
  })

  it('does not render another Run’s Audit event under the requested Run', async () => {
    vi.useFakeTimers()
    const foreign = makeAudit('run-b', ['subject-foreign'])
    installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: (id) => makeTrace(id, []),
      audit: (id) => ({
        ...makeAudit(id, []),
        events: foreign.events,
      }),
    })
    render(<App />)
    fireEvent.change(screen.getByLabelText('Bearer 凭据'), { target: { value: TOKEN } })
    fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
    await settle()
    fireEvent.click(screen.getByRole('tab', { name: /审批审计/ }))
    await settle()
    expect(screen.queryByText('subject-foreign')).toBeNull()
  })

  it('does not render a Trace envelope whose trace_id belongs to another Run', async () => {
    vi.useFakeTimers()
    installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: (id) => {
        const payload = makeTrace(id, ['run_completed'])
        // The envelope keeps the requested run_id but claims a different Trace.
        payload.trace.trace_id = 'trace-somewhere-else'
        payload.trace.events[0].trace_id = 'trace-somewhere-else'
        return payload
      },
      audit: (id) => makeAudit(id, []),
    })
    render(<App />)
    fireEvent.change(screen.getByLabelText('Bearer 凭据'), { target: { value: TOKEN } })
    fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
    await settle()
    fireEvent.click(screen.getByRole('tab', { name: /运行 Trace/ }))
    await settle()
    // The Run's own cache knows its trace id, so the envelope is rejected before React.
    expect(screen.queryByText('run_completed')).toBeNull()
  })
})
