/**
 * FIX-01C / V-C-02 — recoverable error states.
 *
 * Two failure modes are covered here:
 *
 *  1. a **render** exception inside a view, caught by `ReviewErrorBoundary`, which
 *     must leave an escape hatch instead of a blank page and must never replay a
 *     write request when it is reset;
 *  2. a **malformed 2xx body**, which is rejected by the response contract and
 *     surfaces as an explicit, non-destructive error while the rest of the
 *     workbench (other Runs, reload, sign-out) stays usable.
 */

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const control = vi.hoisted(() => ({ armed: false }))

vi.mock('./ReviewViews', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./ReviewViews')>()
  return {
    ...actual,
    ResultView: (props: Parameters<typeof actual.ResultView>[0]) => {
      if (control.armed) throw new Error('SECRET-PAYLOAD-MARKER')
      return actual.ResultView(props)
    },
  }
})

import App from './App'
import type { ReviewRun } from './types'
import {
  OPERATOR_SESSION,
  TOKEN,
  installHarness,
  jsonResponse,
  makeRun,
} from './fix01cHarness'

const CHECKSUM = `sha256:${'e'.repeat(64)}`

function decidable(runId: string, overrides: Partial<ReviewRun> = {}): ReviewRun {
  return makeRun(runId, {
    version: 4,
    revision: 4,
    summary: {
      schema_version: '1.0',
      run_id: runId,
      case_id: 'CASE-LOW-001',
      rules_version: '2.0.0',
      findings: [],
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

afterEach(() => {
  control.armed = false
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('FIX-01C recoverable render boundary', () => {
  it('shows a recoverable boundary instead of a blank page, and logs no payload', async () => {
    vi.useFakeTimers()
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [decidable('run-a')],
    })
    await signIn()
    expect(screen.getByTestId('identity-subject')).toBeInTheDocument()

    control.armed = true
    fireEvent.click(screen.getByRole('button', { name: '刷新当前 Review' }))
    await settle()

    expect(screen.getByTestId('review-error-boundary')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '重新读取视图' })).toBeInTheDocument()
    // The rest of the workbench is still there: this is a view failure, not a crash.
    expect(screen.getByTestId('identity-subject')).toBeInTheDocument()

    // The boundary's own diagnostics say where it failed and nothing about the value it
    // was handed: no name, message, stack or component stack (FIX-04B / C-AUD-07).
    const own = consoleError.mock.calls
      .map((call) => String(call[0] ?? ''))
      .filter((line) => line.includes('[review-workbench]'))
    expect(own).toHaveLength(1)
    expect(own[0]).not.toContain('SECRET-PAYLOAD-MARKER')
    expect(own[0]).toContain('render_failed')
    expect(own[0]).toContain('审查视图')
    // The value is described only by its JavaScript type.
    expect(own[0]).toContain('thrown: error')
    expect(harness.callsTo(/\/confirm$/, 'POST')).toHaveLength(0)
  })

  it('re-reads the view on reset and never replays the human action', async () => {
    vi.useFakeTimers()
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [decidable('run-a')],
      actionResult: () =>
        decidable('run-a', {
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
            idempotency_key_digest: `sha256:${'f'.repeat(64)}`,
            simulation_status: 'local_simulation_recorded',
            simulation_mode: 'local_test_only',
            external_write_performed: false,
          },
        }),
    })
    await signIn()

    fireEvent.click(screen.getByRole('button', { name: '确认建议（本地）' }))
    await settle()
    const decisions = harness.callsTo(/\/confirm$/, 'POST')
    expect(decisions).toHaveLength(1)
    const readsBefore = harness.callsTo(/^\/api\/v2\/reviews\/run-a$/, 'GET').length

    control.armed = true
    fireEvent.click(screen.getByRole('button', { name: '刷新当前 Review' }))
    await settle()
    expect(screen.getByTestId('review-error-boundary')).toBeInTheDocument()

    control.armed = false
    fireEvent.click(screen.getByRole('button', { name: '重新读取视图' }))
    await settle()

    expect(screen.queryByTestId('review-error-boundary')).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '本地模拟回执' })).toBeInTheDocument()
    // A reset is a re-read: the decision that already left the browser is not re-sent.
    expect(harness.callsTo(/\/confirm$/, 'POST')).toHaveLength(1)
    expect(harness.callsTo(/^\/api\/v2\/reviews\/run-a$/, 'GET').length).toBeGreaterThan(
      readsBefore,
    )
  })

  it('keeps a sign-out path open from the boundary', async () => {
    vi.useFakeTimers()
    vi.spyOn(console, 'error').mockImplementation(() => {})
    installHarness({ session: OPERATOR_SESSION, runs: [decidable('run-a')] })
    await signIn()

    control.armed = true
    fireEvent.click(screen.getByRole('button', { name: '刷新当前 Review' }))
    await settle()
    expect(screen.getByTestId('review-error-boundary')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: '清除凭据并返回登录' }))
    await settle()
    expect(screen.getByText('需要本地凭据')).toBeInTheDocument()
    expect(document.querySelectorAll('[data-run-id]')).toHaveLength(0)
  })
})

describe('FIX-01C recoverable response boundary', () => {
  it('reports a malformed success body as a safe error and keeps the workbench usable', async () => {
    vi.useFakeTimers()
    const base = decidable('run-a')
    const baseSummary = base.summary
    if (!baseSummary) throw new Error('fixture must carry a summary')
    const malformed = {
      // A structurally plausible payload that also claims a *different* state, so a
      // boundary that leaked it would visibly change the Run.
      ...decidable('run-a', {
        status: 'failed',
        error_code: 'review_execution_failed',
        error_message: 'should never be rendered',
      }),
      // ...but the summary lost a field the workbench consumes.
      summary: { ...baseSummary, missing_information: undefined },
    }
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [decidable('run-a'), makeRun('run-b')],
      overrideOnce: {
        match: /^\/api\/v2\/reviews\/run-a$/,
        response: () => jsonResponse(malformed),
      },
    })
    await signIn()

    fireEvent.click(document.querySelector('[data-run-id="run-a"]') as Element)
    await settle()

    expect(
      screen.getByText('接口返回了无法校验的响应，已停止展示该对象；请重试或检查服务端版本。'),
    ).toBeInTheDocument()
    // The rejected payload never became state: the Run is not shown as failed, and
    // the view did not fall over either.
    expect(
      screen.queryByRole('heading', { name: 'Review 运行失败并已关闭' }),
    ).not.toBeInTheDocument()
    expect(screen.queryByTestId('review-error-boundary')).not.toBeInTheDocument()

    // The escape hatches AC-05.4 requires: pick another Run and keep working.
    fireEvent.click(document.querySelector('[data-run-id="run-b"]') as Element)
    await settle()
    expect(document.querySelector('.run-title code')?.getAttribute('title')).toBe('run-b')
    expect(harness.callsTo(/^\/api\/v2\/reviews\/run-b$/, 'GET')).toHaveLength(1)
  })
})
