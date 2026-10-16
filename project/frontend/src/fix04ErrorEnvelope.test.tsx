/**
 * FIX-04B / C-AUD-04 — the error envelope is a closed contract, and its text is not UI copy.
 *
 * The audit found `parseErrorEnvelope` only checked that `error` was an object: a missing
 * field, a numeric code or an array `details` were defaulted into a plausible business
 * error, and the server's own `message` was rendered into the page.  A gateway that
 * answers `503` with `{"error":{"message":"..."}}` therefore put arbitrary text in front
 * of the operator and hid the fact that nothing had been validated.
 *
 * These cases pin three things: every required field is validated for presence and type,
 * a malformed envelope becomes `invalid_response` **with the real HTTP status**, and no
 * server-supplied text reaches the page.
 */

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import { createReview, parseErrorEnvelope, ReviewAPIError } from './api'
import { OPERATOR_SESSION, TOKEN, installHarness, jsonResponse, makeAudit, makeRun, makeTrace } from './fix01cHarness'
import { reviewRun } from './runFixtures'

const VALID = {
  code: 'version_conflict',
  message: 'the fixture message must never be rendered',
  request_id: 'request-fixture',
  details: { fixture: true },
}

function envelope(overrides: Record<string, unknown>): Record<string, unknown> {
  return { error: { ...VALID, ...overrides } }
}

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

describe('C-AUD-04 · the envelope is closed', () => {
  it.each([
    ['empty error object', { error: {} }],
    ['error is null', { error: null }],
    ['error is an array', { error: [] }],
    ['error is a string', { error: 'version_conflict' }],
    ['missing code', envelope({ code: undefined })],
    ['null code', envelope({ code: null })],
    ['numeric code', envelope({ code: 42 })],
    ['empty code', envelope({ code: '' })],
    ['missing message', envelope({ message: undefined })],
    ['null message', envelope({ message: null })],
    ['numeric message', envelope({ message: 7 })],
    ['empty message', envelope({ message: '' })],
    ['missing request_id', envelope({ request_id: undefined })],
    ['null request_id', envelope({ request_id: null })],
    ['empty request_id', envelope({ request_id: '' })],
    ['missing details', envelope({ details: undefined })],
    ['null details', envelope({ details: null })],
    ['array details', envelope({ details: [] })],
    ['numeric details', envelope({ details: 3 })],
    ['extra key inside error', envelope({ extra: 'unexplained' })],
    ['extra key beside error', { error: { ...VALID }, detail: 'unexplained' }],
    ['body is an array', [{ error: VALID }]],
    ['body is null', null],
    ['body is a string', 'not json'],
  ])('rejects %s as invalid_response', (_label, body) => {
    const error = parseErrorEnvelope(409, body)
    expect(error).toBeInstanceOf(ReviewAPIError)
    expect(error.code).toBe('invalid_response')
    // The real status is kept: a broken gateway may not masquerade as a business answer.
    expect(error.status).toBe(409)
  })

  it('keeps a valid envelope and its real status', () => {
    const error = parseErrorEnvelope(503, envelope({ code: 'security_dependency_unavailable' }))
    expect(error.code).toBe('security_dependency_unavailable')
    expect(error.status).toBe(503)
  })

  it('does not retain the server message or details in the error object', () => {
    const error = parseErrorEnvelope(
      409,
      envelope({
        code: 'version_conflict',
        message: 'CANARY-SERVER-MESSAGE',
        details: { leaked: 'CANARY-SERVER-DETAIL' },
      }),
    )
    // Nothing that renders or logs an error may carry unverified server text.
    expect(JSON.stringify(error)).not.toContain('CANARY-SERVER')
    expect(error.message).not.toContain('CANARY-SERVER')
    expect(String(error.stack ?? '')).not.toContain('CANARY-SERVER')
    expect(Object.values(error.details).join('|')).not.toContain('CANARY-SERVER')
  })

  it.each([
    ['a proxy HTML body', new Response('<html>CANARY-PROXY-BODY</html>', { status: 503 })],
    ['an empty body', new Response('', { status: 503 })],
    ['a non-JSON body', new Response('gateway said no', { status: 502 })],
    ['a JSON body that is not an envelope', jsonResponse({ detail: 'CANARY-FASTAPI-DEFAULT' }, 404)],
  ])('classifies %s as invalid_response while keeping the status', async (_label, response) => {
    const harness = installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: (id) => makeTrace(id, []),
      audit: (id) => makeAudit(id, []),
      overrideOnce: { match: /\/reviews$/, method: 'POST', response: () => response },
    })
    void harness
    await expect(createReview('CASE-LOW-001', 'v1', 'key', { token: TOKEN })).rejects.toMatchObject({
      code: 'invalid_response',
      status: response.status,
    })
  })

  it('keeps the real 202 status when a success body fails validation', async () => {
    installHarness({
      session: OPERATOR_SESSION,
      runs: [makeRun('run-a')],
      trace: (id) => makeTrace(id, []),
      audit: (id) => makeAudit(id, []),
      overrideOnce: {
        match: /\/reviews$/,
        method: 'POST',
        response: () => jsonResponse({ run_id: 'run-a' }, 202),
      },
    })
    // A malformed 2xx is not a 200: the default must not mask the real classification.
    await expect(createReview('CASE-LOW-001', 'v1', 'key', { token: TOKEN })).rejects.toMatchObject({
      code: 'invalid_response',
      status: 202,
    })
  })
})

describe('C-AUD-04 · the page shows local words only', () => {
  it.each([
    [
      'a complete envelope with hostile message and details',
      { error: { code: 'security_dependency_unavailable', message: 'AUDIT-PRIVATE-CANARY-MESSAGE', request_id: 'req-canary', details: { note: 'AUDIT-PRIVATE-CANARY-DETAIL' } } },
      503,
      '本地 Review 存储暂时不可用，操作未确认，请稍后重试。',
    ],
    [
      'an incomplete envelope',
      { error: { message: 'AUDIT-PRIVATE-CANARY-MESSAGE' } },
      503,
      '接口返回了无法校验的响应，已停止展示该对象；请重试或检查服务端版本。',
    ],
    [
      'an unregistered code',
      { error: { code: 'totally_new_code', message: 'AUDIT-PRIVATE-CANARY-MESSAGE', request_id: 'req-canary', details: {} } },
      503,
      '本地 Review API 返回了未登记的错误；操作未确认，请重试或联系维护者。',
    ],
    [
      'an unregistered code with a request id that is not a local identifier',
      { error: { code: 'totally_new_code', message: 'AUDIT-PRIVATE-CANARY-MESSAGE', request_id: 'not a safe id <script>', details: {} } },
      503,
      '本地 Review API 返回了未登记的错误；操作未确认，请重试或联系维护者。',
    ],
  ])('renders no server text for %s', async (_label, body, status, expected) => {
    installHarness({
      session: OPERATOR_SESSION,
      runs: [],
      trace: (id) => makeTrace(id, []),
      audit: (id) => makeAudit(id, []),
      overrideOnce: {
        match: /\/session$/,
        response: () => jsonResponse(body, status),
      },
    })
    render(<App />)
    fireEvent.change(screen.getByLabelText('Bearer 凭据'), { target: { value: TOKEN } })
    fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
    await settle()
    const text = document.body.textContent ?? ''
    expect(text).not.toContain('AUDIT-PRIVATE-CANARY')
    // It still says something fixed and actionable in local words.
    expect(screen.getByText((content) => content.startsWith(expected))).toBeInTheDocument()
  })

  it('shows a request id only when it satisfies the local identifier contract', async () => {
    installHarness({
      session: OPERATOR_SESSION,
      runs: [],
      trace: (id) => makeTrace(id, []),
      audit: (id) => makeAudit(id, []),
      overrideOnce: {
        match: /\/session$/,
        response: () =>
          jsonResponse(
            {
              error: {
                code: 'totally_new_code',
                message: 'fixture',
                request_id: 'not a safe id <script>',
                details: {},
              },
            },
            503,
          ),
      },
    })
    render(<App />)
    fireEvent.change(screen.getByLabelText('Bearer 凭据'), { target: { value: TOKEN } })
    fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
    await settle()
    expect(document.body.textContent).not.toContain('not a safe id')
    expect(document.body.textContent).not.toContain('<script>')
  })

  it('keeps the registered behaviour for a version conflict', async () => {
    const run = structuredClone(reviewRun('pending_confirmation'))
    run.run_id = 'run-a'
    run.trace_id = 'trace-run-a'
    run.summary!.run_id = 'run-a'
    installHarness({
      session: OPERATOR_SESSION,
      runs: [run],
      trace: (id) => makeTrace(id, []),
      audit: (id) => makeAudit(id, []),
      failOnce: { match: /run-a\/confirm/, method: 'POST', status: 409, code: 'version_conflict' },
    })
    render(<App />)
    fireEvent.change(screen.getByLabelText('Bearer 凭据'), { target: { value: TOKEN } })
    fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
    await settle()
    fireEvent.click(screen.getByRole('button', { name: '确认建议（本地）' }))
    await settle()
    expect(screen.getByText('版本已过期，已重新读取最新审查状态。')).toBeInTheDocument()
  })
})
