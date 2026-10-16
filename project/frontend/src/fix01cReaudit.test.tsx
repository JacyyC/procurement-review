import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import {
  OPERATOR_SESSION,
  TOKEN,
  errorResponse,
  installHarness,
  makeAudit,
  makeRun,
  makeTrace,
} from './fix01cHarness'
import type { HarnessOptions } from './fix01cHarness'

const frameFailure = vi.hoisted(() => ({ armed: false }))

vi.mock('./WorkbenchFrame', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./WorkbenchFrame')>()
  return {
    ...actual,
    WorkbenchFrame: (props: Parameters<typeof actual.WorkbenchFrame>[0]) => {
      if (frameFailure.armed && props.session) throw new Error('synthetic frame failure')
      return actual.WorkbenchFrame(props)
    },
  }
})

async function settle() {
  await act(async () => {
    for (let tick = 0; tick < 100; tick += 1) await Promise.resolve()
  })
}

function install(options: Partial<HarnessOptions> = {}) {
  return installHarness({
    session: OPERATOR_SESSION,
    runs: [makeRun('run-a'), makeRun('run-b')],
    trace: (id) => makeTrace(id, ['run_completed']),
    audit: (id) => makeAudit(id, []),
    ...options,
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

beforeEach(() => vi.useFakeTimers())

afterEach(() => {
  frameFailure.armed = false
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('FIX-01C re-audit: ownership through asynchronous continuations', () => {
  it('keeps the selected Run and tab when a create finishes after navigation', async () => {
    const harness = install({
      cases: [{
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
      }],
      createResult: () => makeRun('run-created'),
    })
    await signIn()
    const held = harness.hold((call) => call.method === 'POST' && call.url.endsWith('/reviews'))
    fireEvent.click(screen.getByRole('button', { name: '运行审查' }))
    await settle()
    select('run-b')
    await settle()
    fireEvent.click(screen.getByRole('tab', { name: /运行 Trace/ }))
    held.resolve()
    await settle()
    expect(screen.getByRole('tab', { name: /运行 Trace/ })).toHaveAttribute('aria-selected', 'true')
    expect(document.querySelector('[data-run-id="run-b"]')).toHaveAttribute('aria-current', 'true')
    expect(harness.callsTo(/\/reviews$/, 'POST')).toHaveLength(1)
  })

  it('keeps the selected Run notice when another Run reports a late failure', async () => {
    const harness = install()
    await signIn()
    const base = globalThis.fetch
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith('/run-a/trace')) return heldResponse
      return base(input, init)
    }))
    let release!: (response: Response) => void
    const heldResponse = new Promise<Response>((resolve) => { release = resolve })
    select('run-a')
    await settle()
    select('run-b')
    await settle()
    refresh()
    await settle()
    expect(screen.getByText('已读取最新状态。')).toBeVisible()
    release(errorResponse(503, 'security_dependency_unavailable'))
    await settle()
    expect(screen.getByText('已读取最新状态。')).toBeVisible()
    expect(harness.callsTo(/\/run-b$/, 'GET')).toHaveLength(2)
  })

  it('does not publish an old refresh notice after a new sign-in during artifact reads', async () => {
    const harness = install()
    await signIn()
    const held = harness.hold((call) => call.url.endsWith('/run-a/trace'))
    refresh()
    await settle()
    fireEvent.click(screen.getByRole('button', { name: '身份与权限' })); fireEvent.click(screen.getByRole('button', { name: '退出并清除' }))
    fireEvent.change(screen.getByLabelText('Bearer 凭据'), { target: { value: TOKEN } })
    fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
    await settle()
    held.resolve()
    await settle()
    fireEvent.click(screen.getByRole('button', { name: '身份与权限' }))
    expect(screen.getByRole('region', { name: '已验证身份' })).toBeVisible()
    expect(screen.queryByText('已读取最新状态。')).toBeNull()
  })

  it('sends no reconciliation request with a credential rejected by a write', async () => {
    const harness = install({ failOnce: { match: /\/cancel$/, method: 'POST', status: 401, code: 'unauthenticated' } })
    await signIn()
    fireEvent.click(screen.getByRole('button', { name: '取消' }))
    await settle()
    expect(screen.getByLabelText('Bearer 凭据')).toBeVisible()
    expect(harness.callsTo(/\/run-a$/, 'GET')).toHaveLength(0)
    expect(harness.callsTo(/\/cancel$/, 'POST')).toHaveLength(1)
  })

  it('ends the session when read-only reconciliation returns 401', async () => {
    const harness = install({
      networkErrorOnce: { match: /\/cancel$/, method: 'POST' },
      failOnce: { match: /\/run-a$/, method: 'GET', status: 401, code: 'unauthenticated' },
    })
    await signIn()
    fireEvent.click(screen.getByRole('button', { name: '取消' }))
    await settle()
    expect(harness.callsTo(/\/cancel$/, 'POST')).toHaveLength(1)
    expect(screen.getByLabelText('Bearer 凭据')).toBeVisible()
  })

  it('returns to sign-in when the outer workbench boundary clears credentials', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const harness = install()
    await signIn()
    frameFailure.armed = true
    refresh()
    await settle()
    expect(screen.getByTestId('review-error-boundary')).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: '清除凭据并返回登录' }))
    await settle()
    expect(screen.getByLabelText('Bearer 凭据')).toBeVisible()
    expect(screen.queryByTestId('review-error-boundary')).toBeNull()
    expect(harness.calls.filter((call) => call.method === 'POST')).toHaveLength(0)
  })

  it('uses the fixed fallback for an unregistered code named constructor', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    install({ failOnce: { match: /\/cancel$/, method: 'POST', status: 409, code: 'constructor' } })
    await signIn()
    fireEvent.click(screen.getByRole('button', { name: '取消' }))
    await settle()
    expect(screen.getByText(/本地 Review API 返回了未登记的错误/)).toBeVisible()
  })
})
