import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { installHarness, OPERATOR_SESSION, TOKEN, makeRun } from './fix01cHarness'
import type { HarnessOptions } from './fix01cHarness'
import type { CaseOption } from './types'

const aCase: CaseOption = { ...makeRun('run-a').request, api_version: 'review-api.v2', case_version: 'case-low-001.v1', index_version: 'retrieval-index-v1', rules_version: '2.0.0', label: '测试案例', case_type: 'purchase_order', scenario: 'low' }
async function settle() { await act(async () => { for (let n = 0; n < 100; n++) await Promise.resolve() }) }
function setup(options: Partial<HarnessOptions> = {}) {
  const harness = installHarness({ session: OPERATOR_SESSION, runs: [], cases: [aCase], ...options })
  render(<App />)
  return harness
}
async function connect() {
  fireEvent.change(screen.getByLabelText('Bearer 凭据'), { target: { value: TOKEN } })
  fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
  await settle()
}
function guide() {
  const element = document.querySelector('details.quick-guide') as HTMLDetailsElement
  element.open = true
  return within(element)
}
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks() })

describe('guide consumes the current App request state', () => {
  it('focuses the credential on repeated clicks, with no request', () => {
    const scroll = vi.fn()
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: scroll })
    const harness = setup()
    for (let i = 0; i < 2; i++) {
      fireEvent.click(guide().getByRole('button', { name: '验证凭据，开始审查' }))
      expect(screen.getByLabelText('Bearer 凭据')).toHaveFocus()
      expect(document.querySelector('details.quick-guide')).not.toHaveAttribute('open')
    }
    expect(scroll).toHaveBeenCalledTimes(2)
    expect(harness.calls).toHaveLength(0)
  })
  it('focuses the available case without changing selection or writing', async () => {
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
    const harness = setup()
    await connect()
    const select = screen.getByLabelText('已授权案例')
    const before = (select as HTMLSelectElement).value
    fireEvent.click(guide().getByRole('button', { name: '选择案例，发起审查' }))
    expect(select).toHaveFocus()
    expect(select).toHaveValue(before)
    expect(harness.calls.filter((call) => call.method !== 'GET')).toHaveLength(0)
  })
  for (const removed of ['case:read', 'review:create']) {
    it(`opens permissions when ${removed} is absent`, async () => {
      const harness = setup({ session: { ...OPERATOR_SESSION, scopes: OPERATOR_SESSION.scopes.filter((scope) => scope !== removed) } })
      await connect()
      expect(guide().getByRole('status')).toHaveTextContent('当前身份无法发起审查')
      fireEvent.click(guide().getByRole('button', { name: '查看权限' }))
      expect(document.getElementById('identity-panel')).toHaveFocus()
      expect(screen.getByRole('button', { name: '退出并清除' })).toBeVisible()
      expect(harness.calls.filter((call) => call.method !== 'GET')).toHaveLength(0)
    })
  }
  for (const state of ['empty', 'failed'] as const) {
    it(`distinguishes ${state} from other states and reuses reconnection`, async () => {
      setup(state === 'empty' ? { cases: [] } : { failOnce: { match: /\/cases$/, status: 503, code: 'review_store_unavailable' } })
      await connect()
      expect(guide().getByRole('status')).toHaveTextContent(state === 'empty' ? '当前没有已授权案例' : '工作台读取失败')
      fireEvent.click(guide().getByRole('button', { name: /查看身份/ }))
      expect(screen.getByRole('button', { name: '退出并清除' })).toBeVisible()
    })
  }
  for (const state of ['connecting', 'loading', 'creating'] as const) {
    it(`disables the guide action during ${state} and updates it when settled`, async () => {
      const harness = setup({ createResult: () => makeRun('run-a') })
      if (state === 'creating') await connect()
      const held = harness.hold((call) => state === 'connecting' ? call.url.endsWith('/session') : state === 'loading' ? call.url.endsWith('/cases') : call.method === 'POST')
      if (state === 'creating') fireEvent.click(screen.getByRole('button', { name: '运行审查' }))
      else await connect()
      await settle()
      expect(guide().getByRole('button', { name: /正在/ })).toBeDisabled()
      held.resolve()
      await settle()
      expect(guide().getByRole('button', { name: '选择案例，发起审查' })).toBeEnabled()
    })
  }
  it('re-evaluates an open guide after the active identity expires', async () => {
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
    const harness = setup({ runs: [makeRun('run-a')] })
    await connect()
    guide()
    harness.expireCredential()
    fireEvent.click(screen.getByRole('button', { name: '刷新当前 Review' }))
    await settle()
    expect(document.querySelector('details.quick-guide')).toHaveAttribute('open')
    fireEvent.click(guide().getByRole('button', { name: '验证凭据，开始审查' }))
    expect(screen.getByLabelText('Bearer 凭据')).toHaveFocus()
    expect(screen.queryByTestId('identity-subject')).toBeNull()
  })
})
