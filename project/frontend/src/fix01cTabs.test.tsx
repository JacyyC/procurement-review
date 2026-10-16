/**
 * FIX-01C / V-C-04 — tabs: roving tabindex, ARIA relationships and what happens
 * when a credential change hides a view.
 *
 * These are the guarantees the keyboard user depends on: exactly one tab is in the
 * page's Tab order, the panel is labelled by the *selected* tab, and a narrower
 * credential neither leaves focus on a detached node nor reads the hidden views.
 */

import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App from './App'
import type { SessionInfo } from './types'
import { OPERATOR_SESSION, TOKEN, installHarness, makeRun } from './fix01cHarness'

const READER_SESSION: SessionInfo = {
  ...OPERATOR_SESSION,
  credential_id: 'cred-reader-fixture',
  roles: ['reader'],
  scopes: ['case:read', 'review:read'],
}

async function settle(rounds = 30): Promise<void> {
  await act(async () => {
    for (let index = 0; index < rounds; index += 1) {
      await Promise.resolve()
    }
  })
}

async function signIn(token = TOKEN): Promise<void> {
  fireEvent.change(screen.getByLabelText('Bearer 凭据'), { target: { value: token } })
  fireEvent.click(screen.getByRole('button', { name: '验证凭据' }))
  await settle()
}

function activeElementIsAttached(): boolean {
  const active = document.activeElement
  return Boolean(active) && active !== document.body ? document.contains(active) : true
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('FIX-01C tab semantics', () => {
  it('keeps exactly one tab in the Tab order and relates it to the panel', async () => {
    vi.useFakeTimers()
    installHarness({ session: OPERATOR_SESSION, runs: [makeRun('run-a')] })
    render(<App />)
    await signIn()
    expect(screen.getByTestId('identity-subject')).toBeInTheDocument()

    const tabs = screen.getAllByRole('tab')
    expect(tabs.map((tab) => tab.textContent)).toEqual([
      '审查结果',
      '运行 Trace3',
      '审批审计0',
    ])
    const panel = screen.getByRole('tabpanel')
    expect(panel).toHaveAttribute('id', 'review-tab-panel')

    for (const tab of tabs) {
      expect(tab).toHaveAttribute('aria-controls', 'review-tab-panel')
      const selected = tab.getAttribute('aria-selected') === 'true'
      expect(tab).toHaveAttribute('tabindex', selected ? '0' : '-1')
    }
    expect(tabs.filter((tab) => tab.getAttribute('tabindex') === '0')).toHaveLength(1)
    expect(panel).toHaveAttribute('aria-labelledby', 'review-tab-result')

    fireEvent.click(screen.getByRole('tab', { name: /审批审计/ }))
    await settle()
    expect(screen.getByRole('tab', { name: /审批审计/ })).toHaveAttribute('tabindex', '0')
    expect(screen.getByRole('tab', { name: '审查结果' })).toHaveAttribute('tabindex', '-1')
    expect(screen.getByRole('tabpanel')).toHaveAttribute('aria-labelledby', 'review-tab-audit')
  })

  it('wraps with the arrow keys and follows Home and End', async () => {
    vi.useFakeTimers()
    installHarness({ session: OPERATOR_SESSION, runs: [makeRun('run-a')] })
    render(<App />)
    await signIn()

    const result = screen.getByRole('tab', { name: '审查结果' })
    const trace = screen.getByRole('tab', { name: /Trace/ })
    const audit = screen.getByRole('tab', { name: /审批审计/ })

    result.focus()
    fireEvent.keyDown(result, { key: 'ArrowLeft' })
    await settle()
    expect(audit).toHaveFocus()
    expect(screen.getByRole('tabpanel')).toHaveAttribute('aria-labelledby', 'review-tab-audit')

    fireEvent.keyDown(audit, { key: 'ArrowRight' })
    await settle()
    expect(result).toHaveFocus()

    fireEvent.keyDown(result, { key: 'End' })
    await settle()
    expect(audit).toHaveFocus()

    fireEvent.keyDown(audit, { key: 'Home' })
    await settle()
    expect(result).toHaveFocus()

    fireEvent.keyDown(result, { key: 'ArrowRight' })
    await settle()
    expect(trace).toHaveFocus()
    expect(screen.getByRole('tabpanel')).toHaveAttribute('aria-labelledby', 'review-tab-trace')
  })

  it('navigates from the tab that has focus, not from the last rendered selection', async () => {
    vi.useFakeTimers()
    installHarness({ session: OPERATOR_SESSION, runs: [makeRun('run-a')] })
    render(<App />)
    await signIn()

    const result = screen.getByRole('tab', { name: '审查结果' })
    const trace = screen.getByRole('tab', { name: /Trace/ })
    const audit = screen.getByRole('tab', { name: /审批审计/ })
    expect(result).toHaveAttribute('aria-selected', 'true')

    // Focus lands on the last tab without selecting it — the state a burst of key
    // presses leaves behind.  ArrowRight from there must wrap to the first tab, which
    // only holds when the origin is the focused tab rather than the stale selection.
    audit.focus()
    fireEvent.keyDown(audit, { key: 'ArrowRight' })
    await settle()
    expect(result).toHaveFocus()
    expect(result).toHaveAttribute('aria-selected', 'true')

    trace.focus()
    fireEvent.keyDown(trace, { key: 'ArrowLeft' })
    await settle()
    expect(result).toHaveFocus()
  })

  it('renders only the tabs a narrower credential may use, without reading the hidden views', async () => {
    vi.useFakeTimers()
    const harness = installHarness({ session: OPERATOR_SESSION, runs: [makeRun('run-a')] })
    render(<App />)
    await signIn()
    screen.getByRole('tab', { name: /审批审计/ }).focus()
    expect(screen.getAllByRole('tab')).toHaveLength(3)

    fireEvent.click(screen.getByRole('button', { name: '身份与权限' })); fireEvent.click(screen.getByRole('button', { name: '退出并清除' }))
    await settle()
    const callsBefore = harness.calls.length
    harness.setSession(READER_SESSION)
    await signIn('reader-credential')

    expect(screen.getByTestId('identity-subject')).toHaveTextContent('subject-local-demo')
    const tabs = screen.getAllByRole('tab')
    expect(tabs).toHaveLength(1)
    expect(tabs[0]).toHaveTextContent('审查结果')
    expect(tabs[0]).toHaveAttribute('tabindex', '0')
    expect(tabs[0]).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tabpanel')).toHaveAttribute('aria-labelledby', 'review-tab-result')
    expect(screen.getByText(/缺少 trace:read 授权，Trace 已隐藏。/)).toBeInTheDocument()

    // The narrower identity never asks for the views it cannot see, and focus was
    // not left on the tab that disappeared with the wider credential.
    const after = harness.calls.slice(callsBefore)
    expect(after.filter((call) => /\/(trace|audit)$/.test(call.url))).toHaveLength(0)
    expect(activeElementIsAttached()).toBe(true)
  })

  it('clears the tablist on revocation and leaves focus on a live element', async () => {
    vi.useFakeTimers()
    const harness = installHarness({ session: OPERATOR_SESSION, runs: [makeRun('run-a')] })
    render(<App />)
    await signIn()
    screen.getByRole('tab', { name: /Trace/ }).focus()

    harness.expireCredential()
    fireEvent.click(screen.getByRole('button', { name: '刷新当前 Review' }))
    await settle()

    expect(screen.queryAllByRole('tab')).toHaveLength(0)
    expect(screen.getByText('需要本地凭据')).toBeInTheDocument()
    expect(activeElementIsAttached()).toBe(true)
    // The credential field is reachable again, so the keyboard path is not trapped.
    const form = screen.getByLabelText('Bearer 凭据')
    form.focus()
    expect(form).toHaveFocus()
    // Nothing was read for the hidden views after the credential stopped working.
    expect(screen.queryByRole('tabpanel')).not.toBeInTheDocument()
    const list = screen.getByRole('navigation', { name: 'Review 运行' })
    expect(within(list).queryAllByRole('button')).toHaveLength(0)
  })
})
