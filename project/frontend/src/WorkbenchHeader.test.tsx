import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { WorkbenchHeader } from './WorkbenchHeader'
import { operatorSession } from './runFixtures'

const guide = { permissions: { caseRead: true, create: true }, connecting: false, loading: false, creating: false, loadFailed: false, hasCases: true }
beforeEach(() => Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() }))

describe('top identity disclosure', () => {
  it('opens a non-modal panel, closes by button/Escape and restores focus', () => {
    render(<WorkbenchHeader guide={guide} session={operatorSession} onSignOut={vi.fn()} executionMode="test_only" />)
    const trigger = screen.getByRole('button', { name: '身份与权限' })
    expect(trigger).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByRole('region', { name: '已验证身份' })).toBeNull()
    fireEvent.click(trigger)
    expect(trigger).toHaveAttribute('aria-expanded', 'true')
    expect(document.getElementById('identity-panel')).toHaveFocus()
    expect(screen.getByTestId('identity-scopes')).toBeVisible()
    expect(screen.queryByRole('dialog')).toBeNull()
    fireEvent.keyDown(document.activeElement!, { key: 'Escape' })
    expect(trigger).toHaveFocus()
    expect(trigger).toHaveAttribute('aria-expanded', 'false')
    fireEvent.click(trigger)
    fireEvent.click(screen.getByRole('button', { name: '关闭身份面板' }))
    expect(trigger).toHaveFocus()
  })
  it('uses the existing sign-out handler and removes the old identity on expiry', () => {
    const signOut = vi.fn()
    const { rerender } = render(<WorkbenchHeader guide={guide} session={operatorSession} onSignOut={signOut} executionMode="test_only" />)
    fireEvent.click(screen.getByRole('button', { name: '身份与权限' }))
    fireEvent.click(screen.getByRole('button', { name: '退出并清除' }))
    expect(signOut).toHaveBeenCalledOnce()
    rerender(<WorkbenchHeader guide={guide} session={null} onSignOut={signOut} executionMode={null} />)
    expect(screen.queryByTestId('identity-subject')).toBeNull()
    expect(screen.getByRole('button', { name: '连接工作台' })).toBeVisible()
  })

  it('states the profile the API reported, never a fixed test-only label', () => {
    const { rerender } = render(
      <WorkbenchHeader guide={guide} session={operatorSession} onSignOut={vi.fn()} executionMode="test_only" />,
    )
    expect(screen.getByTestId('environment-info')).toHaveAttribute('data-mode', 'test_only')
    expect(screen.getByTestId('environment-info').textContent).toContain('TEST_ONLY')
    rerender(
      <WorkbenchHeader guide={guide} session={operatorSession} onSignOut={vi.fn()} executionMode="real" />,
    )
    expect(screen.getByTestId('environment-info')).toHaveAttribute('data-mode', 'real')
    expect(screen.getByTestId('environment-info').textContent).toContain('真实模型环境')
    rerender(
      <WorkbenchHeader guide={guide} session={operatorSession} onSignOut={vi.fn()} executionMode={null} />,
    )
    expect(screen.getByTestId('environment-info').textContent).toContain('环境未知')
  })
})
