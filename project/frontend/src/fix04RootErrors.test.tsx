/**
 * FIX-04B / C-AUD-07 — one safe diagnostic outlet, wired into the React root.
 *
 * The boundary alone cannot cover this: React 19 reports a render error to the root's
 * `onCaughtError`/`onUncaughtError`/`onRecoverableError` as well, and with none configured
 * the production default prints the raw `Error`.  The workbench therefore owns one
 * reporting function and hands the same object to `createRoot`.
 *
 * Everything here is driven through the module the product entry uses, so a change that
 * only fixes the boundary (or only the root) fails.
 */

import { act, cleanup } from '@testing-library/react'
import { createRoot } from 'react-dom/client'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ROOT_ERROR_OPTIONS, reportRenderFailure, scopeOf } from './errorReporting'
import { ReviewErrorBoundary } from './ReviewErrorBoundary'
import { scanConsole } from './testConsoleScan'

function canaryError(): Error {
  const error = new Error('ROOT-CANARY-MESSAGE')
  error.name = 'RootCanaryError'
  error.stack = 'RootCanaryError: ROOT-CANARY-MESSAGE\n    at ROOT-CANARY-FRAME'
  ;(error as { cause?: unknown }).cause = { detail: 'ROOT-CANARY-CAUSE' }
  return error
}

const CANARIES = [
  'ROOT-CANARY-MESSAGE',
  'RootCanaryError',
  'ROOT-CANARY-FRAME',
  'ROOT-CANARY-CAUSE',
  'at renderRoot',
]

function withScan(run: () => void): string {
  const scan = scanConsole()
  try {
    run()
  } finally {
    scan.restore()
  }
  return scan.entries.join('\n')
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('C-AUD-07 · the root callbacks are safe', () => {
  it('exposes exactly the three React 19 root callbacks', () => {
    expect(typeof ROOT_ERROR_OPTIONS.onCaughtError).toBe('function')
    expect(typeof ROOT_ERROR_OPTIONS.onUncaughtError).toBe('function')
    expect(typeof ROOT_ERROR_OPTIONS.onRecoverableError).toBe('function')
  })

  it.each([
    ['onCaughtError', 'onCaughtError'],
    ['onUncaughtError', 'onUncaughtError'],
    ['onRecoverableError', 'onRecoverableError'],
  ] as const)('%s never prints the error it is given', (_label, key) => {
    const handler = ROOT_ERROR_OPTIONS[key] as (error: unknown, info: unknown) => void
    const output = withScan(() => {
      handler(canaryError(), {
        componentStack: 'ROOT-CANARY-COMPONENT-STACK',
        errorBoundary: { componentStack: 'ROOT-CANARY-BOUNDARY-STACK' },
      })
    })
    for (const canary of CANARIES) expect(output).not.toContain(canary)
    expect(output).toContain('[review-workbench]')
  })

  it.each([
    ['a thrown string', 'ROOT-CANARY-MESSAGE'],
    ['a thrown object', { toString: () => 'ROOT-CANARY-MESSAGE' }],
    ['a thrown number', 7],
    ['a thrown undefined', undefined],
    ['a thrown symbol', Symbol('ROOT-CANARY-MESSAGE')],
  ])('survives %s without printing it', (_label, thrown) => {
    const output = withScan(() => {
      ROOT_ERROR_OPTIONS.onCaughtError?.(thrown, { componentStack: 'ROOT-CANARY-COMPONENT-STACK' })
      ROOT_ERROR_OPTIONS.onUncaughtError?.(thrown, { componentStack: 'ROOT-CANARY-COMPONENT-STACK' })
      ROOT_ERROR_OPTIONS.onRecoverableError?.(thrown, { componentStack: 'ROOT-CANARY-COMPONENT-STACK' })
    })
    for (const canary of CANARIES) expect(output).not.toContain(canary)
  })

  it('records a fixed category and a local scope, never a caller-supplied name', () => {
    const output = withScan(() => {
      reportRenderFailure('审查视图', canaryError())
    })
    expect(output).toContain('[review-workbench]')
    expect(output).toContain('审查视图')
    expect(output).toContain('render_failed')
  })

  it('accepts only the scopes the workbench declares', () => {
    expect(scopeOf('工作台')).toBe('工作台')
    expect(scopeOf('审查视图')).toBe('审查视图')
    // Anything else is reported under the fixed fallback, so a future caller cannot
    // smuggle a value into the one field the outlet prints.
    expect(scopeOf('CANARY-SCOPE' as never)).toBe('工作台')
  })
})

describe('C-AUD-07 · the boundary and the root share one outlet', () => {
  it('reports a render failure without printing the thrown value', async () => {
    function Broken(): never {
      throw canaryError()
    }
    const container = document.createElement('div')
    document.body.append(container)
    const scan = scanConsole()
    try {
      // The root is configured exactly as the product entry configures it, so both
      // reporting paths (React's root callback and the boundary) are exercised at once.
      const root = createRoot(container, ROOT_ERROR_OPTIONS)
      await act(async () => {
        root.render(
          <ReviewErrorBoundary scope="审查视图" onReset={() => {}}>
            <Broken />
          </ReviewErrorBoundary>,
        )
        for (let tick = 0; tick < 20; tick += 1) await Promise.resolve()
      })
      expect(container.querySelector('[data-testid="review-error-boundary"]')).not.toBeNull()
    } finally {
      scan.restore()
    }
    const output = scan.entries.join('\n')
    for (const canary of CANARIES) expect(output).not.toContain(canary)
    // Both outlets spoke, and both said only a fixed local category.
    expect(output).toContain('render_failed')
    expect(output).toContain('root_caught')
  })

  it('keeps the credential escape hatch available and never re-sends a write', async () => {
    const onReset = vi.fn()
    const onSignOut = vi.fn()
    function Broken(): never {
      throw canaryError()
    }
    const scan = scanConsole()
    let container: HTMLDivElement
    try {
      container = document.createElement('div')
      document.body.append(container)
      const root = createRoot(container, ROOT_ERROR_OPTIONS)
      await act(async () => {
        root.render(
          <ReviewErrorBoundary scope="工作台" onReset={onReset} onSignOut={onSignOut}>
            <Broken />
          </ReviewErrorBoundary>,
        )
        for (let tick = 0; tick < 20; tick += 1) await Promise.resolve()
      })
    } finally {
      scan.restore()
    }
    const fallback = container!.querySelector('[data-testid="review-error-boundary"]')!
    expect(fallback).not.toBeNull()
    const buttons = Array.from(fallback.querySelectorAll('button')).map((b) => b.textContent)
    expect(buttons).toContain('重新读取视图')
    expect(buttons).toContain('清除凭据并返回登录')
    // Rendering the fallback is not a retry: nothing is called until the human clicks.
    expect(onReset).not.toHaveBeenCalled()
    expect(onSignOut).not.toHaveBeenCalled()
  })
})
