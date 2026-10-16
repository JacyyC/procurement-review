/**
 * FIX-04B / C-AUD-07 — render diagnostics must not print the payload they were handed.
 *
 * The audit reproduced this with the installed React 19 production build: with no root
 * error callback configured, React's `defaultOnCaughtError(error)` calls
 * `console.error(error)` directly, so a component that throws while rendering a payload
 * prints that payload (and its stack) into the console.  The workbench's own boundary log
 * was no better: it interpolated `error.name` and the first `componentStack` frame.
 *
 * `error.name` and `componentStack` are not trusted either — a thrown value controls both.
 * These cases drive a root the test configures itself, so the only thing under test is the
 * workbench's own logging.
 */

import { act, cleanup } from '@testing-library/react'
import { createRoot, type RootOptions } from 'react-dom/client'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ReviewErrorBoundary } from './ReviewErrorBoundary'
import { describeConsoleValue, scanConsole } from './testConsoleScan'

/** Nothing is logged by React itself: the assertion must be about our own outlet. */
const SILENT_ROOT: RootOptions = {
  onCaughtError: () => {},
  onUncaughtError: () => {},
  onRecoverableError: () => {},
}

function mount(element: React.ReactNode, options: RootOptions) {
  const container = document.createElement('div')
  document.body.append(container)
  const root = createRoot(container, options)
  act(() => {
    root.render(element)
  })
  return { container, root }
}

function canaryError(): Error {
  const error = new Error('CANARY-MESSAGE')
  error.name = 'CanaryNameError'
  error.stack = 'CanaryNameError: CANARY-MESSAGE\n    at CANARY-STACK-FRAME'
  ;(error as { cause?: unknown }).cause = { detail: 'CANARY-CAUSE' }
  return error
}

afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
  vi.restoreAllMocks()
})

describe('C-AUD-07 · the default root logs the whole error', () => {
  it('leaks the payload when no root callback is configured (the defect)', async () => {
    const scan = scanConsole()
    try {
      function Broken(): never {
        throw canaryError()
      }
      mount(
        <ReviewErrorBoundary scope="探针" onReset={() => {}}>
          <Broken />
        </ReviewErrorBoundary>,
        {},
      )
      await act(async () => {
        for (let tick = 0; tick < 20; tick += 1) await Promise.resolve()
      })
    } finally {
      scan.restore()
    }
    expect(scan.entries.join('\n')).toContain('CANARY-MESSAGE')
  })
})

describe('C-AUD-07 · the workbench boundary describes without copying', () => {
  it('never prints message, name, stack, cause or componentStack', async () => {
    const scan = scanConsole()
    try {
      function Broken(): never {
        throw canaryError()
      }
      const { container } = mount(
        <ReviewErrorBoundary scope="审查视图" onReset={() => {}}>
          <Broken />
        </ReviewErrorBoundary>,
        SILENT_ROOT,
      )
      await act(async () => {
        for (let tick = 0; tick < 20; tick += 1) await Promise.resolve()
      })
      expect(container.querySelector('[data-testid="review-error-boundary"]')).not.toBeNull()
    } finally {
      scan.restore()
    }
    const output = scan.entries.join('\n')
    expect(output).not.toContain('CANARY-MESSAGE')
    expect(output).not.toContain('CanaryNameError')
    expect(output).not.toContain('CANARY-STACK-FRAME')
    expect(output).not.toContain('CANARY-CAUSE')
    expect(output).not.toContain('at ')
    // It still says something, so a real failure is not silently swallowed.
    expect(output).toContain('[review-workbench]')
  })

  it.each([
    ['a thrown string', 'CANARY-MESSAGE'],
    ['a thrown object', { toString: () => 'CANARY-MESSAGE' }],
    ['a thrown number', 42],
    ['a thrown null', null],
  ])('handles %s without printing it', async (_label, thrown) => {
    const scan = scanConsole()
    try {
      function Broken(): never {
        throw thrown
      }
      mount(
        <ReviewErrorBoundary scope="审查视图" onReset={() => {}}>
          <Broken />
        </ReviewErrorBoundary>,
        SILENT_ROOT,
      )
      await act(async () => {
        for (let tick = 0; tick < 20; tick += 1) await Promise.resolve()
      })
    } finally {
      scan.restore()
    }
    expect(scan.entries.join('\n')).not.toContain('CANARY-MESSAGE')
  })
})

describe('C-AUD-07 · the scanner actually sees what it claims to', () => {
  it('reads non-enumerable Error fields and nested objects', () => {
    const nested = { outer: { inner: canaryError() } }
    const described = describeConsoleValue(nested)
    expect(described).toContain('CANARY-MESSAGE')
    expect(described).toContain('CANARY-STACK-FRAME')
    expect(described).toContain('CANARY-CAUSE')
    // The naive check this replaces would have missed all three: `message` and `stack`
    // are non-enumerable, and `String(value)` was never consulted.
    const naive = JSON.stringify(canaryError())
    expect(naive).not.toContain('CANARY-MESSAGE')
    expect(naive).not.toContain('CANARY-STACK-FRAME')
  })
})
