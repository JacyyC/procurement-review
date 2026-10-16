/**
 * FIX-04B / C-AUD-07 — the product entry must actually hand the safe options to React.
 *
 * `errorReporting.ts` is only a fix if the browser entry uses it.  This file mocks
 * `react-dom/client` so the assertion is about the real `main.tsx` module, not about a
 * test harness that happens to configure a root the product never configures.
 */

import { describe, expect, it, vi } from 'vitest'

const createRoot = vi.fn(() => ({ render: vi.fn(), unmount: vi.fn() }))

vi.mock('react-dom/client', () => ({ createRoot, hydrateRoot: vi.fn() }))
vi.mock('./App', () => ({ default: () => null }))

describe('C-AUD-07 · the entry wires the root options', () => {
  it('passes the shared safe options to createRoot', async () => {
    document.body.innerHTML = '<div id="root"></div>'
    const { ROOT_ERROR_OPTIONS } = await import('./errorReporting')
    await import('./main')
    expect(createRoot).toHaveBeenCalledTimes(1)
    const [element, options] = createRoot.mock.calls[0] as unknown as [unknown, unknown]
    expect(element).toBe(document.getElementById('root'))
    expect(options).toBe(ROOT_ERROR_OPTIONS)
  })
})
