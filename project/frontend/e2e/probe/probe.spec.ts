import { readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, test } from '@playwright/test'

/**
 * FIX-04B / C-AUD-07 — what the *shipped* React build writes to the console.
 *
 * The unit tests prove the workbench's own outlet is safe; they cannot prove the production
 * React implementation reports through it, because React ships a different build for
 * production and the leak the audit reproduced came from that build.  This probe serves a
 * real production build, triggers caught / uncaught / recoverable render errors, and records
 * every console argument (deeply, including non-enumerable Error fields) plus every
 * `error`/`unhandledrejection` event.
 */

const HERE = path.dirname(fileURLToPath(import.meta.url))
const PRODUCT_DIST = path.resolve(HERE, '..', '..', 'dist')
const PROBE_DIST = path.resolve(HERE, 'dist')
const SCAN_SCRIPT = path.join(HERE, 'console-scan.js')

interface ProbeWindow {
  __consoleScan: string[]
  __pageErrors: string[]
  __canary: string
}

async function readProbeWindow(page: import('@playwright/test').Page): Promise<ProbeWindow> {
  return page.evaluate(() => ({
    __consoleScan: (window as unknown as ProbeWindow).__consoleScan ?? [],
    __pageErrors: (window as unknown as ProbeWindow).__pageErrors ?? [],
    __canary: (window as unknown as ProbeWindow).__canary ?? '',
  }))
}

function bundleText(directory: string): string {
  return readdirSync(path.join(directory, 'assets'))
    .filter((name) => name.endsWith('.js'))
    .map((name) => readFileSync(path.join(directory, 'assets', name), 'utf8'))
    .join('\n')
}

test('a production build reports render errors without printing the thrown value', async ({
  page,
}) => {
  await page.addInitScript({ path: SCAN_SCRIPT })
  await page.goto('/')
  await page.waitForFunction(() => document.body.dataset.probeReady === 'true')

  // All three paths really ran: the boundary fallback is on screen and React recovered the
  // hydration mismatch on the client.
  await expect(page.locator('#caught [data-testid="review-error-boundary"]')).toBeVisible()
  await expect(page.locator('#recoverable p')).toHaveText('client-rendered text')

  const { __consoleScan: scan, __pageErrors: pageErrors, __canary: canary } = await readProbeWindow(page)
  expect(canary).toBe('FIX04-PROBE-CANARY')
  // The recorder is not vacuous: something was logged, and the fixed categories are visible.
  expect(scan.length).toBeGreaterThan(0)
  const output = scan.join('\n')
  expect(output).toContain('[review-workbench]')
  expect(output).toContain('render_failed')
  expect(output).toContain('root_caught')
  expect(output).toContain('root_uncaught')
  expect(output).toContain('root_recoverable')

  // …and none of the recorded arguments mentions any part of the thrown value.
  for (const part of ['MESSAGE', 'NAME', 'FRAME', 'CAUSE']) {
    expect(output, `console leaked ${canary}-${part}`).not.toContain(`${canary}-${part}`)
  }
  expect(output).not.toContain(canary)
  expect(pageErrors.join('\n')).not.toContain(canary)
})

test('the product bundle carries no probe canary or fault switch', () => {
  const product = bundleText(PRODUCT_DIST)
  expect(product).not.toContain('FIX04-PROBE-CANARY')
  expect(product).not.toContain('probeReady')
  // The probe build is a separate artifact and is the only one that may contain it.
  expect(bundleText(PROBE_DIST)).toContain('FIX04-PROBE-CANARY')
})
