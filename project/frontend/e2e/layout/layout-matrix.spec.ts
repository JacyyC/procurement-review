import { fullPage } from '../full-page'
import { execFileSync } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { expect, test, type Page } from '@playwright/test'

import { faultKindsByProject, faultOriginsByProject, resolveE2EOrigins } from '../origins.mjs'

/**
 * FIX-04B / V-04B-03 — the layout matrix, measured on the real page.
 *
 * Eight product states are each measured at the five required widths (8 × 5 = 40 sizing and
 * readability checks), and the four states this repair adds — an unknown write outcome, an
 * available replay entry, an in-flight write, and the render-error fallback — are measured at
 * 1440 and 393.  Everything is asserted against the viewport that was actually set and the
 * layout viewport, never against a possibly-widened `window.innerWidth`.
 *
 * The unknown-outcome and in-flight states are produced with Playwright request interception
 * (`page.route`), not with a switch in the product: the browser really does send the write,
 * the server really does commit it, and only the response is withheld or delayed.
 */

const HELPER = fileURLToPath(new URL('../review-db.mjs', import.meta.url))
type CredentialName = 'operator' | 'reader'
type RunKind = 'main' | 'fault-desktop' | 'fault-mobile'

const ORIGINS = resolveE2EOrigins()
const FAULT_ORIGINS: Record<string, string> = faultOriginsByProject(ORIGINS)
const FAULT_RUNS: Record<string, RunKind> = faultKindsByProject(ORIGINS) as Record<string, RunKind>

const WIDTHS = [375, 393, 768, 1024, 1440]
// Written outside Playwright's `outputDir`: that directory is cleared at the start of every
// run, and a later probe run would delete the evidence this matrix produces.
const SHOT_DIR = process.env.REVIEW_E2E_LAYOUT_EVIDENCE_DIR ?? fileURLToPath(
  new URL('../../../.local/fix04-b-20260914/layout-evidence/', import.meta.url),
)

/**
 * Every measurement taken, written to `metrics.json` next to the screenshots.  The numbers
 * (viewport, layout viewport, scroll width, clipped controls) are the auditable part of this
 * matrix; the screenshots are kept alongside them for a human review.
 */
const MEASURED: {
  state: string
  width: number
  clientWidth: number
  innerWidth: number
  scrollWidth: number
  clippedControls: string[]
}[] = []

interface LayoutMetrics {
  expectedWidth: number
  clientWidth: number
  innerWidth: number
  scrollWidth: number
  clippedControls: string[]
}

function credential(name: CredentialName, run: RunKind = 'main'): string {
  return execFileSync(process.execPath, [HELPER, 'token', name, run], { encoding: 'utf8' }).trim()
}

async function signIn(page: Page, token: string): Promise<void> {
  await page.getByLabel('Bearer 凭据').fill(token)
  await page.getByRole('button', { name: '验证凭据' }).click()
  await page.getByRole('button', { name: '身份与权限' }).click()
  await expect(page.getByText('已验证身份')).toBeVisible({ timeout: 20_000 })
}

async function startReview(page: Page, caseId: string): Promise<void> {
  const runId = page.locator('.run-title code').first()
  await page.getByLabel('已授权案例').selectOption(caseId)
  // A preloaded same-case Run can arrive after sign-in. Bind assertions to the
  // POST response, never an old heading or a null-before-load selection.
  const created = page.waitForResponse((response) => response.url().endsWith('/api/v2/reviews') && response.request().method() === 'POST')
  await page.getByRole('button', { name: '运行审查', exact: true }).click()
  const response = await created
  expect(response.status()).toBe(202)
  const body = await response.json()
  expect(body.request.case_id).toBe(caseId)
  await expect(runId).toHaveAttribute('title', body.run_id)

}

async function measureLayout(page: Page, expectedWidth: number): Promise<LayoutMetrics> {
  return page.evaluate((expected) => {
    const root = document.documentElement
    const selectors = [
      '.run-title',
      '.gate-actions button',
      '.view-tabs button',
      '.notice',
      '.scope-hint',
      '.boundary-state',
      '.request-replay',
      '.receipt-panel',
    ]
    const clipped: string[] = []
    for (const selector of selectors) {
      for (const node of Array.from(document.querySelectorAll<HTMLElement>(selector))) {
        const rect = node.getBoundingClientRect()
        if (rect.width === 0 && rect.height === 0) continue
        if (rect.right > root.clientWidth + 1 || rect.left < -1) {
          clipped.push(`${selector} (${Math.round(rect.left)}..${Math.round(rect.right)})`)
        }
      }
    }
    for (const column of document.querySelectorAll<HTMLElement>('.summary-metrics > div')) {
      const bounds = column.getBoundingClientRect()
      for (const child of column.children) {
        const rect = child.getBoundingClientRect()
        if (rect.left < bounds.left - 1 || rect.right > bounds.right + 1) {
          clipped.push('.summary-metrics content exceeds its column')
        }
      }
    }
    return {
      expectedWidth: expected,
      clientWidth: root.clientWidth,
      innerWidth: window.innerWidth,
      scrollWidth: root.scrollWidth,
      clippedControls: clipped.slice(0, 6),
    }
  }, expectedWidth)
}

async function expectFitsViewport(page: Page, width: number): Promise<LayoutMetrics> {
  const metrics = await measureLayout(page, width)
  expect(
    metrics.scrollWidth,
    `horizontal overflow at ${width}px: scrollWidth ${metrics.scrollWidth} > clientWidth ${metrics.clientWidth}`,
  ).toBeLessThanOrEqual(metrics.clientWidth)
  expect(metrics.clippedControls, `clipped controls at ${width}px`).toEqual([])
  expect(metrics.clientWidth).toBeLessThanOrEqual(width)
  expect(metrics.clientWidth).toBeGreaterThanOrEqual(Math.min(width, 320) - 20)
  return metrics
}

/** Measure one state at all five widths and record the screenshots for the two ends. */
async function measureEveryWidth(
  page: Page,
  label: string,
  readable: () => Promise<void>,
): Promise<LayoutMetrics[]> {
  const measured: LayoutMetrics[] = []
  for (const width of WIDTHS) {
    await page.setViewportSize({ width, height: 900 })
    await readable()
    const metrics = await expectFitsViewport(page, width)
    MEASURED.push({ state: label, ...metrics })
    measured.push(metrics)
    if (width === 375 || width === 1440) {
      mkdirSync(SHOT_DIR, { recursive: true })
      // Screenshots are taken only after the credential field has been cleared, and the
      // suite keeps traces/video off, so no credential can end up in an artifact.
      await page.evaluate(() => window.scrollTo(0, 0))
      await fullPage(page, { path: `${SHOT_DIR}/${label}-${width}.png`, fullPage: true })
    }
  }
  expect(measured.map((entry) => entry.expectedWidth)).toEqual(WIDTHS)
  return measured
}

test.afterAll(() => {
  mkdirSync(SHOT_DIR, { recursive: true })
  writeFileSync(
    `${SHOT_DIR}/metrics.json`,
    `${JSON.stringify({ measuredAt: new Date().toISOString(), widths: WIDTHS, entries: MEASURED }, null, 2)}\n`,
  )
})

test.describe('FIX-04B layout matrix', () => {
  test('measures the eight product states at five widths', async ({ page }) => {
    test.setTimeout(180_000)
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await page.goto('/')
    await signIn(page, credential('operator'))

    // 1. A normal result with evidence and business facts expanded.
    await startReview(page, 'CASE-LOW-001')
    await expect(page.getByRole('heading', { name: '建议通过', exact: true })).toBeVisible({
      timeout: 20_000,
    })
    await page.getByRole('region', { name: '可核验证据', exact: true }).locator('summary').first().click()
    await measureEveryWidth(page, 'result-expanded', async () => {
      await expect(page.getByText('内容校验值').first()).toBeVisible()
    })

    // 2. Every worker failed: the failure lists are the longest content in the page.
    await startReview(page, 'CASE-ALL-FAILURE-001')
    await expect(page.getByRole('heading', { name: '信息不足' })).toBeVisible({ timeout: 20_000 })
    await measureEveryWidth(page, 'all-workers-failed', async () => {
      await expect(page.getByRole('heading', { name: 'Worker 失败', exact: true })).toBeVisible()
    })

    // 3. A recorded decision: the receipt carries a 64-hex digest.
    await startReview(page, 'CASE-LOW-001')
    await expect(page.getByRole('heading', { name: '建议通过', exact: true })).toBeVisible({
      timeout: 20_000,
    })
    await page.getByRole('button', { name: '确认建议（本地）', exact: true }).click()
    await expect(page.getByRole('heading', { name: '本地模拟回执' })).toBeVisible({
      timeout: 20_000,
    })
    await measureEveryWidth(page, 'receipt', async () => {
      await expect(page.getByText('幂等键摘要')).toBeVisible()
    })

    // 4. The approval audit: the longest identifiers the page renders.
    await page.getByRole('tab', { name: /审批审计/ }).click()
    await measureEveryWidth(page, 'audit', async () => {
      await expect(page.getByRole('heading', { name: '审批审计' })).toBeVisible()
      await expect(page.locator('.audit-list article')).toHaveCount(1)
      await expect(page.locator('.audit-list .action-label')).toHaveText('confirm')
    })

    // 5. The Trace timeline.
    await page.getByRole('tab', { name: /运行 Trace/ }).click()
    await expect(page.getByRole('heading', { name: '运行 Trace' })).toBeVisible()
    await measureEveryWidth(page, 'trace', async () => {
      await expect(page.getByText('run_completed').first()).toBeVisible({ timeout: 20_000 })
    })

    // 6. A closed runtime failure.
    await startReview(page, 'CASE-RUNTIME-FAILURE-001')
    await expect(page.getByRole('heading', { name: 'Review 运行失败并已关闭' })).toBeVisible({
      timeout: 20_000,
    })
    await measureEveryWidth(page, 'runtime-failure', async () => {
      await expect(page.getByRole('heading', { name: 'Review 运行失败并已关闭' })).toBeVisible()
    })
  })

  test('measures the interrupted and resumed states at five widths', async ({ page }, testInfo) => {
    test.setTimeout(180_000)
    const faultOrigin = FAULT_ORIGINS[testInfo.project.name]
    const faultRun = FAULT_RUNS[testInfo.project.name]
    expect(faultOrigin && faultRun).toBeTruthy()

    await page.goto(`${faultOrigin}/`)
    await signIn(page, credential('operator', faultRun))
    await startReview(page, 'CASE-LOW-001')

    // 7. Interrupted: the densest recovery view.
    await expect(page.getByTestId('interrupted-state')).toBeVisible({ timeout: 45_000 })
    await measureEveryWidth(page, 'interrupted', async () => {
      await expect(page.getByTestId('recovery-budget')).toHaveText('恢复 0 / 2')
    })

    // FIX-FINAL: render the deadline reason using the real interrupted projection
    // with only its bounded reason code replaced. Backend expiry is tested separately.
    const runId = await page.locator('.run-title code').first().getAttribute('title')
    const detailUrl = `**/reviews/${runId}`
    await page.route(detailUrl, async (route) => {
      const response = await route.fetch()
      const body = await response.json()
      await route.fulfill({
        response,
        json: { ...body, error_code: 'generation_deadline_exceeded' },
      })
    })
    await page.getByRole('button', { name: '刷新当前 Review' }).click()
    await expect(page.getByText('本次执行已超过截止时间', { exact: true })).toBeVisible()
    const deadlineMetrics: LayoutMetrics[] = []
    for (const width of WIDTHS) {
      await page.setViewportSize({ width, height: 900 })
      await expect(page.getByTestId('resume-button')).toBeVisible()
      await expect(page.getByTestId('recovery-budget')).toHaveText('恢复 0 / 2')
      deadlineMetrics.push(await expectFitsViewport(page, width))
      if (width === 375 || width === 1440) {
        await fullPage(page, { path: `${SHOT_DIR}/deadline-${width}.png`, fullPage: true })
      }
    }
    writeFileSync(`${SHOT_DIR}/deadline-metrics.json`, JSON.stringify(deadlineMetrics, null, 2))
    await page.unroute(detailUrl)
    await page.getByRole('button', { name: '刷新当前 Review' }).click()

    // The injected outage must be over before an authorized resume.
    await page.waitForTimeout(1500)
    await page.getByTestId('resume-button').click()
    await expect(page.getByRole('heading', { name: '建议通过' })).toBeVisible({ timeout: 45_000 })

    // 8. Resumed, with the consumed budget chip.
    await measureEveryWidth(page, 'resumed', async () => {
      await expect(page.getByTestId('recovery-budget-chip')).toHaveText('恢复 1 / 2')
    })
  })

  test('measures the unknown outcome and the replay entry at desktop and 393px', async ({
    page,
  }) => {
    test.setTimeout(120_000)
    await page.goto('/')
    await signIn(page, credential('operator'))
    await startReview(page, 'CASE-LOW-001')
    await expect(page.getByRole('heading', { name: '建议通过', exact: true })).toBeVisible({
      timeout: 20_000,
    })

    // The write really reaches the server and really commits; only its response is lost.
    await page.route('**/reviews/*/confirm', async (route) => {
      await route.fetch()
      await route.abort('connectionfailed')
    })
    await page.getByRole('button', { name: '确认建议（本地）', exact: true }).click()

    // The workbench says the outcome is unknown and offers the original request,
    // rather than guessing or retrying on its own.
    await expect(page.getByTestId('pending-request-state')).toContainText('结果未知', {
      timeout: 20_000,
    })
    await page.unroute('**/reviews/*/confirm')
    await expect(page.getByRole('button', { name: '重放原动作' })).toBeEnabled()
    await page.getByRole('heading', { name: '本地模拟回执' }).waitFor({ timeout: 20_000 })

    for (const width of [1440, 393]) {
      await page.setViewportSize({ width, height: 900 })
      await expect(page.getByTestId('pending-request')).toBeVisible()
      await expectFitsViewport(page, width)
      mkdirSync(SHOT_DIR, { recursive: true })
      await fullPage(page, { path: `${SHOT_DIR}/unknown-outcome-${width}.png`, fullPage: true })
    }
  })

  test('measures the in-flight write state at desktop and 393px', async ({ page }) => {
    test.setTimeout(120_000)
    await page.goto('/')
    await signIn(page, credential('operator'))
    await startReview(page, 'CASE-LOW-001')
    await expect(page.getByRole('heading', { name: '建议通过', exact: true })).toBeVisible({
      timeout: 20_000,
    })

    // Hold the response: the write is genuinely in flight while the page is measured.
    await page.route('**/reviews/*/confirm', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 20_000))
      await route.continue()
    })
    await page.getByRole('button', { name: '确认建议（本地）', exact: true }).click()

    for (const width of [1440, 393]) {
      await page.setViewportSize({ width, height: 900 })
      await expect(page.locator('.review-view')).toHaveAttribute('aria-busy', 'true')
      await expect(page.getByRole('button', { name: '确认建议（本地）', exact: true })).toBeDisabled()
      await expectFitsViewport(page, width)
      mkdirSync(SHOT_DIR, { recursive: true })
      await fullPage(page, { path: `${SHOT_DIR}/busy-${width}.png`, fullPage: true })
    }
    await page.unroute('**/reviews/*/confirm')
  })

  test('measures the render-error fallback at desktop and 393px', async ({ page }) => {
    test.setTimeout(120_000)
    await page.goto('/')
    await signIn(page, credential('operator'))
    await startReview(page, 'CASE-LOW-001')
    await expect(page.getByRole('heading', { name: '建议通过', exact: true })).toBeVisible({
      timeout: 20_000,
    })

    // A test-side injection: the browser's date formatting is removed *after* the page is
    // healthy, then a re-render formats a timestamp and throws.  The product has no fault
    // switch; this is the real page under a broken browser dependency.
    await page.evaluate(() => {
      ;(window as unknown as { Intl: { DateTimeFormat: unknown } }).Intl.DateTimeFormat =
        function Broken() {
          throw new Error('FIX04-LAYOUT-CANARY')
        }
    })
    await page.getByRole('tab', { name: /运行 Trace/ }).click()
    await expect(page.getByTestId('review-error-boundary')).toBeVisible({ timeout: 20_000 })
    for (const width of [1440, 393]) {
      await page.setViewportSize({ width, height: 900 })
      await expect(page.getByTestId('review-error-boundary')).toBeVisible()
      await expect(
        page.getByRole('button', { name: '重新读取视图' }),
      ).toBeVisible()
      await expect(
        page.getByRole('button', { name: '清除凭据并返回登录' }),
      ).toBeVisible()
      await expectFitsViewport(page, width)
      mkdirSync(SHOT_DIR, { recursive: true })
      await fullPage(page, { path: `${SHOT_DIR}/error-fallback-${width}.png`, fullPage: true })
    }
  })
})
