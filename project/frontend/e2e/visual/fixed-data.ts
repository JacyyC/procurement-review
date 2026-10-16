import { readFileSync } from 'node:fs'
import { expect, type Page } from '@playwright/test'
import { validateAudit, validateCases, validateReviewList, validateSession, validateTrace } from '../../src/apiContract'

// The original remains immutable. All envelopes pass the current runtime validators;
// no product switch, credential, React-state injection or alternate UI is involved.
const original = JSON.parse(readFileSync(new URL('../fixtures/preview-data.json', import.meta.url), 'utf8'))
export const fixed = {
  session: validateSession(original.session),
  cases: validateCases(original.cases),
  reviews: validateReviewList(original.reviews),
  // Historical event/Trace ids predate the parent-identity guard. Align only these
  // identifiers to the frozen Run's trace_id; all visible business data stays intact.
  trace: validateTrace({ ...original.selected_trace, trace: {
    ...original.selected_trace.trace,
    trace_id: original.reviews.runs[0].trace_id,
    events: original.selected_trace.trace.events.map((event: Record<string, unknown>) => ({
      ...event, trace_id: original.reviews.runs[0].trace_id,
    })),
  } }),
  audit: validateAudit(original.selected_audit),
  ui: original.ui as { selectedRun: string; selectedCase: string; runs: string[] },
}

export async function installFixedAPI(page: Page, data = fixed): Promise<void> {
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url())
    const resource = url.pathname.replace('/api/v2/', '')
    const run = data.reviews.runs.find((item) => resource.split('/')[1] === item.run_id)
    let body: unknown
    if (resource === 'session') body = data.session
    else if (resource === 'cases') body = data.cases
    else if (resource === 'reviews' && route.request().method() === 'GET') body = data.reviews
    else if (resource === 'reviews' && route.request().method() === 'POST') {
      // Reconstitute the in-memory create replay using a real click on a different
      // existing Run. The reference's selected Run therefore has no pending request.
      const request = route.request().postDataJSON()
      body = data.reviews.runs.find((item) => item.request.case_id === request.case_id)
    } else if (run && resource.endsWith('/trace')) {
      body = run.run_id === data.ui.selectedRun ? data.trace : {
        ...data.trace, trace: { ...data.trace.trace, run_id: run.run_id, trace_id: run.trace_id, events: [] },
      }
    } else if (run && resource.endsWith('/audit')) {
      body = run.run_id === data.ui.selectedRun ? data.audit : { ...data.audit, run_id: run.run_id, events: [] }
    } else if (run && resource === `reviews/${run.run_id}`) body = run
    else throw new Error(`Unregistered visual fixture request: ${route.request().method()} ${resource}`)
    await route.fulfill({ json: body })
  })
}

export async function signInFixed(page: Page): Promise<void> {
  await page.goto('/')
  await page.addStyleTag({ content: '*, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; }' })
  await page.getByLabel('Bearer 凭据').fill('test-only-visual-not-a-credential')
  await page.getByRole('button', { name: '验证凭据' }).click()
  await expect(page.getByRole('button', { name: '身份与权限' })).toBeVisible()
}

export async function restoreApprovedSelection(page: Page): Promise<void> {
  await signInFixed(page)
  await page.getByLabel('已授权案例').selectOption('CASE-HIGH-001')
  await page.getByRole('button', { name: '运行审查' }).click()
  await expect(page.getByRole('button', { name: '重放最近创建请求' })).toBeEnabled()
  await page.locator(`[data-run-id="${fixed.ui.selectedRun}"]`).click()
  await page.getByLabel('已授权案例').selectOption(fixed.ui.selectedCase)
  const close = page.getByRole('button', { name: '关闭反馈' })
  if (await close.count()) await close.click()
  await expect(page.locator('.run-title code').first()).toHaveAttribute('title', fixed.ui.selectedRun)
  await expect(page.getByRole('heading', { name: '建议通过', exact: true })).toBeVisible()
  await expect(page.locator('[data-run-id]')).toHaveCount(10)
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.getByRole('button', { name: '刷新当前 Review' })).toBeEnabled()
}

export async function settle(page: Page, y = 0): Promise<void> {
  await page.mouse.move(0, 0)
  await page.evaluate(async (scrollY) => {
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur()
    await document.fonts.ready
    window.scrollTo({ top: scrollY, left: 0, behavior: 'instant' })
    await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))
  }, y)
}
