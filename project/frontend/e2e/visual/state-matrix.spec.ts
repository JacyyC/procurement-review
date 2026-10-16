import { mkdirSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { expect, test } from '@playwright/test'
import { fixed, installFixedAPI, settle, signInFixed } from './fixed-data'

const output = process.env.REVIEW_STATE_OUTPUT ?? fileURLToPath(new URL('../../.local/ui-02/states/', import.meta.url))
const widths = [375, 768, 1024, 1440]
const states = [
  ...fixed.reviews.runs.map((run) => `${run.request.case_id}-${run.status}`),
  'credential', 'credential-invalid', 'empty', 'loading', 'permission', 'processing',
  'interrupted', 'exhausted', 'deadline', 'long-identifiers', 'long-error',
  'trace', 'audit', 'expanded',
]

for (const state of states) {
  test(`${state} remains readable at four widths`, async ({ page }) => {
    test.setTimeout(60_000)
    mkdirSync(output, { recursive: true })
    const data = structuredClone(fixed)
    let release = () => {}
    const hold = new Promise<void>((resolve) => { release = resolve })
    const selected = data.reviews.runs.find((run) => `${run.request.case_id}-${run.status}` === state)
    if (selected) data.reviews.runs = [selected]
    if (['processing', 'interrupted', 'exhausted', 'deadline'].includes(state)) {
      const run = data.reviews.runs[0]
      run.status = state === 'processing' ? 'processing' : 'interrupted'
      run.summary = null
      run.summary_checksum = null
      run.execution = null
      run.evidence_refs = []
      run.business_facts = []
      run.recovery = { ...run.recovery!, resumable: state !== 'exhausted', resume_count: state === 'exhausted' ? 2 : 0 }
      if (state === 'deadline') run.error_code = 'generation_deadline_exceeded'
      data.reviews.runs = [run]
    }
    if (state === 'permission') data.session.scopes = ['case:read', 'review:read']
    if (state === 'empty') { data.reviews.runs = []; data.cases.cases = [] }
    if (state === 'long-identifiers') {
      const run = data.reviews.runs[0]
      // Replace the identity consistently in nested summary/execution projections.
      const id = `run-${'long-identifier-'.repeat(24)}`
      data.reviews.runs = [JSON.parse(JSON.stringify(run).replaceAll(run.run_id, id))]
      data.ui.selectedRun = id
      data.trace = { ...data.trace, trace: { ...data.trace.trace, run_id: id, trace_id: data.reviews.runs[0].trace_id, events: [] } }
      data.audit = { ...data.audit, run_id: id, events: [] }
    }
    if (state === 'audit') data.reviews.runs = [data.reviews.runs.find((run) => run.status === 'confirmed')!]
    await installFixedAPI(page, data)
    if (state === 'loading') await page.route('**/api/v2/reviews', async (route) => { await hold; await route.fulfill({ json: data.reviews }) })
    if (state === 'credential-invalid') await page.route('**/api/v2/session', (route) => route.fulfill({ status: 401, json: { error: { code: 'invalid_credential', message: 'test_only', request_id: 'test-only', details: {} } } }))
    try {
      if (state === 'credential' || state === 'credential-invalid') {
        await page.goto('/')
        if (state === 'credential-invalid') {
          await page.getByLabel('Bearer 凭据').fill('test-only-invalid')
          await page.getByRole('button', { name: '验证凭据' }).click()
          await expect(page.getByRole('alert')).toBeVisible()
        }
        await expect(page.getByRole('heading', { name: '需要本地凭据' })).toBeVisible()
      } else {
        await signInFixed(page)
        if (state === 'empty') await expect(page.getByRole('heading', { name: '尚未创建审查' })).toBeVisible()
        else if (state === 'loading') await expect(page.getByRole('heading', { name: '正在读取本地工作台' })).toBeVisible()
        else await expect(page.locator('.review-view')).toBeVisible()
        if (selected) {
          const headings: Record<string, string> = {
            'CASE-LOW-001': '建议通过', 'CASE-HIGH-001': '建议拒绝',
            'CASE-CANCEL-001': 'Review 已取消', 'CASE-CONFLICT-001': '需要人工复核',
            'CASE-RUNTIME-FAILURE-001': 'Review 运行失败并已关闭',
            'CASE-VERSION-001': '信息不足', 'CASE-UNKNOWN-001': '信息不足',
            'CASE-PARTIAL-FAILURE-001': '信息不足', 'CASE-ALL-FAILURE-001': '信息不足',
          }
          await expect(page.getByRole('heading', { name: headings[selected.request.case_id], exact: true })).toBeVisible()
          if (selected.receipt) await expect(page.getByRole('heading', { name: '本地模拟回执' })).toBeVisible()
        }
        await expect(page.getByText(/接口返回了无法校验的响应/)).toHaveCount(0)
        if (state === 'processing') await expect(page.locator('.processing-state')).toBeVisible()
        if (['interrupted', 'deadline', 'exhausted'].includes(state)) {
          await expect(page.getByTestId('interrupted-state')).toBeVisible()
          if (state === 'exhausted') {
            await expect(page.getByTestId('resume-button')).toBeDisabled()
            await expect(page.getByTestId('recovery-budget')).toHaveText('恢复 2 / 2')
          }
          else await expect(page.getByTestId('resume-button')).toBeEnabled()
        }
        if (state === 'permission') {
          await expect(page.getByRole('button', { name: '运行审查' })).toHaveCount(0)
          await expect(page.getByRole('tab')).toHaveCount(1)
          await page.getByRole('tab').focus()
          await page.keyboard.press('End')
          await page.keyboard.press('ArrowRight')
          await expect(page.getByRole('tab')).toBeFocused()
        }
        if (state === 'expanded') {
          for (const region of ['可核验证据', '只读业务事实']) {
            await page.getByRole('region', { name: region, exact: true }).locator('summary').first().click()
            await expect(page.getByRole('region', { name: region, exact: true }).locator('details[open]')).toHaveCount(1)
          }
        }
        if (state === 'trace' || state === 'audit') {
          await page.getByRole('tab', { name: state === 'trace' ? /运行 Trace/ : /审批审计/ }).click()
          await expect(page.getByRole('heading', { name: state === 'trace' ? '运行 Trace' : '审批审计', exact: true })).toBeVisible()
        }
        if (state === 'long-error') {
          await page.route(`**/api/v2/reviews/${data.reviews.runs[0].run_id}`, (route) => route.fulfill({ status: 503, json: { error: { code: 'test_only_unregistered', message: 'DO_NOT_RENDER'.repeat(40), request_id: 'x'.repeat(64), details: {} } } }))
          await page.getByRole('button', { name: '刷新当前 Review' }).click()
          await expect(page.getByRole('alert')).toContainText('x'.repeat(64))
          await expect(page.getByText(/DO_NOT_RENDER/)).toHaveCount(0)
        }
      }
      const metrics = []
      for (const width of widths) {
        await page.setViewportSize({ width, height: 900 })
        await settle(page)
        const measurement = await page.evaluate(() => {
          const width = document.documentElement.clientWidth
          const clipped = Array.from(document.querySelectorAll<HTMLElement>('button, input, select, .notice, .receipt-panel, .interrupted-state, details[open], pre')).flatMap((node) => {
            const rect = node.getBoundingClientRect()
            if (!rect.width || !rect.height) return []
            return rect.left < -1 || rect.right > width + 1 ? [`${node.tagName}.${node.className}`] : []
          })
          return { width, innerWidth, scrollWidth: document.documentElement.scrollWidth, clipped }
        })
        metrics.push({ state, requestedWidth: width, ...measurement })
        expect(measurement).toMatchObject({ width, innerWidth: width, scrollWidth: width, clipped: [] })
        if (width === 375 || width === 1440) await page.screenshot({ path: `${output}/${state}-${width}.png`, fullPage: true, animations: 'disabled' })
      }
      writeFileSync(`${output}/${state}.json`, `${JSON.stringify(metrics, null, 2)}\n`)
    } finally { release() }
  })
}
