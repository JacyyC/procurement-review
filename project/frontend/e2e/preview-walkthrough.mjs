/** Walk a separately running local preview; never persist its credential in evidence. */
import { execFileSync } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { chromium, expect } from '@playwright/test'

const origin = process.env.REVIEW_PREVIEW_ORIGIN ?? 'http://127.0.0.1:5188'
const output = process.env.REVIEW_PREVIEW_OUTPUT ?? fileURLToPath(new URL('../.local/ui-03/preview-evidence/', import.meta.url))
const token = execFileSync(process.execPath, [fileURLToPath(new URL('./review-db.mjs', import.meta.url)), 'token', 'operator'], {
  env: { ...process.env, REVIEW_E2E_STATE_DIR: process.env.REVIEW_E2E_STATE_DIR ?? '.local/ui-03/preview' }, encoding: 'utf8',
}).trim()
const browser = await chromium.launch()
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, locale: 'zh-CN', timezoneId: 'Asia/Shanghai', reducedMotion: 'reduce' })
  await page.goto(origin)
  await page.locator('.quick-guide summary').click()
  await page.getByRole('button', { name: '验证凭据，开始审查' }).click()
  await expect(page.getByLabel('Bearer 凭据')).toBeFocused()
  await page.getByLabel('Bearer 凭据').fill(token)
  await page.getByRole('button', { name: '验证凭据' }).click()
  await page.getByRole('button', { name: '身份与权限' }).click()
  await expect(page.getByText('已验证身份')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('button', { name: '身份与权限' })).toBeFocused()
  await page.locator('.quick-guide summary').click()
  await page.getByRole('button', { name: '选择案例，发起审查' }).click()
  await expect(page.getByLabel('已授权案例')).toBeFocused()
  await expect(page.getByLabel('Bearer 凭据')).toHaveCount(0)
  await expect(page.getByLabel('已授权案例').locator('option')).toHaveCount(9)
  await page.getByLabel('已授权案例').selectOption('CASE-LOW-001')
  const current = page.locator('.run-title code').first()
  const created = page.waitForResponse((response) => response.url().endsWith('/api/v2/reviews') && response.request().method() === 'POST')
  await page.getByRole('button', { name: '运行审查' }).click()
  const createdResponse = await created
  expect(createdResponse.status()).toBe(202)
  await expect(current).toHaveAttribute('title', (await createdResponse.json()).run_id)
  await expect(page.getByRole('heading', { name: '建议通过', exact: true })).toBeVisible({ timeout: 20_000 })
  const runId = await page.locator('.run-title code').first().getAttribute('title')
  const count = await page.locator('[data-run-id]').count()
  const replay = page.waitForResponse((response) => response.url().endsWith('/api/v2/reviews') && response.request().method() === 'POST')
  await page.getByRole('button', { name: '重放最近创建请求' }).click()
  expect((await replay).status()).toBe(200)
  await expect(page.getByRole('button', { name: '运行审查' })).toBeEnabled()
  await expect(page.locator('[data-run-id]')).toHaveCount(count)
  await expect(page.locator('.run-title code').first()).toHaveAttribute('title', runId)
  for (const region of ['可核验证据', '只读业务事实']) {
    const summary = page.getByRole('region', { name: region, exact: true }).locator('summary').first()
    await summary.click()
    await expect(page.getByRole('region', { name: region, exact: true }).locator('details[open]')).toHaveCount(1)
    await summary.click()
  }
  await page.getByRole('tab', { name: /运行 Trace/ }).click()
  await expect(page.getByText('trace.v1 · 等待人工 Gate')).toBeVisible()
  await expect(page.getByText('review_workflow', { exact: true })).toBeVisible()
  await page.getByRole('tab', { name: /审批审计/ }).click()
  await expect(page.getByRole('heading', { name: '审批审计', exact: true })).toBeVisible()
  await page.getByRole('tab', { name: '审查结果', exact: true }).click()
  await expect(page.getByRole('button', { name: '确认建议（本地）', exact: true })).toBeEnabled()
  const close = page.getByRole('button', { name: '关闭反馈' })
  if (await close.count()) await close.click()
  mkdirSync(output, { recursive: true })
  for (const width of [1440, 375]) {
    await page.setViewportSize({ width, height: 1000 })
    await page.locator('.quick-guide summary').click()
    await page.getByRole('button', { name: '选择案例，发起审查' }).click()
    await expect(page.getByLabel('已授权案例')).toBeFocused()
    await page.getByRole('button', { name: '身份与权限' }).click()
    await expect(page.getByText('已验证身份')).toBeVisible()
    await page.getByRole('button', { name: '关闭身份面板' }).click()
    await expect(page.getByRole('button', { name: '身份与权限' })).toBeFocused()
    await page.locator('#page-title').scrollIntoViewIfNeeded()
    expect(await page.locator('html').evaluate((node) => node.scrollWidth)).toBe(width)
    await page.screenshot({ path: `${output}/preview-${width}.png`, fullPage: true, animations: 'disabled' })
  }
  const health = await page.request.get(`${origin}/api/v2/health`)
  expect(health.status()).toBe(200)
  const response = await page.request.get(`${origin}/api/v2/reviews/${runId}`, { headers: { Authorization: `Bearer ${token}` } })
  expect(response.status()).toBe(200)
  const run = await response.json()
  expect(run.status).toBe('pending_confirmation')
  writeFileSync(`${output}/walkthrough.json`, JSON.stringify({
    origin, runId, guideCredentialFocus: true, guideCaseFocus: true, identityFocusReturn: true, cases: 9, createReplayAddedRuns: 0, evidenceAndFactsExpanded: true,
    traceAwaitingGate: true, auditReachable: true, humanActionsAvailable: true,
    healthStatus: health.status(), runStatus: run.status, browser: browser.version(), measuredAt: new Date().toISOString(),
  }, null, 2) + '\n')
  console.log('Preview walkthrough passed; token not persisted in evidence.')
} finally { await browser.close() }
