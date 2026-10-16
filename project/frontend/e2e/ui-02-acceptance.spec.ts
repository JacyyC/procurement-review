import { fullPage } from './full-page'
import { execFileSync } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { expect, test, type Page } from '@playwright/test'
import { resolveE2EOrigins } from './origins.mjs'
import { validateRun } from '../src/apiContract'

const helper = fileURLToPath(new URL('./review-db.mjs', import.meta.url))
const output = process.env.REVIEW_ACCEPTANCE_OUTPUT ?? fileURLToPath(new URL('../.local/ui-02/acceptance/', import.meta.url))
const api = resolveE2EOrigins().main.apiOrigin
const scenarios = [
  ['CASE-LOW-001', '建议通过'], ['CASE-HIGH-001', '建议拒绝'],
  ['CASE-CONFLICT-001', '需要人工复核'], ['CASE-UNKNOWN-001', '信息不足'],
  ['CASE-PARTIAL-FAILURE-001', '信息不足'], ['CASE-ALL-FAILURE-001', '信息不足'],
  ['CASE-VERSION-001', '信息不足'], ['CASE-RUNTIME-FAILURE-001', 'Review 运行失败并已关闭'],
  ['CASE-CANCEL-001', 'Review 已取消'],
]
function credential(name = 'operator') {
  return execFileSync(process.execPath, [helper, 'token', name], { encoding: 'utf8' }).trim()
}
async function signIn(page: Page, token: string) {
  await page.goto('/')
  await page.getByLabel('Bearer 凭据').fill(token)
  await page.getByRole('button', { name: '验证凭据' }).click()
  await page.getByRole('button', { name: '身份与权限' }).click()
  await expect(page.getByText('已验证身份')).toBeVisible()
}
async function readRun(page: Page, token: string, id: string) {
  const response = await page.request.get(`${api}/api/v2/reviews/${id}`, { headers: { Authorization: `Bearer ${token}` } })
  expect(response.status()).toBe(200)
  return validateRun(await response.json(), 'run', { expectRunId: id })
}

test('UI-02 exercises all nine cases, reject replay, search and fast switching on the real API', async ({ page }, info) => {
  test.setTimeout(180_000)
  mkdirSync(output, { recursive: true })
  const token = credential()
  await signIn(page, token)
  const observations = []
  for (const [caseId, title] of scenarios) {
    await page.getByLabel('已授权案例').selectOption(caseId)
    const created = page.waitForResponse((response) => response.url().endsWith('/api/v2/reviews') && response.request().method() === 'POST')
    await page.getByRole('button', { name: '运行审查' }).click()
    const response = await created
    expect(response.status()).toBe(202)
    const body = await response.json()
    const idNode = page.locator('.run-title code').first()
    await expect(idNode).toHaveAttribute('title', body.run_id)
    if (caseId === 'CASE-CANCEL-001') await page.getByRole('button', { name: '取消', exact: true }).click()
    await expect(page.getByRole('heading', { name: title, exact: true })).toBeVisible()
    const id = (await idNode.getAttribute('title'))!
    const run = await readRun(page, token, id)
    expect(run.request.case_id).toBe(caseId)
    if (caseId === 'CASE-CANCEL-001') expect(run.receipt).toBeNull()
    observations.push({ caseId, id, status: run.status, recommendation: run.summary?.recommendation ?? null })
    await fullPage(page, { path: `${output}/${info.project.name}-${caseId}.png`, fullPage: true })
  }
  expect(new Set(observations.map((item) => item.id)).size).toBe(9)
  const search = page.getByRole('textbox', { name: '搜索运行记录' })
  await search.fill('no-match-ui-02')
  await expect(page.getByText('没有匹配的运行')).toBeVisible()
  await expect(page.locator('[data-run-id]')).toHaveCount(0)
  await page.getByRole('button', { name: '清空搜索' }).click()
  await expect(page.locator('[data-run-id]').first()).toBeVisible()
  const low = observations[0].id
  const high = observations[1].id
  await search.fill(high)
  await expect(page.locator('[data-run-id]')).toHaveCount(1)
  await page.getByRole('button', { name: '清空搜索' }).click()
  for (const id of [low, high, low, high]) await page.locator(`[data-run-id="${id}"]`).click()
  await expect(page.locator('.run-title code').first()).toHaveAttribute('title', high)
  await expect(page.getByRole('heading', { name: '建议拒绝', exact: true })).toBeVisible()
  await page.getByRole('button', { name: '拒绝建议（本地）', exact: true }).click()
  await expect(page.getByRole('heading', { name: '本地模拟回执' })).toBeVisible()
  const before = await readRun(page, token, high)
  expect(before.status).toBe('rejected')
  await page.getByRole('button', { name: '重放原动作', exact: true }).click()
  await expect(page.getByTestId('pending-request-state')).toContainText('服务端已响应')
  const after = await readRun(page, token, high)
  expect(after.receipt).toEqual(before.receipt)
  expect(after.version).toBe(before.version)
  expect(after.audit_event_count).toBe(before.audit_event_count)
  expect(after.trace_event_count).toBe(before.trace_event_count)
  await page.getByRole('tab', { name: /审批审计/ }).click()
  await expect(page.locator('.audit-list .action-label')).toHaveText('reject')
  await page.getByRole('button', { name: '退出并清除' }).click()
  await expect(page.getByRole('heading', { name: '需要本地凭据' })).toBeVisible()
  await expect(page.locator('[data-run-id]')).toHaveCount(0)
  writeFileSync(`${output}/${info.project.name}-nine-cases.json`, `${JSON.stringify({ observations, rejectReplay: { before, after } }, null, 2)}\n`)
})

test('UI-02 verifies HTTP 401/403 as well as hidden scope controls', async ({ page }) => {
  const missing = await page.request.get(`${api}/api/v2/cases`)
  expect(missing.status()).toBe(401)
  const reader = credential('reader')
  await signIn(page, reader)
  const rejected = await page.request.post(`${api}/api/v2/reviews`, {
    headers: { Authorization: `Bearer ${reader}`, 'Idempotency-Key': 'ui02-reader-denial' },
    data: { api_version: 'review-api.v2', case_id: 'CASE-LOW-001', case_version: 'case-low-001.v1' },
  })
  expect(rejected.status()).toBe(403)
  expect((await rejected.json()).error.code).toBe('insufficient_scope')
  await expect(page.getByRole('button', { name: '运行审查' })).toHaveCount(0)
  await expect(page.getByRole('tab', { name: /Trace|审批审计/ })).toHaveCount(0)
})
