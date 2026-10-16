import { fullPage } from './full-page'
import { expect, test, type Page } from '@playwright/test'
import { fixed, installFixedAPI, signInFixed } from './visual/fixed-data'

async function openGuide(page: Page) {
  const guide = page.locator('.quick-guide')
  if (!await guide.evaluate((node) => (node as HTMLDetailsElement).open)) await guide.locator('summary').click()
  return guide
}
async function focusedVisible(page: Page, id: string) {
  await expect(page.locator(`#${id}`)).toBeFocused()
  expect(await page.locator(`#${id}`).evaluate((node) => {
    const r = node.getBoundingClientRect()
    return r.top >= 0 && r.left >= 0 && r.right <= innerWidth && r.bottom <= innerHeight
  })).toBe(true)
  await expect(page.locator('.quick-guide')).not.toHaveAttribute('open')
}

test('guide sends disconnected keyboard users to the credential, including repeated in-view use', async ({ page }) => {
  await installFixedAPI(page)
  const writes: string[] = []
  page.on('request', (request) => { if (request.method() !== 'GET') writes.push(request.url()) })
  await page.goto('/')
  await page.emulateMedia({ reducedMotion: 'reduce' })
  for (const repeated of [false, true]) {
    if (repeated) await page.getByLabel('Bearer 凭据').scrollIntoViewIfNeeded()
    const guide = await openGuide(page)
    const action = guide.locator('[data-guide-action], a[href="#case-control"]')
    await action.focus()
    await page.keyboard.press('Enter')
    await focusedVisible(page, 'bearer-credential')
  }
  expect(writes).toEqual([])
})

test('guide focuses available cases without choosing or creating and handles expiry while open', async ({ page }) => {
  await installFixedAPI(page)
  await signInFixed(page)
  await expect(page.getByLabel('已授权案例')).toBeEnabled()
  const selected = await page.getByLabel('已授权案例').inputValue()
  const writes: string[] = []
  page.on('request', (request) => { if (request.method() !== 'GET') writes.push(request.url()) })
  for (const motion of ['no-preference', 'reduce'] as const) {
    await page.emulateMedia({ reducedMotion: motion })
    const guide = await openGuide(page)
    await guide.getByRole('button', { name: '选择案例，发起审查' }).click()
    await focusedVisible(page, 'case-select')
    expect(await page.getByLabel('已授权案例').inputValue()).toBe(selected)
  }
  await openGuide(page)
  await page.route('**/api/v2/reviews/*', (route) => route.fulfill({ status: 401, json: { error: { code: 'invalid_credential', message: 'test_only', request_id: 'guide-expired', details: {} } } }))
  await page.getByRole('button', { name: '刷新当前 Review' }).click()
  const guide = page.locator('.quick-guide')
  await expect(guide.getByRole('button', { name: '验证凭据，开始审查' })).toBeVisible()
  await guide.getByRole('button', { name: '验证凭据，开始审查' }).click()
  await focusedVisible(page, 'bearer-credential')
  await expect(page.getByTestId('identity-subject')).toHaveCount(0)
  expect(writes).toEqual([])
})

for (const state of ['no-case-scope', 'no-create-scope', 'empty', 'failed', 'loading', 'connecting', 'creating']) {
  test(`guide explains ${state} and never targets an unavailable control`, async ({ page }, info) => {
    const data = structuredClone(fixed)
    if (state === 'no-case-scope') data.session.scopes = data.session.scopes.filter((scope) => scope !== 'case:read')
    if (state === 'no-create-scope') data.session.scopes = data.session.scopes.filter((scope) => scope !== 'review:create')
    if (state === 'empty') data.cases.cases = []
    await installFixedAPI(page, data)
    let release = () => {}
    const held = new Promise<void>((resolve) => { release = resolve })
    if (state === 'loading') await page.route('**/api/v2/cases', async (route) => { await held; await route.fulfill({ json: data.cases }) })
    if (state === 'connecting') await page.route('**/api/v2/session', async (route) => { await held; await route.fulfill({ json: data.session }) })
    if (state === 'failed') await page.route('**/api/v2/cases', (route) => route.fulfill({ status: 503, json: { error: { code: 'review_store_unavailable', message: 'test_only', request_id: 'guide-failed', details: {} } } }))
    if (state === 'creating') await page.route('**/api/v2/reviews', async (route) => {
      if (route.request().method() === 'GET') return route.fallback()
      await held
      await route.fulfill({ json: data.reviews.runs[0] })
    })
    try {
      await page.goto('/')
      await page.getByLabel('Bearer 凭据').fill('test-only-guide')
      await page.getByRole('button', { name: '验证凭据' }).click()
      if (state !== 'connecting') await expect(page.getByRole('button', { name: '身份与权限' })).toBeVisible()
      if (state === 'creating') {
        await page.getByLabel('已授权案例').selectOption(data.reviews.runs[0].request.case_id)
        await page.getByRole('button', { name: '运行审查', exact: true }).click()
      }
      const writes: string[] = []
      page.on('request', (request) => { if (request.method() !== 'GET') writes.push(request.url()) })
      const guide = await openGuide(page)
      await fullPage(page, { path: `e2e/ui-03-evidence/guide-open-${state}-${info.project.name}.png`, fullPage: true })
      if (['loading', 'connecting', 'creating'].includes(state)) {
        const label = state === 'loading' ? '正在加载案例…' : state === 'connecting' ? '正在验证凭据…' : '正在发起审查…'
        await expect(guide.getByRole('status')).toContainText(label)
        await expect(guide.getByRole('button', { name: label })).toBeDisabled()
        release()
        await expect(guide.getByRole('button', { name: '选择案例，发起审查' })).toBeEnabled()
        await guide.getByRole('button', { name: '选择案例，发起审查' }).click()
        await focusedVisible(page, 'case-select')
      } else {
        await expect(guide).toContainText(state.startsWith('no-') ? '当前身份无法发起审查' : state === 'empty' ? '当前没有已授权案例' : '工作台读取失败')
        if (state === 'failed') await expect(guide).not.toContainText('当前没有已授权案例')
        await guide.getByRole('button', { name: /查看权限|查看身份/ }).click()
        await expect(page.locator('.quick-guide')).not.toHaveAttribute('open')
        await expect(page.getByRole('button', { name: '身份与权限' })).toHaveAttribute('aria-expanded', 'true')
        await expect(page.locator('#identity-panel')).toBeFocused()
        await expect(page.getByRole('button', { name: '退出并清除' })).toBeVisible()
      }
      expect(writes).toEqual([])
      await fullPage(page, { path: `e2e/ui-03-evidence/guide-${state}-${info.project.name}.png`, fullPage: true })
    } finally { release() }
  })
}
