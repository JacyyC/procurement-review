import { fullPage } from './full-page'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { expect, test, type Page } from '@playwright/test'

import { faultKindsByProject, faultOriginsByProject, resolveE2EOrigins } from './origins.mjs'

/**
 * End-to-end workbench run against the authenticated `review-api.v2`.
 *
 * `e2e/review-db.mjs serve` (wired as a Playwright webServer) has already created a
 * disposable PostgreSQL database, migrated and seeded it, issued the local
 * credentials into 0600 files and started the API with REVIEW_DATABASE_URL pointing
 * at that database.  This spec only ever reads a credential value into memory and
 * types it into the page: it is never printed, asserted by text, or attached to a
 * report.  Traces, automatic screenshots and video are off in the Playwright config
 * for that reason.
 */

const HELPER = fileURLToPath(new URL('./review-db.mjs', import.meta.url))

type CredentialName = 'operator' | 'reader'
/** Run kinds owned by `e2e/review-db.mjs`; `main` is the whole-workbench server. */
type RunKind = 'main' | 'fault-desktop' | 'fault-mobile'

/**
 * Every port and origin is derived by the module the Playwright config and the API
 * bootstrap also use, so a non-default configuration cannot leave the browser on one
 * origin while the servers listen on another (FIX-01C, V-C-05).
 */
const ORIGINS = resolveE2EOrigins()
const FAULT_ORIGINS: Record<string, string> = faultOriginsByProject(ORIGINS)
const FAULT_RUNS: Record<string, RunKind> = faultKindsByProject(ORIGINS) as Record<
  string,
  RunKind
>

/** The five CSS widths AC-06 requires, measured against the real viewport. */
const WIDTHS = [375, 393, 768, 1024, 1440]

function credential(name: CredentialName, run: RunKind = 'main'): string {
  return execFileSync(process.execPath, [HELPER, 'token', name, run], {
    encoding: 'utf8',
  }).trim()
}

/**
 * Issue a credential that only this test owns, so revoking it can never disturb a
 * credential another (parallel) test is still using.
 */
function issueOwnCredential(label: string): string {
  return execFileSync(process.execPath, [HELPER, 'issue', label], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit'],
  }).trim()
}

function revokeCredential(label: string): void {
  execFileSync(process.execPath, [HELPER, 'revoke', label], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit'],
  })
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
  const processing = page.getByText('三个只读 Worker 正在并行审查')
  const terminal = page.getByRole('heading', {
    name: /建议通过|建议拒绝|建议有条件通过|需要人工复核|信息不足|Review 运行失败并已关闭|Review 已取消|运行已中断，等待恢复/,
  })
  // A fast local fixture can complete before the transient processing panel paints.
  await expect(processing.or(terminal).first()).toBeVisible({ timeout: 20_000 })
}

interface LayoutMetrics {
  expectedWidth: number
  clientWidth: number
  innerWidth: number
  scrollWidth: number
  /** Critical controls whose box is not fully inside the layout viewport. */
  clippedControls: string[]
}

/** The parts of a `review-api.v2` Run projection these assertions read. */
interface ApiRun {
  run_id: string
  status: string
  version: number
  revision: number
  trace_event_count: number
  receipt: { receipt_id: string } | null
  recovery: { resume_count: number; generation: number } | null
}

interface ApiRunList {
  runs: ApiRun[]
}

interface ApiAudit {
  events: { audit_id: string; sequence: number }[]
}

/**
 * Read one `review-api.v2` endpoint with the credential the page is using.
 *
 * The UI assertions prove the workbench says the right thing; these reads prove the
 * *server* state behind them, which is what makes "the replay added nothing" a fact
 * rather than a rendered sentence (FIX-04A, V-04A-02).
 */
async function apiGet<T>(page: Page, path: string, token: string): Promise<T> {
  const response = await page.request.get(`${ORIGINS.main.apiOrigin}${path}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  expect(response.status(), `${path} answered ${response.status()}`).toBe(200)
  return (await response.json()) as T
}

/**
 * Measure the page against the viewport that was actually set.
 *
 * `documentElement.clientWidth` is the layout viewport; `window.innerWidth` is
 * recorded but never used as the reference, because a page that overflows widens it
 * and would then certify itself (the defect A-09 names).
 */
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
    return {
      expectedWidth: expected,
      clientWidth: root.clientWidth,
      innerWidth: window.innerWidth,
      scrollWidth: root.scrollWidth,
      clippedControls: clipped.slice(0, 6),
    }
  }, expectedWidth)
}

/** Assert the page fits the viewport that was set, and that nothing is clipped. */
async function expectFitsViewport(page: Page, width: number): Promise<LayoutMetrics> {
  const metrics = await measureLayout(page, width)
  expect(
    metrics.scrollWidth,
    `horizontal overflow at ${width}px: scrollWidth ${metrics.scrollWidth} > clientWidth ${metrics.clientWidth}`,
  ).toBeLessThanOrEqual(metrics.clientWidth)
  expect(metrics.clippedControls, `clipped controls at ${width}px`).toEqual([])
  // The layout viewport must track the requested width (a classic scrollbar can take
  // a few pixels); a much smaller value would mean the page is not really being
  // measured at the width the test claims.
  expect(metrics.clientWidth).toBeLessThanOrEqual(width)
  expect(metrics.clientWidth).toBeGreaterThanOrEqual(Math.min(width, 320) - 20)
  return metrics
}

test.describe('authenticated local review workbench', () => {
  test('keeps results and expanded evidence usable across responsive breakpoints', async ({ page }, testInfo) => {
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await page.goto('/')
    await signIn(page, credential('operator'))
    await startReview(page, 'CASE-LOW-001')
    // `startReview` returns as soon as the new Run is on screen, which may still be the
    // processing panel; every terminal-state assertion in this suite uses the same 20s
    // budget the run itself needs.
    await expect(page.getByRole('heading', { name: '建议通过', exact: true })).toBeVisible({
      timeout: 20_000,
    })
    const evidence = page.getByRole('region', { name: '可核验证据', exact: true })
    await evidence.locator('summary').first().click()
    const measured: LayoutMetrics[] = []
    for (const width of WIDTHS) {
      await page.setViewportSize({ width, height: 900 })
      await expect(page.getByText('TEST_ONLY · LOCAL', { exact: true })).toBeVisible()
      await expect(evidence.getByText('内容校验值').first()).toBeVisible()
      measured.push(await expectFitsViewport(page, width))
      expect(await page.evaluate(() => getComputedStyle(document.documentElement).scrollBehavior)).toBe('auto')
      await expect(page.getByRole('button', { name: '确认建议（本地）', exact: true })).toBeEnabled()
    }
    // Every required width was actually measured, and the mobile one is recorded.
    expect(measured.map((entry) => entry.expectedWidth)).toEqual(WIDTHS)
    await page.setViewportSize({ width: 393, height: 851 })
    await fullPage(page, {
      path: `test-results/${testInfo.project.name}-result-393.png`,
      fullPage: true,
    })
    await page.setViewportSize({ width: 1440, height: 900 })
    await fullPage(page, {
      path: `test-results/${testInfo.project.name}-result-1440.png`,
      fullPage: true,
    })

    await page.setViewportSize({ width: 1024, height: 900 })
    await page.getByRole('tab', { name: '审查结果', exact: true }).focus()
    await page.keyboard.press('ArrowRight')
    await expect(page.getByRole('tab', { name: /Trace/ })).toBeFocused()
    await expect(page.getByRole('heading', { name: '运行 Trace', exact: true })).toBeVisible()
  })

  test('keeps the tabs consistent: one tab stop, correct panel link, arrow keys', async ({
    page,
  }) => {
    await page.goto('/')
    await signIn(page, credential('operator'))
    await startReview(page, 'CASE-LOW-001')
    await expect(page.getByRole('heading', { name: '建议通过' })).toBeVisible({
      timeout: 20_000,
    })

    const tabs = page.getByRole('tab')
    await expect(tabs).toHaveCount(3)
    const panel = page.locator('#review-tab-panel')
    await expect(panel).toHaveCount(1)

    // Exactly one tab is in the page's Tab order, and every tab points at the panel.
    const state = await page.evaluate(() => {
      const nodes = Array.from(document.querySelectorAll('[role="tab"]'))
      return nodes.map((node) => ({
        id: node.id,
        selected: node.getAttribute('aria-selected'),
        tabIndex: node.getAttribute('tabindex'),
        controls: node.getAttribute('aria-controls'),
      }))
    })
    expect(state.map((entry) => entry.controls)).toEqual([
      'review-tab-panel',
      'review-tab-panel',
      'review-tab-panel',
    ])
    expect(state.filter((entry) => entry.tabIndex === '0')).toHaveLength(1)
    expect(state[0]).toMatchObject({ id: 'review-tab-result', selected: 'true', tabIndex: '0' })

    await page.getByRole('tab', { name: '审查结果', exact: true }).focus()
    await page.keyboard.press('ArrowLeft')
    await expect(page.getByRole('tab', { name: /审批审计/ })).toBeFocused()
    await expect(panel).toHaveAttribute('aria-labelledby', 'review-tab-audit')
    await page.keyboard.press('Home')
    await expect(page.getByRole('tab', { name: '审查结果', exact: true })).toBeFocused()
    await page.keyboard.press('End')
    await expect(page.getByRole('tab', { name: /审批审计/ })).toBeFocused()
    await page.keyboard.press('ArrowRight')
    await expect(page.getByRole('tab', { name: '审查结果', exact: true })).toBeFocused()
    await expect(panel).toHaveAttribute('aria-labelledby', 'review-tab-result')
  })

  test('verifies a credential, runs a review, confirms, replays and inspects the trace', async ({
    page,
  }, testInfo) => {
    const token = credential('operator')

    await page.goto('/')
    await expect(page.getByText('Review Desk')).toBeVisible()
    await expect(page.getByText('TEST_ONLY · LOCAL')).toBeVisible()
    if (testInfo.project.name === 'chromium-desktop') {
      // The narrow layout intentionally hides this note (see styles.css).
      await expect(page.getByText('建议不是审批事实')).toBeVisible()
    }
    await expect(page.getByText('需要本地凭据')).toBeVisible()

    await signIn(page, token)
    await expect(page.getByTestId('identity-subject')).toHaveText('subject-local-demo')
    await expect(page.getByTestId('identity-tenant')).toHaveText('tenant-local-demo')
    await expect(page.getByTestId('identity-scopes')).toContainText('review:create')
    await expect(page.getByTestId('identity-scopes')).toContainText('review:decide')
    await expect(page.getByTestId('identity-scopes')).toContainText('trace:read')
    await expect(page.getByTestId('identity-scopes')).toContainText('audit:read')
    // The credential form is replaced by the identity panel: the raw value is never
    // rendered into the DOM after submission.
    await expect(page.getByLabel('Bearer 凭据')).toHaveCount(0)
    await expect(page.locator('input#bearer-credential')).toHaveCount(0)
    // The free-text actor input is gone; the actor is the verified subject.
    await expect(page.getByLabel('操作人')).toHaveCount(0)

    await startReview(page, 'CASE-LOW-001')
    await expect(page.getByRole('heading', { name: '建议通过' })).toBeVisible({
      timeout: 20_000,
    })
    await expect(page.getByText('可核验证据')).toBeVisible()
    await expect(page.getByText('只读业务事实')).toBeVisible()
    await expect(page.getByText('主体 subject-local-demo')).toBeVisible()
    await expect(page.getByText('本地模拟回执')).toHaveCount(0)
    await fullPage(page, {
      path: `test-results/${testInfo.project.name}-result.png`,
      fullPage: true,
    })

    await expect(page.getByRole('button', { name: '确认建议（本地）', exact: true })).toBeEnabled()
    await page.getByRole('button', { name: '确认建议（本地）', exact: true }).click()
    await expect(page.getByRole('heading', { name: '本地模拟回执' })).toBeVisible({
      timeout: 20_000,
    })
    await expect(page.getByText('0 · 未执行')).toBeVisible()
    await expect(page.getByText('人工动作仅在本地记录，未写入采购系统。')).toBeVisible()
    await expect(
      page.getByText('Gate 已关闭；新动作会被拒绝，仅可幂等重放原动作。'),
    ).toBeVisible()

    // Replay is explicit: the same idempotency key and the same payload, and the
    // receipt/audit counters must not grow.
    await page.getByRole('button', { name: '重放原动作' }).click()
    await expect(
      page.getByText('人工动作已幂等重放：回执与审计记录均未增加。'),
    ).toBeVisible({ timeout: 20_000 })

    const runId = await page.locator('.run-title code').first().getAttribute('title')
    expect(runId).toBeTruthy()

    await page.getByRole('tab', { name: /Trace/ }).click()
    await expect(page.getByRole('heading', { name: '运行 Trace' })).toBeVisible()
    await expect(page.getByText('run_completed').first()).toBeVisible({
      timeout: 20_000,
    })

    await page.getByRole('tab', { name: /审批审计/ }).click()
    await expect(page.getByRole('heading', { name: '审批审计' })).toBeVisible()
    await expect(page.getByText('confirm', { exact: true }).first()).toBeVisible()
    await expect(page.getByText('幂等键摘要')).toBeVisible()

    // A reload drops the in-memory credential *and* the in-memory idempotency key.
    // The v2 receipt returns only a key digest, so the run is still readable but its
    // decision can no longer be replayed; the UI says so instead of faking a key.
    await page.reload()
    await expect(page.getByText('需要本地凭据')).toBeVisible()
    await signIn(page, token)
    await page.locator(`[data-run-id="${runId}"]`).click()
    await expect(page.getByRole('heading', { name: '本地模拟回执' })).toBeVisible({
      timeout: 20_000,
    })
    await expect(page.getByRole('button', { name: '重放原动作' })).toBeDisabled()
    await expect(
      page.getByText(
        '幂等重放需要本次会话保留的原始幂等键；服务端只返回幂等键摘要，刷新页面或重新验证凭据后无法重放。',
      ),
    ).toBeVisible()
    await fullPage(page, {
      path: `test-results/${testInfo.project.name}-receipt.png`,
      fullPage: true,
    })
  })

  test('replays a committed decision against the real API without duplicating any effect', async ({
    page,
  }) => {
    const token = credential('operator')
    await page.goto('/')
    await signIn(page, token)
    await startReview(page, 'CASE-LOW-001')
    await expect(page.getByRole('heading', { name: '建议通过', exact: true })).toBeVisible({
      timeout: 20_000,
    })

    await page.getByRole('button', { name: '确认建议（本地）', exact: true }).click()
    await expect(page.getByRole('heading', { name: '本地模拟回执' })).toBeVisible({
      timeout: 20_000,
    })
    const runId = await page.locator('.run-title code').first().getAttribute('title')
    expect(runId).toBeTruthy()
    const runPath = `/api/v2/reviews/${encodeURIComponent(runId!)}`

    const runsBefore = await apiGet<ApiRunList>(page, '/api/v2/reviews', token)
    const runBefore = await apiGet<ApiRun>(page, runPath, token)
    const auditBefore = await apiGet<ApiAudit>(page, `${runPath}/audit`, token)

    await expect(page.getByRole('button', { name: '重放原动作' })).toBeEnabled()
    await page.getByRole('button', { name: '重放原动作' }).click()
    await expect(
      page.getByText('人工动作已幂等重放：回执与审计记录均未增加。'),
    ).toBeVisible({ timeout: 20_000 })

    const runsAfter = await apiGet<ApiRunList>(page, '/api/v2/reviews', token)
    const runAfter = await apiGet<ApiRun>(page, runPath, token)
    const auditAfter = await apiGet<ApiAudit>(page, `${runPath}/audit`, token)

    // No second Run for the same Case: the replay reused the existing one.
    expect(runsAfter.runs.map((run) => run.run_id).sort()).toEqual(
      runsBefore.runs.map((run) => run.run_id).sort(),
    )
    // One receipt, one decision, one revision: the replay returned the existing result.
    expect(runAfter.receipt?.receipt_id).toBe(runBefore.receipt?.receipt_id)
    expect(runAfter.version).toBe(runBefore.version)
    expect(runAfter.revision).toBe(runBefore.revision)
    expect(runAfter.status).toBe(runBefore.status)
    // Neither the recovery budget nor the execution generation moved.
    expect(runAfter.recovery?.resume_count).toBe(runBefore.recovery?.resume_count)
    expect(runAfter.recovery?.generation).toBe(runBefore.recovery?.generation)
    // The audit ledger is byte-for-byte the same list: no human and no recovery row.
    expect(auditAfter.events.map((event) => event.audit_id)).toEqual(
      auditBefore.events.map((event) => event.audit_id),
    )
    expect(auditAfter.events.map((event) => event.sequence)).toEqual(
      auditBefore.events.map((event) => event.sequence),
    )
    // A settled Run may still be emitting Trace for the asynchronous execution, so the
    // Trace count is observed rather than frozen; it may never go backwards.
    console.log(
      `[V-04A-02] replay trace events before=${runBefore.trace_event_count} after=${runAfter.trace_event_count}`,
    )
    expect(runAfter.trace_event_count).toBeGreaterThanOrEqual(runBefore.trace_event_count)
  })

  test('replays a committed create against the real API with no second Run', async ({ page }) => {
    const token = credential('operator')
    await page.goto('/')
    await signIn(page, token)
    const runsBefore = await apiGet<ApiRunList>(page, '/api/v2/reviews', token)
    const beforeIds = runsBefore.runs.map((run) => run.run_id).sort()

    await startReview(page, 'CASE-LOW-001')
    await expect(page.getByRole('heading', { name: '建议通过', exact: true })).toBeVisible({
      timeout: 20_000,
    })
    const created = await apiGet<ApiRunList>(page, '/api/v2/reviews', token)
    expect(created.runs).toHaveLength(beforeIds.length + 1)

    await expect(page.getByRole('button', { name: '重放最近创建请求' })).toBeEnabled()
    await page.getByRole('button', { name: '重放最近创建请求' }).click()
    await expect(
      page.getByText('创建请求已幂等重放：复用同一 Run，未重复执行。'),
    ).toBeVisible({ timeout: 20_000 })

    const replayed = await apiGet<ApiRunList>(page, '/api/v2/reviews', token)
    expect(replayed.runs.map((run) => run.run_id).sort()).toEqual(
      created.runs.map((run) => run.run_id).sort(),
    )
  })

  test('keeps every long-content state inside a 393px viewport', async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 393, height: 851 })
    await page.goto('/')
    await signIn(page, credential('operator'))

    // 1. A result whose failure lists are long and unbroken.
    await startReview(page, 'CASE-ALL-FAILURE-001')
    await expect(page.getByRole('heading', { name: '信息不足' })).toBeVisible({
      timeout: 20_000,
    })
    await expectFitsViewport(page, 393)

    // 2. A recorded decision: the receipt carries a 64-hex digest and a receipt id.
    await startReview(page, 'CASE-LOW-001')
    await expect(page.getByRole('heading', { name: '建议通过' })).toBeVisible({
      timeout: 20_000,
    })
    await page.getByRole('button', { name: '确认建议（本地）', exact: true }).click()
    await expect(page.getByRole('heading', { name: '本地模拟回执' })).toBeVisible({
      timeout: 20_000,
    })
    await expectFitsViewport(page, 393)

    // 3. The audit trail and the trace, whose identifiers are the longest strings in
    //    the UI, plus a recovered Run's budget chip.
    await page.getByRole('tab', { name: /审批审计/ }).click()
    await expect(page.getByRole('heading', { name: '审批审计' })).toBeVisible()
    await expectFitsViewport(page, 393)
    await page.getByRole('tab', { name: /Trace/ }).click()
    await expect(page.getByRole('heading', { name: '运行 Trace' })).toBeVisible()
    await expectFitsViewport(page, 393)

    // 4. A closed runtime failure: the error text is the page's main content.
    await startReview(page, 'CASE-RUNTIME-FAILURE-001')
    await expect(page.getByRole('heading', { name: 'Review 运行失败并已关闭' })).toBeVisible({
      timeout: 20_000,
    })
    await expectFitsViewport(page, 393)
    await fullPage(page, {
      path: `test-results/${testInfo.project.name}-failure-393.png`,
      fullPage: true,
    })
  })

  test('hides every control the credential is not scoped for', async ({ page }) => {
    await page.goto('/')
    await signIn(page, credential('reader'))

    await expect(page.getByTestId('identity-subject')).toHaveText(
      'subject-local-demo-reader',
    )
    await expect(page.getByTestId('identity-scopes')).toHaveText(
      'case:read、review:read',
    )
    await expect(page.getByText(/服务端按已验证凭据与授权范围裁决/)).toBeVisible()

    await expect(page.getByRole('button', { name: '运行审查' })).toHaveCount(0)
    await expect(page.getByText('缺少 review:create 授权，运行审查已隐藏。')).toBeVisible()
    await expect(page.getByRole('tab', { name: /Trace/ })).toHaveCount(0)
    await expect(page.getByRole('tab', { name: /审批审计/ })).toHaveCount(0)
    const hidden = page.getByTestId('hidden-controls')
    await expect(hidden).toContainText('review:create')
    await expect(hidden).toContainText('review:decide')
    await expect(hidden).toContainText('review:cancel')
    await expect(hidden).toContainText('trace:read')
    await expect(hidden).toContainText('audit:read')

    // A narrower credential leaves a legal, single-tab view: the only tab is the
    // selected one and it is the one in the page's Tab order.
    const tabs = page.getByRole('tab')
    await expect(tabs).toHaveCount(1)
    await expect(tabs.first()).toHaveAttribute('aria-selected', 'true')
    await expect(tabs.first()).toHaveAttribute('tabindex', '0')
    await expect(page.locator('#review-tab-panel')).toHaveAttribute(
      'aria-labelledby',
      'review-tab-result',
    )
  })

  test('clears the workbench when a live credential stops being accepted', async ({
    page,
  }, testInfo) => {
    const label = `revoke-${testInfo.project.name}-${Date.now().toString(36)}`
    await page.goto('/')
    await signIn(page, issueOwnCredential(label))
    await expect(page.getByTestId('identity-subject')).toHaveText('subject-local-demo')

    // Give this session a run of its own so the workbench is definitely populated.
    await startReview(page, 'CASE-HIGH-001')
    await expect(page.getByRole('heading', { name: '建议拒绝' })).toBeVisible({
      timeout: 20_000,
    })
    const runsBefore = await page.locator('[data-run-id]').count()
    expect(runsBefore).toBeGreaterThan(0)

    // Stand on the widest view a credential can open, then lose that credential:
    // focus must not be left on a tab that no longer exists.
    await page.getByRole('tab', { name: /审批审计/ }).focus()

    revokeCredential(label)
    await page.getByRole('button', { name: '刷新当前 Review' }).click()

    await expect(page.getByText('凭据无效或已失效，请重新输入')).toBeVisible({
      timeout: 20_000,
    })
    await expect(page.getByTestId('identity-subject')).toHaveCount(0)
    await expect(page.locator('[data-run-id]')).toHaveCount(0)
    await expect(page.getByRole('tab')).toHaveCount(0)
    await expect(page.getByText('需要本地凭据')).toBeVisible()
    await expect(page.getByText('验证凭据后读取运行列表。')).toBeVisible()
    // The keyboard path is still usable: the credential field is reachable again.
    await page.getByLabel('Bearer 凭据').focus()
    await expect(page.getByLabel('Bearer 凭据')).toBeFocused()
  })

  test('rejects an unusable credential without leaving an identity behind', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByLabel('Bearer 凭据').fill('dwv-local-credential-that-was-never-issued')
    await page.getByRole('button', { name: '验证凭据' }).click()

    await expect(page.getByText('凭据无效或已失效，请重新输入')).toBeVisible({
      timeout: 20_000,
    })
    await expect(page.getByTestId('identity-subject')).toHaveCount(0)
    await expect(page.getByText('需要本地凭据')).toBeVisible()
    await expect(page.locator('[data-run-id]')).toHaveCount(0)
  })

  test('keeps worker failures visible and fail-closed', async ({ page }, testInfo) => {
    await page.goto('/')
    await signIn(page, credential('operator'))
    await startReview(page, 'CASE-ALL-FAILURE-001')

    await expect(page.getByRole('heading', { name: '信息不足' })).toBeVisible({
      timeout: 20_000,
    })
    await expect(page.getByRole('heading', { name: 'Worker 失败', exact: true })).toBeVisible()
    await expect(page.getByText('本地模拟回执')).toHaveCount(0)
    await fullPage(page, {
      path: `test-results/${testInfo.project.name}-failure.png`,
      fullPage: true,
    })
  })

  test('cancels a review without producing a receipt', async ({ page }) => {
    await page.goto('/')
    await signIn(page, credential('operator'))
    await startReview(page, 'CASE-CANCEL-001')

    await page.getByRole('button', { name: '取消', exact: true }).click()
    // `pending_confirmation → cancelled` is a legal transition and this fixture is
    // fast, so the Run can settle between the render the operator clicked and the
    // request reaching the server.  The workbench never retries a write on its own,
    // so a stale click is reported as a version conflict and the operator cancels
    // once more explicitly.  Both orders end in the same state; the assertions below
    // are the ones that matter — the cancellation is stated and no receipt is
    // invented for it (FIX-01C: cancelled-after-commit rendered as an ordinary result).
    const cancelled = page.getByText('Review 已取消；没有生成本地模拟回执。')
    const conflict = page.getByText('版本已过期，已重新读取最新审查状态。')
    await expect(cancelled.or(conflict).first()).toBeVisible({ timeout: 20_000 })
    if (await conflict.isVisible()) {
      await page.getByRole('button', { name: '取消', exact: true }).click()
    }
    await expect(cancelled).toBeVisible({ timeout: 20_000 })
    await expect(page.getByRole('main').getByText('已取消', { exact: true })).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Review 已取消' })).toBeVisible()
    await expect(page.getByTestId('cancelled-state')).toBeVisible()
    await expect(page.getByText('本地模拟回执', { exact: true })).toHaveCount(0)

    await page.getByRole('tab', { name: /审批审计/ }).click()
    await expect(page.getByText('cancel', { exact: true }).first()).toBeVisible()
  })

  test('interrupts a Run for real and resumes it from the committed checkpoints', async ({
    page,
  }, testInfo) => {
    // The disposable fault server needs a few seconds to notice the dead lease.
    test.setTimeout(90_000)
    const faultOrigin = FAULT_ORIGINS[testInfo.project.name]
    const faultRun = FAULT_RUNS[testInfo.project.name]
    expect(
      faultOrigin && faultRun,
      `no fault server is configured for project ${testInfo.project.name}`,
    ).toBeTruthy()

    // This origin proxies to its own API, started with a test-only fault plan that
    // breaks the policy checkpoint commit once.  The API process survives, so the
    // Run really does end up interrupted instead of the server dying.
    await page.goto(`${faultOrigin}/`)
    await signIn(page, credential('operator', faultRun))
    await startReview(page, 'CASE-LOW-001')

    await expect(page.getByTestId('interrupted-state')).toBeVisible({ timeout: 45_000 })
    await expect(page.locator('.run-header .status-badge')).toHaveText('已中断 · 等待恢复')
    await expect(page.getByTestId('recovery-budget')).toHaveText('恢复 0 / 2')
    await expect(page.getByTestId('recovery-reason')).not.toBeEmpty()
    // An interrupted Run has no summary and no receipt, so the human Gate must not
    // offer a decision for it.
    await expect(
      page.getByRole('button', { name: '确认建议（本地）', exact: true }),
    ).toHaveCount(0)
    await expect(
      page.getByRole('button', { name: '拒绝建议（本地）', exact: true }),
    ).toHaveCount(0)
    await expect(page.getByText('本地模拟回执')).toHaveCount(0)
    await expect(page.getByTestId('resume-button')).toBeEnabled()
    // The recovery view is the densest state the workbench has; it must fit the
    // project's own viewport too.
    const width = page.viewportSize()?.width
    expect(width).toBeTruthy()
    await expectFitsViewport(page, width as number)
    await fullPage(page, {
      path: `test-results/${testInfo.project.name}-interrupted.png`,
      fullPage: true,
    })

    // The injected outage must be over before an authorized operator resumes: the Run
    // becomes interrupted as soon as the broken commit is reported, which can still be
    // inside the termination window.  A resume that landed there would be a different
    // scenario than the one under test.
    await page.waitForTimeout(1500)

    // Explicit resume: the displayed version plus a fresh idempotency key.
    await page.getByTestId('resume-button').click()

    await expect(page.locator('.run-header .status-badge')).toHaveText('待人工确认', {
      timeout: 45_000,
    })
    await expect(page.getByRole('heading', { name: '建议通过' })).toBeVisible({
      timeout: 45_000,
    })
    // The resumed Run shows the budget it consumed.
    await expect(page.getByTestId('recovery-budget-chip')).toHaveText('恢复 1 / 2')
    await expect(page.getByTestId('interrupted-state')).toHaveCount(0)
    await expect(
      page.getByRole('button', { name: '确认建议（本地）', exact: true }),
    ).toBeEnabled()
    await fullPage(page, {
      path: `test-results/${testInfo.project.name}-resumed.png`,
      fullPage: true,
    })
  })

  test('renders without horizontal overflow on a mobile viewport', async ({
    page,
  }, testInfo) => {
    await page.goto('/')
    await expect(page.getByText('Review Desk')).toBeVisible()
    await signIn(page, credential('operator'))

    const viewport = page.viewportSize()
    expect(viewport?.width).toBeTruthy()
    const dimensions = await measureLayout(page, viewport?.width as number)
    // Referenced against the viewport that was set and the layout viewport, never
    // against a possibly-widened `innerWidth`.
    expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth)
    expect(dimensions.clientWidth).toBeLessThanOrEqual(viewport?.width as number)
    const height = await page.evaluate(() => document.documentElement.scrollHeight)
    expect(height).toBeGreaterThan(400)
    await fullPage(page, {
      path: `test-results/${testInfo.project.name}-identity.png`,
      fullPage: true,
    })
  })
})
