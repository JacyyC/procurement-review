import { defineConfig, devices } from '@playwright/test'

import { resolveE2EOrigins } from './e2e/origins.mjs'

// Every port, origin and proxy target comes from one validated resolver shared with
// `e2e/review-db.mjs` and the spec, so a non-default configuration cannot drift into
// "browser on one port, API on another" (FIX-01C, V-C-05).
const origins = resolveE2EOrigins()
const { main, faults } = origins
const WEB_ORIGIN = main.webOrigin
const API_ORIGIN = main.apiOrigin

export default defineConfig({
  testDir: './e2e',
  // The layout matrix and the production probe are separate runs with their own configs
  // (`npm run e2e:layout`, `npm run e2e:probe`).  They must not be collected here: the
  // layout matrix consumes the fault APIs' one-shot interruption barrier, and the probe
  // expects a static production build rather than this suite's servers.
  testIgnore: ['**/layout/**', '**/probe/**', '**/visual/**'],
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  // The test-only fault hook terminates *every* database connection that carries its
  // runner application name, and that name is a backend constant shared by both fault
  // APIs.  Two fault scenarios running at once would therefore interrupt each other's
  // resumed generation and leave one Run stuck interrupted.  One worker keeps the
  // suite deterministic; the whole run is local and short.
  workers: 1,
  // Every assertion in this suite that waits for a Run to settle already asks for 20s; the
  // rest were left on Playwright's 5s default, which is shorter than the time a local Run
  // takes when the machine is busy.  One budget for all of them keeps a slow host from
  // looking like a product failure without weakening any assertion (2026-09-14: three such
  // flakes were observed on the mobile project under an external load average of ~30).
  expect: { timeout: 20_000 },
  // These scenarios walk a whole review lifecycle (credential → create → decide → replay →
  // trace → audit → reload), so 30s per test is tight once the host is shared with other
  // work.  The budget is per test, not per assertion: a genuine hang still fails.
  timeout: 60_000,
  reporter: [['html', { open: 'never' }], ['list']],
  // The suite types a real local Bearer credential into the page.  Playwright's
  // trace records action arguments and its automatic failure screenshot can catch
  // the still-filled password input, so both stay off for this suite; the spec
  // takes explicit screenshots only after the credential field has been cleared.
  use: {
    baseURL: WEB_ORIGIN,
    trace: 'off',
    screenshot: 'off',
    video: 'off',
  },
  globalTeardown: './e2e/global-teardown.mjs',
  webServer: [
    {
      // Builds a disposable PostgreSQL database, migrates and seeds it, issues the
      // local credentials and then runs the authenticated API with
      // REVIEW_DATABASE_URL pointing at that database.
      command: 'node e2e/review-db.mjs serve',
      url: `${API_ORIGIN}/api/v2/health`,
      reuseExistingServer: false,
      gracefulShutdown: { signal: 'SIGTERM', timeout: 15_000 },
      timeout: 180_000,
    },
    {
      command: `npm run dev -- --port ${main.webPort} --strictPort`,
      url: WEB_ORIGIN,
      reuseExistingServer: false,
      timeout: 120_000,
      // The dev server proxies `/api` to this exact origin, so the main suite follows
      // a non-default API port instead of silently keeping the default.
      env: { REVIEW_E2E_API_ORIGIN: API_ORIGIN },
    },
    // The fault APIs interrupt a Run mid-checkpoint instead of killing the process,
    // so a single API process can serve exactly one interruption.  Two of them keep
    // the desktop and mobile projects from consuming each other's barrier.
    ...faults.map((fault) => ({
      command: `node e2e/review-db.mjs serve-fault ${fault.variant}`,
      url: `${fault.apiOrigin}/api/v2/health`,
      reuseExistingServer: false,
      gracefulShutdown: { signal: 'SIGTERM', timeout: 15_000 } as const,
      timeout: 180_000,
    })),
    ...faults.map((fault) => ({
      command: `npm run dev -- --port ${fault.webPort} --strictPort`,
      url: fault.webOrigin,
      reuseExistingServer: false,
      timeout: 120_000,
      env: { REVIEW_E2E_API_ORIGIN: fault.apiOrigin },
    })),
  ],
  projects: [
    { name: 'chromium-desktop', use: { ...devices['Desktop Chrome'] } },
    { name: 'chromium-mobile', use: { ...devices['Pixel 5'] } },
  ],
})
