import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from '@playwright/test'

const HERE = fileURLToPath(new URL('.', import.meta.url))
const FRONTEND_ROOT = path.resolve(HERE, '..', '..')

// Its own config, so the production probe never interferes with the 26-scenario workbench
// suite (which drives the dev server and the real API).  This one needs no API, no
// PostgreSQL and no credential: it only serves a static production build.
export default defineConfig({
  testDir: HERE,
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  // Never the shared `test-results` tree: this run must not clear another suite's evidence.
  outputDir: path.join(FRONTEND_ROOT, '.local', 'fix04-b-20260914', 'probe-results'),
  use: {
    baseURL: 'http://127.0.0.1:4188',
    trace: 'off',
    screenshot: 'off',
    video: 'off',
  },
  webServer: {
    // Run from the frontend root so the config path is the same one the npm script uses.
    cwd: FRONTEND_ROOT,
    command: 'npx vite preview --config e2e/probe/vite.config.ts',
    url: 'http://127.0.0.1:4188/',
    reuseExistingServer: false,
    timeout: 120_000,
  },
  projects: [{ name: 'chromium-probe', use: { browserName: 'chromium' } }],
})
