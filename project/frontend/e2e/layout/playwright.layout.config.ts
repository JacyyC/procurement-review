import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from '@playwright/test'

import base from '../../playwright.config'

/**
 * FIX-04B layout matrix.
 *
 * Its own config so the 26-scenario workbench suite keeps its own denominator.  It reuses
 * the same six web servers (the real API plus the two fault APIs, all resolved through
 * `e2e/origins.mjs`) and runs only the desktop project, because every case sets the
 * viewport it measures explicitly.
 */
const HERE = fileURLToPath(new URL('.', import.meta.url))
const FRONTEND_ROOT = path.resolve(HERE, '..', '..')

export default defineConfig({
  ...base,
  testDir: HERE,
  testMatch: '**/layout-matrix.spec.ts',
  // The base config excludes this directory so the workbench suite skips it; this run is
  // the one that owns it.
  testIgnore: [],
  projects: [base.projects![0]],
  reporter: [['list']],
  // Outside the shared `test-results` tree, which any Playwright run clears.
  outputDir: path.join(FRONTEND_ROOT, '.local', 'fix04-b-20260914', 'layout-results'),
  // Both of these are relative to the config file, so point them at the frontend root.
  globalTeardown: path.join(FRONTEND_ROOT, 'e2e', 'global-teardown.mjs'),
  // The base commands are relative to the frontend root, so every server is started there.
  webServer: (base.webServer as { command: string }[]).map((server) => ({
    ...server,
    cwd: FRONTEND_ROOT,
  })),
})
