import { defineConfig } from '@playwright/test'
import base from './playwright.visual.config'

export default defineConfig({
  ...base,
  testMatch: '**/state-matrix.spec.ts',
  outputDir: '../../.local/ui-02/state-results',
})
