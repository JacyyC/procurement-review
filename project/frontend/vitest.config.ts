import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['e2e/**'],
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
    css: true,
    globals: true,
  },
})
