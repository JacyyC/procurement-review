import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const HERE = fileURLToPath(new URL('.', import.meta.url))

// Production-build probe for FIX-04B / C-AUD-07.
//
// It is a *second* build, with its own output directory, so the product bundle stays
// exactly what `npm run build` produces: no probe, no canary, no fault switch.  The build
// mode is `production`, which is the React build whose default `onCaughtError` the audit
// reproduced the leak with.
export default defineConfig({
  root: HERE,
  base: './',
  plugins: [react()],
  build: {
    mode: 'production',
    outDir: 'dist',
    emptyOutDir: true,
  },
  preview: {
    port: 4188,
    host: '127.0.0.1',
    strictPort: true,
  },
})
