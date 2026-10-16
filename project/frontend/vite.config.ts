import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The e2e "fault" dev servers proxy to their own API instance so a scenario can
// interrupt a Run for real without disturbing the main suite; everything else keeps
// the usual local API on 127.0.0.1:8000.  `REVIEW_E2E_API_ORIGIN` is set by
// `playwright.config.ts` for *every* e2e dev server (main and fault) from the shared
// resolver in `e2e/origins.mjs`, so a non-default API port is followed instead of
// silently falling back to the default.  `REVIEW_API_ORIGIN` is the delivery override
// used by the Compose `web` service, where the API is reachable as `http://api:8000`.
const apiTarget =
  process.env.REVIEW_E2E_API_ORIGIN ?? process.env.REVIEW_API_ORIGIN ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: false,
      },
    },
  },
  preview: {
    port: 4173,
    host: '127.0.0.1',
    proxy: {
      '/api': {
        // `changeOrigin: false` keeps the caller's loopback Host header, which the API
        // requires; the target is the orchestrator-internal `api` service.
        target: apiTarget,
        changeOrigin: false,
      },
    },
  },
})
