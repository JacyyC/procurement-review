/**
 * FIX-04B / C-AUD-07 production probe entry.
 *
 * This is a *test asset*, built by `e2e/probe/vite.config.ts` into its own output directory so
 * the product bundle never contains it.  It imports the same two modules the product entry
 * uses — `ROOT_ERROR_OPTIONS` from `src/errorReporting.ts` and `ReviewErrorBoundary` — and
 * triggers the three render-error paths React 19 reports:
 *
 *  1. a render error a boundary catches (`onCaughtError` + `componentDidCatch`);
 *  2. a render error no boundary catches (`onUncaughtError`, root unmounted);
 *  3. a hydration mismatch React recovers from on its own (`onRecoverableError`).
 *
 * jsdom cannot prove this: React ships a different production implementation, and the leak
 * the audit reproduced came from that build.  The assertion itself lives in `probe.spec.ts`,
 * which also records every console argument.
 */

import { createRoot, hydrateRoot } from 'react-dom/client'

import { ROOT_ERROR_OPTIONS } from '../../src/errorReporting'
import { ReviewErrorBoundary } from '../../src/ReviewErrorBoundary'

const CANARY = 'FIX04-PROBE-CANARY'

/** A thrown value whose name, message, stack and cause all carry the canary. */
function canaryError(): Error {
  const error = new Error(`${CANARY}-MESSAGE`)
  error.name = `${CANARY}-NAME`
  error.stack = `${CANARY}-NAME: ${CANARY}-MESSAGE\n    at ${CANARY}-FRAME`
  ;(error as { cause?: unknown }).cause = { detail: `${CANARY}-CAUSE` }
  return error
}

function BrokenCaught(): never {
  throw canaryError()
}

function BrokenUncaught(): never {
  throw canaryError()
}

const caught = document.getElementById('caught')
if (caught) {
  createRoot(caught, ROOT_ERROR_OPTIONS).render(
    <ReviewErrorBoundary scope="审查视图" onReset={() => {}}>
      <BrokenCaught />
    </ReviewErrorBoundary>,
  )
}

const uncaught = document.getElementById('uncaught')
if (uncaught) {
  createRoot(uncaught, ROOT_ERROR_OPTIONS).render(<BrokenUncaught />)
}

const recoverable = document.getElementById('recoverable')
if (recoverable) {
  // Server markup that does not match what the client renders: React recovers by
  // re-rendering on the client and reports it through `onRecoverableError`.
  recoverable.innerHTML = '<p>server-rendered text</p>'
  hydrateRoot(recoverable, <p>client-rendered text</p>, ROOT_ERROR_OPTIONS)
}

window.setTimeout(() => {
  document.body.dataset.probeReady = 'true'
}, 50)
