#!/usr/bin/env node
/**
 * One source of truth for every origin the end-to-end run owns.
 *
 * Four processes have to agree on the same numbers: the Playwright config (which
 * starts the servers and points the browser at them), the API bootstrap
 * (`review-db.mjs`, which allow-lists the browser origin for CORS and binds the API
 * port), the Vite dev servers (which proxy `/api` to one specific API), and the spec
 * itself (which navigates to the fault origins).  When each of them read the
 * environment separately, a non-default port configuration silently produced a
 * browser talking to the default API.
 *
 * Everything is derived here, validated once, and imported by all four.  Ports must
 * be integers in range and pairwise distinct: two services on one port is a
 * configuration error worth failing loudly for, not a flaky test.
 */

const LOOPBACK = '127.0.0.1'

/**
 * Ports for the whole run.  Every value has an environment override so the suite can
 * be re-run on a fully non-default configuration (V-C-05) without editing a file.
 */
export const PORT_VARIABLES = {
  api: 'REVIEW_E2E_API_PORT',
  web: 'REVIEW_E2E_WEB_PORT',
  faultDesktopApi: 'REVIEW_E2E_FAULT_DESKTOP_API_PORT',
  faultDesktopWeb: 'REVIEW_E2E_FAULT_DESKTOP_WEB_PORT',
  faultMobileApi: 'REVIEW_E2E_FAULT_MOBILE_API_PORT',
  faultMobileWeb: 'REVIEW_E2E_FAULT_MOBILE_WEB_PORT',
}

export const DEFAULT_PORTS = {
  api: 8000,
  web: 5273,
  faultDesktopApi: 8001,
  faultDesktopWeb: 5274,
  faultMobileApi: 8002,
  faultMobileWeb: 5275,
}

function resolvePort(env, key, fallback) {
  const variable = PORT_VARIABLES[key]
  const raw = env[variable]
  if (raw === undefined || raw === '') return fallback
  if (!/^\d+$/.test(String(raw))) {
    throw new Error(`${variable} must be a port number, got "${raw}"`)
  }
  const port = Number(raw)
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`${variable} must be an integer in 1..65535, got "${raw}"`)
  }
  return port
}

export function originFor(port) {
  return `http://${LOOPBACK}:${port}`
}

/** Named run kinds, so a failure message can say which two services collided. */
export const RUN_KIND = {
  main: 'main',
  faultDesktop: 'fault-desktop',
  faultMobile: 'fault-mobile',
}

/**
 * Resolve and validate the whole port map.
 *
 * @param {NodeJS.ProcessEnv} [env]
 */
export function resolveE2EOrigins(env = process.env) {
  const ports = {}
  for (const key of Object.keys(DEFAULT_PORTS)) {
    ports[key] = resolvePort(env, key, DEFAULT_PORTS[key])
  }
  const seen = new Map()
  for (const [key, port] of Object.entries(ports)) {
    const previous = seen.get(port)
    if (previous) {
      throw new Error(
        `port ${port} is assigned to both ${previous} and ${key}; ` +
          `set a distinct ${PORT_VARIABLES[key]} (or ${PORT_VARIABLES[previous]})`,
      )
    }
    seen.set(port, key)
  }
  return {
    ports,
    main: {
      kind: RUN_KIND.main,
      apiPort: ports.api,
      webPort: ports.web,
      apiOrigin: originFor(ports.api),
      webOrigin: originFor(ports.web),
      fault: false,
    },
    faults: [
      {
        variant: 'desktop',
        kind: RUN_KIND.faultDesktop,
        apiPort: ports.faultDesktopApi,
        webPort: ports.faultDesktopWeb,
        apiOrigin: originFor(ports.faultDesktopApi),
        webOrigin: originFor(ports.faultDesktopWeb),
        fault: true,
      },
      {
        variant: 'mobile',
        kind: RUN_KIND.faultMobile,
        apiPort: ports.faultMobileApi,
        webPort: ports.faultMobileWeb,
        apiOrigin: originFor(ports.faultMobileApi),
        webOrigin: originFor(ports.faultMobileWeb),
        fault: true,
      },
    ],
  }
}

/** Playwright project name -> the fault web origin that project must drive. */
export function faultOriginsByProject(origins = resolveE2EOrigins()) {
  return Object.fromEntries(
    origins.faults.map((fault) => [`chromium-${fault.variant}`, fault.webOrigin]),
  )
}

/** Playwright project name -> the fault run kind `review-db.mjs` bootstraps. */
export function faultKindsByProject(origins = resolveE2EOrigins()) {
  return Object.fromEntries(
    origins.faults.map((fault) => [`chromium-${fault.variant}`, fault.kind]),
  )
}

/** The browser origin the API must allow-list for one run kind. */
export function allowedWebOrigins(kind, origins = resolveE2EOrigins()) {
  const run = kind === RUN_KIND.main ? origins.main : origins.faults.find((f) => f.kind === kind)
  if (!run) throw new Error(`unknown run kind "${kind}"`)
  return [run.webOrigin, `http://localhost:${run.webPort}`]
}
