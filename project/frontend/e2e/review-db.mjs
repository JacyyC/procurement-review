#!/usr/bin/env node
/**
 * Disposable PostgreSQL + credential bootstrap for the review workbench e2e run.
 *
 * This helper owns the whole local security store lifecycle for one Playwright run:
 *
 *   node e2e/review-db.mjs up              create a uniquely named database, migrate
 *                                          it, seed the synthetic fixture, issue two
 *                                          local credentials into 0600 files
 *   node e2e/review-db.mjs serve           `up` (or reuse the current run) and then
 *                                          start the authenticated API in the
 *                                          foreground with REVIEW_DATABASE_URL
 *                                          pointing at it
 *   node e2e/review-db.mjs serve-fault K   the same for a "fault" server: its own
 *                                          database, its own port, its own temporary
 *                                          fault directory and the review timing
 *                                          overrides a recovery scenario needs
 *   node e2e/review-db.mjs down            drop every database and delete the
 *                                          credential files
 *   node e2e/review-db.mjs token ID [RUN]  print one credential value for a test to type
 *   node e2e/review-db.mjs status [RUN]    print the safe run metadata (never a token)
 *
 * Run kinds: `main` (the whole workbench suite), `fault-desktop` and `fault-mobile`.
 * The two fault kinds exist so each Playwright project can interrupt *its own* Run:
 * the fault plan fires once per API process, so sharing one fault server between the
 * desktop and mobile projects would let whichever project ran first consume the
 * barrier and leave the other one waiting for an interruption that never happens.
 *
 * PostgreSQL 17 must already be listening on 127.0.0.1:55433 (role `v1_03_admin`,
 * trust auth).  If it is not, this helper reports that clearly and starts nothing
 * else rather than faking a database.  Credential values only ever live in protected
 * files under a private state directory; they are never printed to stdout, never
 * written into a git-tracked file and never logged by the Playwright run.
 */

import { spawn, spawnSync } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

import { allowedWebOrigins, resolveE2EOrigins, RUN_KIND } from './origins.mjs'

const HERE = path.dirname(fileURLToPath(import.meta.url))
/** Ports and browser origins are resolved once, by the shared config module. */
const ORIGINS = resolveE2EOrigins()
const FRONTEND_ROOT = path.resolve(HERE, '..')
const BACKEND_ROOT = path.resolve(FRONTEND_ROOT, '..', 'backend')

export const DEFAULT_STATE_DIR = path.join(os.tmpdir(), 'review-workbench-e2e')
export const STATE_DIR = process.env.REVIEW_E2E_STATE_DIR
  ? path.resolve(process.env.REVIEW_E2E_STATE_DIR)
  : DEFAULT_STATE_DIR
export const STATE_FILE = path.join(STATE_DIR, 'state.json')

const PG_HOST = process.env.REVIEW_E2E_PG_HOST ?? '127.0.0.1'
const PG_PORT = process.env.REVIEW_E2E_PG_PORT ?? '55433'
const PG_USER = process.env.REVIEW_E2E_PG_USER ?? 'v1_03_admin'
const PG_MAINTENANCE_DB = process.env.REVIEW_E2E_PG_MAINTENANCE_DB ?? 'postgres'
const API_HOST = '127.0.0.1'
const OPERATOR_SUBJECT = 'subject-local-demo'
const READER_SUBJECT = 'subject-local-demo-reader'
const CREDENTIAL_TTL_SECONDS = '3600'
const UV = process.env.REVIEW_E2E_UV ?? 'uv'

/**
 * Everything that makes one API + database + browser origin independent from the
 * others.  The ports mirror the webServer entries in `playwright.config.ts`.
 */
const RUN_KINDS = {
  [RUN_KIND.main]: {
    name: RUN_KIND.main,
    stateFile: STATE_FILE,
    databasePrefix: 'review_e2e_',
    apiPort: String(ORIGINS.main.apiPort),
    webPort: String(ORIGINS.main.webPort),
    fault: false,
  },
  [RUN_KIND.faultDesktop]: {
    name: RUN_KIND.faultDesktop,
    stateFile: path.join(STATE_DIR, 'fault-desktop.json'),
    databasePrefix: 'review_e2e_fault_desktop_',
    apiPort: String(ORIGINS.faults[0].apiPort),
    webPort: String(ORIGINS.faults[0].webPort),
    fault: true,
  },
  [RUN_KIND.faultMobile]: {
    name: RUN_KIND.faultMobile,
    stateFile: path.join(STATE_DIR, 'fault-mobile.json'),
    databasePrefix: 'review_e2e_fault_mobile_',
    apiPort: String(ORIGINS.faults[1].apiPort),
    webPort: String(ORIGINS.faults[1].webPort),
    fault: true,
  },
}
const DEFAULT_KIND = RUN_KIND.main
const FAULT_VARIANTS = { desktop: 'fault-desktop', mobile: 'fault-mobile' }

/**
 * Test-only fault plan.  `terminate` breaks the database connection the policy
 * branch is about to commit on (for a short window), so the Run stops mid-checkpoint
 * without killing the API process; the committed branches of the other workers stay
 * adoptable and the interrupted Run can be resumed.  `once` keeps the barrier armed
 * for exactly one generation, so the resumed generation commits normally.
 */
const FAULT_PLAN = {
  fault_plan_version: 'fault-plan.v1',
  barriers: {
    'during_checkpoint_commit:policy': {
      action: 'terminate',
      // Short window: the termination has to break the checkpoint commit without
      // outliving the resume that the operator triggers right afterwards.
      timeout_seconds: 0.5,
      once: true,
    },
  },
}
const FAULT_APPLICATION_NAME = 'v1_03_security_recovery'
/**
 * Each fault server tags its own connections, so one project's injected outage cannot
 * terminate the other project's sessions in the same PostgreSQL instance.
 */
function faultApplicationName(kind) {
  return `${FAULT_APPLICATION_NAME}_${kind.name}`
}
/** Fast interruption: a 3s lease and a 0.2s scanner make a stuck Run visible quickly. */
const FAULT_TIMING = {
  REVIEW_LEASE_SECONDS: '3',
  REVIEW_HEARTBEAT_SECONDS: '0.5',
  REVIEW_SCAN_INTERVAL_SECONDS: '0.2',
}

function runKind(name) {
  const kind = RUN_KINDS[name ?? DEFAULT_KIND]
  if (!kind) {
    throw new Error(
      `unknown run kind "${name}"; expected one of ${Object.keys(RUN_KINDS).join(', ')}`,
    )
  }
  return kind
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { encoding: 'utf8', ...options })
  if (result.error) {
    throw new Error(`${command} could not be started: ${result.error.message}`)
  }
  if (result.status !== 0) {
    const detail = (result.stderr ?? '').trim().split('\n').slice(-4).join('\n')
    throw new Error(
      `${command} ${args.join(' ')} exited with ${result.status}${detail ? `\n${detail}` : ''}`,
    )
  }
  return result.stdout ?? ''
}

function psql(sql, database = PG_MAINTENANCE_DB) {
  return run('psql', [
    '-X',
    '-q',
    '-v',
    'ON_ERROR_STOP=1',
    '-h',
    PG_HOST,
    '-p',
    PG_PORT,
    '-U',
    PG_USER,
    '-d',
    database,
    '-tAc',
    sql,
  ])
}

/** Fail loudly when the local PostgreSQL 17 instance is not there. */
function assertPostgres() {
  const probe = spawnSync(
    'pg_isready',
    ['-h', PG_HOST, '-p', PG_PORT, '-U', PG_USER],
    { encoding: 'utf8' },
  )
  if (probe.error || probe.status !== 0) {
    throw new Error(
      `PostgreSQL 17 is not accepting connections on ${PG_HOST}:${PG_PORT} as ${PG_USER}.\n` +
        'Start that instance (trust auth, no password) and re-run `npm run e2e`. ' +
        'This helper starts no other database and never fakes one.',
    )
  }
}

function admin(args, dsn) {
  return run(
    UV,
    [
      'run',
      '--offline',
      '--frozen',
      '--python',
      '3.12',
      'python',
      '-m',
      'procurement_review.review_admin',
      ...args,
    ],
    {
      cwd: BACKEND_ROOT,
      env: {
        ...process.env,
        REVIEW_ADMIN_DATABASE_URL: dsn,
        REVIEW_DATABASE_URL: dsn,
      },
    },
  )
}

function ensureStateDir() {
  fs.mkdirSync(STATE_DIR, { recursive: true, mode: 0o700 })
  fs.chmodSync(STATE_DIR, 0o700)
  const mode = fs.statSync(STATE_DIR).mode & 0o777
  if (mode !== 0o700) {
    throw new Error(
      `refusing to store credentials in ${STATE_DIR}: mode is 0${mode.toString(8)}, expected 0700`,
    )
  }
}

export function readState(name) {
  const kind = runKind(name)
  if (!fs.existsSync(kind.stateFile)) return null
  return JSON.parse(fs.readFileSync(kind.stateFile, 'utf8'))
}

function writeState(state, name) {
  const kind = runKind(name)
  ensureStateDir()
  fs.writeFileSync(kind.stateFile, `${JSON.stringify(state, null, 2)}\n`, { mode: 0o600 })
  fs.chmodSync(kind.stateFile, 0o600)
}

function issueCredential(dsn, subject, label) {
  // `credentials issue` refuses to overwrite and prints only safe metadata.
  const tokenFile = path.join(STATE_DIR, `credential-${label}.token`)
  const payload = JSON.parse(
    admin(
      [
        'credentials',
        'issue',
        '--subject',
        subject,
        '--ttl-seconds',
        CREDENTIAL_TTL_SECONDS,
        '--output',
        tokenFile,
      ],
      dsn,
    ),
  )
  if (payload.token_printed !== false) {
    throw new Error('the admin CLI reported an unexpected credential output mode')
  }
  return {
    label,
    subject_id: payload.subject_id,
    tenant_id: payload.tenant_id,
    credential_id: payload.credential_id,
    token_file: tokenFile,
  }
}

function dropDatabase(database) {
  if (!database) return
  psql(`DROP DATABASE IF EXISTS "${database}" WITH (FORCE)`)
}

/** Terminate a still-running API process recorded by `serve`, guarded by its command line. */
function killRecordedApi(state) {
  const pid = Number(state?.api_pid)
  if (!Number.isInteger(pid) || pid <= 0) return false
  const probe = spawnSync('ps', ['-p', String(pid), '-o', 'command='], {
    encoding: 'utf8',
  })
  const command = (probe.stdout ?? '').trim()
  if (probe.status !== 0 || !command.includes('procurement_review.api')) return false
  try {
    process.kill(pid, 'SIGTERM')
  } catch {
    return true
  }
  const deadline = Date.now() + 5_000
  while (Date.now() < deadline) {
    const alive = spawnSync('ps', ['-p', String(pid), '-o', 'command='], {
      encoding: 'utf8',
    })
    if (alive.status !== 0) return true
    spawnSync('sleep', ['0.2'])
  }
  try {
    process.kill(pid, 'SIGKILL')
  } catch {
    // Already gone.
  }
  return true
}

function removeCredentialArtifacts(state, name) {
  const kind = runKind(name)
  const files = [
    ...(state?.credentials
      ? Object.values(state.credentials).map((item) => item.token_file)
      : []),
    kind.stateFile,
  ]
  for (const file of files) {
    if (file && fs.existsSync(file)) fs.rmSync(file, { force: true })
  }
  try {
    if (fs.existsSync(STATE_DIR) && fs.readdirSync(STATE_DIR).length === 0) {
      fs.rmdirSync(STATE_DIR)
    }
  } catch {
    // A leftover empty directory is harmless.
  }
}

/** Directory holding one fault run's plan and its `*.reached` markers. */
function faultDir(kind) {
  return path.join(STATE_DIR, `fault-${kind.name}`)
}

function writeFaultPlan(kind) {
  const directory = faultDir(kind)
  fs.rmSync(directory, { recursive: true, force: true })
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 })
  const planPath = path.join(directory, 'plan.json')
  fs.writeFileSync(planPath, `${JSON.stringify(FAULT_PLAN, null, 2)}\n`, { mode: 0o600 })
  return { directory, planPath }
}

/** The fault hooks only terminate connections that carry the runner's name. */
function withApplicationName(dsn, name) {
  const separator = dsn.includes('?') ? '&' : '?'
  return `${dsn}${separator}application_name=${name}`
}

export function up(name) {
  const kind = runKind(name)
  assertPostgres()
  down(kind.name)
  ensureStateDir()
  const suffix = `${Date.now().toString(36)}_${randomBytes(3).toString('hex')}`
  const database = `${kind.databasePrefix}${suffix}`
  const dsn = `postgresql+psycopg://${PG_USER}@${PG_HOST}:${PG_PORT}/${database}`
  psql(`CREATE DATABASE "${database}"`)
  let state
  try {
    admin(['migrate'], dsn)
    admin(['seed'], dsn)
    state = {
      run_id: suffix,
      run_kind: kind.name,
      created_at: new Date().toISOString(),
      database,
      dsn,
      pg_host: PG_HOST,
      pg_port: PG_PORT,
      pg_user: PG_USER,
      api_url: `http://${API_HOST}:${kind.apiPort}`,
      web_origin: `http://127.0.0.1:${kind.webPort}`,
      credentials: {
        operator: issueCredential(dsn, OPERATOR_SUBJECT, `${kind.name}-operator`),
        reader: issueCredential(dsn, READER_SUBJECT, `${kind.name}-reader`),
      },
    }
    writeState(state, kind.name)
  } catch (error) {
    dropDatabase(database)
    throw error
  }
  return state
}

/** Drop one run kind, or every kind when no name is given. */
export function down(name) {
  const kinds = name ? [runKind(name)] : Object.values(RUN_KINDS)
  let dropped = false
  for (const kind of kinds) {
    const state = readState(kind.name)
    if (state) dropped = true
    killRecordedApi(state ?? {})
    if (state) {
      try {
        dropDatabase(state.database)
      } finally {
        removeCredentialArtifacts(state, kind.name)
      }
    }
    fs.rmSync(faultDir(kind), { recursive: true, force: true })
  }
  return dropped
}

function serve(name) {
  const kind = runKind(name)
  const state =
    kind.fault || !fs.existsSync(kind.stateFile) ? up(kind.name) : readState(kind.name)
  const env = {
    ...process.env,
    REVIEW_DATABASE_URL: kind.fault
      ? withApplicationName(state.dsn, faultApplicationName(kind))
      : state.dsn,
    REVIEW_ADMIN_DATABASE_URL: state.dsn,
    // The workbench origin is served from a private port so the e2e run never
    // collides with anything else already listening on 5173.  The list comes from
    // the same resolver the browser origins do.
    REVIEW_ALLOWED_ORIGINS: allowedWebOrigins(kind.name, ORIGINS).join(','),
  }
  if (kind.fault) {
    const { directory, planPath } = writeFaultPlan(kind)
    env.V1_03_FAULT_PLAN = planPath
    env.V1_03_FAULT_DIR = directory
    env.V1_03_FAULT_ADMIN_DSN = state.dsn
    env.V1_03_FAULT_APPLICATION_NAME = faultApplicationName(kind)
    Object.assign(env, FAULT_TIMING)
  } else {
    delete env.V1_03_FAULT_PLAN
    delete env.V1_03_FAULT_DIR
    delete env.V1_03_FAULT_ADMIN_DSN
  }
  console.log(
    `[review-db] ${kind.name}: database ${state.database} ready; starting the API on ${state.api_url}`,
  )
  const child = spawn(
    UV,
    [
      'run',
      '--offline',
      '--frozen',
      '--python',
      '3.12',
      'uvicorn',
      'procurement_review.api:app',
      '--host',
      API_HOST,
      '--port',
      kind.apiPort,
    ],
    { cwd: BACKEND_ROOT, env, stdio: 'inherit' },
  )
  writeState({ ...state, api_pid: child.pid }, kind.name)

  let stopping = false
  const stop = (signal) => {
    if (stopping) return
    stopping = true
    try {
      child.kill(signal)
    } catch {
      // Already gone.
    }
  }
  process.on('SIGINT', () => stop('SIGINT'))
  process.on('SIGTERM', () => stop('SIGTERM'))

  child.on('exit', (code, signal) => {
    try {
      down(kind.name)
      console.log(`[review-db] ${kind.name}: database ${state.database} dropped`)
    } catch (error) {
      console.error(`[review-db] ${kind.name}: teardown failed: ${error.message}`)
    }
    process.exit(code ?? (signal ? 1 : 0))
  })
}

function token(label, name) {
  const state = readState(name)
  const credential = state?.credentials?.[label]
  if (!credential) {
    throw new Error(
      `no issued credential named "${label}" for run "${name ?? DEFAULT_KIND}"; ` +
        'run `review-db.mjs up` first',
    )
  }
  return fs.readFileSync(credential.token_file, 'utf8').trim()
}

/** Revoke one issued credential so a test can exercise a live 401. */
function revoke(label, name) {
  const state = readState(name)
  const credential = state?.credentials?.[label]
  if (!credential) {
    throw new Error(`no issued credential named "${label}"`)
  }
  admin(
    ['credentials', 'revoke', '--credential-id', credential.credential_id],
    state.dsn,
  )
  return credential.credential_id
}

/**
 * Issue one extra credential for the operator subject under a caller-chosen label.
 * Tests that revoke a credential use this so they can never disturb a shared one.
 */
function issue(label, name) {
  if (!label) throw new Error('an issue label is required')
  const kind = runKind(name)
  const state = readState(kind.name)
  if (!state) throw new Error('no active run; start `review-db.mjs serve` first')
  const credential = issueCredential(state.dsn, OPERATOR_SUBJECT, label)
  writeState(
    {
      ...state,
      credentials: { ...state.credentials, [label]: credential },
    },
    kind.name,
  )
  return credential
}

function main(argv) {
  const [command, argument, runArgument] = argv
  switch (command) {
    case 'up': {
      const state = up(runArgument)
      console.log(
        `[review-db] database ${state.database} ready (state: ${runKind(runArgument).stateFile})`,
      )
      return 0
    }
    case 'down': {
      const dropped = down(runArgument)
      console.log(dropped ? '[review-db] database dropped' : '[review-db] nothing to drop')
      return 0
    }
    case 'serve':
      serve(runArgument)
      return 0
    case 'serve-fault': {
      const kindName = FAULT_VARIANTS[argument]
      if (!kindName) {
        console.error('usage: review-db.mjs serve-fault <desktop|mobile>')
        return 2
      }
      serve(kindName)
      return 0
    }
    case 'token':
      // Never printed with a newline-free trailer that a log could capture twice.
      process.stdout.write(token(argument ?? 'operator', runArgument))
      return 0
    case 'revoke': {
      const credentialId = revoke(argument ?? 'revocable', runArgument)
      console.log(`[review-db] revoked ${credentialId}`)
      return 0
    }
    case 'issue': {
      const credential = issue(argument, runArgument)
      // Status goes to stderr so stdout carries exactly one credential value.
      console.error(`[review-db] issued ${credential.credential_id} as "${argument}"`)
      process.stdout.write(token(argument, runArgument))
      return 0
    }
    case 'status': {
      const kind = runKind(runArgument)
      const state = readState(kind.name)
      if (!state) {
        console.log(`[review-db] no active run for ${kind.name}`)
        return 1
      }
      console.log(
        JSON.stringify(
          {
            run_id: state.run_id,
            run_kind: kind.name,
            database: state.database,
            api_url: state.api_url,
            web_origin: state.web_origin,
            api_pid: state.api_pid ?? null,
            fault_plan: kind.fault ? path.join(faultDir(kind), 'plan.json') : null,
            credentials: Object.fromEntries(
              Object.entries(state.credentials).map(([key, value]) => [
                key,
                {
                  subject_id: value.subject_id,
                  tenant_id: value.tenant_id,
                  credential_id: value.credential_id,
                  token_file: value.token_file,
                  token_printed: false,
                },
              ]),
            ),
          },
          null,
          2,
        ),
      )
      return 0
    }
    default:
      console.error(
        'usage: review-db.mjs <up|down|serve|serve-fault <desktop|mobile>|' +
          'token <name> [run]|issue <name> [run]|revoke <name> [run]|status [run]>',
      )
      return 2
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    process.exitCode = main(process.argv.slice(2))
  } catch (error) {
    console.error(`[review-db] ${error.message}`)
    process.exitCode = 1
  }
}
