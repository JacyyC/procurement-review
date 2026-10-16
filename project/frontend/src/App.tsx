import { Activity, AlertCircle, Ban, Check, CheckCircle2, FileSearch, History, LoaderCircle, Play, RefreshCw, RotateCcw, Undo2, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ReviewAPIError,
  applyReviewAction,
  createReview,
  getAudit,
  getCases,
  getReview,
  getSession,
  getTrace,
  isUnauthenticated,
  listReviews,
  resumeReview,
} from './api'
import { isDisplayableRequestId } from './apiErrors'
import type { RequestAuth, ReviewAction } from './api'
import type { AuditResponse, CaseOption, ExecutionMode, Permissions, RequestOutcome, ReviewRun, ReviewScope, SessionInfo, TraceResponse } from './types'
import type { ArtifactCache, TicketCounter } from './resourceBinding'
import {
  applyArtifact,
  artifactFor,
  isLatestTicket,
  openTicket,
  selectionStillApplies,
} from './resourceBinding'
import { useRunPolling } from './detailReads'
import type { ProjectionSource } from './projectionOrdering'
import { ProjectionOrdering, projectRunList, upsertRun } from './projectionOrdering'
import {
  associateCreatedRun,
  classifyWriteFailure,
  offeredCreate,
  offeredWriteForRun,
  registerWrite,
  settleWrite,
  writeScope,
  writeTargetRun,
} from './writeOperations'
import type { WriteOperation, WriteOutcome, WriteRecord } from './writeOperations'

export { RUN_POLL_INTERVAL_MS } from './detailReads'

import { CASE_TYPE_LABELS, compactId, CREATE_NEW_REVIEW_STEP, recoveryGuidance, SCENARIO_LABELS, workerLabel } from './reviewPresentation'
import { AuditView, CredentialRequired, EmptyState, ResultView, StatusBadge, TraceView } from './ReviewViews'
import { ReviewErrorBoundary } from './ReviewErrorBoundary'
import { CredentialForm } from './SessionPanels'
import { WorkbenchFrame, RunList } from './WorkbenchFrame'

type ViewTab = 'result' | 'trace' | 'audit'
type HumanAction = ReviewAction
/** A notice always names the object it describes; `null` means the whole workspace. */
type Notice = { tone: 'success' | 'error' | 'info'; text: string; runId: string | null }
type NoticeBody = Omit<Notice, 'runId'> & { runId: string | null }
/**
 * One in-flight operation, keyed by its own token.  Writes are mutually exclusive per
 * target (a Run, or the workspace for a create); reads serialise per Run.  The token —
 * never the Run — is what a completion removes, so one operation can never clear
 * another (FIX-04A / C-AUD-03).
 */
type BusyKind = 'create' | 'action' | 'resume' | 'refresh'
type BusyOperation = { token: number; kind: BusyKind; runId: string | null; generation: number }

const NO_PERMISSIONS: Permissions = {
  caseRead: false,
  reviewRead: false,
  create: false,
  decide: false,
  cancel: false,
  resume: false,
  trace: false,
  audit: false,
}

const INVALID_CREDENTIAL_MESSAGE = '凭据无效或已失效，请重新输入'
/** The exact sentence shown when this session no longer holds the original key. */
export const REPLAY_UNAVAILABLE_HINT =
  '幂等重放需要本次会话保留的原始幂等键；服务端只返回幂等键摘要，刷新页面或重新验证凭据后无法重放。'

function operationKey(prefix: string): string {
  const suffix =
    typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`
  return `${prefix}-${suffix}`
}

function permissionsFor(session: SessionInfo | null): Permissions {
  if (!session) return NO_PERMISSIONS
  const scopes = new Set<ReviewScope>(session.scopes)
  return {
    caseRead: scopes.has('case:read'),
    reviewRead: scopes.has('review:read'),
    create: scopes.has('review:create'),
    decide: scopes.has('review:decide'),
    cancel: scopes.has('review:cancel'),
    resume: scopes.has('review:resume'),
    trace: scopes.has('trace:read'),
    audit: scopes.has('audit:read'),
  }
}

/** Which granted permission authorises (re)sending one kind of write. */
const WRITE_PERMISSIONS: Record<WriteOperation['kind'], keyof Permissions> = {
  create: 'create',
  decide: 'decide',
  cancel: 'cancel',
  resume: 'resume',
}

/** Human-facing label of one pending original request — never its key or payload. */
function writeLabel(operation: WriteOperation): string {
  switch (operation.kind) {
    case 'create':
      return '创建审查请求'
    case 'decide':
      return operation.action === 'confirm' ? '确认建议请求' : '拒绝建议请求'
    case 'cancel':
      return '取消审查请求'
    case 'resume':
      return '恢复审查请求'
  }
}

/** What the workbench knows about the request, in operator language. */
function writeStateText(outcome: WriteOutcome): string {
  if (outcome === 'in-flight') return '已发送，正在等待服务端响应。'
  if (outcome === 'unknown') {
    return '响应丢失或无法校验，结果未知：这里不会替你猜，也不会用新幂等键重发。'
  }
  return '服务端已响应；重放会复用同一幂等键并返回既有结果。'
}

/**
 * The complete map from a server code to the words the operator sees.
 *
 * A code that is not here is *not* rendered: the fallback below is a fixed sentence, and
 * the code, message and details of the response never reach the page (FIX-04B / C-AUD-04).
 * `resource_not_found` is also reached through a bare 404, which a proxy can answer.
 */
const LOCAL_ERROR_MESSAGES = {
  invalid_response: '接口返回了无法校验的响应，已停止展示该对象；请重试或检查服务端版本。',
  version_conflict: '版本已过期，已重新读取最新审查状态。',
  invalid_state_transition: '当前状态不允许该操作，Gate 未产生新记录。',
  idempotency_conflict: '幂等键已用于不同内容，请重新发起操作。',
  summary_checksum_mismatch: '展示的建议校验值已过期，已重新读取最新结果。',
  insufficient_scope: '当前凭据缺少该操作的授权范围（insufficient_scope）。',
  api_version_retired: '接口版本已停用，请使用 review-api.v2。',
  security_dependency_unavailable: '本地 Review 存储暂时不可用，操作未确认，请稍后重试。',
  review_record_unavailable: '本地记录未通过完整性校验，已停止展示该对象。',
  resource_not_found: '对象不存在或当前凭据未被授予（resource_not_found）。',
  expected_version_required: '恢复请求缺少 expected_version，未发送。',
  review_not_recoverable: '该 Run 已不可恢复，请为同一 Case 创建新的审查。',
} as const satisfies Record<string, string>

const UNREGISTERED_ERROR_MESSAGE =
  '本地 Review API 返回了未登记的错误；操作未确认，请重试或联系维护者。'
const TRANSPORT_ERROR_MESSAGE = '本地 Review API 暂时不可用，请检查服务状态。'

function errorNotice(error: unknown, runId: string | null): NoticeBody {
  const bound = (notice: Omit<NoticeBody, 'runId'>): NoticeBody => ({ ...notice, runId })
  if (error instanceof ReviewAPIError) {
    const registered = Object.hasOwn(LOCAL_ERROR_MESSAGES, error.code)
      ? LOCAL_ERROR_MESSAGES[error.code as keyof typeof LOCAL_ERROR_MESSAGES]
      : undefined
    if (registered) return bound({ tone: 'error', text: registered })
    if (error.status === 404) {
      return bound({ tone: 'error', text: LOCAL_ERROR_MESSAGES.resource_not_found })
    }
    // An unregistered code gets a fixed sentence plus the request id, and only when that
    // id itself satisfies the local identifier contract — being a string is not enough.
    const suffix = isDisplayableRequestId(error.requestId)
      ? `（请求 ID：${error.requestId}）`
      : ''
    return bound({ tone: 'error', text: `${UNREGISTERED_ERROR_MESSAGE}${suffix}` })
  }
  return bound({ tone: 'error', text: TRANSPORT_ERROR_MESSAGE })
}

/** Resume-specific failures: the server, never the client, decides recoverability. */
function resumeNotice(error: unknown, runId: string | null): NoticeBody {
  if (error instanceof ReviewAPIError) {
    if (error.code === 'review_not_recoverable') {
      return {
        tone: 'error',
        runId,
        text: `恢复预算已用尽，该 Run 已转为失败：请${CREATE_NEW_REVIEW_STEP}。`,
      }
    }
    if (error.code === 'invalid_state_transition') {
      return {
        tone: 'error',
        runId,
        text: '当前状态不可恢复：只有已中断的 Run 可以恢复。',
      }
    }
  }
  return errorNotice(error, runId)
}

/**
 * Send exactly the frozen request.  Nothing is derived from current state here: the
 * replay path and the first send run through the same function with the same object.
 */
function sendWrite(operation: WriteOperation, auth: RequestAuth): Promise<ReviewRun> {
  switch (operation.kind) {
    case 'create':
      return createReview(operation.caseId, operation.caseVersion, operation.idempotencyKey, auth)
    case 'decide':
      return applyReviewAction(
        operation.runId,
        operation.action,
        {
          expectedVersion: operation.expectedVersion,
          summaryChecksum: operation.summaryChecksum,
        },
        operation.idempotencyKey,
        auth,
      )
    case 'cancel':
      return applyReviewAction(
        operation.runId,
        'cancel',
        { expectedVersion: operation.expectedVersion },
        operation.idempotencyKey,
        auth,
      )
    case 'resume':
      return resumeReview(
        operation.runId,
        operation.expectedVersion,
        operation.idempotencyKey,
        auth,
      )
  }
}

export default function App() {
  // The credential lives only in component memory: never in storage, cookies,
  // the URL, or any build artifact.
  const [token, setToken] = useState<string | null>(null)
  const [credentialDraft, setCredentialDraft] = useState('')
  const [credentialVisible, setCredentialVisible] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const [credentialError, setCredentialError] = useState<string | null>(null)
  const [session, setSession] = useState<SessionInfo | null>(null)

  const [cases, setCases] = useState<CaseOption[]>([])
  // The API's own answer for the profile it runs; the header never guesses it.
  const [executionMode, setExecutionMode] = useState<ExecutionMode | null>(null)
  const [runs, setRuns] = useState<ReviewRun[]>([])
  const [selectedCaseId, setSelectedCaseId] = useState('')
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  const [requestedTab, setRequestedTab] = useState<ViewTab>('result')
  const [trace, setTrace] = useState<ArtifactCache<TraceResponse>>({})
  const [audit, setAudit] = useState<ArtifactCache<AuditResponse>>({})
  const [notices, setNotices] = useState<Readonly<Record<string, Notice>>>({})
  const [loading, setLoading] = useState(false)
  const [loadFailed, setLoadFailed] = useState(false)
  const [busyOps, setBusyOps] = useState<readonly BusyOperation[]>([])
  const [writes, setWrites] = useState<readonly WriteRecord[]>([])

  const permissions = useMemo(() => permissionsFor(session), [session])

  // ------------------------------------------------------------- request scope
  //
  // Four separate ownership mechanisms, each responsible for one kind of
  // invalidation (see `resourceBinding.ts`):
  //   * `generationRef`  — identity: a credential change or unmount aborts and
  //                        discards everything, including writes;
  //   * `selectionRef`   — selection: a user click bumps it, so a late response
  //                        updates its own Run's cache but never re-selects;
  //   * `ticketRef`      — per (Run, resource): only the newest outstanding read may
  //                        write its result, successes and failures alike;
  //   * `busyRef`        — per operation: two Runs may write at once, one completion
  //                        clears only its own token.
  // AbortController is only resource cleanup here; correctness never depends on a
  // mock honouring the signal.
  const generationRef = useRef(0)
  const controllersRef = useRef(new Set<AbortController>())
  const selectionRef = useRef(0)
  const ticketCounterRef = useRef<TicketCounter>({ current: 0 })
  const latestTicketsRef = useRef(new Map<string, number>())
  const noticeTicketsRef = useRef(new Map<string, number>())
  const busyRef = useRef<readonly BusyOperation[]>([])
  const busyTokenRef = useRef(0)
  const writesRef = useRef<readonly WriteRecord[]>([])
  // The one ordering state every `ReviewRun` projection is decided by: one watermark per
  // Run, cleared whenever the identity changes (FIX-07 / S-02).
  const orderingRef = useRef(new ProjectionOrdering(0))
  /**
   * The `trace_id` each cached Run declared.  A Trace response must name the Trace of the
   * Run it was requested for, and this is where that expectation comes from (C-AUD-05).
   */
  const traceIdsRef = useRef(new Map<string, string>())
  /** The Run cache as a ref, so the one writer below can read it synchronously. */
  const runsRef = useRef<readonly ReviewRun[]>([])

  // The **one** writer of the Run cache: it keeps the ref, the rendered state and the
  // `trace_id` map in step, so a Run's own `trace_id` is registered before its Trace read.
  const applyRuns = useCallback(
    (update: (current: readonly ReviewRun[]) => readonly ReviewRun[]) => {
      const next = update(runsRef.current)
      if (next === runsRef.current) return
      runsRef.current = next
      for (const run of next) traceIdsRef.current.set(run.run_id, run.trace_id)
      setRuns([...next])
    },
    [],
  )

  // The monotonic request sequence: reads, writes and list reconciliations all take it
  // from this one counter, so "newer" is a fact about the client, not a guess.
  const nextSequence = useCallback(() => (ticketCounterRef.current.current += 1), [])

  const openRequest = useCallback(() => {
    const controller = new AbortController()
    controllersRef.current.add(controller)
    return {
      generation: generationRef.current,
      signal: controller.signal,
      release: () => {
        controllersRef.current.delete(controller)
      },
    }
  }, [])

  const invalidateRequests = useCallback(() => {
    generationRef.current += 1
    // A new identity owns a new ordering state: nothing a previous credential's
    // requests established may decide where the new identity's responses land.
    orderingRef.current.reset(generationRef.current)
    for (const controller of controllersRef.current) controller.abort()
    controllersRef.current.clear()
  }, [])

  const request = useCallback(
    async <T,>(
      authToken: string,
      task: (auth: RequestAuth) => Promise<T>,
    ): Promise<RequestOutcome<T>> => {
      const ticket = openRequest()
      try {
        const value = await task({ token: authToken, signal: ticket.signal })
        if (generationRef.current !== ticket.generation) return { status: 'stale' }
        return { status: 'ok', value }
      } catch (error) {
        // Only an identity change supersedes a result.  An abort or timeout that
        // happened *within* the same identity is a real failure: the caller must be
        // able to keep the request context ("result unknown"), not discard it.
        if (generationRef.current !== ticket.generation) return { status: 'stale' }
        return { status: 'error', error }
      } finally {
        ticket.release()
      }
    },
    [openRequest],
  )

  useEffect(() => () => invalidateRequests(), [invalidateRequests])

  // ------------------------------------------------------------- notices
  //
  // A notice carries the ticket of the request that produced it.  An older request
  // that settles late may not replace the message a newer one already wrote.
  const noticeKey = useCallback((runId: string | null) => runId ?? '__workspace__', [])

  const publishNotice = useCallback(
    (body: NoticeBody, sourceTicket?: number) => {
      const key = noticeKey(body.runId)
      const ticket = sourceTicket ?? (ticketCounterRef.current.current += 1)
      const applied = noticeTicketsRef.current.get(key)
      if (applied !== undefined && ticket < applied) return
      noticeTicketsRef.current.set(key, ticket)
      setNotices((current) => ({ ...current, [key]: body }))
    },
    [noticeKey],
  )

  const dismissNotice = useCallback(
    (runId: string | null) => {
      const key = noticeKey(runId)
      noticeTicketsRef.current.set(key, (ticketCounterRef.current.current += 1))
      setNotices((current) => {
        const next = { ...current }
        delete next[key]
        return next
      })
    },
    [noticeKey],
  )

  // ------------------------------------------------------------- busy registry
  //
  // Admission is synchronous: it reads a ref, not React state, so two activations in
  // one render batch can never both be admitted.  Releases are by token, so a late
  // completion of a superseded identity cannot clear the operation that replaced it.
  const applyBusy = useCallback(
    (update: (current: readonly BusyOperation[]) => readonly BusyOperation[]) => {
      const next = update(busyRef.current)
      if (next === busyRef.current) return
      busyRef.current = next
      setBusyOps(next)
    },
    [],
  )

  const admitOperation = useCallback(
    (kind: BusyKind, runId: string | null): number | null => {
      const conflict = busyRef.current.some((operation) => {
        if (operation.runId !== runId) return false
        if (kind === 'refresh' && operation.kind === 'refresh') return true
        if (kind === 'refresh' || operation.kind === 'refresh') return false
        return true
      })
      if (conflict) return null
      const token = (busyTokenRef.current += 1)
      applyBusy((current) => [
        ...current,
        { token, kind, runId, generation: generationRef.current },
      ])
      return token
    },
    [applyBusy],
  )

  const releaseOperation = useCallback(
    (token: number) => {
      applyBusy((current) => {
        const next = current.filter((operation) => operation.token !== token)
        return next.length === current.length ? current : next
      })
    },
    [applyBusy],
  )

  // ------------------------------------------------------------- write registry
  const applyWrites = useCallback(
    (update: (current: readonly WriteRecord[]) => readonly WriteRecord[]) => {
      const next = update(writesRef.current)
      if (next === writesRef.current) return
      writesRef.current = next
      setWrites(next)
    },
    [],
  )

  /**
   * The **only** place a `ReviewRun` projection may reach the cache (FIX-07 / S-02).
   *
   * Every asynchronous source — select, refresh, poll, write success, idempotent replay,
   * create/list reconciliation, action and resume reconciliation, and the boundary's
   * recovery read — calls this with the sequence its request took when it was sent.  The
   * decision lives in `projectionOrdering.ts`; this only writes once it is allowed to,
   * and it never changes the selection.
   */
  const applyRunProjection = useCallback(
    (
      runId: string,
      sequence: number,
      next: ReviewRun,
      source: ProjectionSource,
    ): boolean => {
      const decision = orderingRef.current.decide({
        generation: generationRef.current,
        runId,
        sequence,
        revision: next.revision,
        source,
      })
      if (!decision.accepted) return false
      applyRuns((current) => upsertRun(current, next))
      return true
    },
    [applyRuns],
  )

  const clearWorkspace = useCallback(() => {
    setCases([])
    runsRef.current = []
    setRuns([])
    setSelectedCaseId('')
    setActiveRunId(null)
    setRequestedTab('result')
    setTrace({})
    setAudit({})
    setNotices({})
    setLoading(false)
    setLoadFailed(false)
    busyRef.current = []
    setBusyOps([])
    writesRef.current = []
    setWrites([])
    traceIdsRef.current.clear()
    latestTicketsRef.current.clear()
    orderingRef.current.reset(generationRef.current)
    noticeTicketsRef.current.clear()
  }, [])

  /** Drop the identity and every artifact derived from it, and abort in-flight work. */
  const endSession = useCallback(
    (message: string | null) => {
      invalidateRequests()
      selectionRef.current += 1
      setToken(null)
      setSession(null)
      setCredentialDraft('')
      setCredentialVisible(false)
      setConnecting(false)
      setCredentialError(message)
      clearWorkspace()
    },
    [clearWorkspace, invalidateRequests],
  )

  /**
   * Every failure is reported against the object it belongs to, so switching Runs
   * cannot leave another Run's error on screen.
   */
  const handleRequestError = useCallback(
    (error: unknown, runId: string | null, sourceTicket?: number) => {
      if (isUnauthenticated(error)) {
        endSession(INVALID_CREDENTIAL_MESSAGE)
        return
      }
      publishNotice(errorNotice(error, runId), sourceTicket)
    },
    [endSession, publishNotice],
  )

  // ------------------------------------------------------------- data loading

  /**
   * Read one independent resource (Trace or Audit) for one Run.
   *
   * The ticket is taken *before* the call and the result is applied only while it is
   * still the newest for that pair, so a superseded success and a superseded failure
   * are both no-ops.  Trace and Audit never wait for each other: each job applies its
   * own outcome as soon as it settles.
   */
  const loadArtifact = useCallback(
    async <T,>(
      resource: 'trace' | 'audit',
      activeToken: string,
      runId: string,
      read: (runId: string, auth: RequestAuth) => Promise<T>,
      apply: (updater: (current: ArtifactCache<T>) => ArtifactCache<T>) => void,
    ): Promise<void> => {
      const key = `${resource}:${runId}`
      const ticket = openTicket(ticketCounterRef.current, latestTicketsRef.current, key)
      const outcome = await request(activeToken, (auth) => read(runId, auth))
      if (outcome.status === 'stale') return
      if (outcome.status === 'error' && isUnauthenticated(outcome.error)) {
        endSession(INVALID_CREDENTIAL_MESSAGE)
        return
      }
      if (!isLatestTicket(latestTicketsRef.current, key, ticket)) return
      apply((current) => applyArtifact(current, runId, ticket, outcome))
      if (outcome.status === 'error') publishNotice(errorNotice(outcome.error, runId), ticket)
    },
    [endSession, publishNotice, request],
  )

  const refreshArtifacts = useCallback(
    async (activeToken: string, runId: string, granted: Permissions): Promise<void> => {
      const jobs: Promise<void>[] = []
      if (granted.trace) {
        jobs.push(
          loadArtifact(
            'trace',
            activeToken,
            runId,
            (id, auth) => getTrace(id, auth, traceIdsRef.current.get(id)),
            setTrace,
          ),
        )
      }
      if (granted.audit) {
        jobs.push(loadArtifact('audit', activeToken, runId, getAudit, setAudit))
      }
      await Promise.all(jobs)
    },
    [loadArtifact],
  )

  const loadWorkspace = useCallback(
    async (activeToken: string, info: SessionInfo) => {
      const granted = permissionsFor(info)
      setLoading(true)
      setLoadFailed(false)
      // The list is one request carrying many projections: it takes one sequence, and
      // every Run in the payload is arbitrated against that Run's own watermark.
      const sequence = nextSequence()
      const outcome = await request(activeToken, async (auth) => {
        const [caseResponse, reviewResponse] = await Promise.all([
          granted.caseRead ? getCases(auth) : Promise.resolve(null),
          granted.reviewRead ? listReviews(auth) : Promise.resolve(null),
        ])
        return { caseResponse, reviewResponse }
      })
      if (outcome.status === 'stale') return
      setLoading(false)
      if (outcome.status === 'error') {
        setLoadFailed(true)
        handleRequestError(outcome.error, null)
        return
      }
      const { caseResponse, reviewResponse } = outcome.value
      const nextCases = caseResponse?.cases ?? []
      const nextRuns = reviewResponse?.runs ?? []
      setCases(nextCases)
      if (caseResponse) setExecutionMode(caseResponse.execution_mode)
      setSelectedCaseId(nextCases[0]?.case_id ?? '')
      applyRuns((current) =>
        projectRunList(orderingRef.current, current, nextRuns, generationRef.current, sequence),
      )
      if (!granted.caseRead) {
        publishNotice({ tone: 'info', runId: null, text: '当前凭据缺少 case:read 授权，案例列表已隐藏。' })
      } else if (!granted.reviewRead) {
        publishNotice({ tone: 'info', runId: null, text: '当前凭据缺少 review:read 授权，运行列表已隐藏。' })
      }
      const firstRun = nextRuns[0]
      if (firstRun) {
        // A fresh identity starts a fresh selection epoch: nothing that was in
        // flight for the previous credential can claim this selection.
        selectionRef.current += 1
        setActiveRunId(firstRun.run_id)
        await refreshArtifacts(activeToken, firstRun.run_id, granted)
      }
    },
    [applyRuns, handleRequestError, nextSequence, publishNotice, refreshArtifacts, request],
  )

  // Derived selection.  Declared before the action functions on purpose: they are
  // hoisted function declarations that read `activeRun`/`selectedCase`, and a memo
  // that a hoisted function can reach before initialisation is one the React
  // Compiler cannot preserve (eslint `react-hooks/preserve-manual-memoization`).
  const activeRun = useMemo(
    () => runs.find((run) => run.run_id === activeRunId) ?? null,
    [activeRunId, runs],
  )
  const selectedCase = useMemo(
    () => cases.find((item) => item.case_id === selectedCaseId) ?? null,
    [cases, selectedCaseId],
  )

  async function connect() {
    const candidate = credentialDraft.trim()
    if (!candidate) return
    const previousSession = session
    // A different credential must never inherit the previous identity's view or its
    // in-memory idempotency keys — not even when the previous session had ended.
    invalidateRequests()
    clearWorkspace()
    if (previousSession) setSession(null)
    setToken(null)
    // A second connect supersedes the first: bump the generation so an earlier
    // credential's response can never become the session.
    invalidateRequests()
    selectionRef.current += 1
    setCredentialDraft('')
    setCredentialError(null)
    setConnecting(true)
    const outcome = await request(candidate, (auth) => getSession(auth))
    if (outcome.status === 'stale') return
    if (outcome.status === 'error') {
      setConnecting(false)
      if (isUnauthenticated(outcome.error)) {
        endSession(INVALID_CREDENTIAL_MESSAGE)
        return
      }
      setCredentialError(errorNotice(outcome.error, null).text)
      return
    }
    setToken(candidate)
    setSession(outcome.value)
    setConnecting(false)
    await loadWorkspace(candidate, outcome.value)
  }

  // -------------------------------------------------------------- run actions

  async function selectRun(runId: string) {
    if (!token) return
    // Eager selection: the human's click wins immediately.  The epoch bump is what
    // makes later responses for other Runs harmless.
    const epoch = (selectionRef.current += 1)
    setActiveRunId(runId)
    setRequestedTab('result')
    dismissNotice(runId)
    const sequence = nextSequence()
    const outcome = await request(token, (auth) => getReview(runId, auth))
    if (outcome.status === 'stale') return
    if (outcome.status === 'error') {
      handleRequestError(outcome.error, runId, sequence)
      return
    }
    // A superseded read may not touch the cache or start any follow-up read.
    if (!applyRunProjection(runId, sequence, outcome.value, 'read')) return
    if (!selectionStillApplies(epoch, selectionRef.current)) {
      // The human moved on while this read was in flight: keep the cache warm and
      // skip the artifact read for a Run that is no longer on screen.
      return
    }
    await refreshArtifacts(token, runId, permissions)
  }

  /**
   * Send one write — either a fresh human action or an explicit replay of a request
   * this session already sent.  The operation object is immutable and was frozen
   * before this call, so a replay can never be re-derived from current state.
   */
  async function submitWrite(operation: WriteOperation, replay: boolean) {
    if (!token) return
    const target = writeTargetRun(operation)
    const kind: BusyKind =
      operation.kind === 'create' ? 'create' : operation.kind === 'resume' ? 'resume' : 'action'
    const slot = admitOperation(kind, target)
    if (slot === null) return
    // Registered *before* the request leaves: if the response never arrives, the key
    // and the exact body are still here for an explicit same-key replay.
    const epoch = selectionRef.current
    const generation = generationRef.current
    // One sequence for this write, taken before it is sent: its response is a projection
    // like any other, and it must lose to a read that was issued *after* it.
    const sequence = nextSequence()
    applyWrites((current) => registerWrite(current, operation, generation))
    dismissNotice(target)
    try {
      const outcome = await request(token, (auth) => sendWrite(operation, auth))
      if (outcome.status === 'stale') return
      if (outcome.status === 'error') {
        applyWrites((current) =>
          settleWrite(current, operation.operationId, writeFailureOutcome(outcome.error)),
        )
        if (isUnauthenticated(outcome.error)) {
          endSession(INVALID_CREDENTIAL_MESSAGE)
          return
        } else {
          publishNotice(
            operation.kind === 'resume'
              ? resumeNotice(outcome.error, target)
              : errorNotice(outcome.error, target),
          )
        }
        // Re-read so the operator sees the real state.  The write itself is never
        // retried here: only an explicit replay may send it again.
        await reconcileWrite(operation)
        return
      }
      applyWrites((current) => settleWrite(current, operation.operationId, 'responded'))
      const next = outcome.value
      applyRunProjection(next.run_id, sequence, next, 'write')
      if (operation.kind === 'create') {
        applyWrites((current) =>
          associateCreatedRun(current, operation.operationId, next.run_id),
        )
        // The new Run takes the selection only if the human has not selected another
        // Run while the create was in flight.
        if (selectionStillApplies(epoch, selectionRef.current)) {
          setRequestedTab('result')
          setActiveRunId(next.run_id)
        }
      }
      if (operation.kind !== 'create') {
        await refreshArtifacts(token, next.run_id, permissions)
      }
      if (generationRef.current !== generation) return
      publishNotice(successNotice(operation, next, replay))
    } finally {
      releaseOperation(slot)
    }
  }

  /** Read-only reconciliation after a write whose outcome is unknown. */
  async function reconcileWrite(operation: WriteOperation) {
    if (!token) return
    const sequence = nextSequence()
    if (operation.kind === 'create') {
      // The Run may exist even though its response was lost; a list read is the only
      // honest way to find out, and it never re-sends the create.  It is arbitrated per
      // Run, so a stale list cannot replace a newer cached projection.
      const outcome = await request(token, (auth) => listReviews(auth))
      if (outcome.status !== 'ok') {
        if (outcome.status === 'error') handleRequestError(outcome.error, null)
        return
      }
      const generation = generationRef.current
      applyRuns((current) =>
        projectRunList(orderingRef.current, current, outcome.value.runs, generation, sequence),
      )
      return
    }
    const runId = operation.runId
    const latest = await request(token, (auth) => getReview(runId, auth))
    if (latest.status !== 'ok') {
      if (latest.status === 'error') handleRequestError(latest.error, runId)
      return
    }
    applyRunProjection(runId, sequence, latest.value, 'reconcile')
    await refreshArtifacts(token, runId, permissions)
  }

  async function startReview() {
    if (!selectedCase || !permissions.create) return
    await submitWrite(
      {
        kind: 'create',
        operationId: operationKey('op-create'),
        caseId: selectedCase.case_id,
        caseVersion: selectedCase.case_version,
        idempotencyKey: operationKey('ui-create'),
        runId: null,
      },
      false,
    )
  }

  async function runAction(actionName: HumanAction) {
    if (!token || !activeRun) return
    // The object identity, version, checksum and idempotency key are frozen at the
    // moment of the click.  Nothing below re-reads `activeRun`, so a selection
    // change cannot redirect an in-flight decision to another Run.
    if (actionName !== 'cancel' && !activeRun.summary_checksum) {
      publishNotice({
        tone: 'error',
        runId: activeRun.run_id,
        text: '当前 Run 没有可核对 summary_checksum，无法记录人工决定。',
      })
      return
    }
    const operation: WriteOperation =
      actionName === 'cancel'
        ? {
            kind: 'cancel',
            operationId: operationKey('op-cancel'),
            runId: activeRun.run_id,
            idempotencyKey: operationKey('ui-cancel'),
            expectedVersion: activeRun.version,
          }
        : {
            kind: 'decide',
            operationId: operationKey('op-decide'),
            runId: activeRun.run_id,
            action: actionName,
            idempotencyKey: operationKey(`ui-${actionName}`),
            expectedVersion: activeRun.version,
            summaryChecksum: activeRun.summary_checksum,
          }
    await submitWrite(operation, false)
  }

  /**
   * Explicit replay of a request this session already sent.  It reuses the stored
   * operation verbatim — including its original `expected_version` and checksum —
   * so a Run that has since moved to another state is still replayable rather than
   * being silently treated as a new action.
   */
  async function replayWrite(record: WriteRecord) {
    if (record.generation !== generationRef.current) return
    await submitWrite(record.operation, true)
  }

  /**
   * Resume one interrupted Run.  It is only ever triggered by an explicit click,
   * it always sends the *displayed* `version` plus a fresh idempotency key, and it
   * never retries on its own: a rejected resume surfaces the server's reason.
   */
  async function resumeActiveRun() {
    if (!token || !activeRun || !canResume) return
    await submitWrite(
      {
        kind: 'resume',
        operationId: operationKey('op-resume'),
        runId: activeRun.run_id,
        idempotencyKey: operationKey('ui-resume'),
        expectedVersion: activeRun.version,
      },
      false,
    )
  }

  async function refreshCurrent() {
    if (!activeRun || !token) return
    const runId = activeRun.run_id
    const generation = generationRef.current
    const slot = admitOperation('refresh', runId)
    if (slot === null) return
    try {
      const sequence = nextSequence()
      const outcome = await request(token, (auth) => getReview(runId, auth))
      if (outcome.status === 'stale') return
      if (outcome.status === 'error') {
        handleRequestError(outcome.error, runId, sequence)
        return
      }
      if (!applyRunProjection(runId, sequence, outcome.value, 'read')) return
      await refreshArtifacts(token, runId, permissions)
      if (generationRef.current !== generation) return
      publishNotice({ tone: 'info', runId, text: '已读取最新状态。' }, sequence)
    } finally {
      releaseOperation(slot)
    }
  }

  /**
   * Recovery from a render failure: clear derived view state and re-issue *reads*
   * only.  No create/resume/cancel/decide request is ever re-sent here — the
   * workbench has no write-retry path at all — and the session's record of an
   * unresolved original request is deliberately left untouched.
   */
  const latestRef = useRef<{ token: string | null; runId: string | null; granted: Permissions }>({
    token: null,
    runId: null,
    granted: NO_PERMISSIONS,
  })
  useEffect(() => {
    latestRef.current = { token, runId: activeRunId, granted: permissions }
  }, [activeRunId, permissions, token])

  const recoverView = useCallback(() => {
    const { token: activeToken, runId, granted } = latestRef.current
    dismissNotice(runId)
    setTrace({})
    setAudit({})
    if (!activeToken || !runId) return
    void (async () => {
      const sequence = nextSequence()
      const outcome = await request(activeToken, (auth) => getReview(runId, auth))
      if (outcome.status !== 'ok') {
        if (outcome.status === 'error') handleRequestError(outcome.error, runId, sequence)
        return
      }
      if (!applyRunProjection(runId, sequence, outcome.value, 'read')) return
      await refreshArtifacts(activeToken, runId, granted)
    })()
  }, [applyRunProjection, dismissNotice, handleRequestError, nextSequence, refreshArtifacts, request])

  // Poll only while the active run is processing; every tick takes its own detail
  // read ticket, so a late tick can never roll a newer read back (see `detailReads`).
  const processingRunId = activeRun?.status === 'processing' ? activeRun.run_id : null
  useRunPolling({
    processingRunId,
    token,
    permissions,
    nextSequence,
    request,
    applyProjection: applyRunProjection,
    handleRequestError,
    refreshArtifacts,
  })
  const availableTabs = useMemo(() => {
    const tabs: ViewTab[] = ['result']
    if (permissions.trace) tabs.push('trace')
    if (permissions.audit) tabs.push('audit')
    return tabs
  }, [permissions.audit, permissions.trace])
  const activeTab = availableTabs.includes(requestedTab) ? requestedTab : 'result'

  // Roving focus for the arrow-key navigation: the tablist is rendered from
  // `availableTabs`, so a scope change can only ever hand focus to a tab that is
  // actually rendered.  `activeTab` above is the single source of truth for both the
  // selection and which tab carries `tabIndex=0`.
  const tabRefs = useRef<Partial<Record<ViewTab, HTMLButtonElement | null>>>({})

  const runDecidable =
    activeRun?.status === 'pending_confirmation' ||
    activeRun?.status === 'pending_human_review'
  const canDecide = Boolean(permissions.decide && runDecidable)
  const canCancel = Boolean(
    permissions.cancel &&
      activeRun &&
      [
        'processing',
        'pending_confirmation',
        'pending_human_review',
        'interrupted',
      ].includes(activeRun.status),
  )
  // Resume is offered only for an interrupted Run the server still calls resumable.
  const runInterrupted = activeRun?.status === 'interrupted'
  const canResume = Boolean(
    permissions.resume && runInterrupted && activeRun?.recovery?.resumable,
  )
  const recoveryBlockReason = !activeRun?.recovery
    ? '服务端未返回恢复信息，无法从检查点恢复'
    : recoveryGuidance(activeRun).budgetExhausted
      ? '恢复预算已用尽，请为同一 Case 创建新的审查'
      : '当前 Run 不可恢复，请为同一 Case 创建新的审查'
  const decideNeedsChecksum = canDecide && !activeRun?.summary_checksum

  // ------------------------------------------------- original-request entry
  //
  // Exactly one replay entry per Run: the oldest request whose result is still
  // unresolved, otherwise the most recent one this session sent.  The stored key and
  // body are the only things a replay may send, and the action's own Scope — never
  // `review:decide` for everything — decides whether that entry is usable.
  const offeredRecord = activeRunId ? offeredWriteForRun(writes, activeRunId) : null
  const offeredCreateRecord = offeredCreate(writes)
  const offeredOperation = offeredRecord?.operation ?? null
  const offeredGranted = offeredOperation
    ? permissions[WRITE_PERMISSIONS[offeredOperation.kind]]
    : false
  const replayInFlight = offeredRecord?.outcome === 'in-flight'
  const canReplayOffered = Boolean(offeredRecord && offeredGranted && !replayInFlight)
  const replayHint = !offeredOperation
    ? REPLAY_UNAVAILABLE_HINT
    : !offeredGranted
      ? `缺少 ${writeScope(offeredOperation)} 授权，无法重放该请求。`
      : ''
  const showRequestPanel = Boolean(activeRun && (activeRun.receipt !== null || offeredRecord))
  const canReplayCreate = Boolean(
    offeredCreateRecord && permissions.create && offeredCreateRecord.outcome !== 'in-flight',
  )

  // Only the selected Run's own artifacts are rendered, and an operation in flight
  // for another Run never disables this Run's controls.
  const activeTrace = artifactFor(trace, activeRunId)
  const activeAudit = artifactFor(audit, activeRunId)
  const visibleNotice = notices[noticeKey(activeRunId)] ?? notices[noticeKey(null)] ?? null
  const busyForActiveRun = busyOps.some(
    (operation) => operation.runId === null || operation.runId === activeRunId,
  )
  const createInFlight = busyOps.some((operation) => operation.kind === 'create')
  const resumeInFlight = busyOps.some(
    (operation) => operation.kind === 'resume' && operation.runId === activeRunId,
  )

  return (
    <ReviewErrorBoundary scope="工作台" onReset={recoverView} onSignOut={() => endSession(null)}>
    <WorkbenchFrame session={session} cases={cases} runs={runs} executionMode={executionMode} guide={{ permissions, connecting, loading, creating: createInFlight, loadFailed, hasCases: cases.length > 0 }} onSignOut={() => endSession(null)}>
      {!session && <div className="identity-band" id="identity">
        <CredentialForm
          draft={credentialDraft}
          visible={credentialVisible}
          connecting={connecting}
          error={credentialError}
          onDraftChange={setCredentialDraft}
          onToggleVisible={() => setCredentialVisible((value) => !value)}
          onSubmit={() => void connect()}
        />
      </div>}

      <div className="workspace-layout">
        <aside className="sidebar" aria-label="案例选择">
          <section className="case-control" id="case-control">
            <div className="sidebar-heading">
              <span>发起新审查</span>
              <FileSearch size={18} aria-hidden="true" />
            </div>
            <label htmlFor="case-select">已授权案例</label>
            <select
              id="case-select"
              value={selectedCaseId}
              onChange={(event) => setSelectedCaseId(event.target.value)}
              disabled={!session || !permissions.caseRead || loading || createInFlight || loadFailed || cases.length === 0}
            >
              {cases.length === 0 && <option value="">{loading ? '正在加载案例…' : loadFailed ? '案例读取失败' : '无可用案例'}</option>}
              {cases.map((item) => (
                <option key={item.case_id} value={item.case_id}>
                  {item.label}
                </option>
              ))}
            </select>
            <p className="case-description">
              {selectedCase
                ? `${CASE_TYPE_LABELS[selectedCase.case_type]} · ${SCENARIO_LABELS[selectedCase.scenario]} · 规则 ${selectedCase.rules_version} · ${selectedCase.case_version}`
                : loading ? '正在加载案例…'
                : loadFailed ? '读取工作台失败，请退出后重新验证凭据。'
                : session
                  ? '没有可用的已授权案例。'
                  : '验证凭据后读取已授权案例。'}
            </p>
            {permissions.create ? (
              <div className="create-actions">
                <button
                  className="primary-button"
                  type="button"
                  onClick={() => void startReview()}
                  disabled={!selectedCase || createInFlight}
                >
                  {createInFlight ? (
                    <LoaderCircle className="spin" size={18} aria-hidden="true" />
                  ) : (
                    <Play size={18} aria-hidden="true" />
                  )}
                  运行审查
                </button>
                <button
                  className="icon-button"
                  type="button"
                  aria-label="重放最近创建请求"
                  title={
                    offeredCreateRecord
                      ? '使用同一幂等键重放最近的创建请求'
                      : '本次会话没有可重放的创建请求'
                  }
                  onClick={() => offeredCreateRecord && void replayWrite(offeredCreateRecord)}
                  disabled={!canReplayCreate}
                >
                  <RotateCcw size={18} aria-hidden="true" />
                </button>
              </div>
            ) : (
              <p className="scope-hint">
                缺少 review:create 授权，运行审查已隐藏。
              </p>
            )}
          </section>

        </aside>

        <main className="review-main" id="review-content" tabIndex={-1}>
          {visibleNotice && (
            <div className={`notice notice-${visibleNotice.tone}`} role={visibleNotice.tone === 'error' ? 'alert' : 'status'}>
              {visibleNotice.tone === 'error' ? (
                <AlertCircle aria-hidden="true" />
              ) : visibleNotice.tone === 'success' ? (
                <CheckCircle2 aria-hidden="true" />
              ) : (
                <Activity aria-hidden="true" />
              )}
              <span>{visibleNotice.text}</span>
              <button
                className="icon-button small"
                type="button"
                aria-label="关闭反馈"
                title="关闭反馈"
                onClick={() => dismissNotice(visibleNotice.runId)}
              >
                <X size={16} aria-hidden="true" />
              </button>
            </div>
          )}

          {!session ? (
            <CredentialRequired />
          ) : !activeRun ? (
            <EmptyState loading={loading} />
          ) : (
            <ReviewErrorBoundary scope="审查视图" onReset={recoverView} onSignOut={() => endSession(null)}>
            <div className="review-view" aria-busy={busyForActiveRun}>
              <section className="run-header">
                <div className="run-title">
                  <div>
                    <StatusBadge status={activeRun.status} />
                    <span className="mode-inline">本地合成案例</span>
                  </div>
                  <h2>{cases.find((item) => item.case_id === activeRun.request.case_id)?.label ?? activeRun.request.case_id}</h2>
                  <span className="case-reference">{activeRun.request.case_id}</span>
                  <p>
                    <code title={activeRun.run_id}>{compactId(activeRun.run_id)}</code>
                    <span>v{activeRun.version}</span>
                    <span>{activeRun.request.purchase_order_id}</span>
                    <span title="服务端根据已验证凭据推导">
                      主体 {activeRun.request.initiator}
                    </span>
                  </p>
                  {activeRun.summary_checksum && (
                    <p className="checksum-line">
                      <code title={activeRun.summary_checksum}>
                        校验值 {compactId(activeRun.summary_checksum)}
                      </code>
                    </p>
                  )}
                  {activeRun.recovery &&
                    (activeRun.recovery.resume_count > 0 || runInterrupted) && (
                      <p className="recovery-line">
                        <span
                          className="recovery-chip"
                          data-testid="recovery-budget-chip"
                          title="服务端记录的恢复次数 / 恢复上限"
                        >
                          恢复 {activeRun.recovery.resume_count} /{' '}
                          {activeRun.recovery.resume_limit}
                        </span>
                        {activeRun.recovery.adopted_workers.length > 0 && (
                          <span className="recovery-chip subtle">
                            已复用分支{' '}
                            {activeRun.recovery.adopted_workers.map(workerLabel).join('、')}
                          </span>
                        )}
                      </p>
                    )}
                </div>
                <div className="gate-actions" aria-label="人工 Gate">
                  {runInterrupted ? (
                    <span className="scope-hint inline" data-testid="interrupted-gate-note">
                      已中断的 Run 没有建议与回执，确认/拒绝不适用。
                    </span>
                  ) : permissions.decide ? (
                    <>
                      <button
                        className="confirm-button"
                        type="button"
                        onClick={() => void runAction('confirm')}
                        disabled={!canDecide || busyForActiveRun || decideNeedsChecksum}
                        title={
                          decideNeedsChecksum
                            ? '缺少 summary_checksum，无法记录人工决定'
                            : canDecide
                              ? '确认当前建议并记录本地模拟回执'
                              : '当前状态不允许确认'
                        }
                      >
                        <Check size={18} aria-hidden="true" />
                        确认建议（本地）
                      </button>
                      <button
                        className="reject-button"
                        type="button"
                        onClick={() => void runAction('reject')}
                        disabled={!canDecide || busyForActiveRun || decideNeedsChecksum}
                        title={
                          decideNeedsChecksum
                            ? '缺少 summary_checksum，无法记录人工决定'
                            : canDecide
                              ? '拒绝当前建议并记录本地模拟回执'
                              : '当前状态不允许拒绝'
                        }
                      >
                        <X size={18} aria-hidden="true" />
                        拒绝建议（本地）
                      </button>
                    </>
                  ) : (
                    <span className="scope-hint inline">
                      缺少 review:decide 授权，确认/拒绝已隐藏。
                    </span>
                  )}
                  {permissions.resume ? (
                    runInterrupted && (
                      <button
                        className="resume-button"
                        type="button"
                        data-testid="resume-button"
                        onClick={() => void resumeActiveRun()}
                        disabled={!canResume || busyForActiveRun}
                        title={
                          canResume
                            ? '按当前显示的版本从已提交的检查点恢复执行'
                            : recoveryBlockReason
                        }
                      >
                        {resumeInFlight ? (
                          <LoaderCircle className="spin" size={18} aria-hidden="true" />
                        ) : (
                          <Undo2 size={18} aria-hidden="true" />
                        )}
                        恢复审查
                      </button>
                    )
                  ) : (
                    runInterrupted && (
                      <span className="scope-hint inline">
                        缺少 review:resume 授权，恢复审查已隐藏。
                      </span>
                    )
                  )}
                  {permissions.cancel ? (
                    <button
                      className="cancel-button"
                      type="button"
                      onClick={() => void runAction('cancel')}
                      disabled={!canCancel || busyForActiveRun}
                      title={canCancel ? '取消 Review，不生成回执' : '当前状态不允许取消'}
                    >
                      <Ban size={18} aria-hidden="true" />
                      取消
                    </button>
                  ) : (
                    <span className="scope-hint inline">缺少 review:cancel 授权，取消已隐藏。</span>
                  )}
                  <button
                    className="icon-button"
                    type="button"
                    onClick={() => void refreshCurrent()}
                    disabled={busyForActiveRun}
                    aria-label="刷新当前 Review"
                    title="刷新当前 Review"
                  >
                    <RefreshCw size={18} aria-hidden="true" />
                  </button>
                </div>
              </section>

              {!canDecide && !canCancel && !runInterrupted && (
                <p className="gate-closed">
                  Gate 已关闭；新动作会被拒绝，仅可幂等重放原动作。
                </p>
              )}
              {runInterrupted && !canResume && (
                <p className="gate-closed" data-testid="resume-closed">
                  当前不可恢复：{recoveryBlockReason}。
                </p>
              )}
              {decideNeedsChecksum && (
                <p className="gate-closed">
                  服务端未返回 summary_checksum，界面不会代替服务端计算，人工决定保持禁用。
                </p>
              )}

              {showRequestPanel && (
                <section
                  className="receipt-panel request-replay"
                  data-testid="pending-request"
                  aria-labelledby="pending-request-title"
                >
                  <div className="section-heading">
                    <div>
                      <span className="section-kicker">SESSION · IDEMPOTENT REPLAY</span>
                      <h2 id="pending-request-title">本会话原请求</h2>
                    </div>
                    <RotateCcw aria-hidden="true" />
                  </div>
                  <p className="scope-hint" data-testid="pending-request-state">
                    {offeredOperation
                      ? `${writeLabel(offeredOperation)}：${writeStateText(offeredRecord!.outcome)}`
                      : '本次会话没有保留该对象的原始幂等键。'}
                  </p>
                  <div className="receipt-footer">
                    <code>
                      {offeredOperation
                        ? `范围 ${writeScope(offeredOperation)}`
                        : '范围 review:decide'}
                    </code>
                    <button
                      className="icon-text-button"
                      type="button"
                      onClick={() => offeredRecord && void replayWrite(offeredRecord)}
                      disabled={!canReplayOffered}
                      title={
                        canReplayOffered
                          ? '使用同一幂等键与同一请求体重放原动作'
                          : replayInFlight
                            ? '原请求仍在飞行中，请等待结果'
                            : '当前会话没有可重放的原请求'
                      }
                    >
                      {replayInFlight ? (
                        <LoaderCircle className="spin" size={17} aria-hidden="true" />
                      ) : (
                        <RotateCcw size={17} aria-hidden="true" />
                      )}
                      重放原动作
                    </button>
                  </div>
                  {!canReplayOffered && replayHint && (
                    <p className="scope-hint">{replayHint}</p>
                  )}
                </section>
              )}

              <div className="view-tabs" role="tablist" aria-label="Review 视图"
                onKeyDown={(event) => {
                  const keys = ['ArrowLeft', 'ArrowRight', 'Home', 'End']
                  if (!keys.includes(event.key)) return
                  event.preventDefault()
                  // Navigate from the tab that actually has focus, not from the last
                  // rendered selection: after a rapid key sequence the render may not
                  // have committed yet, and deriving the origin from stale state
                  // sends the second press to the wrong tab (observed on the mobile
                  // project before this was fixed).
                  const focusedId = (event.target as HTMLElement | null)?.id
                  const focused = focusedId?.startsWith('review-tab-')
                    ? (focusedId.slice('review-tab-'.length) as ViewTab)
                    : null
                  const index =
                    focused && availableTabs.includes(focused)
                      ? availableTabs.indexOf(focused)
                      : availableTabs.indexOf(activeTab)
                  const next = event.key === 'Home' ? 0 : event.key === 'End' ? availableTabs.length - 1
                    : (index + (event.key === 'ArrowRight' ? 1 : -1) + availableTabs.length) % availableTabs.length
                  setRequestedTab(availableTabs[next])
                  tabRefs.current[availableTabs[next]]?.focus()
                }}>
                <button
                  role="tab"
                  aria-selected={activeTab === 'result'}
                  id="review-tab-result"
                  aria-controls="review-tab-panel"
                  tabIndex={activeTab === 'result' ? 0 : -1}
                  className={activeTab === 'result' ? 'active' : ''}
                  ref={(node) => { tabRefs.current.result = node }}
                  onClick={() => setRequestedTab('result')}
                  type="button"
                >
                  <FileSearch size={17} aria-hidden="true" />
                  审查结果
                </button>
                {permissions.trace && (
                  <button
                    role="tab"
                    aria-selected={activeTab === 'trace'}
                  id="review-tab-trace"
                  aria-controls="review-tab-panel"
                  tabIndex={activeTab === 'trace' ? 0 : -1}
                    className={activeTab === 'trace' ? 'active' : ''}
                    ref={(node) => { tabRefs.current.trace = node }}
                    onClick={() => setRequestedTab('trace')}
                    type="button"
                  >
                    <Activity size={17} aria-hidden="true" />
                    运行 Trace
                    <span>{activeRun.trace_event_count}</span>
                  </button>
                )}
                {permissions.audit && (
                  <button
                    role="tab"
                    aria-selected={activeTab === 'audit'}
                  id="review-tab-audit"
                  aria-controls="review-tab-panel"
                  tabIndex={activeTab === 'audit' ? 0 : -1}
                    className={activeTab === 'audit' ? 'active' : ''}
                    ref={(node) => { tabRefs.current.audit = node }}
                    onClick={() => setRequestedTab('audit')}
                    type="button"
                  >
                    <History size={17} aria-hidden="true" />
                    审批审计
                    <span>{activeRun.audit_event_count}</span>
                  </button>
                )}
              </div>
              {(!permissions.trace || !permissions.audit) && (
                <p className="scope-hint">
                  {!permissions.trace && '缺少 trace:read 授权，Trace 已隐藏。'}
                  {!permissions.trace && !permissions.audit && ' '}
                  {!permissions.audit && '缺少 audit:read 授权，审批审计已隐藏。'}
                </p>
              )}

              <div className="tab-content" role="tabpanel" id="review-tab-panel" aria-labelledby={`review-tab-${activeTab}`} tabIndex={0}>
                {activeTab === 'result' && <ResultView run={activeRun} />}
                {activeTab === 'trace' && <TraceView trace={activeTrace} />}
                {activeTab === 'audit' && <AuditView audit={activeAudit} />}
              </div>
            </div>
            </ReviewErrorBoundary>
          )}
        </main>
        <RunList key={session?.credential_id ?? 'anonymous'} runs={runs} cases={cases} activeRunId={activeRunId} signedIn={Boolean(session)} onSelect={(id) => void selectRun(id)} />
      </div>
    </WorkbenchFrame>
    </ReviewErrorBoundary>
  )
}

/** The message shown after a write the server answered. */
function successNotice(operation: WriteOperation, next: ReviewRun, replay: boolean): NoticeBody {
  const runId = next.run_id
  if (operation.kind === 'create') {
    return {
      tone: replay ? 'info' : 'success',
      runId,
      text: replay
        ? '创建请求已幂等重放：复用同一 Run，未重复执行。'
        : 'Review 已创建，可在下方查看审查进度与结果。',
    }
  }
  if (operation.kind === 'resume') {
    return {
      tone: 'success',
      runId,
      text:
        next.status === 'processing'
          ? '恢复请求已提交：已从已提交的检查点继续执行，本次恢复计入预算。'
          : '恢复请求已被服务端幂等重放：未开启新的执行代次。',
    }
  }
  if (replay) {
    return { tone: 'info', runId, text: '人工动作已幂等重放：回执与审计记录均未增加。' }
  }
  return {
    tone: 'success',
    runId,
    text:
      operation.kind === 'cancel'
        ? 'Review 已取消；没有生成本地模拟回执。'
        : '人工动作仅在本地记录，未写入采购系统。',
  }
}

/** What a failed send tells us about the server's state. */
function writeFailureOutcome(error: unknown): WriteOutcome {
  if (error instanceof ReviewAPIError) return classifyWriteFailure(error.status, error.code)
  return 'unknown'
}
