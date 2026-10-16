/**
 * Transport types for the authenticated `review-api.v2` surface.
 *
 * Contract notes verified against the running backend:
 *  - `session`, `cases` and `health` are stamped `review-api.v2`.
 *  - `ReviewRun` / review list payloads are still emitted by the shared
 *    `ReviewRunResponse` contract, so their `api_version` literal is
 *    `review-api.v1` even though they are only served under `/api/v2`.
 *  - The trace envelope, the approval audit envelope and the error envelope come
 *    from the same shared v1-stamped contracts.
 *
 * These literals mirror the server exactly; they are documentation of the wire
 * format, not an instruction to send them back. Request bodies are always built
 * with `API_VERSION_V2`.
 */

export const API_VERSION_V2 = 'review-api.v2'
export const API_VERSION_V1 = 'review-api.v1'

export type ApiVersionV2 = typeof API_VERSION_V2
export type ApiVersionV1 = typeof API_VERSION_V1

export type ReviewScenario =
  | 'low'
  | 'high'
  | 'conflict'
  | 'unknown'
  | 'partial_failure'
  | 'all_failure'
  | 'version_mismatch'
  | 'cancellable'
  | 'runtime_failure'

export type ReviewStatus =
  | 'processing'
  | 'pending_confirmation'
  | 'pending_human_review'
  | 'confirmed'
  | 'rejected'
  | 'cancelled'
  | 'failed'
  /**
   * A product run state, not a recommendation: the execution generation stopped
   * before it committed a result.  Such a run has `summary: null`, `receipt: null`
   * and `error_code` set to the interruption reason, and it is the only state an
   * authorized operator may resume.
   */
  | 'interrupted'

export type RiskLevel = 'unknown' | 'low' | 'medium' | 'high' | 'critical'

/** Scopes the credential may carry; the server decides, the UI only hides controls. */
export type ReviewScope =
  | 'case:read'
  | 'review:read'
  | 'review:create'
  | 'review:cancel'
  | 'review:resume'
  | 'review:decide'
  | 'trace:read'
  | 'audit:read'

export type SessionRole = 'reader' | 'operator' | 'approver' | 'auditor'

/** Which read/write affordances the current session's scopes grant, derived client-side. */
export interface Permissions {
  caseRead: boolean
  reviewRead: boolean
  create: boolean
  decide: boolean
  cancel: boolean
  resume: boolean
  trace: boolean
  audit: boolean
}

/**
 * Result of one authenticated call.  `stale` means the response arrived after an
 * identity change, so it must never be applied.  A response that is merely late for
 * the current *selection* is still `ok`: it updates its own Run's cache entry and is
 * prevented from moving the selection by the selection epoch check at the call site.
 */
export type RequestOutcome<T> =
  | { status: 'ok'; value: T }
  | { status: 'error'; error: unknown }
  | { status: 'stale' }

/** `GET /api/v2/session` — the caller's own verified identity, never a secret. */
export interface SessionInfo {
  api_version: ApiVersionV2
  subject_id: string
  tenant_id: string
  credential_id: string
  authorization_revision: number
  roles: SessionRole[]
  scopes: ReviewScope[]
  issued_at: string
  expires_at: string
}

/** `GET /api/v2/health` — the only anonymous endpoint. */
/** The execution profiles the workbench can run under. */
export type ExecutionMode = 'test_only' | 'real'

export interface HealthResponse {
  api_version: ApiVersionV2
  service: 'review-api'
  status: 'ok'
  execution_mode: ExecutionMode
  security_store: 'ready' | 'not_configured'
}

/**
 * One Case the credential is granted. v2 carries the pinned `case_version` that
 * must be echoed on create; it no longer exposes a purchase order id, a free
 * scenario choice or a description.
 */
export interface CaseOption {
  api_version: ApiVersionV2
  case_id: string
  case_version: string
  label: string
  case_type: 'purchase_order' | 'contract_change'
  scenario: ReviewScenario
  document_version: string
  dataset_version: string
  index_version: string
  rules_version: string
}

export interface CaseListResponse {
  api_version: ApiVersionV2
  cases: CaseOption[]
  execution_mode: ExecutionMode
}

export interface EvidenceRef {
  schema_version: '1.0'
  source_kind: 'public_document' | 'case_document'
  source_id: string
  version: string
  page_number: number | null
  section: string | null
  chunk_id: string | null
  content_checksum: string
}

export interface BusinessFact {
  schema_version: '1.0'
  tool_name: string
  record_id: string
  dataset_version: string
  as_of: string
  status: 'known' | 'unknown' | 'upstream_failure'
  result: Record<string, unknown>
  failure_reason: string | null
  read_only: true
}

export interface Finding {
  schema_version: '1.0'
  worker: 'policy' | 'contract' | 'procurement_data'
  finding_key: string
  category: string
  risk_level: RiskLevel
  assertion: 'risk_present' | 'risk_absent' | 'unknown'
  statement: string
  status:
    | 'supported'
    | 'unknown'
    | 'insufficient_evidence'
    | 'upstream_failure'
    | 'invalid_input'
  evidence_refs: EvidenceRef[]
  business_fact_refs: BusinessFact[]
  missing_information: string[]
  recommended_action: string
}

export interface FindingConflict {
  schema_version: '1.0'
  finding_key: string
  reason: 'opposing_assertions' | 'category_mismatch'
  assertions: string[]
  categories: string[]
  statements: string[]
}

export interface ReviewSummary {
  schema_version: '1.0'
  run_id: string
  case_id: string
  rules_version: string
  findings: Finding[]
  highest_risk: RiskLevel
  recommendation:
    | 'recommend_approve'
    | 'recommend_conditional'
    | 'recommend_reject'
    | 'insufficient_information'
    | 'needs_human_review'
  human_review_status: 'pending_confirmation' | 'required' | 'completed'
  conflicts: FindingConflict[]
  missing_information: string[]
  failure_reasons: string[]
  is_final_decision: false
}

export interface BranchFailure {
  api_version: ApiVersionV1
  worker: 'policy' | 'contract' | 'procurement_data'
  code: string
  detail: string
}

export interface ConfirmationReceipt {
  /**
   * `ReceiptViewV2` is an `APIContractV2`: it carries `api_version` and has no
   * `schema_version`.  The domain `ConfirmationReceipt` that does carry one is the
   * retired v1 receipt, which also returned the raw idempotency key.
   */
  api_version: ApiVersionV2
  receipt_id: string
  run_id: string
  case_id: string
  /** Server-derived verified subject id; no client supplied actor exists in v2. */
  human_actor: string
  decision: 'confirm_recommendation' | 'reject_recommendation'
  reviewed_recommendation: ReviewSummary['recommendation']
  recorded_at: string
  /**
   * The v2 receipt projection deliberately withholds the raw idempotency key and
   * exposes only its checksum, so a client can never replay another caller's key.
   * A replay is therefore only possible while this session still holds the key it
   * generated.
   */
  idempotency_key_digest: string
  simulation_status: 'local_simulation_recorded'
  simulation_mode: 'local_test_only'
  external_write_performed: false
}

/**
 * `recovery` on `ReviewRun` — everything the UI needs to explain an interruption
 * and to decide whether it may offer a resume.
 *
 * `resumable` is the server's own verdict (`status === 'interrupted'` and the
 * resume budget is not exhausted); the UI never derives it.  `attempt_totals`
 * counts worker attempts per worker for the current generation.
 */
export interface RecoveryInfo {
  /** Generations already started by an operator resume. */
  resume_count: number
  /** Server-configured cap on resumes for this run. */
  resume_limit: number
  resumable: boolean
  /** Execution generation; 0 when no generation was ever opened. */
  generation: number
  lease_expires_at: string | null
  /** Workers whose committed checkpoint the next generation would adopt. */
  adopted_workers: string[]
  attempt_totals: Record<string, number>
  /** The interruption reason, i.e. the run's `error_code` while it is interrupted. */
  interrupted_reason: string | null
}

/**
 * One document branch of a rules 3.1.0 execution, as the page receives it: the bounded
 * counts the validator admits, never the raw answer.
 */
export interface PreciseBranchView {
  worker: string
  outcome: 'completed' | 'failed' | 'no_attempts_left' | 'stopped'
  reported: number
  support_failures: string[]
  attempts: number
  evidence_refs: number
  failure_code: string | null
}

/** What one rules 3.1.0 execution cost, recomputed from its persisted ledger.
 *
 * `physical_attempts` counts the attempt rows the Run opened; only `proven_sends` counts
 * the attempts whose own receipt proves a request was sent and answered, and the fields
 * that split corrections by kind are null on a Run recorded before the split existed.
 */
export interface PreciseMeteringView {
  physical_attempts: number
  first_sends: number
  corrections: number
  unknown_usage_attempts: number
  unreturned_attempts: number
  input_tokens_known: number | null
  output_tokens_known: number | null
  branch_latency_ms: Array<[string, number]>
  requested_model: string | null
  resolved_models: string[]
  attempts_opened: number | null
  proven_sends: number | null
  unresolved_attempts: number | null
  open_attempts: number | null
  contract_corrections: number | null
  citation_corrections: number | null
  transport_retries: number | null
  other_extra_attempts: number | null
  usage_complete: boolean | null
  reused_branches: number | null
  explicit_recoveries: number | null
}

/** One evidence location a rules 3.1.0 conclusion was bound to. */
export interface PreciseEvidenceView {
  source_id: string
  version: string
  locator: string
  content_sha256: string
}

/** One duty of a rules 3.1.0 decision, with the reason and the missing dimensions it carries. */
export interface PreciseFindingView {
  obligation_id: string
  state: 'supported' | 'violated' | 'unknown' | 'not_applicable'
  reason: string
  risk: string
  blocking: boolean
  critical: boolean
  remediable: boolean
  missing: string[]
  evidence: PreciseEvidenceView[]
}

/** One side of a business conflict: the proposition, its value and its own source. */
export interface PreciseConflictSideView {
  check_id: string
  field_path: string
  subject_id: string
  effective_at: string
  value: string | number | boolean | null
  source_id: string
  version: string
  locator: string
}

/** One conflict pair with both sides' identities, as the decision recorded them. */
export interface PreciseConflictView {
  pair_id: string
  kind: string
  relation: string
  left: PreciseConflictSideView
  right: PreciseConflictSideView
}

/** One trusted, redacted security fact: a fixed code, its origin and a hashed location. */
export interface PreciseSecurityEventView {
  code: string
  origin: string
  location_sha256: string
}

/** The recommendation one rules 3.1.0 decision reached, from the closed five-class set. */
export type PreciseRecommendation =
  | 'recommend_approve'
  | 'recommend_conditional'
  | 'recommend_reject'
  | 'insufficient_information'
  | 'needs_human_review'

/** What one rules 3.1.0 execution decided, as the page verifies it. */
export interface PreciseDecisionView {
  recommendation: PreciseRecommendation
  findings: PreciseFindingView[]
  violations: string[]
  unknown_obligations: string[]
  pipeline_failures: string[]
  security_events: string[]
  conflict_pairs: string[]
}

/** The rules 3.1.0 execution record the result view renders. */
export interface PreciseExecutionView {
  execution_version: 'review-execution.v2'
  rules_version: '3.1.0'
  stop_reason: string | null
  policy: PreciseBranchView
  contract: PreciseBranchView
  procurement: { status: string; tool_calls: number }
  metering: PreciseMeteringView
  decision: PreciseDecisionView
  conflicts: PreciseConflictView[]
  security: PreciseSecurityEventView[]
}

/**
 * The pre-3.1.0 record: validated at the boundary, not rendered by any component.  Its
 * shape is stated only as far as the page's own fixtures touch it (the branch results
 * and the summary the record restates).
 */
export interface FrozenExecutionView {
  execution_version: 'review-execution.v1'
  schema_version: string
  state_version: string
  branch_results: unknown[]
  summary: { run_id: string } | null
}

export type RunExecution = PreciseExecutionView | FrozenExecutionView | null

/** Whether one validated record is the rules 3.1.0 record the result view renders. */
export function isPreciseExecution(
  execution: RunExecution,
): execution is PreciseExecutionView {
  return (
    execution !== null && execution.execution_version === 'review-execution.v2'
  )
}

export interface ReviewRun {
  /**
   * The current `/api/v2` run view (`ReviewRunResponseV2`) carries no version
   * literal at all; it stays optional here so both the earlier v1-stamped and the
   * current payloads type-check. The UI never renders it.
   */
  api_version?: ApiVersionV1 | ApiVersionV2
  run_id: string
  trace_id: string
  status: ReviewStatus
  version: number
  revision: number
  execution_mode: ExecutionMode
  request: {
    schema_version: '1.0'
    case_id: string
    case_type: 'purchase_order' | 'contract_change'
    document_version: string
    purchase_order_id: string
    dataset_version: string
    /** Server-derived subject id of the credential that started the run. */
    initiator: string
  }
  summary: ReviewSummary | null
  /**
   * Non-null exactly when `summary` is non-null. Must be echoed back verbatim as
   * `expected_summary_checksum`; the client never computes it.
   */
  summary_checksum: string | null
  /**
   * The persisted execution record.  The runtime contract in `apiContract.ts` still
   * validates the closed envelope in either version; the rules 3.1.0 record is the one
   * the result view renders, because its metering, stop reason and deterministic branch
   * are what a human verifies before confirming.
   */
  execution: RunExecution
  branch_failures: BranchFailure[]
  evidence_refs: EvidenceRef[]
  business_facts: BusinessFact[]
  receipt: ConfirmationReceipt | null
  created_at: string
  updated_at: string
  trace_event_count: number
  audit_event_count: number
  error_code: string | null
  error_message: string | null
  /**
   * `null` means the backend returned no recovery information at all; the UI then
   * shows no resume offer and says so instead of guessing a budget.
   */
  recovery: RecoveryInfo | null
}

export interface ReviewListResponse {
  api_version: ApiVersionV2
  runs: ReviewRun[]
}

export interface RunEvent {
  schema_version: '1.0'
  event_id: string
  run_id: string
  trace_id: string
  event_type: string
  step: string
  occurred_at: string
  input_summary: Record<string, unknown>
  output_summary: Record<string, unknown>
  /**
   * Always present on the wire but never rendered, so it is declared optional
   * rather than required: the client validates what it consumes and ignores the
   * rest instead of pretending to own those fields.
   */
  input_checksum?: string
  output_checksum?: string
  source_versions: Record<string, string>
  duration_ms: number
  token_count: number | null
  cost_amount: string | number | null
  cost_currency: string | null
  redaction_status: 'not_required' | 'redacted'
  error: string | null
}

export interface TraceResponse {
  /** Shared contract: still stamped with the v1 literal. */
  api_version: ApiVersionV1
  trace: {
    schema_version: '1.0'
    trace_schema_version: 'trace.v1'
    trace_id: string
    run_id: string
    events: RunEvent[]
    complete: boolean
  }
}

/**
 * `approval-audit.v2` — the persisted product audit record.
 *
 * Verified against the running backend: every event carries a database-assigned
 * `sequence`, a tenant/Case anchor, the authorization revision, the before/after
 * revision, the result checksum and the idempotency **digest** (the raw key is never
 * returned).  Lifecycle and recovery events have no `action_type`, no actor and no
 * digest, so every one of those fields is nullable here.
 */
export type AuditEventClass =
  | 'approval'
  | 'recovery'
  | 'lifecycle'
  | 'access_denial'

export interface AuditEvent {
  schema_version: '1.0'
  audit_schema_version: 'approval-audit.v2'
  audit_id: string
  sequence: number
  tenant_id: string | null
  case_id: string | null
  run_id: string | null
  event_class: AuditEventClass
  action_type: 'confirm' | 'reject' | 'cancel' | null
  actor: string | null
  authorization_revision: number | null
  previous_status: ReviewStatus | null
  new_status: ReviewStatus | null
  previous_revision: number | null
  new_revision: number | null
  result_checksum: string | null
  /** Raw idempotency keys are withheld by the v2 projection. */
  idempotency_digest: string | null
  receipt_id: string | null
  api_version: 'review-api.v2'
  occurred_at: string
  detail_code: string | null
  simulation_only: true
  external_write_performed: false
}

export interface AuditResponse {
  api_version: ApiVersionV2
  audit_schema_version: 'approval-audit.v2'
  run_id: string
  events: AuditEvent[]
  /** Always present on the wire; the workbench does not render either field. */
  sequence_projection_version?: string
  legacy_sequence_anomalies?: string[]
}

export interface APIErrorBody {
  error: {
    api_version?: ApiVersionV1
    code: string
    message: string
    request_id: string
    details: Record<string, unknown>
  }
}
