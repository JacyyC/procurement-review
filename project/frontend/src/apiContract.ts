/**
 * Runtime contract of every `review-api.v2` response the workbench consumes.
 *
 * Route chosen in FIX-01C (see the work-item Log for the full comparison): the
 * backend Pydantic models are the source of truth for the *shape*, but the generated
 * OpenAPI document is not a faithful enough runtime validator on its own — its
 * `required` lists omit every defaulted field that the serializer always emits, it
 * carries no error-envelope schema at all, and it leaves the session role/scope sets
 * as unconstrained `string` arrays.  The client therefore keeps a closed hand-written
 * narrow validator, cross-checked field by field against the live `review-api.v2`
 * payload and the backend enums (`contracts.py`, `worker_contracts.py`,
 * `audit_contracts.py`, `review_graph.py`).
 *
 * Field-level rules — what one field may be — live in `apiPrimitives.ts`; this module
 * only says which fields each record and payload has, so the two kinds of change
 * never land in the same edit.  Rules preserved here:
 *
 *  - every field a component reads is validated before React sees it;
 *  - no `as` coercion, no default filling, no "missing field becomes an empty array";
 *  - legal `null`, empty arrays and empty open maps stay legal;
 *  - a response must describe the resource that was requested;
 *  - fields the client does not consume are ignored, never copied into state.
 *
 * Invariants are enforced where violating them would make the UI misrender or
 * misbehave (summary/checksum agreement, nested identities, closed execution shape).
 * Invariants the UI already handles explicitly — an evidence ref with no locator, an
 * empty trace event list, a `source_versions` map with no entries — are deliberately
 * not enforced, because turning a benign server state into `invalid_response` would
 * replace a readable hint with an error.  That choice is recorded, not implicit.
 *
 * Drift detection: `apiContract.test.ts` pins the accepted shape, and the Playwright
 * suite runs the real API, so a backend contract change this file has not learned
 * about fails loudly as `invalid_response` instead of rendering a half-empty view.
 */

import {
  asRecord,
  enumValue,
  present,
  reqArray,
  reqBoolean,
  reqCostAmount,
  reqCostCurrency,
  reqDateTime,
  reqDateTimeOrNull,
  reqEnum,
  reqEnumArray,
  reqEnumOrNull,
  reqIdentity,
  reqInteger,
  reqIntegerOrNull,
  reqLiteral,
  reqOpenObject,
  reqString,
  reqStringMap,
  reqStringOrNull,
  AUDIT_ACTION_TYPES,
  AUDIT_EVENT_CLASSES,
  AUDIT_SCHEMA_VERSION,
  CASE_TYPES,
  EXECUTION_VERSION,
  EXECUTION_WORKER_COUNT,
  RECEIPT_DECISIONS,
  RECOMMENDATIONS,
  REDACTION_STATUSES,
  REVIEW_SCENARIOS,
  REVIEW_SCOPES,
  REVIEW_STATUSES,
  RUN_API_VERSIONS,
  SCHEMA_VERSION_1,
  SESSION_ROLES,
  STATE_VERSION,
  TRACE_SCHEMA_VERSION,
  WORKER_KINDS,
  WORKER_OUTCOMES,
  WORKER_RESULT_VERSION,
} from './apiPrimitives'
import { invalidResponse } from './apiErrors'
import { validateBusinessFact, validateEvidenceRef } from './summaryContract'
import { validateFinding, validateSummary } from './summaryContract'
import {
  EXECUTION_VERSION_V2,
  validatePreciseExecution,
} from './preciseExecutionContract'

/** The modes a workbench may run in. */
const EXECUTION_MODES = ['test_only', 'real'] as const
import { API_VERSION_V1, API_VERSION_V2 } from './types'
import type { PlainRecord } from './apiPrimitives'
import type {
  AuditEvent,
  AuditResponse,
  BranchFailure,
  FrozenExecutionView,
  CaseListResponse,
  CaseOption,
  ConfirmationReceipt,
  RecoveryInfo,
  PreciseExecutionView,
  ReviewListResponse,
  ReviewRun,
  RunEvent,
  SessionInfo,
  TraceResponse,
} from './types'

/**
 * The full summary contract, as served both at the top level of a Run and inside
 * `execution`.  One alias keeps the two call sites provably identical.
 */

/** What one validated execution record is: the precise record, or the frozen one. */
export type ValidatedExecution = PreciseExecutionView | FrozenExecutionView

// ---------------------------------------------------------------- leaf contracts



function validateReceipt(value: unknown, path: string): ConfirmationReceipt {
  const record = asRecord(value, path)
  // `ReceiptViewV2` is an APIContractV2: it carries `api_version` and deliberately no
  // `schema_version`.  Requiring one would reject every real receipt.
  return {
    api_version: reqLiteral(record, 'api_version', API_VERSION_V2, path),
    receipt_id: reqString(record, 'receipt_id', path),
    run_id: reqString(record, 'run_id', path),
    case_id: reqString(record, 'case_id', path),
    human_actor: reqString(record, 'human_actor', path),
    decision: reqEnum(record, 'decision', RECEIPT_DECISIONS, path),
    reviewed_recommendation: reqEnum(record, 'reviewed_recommendation', RECOMMENDATIONS, path),
    recorded_at: reqDateTime(record, 'recorded_at', path),
    idempotency_key_digest: reqString(record, 'idempotency_key_digest', path),
    simulation_status: reqLiteral(record, 'simulation_status', 'local_simulation_recorded', path),
    simulation_mode: reqLiteral(record, 'simulation_mode', 'local_test_only', path),
    external_write_performed: reqLiteral(record, 'external_write_performed', false, path),
  }
}

function validateAttemptTotals(
  record: PlainRecord,
  path: string,
): Record<string, number> {
  const totals = asRecord(present(record, 'attempt_totals', path), `${path}.attempt_totals`)
  const attemptTotals: Record<string, number> = {}
  for (const [worker, total] of Object.entries(totals)) {
    if (!(WORKER_KINDS as readonly string[]).includes(worker)) {
      throw invalidResponse(`${path}.attempt_totals.${worker}`, `one of ${WORKER_KINDS.join('/')}`)
    }
    if (typeof total !== 'number' || !Number.isInteger(total) || total < 0) {
      throw invalidResponse(`${path}.attempt_totals.${worker}`, 'an integer >= 0')
    }
    attemptTotals[worker] = total
  }
  return attemptTotals
}

function validateRecovery(value: unknown, path: string): RecoveryInfo {
  const record = asRecord(value, path)
  // Fields are validated in declaration order, so the reported path is the first
  // thing that is actually wrong rather than an arbitrary later field.
  return {
    resume_count: reqInteger(record, 'resume_count', path, 0),
    resume_limit: reqInteger(record, 'resume_limit', path, 0),
    resumable: reqBoolean(record, 'resumable', path),
    generation: reqInteger(record, 'generation', path, 0),
    lease_expires_at: reqDateTimeOrNull(record, 'lease_expires_at', path),
    adopted_workers: reqEnumArray(record, 'adopted_workers', WORKER_KINDS, path),
    attempt_totals: validateAttemptTotals(record, path),
    interrupted_reason: reqStringOrNull(record, 'interrupted_reason', path),
  }
}

function validateBranchFailure(value: unknown, path: string): BranchFailure {
  const record = asRecord(value, path)
  // Shared with the retired v1 contract, so the literal inside a v2 payload is v1.
  return {
    api_version: reqLiteral(record, 'api_version', API_VERSION_V1, path),
    worker: reqEnum(record, 'worker', WORKER_KINDS, path),
    code: reqString(record, 'code', path),
    detail: reqString(record, 'detail', path),
  }
}

/**
 * `execution` is the persisted worker ledger.  No component renders it, but it is
 * part of the same 2xx body, so its envelope, its exact worker cardinality and every
 * branch record are still validated rather than stored as an opaque blob.
 */
function validateExecution(value: unknown, path: string): ValidatedExecution {
  const record = asRecord(value, path)
  const version = reqString(record, 'execution_version', path)
  if (version === EXECUTION_VERSION_V2) {
    return validatePreciseExecution(record, path)
  }
  if (version !== EXECUTION_VERSION) {
    throw invalidResponse(`${path}.execution_version`, 'a known execution version')
  }
  return validateFrozenExecution(record, path)
}

/**
 * The rules 3.1.0 precise record.  It declares its own version, so an unknown version is
 * refused rather than being stored as an opaque blob; every field a component or an
 * auditor may read is validated, including the metering a real run reports and the
 * stop reason an explicit control path wrote.
 */

function validateFrozenExecution(
  record: PlainRecord,
  path: string,
): FrozenExecutionView {
  reqLiteral(record, 'schema_version', SCHEMA_VERSION_1, path)
  reqLiteral(record, 'execution_version', EXECUTION_VERSION, path)
  reqLiteral(record, 'state_version', STATE_VERSION, path)
  const branches = reqArray(record, 'branch_results', path)
  if (branches.length !== EXECUTION_WORKER_COUNT) {
    throw invalidResponse(
      `${path}.branch_results`,
      `exactly ${EXECUTION_WORKER_COUNT} worker results`,
    )
  }
  branches.forEach((item, index) => {
    const branchPath = `${path}.branch_results[${index}]`
    const branch = asRecord(item, branchPath)
    reqLiteral(branch, 'schema_version', SCHEMA_VERSION_1, branchPath)
    reqLiteral(branch, 'result_version', WORKER_RESULT_VERSION, branchPath)
    reqEnum(branch, 'worker', WORKER_KINDS, branchPath)
    reqEnum(branch, 'outcome', WORKER_OUTCOMES, branchPath)
    const findings = reqArray(branch, 'findings', branchPath)
    if (findings.length === 0) {
      throw invalidResponse(`${branchPath}.findings`, 'at least one worker finding')
    }
    findings.forEach((finding, findingIndex) =>
      validateFinding(finding, `${branchPath}.findings[${findingIndex}]`),
    )
    reqStringOrNull(branch, 'failure_code', branchPath)
    reqStringOrNull(branch, 'failure_detail', branchPath)
  })
  const summary = present(record, 'summary', path)
  if (summary !== null) validateSummary(summary, `${path}.summary`)
  return {
    schema_version: SCHEMA_VERSION_1,
    execution_version: EXECUTION_VERSION,
    state_version: STATE_VERSION,
    branch_results: branches.map((item, index) =>
      validateBranchResult(item, `${path}.branch_results[${index}]`),
    ),
    summary: summary === null ? null : validateSummary(summary, `${path}.summary`),
  }
}

function validateBranchResult(value: unknown, path: string): Record<string, unknown> {
  const record = asRecord(value, path)
  return {
    schema_version: reqLiteral(record, 'schema_version', SCHEMA_VERSION_1, path),
    result_version: reqLiteral(record, 'result_version', WORKER_RESULT_VERSION, path),
    worker: reqEnum(record, 'worker', WORKER_KINDS, path),
    outcome: reqEnum(record, 'outcome', WORKER_OUTCOMES, path),
    findings: reqArray(record, 'findings', path).map((item, index) =>
      validateFinding(item, `${path}.findings[${index}]`),
    ),
    failure_code: reqStringOrNull(record, 'failure_code', path),
    failure_detail: reqStringOrNull(record, 'failure_detail', path),
  }
}

// ------------------------------------------------------------- top-level payloads

export interface RunIdentity {
  /** When set, the payload must describe exactly this run. */
  expectRunId?: string
  /**
   * When set, the payload's `request.case_id` must be exactly this Case.
   *
   * The Case is frozen by the caller *before* the request is sent, so a create
   * response that describes another Case is a bad success payload — never a new
   * review to adopt, and never something to re-derive from current state.
   */
  expectCaseId?: string
}

export function validateRun(
  value: unknown,
  path = 'run',
  identity: RunIdentity = {},
): ReviewRun {
  const record = asRecord(value, path)
  const runId = identity.expectRunId
    ? reqIdentity(record, 'run_id', identity.expectRunId, path)
    : reqString(record, 'run_id', path)

  const request = asRecord(present(record, 'request', path), `${path}.request`)
  if (identity.expectCaseId !== undefined) {
    reqIdentity(request, 'case_id', identity.expectCaseId, `${path}.request`)
  }

  // One rule, no guessing: the checksum the human Gate echoes exists exactly when a
  // summary exists.  A payload that breaks it would silently disable confirm/reject,
  // so it is rejected instead of patched with a default.
  const summaryValue = present(record, 'summary', path)
  const checksum = reqStringOrNull(record, 'summary_checksum', path)
  if ((summaryValue === null) !== (checksum === null)) {
    throw invalidResponse(
      `${path}.summary_checksum`,
      'a checksum exactly when summary is present',
      'The Review API returned a malformed response: summary and summary_checksum disagree.',
    )
  }
  const summary = summaryValue === null ? null : validateSummary(summaryValue, `${path}.summary`)
  if (summary && summary.run_id !== runId) {
    throw invalidResponse(
      `${path}.summary.run_id`,
      'the run it belongs to',
      'The Review API returned a malformed response: summary.run_id is not the run it was returned for.',
    )
  }

  const receiptValue = present(record, 'receipt', path)
  const receipt = receiptValue === null ? null : validateReceipt(receiptValue, `${path}.receipt`)
  if (receipt && receipt.run_id !== runId) {
    throw invalidResponse(
      `${path}.receipt.run_id`,
      'the run it belongs to',
      'The Review API returned a malformed response: receipt.run_id is not the run it was returned for.',
    )
  }

  const executionValue = present(record, 'execution', path)
  const recoveryValue = present(record, 'recovery', path)

  return {
    ...(record.api_version === undefined
      ? {}
      : { api_version: enumValue(record.api_version, RUN_API_VERSIONS, `${path}.api_version`) }),
    run_id: runId,
    trace_id: reqString(record, 'trace_id', path),
    status: reqEnum(record, 'status', REVIEW_STATUSES, path),
    version: reqInteger(record, 'version', path, 1),
    revision: reqInteger(record, 'revision', path, 1),
    execution_mode: reqEnum(record, 'execution_mode', EXECUTION_MODES, path),
    request: {
      schema_version: reqLiteral(request, 'schema_version', SCHEMA_VERSION_1, `${path}.request`),
      case_id: reqString(request, 'case_id', `${path}.request`),
      case_type: reqEnum(request, 'case_type', CASE_TYPES, `${path}.request`),
      document_version: reqString(request, 'document_version', `${path}.request`),
      purchase_order_id: reqString(request, 'purchase_order_id', `${path}.request`),
      dataset_version: reqString(request, 'dataset_version', `${path}.request`),
      initiator: reqString(request, 'initiator', `${path}.request`),
    },
    summary,
    summary_checksum: checksum,
    execution: executionValue === null ? null : validateExecution(executionValue, `${path}.execution`),
    branch_failures: reqArray(record, 'branch_failures', path).map((item, index) =>
      validateBranchFailure(item, `${path}.branch_failures[${index}]`),
    ),
    evidence_refs: reqArray(record, 'evidence_refs', path).map((item, index) =>
      validateEvidenceRef(item, `${path}.evidence_refs[${index}]`),
    ),
    business_facts: reqArray(record, 'business_facts', path).map((item, index) =>
      validateBusinessFact(item, `${path}.business_facts[${index}]`),
    ),
    receipt,
    created_at: reqDateTime(record, 'created_at', path),
    updated_at: reqDateTime(record, 'updated_at', path),
    trace_event_count: reqInteger(record, 'trace_event_count', path, 0),
    audit_event_count: reqInteger(record, 'audit_event_count', path, 0),
    error_code: reqStringOrNull(record, 'error_code', path),
    error_message: reqStringOrNull(record, 'error_message', path),
    recovery: recoveryValue === null ? null : validateRecovery(recoveryValue, `${path}.recovery`),
  }
}

export function validateSession(value: unknown, path = 'session'): SessionInfo {
  const record = asRecord(value, path)
  return {
    api_version: reqLiteral(record, 'api_version', API_VERSION_V2, path),
    subject_id: reqString(record, 'subject_id', path),
    tenant_id: reqString(record, 'tenant_id', path),
    credential_id: reqString(record, 'credential_id', path),
    authorization_revision: reqInteger(record, 'authorization_revision', path, 1),
    roles: reqEnumArray(record, 'roles', SESSION_ROLES, path),
    scopes: reqEnumArray(record, 'scopes', REVIEW_SCOPES, path),
    issued_at: reqDateTime(record, 'issued_at', path),
    expires_at: reqDateTime(record, 'expires_at', path),
  }
}

export function validateCaseOption(value: unknown, path: string): CaseOption {
  const record = asRecord(value, path)
  return {
    api_version: reqLiteral(record, 'api_version', API_VERSION_V2, path),
    case_id: reqString(record, 'case_id', path),
    case_version: reqString(record, 'case_version', path),
    label: reqString(record, 'label', path),
    case_type: reqEnum(record, 'case_type', CASE_TYPES, path),
    scenario: reqEnum(record, 'scenario', REVIEW_SCENARIOS, path),
    document_version: reqString(record, 'document_version', path),
    dataset_version: reqString(record, 'dataset_version', path),
    index_version: reqString(record, 'index_version', path),
    rules_version: reqString(record, 'rules_version', path),
  }
}

export function validateCases(value: unknown, path = 'cases_response'): CaseListResponse {
  const record = asRecord(value, path)
  return {
    api_version: reqLiteral(record, 'api_version', API_VERSION_V2, path),
    cases: reqArray(record, 'cases', path).map((item, index) =>
      validateCaseOption(item, `${path}.cases[${index}]`),
    ),
    execution_mode: reqEnum(record, 'execution_mode', EXECUTION_MODES, path),
  }
}

export function validateReviewList(value: unknown, path = 'review_list'): ReviewListResponse {
  const record = asRecord(value, path)
  return {
    api_version: reqLiteral(record, 'api_version', API_VERSION_V2, path),
    runs: reqArray(record, 'runs', path).map((item, index) =>
      validateRun(item, `${path}.runs[${index}]`),
    ),
  }
}

/**
 * One Trace event, which must belong to the envelope that carries it.
 *
 * The envelope identity is validated first and then handed down: an event whose `run_id` or
 * `trace_id` names something else is a payload describing another object, and rendering it
 * under this Run's heading is exactly the defect this rule removes (FIX-04B / C-AUD-05).
 * The whole payload is rejected — bad events are never filtered out and identifiers are
 * never rewritten.
 */
function validateRunEvent(
  value: unknown,
  path: string,
  parentRunId: string,
  parentTraceId: string,
): RunEvent {
  const record = asRecord(value, path)
  const eventRunId = reqIdentity(record, 'run_id', parentRunId, path)
  const eventTraceId = reqIdentity(record, 'trace_id', parentTraceId, path)
  // The backend model validator requires the amount and its currency together.
  const costAmount = reqCostAmount(record, 'cost_amount', path)
  const costCurrency = reqCostCurrency(record, 'cost_currency', path)
  if ((costAmount === null) !== (costCurrency === null)) {
    throw invalidResponse(
      `${path}.cost_currency`,
      'a currency exactly when cost_amount is present',
      'The Review API returned a malformed response: cost_amount and cost_currency disagree.',
    )
  }
  return {
    schema_version: reqLiteral(record, 'schema_version', SCHEMA_VERSION_1, path),
    event_id: reqString(record, 'event_id', path),
    run_id: eventRunId,
    trace_id: eventTraceId,
    event_type: reqString(record, 'event_type', path),
    step: reqString(record, 'step', path),
    occurred_at: reqDateTime(record, 'occurred_at', path),
    input_summary: reqOpenObject(record, 'input_summary', path),
    output_summary: reqOpenObject(record, 'output_summary', path),
    source_versions: reqStringMap(record, 'source_versions', path),
    duration_ms: reqInteger(record, 'duration_ms', path, 0),
    token_count: reqIntegerOrNull(record, 'token_count', path, 0),
    cost_amount: costAmount,
    cost_currency: costCurrency,
    redaction_status: reqEnum(record, 'redaction_status', REDACTION_STATUSES, path),
    error: reqStringOrNull(record, 'error', path),
  }
}

/**
 * A Trace response.
 *
 * `expectRunId` / `expectTraceId` are supplied by the caller when it already knows which
 * object it asked for; without them the envelope and its events are still required to agree
 * with each other, because a payload is only self-consistent if its own parts match.
 */
export function validateTrace(
  value: unknown,
  path = 'trace_response',
  expectRunId?: string,
  expectTraceId?: string,
): TraceResponse {
  const record = asRecord(value, path)
  const trace = asRecord(present(record, 'trace', path), `${path}.trace`)
  const tracePath = `${path}.trace`
  const runId = expectRunId
    ? reqIdentity(trace, 'run_id', expectRunId, tracePath)
    : reqString(trace, 'run_id', tracePath)
  const traceId = expectTraceId
    ? reqIdentity(trace, 'trace_id', expectTraceId, tracePath)
    : reqString(trace, 'trace_id', tracePath)
  return {
    api_version: reqLiteral(record, 'api_version', API_VERSION_V1, path),
    trace: {
      schema_version: reqLiteral(trace, 'schema_version', SCHEMA_VERSION_1, tracePath),
      trace_schema_version: reqLiteral(
        trace,
        'trace_schema_version',
        TRACE_SCHEMA_VERSION,
        tracePath,
      ),
      trace_id: traceId,
      run_id: runId,
      events: reqArray(trace, 'events', tracePath).map((item, index) =>
        validateRunEvent(item, `${tracePath}.events[${index}]`, runId, traceId),
      ),
      complete: reqBoolean(trace, 'complete', tracePath),
    },
  }
}

/**
 * One audit event of a Run-scoped ledger.
 *
 * `GET /reviews/{id}/audit` loads `SELECT * FROM review_audit_event WHERE run_id = :run_id`,
 * and every writer inserts the audited Run as `run_id`, so a member of that ledger is
 * neither another Run nor `null`.  Requiring equality here — rather than inheriting the
 * contract's wider nullable field — is what keeps an unanchored event from rendering as
 * this Run's (FIX-04B / C-AUD-05; the endpoint fact is recorded in the FIX-04 evidence).
 */
function validateAuditEvent(value: unknown, path: string, parentRunId: string): AuditEvent {
  const record = asRecord(value, path)
  return {
    schema_version: reqLiteral(record, 'schema_version', SCHEMA_VERSION_1, path),
    audit_schema_version: reqLiteral(record, 'audit_schema_version', AUDIT_SCHEMA_VERSION, path),
    audit_id: reqString(record, 'audit_id', path),
    sequence: reqInteger(record, 'sequence', path, 1),
    tenant_id: reqStringOrNull(record, 'tenant_id', path),
    case_id: reqStringOrNull(record, 'case_id', path),
    run_id: reqIdentity(record, 'run_id', parentRunId, path),
    event_class: reqEnum(record, 'event_class', AUDIT_EVENT_CLASSES, path),
    action_type: reqEnumOrNull(record, 'action_type', AUDIT_ACTION_TYPES, path),
    actor: reqStringOrNull(record, 'actor', path),
    authorization_revision: reqIntegerOrNull(record, 'authorization_revision', path, 1),
    previous_status: reqEnumOrNull(record, 'previous_status', REVIEW_STATUSES, path),
    new_status: reqEnumOrNull(record, 'new_status', REVIEW_STATUSES, path),
    previous_revision: reqIntegerOrNull(record, 'previous_revision', path, 0),
    new_revision: reqIntegerOrNull(record, 'new_revision', path, 1),
    result_checksum: reqStringOrNull(record, 'result_checksum', path),
    idempotency_digest: reqStringOrNull(record, 'idempotency_digest', path),
    receipt_id: reqStringOrNull(record, 'receipt_id', path),
    api_version: reqLiteral(record, 'api_version', API_VERSION_V2, path),
    occurred_at: reqDateTime(record, 'occurred_at', path),
    detail_code: reqStringOrNull(record, 'detail_code', path),
    simulation_only: reqLiteral(record, 'simulation_only', true, path),
    external_write_performed: reqLiteral(record, 'external_write_performed', false, path),
  }
}

export function validateAudit(
  value: unknown,
  path = 'audit_response',
  expectRunId?: string,
): AuditResponse {
  const record = asRecord(value, path)
  const runId = expectRunId
    ? reqIdentity(record, 'run_id', expectRunId, path)
    : reqString(record, 'run_id', path)
  return {
    api_version: reqLiteral(record, 'api_version', API_VERSION_V2, path),
    audit_schema_version: reqLiteral(record, 'audit_schema_version', AUDIT_SCHEMA_VERSION, path),
    run_id: runId,
    events: reqArray(record, 'events', path).map((item, index) =>
      validateAuditEvent(item, `${path}.events[${index}]`, runId),
    ),
  }
}
