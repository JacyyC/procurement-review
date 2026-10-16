/**
 * The rules 3.1.0 execution record the page validates before it renders anything.
 *
 * The record declares its own version, so it is judged field by field instead of being
 * stored as an opaque blob: the five-class recommendation, every duty with its reason and
 * missing dimensions, the evidence location each duty was bound to, both sides of every
 * conflict pair and every trusted security fact.  A field this record declares but the page
 * cannot read is a contract break, and an unknown version is refused by the caller.
 */

import { validateSummary } from './summaryContract'
import { invalidResponse } from './apiErrors'
import {
  asRecord,
  optionalInteger,
  present,
  reqArray,
  reqBoolean,
  reqBooleanOrNull,
  reqEnum,
  reqInteger,
  reqIntegerOrNull,
  reqLiteral,
  reqString,
  reqStringArray,
  reqStringOrNull,
  type PlainRecord,
} from './apiPrimitives'
import type {
  PreciseBranchView,
  PreciseConflictSideView,
  PreciseConflictView,
  PreciseDecisionView,
  PreciseEvidenceView,
  PreciseExecutionView,
  PreciseFindingView,
  PreciseMeteringView,
  PreciseSecurityEventView,
} from './types'

/** The two execution records a run may carry, and the modes a workbench may run in. */
const EXECUTION_VERSION_V2 = 'review-execution.v2'
const RULES_VERSION_V31 = '3.1.0'
/** The closed five-class recommendation set a rules 3.1.0 decision may reach. */
const PRECISE_RECOMMENDATIONS = [
  'recommend_approve',
  'recommend_conditional',
  'recommend_reject',
  'insufficient_information',
  'needs_human_review',
] as const

function validateEvidenceIdentity(value: unknown, path: string): PreciseEvidenceView {
  const item = asRecord(value, path)
  const checksum = reqString(item, 'content_sha256', path)
  if (!/^[0-9a-f]{64}$/.test(checksum)) {
    throw invalidResponse(`${path}.content_sha256`, 'a bare sha256 digest')
  }
  return {
    source_id: reqString(item, 'source_id', path),
    version: reqString(item, 'version', path),
    locator: reqString(item, 'locator', path),
    content_sha256: checksum,
  }
}

function validatePreciseFinding(value: unknown, path: string): PreciseFindingView {
  const item = asRecord(value, path)
  const state = reqEnum(
    item,
    'state',
    ['supported', 'violated', 'unknown', 'not_applicable'] as const,
    path,
  )
  reqLiteral(item, 'rules_version', RULES_VERSION_V31, path)
  return {
    obligation_id: reqString(item, 'obligation_id', path),
    state,
    reason: reqString(item, 'reason', path),
    risk: reqString(item, 'risk', path),
    blocking: reqBoolean(item, 'blocking', path),
    critical: reqBoolean(item, 'critical', path),
    remediable: reqBoolean(item, 'remediable', path),
    missing: reqStringArray(item, 'missing', path),
    evidence: reqArray(item, 'evidence', path).map((entry, index) =>
      validateEvidenceIdentity(entry, `${path}.evidence[${index}]`),
    ),
  }
}

/** One trusted, redacted security fact; the location travels as a digest, never a payload. */
function validateSecurityEvent(value: unknown, path: string): PreciseSecurityEventView {
  const item = asRecord(value, path)
  const location = reqString(item, 'location_sha256', path)
  if (!/^[0-9a-f]{64}$/.test(location)) {
    throw invalidResponse(`${path}.location_sha256`, 'a bare sha256 digest')
  }
  reqLiteral(item, 'forbidden_export_succeeded', false, path)
  return {
    code: reqString(item, 'code', path),
    origin: reqEnum(item, 'origin', ['input', 'output', 'transport'] as const, path),
    location_sha256: location,
  }
}

/** One side of a conflict: the proposition, its value and the source it came from. */
function validateConflictSide(value: unknown, path: string): PreciseConflictSideView {
  const item = asRecord(value, path)
  const proposition = asRecord(present(item, 'proposition', path), `${path}.proposition`)
  const evidence = reqArray(item, 'evidence', path)
  if (evidence.length === 0) {
    throw invalidResponse(`${path}.evidence`, 'one located side of a conflict')
  }
  const source = validateEvidenceIdentity(evidence[0], `${path}.evidence[0]`)
  const raw = present(item, 'value', path)
  if (raw !== null && !['string', 'number', 'boolean'].includes(typeof raw)) {
    throw invalidResponse(`${path}.value`, 'a scalar value or null')
  }
  return {
    check_id: reqString(proposition, 'check_id', `${path}.proposition`),
    field_path: reqString(proposition, 'field_path', `${path}.proposition`),
    subject_id: reqString(proposition, 'subject_id', `${path}.proposition`),
    effective_at: reqString(proposition, 'effective_at', `${path}.proposition`),
    value: raw as string | number | boolean | null,
    source_id: source.source_id,
    version: source.version,
    locator: source.locator,
  }
}

function validateConflictPair(value: unknown, path: string): PreciseConflictView {
  const item = asRecord(value, path)
  return {
    pair_id: reqString(item, 'pair_id', path),
    kind: reqString(item, 'kind', path),
    relation: reqString(item, 'relation', path),
    left: validateConflictSide(present(item, 'left', path), `${path}.left`),
    right: validateConflictSide(present(item, 'right', path), `${path}.right`),
  }
}

function validatePreciseExecution(
  record: PlainRecord,
  path: string,
): PreciseExecutionView {
  const branchOutcomes = ['completed', 'failed', 'no_attempts_left', 'stopped'] as const
  const branch = (value: unknown, branchPath: string) => {
    const item = asRecord(value, branchPath)
    const reported = reqArray(item, 'reported', branchPath)
    reqEnum(item, 'worker', ['policy', 'contract'], branchPath)
    const outcome = reqEnum(item, 'outcome', branchOutcomes, branchPath)
    reqArray(item, 'spans', branchPath)
    const supportFailures = reqStringArray(item, 'support_failures', branchPath)
    reqStringOrNull(item, 'request_sha256', branchPath)
    reqStringOrNull(item, 'answer_sha256', branchPath)
    const attempts = reqArray(item, 'attempts', branchPath)
    const evidenceRefs = reqArray(item, 'evidence_refs', branchPath)
    const failureCode = reqStringOrNull(item, 'failure_code', branchPath)
    const view: PreciseBranchView = {
      worker: String(item.worker),
      outcome,
      reported: reported.length,
      support_failures: supportFailures,
      attempts: attempts.length,
      evidence_refs: evidenceRefs.length,
      failure_code: failureCode,
    }
    return view
  }
  const decision = asRecord(present(record, 'decision', path), `${path}.decision`)
  reqLiteral(decision, 'rules_version', RULES_VERSION_V31, `${path}.decision`)
  const findings = reqArray(decision, 'findings', `${path}.decision`).map(
    (entry, index) => validatePreciseFinding(entry, `${path}.decision.findings[${index}]`),
  )
  const recommendation = reqEnum(
    decision,
    'recommendation',
    PRECISE_RECOMMENDATIONS,
    `${path}.decision`,
  )
  const decisionView: PreciseDecisionView = {
    recommendation,
    findings,
    violations: reqStringArray(decision, 'violations', `${path}.decision`),
    unknown_obligations: reqStringArray(
      decision,
      'unknown_obligations',
      `${path}.decision`,
    ),
    pipeline_failures: reqStringArray(decision, 'pipeline_failures', `${path}.decision`),
    security_events: reqStringArray(decision, 'security_events', `${path}.decision`),
    conflict_pairs: reqStringArray(decision, 'conflict_pairs', `${path}.decision`),
  }
  const security = reqArray(record, 'security', path).map((entry, index) =>
    validateSecurityEvent(entry, `${path}.security[${index}]`),
  )
  const conflicts = reqArray(record, 'conflicts', path).map((entry, index) =>
    validateConflictPair(entry, `${path}.conflicts[${index}]`),
  )
  const policy = branch(present(record, 'policy', path), `${path}.policy`)
  const contract = branch(present(record, 'contract', path), `${path}.contract`)
  if (policy.outcome === undefined || contract.outcome === undefined) {
    throw invalidResponse(`${path}.policy`, 'one record per document branch')
  }
  const procurement = asRecord(
    present(record, 'procurement', path),
    `${path}.procurement`,
  )
  reqLiteral(procurement, 'worker', 'procurement_data', `${path}.procurement`)
  const procurementStatus = reqString(procurement, 'status', `${path}.procurement`)
  const toolCalls = reqInteger(procurement, 'tool_calls', `${path}.procurement`, 0)
  reqInteger(procurement, 'model_calls', `${path}.procurement`, 0)
  if (procurement.model_calls !== 0) {
    throw invalidResponse(
      `${path}.procurement.model_calls`,
      'the deterministic branch calls no model',
    )
  }
  const meterPath = `${path}.metering`
  const metering = asRecord(present(record, 'metering', path), meterPath)
  const counter = (key: string) => reqInteger(metering, key, meterPath, 0)
  const latency = reqArray(metering, 'branch_latency_ms', meterPath)
  const branchLatency = latency.map((entry, index) => {
    const pairPath = `${meterPath}.branch_latency_ms[${index}]`
    const pair = Array.isArray(entry) ? entry : []
    if (pair.length !== 2 || typeof pair[0] !== 'string' || !Number.isInteger(pair[1])) {
      throw invalidResponse(pairPath, 'a [worker, milliseconds] pair')
    }
    return [pair[0], Number(pair[1])] as [string, number]
  })
  const meteringView: PreciseMeteringView = {
    physical_attempts: counter('physical_attempts'),
    first_sends: counter('first_sends'),
    corrections: counter('corrections'),
    unknown_usage_attempts: counter('unknown_usage_attempts'),
    unreturned_attempts: counter('unreturned_attempts'),
    input_tokens_known: reqIntegerOrNull(
      metering,
      'input_tokens_known',
      meterPath,
      0,
    ),
    output_tokens_known: reqIntegerOrNull(
      metering,
      'output_tokens_known',
      meterPath,
      0,
    ),
    branch_latency_ms: branchLatency,
    requested_model: reqStringOrNull(metering, 'requested_model', meterPath),
    resolved_models: reqStringArray(metering, 'resolved_models', meterPath),
    attempts_opened: optionalInteger(metering, 'attempts_opened', meterPath, 0),
    proven_sends: optionalInteger(metering, 'proven_sends', meterPath, 0),
    unresolved_attempts: optionalInteger(metering, 'unresolved_attempts', meterPath, 0),
    open_attempts: optionalInteger(metering, 'open_attempts', meterPath, 0),
    contract_corrections: optionalInteger(metering, 'contract_corrections', meterPath, 0),
    citation_corrections: optionalInteger(metering, 'citation_corrections', meterPath, 0),
    transport_retries: optionalInteger(metering, 'transport_retries', meterPath, 0),
    other_extra_attempts: optionalInteger(metering, 'other_extra_attempts', meterPath, 0),
    usage_complete: reqBooleanOrNull(metering, 'usage_complete', meterPath),
    reused_branches: optionalInteger(metering, 'reused_branches', meterPath, 0),
    explicit_recoveries: optionalInteger(metering, 'explicit_recoveries', meterPath, 0),
  }
  const summary = present(record, 'summary', path)
  if (summary === null) {
    throw invalidResponse(`${path}.summary`, 'a committed precise run summary')
  }
  validateSummary(summary, `${path}.summary`)
  return {
    execution_version: EXECUTION_VERSION_V2,
    decision: decisionView,
    conflicts,
    security,
    rules_version: RULES_VERSION_V31,
    stop_reason: reqStringOrNull(record, 'stop_reason', path),
    policy,
    contract,
    procurement: { status: procurementStatus, tool_calls: toolCalls },
    metering: meteringView,
  }
}


export { EXECUTION_VERSION_V2, RULES_VERSION_V31, validatePreciseExecution }
