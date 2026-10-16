/**
 * The committed summary the result view renders: the frozen projection every Run carries,
 * validated field by field so a malformed summary fails closed instead of being rendered.
 */

import {
  asRecord,
  CONFLICT_REASONS,
  EVIDENCE_SOURCE_KINDS,
  FACT_STATUSES,
  FINDING_ASSERTIONS,
  FINDING_STATUSES,
  HUMAN_REVIEW_STATUSES,
  RECOMMENDATIONS,
  reqArray,
  reqDateTime,
  reqEnum,
  reqIntegerOrNull,
  reqOpenObject,
  reqLiteral,
  reqString,
  reqStringArray,
  reqStringOrNull,
  RISK_LEVELS,
  SCHEMA_VERSION_1,
  WORKER_KINDS,
} from './apiPrimitives'
import type {
  BusinessFact,
  EvidenceRef,
  Finding,
  FindingConflict,
  ReviewRun,
} from './types'

/** The committed summary shape the result view renders. */
type Summary = NonNullable<ReviewRun['summary']>

export function validateEvidenceRef(value: unknown, path: string): EvidenceRef {
  const record = asRecord(value, path)
  return {
    schema_version: reqLiteral(record, 'schema_version', SCHEMA_VERSION_1, path),
    source_kind: reqEnum(record, 'source_kind', EVIDENCE_SOURCE_KINDS, path),
    source_id: reqString(record, 'source_id', path),
    version: reqString(record, 'version', path),
    page_number: reqIntegerOrNull(record, 'page_number', path, 1),
    section: reqStringOrNull(record, 'section', path),
    chunk_id: reqStringOrNull(record, 'chunk_id', path),
    content_checksum: reqString(record, 'content_checksum', path),
  }
}


export function validateBusinessFact(value: unknown, path: string): BusinessFact {
  const record = asRecord(value, path)
  return {
    schema_version: reqLiteral(record, 'schema_version', SCHEMA_VERSION_1, path),
    tool_name: reqString(record, 'tool_name', path),
    record_id: reqString(record, 'record_id', path),
    dataset_version: reqString(record, 'dataset_version', path),
    as_of: reqDateTime(record, 'as_of', path),
    status: reqEnum(record, 'status', FACT_STATUSES, path),
    result: reqOpenObject(record, 'result', path),
    failure_reason: reqStringOrNull(record, 'failure_reason', path),
    read_only: reqLiteral(record, 'read_only', true, path),
  }
}


export function validateFinding(value: unknown, path: string): Finding {
  const record = asRecord(value, path)
  return {
    schema_version: reqLiteral(record, 'schema_version', SCHEMA_VERSION_1, path),
    worker: reqEnum(record, 'worker', WORKER_KINDS, path),
    finding_key: reqString(record, 'finding_key', path),
    category: reqString(record, 'category', path),
    risk_level: reqEnum(record, 'risk_level', RISK_LEVELS, path),
    assertion: reqEnum(record, 'assertion', FINDING_ASSERTIONS, path),
    statement: reqString(record, 'statement', path),
    status: reqEnum(record, 'status', FINDING_STATUSES, path),
    evidence_refs: reqArray(record, 'evidence_refs', path).map((item, index) =>
      validateEvidenceRef(item, `${path}.evidence_refs[${index}]`),
    ),
    business_fact_refs: reqArray(record, 'business_fact_refs', path).map((item, index) =>
      validateBusinessFact(item, `${path}.business_fact_refs[${index}]`),
    ),
    missing_information: reqStringArray(record, 'missing_information', path),
    recommended_action: reqString(record, 'recommended_action', path),
  }
}

function validateConflict(value: unknown, path: string): FindingConflict {
  const record = asRecord(value, path)
  return {
    schema_version: reqLiteral(record, 'schema_version', SCHEMA_VERSION_1, path),
    finding_key: reqString(record, 'finding_key', path),
    reason: reqEnum(record, 'reason', CONFLICT_REASONS, path),
    assertions: reqStringArray(record, 'assertions', path),
    categories: reqStringArray(record, 'categories', path),
    statements: reqStringArray(record, 'statements', path),
  }
}

export function validateSummary(value: unknown, path: string): Summary {
  const record = asRecord(value, path)
  return {
    schema_version: reqLiteral(record, 'schema_version', SCHEMA_VERSION_1, path),
    run_id: reqString(record, 'run_id', path),
    case_id: reqString(record, 'case_id', path),
    rules_version: reqString(record, 'rules_version', path),
    findings: reqArray(record, 'findings', path).map((item, index) =>
      validateFinding(item, `${path}.findings[${index}]`),
    ),
    highest_risk: reqEnum(record, 'highest_risk', RISK_LEVELS, path),
    recommendation: reqEnum(record, 'recommendation', RECOMMENDATIONS, path),
    human_review_status: reqEnum(record, 'human_review_status', HUMAN_REVIEW_STATUSES, path),
    conflicts: reqArray(record, 'conflicts', path).map((item, index) =>
      validateConflict(item, `${path}.conflicts[${index}]`),
    ),
    missing_information: reqStringArray(record, 'missing_information', path),
    failure_reasons: reqStringArray(record, 'failure_reasons', path),
    is_final_decision: reqLiteral(record, 'is_final_decision', false, path),
  }
}
