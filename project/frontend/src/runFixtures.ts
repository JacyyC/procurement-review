/**
 * Shared Run payload fixtures for the frontend test suites.
 *
 * These builders describe the payload the API actually serves, so a test that mocks
 * `fetch` still exercises the real response contract:
 *
 *  - `execution` is a closed `review-execution.v1` record — one result per worker,
 *    each carrying at least one finding, plus a copy of the summary.  Every success
 *    payload that contains a summary contains it, so a fixture that omits it (or
 *    sends `{}`) describes a payload the API never returns, and the contract rejects
 *    both;
 *  - `revision` is monotonically non-decreasing on the real service, so fixtures that
 *    represent a *later* state must carry a higher revision than the state before it.
 */

import type { BusinessFact, EvidenceRef, ReviewRun, SessionInfo } from './types'

/** The local Bearer value the harness signs in with; never a real credential. */
export const TOKEN = 'local-test-credential-value'

/** The operator session every workbench suite starts from. */
export const operatorSession: SessionInfo = {
  api_version: 'review-api.v2',
  subject_id: 'subject-local-demo',
  tenant_id: 'tenant-local-demo',
  credential_id: 'cred-operator-0001',
  authorization_revision: 1,
  roles: ['approver', 'operator'],
  scopes: [
    'audit:read',
    'case:read',
    'review:cancel',
    'review:create',
    'review:decide',
    'review:read',
    'review:resume',
    'trace:read',
  ],
  issued_at: '2026-08-31T11:00:00Z',
  expires_at: '2026-08-31T12:00:00Z',
}

/** A read-only credential for the same tenant: no write affordance may appear. */
export const readerSession: SessionInfo = {
  ...operatorSession,
  subject_id: 'subject-local-demo-reader',
  credential_id: 'cred-reader-0001',
  roles: ['reader'],
  scopes: ['case:read', 'review:read'],
}

/** An operator credential without `review:resume`: 恢复审查 must be hidden. */
export const resumeLessSession: SessionInfo = {
  ...operatorSession,
  scopes: operatorSession.scopes.filter((scope) => scope !== 'review:resume'),
}

export const FIXTURE_RUN_ID = 'run-001'
export const FIXTURE_CHECKSUM = `sha256:${'a'.repeat(64)}`

export const evidence: EvidenceRef = {
  schema_version: '1.0',
  source_kind: 'public_document',
  source_id: 'fixture-competition',
  version: 'source-version-001',
  page_number: null,
  section: 'lines:1-1',
  chunk_id: 'chunk-001',
  content_checksum: `sha256:${'1'.repeat(64)}`,
}

export const businessFact: BusinessFact = {
  schema_version: '1.0',
  tool_name: 'get_purchase_order',
  record_id: 'PO-001',
  dataset_version: 'synthetic-procurement.v1',
  as_of: '2026-06-30T00:00:00Z',
  status: 'known',
  result: { purchase_order_id: 'PO-001', status: 'pending_review' },
  failure_reason: null,
  read_only: true,
}

type Summary = NonNullable<ReviewRun['summary']>

/** The worker ledger that ships inside every payload carrying a summary. */
export function executionFor(summary: Summary): ReviewRun['execution'] {
  return {
    schema_version: '1.0',
    execution_version: 'review-execution.v1',
    state_version: 'review-state.v1',
    branch_results: (['contract', 'policy', 'procurement_data'] as const).map((worker) => {
      const own = summary.findings.filter((finding) => finding.worker === worker)
      return {
        schema_version: '1.0',
        result_version: 'worker-result.v1',
        worker,
        outcome: 'completed',
        findings: own.length > 0 ? own : summary.findings.slice(0, 1),
        failure_code: null,
        failure_detail: null,
      }
    }),
    summary,
  }
}

/** One Run payload of the requested lifecycle status, with a matching summary. */
export function reviewRun(
  status: ReviewRun['status'],
  caseId = 'CASE-LOW-001',
): ReviewRun {
  const hasSummary = !['processing', 'failed', 'cancelled', 'interrupted'].includes(status)
  const confirmed = status === 'confirmed'
  const interrupted = status === 'interrupted'
  const summary: ReviewRun['summary'] = hasSummary
    ? {
        schema_version: '1.0',
        run_id: FIXTURE_RUN_ID,
        case_id: caseId,
        rules_version: '2.0.0',
        findings: [
          {
            schema_version: '1.0',
            worker: 'policy',
            finding_key: 'policy.required_policy_evidence',
            category: 'policy_compliance',
            risk_level: 'low',
            assertion: 'risk_absent',
            statement: '政策阈值证据完整。',
            status: 'supported',
            evidence_refs: [evidence],
            business_fact_refs: [],
            missing_information: [],
            recommended_action: '保留证据供人工确认。',
          },
        ],
        highest_risk: 'low',
        recommendation: 'recommend_approve',
        human_review_status: 'pending_confirmation',
        conflicts: [],
        missing_information: [],
        failure_reasons: [],
        is_final_decision: false,
      }
    : null
  return {
    run_id: FIXTURE_RUN_ID,
    trace_id: 'trace-001',
    status,
    version: status === 'processing' ? 1 : confirmed ? 3 : 2,
    revision: status === 'processing' ? 1 : confirmed ? 3 : 2,
    execution_mode: 'test_only',
    request: {
      schema_version: '1.0',
      case_id: caseId,
      case_type: 'purchase_order',
      document_version: 'retrieval-fixture-v1',
      purchase_order_id: 'PO-001',
      dataset_version: 'synthetic-procurement.v1',
      initiator: 'subject-local-demo',
    },
    summary,
    // The checksum is non-null exactly when the summary is non-null.
    summary_checksum: summary ? FIXTURE_CHECKSUM : null,
    execution: summary ? executionFor(summary) : null,
    branch_failures: [],
    evidence_refs: summary ? [evidence] : [],
    business_facts: summary ? [businessFact] : [],
    receipt: confirmed
      ? {
          // The v2 receipt projection is an APIContractV2: api_version only, no
          // schema_version, and a key digest instead of the raw idempotency key.
          api_version: 'review-api.v2',
          receipt_id: 'receipt-001',
          run_id: FIXTURE_RUN_ID,
          case_id: caseId,
          human_actor: 'subject-local-demo',
          decision: 'confirm_recommendation',
          reviewed_recommendation: 'recommend_approve',
          recorded_at: '2026-08-31T12:00:00Z',
          idempotency_key_digest: `sha256:${'9'.repeat(64)}`,
          simulation_status: 'local_simulation_recorded',
          simulation_mode: 'local_test_only',
          external_write_performed: false,
        }
      : null,
    created_at: '2026-08-31T11:59:59Z',
    updated_at: '2026-08-31T12:00:00Z',
    trace_event_count: confirmed ? 13 : 11,
    audit_event_count: confirmed ? 1 : 0,
    error_code: status === 'failed'
      ? 'review_execution_failed'
      : interrupted
        ? 'lease_expired'
        : null,
    error_message: status === 'failed'
      ? 'The local test-only review execution failed closed.'
      : interrupted
        ? 'The execution lease expired before a result was committed.'
        : null,
    recovery: {
      resume_count: 0,
      resume_limit: 2,
      resumable: interrupted,
      generation: 1,
      lease_expires_at: '2026-08-31T12:00:03Z',
      adopted_workers: interrupted ? ['contract', 'procurement_data'] : [],
      attempt_totals: { policy: 1 },
      interrupted_reason: interrupted ? 'lease_expired' : null,
    },
  }
}
