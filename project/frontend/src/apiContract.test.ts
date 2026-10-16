/**
 * FIX-01C / V-C-02 — the runtime boundary between the API and React.
 *
 * Every payload here is built from a *valid* one and then mutated one field at a
 * time, so each case proves the boundary rejects exactly what it claims to.  The
 * positive controls matter as much as the rejections: legal `null`, empty arrays and
 * empty open maps must keep working, because a validator that rejects them would
 * turn a normal server state into an error screen.
 *
 * The last block pins the error envelope: the v2 envelope is validated like a success
 * body, and the two bodies the server does *not* wrap (FastAPI's default
 * `{"detail": ...}` 404/405) are still reported against their real HTTP status.
 */

import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  getAudit,
  getReview,
  getSession,
  getTrace,
  listReviews,
  ReviewAPIError,
} from './api'
import type { ReviewRun } from './types'

const AUTH = { token: 'fix01c-contract-credential' }
const RUN_ID = 'run-001'
const CHECKSUM = `sha256:${'a'.repeat(64)}`

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function stubFetch(body: unknown, status = 200) {
  const fetchMock = vi.fn(async () => jsonResponse(body, status))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function stubRawFetch(body: string, status: number, contentType = 'application/json') {
  vi.stubGlobal(
    'fetch',
    vi.fn(
      async () =>
        new Response(body, { status, headers: { 'Content-Type': contentType } }),
    ),
  )
}

function finding(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: '1.0',
    worker: 'policy',
    finding_key: 'policy.required_policy_evidence',
    category: 'policy_compliance',
    risk_level: 'low',
    assertion: 'risk_absent',
    statement: '政策阈值证据完整。',
    status: 'supported',
    evidence_refs: [],
    business_fact_refs: [],
    missing_information: [],
    recommended_action: '保留证据供人工确认。',
    ...overrides,
  }
}

function summary(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: '1.0',
    run_id: RUN_ID,
    case_id: 'CASE-LOW-001',
    rules_version: '2.0.0',
    findings: [finding()],
    highest_risk: 'low',
    recommendation: 'recommend_approve',
    human_review_status: 'pending_confirmation',
    conflicts: [],
    missing_information: [],
    failure_reasons: [],
    is_final_decision: false,
    ...overrides,
  }
}

function execution(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: '1.0',
    execution_version: 'review-execution.v1',
    state_version: 'review-state.v1',
    branch_results: (['contract', 'policy', 'procurement_data'] as const).map((worker) => ({
      schema_version: '1.0',
      result_version: 'worker-result.v1',
      worker,
      outcome: 'completed',
      findings: [finding({ worker })],
      failure_code: null,
      failure_detail: null,
    })),
    summary: summary(),
    ...overrides,
  }
}

function preciseBranch(worker: string, overrides: Record<string, unknown> = {}) {
  return {
    worker,
    outcome: 'completed',
    reported: [{ obligation_id: 'O-04', status: 'supported', normalized: '{"minimum_quotes":3}' }],
    spans: [],
    support_failures: [],
    request_sha256: 'a'.repeat(64),
    answer_sha256: 'b'.repeat(64),
    attempts: [],
    evidence_refs: [],
    failure_code: null,
    ...overrides,
  }
}

function preciseExecution(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    execution_version: 'review-execution.v2',
    run_id: RUN_ID,
    case_id: 'CASE-LOW-001',
    rules_version: '3.1.0',
    decision: {
      rules_version: '3.1.0',
      recommendation: 'recommend_approve',
      findings: [
        {
          obligation_id: 'O-04',
          state: 'supported',
          reason: 'quote_threshold_met',
          risk: 'none',
          blocking: false,
          critical: false,
          remediable: false,
          missing: [],
          evidence: [
            {
              source_id: 'CASE-LOW-001-POLICY',
              version: 'synthetic-gold-document.v3.1',
              locator: `chunk-${'a'.repeat(64)}`,
              content_sha256: 'b'.repeat(64),
            },
          ],
          rules_version: '3.1.0',
        },
      ],
      violations: [],
      unknown_obligations: [],
      pipeline_failures: [],
      security_events: [],
      conflict_pairs: [],
    },
    security: [],
    conflicts: [],
    policy: preciseBranch('policy'),
    contract: preciseBranch('contract'),
    procurement: {
      worker: 'procurement_data',
      status: 'known',
      dataset_version: 'synthetic-gold-procurement.v3',
      tool_calls: 3,
      record_count: 3,
      model_calls: 0,
      detail: null,
    },
    metering: {
      physical_attempts: 2,
      first_sends: 2,
      corrections: 0,
      input_tokens_known: 1400,
      output_tokens_known: 520,
      unknown_usage_attempts: 0,
      unreturned_attempts: 0,
      branch_latency_ms: [['policy', 41]],
      requested_model: 'deepseek-v4.1-flash',
      resolved_models: ['deepseek-v4-1-flash'],
    },
    stop_reason: null,
    summary: summary(),
    ...overrides,
  }
}

function receipt(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    api_version: 'review-api.v2',
    receipt_id: 'receipt-001',
    run_id: RUN_ID,
    case_id: 'CASE-LOW-001',
    human_actor: 'subject-local-demo',
    decision: 'confirm_recommendation',
    reviewed_recommendation: 'recommend_approve',
    recorded_at: '2026-09-01T00:00:06Z',
    idempotency_key_digest: `sha256:${'b'.repeat(64)}`,
    simulation_status: 'local_simulation_recorded',
    simulation_mode: 'local_test_only',
    external_write_performed: false,
    ...overrides,
  }
}

function validRun(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    run_id: RUN_ID,
    trace_id: 'trace-001',
    status: 'pending_confirmation',
    version: 2,
    revision: 2,
    execution_mode: 'test_only',
    request: {
      schema_version: '1.0',
      case_id: 'CASE-LOW-001',
      case_type: 'purchase_order',
      document_version: 'retrieval-fixture-v1',
      purchase_order_id: 'PO-001',
      dataset_version: 'synthetic-procurement.v1',
      initiator: 'subject-local-demo',
    },
    summary: summary(),
    summary_checksum: CHECKSUM,
    execution: execution(),
    branch_failures: [],
    evidence_refs: [],
    business_facts: [],
    receipt: null,
    created_at: '2026-09-01T00:00:00Z',
    updated_at: '2026-09-01T00:00:05Z',
    trace_event_count: 4,
    audit_event_count: 0,
    error_code: null,
    error_message: null,
    recovery: {
      resume_count: 0,
      resume_limit: 2,
      resumable: false,
      generation: 1,
      lease_expires_at: null,
      adopted_workers: [],
      attempt_totals: {},
      interrupted_reason: null,
    },
    ...overrides,
  }
}

function validSession(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    api_version: 'review-api.v2',
    subject_id: 'subject-local-demo',
    tenant_id: 'tenant-local-demo',
    credential_id: 'cred-001',
    authorization_revision: 1,
    roles: ['operator'],
    scopes: ['review:read'],
    issued_at: '2026-09-01T00:00:00Z',
    expires_at: '2026-09-01T01:00:00Z',
    ...overrides,
  }
}

function validTrace(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    api_version: 'review-api.v1',
    trace: {
      schema_version: '1.0',
      trace_schema_version: 'trace.v1',
      trace_id: 'trace-001',
      run_id: RUN_ID,
      events: [
        {
          schema_version: '1.0',
          event_id: 'event-run-001-0001',
          run_id: RUN_ID,
          trace_id: 'trace-001',
          event_type: 'run_started',
          step: 'review_run',
          occurred_at: '2026-09-01T00:00:00Z',
          input_summary: {},
          output_summary: {},
          source_versions: { api: 'review-api.v2' },
          duration_ms: 0,
          token_count: 0,
          cost_amount: '0',
          cost_currency: 'CNY',
          redaction_status: 'not_required',
          error: null,
        },
      ],
      complete: false,
    },
    ...overrides,
  }
}

function validAudit(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    api_version: 'review-api.v2',
    audit_schema_version: 'approval-audit.v2',
    run_id: RUN_ID,
    events: [],
    ...overrides,
  }
}

/** One audit event anchored to the Run the ledger belongs to. */
function validAuditEvent(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: '1.0',
    audit_schema_version: 'approval-audit.v2',
    audit_id: 'audit-1',
    sequence: 1,
    tenant_id: null,
    case_id: null,
    run_id: RUN_ID,
    event_class: 'lifecycle',
    action_type: null,
    actor: null,
    authorization_revision: null,
    previous_status: null,
    new_status: null,
    previous_revision: null,
    new_revision: null,
    result_checksum: null,
    idempotency_digest: null,
    receipt_id: null,
    api_version: 'review-api.v2',
    occurred_at: '2026-09-01T00:00:06Z',
    detail_code: null,
    simulation_only: true,
    external_write_performed: false,
    ...overrides,
  }
}

function withRun(mutate: (payload: Record<string, unknown>) => void): Record<string, unknown> {
  const payload = validRun()
  mutate(payload)
  return payload
}

function withSummaryField(
  mutate: (value: Record<string, unknown>) => void,
): Record<string, unknown> {
  const payload = summary()
  mutate(payload)
  return withRun((run) => {
    run.summary = payload
  })
}

async function rejection(task: Promise<unknown>): Promise<ReviewAPIError> {
  const error = await task.then(
    () => null,
    (reason: unknown) => reason,
  )
  expect(error).toBeInstanceOf(ReviewAPIError)
  return error as ReviewAPIError
}

async function expectRejected(task: Promise<unknown>, pathFragment: string): Promise<void> {
  const error = await rejection(task)
  expect(error.code).toBe('invalid_response')
  const described = `${String(error.details.path ?? '')} ${error.message}`
  expect(described).toContain(pathFragment)
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('FIX-01C run response contract', () => {
  it('accepts the real payload shape and keeps every consumed field', async () => {
    stubFetch(validRun())
    const value: ReviewRun = await getReview(RUN_ID, AUTH)
    expect(value.run_id).toBe(RUN_ID)
    expect(value.summary?.findings[0].statement).toBe('政策阈值证据完整。')
    expect(value.summary_checksum).toBe(CHECKSUM)
    expect(value.recovery?.resume_limit).toBe(2)
  })

  it('ignores fields the workbench does not consume instead of copying them', async () => {
    const payload = withRun((run) => {
      run.future_field = { anything: true }
      ;(run.summary as Record<string, unknown>).future_summary_field = 1
      ;((run.summary as Record<string, unknown>).findings as Record<string, unknown>[])[0]
        .future_finding_field = 'x'
    })
    stubFetch(payload)
    const value = await getReview(RUN_ID, AUTH)
    expect(value).not.toHaveProperty('future_field')
    expect(value.summary).not.toHaveProperty('future_summary_field')
    expect(value.summary?.findings[0]).not.toHaveProperty('future_finding_field')
    // The consumed fields are still there: ignoring extras is not dropping data.
    expect(value.summary?.findings[0].finding_key).toBe('policy.required_policy_evidence')
  })

  it('rejects a payload that is missing a top-level consumed field', async () => {
    const payload = withRun((run) => {
      delete run.summary
    })
    stubFetch(payload)
    await expectRejected(getReview(RUN_ID, AUTH), 'run.summary')
  })

  it('rejects a summary that lost missing_information', async () => {
    const payload = withRun((run) => {
      delete (run.summary as Record<string, unknown>).missing_information
    })
    stubFetch(payload)
    await expectRejected(getReview(RUN_ID, AUTH), 'run.summary.missing_information')
  })

  it('rejects a finding that lost a nested consumed field', async () => {
    const payload = withSummaryField((value) => {
      delete ((value.findings as Record<string, unknown>[])[0]).statement
    })
    stubFetch(payload)
    await expectRejected(getReview(RUN_ID, AUTH), 'run.summary.findings[0].statement')
  })

  it('rejects a wrong field type at any depth', async () => {
    stubFetch(withRun((run) => { run.trace_event_count = '4' }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.trace_event_count')

    stubFetch(withSummaryField((value) => { value.findings = {} }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.summary.findings')
  })

  it('rejects an illegal enum and accepts the legal one', async () => {
    stubFetch(withRun((run) => { run.status = 'auto_approved' }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.status')

    stubFetch(withRun((run) => { run.status = 'interrupted' }))
    const value = await getReview(RUN_ID, AUTH)
    expect(value.status).toBe('interrupted')
  })

  it('rejects a non-existent calendar date and a date without a time', async () => {
    stubFetch(withRun((run) => { run.created_at = '2026-06-31T99:99:99Z' }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.created_at')

    stubFetch(withRun((run) => { run.updated_at = '2026-06-30' }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.updated_at')
  })

  it('rejects null where the contract requires a value but keeps legal nulls legal', async () => {
    stubFetch(withRun((run) => { run.run_id = null }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.run_id')

    stubFetch(withRun((run) => {
      run.summary = null
      run.summary_checksum = null
      run.execution = null
      run.recovery = null
      run.error_code = null
      run.error_message = null
    }))
    const value = await getReview(RUN_ID, AUTH)
    expect(value.summary).toBeNull()
    expect(value.recovery).toBeNull()
  })

  it('rejects a checksum that disagrees with the presence of a summary', async () => {
    stubFetch(withRun((run) => { run.summary_checksum = null }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.summary_checksum')
  })

  it('rejects a structurally valid payload that describes another Run', async () => {
    stubFetch(validRun({ run_id: 'run-other' }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.run_id')

    stubFetch(withSummaryField((value) => { value.run_id = 'run-other' }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.summary.run_id')

    stubFetch(validRun({ receipt: receipt({ run_id: 'run-other' }) }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.receipt.run_id')
  })

  it('rejects an empty object where a closed sub-record is required', async () => {
    stubFetch(withRun((run) => { run.execution = {} }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.execution')

    stubFetch(withSummaryField(() => {}))
    stubFetch(withRun((run) => { run.summary = {} }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.summary.schema_version')

    stubFetch(withRun((run) => { run.recovery = {} }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.recovery.resume_count')
  })

  it('keeps legal empty arrays legal but rejects a truncated worker ledger', async () => {
    stubFetch(withSummaryField((value) => { value.findings = [] }))
    const empty = await getReview(RUN_ID, AUTH)
    expect(empty.summary?.findings).toEqual([])

    stubFetch(withRun((run) => {
      ;(run.execution as Record<string, unknown>).branch_results = []
    }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.execution.branch_results')

    stubFetch(withRun((run) => {
      const branch = ((run.execution as Record<string, unknown>).branch_results as unknown[])[0]
      ;(branch as Record<string, unknown>).findings = []
    }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.execution.branch_results[0].findings')
  })

  it('rejects a receipt without its v2 version literal', async () => {
    stubFetch(validRun({ receipt: receipt({ api_version: undefined }) }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.receipt.api_version')

    stubFetch(validRun({ receipt: receipt({ decision: 'approve_forever' }) }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.receipt.decision')

    // The real v2 receipt has no schema_version; a legal one is accepted whole.
    stubFetch(validRun({ receipt: receipt() }))
    const value = await getReview(RUN_ID, AUTH)
    expect(value.receipt?.receipt_id).toBe('receipt-001')
  })

  it('rejects an unknown worker in the recovery totals and in branch failures', async () => {
    stubFetch(withRun((run) => {
      ;(run.recovery as Record<string, unknown>).attempt_totals = { unknown_worker: 1 }
    }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.recovery.attempt_totals.unknown_worker')

    stubFetch(withRun((run) => {
      run.branch_failures = [
        { api_version: 'review-api.v2', worker: 'policy', code: 'model_failure', detail: 'x' },
      ]
    }))
    await expectRejected(getReview(RUN_ID, AUTH), 'run.branch_failures[0].api_version')
  })

  it('rejects a Run list whose item is malformed, naming the entry', async () => {
    const second = validRun({ run_id: 'run-002', trace_id: null })
    second.summary = summary({ run_id: 'run-002' })
    stubFetch({
      api_version: 'review-api.v2',
      runs: [validRun(), second],
    })
    await expectRejected(listReviews(AUTH), 'review_list.runs[1].trace_id')
  })
})

describe('FIX-01C session, trace and audit contracts', () => {
  it('rejects a session with an unknown api_version literal', async () => {
    stubFetch(validSession({ api_version: 'review-api.v3' }))
    await expectRejected(getSession(AUTH), 'session.api_version')

    stubFetch(validSession({ scopes: ['review:read', 'review:destroy'] }))
    await expectRejected(getSession(AUTH), 'session.scopes[1]')
  })

  it('rejects a malformed Trace event and keeps the legal ones', async () => {
    stubFetch(
      validTrace({
        trace: {
          ...(validTrace().trace as Record<string, unknown>),
          events: [
            {
              ...((validTrace().trace as Record<string, unknown>).events as unknown[])[0] as object,
              redaction_status: 'maybe',
            },
          ],
        },
      }),
    )
    await expectRejected(getTrace(RUN_ID, AUTH), 'trace_response.trace.events[0].redaction_status')

    stubFetch(
      validTrace({
        trace: {
          ...(validTrace().trace as Record<string, unknown>),
          events: [
            {
              ...((validTrace().trace as Record<string, unknown>).events as unknown[])[0] as object,
              occurred_at: 'yesterday',
            },
          ],
        },
      }),
    )
    await expectRejected(getTrace(RUN_ID, AUTH), 'trace_response.trace.events[0].occurred_at')
  })

  it('rejects a Trace envelope that is stamped with the wrong version or another Run', async () => {
    stubFetch(validTrace({ api_version: 'review-api.v2' }))
    await expectRejected(getTrace(RUN_ID, AUTH), 'trace_response.api_version')

    stubFetch(
      validTrace({
        trace: { ...(validTrace().trace as Record<string, unknown>), run_id: 'run-other' },
      }),
    )
    await expectRejected(getTrace(RUN_ID, AUTH), 'trace_response.trace.run_id')
  })

  it('accepts an empty Trace event list and a null cost, which the UI renders explicitly', async () => {
    stubFetch(
      validTrace({
        trace: {
          ...(validTrace().trace as Record<string, unknown>),
          events: [],
          complete: true,
        },
      }),
    )
    const value = await getTrace(RUN_ID, AUTH)
    expect(value.trace.events).toEqual([])

    stubFetch(
      validTrace({
        trace: {
          ...(validTrace().trace as Record<string, unknown>),
          events: [
            {
              ...((validTrace().trace as Record<string, unknown>).events as unknown[])[0] as object,
              cost_amount: null,
              cost_currency: null,
              token_count: null,
            },
          ],
        },
      }),
    )
    const nulls = await getTrace(RUN_ID, AUTH)
    expect(nulls.trace.events[0].cost_amount).toBeNull()
  })

  it('rejects an Audit envelope for another Run and an unknown event class', async () => {
    stubFetch(validAudit({ run_id: 'run-other' }))
    await expectRejected(getAudit(RUN_ID, AUTH), 'audit_response.run_id')

    stubFetch(
      validAudit({
        events: [validAuditEvent({ run_id: null, event_class: 'invented_class' })],
      }),
    )
    // `run_id` is validated before `event_class`, so an event with no Run anchor is
    // rejected on that field first (FIX-04B / C-AUD-05).
    await expectRejected(getAudit(RUN_ID, AUTH), 'audit_response.events[0].run_id')
  })

  it('rejects an unknown audit event class once the event belongs to the Run', async () => {
    stubFetch(
      validAudit({
        events: [{ ...validAuditEvent(), event_class: 'invented_class' }],
      }),
    )
    await expectRejected(getAudit(RUN_ID, AUTH), 'audit_response.events[0].event_class')
  })

  it('rejects an audit event anchored to another Run', async () => {
    stubFetch(validAudit({ events: [{ ...validAuditEvent(), run_id: 'run-other' }] }))
    await expectRejected(getAudit(RUN_ID, AUTH), 'audit_response.events[0].run_id')
  })

  it('rejects a sequence below one but keeps an empty audit legal', async () => {
    const event = {
      schema_version: '1.0',
      audit_schema_version: 'approval-audit.v2',
      audit_id: 'audit-1',
      sequence: 0,
      tenant_id: null,
      case_id: null,
      run_id: null,
      event_class: 'lifecycle',
      action_type: null,
      actor: null,
      authorization_revision: null,
      previous_status: null,
      new_status: null,
      previous_revision: null,
      new_revision: null,
      result_checksum: null,
      idempotency_digest: null,
      receipt_id: null,
      api_version: 'review-api.v2',
      occurred_at: '2026-09-01T00:00:06Z',
      detail_code: null,
      simulation_only: true,
      external_write_performed: false,
    }
    stubFetch(validAudit({ events: [event] }))
    await expectRejected(getAudit(RUN_ID, AUTH), 'audit_response.events[0].sequence')

    stubFetch(validAudit())
    const value = await getAudit(RUN_ID, AUTH)
    expect(value.events).toEqual([])
  })
})

describe('FIX-01C error envelope boundary', () => {
  it('keeps the server code, request id and details of a real envelope', async () => {
    stubFetch(
      {
        error: {
          code: 'version_conflict',
          message: 'stale',
          request_id: 'request-001',
          details: { expected_version: 1, current_version: 3 },
        },
      },
      409,
    )
    const error = await rejection(getReview(RUN_ID, AUTH))
    expect(error.status).toBe(409)
    expect(error.code).toBe('version_conflict')
    expect(error.requestId).toBe('request-001')
    // The code is kept (the UI maps it through a local vocabulary) and so is the request
    // id (rendered only when it matches the local identifier contract).  The server's
    // `message` and `details` are validated for shape and then dropped: an error is
    // something that gets logged, and unverified server text must not ride along
    // (FIX-04B / C-AUD-04).
    expect(error.details).toEqual({})
    expect(error.message).not.toContain('stale')
    expect(JSON.stringify(error)).not.toContain('current_version')
  })

  it('reports the FastAPI default {"detail": ...} body as an unreadable envelope', async () => {
    stubRawFetch(JSON.stringify({ detail: 'Not Found' }), 404)
    const error = await rejection(getReview(RUN_ID, AUTH))
    // The real status survives; the malformed envelope becomes invalid_response
    // instead of being reported as a business outcome the server never sent.
    expect(error.status).toBe(404)
    expect(error.code).toBe('invalid_response')

    stubRawFetch(JSON.stringify({ detail: 'Method Not Allowed' }), 405)
    const method = await rejection(getReview(RUN_ID, AUTH))
    expect(method.status).toBe(405)
    expect(method.code).toBe('invalid_response')
  })

  it('reports a non-JSON failure body without inventing a business code', async () => {
    stubRawFetch('<html>gateway</html>', 503, 'text/html')
    const error = await rejection(getReview(RUN_ID, AUTH))
    expect(error.status).toBe(503)
    expect(error.code).toBe('invalid_response')
    expect(error.message).not.toContain('gateway')
  })

  it('rejects a success body that is JSON null', async () => {
    stubRawFetch('null', 200)
    const error = await rejection(getReview(RUN_ID, AUTH))
    expect(error.code).toBe('invalid_response')
  })
})

describe('the precise execution record on the page', () => {
  it('accepts review-execution.v2, its metering and the real execution mode', async () => {
    stubFetch(
      withRun((run) => {
        run.execution_mode = 'real'
        run.execution = preciseExecution()
      }),
    )
    const value: ReviewRun = await getReview(RUN_ID, AUTH)
    expect(value.execution_mode).toBe('real')
  })

  it('refuses an execution record whose version it does not know', async () => {
    stubFetch(
      withRun((run) => {
        run.execution = preciseExecution({ execution_version: 'review-execution.v9' })
      }),
    )
    await expect(rejection(getReview(RUN_ID, AUTH))).resolves.toMatchObject({
      code: 'invalid_response',
    })
  })

  it('refuses a deterministic branch that claims model calls', async () => {
    stubFetch(
      withRun((run) => {
        const execution = preciseExecution()
        execution.procurement = {
          ...(execution.procurement as Record<string, unknown>),
          model_calls: 2,
        }
        run.execution = execution
      }),
    )
    const error = await rejection(getReview(RUN_ID, AUTH))
    expect(JSON.stringify(error)).toMatch(/model_calls/)
  })
})
