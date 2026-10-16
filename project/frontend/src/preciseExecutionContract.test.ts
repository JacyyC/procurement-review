/**
 * The rules 3.1.0 execution contract: every field the page renders is judged, and every
 * way a record can lie about one of them is refused.
 *
 * The negative cases are the point: an unreadable evidence identity, a security event
 * that claims a successful export, a conflict side without its own source — each one has
 * to fail closed rather than reach a component.
 */
import { describe, expect, it } from 'vitest'
import { validatePreciseExecution } from './preciseExecutionContract'
import { validateSummary } from './summaryContract'
import type { PlainRecord } from './apiPrimitives'

const CHECKSUM = 'b'.repeat(64)

function evidence(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    source_id: 'CASE-POLICY',
    version: 'synthetic-gold-document.v3.1',
    locator: `chunk-${'a'.repeat(64)}`,
    content_sha256: CHECKSUM,
    ...overrides,
  }
}

function finding(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    obligation_id: 'O-04',
    state: 'supported',
    reason: 'quote_threshold_met',
    risk: 'none',
    blocking: false,
    critical: true,
    remediable: false,
    missing: [],
    evidence: [evidence()],
    rules_version: '3.1.0',
    ...overrides,
  }
}

function decision(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    rules_version: '3.1.0',
    recommendation: 'recommend_approve',
    findings: [finding()],
    violations: [],
    unknown_obligations: [],
    pipeline_failures: [],
    security_events: [],
    conflict_pairs: [],
    ...overrides,
  }
}

function side(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    proposition: {
      check_id: 'O-04',
      field_path: 'contract.o_04.category',
      subject_id: 'PO-1',
      effective_at: '2026-06-30',
    },
    value: 'lease',
    evidence: [evidence()],
    ...overrides,
  }
}

function securityEvent(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    code: 'canary_output_blocked',
    origin: 'output',
    location_sha256: CHECKSUM,
    forbidden_export_succeeded: false,
    ...overrides,
  }
}

function branch(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    worker: 'policy',
    outcome: 'completed',
    reported: [],
    spans: [],
    support_failures: [],
    request_sha256: CHECKSUM,
    answer_sha256: CHECKSUM,
    attempts: [],
    evidence_refs: [],
    failure_code: null,
    ...overrides,
  }
}

function record(overrides: Record<string, unknown> = {}): PlainRecord {
  return {
    execution_version: 'review-execution.v2',
    run_id: 'run-1',
    case_id: 'CASE-LOW-001',
    rules_version: '3.1.0',
    decision: decision(),
    security: [securityEvent()],
    conflicts: [
      {
        pair_id: 'pair-1',
        kind: 'business_conflict',
        relation: 'same_proposition_exclusive_values',
        left: side(),
        right: side({ value: 'purchase', evidence: [evidence({ source_id: 'ORDER-B' })] }),
      },
    ],
    policy: branch(),
    contract: branch({ worker: 'contract' }),
    procurement: {
      worker: 'procurement_data',
      status: 'known',
      tool_calls: 3,
      model_calls: 0,
    },
    metering: {
      physical_attempts: 2,
      first_sends: 2,
      corrections: 0,
      unknown_usage_attempts: 0,
      unreturned_attempts: 0,
      input_tokens_known: 1400,
      output_tokens_known: 520,
      branch_latency_ms: [['policy', 41]],
      requested_model: 'deepseek-v4.1-flash',
      resolved_models: ['deepseek-v4-1-flash'],
    },
    stop_reason: null,
    summary: {
      schema_version: '1.0',
      run_id: 'run-1',
      case_id: 'CASE-LOW-001',
      rules_version: '3.1.0',
      findings: [],
      highest_risk: 'low',
      recommendation: 'recommend_approve',
      human_review_status: 'pending_confirmation',
      conflicts: [],
      missing_information: [],
      failure_reasons: [],
      is_final_decision: false,
    },
    ...overrides,
  }
}

function rejects(mutate: (value: PlainRecord) => void, pattern: RegExp) {
  const value = record()
  mutate(value)
  expect(() => validatePreciseExecution(value, 'run.execution')).toThrowError(pattern)
}

describe('the rules 3.1.0 execution contract', () => {
  it('accepts a complete record and reads every part of it', () => {
    const view = validatePreciseExecution(record(), 'run.execution')
    expect(view.decision.recommendation).toBe('recommend_approve')
    expect(view.decision.findings[0].obligation_id).toBe('O-04')
    expect(view.decision.findings[0].evidence[0].locator.startsWith('chunk-')).toBe(true)
    expect(view.conflicts[0].left.source_id).toBe('CASE-POLICY')
    expect(view.conflicts[0].right.value).toBe('purchase')
    expect(view.security[0].code).toBe('canary_output_blocked')
    expect(view.security[0].origin).toBe('output')
  })

  it('refuses an unknown recommendation or an unknown obligation state', () => {
    rejects((value) => {
      ;(value.decision as PlainRecord).recommendation = 'looks_fine'
    }, /recommendation/)
    rejects((value) => {
      ;((value.decision as PlainRecord).findings as PlainRecord[])[0].state = 'maybe'
    }, /state/)
  })

  it('refuses an evidence identity that is not a located digest', () => {
    rejects((value) => {
      ;(
        ((value.decision as PlainRecord).findings as PlainRecord[])[0]
          .evidence as PlainRecord[]
      )[0].content_sha256 = 'sha256:short'
    }, /content_sha256/)
    rejects((value) => {
      delete (
        ((value.decision as PlainRecord).findings as PlainRecord[])[0]
          .evidence as PlainRecord[]
      )[0].locator
    }, /locator/)
    rejects((value) => {
      ;((value.decision as PlainRecord).findings as PlainRecord[])[0].rules_version =
        '2.0.0'
    }, /rules_version/)
    rejects((value) => {
      delete ((value.decision as PlainRecord).findings as PlainRecord[])[0].blocking
    }, /blocking/)
    rejects((value) => {
      ;((value.decision as PlainRecord).findings as PlainRecord[])[0].missing = 'none'
    }, /missing/)
  })

  it('refuses a security event that is unlocatable or claims an export', () => {
    rejects((value) => {
      ;(value.security as PlainRecord[])[0].location_sha256 = 'not-a-digest'
    }, /location_sha256/)
    rejects((value) => {
      ;(value.security as PlainRecord[])[0].origin = 'somewhere'
    }, /origin/)
    rejects((value) => {
      ;(value.security as PlainRecord[])[0].forbidden_export_succeeded = true
    }, /forbidden_export_succeeded/)
    rejects((value) => {
      delete (value.security as PlainRecord[])[0].code
    }, /code/)
  })

  it('refuses a conflict side that has no source or no scalar value', () => {
    rejects((value) => {
      ;((value.conflicts as PlainRecord[])[0].left as PlainRecord).evidence = []
    }, /evidence/)
    rejects((value) => {
      ;((value.conflicts as PlainRecord[])[0].right as PlainRecord).value = { any: 'object' }
    }, /value/)
    rejects((value) => {
      delete ((value.conflicts as PlainRecord[])[0] as PlainRecord).left
    }, /left/)
    rejects((value) => {
      ;(
        (((value.conflicts as PlainRecord[])[0].left as PlainRecord)
          .proposition as PlainRecord)
      ).check_id = 7
    }, /check_id/)
  })

  it('refuses a malformed latency pair and a missing summary', () => {
    rejects((value) => {
      ;(value.metering as PlainRecord).branch_latency_ms = [['policy']]
    }, /branch_latency_ms/)
    rejects((value) => {
      ;(value.metering as PlainRecord).branch_latency_ms = [['policy', 1.5]]
    }, /branch_latency_ms/)
    rejects((value) => {
      ;(value.metering as PlainRecord).branch_latency_ms = ['policy']
    }, /branch_latency_ms/)
    rejects((value) => {
      value.summary = null
    }, /summary/)
    rejects((value) => {
      delete value.decision
    }, /decision/)
    rejects((value) => {
      delete value.conflicts
    }, /conflicts/)
    rejects((value) => {
      delete value.security
    }, /security/)
  })

  it('carries the optional meterings of a legacy record as unrecorded', () => {
    const value = record()
    const metering = value.metering as PlainRecord
    delete metering.attempts_opened
    delete metering.usage_complete
    const view = validatePreciseExecution(value, 'run.execution')
    expect(view.metering.attempts_opened).toBeNull()
    expect(view.metering.usage_complete).toBeNull()
  })

  it('refuses a malformed counter and requires the deterministic branch to make no call', () => {
    rejects((value) => {
      ;(value.metering as PlainRecord).attempts_opened = -1
    }, /attempts_opened/)
    rejects((value) => {
      ;(value.metering as PlainRecord).usage_complete = 'yes'
    }, /usage_complete/)
    rejects((value) => {
      ;(value.metering as PlainRecord).unknown_usage_attempts = 1.5
    }, /unknown_usage_attempts/)
    rejects((value) => {
      ;(value.procurement as PlainRecord).model_calls = 1
    }, /model_calls/)
  })

  it('refuses a branch that is not one of the two document Workers', () => {
    rejects((value) => {
      ;(value.policy as PlainRecord).worker = 'procurement_data'
    }, /worker/)
    rejects((value) => {
      ;(value.policy as PlainRecord).outcome = 'fine'
    }, /outcome/)
  })
})

describe('the committed summary contract', () => {
  it('accepts the summary every Run carries and reads its own fields', () => {
    const view = validateSummary((record().summary as PlainRecord), 'run.summary')
    expect(view.run_id).toBe('run-1')
    expect(view.is_final_decision).toBe(false)
  })

  it('refuses a summary that claims to be final or carries an unknown risk word', () => {
    const value = record().summary as PlainRecord
    value.is_final_decision = true
    expect(() => validateSummary(value, 'run.summary')).toThrowError(/is_final_decision/)
    const second = record().summary as PlainRecord
    second.highest_risk = 'catastrophic'
    expect(() => validateSummary(second, 'run.summary')).toThrowError(/highest_risk/)
  })
})
