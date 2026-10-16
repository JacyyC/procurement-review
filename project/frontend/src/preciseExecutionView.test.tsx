/**
 * The rules 3.1.0 execution record on the page: the mode is stated, the metering is the
 * API's own numbers, and a frozen record never grows a precise panel it does not have.
 *
 * The distinction matters for the human Gate: a run under the synthetic profile and a run
 * under the real endpoint look identical in a recommendation, and the reviewer has to be
 * able to read which one produced it — and what it cost — before deciding.
 */

import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { PreciseExecutionPanel, ResultView } from './ReviewViews'
import { reviewRun } from './runFixtures'
import type { PreciseExecutionView, ReviewRun } from './types'

function preciseExecution(
  overrides: Partial<PreciseExecutionView> = {},
): PreciseExecutionView {
  return {
    execution_version: 'review-execution.v2',
    rules_version: '3.1.0',
    stop_reason: null,
    policy: {
      worker: 'policy',
      outcome: 'completed',
      reported: 3,
      support_failures: [],
      attempts: 1,
      evidence_refs: 2,
      failure_code: null,
    },
    contract: {
      worker: 'contract',
      outcome: 'failed',
      reported: 0,
      support_failures: [],
      attempts: 2,
      evidence_refs: 1,
      failure_code: 'invalid_schema',
    },
    procurement: { status: 'known', tool_calls: 3 },
    decision: {
      recommendation: 'recommend_conditional',
      findings: [
        {
          obligation_id: 'O-04',
          state: 'violated',
          reason: 'quote_threshold_violated',
          risk: 'medium',
          blocking: false,
          critical: false,
          remediable: true,
          missing: [],
          evidence: [
            {
              source_id: 'CASE-POLICY',
              version: 'synthetic-gold-document.v3.1',
              locator: `chunk-${'a'.repeat(64)}`,
              content_sha256: 'b'.repeat(64),
            },
          ],
        },
      ],
      violations: ['O-04'],
      unknown_obligations: [],
      pipeline_failures: [],
      security_events: [],
      conflict_pairs: [],
    },
    conflicts: [],
    security: [],
    metering: {
      physical_attempts: 3,
      first_sends: 2,
      corrections: 1,
      unknown_usage_attempts: 1,
      unreturned_attempts: 0,
      input_tokens_known: 1400,
      output_tokens_known: 520,
      branch_latency_ms: [
        ['policy', 41],
        ['contract', 88],
      ],
      requested_model: 'deepseek-v4.1-flash',
      resolved_models: ['deepseek-v4-1-flash'],
      attempts_opened: 3,
      proven_sends: 3,
      unresolved_attempts: 0,
      open_attempts: 0,
      contract_corrections: 0,
      citation_corrections: 1,
      transport_retries: 0,
      other_extra_attempts: 0,
      usage_complete: true,
      reused_branches: 0,
      explicit_recoveries: 0,
    },
    ...overrides,
  }
}

function runWith(execution: ReviewRun['execution'], mode: ReviewRun['execution_mode']) {
  return { ...reviewRun('pending_confirmation'), execution_mode: mode, execution }
}

describe('the execution profile chip', () => {
  it('names the synthetic profile instead of leaving the mode implicit', () => {
    render(<ResultView run={runWith(preciseExecution(), 'test_only')} />)
    const chip = screen.getByTestId('execution-mode')
    expect(chip.getAttribute('data-mode')).toBe('test_only')
    expect(chip.textContent).toContain('test_only')
  })

  it('names the real endpoint when the run used it', () => {
    render(<ResultView run={runWith(preciseExecution(), 'real')} />)
    const chip = screen.getByTestId('execution-mode')
    expect(chip.getAttribute('data-mode')).toBe('real')
    expect(chip.textContent).toContain('真实模型端点')
  })
})

describe('the precise execution panel', () => {
  it('renders the metering, the branches and the deterministic branch', () => {
    render(<PreciseExecutionPanel execution={preciseExecution()} />)
    expect(screen.getByTestId('precise-metering').textContent).toContain('1400')
    expect(screen.getByTestId('precise-metering').textContent).toContain('未知用量尝试')
    expect(screen.getByText(/报告 3 项/)).toBeTruthy()
    expect(screen.getByText('invalid_schema')).toBeTruthy()
    expect(screen.getByText(/3 次只读工具调用 · 0 次模型调用/)).toBeTruthy()
    expect(screen.queryByTestId('precise-stop-reason')).toBeNull()
  })

  it('states an explicit stop and an unknown token total as such', () => {
    render(
      <PreciseExecutionPanel
        execution={preciseExecution({
          stop_reason: 'model_identity_drift',
          metering: {
            ...preciseExecution().metering,
            input_tokens_known: null,
            output_tokens_known: null,
            resolved_models: [],
          },
        })}
      />,
    )
    expect(screen.getByTestId('precise-stop-reason').textContent).toContain(
      'model_identity_drift',
    )
    const metering = screen.getByTestId('precise-metering').textContent ?? ''
    expect(metering).toContain('未知')
    expect(metering).not.toContain('0 未知')
  })

  it('renders the five-class recommendation and every duty reason it rests on', () => {
    render(<PreciseExecutionPanel execution={preciseExecution()} />)
    const decision = screen.getByTestId('precise-decision').textContent ?? ''
    expect(decision).toContain('建议有条件通过')
    expect(decision).toContain('O-04')
    expect(decision).toContain('已违反')
    expect(decision).toContain('quote_threshold_violated')
    // The support location of the duty is stated, so the conclusion can be checked.
    expect(decision).toContain('CASE-POLICY')
    expect(decision).toContain('synthetic-gold-document.v3.1')
  })

  it('renders both sides of a conflict with their own sources', () => {
    const side = (source: string, value: string) => ({
      check_id: 'O-04',
      field_path: 'contract.o_04.category',
      subject_id: 'PO-1',
      effective_at: '2026-06-30',
      value,
      source_id: source,
      version: 'synthetic-gold-document.v3.1',
      locator: `chunk-${'c'.repeat(64)}`,
    })
    render(
      <PreciseExecutionPanel
        execution={preciseExecution({
          conflicts: [
            {
              pair_id: 'pair-1',
              kind: 'business_conflict',
              relation: 'same_proposition_exclusive_values',
              left: side('CONTRACT-A', 'lease'),
              right: side('ORDER-B', 'purchase'),
            },
          ],
        })}
      />,
    )
    const conflicts = screen.getByTestId('precise-conflicts').textContent ?? ''
    expect(conflicts).toContain('business_conflict')
    expect(conflicts).toContain('CONTRACT-A')
    expect(conflicts).toContain('ORDER-B')
    expect(conflicts).toContain('lease')
    expect(conflicts).toContain('purchase')
  })

  it('names a security reason by its fixed code, origin and location digest', () => {
    render(
      <PreciseExecutionPanel
        execution={preciseExecution({
          security: [
            { code: 'canary_output_blocked', origin: 'output', location_sha256: 'd'.repeat(64) },
          ],
        })}
      />,
    )
    const security = screen.getByTestId('precise-security').textContent ?? ''
    expect(security).toContain('canary_output_blocked')
    expect(security).toContain('output')
    expect(security).toContain('未发生外写')
  })

  it('shows no precise panel for a frozen execution record', () => {
    render(<ResultView run={runWith(reviewRun('pending_confirmation').execution, 'test_only')} />)
    expect(screen.queryByTestId('precise-execution')).toBeNull()
  })
})
