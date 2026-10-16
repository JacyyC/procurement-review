import { Activity, Ban, CheckCircle2, ChevronDown, ClipboardCheck, Database, FileSearch, FileText, History, ListChecks, LoaderCircle, PauseCircle, ShieldAlert, TriangleAlert, XCircle } from 'lucide-react'
import { ReviewIllustration } from './WorkbenchFrame'
import type { AuditResponse, BusinessFact, ConfirmationReceipt, EvidenceRef, Finding, PreciseDecisionView, PreciseExecutionView, ReviewRun, ReviewStatus, RiskLevel, TraceResponse } from './types'
import { isPreciseExecution } from './types'
import { compactId, formatTime, OBLIGATION_STATE_LABELS, RECOMMENDATION_LABELS, recoveryGuidance, RISK_LABELS, STATUS_LABELS, workerLabel, WORKER_LABELS } from './reviewPresentation'

export function StatusBadge({ status }: { status: ReviewStatus }) {
  return <span className={`status-badge status-${status}`}>{STATUS_LABELS[status]}</span>
}

function RiskBadge({ risk }: { risk: RiskLevel }) {
  return <span className={`risk-badge risk-${risk}`}>风险 {RISK_LABELS[risk]}</span>
}

export function EmptyState({ loading }: { loading: boolean }) {
  return (
    <section className="empty-state" aria-busy={loading}>
      {loading ? (
        <LoaderCircle className="spin" aria-hidden="true" />
      ) : (
        <ReviewIllustration />
      )}
      <h2>{loading ? '正在读取本地工作台' : '尚未创建审查'}</h2>
      <p>{loading ? '读取案例和运行状态…' : '选择一个已授权案例，开启你的第一次合规审查。'}</p>
    </section>
  )
}

export function CredentialRequired() {
  return (
    <section className="locked-state" aria-label="需要凭据">
      <ReviewIllustration />
      <span className="section-kicker">YOUR REVIEW STARTS HERE</span>
      <h2>需要本地凭据</h2>
      <p>
        连接工作台后，即可查看已授权案例、追溯历史运行，
        并从政策、合同与采购数据三个维度开始审查。
      </p>
    </section>
  )
}

function ReceiptPanel({ receipt }: { receipt: ConfirmationReceipt }) {
  return (
    <section className="receipt-panel" aria-labelledby="receipt-title">
      <div className="section-heading">
        <div>
          <span className="section-kicker">LOCAL · TEST_ONLY</span>
          <h2 id="receipt-title">本地模拟回执</h2>
        </div>
        <CheckCircle2 aria-hidden="true" />
      </div>
      <dl className="receipt-grid">
        <div>
          <dt>本地动作</dt>
          <dd>
            {receipt.decision === 'confirm_recommendation'
              ? '确认建议'
              : '拒绝建议'}
          </dd>
        </div>
        <div>
          <dt>操作人（服务端主体）</dt>
          <dd>{receipt.human_actor}</dd>
        </div>
        <div>
          <dt>记录时间</dt>
          <dd>{formatTime(receipt.recorded_at)}</dd>
        </div>
        <div>
          <dt>外部写入</dt>
          <dd>0 · 未执行</dd>
        </div>
        <div>
          <dt>幂等键摘要</dt>
          <dd>
            <code title={receipt.idempotency_key_digest}>
              {compactId(receipt.idempotency_key_digest)}
            </code>
          </dd>
        </div>
      </dl>
      <div className="receipt-footer">
        <code title={receipt.receipt_id}>{compactId(receipt.receipt_id)}</code>
      </div>
    </section>
  )
}

function FindingCard({ finding, index }: { finding: Finding; index: number }) {
  return (
    <article className="finding-card">
      <div className="finding-header">
        <span className="finding-number">{String(index + 1).padStart(2, '0')}</span>
        <div>
          <span className="worker-label">{WORKER_LABELS[finding.worker]}</span>
          <h3>{finding.statement}</h3>
        </div>
        <RiskBadge risk={finding.risk_level} />
      </div>
      <div className="finding-meta">
        <span>{finding.category}</span>
        <span>{finding.assertion}</span>
        <span>{finding.status}</span>
        <span>
          {finding.evidence_refs.length} 份证据 · {finding.business_fact_refs.length}{' '}
          条事实
        </span>
      </div>
      <p className="finding-action"><span>建议行动</span>{finding.recommended_action}</p>
    </article>
  )
}

function EvidenceItem({ item }: { item: EvidenceRef }) {
  const locator = item.page_number
    ? `页码 ${item.page_number}`
    : item.section ?? item.chunk_id ?? '无定位'
  return (
    <details className="record-row">
      <summary>
        <FileText size={18} aria-hidden="true" />
        <span>
          <strong>{item.source_id}</strong>
          <small>{locator}</small>
        </span>
        <span className="record-type">查看来源 <ChevronDown className="detail-chevron" size={14} aria-hidden="true" /></span>
      </summary>
      <dl className="record-details">
        <div>
          <dt>来源类型</dt>
          <dd>{item.source_kind}</dd>
        </div>
        <div>
          <dt>来源版本</dt>
          <dd>{item.version}</dd>
        </div>
        <div>
          <dt>Chunk</dt>
          <dd>{item.chunk_id ?? '—'}</dd>
        </div>
        <div>
          <dt>内容校验值</dt>
          <dd>{item.content_checksum}</dd>
        </div>
      </dl>
    </details>
  )
}

function FactItem({ item }: { item: BusinessFact }) {
  return (
    <details className="record-row">
      <summary>
        <Database size={18} aria-hidden="true" />
        <span>
          <strong>{item.record_id}</strong>
          <small>{item.tool_name}</small>
        </span>
        <span className="record-type">查看事实 <ChevronDown className="detail-chevron" size={14} aria-hidden="true" /></span>
      </summary>
      <dl className="record-details">
        <div>
          <dt>数据集版本</dt>
          <dd>{item.dataset_version}</dd>
        </div>
        <div>
          <dt>读取状态</dt>
          <dd>{item.status}</dd>
        </div>
        <div>
          <dt>只读</dt>
          <dd>{item.read_only ? '是' : '否'}</dd>
        </div>
        <div>
          <dt>截至时间</dt>
          <dd>{formatTime(item.as_of)}</dd>
        </div>
      </dl>
      <pre>{JSON.stringify(item.result, null, 2)}</pre>
    </details>
  )
}

function recommendationLabel(value: keyof typeof RECOMMENDATION_LABELS): string {
  return RECOMMENDATION_LABELS[value]
}

function stateLabel(value: keyof typeof OBLIGATION_STATE_LABELS): string {
  return OBLIGATION_STATE_LABELS[value]
}

/** Which profile one Run executed under, stated rather than inferred. */
export function ExecutionModeChip({ mode }: { mode: ReviewRun['execution_mode'] }) {
  const real = mode === 'real'
  return (
    <span
      className={`mode-chip mode-${mode}`}
      data-testid="execution-mode"
      data-mode={mode}
    >
      {real ? '真实模型端点' : '合成夹具 · test_only'}
    </span>
  )
}

/**
 * What one rules 3.1.0 execution reported and cost: the two document branches, the
 * deterministic procurement branch, the explicit stop reason (if any) and the metering
 * the ledger recomputes.  The page never derives a number of its own.
 */
export function PreciseExecutionPanel({
  execution,
}: {
  execution: PreciseExecutionView
}) {
  const metering = execution.metering
  const branches = [execution.policy, execution.contract]
  return (
    <section
      className="records-section"
      aria-labelledby="precise-execution-title"
      data-testid="precise-execution"
    >
      <div className="section-heading compact">
        <Activity aria-hidden="true" />
        <div>
          <h2 id="precise-execution-title">执行与计量</h2>
          <p>{`rules ${execution.rules_version} · 计量来自持久账本重算`}</p>
        </div>
      </div>
      <dl className="metering-grid" data-testid="precise-metering">
        <div>
          <dt>尝试行</dt>
          <dd>{metering.physical_attempts}</dd>
        </div>
        <div>
          <dt>已证发送</dt>
          <dd>{metering.proven_sends == null ? '未记录' : metering.proven_sends}</dd>
        </div>
        <div>
          <dt>首次发送</dt>
          <dd>{metering.first_sends}</dd>
        </div>
        <div>
          <dt>合同纠正</dt>
          <dd>
            {metering.contract_corrections == null
              ? '未记录'
              : metering.contract_corrections}
          </dd>
        </div>
        <div>
          <dt>引用纠正</dt>
          <dd>
            {metering.citation_corrections == null
              ? '未记录'
              : metering.citation_corrections}
          </dd>
        </div>
        <div>
          <dt>传输重试</dt>
          <dd>{metering.transport_retries == null ? '未记录' : metering.transport_retries}</dd>
        </div>
        <div>
          <dt>未决尝试</dt>
          <dd>
            {metering.unresolved_attempts == null
              ? '未记录'
              : metering.unresolved_attempts + (metering.open_attempts ?? 0)}
          </dd>
        </div>
        <div>
          <dt>未知用量尝试</dt>
          <dd>{metering.unknown_usage_attempts}</dd>
        </div>
        <div>
          <dt>未返回尝试</dt>
          <dd>{metering.unreturned_attempts}</dd>
        </div>
        <div>
          <dt>复用分支</dt>
          <dd>{metering.reused_branches == null ? '未记录' : metering.reused_branches}</dd>
        </div>
        <div>
          <dt>显式恢复</dt>
          <dd>
            {metering.explicit_recoveries == null ? '未记录' : metering.explicit_recoveries}
          </dd>
        </div>
        <div>
          <dt>已知输入 Token</dt>
          <dd>{metering.input_tokens_known ?? '未知'}</dd>
        </div>
        <div>
          <dt>已知输出 Token</dt>
          <dd>{metering.output_tokens_known ?? '未知'}</dd>
        </div>
        <div>
          <dt>用量完整</dt>
          <dd>
            {metering.usage_complete == null
              ? '未记录'
              : metering.usage_complete
                ? '是'
                : '否（已知数为下界）'}
          </dd>
        </div>
        <div>
          <dt>解析模型</dt>
          <dd>
            {metering.resolved_models.length
              ? metering.resolved_models.join('、')
              : '无'}
          </dd>
        </div>
      </dl>
      <ul className="issue-list">
        {branches.map((branch) => (
          <li key={branch.worker}>
            <strong>{workerLabel(branch.worker)}</strong>
            <span>{`${branch.outcome} · 报告 ${branch.reported} 项 · 引用 ${branch.evidence_refs} 处 · 尝试 ${branch.attempts} 次`}</span>
            {branch.failure_code && <p>{branch.failure_code}</p>}
          </li>
        ))}
        <li>
          <strong>确定性采购分支</strong>
          <span>{`${execution.procurement.status} · ${execution.procurement.tool_calls} 次只读工具调用 · 0 次模型调用`}</span>
        </li>
      </ul>
      {execution.stop_reason && (
        <p className="recovery-detail" data-testid="precise-stop-reason">
          {`显式停止原因：${execution.stop_reason}`}
        </p>
      )}
      <PreciseDecisionPanel decision={execution.decision} />
      {execution.conflicts.length > 0 && (
        <div className="records-section" data-testid="precise-conflicts">
          <h3>业务冲突</h3>
          <ul className="issue-list">
            {execution.conflicts.map((pair) => (
              <li key={pair.pair_id}>
                <strong>{`${pair.kind} · ${pair.relation}`}</strong>
                {[pair.left, pair.right].map((side, index) => (
                  <span key={`${pair.pair_id}-${index}`}>
                    {`${side.check_id} ${side.field_path} = ${String(side.value)}；来源 ${side.source_id} ${side.version} ${side.locator}`}
                  </span>
                ))}
              </li>
            ))}
          </ul>
        </div>
      )}
      {execution.security.length > 0 && (
        <div className="records-section" data-testid="precise-security">
          <h3>安全事件</h3>
          <ul className="issue-list">
            {execution.security.map((event) => (
              <li key={`${event.code}-${event.location_sha256}`}>
                <strong>{event.code}</strong>
                <span>{`来源 ${event.origin} · 位置摘要 ${event.location_sha256.slice(0, 12)}… · 未发生外写`}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}

/** The five-class recommendation and every duty it rests on, with each duty's own reason. */
export function PreciseDecisionPanel({ decision }: { decision: PreciseDecisionView }) {
  return (
    <div className="records-section" data-testid="precise-decision">
      <h3>审查结论</h3>
      <p data-testid="precise-recommendation">{recommendationLabel(decision.recommendation)}</p>
      <ul className="issue-list">
        {decision.findings.map((finding) => (
          <li key={finding.obligation_id}>
            <strong>{`${finding.obligation_id} · ${stateLabel(finding.state)}`}</strong>
            <span>{`${finding.reason} · 风险 ${finding.risk}`}</span>
            {finding.missing.length > 0 && (
              <span>{`缺少 ${finding.missing.join('、')}`}</span>
            )}
            {finding.evidence.length > 0 && (
              <span>
                {finding.evidence
                  .map(
                    (item) => `${item.source_id} ${item.version} ${item.locator.slice(0, 16)}…`,
                  )
                  .join('；')}
              </span>
            )}
          </li>
        ))}
      </ul>
      {decision.pipeline_failures.length > 0 && (
        <p className="recovery-detail">{`流水线失败：${decision.pipeline_failures.join('、')}`}</p>
      )}
    </div>
  )
}

export function ResultView({ run }: { run: ReviewRun }) {
  if (run.status === 'processing') {
    return (
      <section className="processing-state" aria-live="polite" aria-busy="true">
        <LoaderCircle className="spin" aria-hidden="true" />
        <div>
          <h2>三个只读 Worker 正在并行审查</h2>
          <p>结果仍是建议，人工 Gate 尚未开放。</p>
          <ExecutionModeChip mode={run.execution_mode} />
        </div>
      </section>
    )
  }

  if (run.status === 'failed') {
    return (
      <section className="failure-state" role="alert">
        <XCircle aria-hidden="true" />
        <div>
          <h2>Review 运行失败并已关闭</h2>
          <p>{run.error_message}</p>
          <code>{run.error_code}</code>
        </div>
      </section>
    )
  }

  if (run.status === 'interrupted') {
    // A product run state, not a recommendation: no summary exists, so there is
    // nothing to confirm or reject and no receipt to replay.
    const guidance = recoveryGuidance(run)
    const recovery = run.recovery
    const adopted = recovery?.adopted_workers?.length
      ? recovery.adopted_workers.map(workerLabel).join('、')
      : '无（没有可复用的已提交分支）'
    const attempts = recovery && Object.keys(recovery.attempt_totals).length
      ? Object.entries(recovery.attempt_totals)
          .map(([worker, count]) => `${workerLabel(worker)} ${count}`)
          .join('、')
      : '无'
    return (
      <section
        className="interrupted-state"
        role="status"
        data-testid="interrupted-state"
      >
        <div className="interrupted-heading">
          <PauseCircle aria-hidden="true" />
          <div>
            <h2>运行已中断，等待恢复</h2>
            <p>
              本次执行代次没有提交结果，所以没有建议、没有回执，也不会生成任何人工决定。
            </p>
          </div>
        </div>
        <dl className="recovery-grid">
          <div>
            <dt>中断原因</dt>
            <dd data-testid="recovery-reason">{guidance.reason}</dd>
          </div>
          <div>
            <dt>恢复预算</dt>
            <dd data-testid="recovery-budget">{guidance.budget}</dd>
          </div>
          <div>
            <dt>已复用分支</dt>
            <dd data-testid="recovery-adopted">{adopted}</dd>
          </div>
          <div>
            <dt>Worker 尝试次数</dt>
            <dd>{attempts}</dd>
          </div>
          <div>
            <dt>执行代次</dt>
            <dd>{recovery ? recovery.generation : '未知'}</dd>
          </div>
          <div>
            <dt>租约到期</dt>
            <dd>
              {recovery?.lease_expires_at
                ? formatTime(recovery.lease_expires_at)
                : '无活跃租约'}
            </dd>
          </div>
        </dl>
        <p className="recovery-next" data-testid="recovery-next-step">
          <strong>下一步：</strong>
          {guidance.nextStep}
        </p>
        {run.error_message && <p className="recovery-detail">{run.error_message}</p>}
        {run.error_code && <code className="recovery-code">{run.error_code}</code>}
      </section>
    )
  }

  if (run.status === 'cancelled' && !run.summary) {
    return (
      <section className="cancelled-state" role="status" data-testid="cancelled-state">
        <Ban aria-hidden="true" />
        <div>
          <h2>Review 已取消</h2>
          <p>本地执行已停止；未生成建议回执，也没有外部业务写入。</p>
        </div>
      </section>
    )
  }

  if (!run.summary) return null
  const summary = run.summary
  // `pending_confirmation → cancelled` is a legal transition, so a Run can be
  // cancelled *after* its summary was committed.  The workbench says both things at
  // once instead of hiding the cancellation behind an apparently normal result, and
  // it never invents a receipt for a cancellation.
  const cancelledAfterCommit = run.status === 'cancelled'

  return (
    <div className="result-stack">
      {cancelledAfterCommit && (
        <section className="cancelled-state" role="status" data-testid="cancelled-state">
          <Ban aria-hidden="true" />
          <div>
            <h2>Review 已取消</h2>
            <p>
              取消发生在结果提交之后：已提交的建议仍然可以核对，但取消不会生成人工回执，
              也没有外部业务写入。
            </p>
          </div>
        </section>
      )}
      {run.receipt && (
        <ReceiptPanel receipt={run.receipt} />
      )}

      {isPreciseExecution(run.execution) && (
        <PreciseExecutionPanel execution={run.execution} />
      )}

      <section className={`summary-band summary-${summary.recommendation}`} aria-labelledby="summary-title">
        <span className="summary-emblem" aria-hidden="true">{summary.recommendation === 'recommend_approve' ? <CheckCircle2 size={30} strokeWidth={1.5} /> : <ShieldAlert size={30} strokeWidth={1.5} />}</span>
        <div className="recommendation-block">
          <span className="section-kicker">
            系统建议 · 非审批事实
            <ExecutionModeChip mode={run.execution_mode} />
          </span>
          <h2 id="summary-title">
            {RECOMMENDATION_LABELS[summary.recommendation]}
          </h2>
          <p>规则版本 {summary.rules_version} · 结果不可自动触发业务动作</p>
        </div>
        <div className="summary-metrics">
          <div>
            <span>最高风险</span>
            <RiskBadge risk={summary.highest_risk} />
          </div>
          <div>
            <span>审查发现</span>
            <strong>{summary.findings.length}</strong>
          </div>
          <div>
            <span>缺失 / 冲突</span>
            <strong>
              {summary.missing_information.length} / {summary.conflicts.length}
            </strong>
          </div>
        </div>
      </section>

      {(run.branch_failures.length > 0 || summary.failure_reasons.length > 0) && (
        <section className="issue-section failure-section" aria-labelledby="failure-title">
          <div className="section-heading compact">
            <ShieldAlert aria-hidden="true" />
            <div>
              <h2 id="failure-title">Worker 失败</h2>
              <p>失败已进入聚合结果，没有默认建议通过。</p>
            </div>
          </div>
          <ul className="issue-list">
            {run.branch_failures.map((failure) => (
              <li key={`${failure.worker}-${failure.code}`}>
                <strong>{WORKER_LABELS[failure.worker]}</strong>
                <span>{failure.code}</span>
                <p>{failure.detail}</p>
              </li>
            ))}
            {summary.failure_reasons.map((reason) => (
              <li key={reason}>
                <strong>聚合失败原因</strong>
                <p>{reason}</p>
              </li>
            ))}
          </ul>
        </section>
      )}

      {(summary.missing_information.length > 0 || summary.conflicts.length > 0) && (
        <section className="issue-section" aria-labelledby="unresolved-title">
          <div className="section-heading compact">
            <TriangleAlert aria-hidden="true" />
            <div>
              <h2 id="unresolved-title">未决信息</h2>
              <p>缺失信息与证据冲突保留为独立结果。</p>
            </div>
          </div>
          {summary.missing_information.length > 0 && (
            <ul className="plain-list">
              {summary.missing_information.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          )}
          {summary.conflicts.map((conflict) => (
            <article className="conflict-row" key={`${conflict.finding_key}-${conflict.reason}`}>
              <strong>{conflict.finding_key}</strong>
              <span>{conflict.reason}</span>
              <p>{conflict.statements.join(' / ')}</p>
            </article>
          ))}
        </section>
      )}

      <section className="findings-section" aria-labelledby="findings-title">
        <div className="section-heading compact">
          <ListChecks aria-hidden="true" />
          <div>
            <h2 id="findings-title">审查发现 <span className="section-count">{summary.findings.length}</span></h2>
            <p>政策、合同与采购数据的独立审查结论</p>
          </div>
        </div>
        <div className="finding-list">
          {summary.findings.map((finding, index) => (
            <FindingCard key={`${finding.finding_key}-${index}`} finding={finding} index={index} />
          ))}
        </div>
      </section>

      <section className="records-section" aria-labelledby="evidence-title">
        <div className="section-heading compact">
          <FileSearch aria-hidden="true" />
          <div>
            <h2 id="evidence-title">可核验证据</h2>
            <p>{run.evidence_refs.length} 份可定位、可复核的文档引用</p>
          </div>
        </div>
        {run.evidence_refs.length ? (
          <div className="record-list">
            {run.evidence_refs.map((item) => (
              <EvidenceItem key={`${item.source_id}-${item.chunk_id}`} item={item} />
            ))}
          </div>
        ) : (
          <p className="inline-empty">本次结果没有可核验文档引用。</p>
        )}
      </section>

      <section className="records-section" aria-labelledby="facts-title">
        <div className="section-heading compact">
          <Database aria-hidden="true" />
          <div>
            <h2 id="facts-title">只读业务事实</h2>
            <p>{run.business_facts.length} 条来源明确的采购事实</p>
          </div>
        </div>
        {run.business_facts.length ? (
          <div className="record-list">
            {run.business_facts.map((item) => (
              <FactItem key={`${item.tool_name}-${item.record_id}`} item={item} />
            ))}
          </div>
        ) : (
          <p className="inline-empty">本次结果没有可核验采购事实。</p>
        )}
      </section>
    </div>
  )
}

export function TraceView({ trace }: { trace: TraceResponse | null }) {
  if (!trace) {
    return <p className="inline-empty">Trace 正在读取。</p>
  }
  return (
    <section className="timeline-section" aria-labelledby="trace-title">
      <div className="section-heading compact">
        <Activity aria-hidden="true" />
        <div>
          <h2 id="trace-title">运行 Trace</h2>
          <p>
            {trace.trace.trace_schema_version} ·{' '}
            {trace.trace.complete ? '完整' : '等待人工 Gate'}
          </p>
        </div>
      </div>
      <ol className="timeline">
        {trace.trace.events.map((event) => (
          <li key={event.event_id} className={event.error ? 'timeline-error' : ''}>
            <span className="timeline-marker" aria-hidden="true" />
            <div className="timeline-meta">
              <code>{event.event_type}</code>
              <time>{formatTime(event.occurred_at)}</time>
            </div>
            <strong>{event.step}</strong>
            <p>{event.error ?? JSON.stringify(event.output_summary)}</p>
          </li>
        ))}
      </ol>
    </section>
  )
}

export function AuditView({ audit }: { audit: AuditResponse | null }) {
  if (!audit) return <p className="inline-empty">审批审计正在读取。</p>
  return (
    <section className="audit-section" aria-labelledby="audit-title">
      <div className="section-heading compact">
        <History aria-hidden="true" />
        <div>
          <h2 id="audit-title">审批审计</h2>
          <p>{audit.audit_schema_version} · 人工动作与运行生命周期留痕</p>
        </div>
      </div>
      {audit.events.length === 0 ? (
        <div className="audit-empty">
          <ClipboardCheck aria-hidden="true" />
          <p>尚无人工动作，审计记录与本地回执均为空。</p>
        </div>
      ) : (
        <div className="audit-list">
          {audit.events.map((event) => (
            <article key={event.audit_id}>
              <div>
                <span className="action-label">
                  {event.action_type ?? event.event_class}
                </span>
                <strong>{event.actor ?? '（系统事件）'}</strong>
                <time>{formatTime(event.occurred_at)}</time>
              </div>
              <p>
                {event.previous_status
                  ? STATUS_LABELS[event.previous_status]
                  : event.event_class}{' '}
                → {event.new_status ? STATUS_LABELS[event.new_status] : '无状态变更'}
              </p>
              <dl>
                <div>
                  <dt>事件序号</dt>
                  <dd>#{event.sequence}</dd>
                </div>
                <div>
                  <dt>幂等键摘要</dt>
                  <dd>
                    {event.idempotency_digest ? (
                      <code title={event.idempotency_digest}>
                        {compactId(event.idempotency_digest)}
                      </code>
                    ) : (
                      '无（该事件没有幂等键）'
                    )}
                  </dd>
                </div>
                <div>
                  <dt>回执关联</dt>
                  <dd>{event.receipt_id ?? '无'}</dd>
                </div>
                <div>
                  <dt>外部写入</dt>
                  <dd>{event.external_write_performed ? '已执行' : '0 · 未执行'}</dd>
                </div>
              </dl>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}
