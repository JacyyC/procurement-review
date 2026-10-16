import type { CaseOption, RecoveryInfo, ReviewRun, ReviewScope, ReviewStatus, RiskLevel, SessionInfo } from './types'

export const STATUS_LABELS: Record<ReviewStatus, string> = {
  processing: '处理中',
  pending_confirmation: '待人工确认',
  pending_human_review: '待人工复核',
  confirmed: '已确认建议（本地）',
  rejected: '已拒绝建议（本地）',
  cancelled: '已取消',
  failed: '运行失败',
  // A product run state: no recommendation, no receipt, only a resume offer.
  interrupted: '已中断 · 等待恢复',
}

export const RECOMMENDATION_LABELS = {
  recommend_approve: '建议通过',
  recommend_conditional: '建议有条件通过',
  recommend_reject: '建议拒绝',
  insufficient_information: '信息不足',
  needs_human_review: '需要人工复核',
} as const

/** The four obligation states one rules 3.1.0 duty can carry. */
export const OBLIGATION_STATE_LABELS = {
  supported: '已支持',
  violated: '已违反',
  unknown: '未知',
  not_applicable: '不适用',
} as const

export const RISK_LABELS: Record<RiskLevel, string> = {
  unknown: '未知',
  low: '低',
  medium: '中',
  high: '高',
  critical: '严重',
}

export const WORKER_LABELS = {
  policy: '政策规则',
  contract: '合同条款',
  procurement_data: '采购数据',
} as const

export const CASE_TYPE_LABELS: Record<CaseOption['case_type'], string> = {
  purchase_order: '采购订单',
  contract_change: '合同变更',
}

/** Scope -> the control it enables, for the "what is hidden" usability hint. */
export const GATED_CONTROLS: ReadonlyArray<[ReviewScope, string]> = [
  ['review:create', '运行审查'],
  ['review:resume', '恢复审查'],
  ['review:decide', '确认/拒绝'],
  ['review:cancel', '取消'],
  ['trace:read', 'Trace'],
  ['audit:read', '审批审计'],
]

export const SCENARIO_LABELS: Record<CaseOption['scenario'], string> = {
  low: '低风险完整证据',
  high: '高风险政策例外',
  conflict: '证据结论冲突',
  unknown: '关键证据缺失',
  partial_failure: '单 Worker 失败',
  all_failure: '全部 Worker 失败',
  version_mismatch: '数据版本不匹配',
  cancellable: '处理中取消',
  runtime_failure: '运行时失败',
}

export function formatTime(value: string): string {
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).format(new Date(value))
}

export function compactId(value: string): string {
  return value.length > 22 ? `${value.slice(0, 12)}…${value.slice(-7)}` : value
}

export function workerLabel(worker: string): string {
  return (WORKER_LABELS as Record<string, string>)[worker] ?? worker
}

/**
 * Interruption reasons the backend may put in `error_code`.  Anything unknown is
 * rendered verbatim: the UI never invents a friendlier cause than the server gave.
 */
export const INTERRUPTION_REASON_LABELS: Record<string, string> = {
  lease_expired: '上次执行租约已失效',
  generation_deadline_exceeded: '本次执行已超过截止时间',
  storage_unavailable: '存储暂时不可用，执行已停止',
  execution_stopped_without_result: '执行已停止且没有提交结果',
  checkpoint_incompatible: '检查点不可信或版本不符',
  resume_budget_exhausted: '恢复预算已用尽',
}

export const CREATE_NEW_REVIEW_STEP = '创建同一 Case 的新审查（不复制旧回执）'

interface RecoveryGuidance {
  /** Plain-Chinese explanation of why this run stopped. */
  reason: string
  /** `恢复 0 / 2` — the operator-visible resume budget. */
  budget: string
  /** What the operator must do next, in Chinese. */
  nextStep: string
  budgetExhausted: boolean
}

/**
 * Derive the operator-facing recovery explanation.  `resumable` always comes from
 * the server; the client only decides how to phrase it.
 */
export function recoveryGuidance(run: ReviewRun): RecoveryGuidance {
  const recovery: RecoveryInfo | null = run.recovery
  const code = run.error_code ?? recovery?.interrupted_reason ?? null
  const budgetExhausted = recovery ? recovery.resume_count >= recovery.resume_limit : false
  const reason = budgetExhausted
    ? INTERRUPTION_REASON_LABELS.resume_budget_exhausted
    : code
      ? INTERRUPTION_REASON_LABELS[code] ?? code
      : '服务端未提供中断原因'

  if (!recovery) {
    return {
      reason,
      budget: '服务端未返回恢复信息',
      nextStep: `当前没有可用的恢复信息，无法从检查点继续；请${CREATE_NEW_REVIEW_STEP}。`,
      budgetExhausted: false,
    }
  }

  let nextStep: string
  if (recovery.resumable) {
    nextStep = '可从已提交的检查点恢复：点击“恢复审查”继续执行本次 Run。'
  } else if (budgetExhausted) {
    nextStep = `恢复预算已用尽：请${CREATE_NEW_REVIEW_STEP}。`
  } else if (code === 'checkpoint_incompatible' || code === 'failed') {
    nextStep = '检查点不可信或版本不符，需新建审查。'
  } else {
    nextStep = `当前不可恢复：请${CREATE_NEW_REVIEW_STEP}。`
  }

  return {
    reason,
    budget: `恢复 ${recovery.resume_count} / ${recovery.resume_limit}`,
    nextStep,
    budgetExhausted,
  }
}

export function scopeList(session: SessionInfo): string {
  return [...session.scopes].sort().join('、')
}
