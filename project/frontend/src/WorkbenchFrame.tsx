import { Activity, ArrowDownLeft, FileCheck2, FileSearch, FolderOpen, History, Layers3, Search, ShieldCheck, X } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import type { CaseOption, ExecutionMode, ReviewRun, SessionInfo } from './types'
import type { GuideState } from './workbenchGuide'
import { WorkbenchHeader } from './WorkbenchHeader'
import { formatTime, STATUS_LABELS } from './reviewPresentation'

export function WorkbenchFrame({ session, cases, runs, guide, onSignOut, children, executionMode }: {
  session: SessionInfo | null
  cases: CaseOption[]
  executionMode: ExecutionMode | null
  runs: ReviewRun[]
  guide: GuideState
  onSignOut: () => void
  children: ReactNode
}) {
  const pending = runs.filter((run) => ['pending_confirmation', 'pending_human_review'].includes(run.status)).length
  const recorded = runs.filter((run) => run.receipt !== null).length
  const stats = [
    { label: '已授权案例', value: cases.length, unit: '个案例', icon: FolderOpen, scope: 'case:read' },
    { label: '审查运行', value: runs.length, unit: '次运行', icon: Activity, scope: 'review:read' },
    { label: '待人工处理', value: pending, unit: '项待办', icon: FileSearch, scope: 'review:read' },
    { label: '已记录决定', value: recorded, unit: '份本地回执', icon: FileCheck2, scope: 'review:read' },
  ]

  return (
    <div className="app-shell" id="app-top">
      <a className="skip-link" href="#review-content">跳到审查内容</a>
      <div className="workspace-body" id="workspace">
        <WorkbenchHeader session={session} guide={guide} onSignOut={onSignOut} executionMode={executionMode} />

        <div className="workspace-content">
          <section className="page-heading" aria-labelledby="page-title">
            <div><p className="eyebrow"><span />EVIDENCE BEFORE DECISIONS</p><h1 id="page-title">合规审查工作台<span className="preview-badge">本地演示</span></h1><p className="page-description">从证据出发，让每一次决策更清晰。</p></div>
          </section>

          <section className="overview-strip" aria-label="当前授权范围概览">
            {stats.map(({ label, value, unit, icon: Icon, scope }, index) => (
              <article className={`overview-stat stat-${index}`} key={label}>
                <div className="stat-label"><span>{label}</span><Icon size={17} aria-hidden="true" /></div>
                <div className="stat-value"><strong>{session?.scopes.some((granted) => granted === scope) ? String(value).padStart(2, '0') : '—'}</strong><span>{unit}</span></div>
                <span className="stat-description">{index === 0 ? '当前身份可访问' : index === 1 ? '当前已加载的运行' : index === 2 ? '等待你的判断' : '人工动作可追溯'}</span>
              </article>
            ))}
          </section>

          {children}
          <footer className="workspace-footer"><span><ShieldCheck size={14} aria-hidden="true" />只读审查 · 人工决策 · 全程留痕</span><span>LOCAL WORKSPACE / V1.0</span></footer>
        </div>
      </div>
    </div>
  )
}

export function RunList({ runs, cases, activeRunId, signedIn, onSelect }: {
  runs: ReviewRun[]
  cases: CaseOption[]
  activeRunId: string | null
  signedIn: boolean
  onSelect: (id: string) => void
}) {
  const [search, setSearch] = useState('')
  const query = search.trim().toLocaleLowerCase()
  const caseLabel = (run: ReviewRun) => cases.find((item) => item.case_id === run.request.case_id)?.label ?? run.request.case_id
  const filtered = runs.filter((run) => `${run.run_id} ${run.request.case_id} ${caseLabel(run)} ${STATUS_LABELS[run.status]}`.toLocaleLowerCase().includes(query))
  return (
    <nav className="run-nav" id="recent-runs" aria-label="Review 运行">
      <div className="sidebar-heading"><span>最近运行</span><small>{runs.length}</small></div>
      {runs.length > 0 && <div className="run-search"><Search size={16} aria-hidden="true" /><input aria-label="搜索运行记录" placeholder="搜索案例或运行…" value={search} onChange={(event) => setSearch(event.target.value)} />{search && <button type="button" aria-label="清空搜索" onClick={() => setSearch('')}><X size={14} aria-hidden="true" /></button>}</div>}
      {runs.length === 0 ? <div className="run-empty"><History size={24} aria-hidden="true" /><p>{signedIn ? '当前没有 Run。' : '验证凭据后读取运行列表。'}</p><small>运行记录将在这里归档</small></div> : filtered.length === 0 ? <p className="run-empty" role="status">没有匹配的运行，请调整搜索内容。</p> : (
        <ul>{filtered.map((run) => <li key={run.run_id}>
          <button type="button" data-run-id={run.run_id} aria-current={run.run_id === activeRunId ? 'true' : undefined} className={run.run_id === activeRunId ? 'active' : ''} onClick={() => onSelect(run.run_id)}>
            <span className="run-item-heading"><span className="run-file-icon"><FileSearch size={16} aria-hidden="true" /></span><strong>{caseLabel(run)}</strong><ArrowDownLeft size={14} aria-hidden="true" /></span>
            <span className="run-item-id">{run.request.case_id}</span>
            <span className="run-item-footer"><span className={`status-badge status-${run.status}`}>{STATUS_LABELS[run.status]}</span><small>{formatTime(run.created_at)}</small></span>
          </button>
        </li>)}</ul>
      )}
    </nav>
  )
}

export function ReviewIllustration() {
  return <div className="review-illustration" aria-hidden="true"><div className="illustration-grid" /><div className="document-tile tile-policy"><ShieldCheck size={20} /><span>政策制度</span><i /><i /></div><div className="document-tile tile-contract"><FileCheck2 size={20} /><span>合同条款</span><i /><i /></div><div className="document-tile tile-data"><Layers3 size={20} /><span>采购事实</span><i /><i /></div><div className="illustration-ring"><span><ShieldCheck size={36} strokeWidth={1.4} /></span></div><span className="illustration-caption">独立审查，共同印证</span></div>
}
