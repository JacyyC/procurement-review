import { ArrowRight, CircleHelp } from 'lucide-react'
import { useRef } from 'react'
import type { SessionInfo } from './types'
import { guideStep, type GuideState } from './workbenchGuide'
import { focusControl } from './focusControl'

export function QuickGuide({ session, state, onIdentity }: {
  session: SessionInfo | null
  state: GuideState
  onIdentity: () => void
}) {
  const guide = useRef<HTMLDetailsElement>(null)
  const step = guideStep(session, state)
  function proceed() {
    // Re-evaluate at activation, never keep the step captured when the guide opened.
    const target = guideStep(session, state).target
    if (!target) return
    if (guide.current) guide.current.open = false
    if (target === 'identity') onIdentity()
    else focusControl(target === 'credential' ? 'bearer-credential' : 'case-select')
  }
  return (
    <details ref={guide} className="quick-guide" onKeyDown={(event) => {
      if (event.key === 'Escape' && guide.current?.open) {
        event.stopPropagation()
        guide.current.open = false
        guide.current.querySelector('summary')?.focus()
      }
    }}>
      <summary><CircleHelp size={17} aria-hidden="true" />操作指南</summary>
      <div>
        <strong>开启一次完整审查</strong>
        <ol><li>验证本地凭据，读取已授权案例。</li><li>选择案例并运行，查看建议与证据。</li><li>人工确认或拒绝，在审批审计中核对记录。</li></ol>
        <p>所有决定仅在本地模拟，不写入采购系统。</p>
        <p role="status" aria-live="polite" aria-atomic="true">{step.text}</p>
        <button type="button" data-guide-action className="guide-action" disabled={!step.target} aria-busy={!step.target} onClick={proceed}>{step.label}<ArrowRight size={15} aria-hidden="true" /></button>
      </div>
    </details>
  )
}
