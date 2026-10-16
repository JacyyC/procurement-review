import { ChevronRight, Layers3, X } from 'lucide-react'
import { useLayoutEffect, useRef, useState } from 'react'
import type { ExecutionMode, SessionInfo } from './types'
import { QuickGuide } from './QuickGuide'
import type { GuideState } from './workbenchGuide'
import { IdentityPanel } from './SessionPanels'

import { focusControl } from './focusControl'

export function WorkbenchHeader({ session, guide, onSignOut, executionMode }: {
  session: SessionInfo | null
  guide: GuideState
  /** What the API reported it is running: the label never guesses the profile. */
  executionMode: ExecutionMode | null
  onSignOut: () => void
}) {
  const credential = session?.credential_id
  const [disclosure, setDisclosure] = useState({ credential, open: false })
  if (disclosure.credential !== credential) setDisclosure({ credential, open: false })
  const identityOpen = disclosure.credential === credential && disclosure.open && Boolean(session)
  const setIdentityOpen = (open: boolean) => setDisclosure({ credential, open })
  const trigger = useRef<HTMLButtonElement>(null)
  const panel = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    if (!identityOpen) return
    panel.current?.focus({ preventScroll: true })
    panel.current?.scrollIntoView({ block: 'nearest', behavior: 'instant' })
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { setDisclosure({ credential, open: false }); trigger.current?.focus() }
    }
    document.addEventListener('keydown', escape)
    return () => document.removeEventListener('keydown', escape)
  }, [identityOpen, credential])
  function closeIdentity() {
    setIdentityOpen(false)
    trigger.current?.focus()
  }
  return (
    <div className="workbench-header">
      <header className="topbar">
        <a className="brand-lockup" href="#app-top" aria-label="Review Desk 首页">
          <span className="brand-mark"><Layers3 size={25} aria-hidden="true" /></span>
          <span><strong>Review Desk</strong><small>采购与合同合规审查</small></span>
        </a>
        <span className="environment-info" data-testid="environment-info" data-mode={executionMode ?? 'unknown'}>
          <span className="environment-dot" />
          <span>
            {executionMode === 'real' ? '真实模型环境' : executionMode === 'test_only' ? '本地演示环境' : '环境未知'}
            <small>{executionMode === 'real' ? 'REAL ENDPOINT · 真实请求计入计量' : executionMode === 'test_only' ? 'TEST_ONLY · LOCAL' : 'MODE UNKNOWN'}</small>
          </span>
        </span>
        <div className="topbar-status">
          <span className="decision-note">建议不是审批事实</span>
          <QuickGuide session={session} state={guide} onIdentity={() => { setIdentityOpen(true); if (identityOpen) focusControl('identity-panel') }} />
          <button ref={trigger} type="button" className="account-link" aria-expanded={session ? identityOpen : undefined} aria-controls={session ? 'identity-panel' : 'bearer-credential'} onClick={() => session ? setIdentityOpen(!identityOpen) : focusControl('bearer-credential')}>
            <span className={`account-dot ${session ? 'connected' : ''}`} />{session ? '身份与权限' : '连接工作台'}<ChevronRight size={14} aria-hidden="true" />
          </button>
        </div>
      </header>
      {session && <div id="identity-panel" ref={panel} className="header-identity" hidden={!identityOpen} tabIndex={-1}>
        <button type="button" className="icon-text-button identity-close" onClick={closeIdentity}><X size={16} aria-hidden="true" />关闭身份面板</button>
        <IdentityPanel session={session} onSignOut={onSignOut} />
      </div>}
    </div>
  )
}
