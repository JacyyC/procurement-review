import { AlertCircle, Eye, EyeOff, KeyRound, LoaderCircle, LogOut, ShieldCheck } from 'lucide-react'
import { useMemo, type FormEvent } from 'react'
import type { ReviewScope, SessionInfo } from './types'
import { compactId, GATED_CONTROLS, scopeList } from './reviewPresentation'

export function CredentialForm({
  draft,
  visible,
  connecting,
  error,
  onDraftChange,
  onToggleVisible,
  onSubmit,
}: {
  draft: string
  visible: boolean
  connecting: boolean
  error: string | null
  onDraftChange: (value: string) => void
  onToggleVisible: () => void
  onSubmit: () => void
}) {
  return (
    <form
      className="credential-form"
      onSubmit={(event: FormEvent<HTMLFormElement>) => {
        event.preventDefault()
        onSubmit()
      }}
    >
      <div className="credential-heading">
        <KeyRound size={18} aria-hidden="true" />
        <div>
          <strong>本地凭据</strong>
          <small>只保存在当前页面内存中，不写入浏览器存储、Cookie 或 URL。</small>
        </div>
      </div>
      <div className="credential-row">
        <div className="credential-field">
          <label htmlFor="bearer-credential">Bearer 凭据</label>
          <span className="credential-input">
            <input
              id="bearer-credential"
              aria-invalid={Boolean(error)}
              aria-describedby={error ? 'credential-error' : undefined}
              name="bearer-credential"
              type={visible ? 'text' : 'password'}
              value={draft}
              autoComplete="off"
              autoCorrect="off"
              spellCheck={false}
              placeholder="粘贴本地签发的 Bearer 凭据"
              onChange={(event) => onDraftChange(event.target.value)}
              disabled={connecting}
            />
            <button
              className="icon-button small"
              type="button"
              aria-label={visible ? '隐藏凭据' : '显示凭据'}
              aria-pressed={visible}
              title={visible ? '隐藏凭据' : '显示凭据'}
              onClick={onToggleVisible}
              disabled={connecting}
            >
              {visible ? <EyeOff size={16} aria-hidden="true" /> : <Eye size={16} aria-hidden="true" />}
            </button>
          </span>
        </div>
        <button
          className="primary-button"
          type="submit"
          disabled={connecting || draft.trim().length === 0}
        >
          {connecting ? (
            <LoaderCircle className="spin" size={18} aria-hidden="true" />
          ) : (
            <ShieldCheck size={18} aria-hidden="true" />
          )}
          验证凭据
        </button>
      </div>
      {error && (
        <p id="credential-error" className="credential-error" role="alert">
          <AlertCircle size={16} aria-hidden="true" />
          {error}
        </p>
      )}
    </form>
  )
}

export function IdentityPanel({
  session,
  onSignOut,
}: {
  session: SessionInfo
  onSignOut: () => void
}) {
  const roles = useMemo(() => [...session.roles].sort(), [session.roles])
  const hidden = useMemo(() => {
    const granted = new Set<ReviewScope>(session.scopes)
    return GATED_CONTROLS.filter(([scope]) => !granted.has(scope))
  }, [session.scopes])
  return (
    <section className="identity-panel" aria-label="已验证身份">
      <div className="identity-heading">
        <ShieldCheck size={18} aria-hidden="true" />
        <div>
          <strong>已验证身份</strong>
          <small title={session.credential_id}>
            凭据 {'•'.repeat(12)} · {compactId(session.credential_id)}
          </small>
        </div>
      </div>
      <dl className="identity-grid">
        <div>
          <dt>主体</dt>
          <dd data-testid="identity-subject">{session.subject_id}</dd>
        </div>
        <div>
          <dt>租户</dt>
          <dd data-testid="identity-tenant">{session.tenant_id}</dd>
        </div>
        <div>
          <dt>凭据 ID</dt>
          <dd>{session.credential_id}</dd>
        </div>
        <div>
          <dt>授权修订</dt>
          <dd>{session.authorization_revision}</dd>
        </div>
        <div>
          <dt>角色</dt>
          <dd>{roles.join('、') || '无'}</dd>
        </div>
        <div className="identity-scopes">
          <dt>授权范围</dt>
          <dd data-testid="identity-scopes">{scopeList(session)}</dd>
        </div>
      </dl>
      <div className="identity-footer">
        <p>
          服务端按已验证凭据与授权范围裁决；界面隐藏或禁用控件只是可用性提示，不构成授权。
        </p>
        <button className="icon-text-button" type="button" onClick={onSignOut}>
          <LogOut size={17} aria-hidden="true" />
          退出并清除
        </button>
      </div>
      {hidden.length > 0 && (
        <p className="scope-hint" data-testid="hidden-controls">
          未授予 {hidden.map(([scope]) => scope).join('、')}，
          {hidden.map(([, label]) => label).join('、')} 控件已隐藏。
        </p>
      )}
    </section>
  )
}
