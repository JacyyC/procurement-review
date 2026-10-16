import type { Permissions, SessionInfo } from './types'

export type GuideState = {
  permissions: Pick<Permissions, 'caseRead' | 'create'>
  connecting: boolean
  loading: boolean
  creating: boolean
  loadFailed: boolean
  hasCases: boolean
}
type Step = { text: string; label: string; target: 'credential' | 'identity' | 'case' | null }

/** Presentation only; all permission and request facts are supplied by App. */
export function guideStep(session: SessionInfo | null, state: GuideState): Step {
  if (!session) return state.connecting
    ? { text: '正在验证凭据…', label: '正在验证凭据…', target: null }
    : { text: '请先验证本地凭据，读取已授权案例。', label: '验证凭据，开始审查', target: 'credential' }
  if (!state.permissions.caseRead || !state.permissions.create) return {
    text: '当前身份无法发起审查；缺少案例读取或审查创建权限。', label: '查看权限', target: 'identity',
  }
  if (state.loading) return { text: '正在加载案例…', label: '正在加载案例…', target: null }
  if (state.creating) return { text: '正在发起审查…', label: '正在发起审查…', target: null }
  if (state.loadFailed) return {
    text: '工作台读取失败。可从身份面板退出后重新验证凭据。', label: '查看身份，重新连接', target: 'identity',
  }
  if (!state.hasCases) return {
    text: '当前没有已授权案例。可核对当前身份和权限，或退出后重新验证凭据。', label: '查看身份与权限', target: 'identity',
  }
  return { text: '选择已授权案例，再由你发起审查。', label: '选择案例，发起审查', target: 'case' }
}
