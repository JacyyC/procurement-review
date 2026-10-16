import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import * as api from './api'
import { RunList } from './WorkbenchFrame'
import type { ReviewRun, SessionInfo } from './types'

const session: SessionInfo = {
  api_version: 'review-api.v2', subject_id: 'navigation-reader', tenant_id: 'local',
  credential_id: 'navigation-credential', authorization_revision: 1,
  roles: ['reader', 'auditor'], scopes: ['case:read', 'review:read', 'trace:read', 'audit:read'],
  issued_at: '2026-09-12T00:00:00Z', expires_at: '2026-09-13T00:00:00Z',
}
const run: ReviewRun = {
  run_id: 'run-navigation', trace_id: 'trace-navigation', status: 'cancelled', version: 2, revision: 2,
  execution_mode: 'test_only', request: { schema_version: '1.0', case_id: 'CASE-NAVIGATION',
    case_type: 'purchase_order', document_version: 'v1', purchase_order_id: 'PO-1', dataset_version: 'v1', initiator: 'navigation-reader' },
  summary: null, summary_checksum: null, execution: null, branch_failures: [], evidence_refs: [], business_facts: [],
  receipt: null, created_at: '2026-09-12T00:00:00Z', updated_at: '2026-09-12T00:00:00Z',
  trace_event_count: 0, audit_event_count: 0, error_code: null, error_message: null, recovery: null,
}

afterEach(() => { cleanup(); vi.restoreAllMocks() })

async function openWorkbench(scopes = session.scopes) {
  vi.spyOn(api, 'getSession').mockResolvedValue({ ...session, scopes })
  vi.spyOn(api, 'getCases').mockResolvedValue({ api_version: 'review-api.v2', execution_mode: 'test_only', cases: [] })
  vi.spyOn(api, 'listReviews').mockResolvedValue({ api_version: 'review-api.v2', runs: [run] })
  vi.spyOn(api, 'getTrace').mockResolvedValue({ api_version: 'review-api.v1', trace: {
    schema_version: '1.0', trace_schema_version: 'trace.v1', trace_id: run.trace_id,
    run_id: run.run_id, events: [], complete: true,
  } })
  vi.spyOn(api, 'getAudit').mockResolvedValue({ api_version: 'review-api.v2', audit_schema_version: 'approval-audit.v2', run_id: run.run_id, events: [] })
  const user = userEvent.setup()
  render(<App />)
  await user.type(screen.getByLabelText('Bearer 凭据'), 'local-navigation-fixture')
  await user.click(screen.getByRole('button', { name: '验证凭据' }))
  await screen.findByRole('heading', { name: 'Review 已取消' })
  return user
}

it('keeps focus, selection and the visible panel together for keyboard tab navigation', async () => {
  const user = await openWorkbench()
  const result = screen.getByRole('tab', { name: '审查结果' })
  const trace = screen.getByRole('tab', { name: /Trace/ })
  const audit = screen.getByRole('tab', { name: /审批审计/ })
  await user.click(result)
  await user.keyboard('{ArrowRight}')
  expect(trace).toHaveFocus()
  expect(trace).toHaveAttribute('aria-selected', 'true')
  expect(screen.getByRole('tabpanel')).toHaveAccessibleName(/运行 Trace/)
  await user.keyboard('{End}')
  expect(audit).toHaveFocus()
  expect(screen.getByRole('heading', { name: '审批审计' })).toBeVisible()
  await user.keyboard('{ArrowRight}')
  expect(result).toHaveFocus()
  await user.keyboard('{ArrowLeft}')
  expect(audit).toHaveFocus()
  await user.keyboard('{Home}')
  expect(result).toHaveFocus()
  await user.keyboard('{Tab}')
  expect(screen.getByRole('tabpanel')).toHaveFocus()
})

it('never exposes unavailable views while navigating with the keyboard', async () => {
  const user = await openWorkbench(['case:read', 'review:read'])
  const result = screen.getByRole('tab', { name: '审查结果' })
  await user.click(result)
  await user.keyboard('{ArrowLeft}{End}{ArrowRight}')
  expect(screen.getAllByRole('tab')).toHaveLength(1)
  expect(result).toHaveFocus()
  expect(api.getTrace).not.toHaveBeenCalled()
  expect(api.getAudit).not.toHaveBeenCalled()
})

it('searches case IDs and statuses, recovers from empty results and opens the selected run', async () => {
  const user = userEvent.setup()
  const onSelect = vi.fn()
  render(<RunList runs={[run, { ...run, run_id: 'run-risk', status: 'failed', request: { ...run.request, case_id: 'CASE-RISK' } }]} cases={[]} activeRunId={run.run_id} signedIn onSelect={onSelect} />)
  const list = screen.getByRole('navigation', { name: 'Review 运行' })
  const input = within(list).getByRole('textbox', { name: '搜索运行记录' })
  await user.type(input, 'case-risk')
  expect(within(list).getAllByRole('listitem')).toHaveLength(1)
  await user.click(within(list).getByRole('button', { name: /CASE-RISK/ }))
  expect(onSelect).toHaveBeenCalledWith('run-risk')
  await user.clear(input)
  await user.type(input, '已取消')
  expect(within(list).getByRole('button', { name: /CASE-NAVIGATION/ })).toHaveAttribute('aria-current', 'true')
  await user.type(input, '没有匹配')
  expect(within(list).getByRole('status')).toHaveTextContent('没有匹配的运行')
  await user.click(within(list).getByRole('button', { name: '清空搜索' }))
  expect(within(list).getAllByRole('listitem')).toHaveLength(2)
})
