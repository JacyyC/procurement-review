/**
 * V1-06 AC-09: every Review API response is validated field by field before the
 * workbench touches it, and a malformed body fails loudly with an explicit code
 * instead of being cast into a shape the UI then mis-renders.
 */

import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  getAudit,
  getReview,
  getSession,
  listReviews,
  ReviewAPIError,
} from './api'

const AUTH = { token: 'rvk1.test.credentialmaterial0000' }

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function stubFetch(body: unknown, status = 200) {
  const fetchMock = vi.fn(async () => jsonResponse(body, status))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function validRun(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    api_version: 'review-api.v2',
    run_id: 'run-001',
    trace_id: 'trace-001',
    status: 'pending_confirmation',
    version: 2,
    revision: 2,
    execution_mode: 'test_only',
    request: {
      schema_version: '1.0',
      case_id: 'CASE-LOW-001',
      case_type: 'purchase_order',
      document_version: 'retrieval-fixture-v1',
      purchase_order_id: 'PO-001',
      dataset_version: 'synthetic-procurement.v1',
      initiator: 'subject-local-demo',
    },
    summary: null,
    summary_checksum: null,
    execution: null,
    branch_failures: [],
    evidence_refs: [],
    business_facts: [],
    receipt: null,
    created_at: '2026-07-01T00:00:00Z',
    updated_at: '2026-07-01T00:00:05Z',
    trace_event_count: 4,
    audit_event_count: 0,
    error_code: null,
    error_message: null,
    recovery: null,
    ...overrides,
  }
}

async function rejection(task: Promise<unknown>): Promise<ReviewAPIError> {
  const error = await task.then(
    () => null,
    (reason: unknown) => reason,
  )
  expect(error).toBeInstanceOf(ReviewAPIError)
  return error as ReviewAPIError
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('Review API response validation', () => {
  it('keeps a well-formed run payload intact for the workbench', async () => {
    stubFetch(validRun())
    const value = await getReview('run-001', AUTH)
    expect(value.run_id).toBe('run-001')
    expect(value.status).toBe('pending_confirmation')
    expect(value.version).toBe(2)
  })

  it('rejects a run whose discriminating status is missing', async () => {
    stubFetch(validRun({ status: undefined }))
    const error = await rejection(getReview('run-001', AUTH))
    expect(error.code).toBe('invalid_response')
    expect(error.message).toContain('run.status')
  })

  it('rejects a status the lifecycle union does not define', async () => {
    stubFetch(validRun({ status: 'auto_approved' }))
    const error = await rejection(getReview('run-001', AUTH))
    expect(error.code).toBe('invalid_response')
    expect(error.details).toMatchObject({ path: 'run.status' })
    expect(error.message).toContain('one of')
  })

  it('rejects a version that arrived as a string instead of a number', async () => {
    stubFetch(validRun({ version: '2' }))
    const error = await rejection(getReview('run-001', AUTH))
    expect(error.code).toBe('invalid_response')
    expect(error.details).toMatchObject({ path: 'run.version', expected: 'a number' })
  })

  it('rejects a run without its nested request envelope', async () => {
    stubFetch(validRun({ request: 'CASE-LOW-001' }))
    const error = await rejection(getReview('run-001', AUTH))
    expect(error.code).toBe('invalid_response')
    expect(error.message).toContain('run.request must be an object')
  })

  it('names the offending entry when a run list carries a malformed item', async () => {
    stubFetch({
      api_version: 'review-api.v2',
      runs: [validRun(), validRun({ run_id: 'run-002', trace_id: null })],
    })
    const error = await rejection(listReviews(AUTH))
    expect(error.code).toBe('invalid_response')
    expect(error.details).toMatchObject({ path: 'review_list.runs[1].trace_id' })
  })

  it('rejects a session whose roles are not the closed enum set', async () => {
    stubFetch({
      api_version: 'review-api.v2',
      subject_id: 'subject-local-demo',
      tenant_id: 'tenant-local',
      credential_id: 'credential-001',
      authorization_revision: 1,
      roles: ['superuser'],
      scopes: ['review:read'],
      issued_at: '2026-07-01T00:00:00Z',
      expires_at: '2026-07-08T00:00:00Z',
    })
    const error = await rejection(getSession(AUTH))
    expect(error.code).toBe('invalid_response')
    expect(error.details).toMatchObject({ path: 'session.roles[0]' })
  })

  it('rejects an unreadable success body instead of treating it as empty', async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response('<html>gateway</html>', {
          status: 200,
          headers: { 'Content-Type': 'text/html' },
        }),
    )
    vi.stubGlobal('fetch', fetchMock)
    const error = await rejection(getAudit('run-001', AUTH))
    expect(error.code).toBe('invalid_response')
    expect(error.message).toContain('unreadable')
  })

  it('still surfaces the server error code for a real failure response', async () => {
    stubFetch(
      {
        error: {
          code: 'review_not_found',
          message: 'No such review.',
          request_id: 'request-001',
          details: {},
        },
      },
      404,
    )
    const error = await rejection(getReview('run-404', AUTH))
    expect(error.code).toBe('review_not_found')
    expect(error.status).toBe(404)
    expect(error.requestId).toBe('request-001')
  })
})
