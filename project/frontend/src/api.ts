/**
 * Authenticated `review-api.v2` client.
 *
 * Every call takes an explicit in-memory token and an optional `AbortSignal`.
 * The token is only ever placed in an `Authorization` header: it is never logged,
 * persisted, or embedded in a URL.  Aborted requests surface as
 * `ReviewRequestAborted` so a caller can drop responses from a superseded
 * identity instead of applying them.
 *
 * The transport is deliberately thin.  Every success body and every error body is
 * handed to the closed runtime contract in `apiContract.ts` / `apiErrors.ts` before
 * a caller can touch it: no `as T` coercion, no default filling, no silently
 * tolerated shape.  A malformed payload raises `ReviewAPIError` with code
 * `invalid_response` naming the offending field, instead of propagating a
 * silently wrong shape into React state.
 *
 * Resource identity is part of that contract: `getReview`, `getTrace` and `getAudit`
 * pass the requested run id down to the validator, so a structurally valid payload
 * describing a *different* run is rejected rather than rendered under the wrong
 * selection.
 */

import {
  isAbortError,
  isUnauthenticated,
  parseErrorEnvelope,
  ReviewAPIError,
  ReviewRequestAborted,
  withResponseStatus,
} from './apiErrors'
import {
  validateAudit,
  validateCases,
  validateReviewList,
  validateRun,
  validateSession,
  validateTrace,
} from './apiContract'
import { API_VERSION_V2 } from './types'
import type {
  AuditResponse,
  CaseListResponse,
  ReviewListResponse,
  ReviewRun,
  SessionInfo,
  TraceResponse,
} from './types'

const API_ROOT = '/api/v2'

/** In-memory credential plus the cancellation channel for one request. */
export interface RequestAuth {
  token: string
  signal?: AbortSignal
}

export type ReviewAction = 'confirm' | 'reject' | 'cancel'

export interface ActionExpectation {
  /** The `version` of the run payload that was rendered to the human. */
  expectedVersion: number
  /**
   * The `summary_checksum` of that same payload. Required by confirm/reject and
   * rejected by cancel.
   */
  summaryChecksum?: string | null
}

// ------------------------------------------------------------------ transport

async function readBody(response: Response): Promise<unknown> {
  const text = await response.text()
  if (!text.trim()) return null
  try {
    return JSON.parse(text) as unknown
  } catch {
    return null
  }
}

async function apiRequest<T>(
  path: string,
  auth: RequestAuth,
  validate: (body: unknown) => T,
  init?: RequestInit,
): Promise<T> {
  const headers = new Headers(init?.headers)
  headers.set('Accept', 'application/json')
  headers.set('Authorization', `Bearer ${auth.token}`)
  if (init?.body) headers.set('Content-Type', 'application/json')

  let response: Response
  try {
    response = await fetch(`${API_ROOT}${path}`, {
      ...init,
      headers,
      signal: auth.signal,
    })
  } catch (error) {
    if (auth.signal?.aborted || isAbortError(error)) {
      throw new ReviewRequestAborted()
    }
    throw error
  }

  if (auth.signal?.aborted) throw new ReviewRequestAborted()
  const body = await readBody(response)
  if (!response.ok) throw parseErrorEnvelope(response.status, body)
  if (body === null) {
    // A 2xx with an empty or unparseable body is not an empty result: report it as
    // `invalid_response` while keeping the real status.
    throw new ReviewAPIError(
      response.status,
      'invalid_response',
      'The Review API returned an unreadable response body.',
      'unknown',
      { expected: 'a JSON body' },
    )
  }
  try {
    return validate(body)
  } catch (error) {
    // A validator runs before the caller knows the status, so it reports `200`.  Re-stamp
    // the real one: a 202 create or resume must not be reported as a 200.
    throw withResponseStatus(error, response.status)
  }
}

/** `GET /api/v2/session` — anonymous-adjacent identity probe; 401 means unusable. */
export function getSession(auth: RequestAuth): Promise<SessionInfo> {
  return apiRequest('/session', auth, validateSession)
}

/** `GET /api/v2/cases` — only the Cases this credential is granted. */
export function getCases(auth: RequestAuth): Promise<CaseListResponse> {
  return apiRequest('/cases', auth, validateCases)
}

/** `GET /api/v2/reviews` */
export function listReviews(auth: RequestAuth): Promise<ReviewListResponse> {
  return apiRequest('/reviews', auth, validateReviewList)
}

/** `GET /api/v2/reviews/{run_id}` — the payload must describe exactly this run. */
export function getReview(runId: string, auth: RequestAuth): Promise<ReviewRun> {
  return apiRequest(`/reviews/${encodeURIComponent(runId)}`, auth, (body) =>
    validateRun(body, 'run', { expectRunId: runId }),
  )
}

/**
 * `POST /api/v2/reviews` — the Case version is pinned by the caller; identity,
 * tenant, order, dataset and model fixtures are all server-derived.
 *
 * The response is bound to the Case that was frozen before the request was sent:
 * a structurally valid payload for another Case is an `invalid_response`, so it can
 * neither be cached nor selected as this request's review.
 */
export function createReview(
  caseId: string,
  caseVersion: string,
  idempotencyKey: string,
  auth: RequestAuth,
): Promise<ReviewRun> {
  return apiRequest(
    '/reviews',
    auth,
    (body) => validateRun(body, 'run', { expectCaseId: caseId }),
    {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey },
      body: JSON.stringify({
        api_version: API_VERSION_V2,
        case_id: caseId,
        case_version: caseVersion,
      }),
    },
  )
}

/**
 * `POST /api/v2/reviews/{run_id}/{action}`.
 *
 * The body carries no actor: the server records the verified subject.  A replay
 * must resend the identical body and idempotency key, so callers should keep the
 * exact expectation they used the first time.
 */
export function applyReviewAction(
  runId: string,
  action: ReviewAction,
  expectation: ActionExpectation,
  idempotencyKey: string,
  auth: RequestAuth,
): Promise<ReviewRun> {
  const body: Record<string, unknown> = {
    api_version: API_VERSION_V2,
    expected_version: expectation.expectedVersion,
  }
  if (action !== 'cancel') {
    body.expected_summary_checksum = expectation.summaryChecksum ?? null
  }
  return apiRequest(
    `/reviews/${encodeURIComponent(runId)}/${action}`,
    auth,
    (payload) => validateRun(payload, 'run', { expectRunId: runId }),
    {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey },
      body: JSON.stringify(body),
    },
  )
}

/**
 * `POST /api/v2/reviews/{run_id}/resume` — requires `review:resume`.
 *
 * `expected_version` is mandatory and is the `version` of the run payload the human
 * was shown.  The server answers `202` when it opened a new execution generation
 * and `200` when the same request was replayed, so callers must send a fresh
 * idempotency key per human click and never retry on their own.
 */
export function resumeReview(
  runId: string,
  expectedVersion: number,
  idempotencyKey: string,
  auth: RequestAuth,
): Promise<ReviewRun> {
  return apiRequest(
    `/reviews/${encodeURIComponent(runId)}/resume`,
    auth,
    (body) => validateRun(body, 'run', { expectRunId: runId }),
    {
      method: 'POST',
      headers: { 'Idempotency-Key': idempotencyKey },
      body: JSON.stringify({
        api_version: API_VERSION_V2,
        expected_version: expectedVersion,
      }),
    },
  )
}

/**
 * `GET /api/v2/reviews/{run_id}/trace` — requires `trace:read`.
 *
 * `expectedTraceId` is passed when the caller already holds the Run's own `trace_id`, so a
 * self-consistent but differently-anchored envelope is rejected rather than rendered.
 */
export function getTrace(
  runId: string,
  auth: RequestAuth,
  expectedTraceId?: string,
): Promise<TraceResponse> {
  return apiRequest(`/reviews/${encodeURIComponent(runId)}/trace`, auth, (body) =>
    validateTrace(body, 'trace_response', runId, expectedTraceId),
  )
}

/** `GET /api/v2/reviews/{run_id}/audit` — requires `audit:read`. */
export function getAudit(runId: string, auth: RequestAuth): Promise<AuditResponse> {
  return apiRequest(`/reviews/${encodeURIComponent(runId)}/audit`, auth, (body) =>
    validateAudit(body, 'audit_response', runId),
  )
}

export {
  isAbortError,
  isUnauthenticated,
  parseErrorEnvelope,
  ReviewAPIError,
  ReviewRequestAborted,
}
export { API_ROOT }
