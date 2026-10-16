/**
 * Error vocabulary of the authenticated `review-api.v2` client.
 *
 * Kept in its own module so the transport (`api.ts`) and the response contract
 * (`apiContract.ts`) can both raise and classify the same errors without a
 * circular import.  Nothing here ever carries a response body verbatim: a parsed envelope
 * keeps only its (locally mapped) `code` and its request id, and those are displayed only
 * through the fixed vocabulary in `App.tsx`.
 */

export class ReviewAPIError extends Error {
  readonly code: string
  readonly requestId: string
  readonly status: number
  readonly details: Record<string, unknown>

  constructor(
    status: number,
    code: string,
    message: string,
    requestId = 'unknown',
    details: Record<string, unknown> = {},
  ) {
    super(message)
    this.name = 'ReviewAPIError'
    this.code = code
    this.requestId = requestId
    this.status = status
    this.details = details
  }
}

/** Raised when a request was superseded by an identity change or a newer read. */
export class ReviewRequestAborted extends Error {
  constructor() {
    super('The request was superseded before it completed.')
    this.name = 'ReviewRequestAborted'
  }
}

export function isAbortError(error: unknown): boolean {
  if (error instanceof ReviewRequestAborted) return true
  if (!(error instanceof Error)) return false
  return error.name === 'AbortError' || error.name === 'TimeoutError'
}

/** 401 is the only "this credential is unusable" signal in v2. */
export function isUnauthenticated(error: unknown): boolean {
  return (
    error instanceof ReviewAPIError &&
    (error.status === 401 || error.code === 'unauthenticated')
  )
}

/** A malformed *success* body: the server said 2xx but the payload is unusable. */
export function invalidResponse(
  path: string,
  expected: string,
  message?: string,
): ReviewAPIError {
  return new ReviewAPIError(
    200,
    'invalid_response',
    message ??
      `The Review API returned a malformed response: ${path} must be ${expected}.`,
    'unknown',
    { path, expected },
  )
}

// ------------------------------------------------------- error envelope parsing
//
// The v2 error envelope is exactly `{ error: { code, message, request_id, details } }`.
// It is validated like any other response — presence *and* type for every field, and no
// unexplained extra key at either level — so a broken gateway cannot be dressed up as a
// business outcome.  Anything that does not match becomes `invalid_response` while
// keeping the real HTTP status.
//
// Two fields are deliberately *dropped* after validation: the server's `message` and its
// `details`.  They are unverified text, and an error object is something that gets logged;
// keeping them would recreate the leak this repair removes.  The `code` is kept because
// the UI only ever maps it through a local vocabulary, and the `request_id` because the
// UI shows it only when it matches the local identifier contract below.

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0
}

/** Every envelope the API actually serves uses a lowercase snake_case code. */
const ERROR_CODE_SHAPE = /^[a-z][a-z0-9_]{0,63}$/

/**
 * What the UI is allowed to render as a request id.  The value is still validated as a
 * non-empty string first; this narrower contract decides whether it may be *displayed*.
 */
const REQUEST_ID_SHAPE = /^[A-Za-z0-9._:-]{1,64}$/

export function isDisplayableRequestId(value: string): boolean {
  return value !== 'unknown' && REQUEST_ID_SHAPE.test(value)
}

const ENVELOPE_KEYS = ['code', 'details', 'message', 'request_id'] as const

export function parseErrorEnvelope(status: number, body: unknown): ReviewAPIError {
  const invalid = (): ReviewAPIError =>
    new ReviewAPIError(
      status,
      'invalid_response',
      `The Review API returned HTTP ${status} without a readable error envelope.`,
      'unknown',
      { expected: 'error{code,message,request_id,details}' },
    )

  if (!isPlainObject(body)) return invalid()
  const topLevel = Object.keys(body)
  if (topLevel.length !== 1 || topLevel[0] !== 'error') return invalid()

  const raw = body.error
  if (!isPlainObject(raw)) return invalid()
  const keys = Object.keys(raw).sort()
  if (
    keys.length !== ENVELOPE_KEYS.length ||
    keys.some((key, index) => key !== ENVELOPE_KEYS[index])
  ) {
    return invalid()
  }

  const { code, message, request_id: requestId, details } = raw
  if (!isNonEmptyString(code) || !ERROR_CODE_SHAPE.test(code)) return invalid()
  if (!isNonEmptyString(message)) return invalid()
  if (!isNonEmptyString(requestId)) return invalid()
  if (!isPlainObject(details)) return invalid()

  return new ReviewAPIError(
    status,
    code,
    `The Review API returned an error envelope for this request (HTTP ${status}).`,
    requestId,
    {},
  )
}

/**
 * Re-stamp an `invalid_response` with the HTTP status the response really carried.
 *
 * `invalidResponse()` builds its error before the caller knows the status, so every
 * malformed **success** body used to be reported as `200` — which hides the difference
 * between a 200 and the 202 a create or a resume actually answers with.
 */
export function withResponseStatus(error: unknown, status: number): unknown {
  if (
    error instanceof ReviewAPIError &&
    error.code === 'invalid_response' &&
    error.status !== status
  ) {
    return new ReviewAPIError(status, error.code, error.message, error.requestId, error.details)
  }
  return error
}
