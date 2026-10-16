import { useEffect } from 'react'

import { ReviewAPIError, getReview } from './api'
import type { RequestAuth } from './api'
import type { ProjectionSource } from './projectionOrdering'
import type { Permissions, RequestOutcome, ReviewRun } from './types'

/**
 * How often a Run that is still `processing` is re-read (exported as the contract the
 * unit and browser suites assert against).
 */
export const RUN_POLL_INTERVAL_MS = 150

/**
 * A `5xx`, `429` or transport failure while a Run is still processing is transient
 * and worth another poll tick; a definitive `4xx` answer is not.
 */
function isTransientPollError(error: unknown): boolean {
  if (error instanceof ReviewAPIError) {
    return error.status >= 500 || error.status === 429
  }
  return true
}

export interface RunPollingOptions {
  processingRunId: string | null
  token: string | null
  permissions: Permissions
  /** Allocate the monotonic request sequence a projection is stamped with. */
  nextSequence: () => number
  request: <T>(
    token: string,
    task: (auth: RequestAuth) => Promise<T>,
  ) => Promise<RequestOutcome<T>>
  /**
   * Apply one Run projection through the single ordering entry point.  It returns
   * whether the projection was accepted; a refused projection starts no follow-up work.
   */
  applyProjection: (
    runId: string,
    sequence: number,
    next: ReviewRun,
    source: ProjectionSource,
  ) => boolean
  handleRequestError: (
    error: unknown,
    runId: string | null,
    sourceTicket?: number,
  ) => void
  refreshArtifacts: (
    token: string,
    runId: string,
    permissions: Permissions,
  ) => Promise<void>
}

/**
 * Poll only while the active run is processing.  The deps carry the stable run
 * identity (not the per-tick run object), so the interval survives the re-renders its
 * own results cause instead of being rebuilt every tick.
 *
 * Every tick takes its own projection sequence, so a tick that resolves after a newer
 * manual refresh — or after a write response, or after a reconciliation read — is a
 * no-op instead of a rollback.  A storage outage can answer a poll with a bare 500
 * while the Run is still running; stopping there would freeze the workbench at 处理中
 * forever, so a transient failure keeps watching, a definitive 4xx ends the watch, and
 * a 401 has already ended the session, which unmounts this effect.
 */
export function useRunPolling(options: RunPollingOptions): void {
  const {
    processingRunId,
    token,
    permissions,
    nextSequence,
    request,
    applyProjection,
    handleRequestError,
    refreshArtifacts,
  } = options
  useEffect(() => {
    if (!processingRunId || !token) return
    let stopped = false
    const timer = window.setInterval(async () => {
      const sequence = nextSequence()
      const outcome = await request(token, (auth) => getReview(processingRunId, auth))
      if (stopped || outcome.status === 'stale') return
      if (outcome.status === 'error') {
        handleRequestError(outcome.error, processingRunId, sequence)
        if (!isTransientPollError(outcome.error)) window.clearInterval(timer)
        return
      }
      // Cache-only: a poll tick must never move the human's selection, even when it
      // resolves after they switched to another Run.  A tick that a newer projection
      // has already overtaken starts no follow-up work; the next tick simply applies.
      if (!applyProjection(processingRunId, sequence, outcome.value, 'read')) return
      if (outcome.value.status !== 'processing') {
        window.clearInterval(timer)
        await refreshArtifacts(token, outcome.value.run_id, permissions)
      }
    }, RUN_POLL_INTERVAL_MS)
    return () => {
      stopped = true
      window.clearInterval(timer)
    }
  }, [
    processingRunId,
    token,
    permissions,
    nextSequence,
    request,
    applyProjection,
    handleRequestError,
    refreshArtifacts,
  ])
}
