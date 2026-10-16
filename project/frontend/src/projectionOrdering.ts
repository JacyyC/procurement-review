/**
 * One ordering decision for every asynchronous source of a `ReviewRun` projection.
 *
 * `revision` orders a Run's lifecycle, but its *progress* counters move without it
 * changing, so two responses can describe the same `revision` while one of them is
 * strictly newer.  FIX-06C handled that between two *reads* only, with a per-Run read
 * ticket.  The 2026-09-17 review reproduced what was left: the write success and the
 * create/action reconciliation wrote the cache directly, so an earlier detail GET that
 * arrived afterwards was still accepted and rolled the displayed trace count from 8
 * back to 3.
 *
 * Every source now goes through this module — select, manual refresh, poll tick, write
 * success, explicit idempotent replay, create/list reconciliation, action and resume
 * reconciliation, and the render-boundary recovery read.  The rules are closed-form:
 *
 *  1. a projection stamped with another identity generation is refused;
 *  2. a projection whose `run_id` is not the Run the caller asked for is refused by the
 *     caller (the API layer binds responses to the requested Run and Case);
 *  3. a **lower** revision is refused, however new the request that produced it;
 *  4. a **higher** revision is accepted;
 *  5. at the same revision, a projection is accepted only when its request sequence is
 *     newer than the newest sequence already applied for that Run.  An earlier request
 *     therefore cannot overwrite a projection a newer request already delivered, while
 *     a genuinely newer request still can.
 *
 * The sequence is one monotonic counter per identity, shared by reads and writes, and
 * taken when the request is *sent*.  Wall clocks, random values and guesses based on how
 * large a payload's dynamic fields are cannot order two responses the client actually
 * received in a known order, so none of them is used here.
 *
 * The watermark never moves backwards.  Accepting a higher revision that an older
 * request happened to deliver therefore cannot reopen the door to projections that a
 * newer request had already superseded.
 */

import type { ReviewRun } from './types'

/** Where one projection came from.  It never changes the rule, only the diagnostics. */
export type ProjectionSource = 'read' | 'write' | 'reconcile' | 'list'

/** One candidate write of a `ReviewRun` into the cache. */
export interface RunProjection {
  /** Identity generation the request that produced this projection was sent under. */
  generation: number
  /** The Run this projection describes. */
  runId: string
  /** Monotonic request sequence, taken when the request was sent. */
  sequence: number
  /** The server revision the projection carries. */
  revision: number
  source: ProjectionSource
}

/** What has already been applied for one Run. */
export interface RunWatermark {
  revision: number
  sequence: number
}

/** Why one projection was refused; used by tests and by nothing else. */
export type ProjectionRefusal =
  | 'identity'
  | 'lower-revision'
  | 'overtaken'

export interface ProjectionDecision {
  accepted: boolean
  /** `null` exactly when the projection was accepted. */
  refusal: ProjectionRefusal | null
}

/**
 * The per-identity ordering state: one watermark per `run_id`.
 *
 * Runs never share a watermark, so a newer request for Run B can neither block nor be
 * blocked by Run A.  Switching identity clears everything, because no sequence or
 * revision of a previous credential orders anything in the new one.
 */
export class ProjectionOrdering {
  private generation: number
  private readonly watermarks = new Map<string, RunWatermark>()

  constructor(generation = 0) {
    this.generation = generation
  }

  get identity(): number {
    return this.generation
  }

  /** Drop every watermark and adopt a new identity generation. */
  reset(generation: number): void {
    this.generation = generation
    this.watermarks.clear()
  }

  /** What has been applied for one Run, or `null` when nothing has been applied yet. */
  watermark(runId: string): RunWatermark | null {
    return this.watermarks.get(runId) ?? null
  }

  /**
   * Decide one projection and, when it is accepted, record that it was applied.
   *
   * The caller writes the cache only when this returns `accepted: true`; that single
   * call site is what makes "the newest request wins" true for successes and failures
   * of every asynchronous source alike.
   */
  decide(projection: RunProjection): ProjectionDecision {
    if (projection.generation !== this.generation) {
      return { accepted: false, refusal: 'identity' }
    }
    const applied = this.watermarks.get(projection.runId)
    if (applied) {
      if (projection.revision < applied.revision) {
        return { accepted: false, refusal: 'lower-revision' }
      }
      if (projection.revision === applied.revision && projection.sequence <= applied.sequence) {
        return { accepted: false, refusal: 'overtaken' }
      }
    }
    this.watermarks.set(projection.runId, {
      revision: applied ? Math.max(applied.revision, projection.revision) : projection.revision,
      sequence: applied ? Math.max(applied.sequence, projection.sequence) : projection.sequence,
    })
    return { accepted: true, refusal: null }
  }
}

/**
 * Store one accepted Run without disturbing the position of the entries around it.
 *
 * A Run the cache does not hold yet is prepended: that is where a freshly created Run
 * belongs, and it is the one ordering decision this module leaves to the list.
 */
export function upsertRun(cache: readonly ReviewRun[], next: ReviewRun): ReviewRun[] {
  const index = cache.findIndex((run) => run.run_id === next.run_id)
  if (index < 0) return [next, ...cache]
  if (cache[index] === next) return cache as ReviewRun[]
  const copy = [...cache]
  copy[index] = next
  return copy
}

/**
 * Fold one list response into the cache, Run by Run.
 *
 * A list response is one request carrying many projections, so it is arbitrated with
 * the *list request's own* sequence: each Run in the payload is compared against that
 * Run's watermark and only the ones that win are taken.  The response therefore can
 * never replace wholesale a newer cache — a Run whose cached projection is newer keeps
 * it, and a cached Run the payload no longer mentions is not silently dropped.
 */
export function projectRunList(
  ordering: ProjectionOrdering,
  cache: readonly ReviewRun[],
  incoming: readonly ReviewRun[],
  generation: number,
  sequence: number,
): ReviewRun[] {
  const cached = new Map(cache.map((run) => [run.run_id, run]))
  const projected: ReviewRun[] = []
  const mentioned = new Set<string>()
  for (const run of incoming) {
    mentioned.add(run.run_id)
    const decision = ordering.decide({
      generation,
      runId: run.run_id,
      sequence,
      revision: run.revision,
      source: 'list',
    })
    const value = decision.accepted ? run : cached.get(run.run_id)
    if (value) projected.push(value)
  }
  for (const run of cache) {
    if (!mentioned.has(run.run_id)) projected.push(run)
  }
  return projected
}
