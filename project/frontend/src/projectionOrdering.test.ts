/**
 * FIX-07 S-02 — the ordering rules of the single `ReviewRun` projection arbiter.
 *
 * These are the pure cases; the App-level interleavings live in
 * `fix07ProjectionOrder.test.tsx`.  Keeping the decision itself testable without React
 * is what makes the rule reviewable: revision orders the lifecycle, the request
 * sequence orders two projections that share one revision, and neither is guessed from
 * a clock, a random value or the size of a payload's dynamic fields.
 */

import { describe, expect, it } from 'vitest'

import { ProjectionOrdering, projectRunList, upsertRun } from './projectionOrdering'
import type { RunProjection } from './projectionOrdering'
import { makeRun } from './fix01cHarness'
import type { ReviewRun } from './types'

function projection(overrides: Partial<RunProjection> = {}): RunProjection {
  return {
    generation: 1,
    runId: 'run-a',
    sequence: 1,
    revision: 1,
    source: 'read',
    ...overrides,
  }
}

describe('ProjectionOrdering', () => {
  it('refuses a projection stamped with another identity generation', () => {
    const ordering = new ProjectionOrdering(2)

    expect(ordering.decide(projection({ generation: 1 }))).toEqual({
      accepted: false,
      refusal: 'identity',
    })
    expect(ordering.watermark('run-a')).toBeNull()
  })

  it('refuses a lower revision even when its request sequence is higher', () => {
    const ordering = new ProjectionOrdering(1)
    expect(ordering.decide(projection({ sequence: 4, revision: 3 })).accepted).toBe(true)

    expect(ordering.decide(projection({ sequence: 9, revision: 2 }))).toEqual({
      accepted: false,
      refusal: 'lower-revision',
    })
    expect(ordering.watermark('run-a')).toEqual({ revision: 3, sequence: 4 })
  })

  it('accepts a higher revision even when its request sequence is lower', () => {
    const ordering = new ProjectionOrdering(1)
    ordering.decide(projection({ sequence: 7, revision: 3 }))

    expect(ordering.decide(projection({ sequence: 5, revision: 4 })).accepted).toBe(true)
    // The watermark never moves backwards: the requests that were already superseded
    // stay superseded even though this one arrived with a higher revision.
    expect(ordering.watermark('run-a')).toEqual({ revision: 4, sequence: 7 })
  })

  it('decides two same-revision projections by request sequence alone', () => {
    const ordering = new ProjectionOrdering(1)
    expect(ordering.decide(projection({ sequence: 6, revision: 2 })).accepted).toBe(true)

    expect(ordering.decide(projection({ sequence: 5, revision: 2 }))).toEqual({
      accepted: false,
      refusal: 'overtaken',
    })
    expect(ordering.decide(projection({ sequence: 6, revision: 2 }))).toEqual({
      accepted: false,
      refusal: 'overtaken',
    })
    expect(ordering.decide(projection({ sequence: 7, revision: 2 })).accepted).toBe(true)
    expect(ordering.watermark('run-a')).toEqual({ revision: 2, sequence: 7 })
  })

  it('keeps one watermark per Run', () => {
    const ordering = new ProjectionOrdering(1)
    ordering.decide(projection({ runId: 'run-a', sequence: 5, revision: 2 }))

    // A newer request for another Run is neither blocked by run-a nor blocks it.
    expect(
      ordering.decide(projection({ runId: 'run-b', sequence: 1, revision: 1 })).accepted,
    ).toBe(true)
    expect(
      ordering.decide(projection({ runId: 'run-a', sequence: 4, revision: 2 })).accepted,
    ).toBe(false)
    expect(ordering.watermark('run-b')).toEqual({ revision: 1, sequence: 1 })
  })

  it('drops every watermark when the identity changes', () => {
    const ordering = new ProjectionOrdering(1)
    ordering.decide(projection({ sequence: 9, revision: 4 }))

    ordering.reset(2)

    expect(ordering.identity).toBe(2)
    expect(ordering.watermark('run-a')).toBeNull()
    // The previous identity's request sequence means nothing in the new one.
    expect(ordering.decide(projection({ generation: 2, sequence: 1, revision: 1 })).accepted).toBe(
      true,
    )
  })
})

describe('projectRunList', () => {
  const cached = makeRun('run-a', { revision: 4, trace_event_count: 9 })
  const incomingA = makeRun('run-a', { revision: 4, trace_event_count: 2 })
  const incomingNew = makeRun('run-b', { revision: 1, trace_event_count: 1 })

  it('keeps a newer cached Run while taking the rest of the payload, in payload order', () => {
    const ordering = new ProjectionOrdering(1)
    ordering.decide({
      generation: 1,
      runId: 'run-a',
      sequence: 8,
      revision: 4,
      source: 'write',
    })

    const projected = projectRunList(ordering, [cached], [incomingA, incomingNew], 1, 3)

    expect(projected.map((run) => run.run_id)).toEqual(['run-a', 'run-b'])
    expect(projected[0]).toBe(cached)
    expect(projected[1]).toBe(incomingNew)
  })

  it('accepts a list that was issued after the cached projection', () => {
    const ordering = new ProjectionOrdering(1)
    ordering.decide({
      generation: 1,
      runId: 'run-a',
      sequence: 3,
      revision: 4,
      source: 'read',
    })

    const projected = projectRunList(ordering, [cached], [incomingA, incomingNew], 1, 4)

    expect(projected[0]).toBe(incomingA)
  })

  it('does not drop a cached Run the payload no longer mentions', () => {
    const ordering = new ProjectionOrdering(1)
    const projected = projectRunList(ordering, [cached], [incomingNew], 1, 1)

    expect(projected.map((run) => run.run_id)).toEqual(['run-b', 'run-a'])
  })

  it('refuses a whole list from another identity', () => {
    const ordering = new ProjectionOrdering(2)
    const projected = projectRunList(ordering, [cached], [incomingNew], 1, 5)

    expect(projected).toEqual([cached])
  })
})

describe('upsertRun', () => {
  const first = makeRun('run-a', { revision: 1 })
  const second = makeRun('run-b', { revision: 1 })

  it('prepends a Run the cache does not hold yet', () => {
    expect(upsertRun([first], second).map((run: ReviewRun) => run.run_id)).toEqual([
      'run-b',
      'run-a',
    ])
  })

  it('replaces a Run in place instead of reshuffling the list', () => {
    const updated = makeRun('run-a', { revision: 2 })
    const next = upsertRun([first, second], updated)

    expect(next.map((run) => run.run_id)).toEqual(['run-a', 'run-b'])
    expect(next[0]).toBe(updated)
  })

  it('returns the same array when the payload is the very same object', () => {
    const cache = [first, second]
    expect(upsertRun(cache, first)).toBe(cache)
  })
})
