/**
 * FIX-04A / C-AUD-01 — resource ownership at the pure-helper boundary.
 *
 * The audit found that `applyArtifact` gated the *success* branch on the request
 * sequence but let **any** failure clear the entry unconditionally.  Clearing the
 * entry also threw the sequence away, so an older success that settled afterwards
 * could refill a slot that a newer request had already superseded.  These cases pin
 * the two halves of the fix:
 *
 *  - a late failure may not destroy a newer, already-applied response;
 *  - a newer failure may not be undone by an older success released afterwards.
 *
 * They drive only the exported helpers, so they describe the contract rather than an
 * implementation detail of the cache.
 */

import { describe, expect, it } from 'vitest'

import { applyArtifact, artifactFor } from './resourceBinding'
import type { ArtifactCache, Settled } from './resourceBinding'

const ok = <T,>(value: T): Settled<T> => ({ status: 'ok', value })
const failed = <T,>(): Settled<T> => ({ status: 'error' })
const stale = <T,>(): Settled<T> => ({ status: 'stale' })

/** Apply a whole scripted sequence of settlements for one Run. */
function fold(
  steps: readonly { seq: number; outcome: Settled<string> }[],
): ArtifactCache<string> {
  return steps.reduce<ArtifactCache<string>>(
    (cache, step) => applyArtifact(cache, 'run-a', step.seq, step.outcome),
    {},
  )
}

describe('applyArtifact keeps the newest request, success or failure', () => {
  it('applies a newer success over an older one', () => {
    const cache = fold([
      { seq: 1, outcome: ok('first') },
      { seq: 2, outcome: ok('second') },
    ])
    expect(artifactFor(cache, 'run-a')).toBe('second')
  })

  it('ignores an older success that settles after a newer one', () => {
    const cache = fold([
      { seq: 2, outcome: ok('second') },
      { seq: 1, outcome: ok('first') },
    ])
    expect(artifactFor(cache, 'run-a')).toBe('second')
  })

  it('lets a newer failure clear the readable value', () => {
    const cache = fold([
      { seq: 1, outcome: ok('first') },
      { seq: 2, outcome: failed() },
    ])
    expect(artifactFor(cache, 'run-a')).toBeNull()
  })

  it('does not let an older failure clear a newer response', () => {
    const cache = fold([
      { seq: 2, outcome: ok('second') },
      { seq: 1, outcome: failed() },
    ])
    expect(artifactFor(cache, 'run-a')).toBe('second')
  })

  it('does not let an older success refill a slot a newer failure emptied', () => {
    const cache = fold([
      { seq: 2, outcome: failed() },
      { seq: 1, outcome: ok('first') },
    ])
    expect(artifactFor(cache, 'run-a')).toBeNull()
  })

  it('keeps the newest failure when two failures settle out of order', () => {
    const cache = fold([
      { seq: 2, outcome: failed() },
      { seq: 1, outcome: failed() },
    ])
    expect(artifactFor(cache, 'run-a')).toBeNull()
  })

  it('recovers once a request newer than the failure succeeds', () => {
    const cache = fold([
      { seq: 1, outcome: ok('first') },
      { seq: 2, outcome: failed() },
      { seq: 3, outcome: ok('recovered') },
    ])
    expect(artifactFor(cache, 'run-a')).toBe('recovered')
  })

  it('leaves the cache untouched for a stale (superseded identity) outcome', () => {
    const cache = fold([
      { seq: 2, outcome: ok('second') },
      { seq: 3, outcome: stale() },
    ])
    expect(artifactFor(cache, 'run-a')).toBe('second')
  })

  it('keeps Runs apart: one Run failing never clears another Run', () => {
    let cache: ArtifactCache<string> = {}
    cache = applyArtifact(cache, 'run-a', 1, ok('a'))
    cache = applyArtifact(cache, 'run-b', 2, ok('b'))
    cache = applyArtifact(cache, 'run-b', 3, failed())
    expect(artifactFor(cache, 'run-a')).toBe('a')
    expect(artifactFor(cache, 'run-b')).toBeNull()
  })

  it('returns null for an unselected or unknown Run', () => {
    const cache = fold([{ seq: 1, outcome: ok('a') }])
    expect(artifactFor(cache, null)).toBeNull()
    expect(artifactFor(cache, 'run-missing')).toBeNull()
  })
})
