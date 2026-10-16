/**
 * Ownership rules for asynchronous work in the workbench.
 *
 * The notions of "the current request" that stay apart on purpose:
 *
 *  1. **identity generation** — a credential change or an unmount invalidates
 *     *everything* (`App` keeps that counter; see `invalidateRequests`);
 *  2. **selection epoch** — which Run the human is looking at right now.  A late
 *     response may update its own Run's cache, but it may never change a selection
 *     the human made after the request started;
 *  3. **per-resource ticket** — Trace and Audit are fetched independently, so only
 *     the newest outstanding read for one (Run, resource) pair may write its result;
 *  4. **artifact entry** — one stored response per Run, which remembers the ticket
 *     that produced it *including when that ticket failed* (FIX-04A / C-AUD-01).
 *
 * Ordering of the `ReviewRun` projections themselves is **not** here: every source of
 * one — reads, write responses and reconciliations alike — is decided by
 * `projectionOrdering.ts`, which is the single place that compares revisions and
 * request sequences (FIX-07 / S-02).
 *
 * The helpers here are pure and synchronous on purpose: the correctness argument is
 * "the guard is evaluated when the response is applied", which holds even when a
 * mock (or a proxy) ignores `AbortController` and the request still resolves.
 *
 * Why a failed read keeps its ticket: clearing the entry on failure used to throw the
 * ordering information away with it, so a *superseded* request that succeeded later
 * could refill a slot a newer request had already replaced.  Recording the failure
 * (rather than deleting the key) is what makes "newest request wins" true for
 * successes and failures alike.
 */

/** One stored outcome, stamped with the ticket of the request that produced it. */
export type ArtifactEntry<T> =
  | { ticket: number; state: 'ok'; value: T }
  | { ticket: number; state: 'error' }

/**
 * Per-Run artifact cache.  Entries are keyed by `run_id`, not held in one slot: a
 * late response for Run A must not evict Run B's already-loaded Trace, and it must
 * never be rendered while B is selected.
 */
export type ArtifactCache<T> = Readonly<Record<string, ArtifactEntry<T>>>

/** The outcome shape the transport produces; declared structurally to stay pure. */
export type Settled<T> =
  | { status: 'ok'; value: T }
  | { status: 'error' }
  | { status: 'stale' }

/** What to render: only the selected Run's own *successfully* read entry. */
export function artifactFor<T>(
  cache: ArtifactCache<T>,
  runId: string | null | undefined,
): T | null {
  if (!runId) return null
  const entry = cache[runId]
  return entry && entry.state === 'ok' ? entry.value : null
}

/**
 * Fold one independent read into the cache.
 *
 * - only the newest ticket for that Run may change anything;
 * - `ok` stores the value, `error` records the failure with its ticket so the slot
 *   stays claimed and an older success cannot refill it later;
 * - `stale` (identity changed) leaves the cache alone — the identity change already
 *   cleared the workspace.
 */
export function applyArtifact<T>(
  cache: ArtifactCache<T>,
  runId: string,
  ticket: number,
  outcome: Settled<T>,
): ArtifactCache<T> {
  if (outcome.status === 'stale') return cache
  const existing = cache[runId]
  if (existing && existing.ticket > ticket) return cache
  const entry: ArtifactEntry<T> =
    outcome.status === 'ok'
      ? { ticket, state: 'ok', value: outcome.value }
      : { ticket, state: 'error' }
  return { ...cache, [runId]: entry }
}

// ------------------------------------------------------------------- tickets
//
// One monotonically increasing counter feeds every scoped registry, so a ticket is
// unique across the whole identity.  A request registers its ticket *when it starts*;
// a completion is applied only if its ticket is still the newest for that key.

export type TicketCounter = { current: number }

/** Register a new request for `key` and return its ticket. */
export function openTicket(
  counter: TicketCounter,
  latest: Map<string, number>,
  key: string,
): number {
  const ticket = (counter.current += 1)
  latest.set(key, ticket)
  return ticket
}

/** Whether this ticket is still the newest one registered for `key`. */
export function isLatestTicket(
  latest: Map<string, number>,
  key: string,
  ticket: number,
): boolean {
  return latest.get(key) === ticket
}

/**
 * Whether an asynchronous result may still move the selection.
 *
 * `App` keeps one monotonically increasing epoch in a ref.  A user click that
 * changes the selection bumps it; an asynchronous completion captures the epoch it
 * started under and only selects its Run when the epoch is unchanged, so a late
 * response updates its own cache entry without re-selecting an object the human
 * already navigated away from.
 */
export function selectionStillApplies(startedAt: number, current: number): boolean {
  return startedAt === current
}
