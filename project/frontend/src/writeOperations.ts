/**
 * Session-scoped registry of the writes the workbench has sent (FIX-04A / C-AUD-02).
 *
 * The audit found the previous design remembered a decision only *after* the server
 * answered, in a single slot that selecting another Run cleared.  A write that had
 * committed but whose response never arrived was therefore unreplayable, and create
 * and resume kept no context at all.
 *
 * The rules implemented here:
 *
 *  - the request is registered **before** it is sent, holding the immutable identity
 *    generation, the original idempotency key and the exact body it will send;
 *  - it is keyed by the object it belongs to — the Case while a create has no run id
 *    yet, the Run afterwards — so switching Runs never discards it;
 *  - its outcome is at least `in-flight`, `unknown` or `responded`.  A lost response,
 *    a timeout and an unreadable 2xx body are all `unknown`: the client may not claim
 *    the write definitely did not happen;
 *  - an unresolved record is *offered* for replay ahead of a newer settled one, so a
 *    later click can never silently bury the request whose result is still unknown;
 *  - the record is memory-only.  Nothing here is written to storage, the URL or a log,
 *    so a reload or a new credential leaves no replayable context behind.
 *
 * This module holds no React state: `App` owns the list, these helpers decide what it
 * means.
 */

import type { ReviewScope } from './types'

/** A human action with idempotent same-key semantics. */
export type DecideAction = 'confirm' | 'reject'

/** `POST /reviews` — pinned to a Case version; the Run identity arrives with it. */
export interface CreateOperation {
  kind: 'create'
  operationId: string
  caseId: string
  caseVersion: string
  idempotencyKey: string
  /** Set once the server answered with the Run it created. */
  runId: string | null
}

export interface DecideOperation {
  kind: 'decide'
  operationId: string
  runId: string
  action: DecideAction
  idempotencyKey: string
  expectedVersion: number
  summaryChecksum: string | null
}

export interface CancelOperation {
  kind: 'cancel'
  operationId: string
  runId: string
  idempotencyKey: string
  expectedVersion: number
}

export interface ResumeOperation {
  kind: 'resume'
  operationId: string
  runId: string
  idempotencyKey: string
  expectedVersion: number
}

/** Narrow union of every write the workbench can send. */
export type WriteOperation =
  | CreateOperation
  | DecideOperation
  | CancelOperation
  | ResumeOperation

/** How much is known about a sent write.  `unknown` is never "definitely not sent". */
export type WriteOutcome = 'in-flight' | 'unknown' | 'responded'

export interface WriteRecord {
  operation: WriteOperation
  /** Identity generation the request was frozen under. */
  generation: number
  outcome: WriteOutcome
}

/** Bound on retained records: enough for a session's manual actions, never unbounded. */
export const MAX_WRITE_RECORDS = 32

/**
 * The object a write belongs to.  A create has no run id until the server answers, so
 * its identity is the Case plus the operation's own id — never another Run's slot.
 */
export function writeTargetRun(operation: WriteOperation): string | null {
  return operation.runId
}

/** The Scope that authorises *sending* (and therefore replaying) this operation. */
export function writeScope(operation: WriteOperation): ReviewScope {
  switch (operation.kind) {
    case 'create':
      return 'review:create'
    case 'decide':
      return 'review:decide'
    case 'cancel':
      return 'review:cancel'
    case 'resume':
      return 'review:resume'
  }
}

/**
 * Classify what a failed send tells us about the server's state.
 *
 * A parsed business error below `500` is the server's own answer, so the write is
 * settled.  Everything else — a transport failure, a timeout, an unreadable 2xx body
 * or a `5xx` — leaves the outcome unknown, and the operator gets an explicit replay
 * instead of a guess.
 */
export function classifyWriteFailure(status: number, code: string): WriteOutcome {
  if (code === 'invalid_response') return 'unknown'
  return status < 500 ? 'responded' : 'unknown'
}

/** Append (or refresh) the record for a request that is about to be sent. */
export function registerWrite(
  records: readonly WriteRecord[],
  operation: WriteOperation,
  generation: number,
): WriteRecord[] {
  const withoutSame = records.filter(
    (record) => record.operation.operationId !== operation.operationId,
  )
  const next = [...withoutSame, { operation, generation, outcome: 'in-flight' as const }]
  return trimRecords(next)
}

/** Record what is now known about one sent request. */
export function settleWrite(
  records: readonly WriteRecord[],
  operationId: string,
  outcome: WriteOutcome,
): WriteRecord[] {
  let changed = false
  const next = records.map((record) => {
    if (record.operation.operationId !== operationId || record.outcome === outcome) {
      return record
    }
    changed = true
    return { ...record, outcome }
  })
  return changed ? next : (records as WriteRecord[])
}

/** Bind the Run the server created back to the create request that asked for it. */
export function associateCreatedRun(
  records: readonly WriteRecord[],
  operationId: string,
  runId: string,
): WriteRecord[] {
  let changed = false
  const next = records.map((record) => {
    if (record.operation.operationId !== operationId) return record
    if (record.operation.kind !== 'create' || record.operation.runId === runId) return record
    changed = true
    return { ...record, operation: { ...record.operation, runId } }
  })
  return changed ? next : (records as WriteRecord[])
}

/**
 * The record offered for explicit replay on one Run: the oldest request whose result
 * is still unresolved first, otherwise the most recently sent one.
 */
export function offeredWriteForRun(
  records: readonly WriteRecord[],
  runId: string,
): WriteRecord | null {
  const forRun = records.filter((record) => writeTargetRun(record.operation) === runId)
  return forRun.find((record) => record.outcome !== 'responded') ?? forRun.at(-1) ?? null
}

/** The create request offered for explicit replay, with the same rules. */
export function offeredCreate(records: readonly WriteRecord[]): WriteRecord | null {
  const creates = records.filter((record) => record.operation.kind === 'create')
  return creates.find((record) => record.outcome !== 'responded') ?? creates.at(-1) ?? null
}

/**
 * Drop the oldest already-settled record once the list grows past the bound.  An
 * unresolved request is never discarded: it is the only remaining way to replay it.
 */
function trimRecords(records: readonly WriteRecord[]): WriteRecord[] {
  if (records.length <= MAX_WRITE_RECORDS) return [...records]
  const overflow = records.length - MAX_WRITE_RECORDS
  const drop = new Set<string>()
  for (const record of records) {
    if (drop.size >= overflow) break
    if (record.outcome === 'responded') drop.add(record.operation.operationId)
  }
  return records.filter((record) => !drop.has(record.operation.operationId))
}
