/**
 * FIX-04A / C-AUD-02 — the write registry's own contracts.
 *
 * These cases pin the rules `App` relies on: what a failure tells us about the
 * server's state, which request is offered for replay, and that the Scope which
 * authorises a replay is the action's own rather than one shared scope.
 */

import { describe, expect, it } from 'vitest'

import {
  associateCreatedRun,
  classifyWriteFailure,
  MAX_WRITE_RECORDS,
  offeredCreate,
  offeredWriteForRun,
  registerWrite,
  settleWrite,
  writeScope,
  writeTargetRun,
} from './writeOperations'
import type { WriteOperation, WriteRecord } from './writeOperations'

const decide = (operationId: string, runId = 'run-a'): WriteOperation => ({
  kind: 'decide',
  operationId,
  runId,
  action: 'confirm',
  idempotencyKey: `key-${operationId}`,
  expectedVersion: 2,
  summaryChecksum: `sha256:${'a'.repeat(64)}`,
})

const cancel = (operationId: string, runId = 'run-a'): WriteOperation => ({
  kind: 'cancel',
  operationId,
  runId,
  idempotencyKey: `key-${operationId}`,
  expectedVersion: 2,
})

const resume = (operationId: string, runId = 'run-a'): WriteOperation => ({
  kind: 'resume',
  operationId,
  runId,
  idempotencyKey: `key-${operationId}`,
  expectedVersion: 2,
})

const create = (operationId: string, runId: string | null = null): WriteOperation => ({
  kind: 'create',
  operationId,
  caseId: 'CASE-LOW-001',
  caseVersion: 'case-version-001',
  idempotencyKey: `key-${operationId}`,
  runId,
})

describe('classifyWriteFailure keeps "unknown" honest', () => {
  it('treats a lost response, a timeout and a broken 2xx body as unknown', () => {
    expect(classifyWriteFailure(503, 'security_dependency_unavailable')).toBe('unknown')
    expect(classifyWriteFailure(500, 'internal_error')).toBe('unknown')
    // `invalid_response` keeps the real 200/202 status, so the code decides.
    expect(classifyWriteFailure(202, 'invalid_response')).toBe('unknown')
  })

  it('treats a server business answer as settled', () => {
    expect(classifyWriteFailure(409, 'version_conflict')).toBe('responded')
    expect(classifyWriteFailure(403, 'insufficient_scope')).toBe('responded')
    expect(classifyWriteFailure(404, 'resource_not_found')).toBe('responded')
    expect(classifyWriteFailure(401, 'unauthenticated')).toBe('responded')
  })
})

describe('the registry is keyed by the object the request belongs to', () => {
  it('binds a create to the Case until the server answers with a Run', () => {
    let records: readonly WriteRecord[] = []
    records = registerWrite(records, create('op-1'), 3)
    expect(writeTargetRun(records[0].operation)).toBeNull()
    expect(offeredCreate(records)?.operation.operationId).toBe('op-1')
    expect(offeredWriteForRun(records, 'run-a')).toBeNull()

    records = associateCreatedRun(records, 'op-1', 'run-new')
    expect(writeTargetRun(records[0].operation)).toBe('run-new')
    expect(offeredWriteForRun(records, 'run-new')?.operation.operationId).toBe('op-1')
  })

  it('registers before the outcome is known and records the identity generation', () => {
    const records = registerWrite([], decide('op-1'), 7)
    expect(records).toHaveLength(1)
    expect(records[0].outcome).toBe('in-flight')
    expect(records[0].generation).toBe(7)
  })

  it('replaces the record of the same operation instead of duplicating it', () => {
    let records = registerWrite([], decide('op-1'), 1)
    records = settleWrite(records, 'op-1', 'unknown')
    records = registerWrite(records, decide('op-1'), 1)
    expect(records).toHaveLength(1)
    expect(records[0].outcome).toBe('in-flight')
  })

  it('keeps a different Run out of this Run’s replay entry', () => {
    let records = registerWrite([], decide('op-a', 'run-a'), 1)
    records = registerWrite(records, decide('op-b', 'run-b'), 1)
    expect(offeredWriteForRun(records, 'run-a')?.operation.operationId).toBe('op-a')
    expect(offeredWriteForRun(records, 'run-b')?.operation.operationId).toBe('op-b')
  })
})

describe('an unresolved request is offered ahead of a newer settled one', () => {
  it('keeps offering the unknown request after a later action was recorded', () => {
    let records = registerWrite([], decide('op-1'), 1)
    records = settleWrite(records, 'op-1', 'unknown')
    records = registerWrite(records, cancel('op-2'), 1)
    records = settleWrite(records, 'op-2', 'responded')
    expect(offeredWriteForRun(records, 'run-a')?.operation.operationId).toBe('op-1')
    expect(offeredWriteForRun(records, 'run-a')?.outcome).toBe('unknown')
  })

  it('falls back to the most recent settled request', () => {
    let records = registerWrite([], decide('op-1'), 1)
    records = settleWrite(records, 'op-1', 'responded')
    records = registerWrite(records, cancel('op-2'), 1)
    records = settleWrite(records, 'op-2', 'responded')
    expect(offeredWriteForRun(records, 'run-a')?.operation.operationId).toBe('op-2')
  })

  it('offers an in-flight request so its control can stay disabled', () => {
    const records = registerWrite([], resume('op-1'), 1)
    const offered = offeredWriteForRun(records, 'run-a')
    expect(offered?.outcome).toBe('in-flight')
  })

  it('reports nothing when the Run has no request in this session', () => {
    const records = registerWrite([], decide('op-1', 'run-b'), 1)
    expect(offeredWriteForRun(records, 'run-a')).toBeNull()
  })
})

describe('retention stays bounded without discarding an unresolved request', () => {
  it('drops only settled records once the bound is exceeded', () => {
    let records: readonly WriteRecord[] = []
    for (let index = 0; index < MAX_WRITE_RECORDS + 2; index += 1) {
      records = registerWrite(records, decide(`op-${index}`, 'run-a'), 1)
      records = settleWrite(records, `op-${index}`, 'responded')
    }
    // The oldest settled records were trimmed…
    expect(records.length).toBeLessThanOrEqual(MAX_WRITE_RECORDS)
    expect(offeredWriteForRun(records, 'run-a')?.operation.operationId).toBe(
      `op-${MAX_WRITE_RECORDS + 1}`,
    )

    // …but an unresolved one is never the record that gets dropped.
    let mixed: readonly WriteRecord[] = registerWrite([], decide('op-unknown'), 1)
    mixed = settleWrite(mixed, 'op-unknown', 'unknown')
    for (let index = 0; index < MAX_WRITE_RECORDS + 2; index += 1) {
      mixed = registerWrite(mixed, decide(`op-${index}`, 'run-a'), 1)
      mixed = settleWrite(mixed, `op-${index}`, 'responded')
    }
    expect(mixed.some((record) => record.operation.operationId === 'op-unknown')).toBe(true)
    expect(offeredWriteForRun(mixed, 'run-a')?.operation.operationId).toBe('op-unknown')
  })
})

describe('replay is authorised by the action’s own Scope', () => {
  it('maps each operation to the scope that permits sending it', () => {
    expect(writeScope(create('op'))).toBe('review:create')
    expect(writeScope(decide('op'))).toBe('review:decide')
    expect(writeScope(cancel('op'))).toBe('review:cancel')
    expect(writeScope(resume('op'))).toBe('review:resume')
  })
})
