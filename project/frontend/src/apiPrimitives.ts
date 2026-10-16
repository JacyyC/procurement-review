/**
 * Field-level primitives and closed value domains of the `review-api.v2` contract.
 *
 * Split out of `apiContract.ts` by responsibility (FIX-01C): this module owns *what a
 * single field may be* — the closed enum sets, the RFC 3339 and decimal rules, the
 * integer/range checks, and the required-present/identity readers.  `apiContract.ts`
 * owns *which fields a response has*, built only from these primitives, so a value
 * domain change and a shape change never land in the same edit.
 *
 * As everywhere on this boundary: no `as` coercion, no default filling, and every
 * rejection names the offending path.
 */

import { invalidResponse } from './apiErrors'
import { API_VERSION_V1, API_VERSION_V2 } from './types'
import type {
  AuditEvent,
  AuditEventClass,
  BusinessFact,
  CaseOption,
  ConfirmationReceipt,
  EvidenceRef,
  Finding,
  FindingConflict,
  ReviewRun,
  ReviewScope,
  ReviewStatus,
  RiskLevel,
  RunEvent,
  SessionRole,
} from './types'

type Summary = NonNullable<ReviewRun['summary']>

// ------------------------------------------------------------------- vocabularies

export const REVIEW_STATUSES: readonly ReviewStatus[] = [
  'processing',
  'pending_confirmation',
  'pending_human_review',
  'confirmed',
  'rejected',
  'cancelled',
  'failed',
  'interrupted',
]
export const SESSION_ROLES: readonly SessionRole[] = ['reader', 'operator', 'approver', 'auditor']
export const REVIEW_SCOPES: readonly ReviewScope[] = [
  'case:read',
  'review:read',
  'review:create',
  'review:cancel',
  'review:resume',
  'review:decide',
  'trace:read',
  'audit:read',
]
export const CASE_TYPES: readonly CaseOption['case_type'][] = ['purchase_order', 'contract_change']
export const REVIEW_SCENARIOS: readonly CaseOption['scenario'][] = [
  'low',
  'high',
  'conflict',
  'unknown',
  'partial_failure',
  'all_failure',
  'version_mismatch',
  'cancellable',
  'runtime_failure',
]
export const WORKER_KINDS = ['policy', 'contract', 'procurement_data'] as const
export const RISK_LEVELS: readonly RiskLevel[] = ['unknown', 'low', 'medium', 'high', 'critical']
export const FINDING_ASSERTIONS: readonly Finding['assertion'][] = [
  'risk_present',
  'risk_absent',
  'unknown',
]
export const FINDING_STATUSES: readonly Finding['status'][] = [
  'supported',
  'unknown',
  'insufficient_evidence',
  'upstream_failure',
  'invalid_input',
]
export const CONFLICT_REASONS: readonly FindingConflict['reason'][] = [
  'opposing_assertions',
  'category_mismatch',
]
export const RECOMMENDATIONS: readonly Summary['recommendation'][] = [
  'recommend_approve',
  'recommend_conditional',
  'recommend_reject',
  'insufficient_information',
  'needs_human_review',
]
export const HUMAN_REVIEW_STATUSES: readonly Summary['human_review_status'][] = [
  'pending_confirmation',
  'required',
  'completed',
]
export const EVIDENCE_SOURCE_KINDS: readonly EvidenceRef['source_kind'][] = [
  'public_document',
  'case_document',
]
export const FACT_STATUSES: readonly BusinessFact['status'][] = [
  'known',
  'unknown',
  'upstream_failure',
]
export const RECEIPT_DECISIONS: readonly ConfirmationReceipt['decision'][] = [
  'confirm_recommendation',
  'reject_recommendation',
]
export const REDACTION_STATUSES: readonly RunEvent['redaction_status'][] = [
  'not_required',
  'redacted',
]
export const AUDIT_EVENT_CLASSES: readonly AuditEventClass[] = [
  'approval',
  'recovery',
  'lifecycle',
  'access_denial',
]
export const AUDIT_ACTION_TYPES: readonly NonNullable<AuditEvent['action_type']>[] = [
  'confirm',
  'reject',
  'cancel',
]
export const WORKER_OUTCOMES = ['completed', 'failed'] as const
export const RUN_API_VERSIONS = [API_VERSION_V1, API_VERSION_V2] as const

export const SCHEMA_VERSION_1 = '1.0'
export const AUDIT_SCHEMA_VERSION = 'approval-audit.v2'
export const TRACE_SCHEMA_VERSION = 'trace.v1'
export const EXECUTION_VERSION = 'review-execution.v1'
export const STATE_VERSION = 'review-state.v1'
export const WORKER_RESULT_VERSION = 'worker-result.v1'
export const EXECUTION_WORKER_COUNT = WORKER_KINDS.length

// ---------------------------------------------------------------------- helpers

export type PlainRecord = Record<string, unknown>

export function isRecord(value: unknown): value is PlainRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function asRecord(value: unknown, path: string): PlainRecord {
  if (!isRecord(value)) throw invalidResponse(path, 'an object')
  return value
}

/** Required-and-present: an absent key is a contract break, not `undefined`. */
export function present(record: PlainRecord, key: string, path: string): unknown {
  const value = record[key]
  if (value === undefined) {
    throw invalidResponse(
      `${path}.${key}`,
      'present',
      `The Review API returned a malformed response: ${path}.${key} is missing.`,
    )
  }
  return value
}

export function reqString(record: PlainRecord, key: string, path: string): string {
  const value = present(record, key, path)
  if (typeof value !== 'string') throw invalidResponse(`${path}.${key}`, 'a string')
  return value
}

export function reqStringOrNull(record: PlainRecord, key: string, path: string): string | null {
  const value = present(record, key, path)
  if (value !== null && typeof value !== 'string') {
    throw invalidResponse(`${path}.${key}`, 'a string or null')
  }
  return value
}

export function reqBoolean(record: PlainRecord, key: string, path: string): boolean {
  const value = present(record, key, path)
  if (typeof value !== 'boolean') throw invalidResponse(`${path}.${key}`, 'a boolean')
  return value
}

/** A declared boolean that a record written before the field existed leaves out. */
export function reqBooleanOrNull(
  record: PlainRecord,
  key: string,
  path: string,
): boolean | null {
  const value = record[key]
  if (value === undefined || value === null) return null
  if (typeof value !== 'boolean') {
    throw invalidResponse(`${path}.${key}`, 'a boolean or null')
  }
  return value
}

/**
 * A declared counter that a record written before the field existed leaves out.
 *
 * A present value is validated exactly like a required one; an absent key means the record
 * predates the field, and a consumer must report that as unrecorded rather than invent a
 * number for it.
 */
export function optionalInteger(
  record: PlainRecord,
  key: string,
  path: string,
  minimum: number,
): number | null {
  const value = record[key]
  if (value === undefined || value === null) return null
  if (typeof value !== 'number' || !Number.isInteger(value) || value < minimum) {
    throw invalidResponse(`${path}.${key}`, `an integer >= ${minimum} or null`)
  }
  return value
}

export function reqLiteral<T extends string | boolean | number>(
  record: PlainRecord,
  key: string,
  expected: T,
  path: string,
): T {
  const value = present(record, key, path)
  if (value !== expected) {
    throw invalidResponse(`${path}.${key}`, `the literal ${JSON.stringify(expected)}`)
  }
  return expected
}

export function reqIntegerOrNull(
  record: PlainRecord,
  key: string,
  path: string,
  minimum: number,
): number | null {
  const value = present(record, key, path)
  if (value === null) return null
  if (typeof value !== 'number' || !Number.isInteger(value) || value < minimum) {
    throw invalidResponse(`${path}.${key}`, `an integer >= ${minimum} or null`)
  }
  return value
}

export function reqInteger(
  record: PlainRecord,
  key: string,
  path: string,
  minimum: number,
): number {
  const value = present(record, key, path)
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw invalidResponse(`${path}.${key}`, 'a number')
  }
  if (!Number.isInteger(value)) throw invalidResponse(`${path}.${key}`, 'an integer')
  if (value < minimum) {
    throw invalidResponse(`${path}.${key}`, `an integer >= ${minimum}`)
  }
  return value
}

export function enumValue<T extends string>(
  value: unknown,
  allowed: readonly T[],
  path: string,
): T {
  if (typeof value !== 'string' || !allowed.includes(value as T)) {
    throw invalidResponse(path, `one of ${allowed.join('/')}`)
  }
  return value as T
}

export function reqEnum<T extends string>(
  record: PlainRecord,
  key: string,
  allowed: readonly T[],
  path: string,
): T {
  return enumValue(present(record, key, path), allowed, `${path}.${key}`)
}

export function reqEnumOrNull<T extends string>(
  record: PlainRecord,
  key: string,
  allowed: readonly T[],
  path: string,
): T | null {
  const value = present(record, key, path)
  if (value === null) return null
  return enumValue(value, allowed, `${path}.${key}`)
}

export function reqArray(record: PlainRecord, key: string, path: string): unknown[] {
  const value = present(record, key, path)
  if (!Array.isArray(value)) throw invalidResponse(`${path}.${key}`, 'an array')
  return value
}

export function reqStringArray(record: PlainRecord, key: string, path: string): string[] {
  return reqArray(record, key, path).map((item, index) => {
    if (typeof item !== 'string') throw invalidResponse(`${path}.${key}[${index}]`, 'a string')
    return item
  })
}

export function reqEnumArray<T extends string>(
  record: PlainRecord,
  key: string,
  allowed: readonly T[],
  path: string,
): T[] {
  return reqArray(record, key, path).map((item, index) =>
    enumValue(item, allowed, `${path}.${key}[${index}]`),
  )
}

/** An open string map, e.g. `source_versions`.  Keys are free, values are typed. */
export function reqStringMap(record: PlainRecord, key: string, path: string): Record<string, string> {
  const source = asRecord(present(record, key, path), `${path}.${key}`)
  const result: Record<string, string> = {}
  for (const [mapKey, mapValue] of Object.entries(source)) {
    if (typeof mapValue !== 'string') {
      throw invalidResponse(`${path}.${key}.${mapKey}`, 'a string')
    }
    result[mapKey] = mapValue
  }
  return result
}

/** An open object map, e.g. `input_summary` / `result`; contents stay opaque. */
export function reqOpenObject(record: PlainRecord, key: string, path: string): Record<string, unknown> {
  return { ...asRecord(present(record, key, path), `${path}.${key}`) }
}

/**
 * RFC 3339 date-times, checked against a real calendar.
 *
 * The shape is pinned first, then the value: `Date.parse` alone accepts `"2026"`, and it
 * *normalises* an impossible day — `2026-02-30T12:00:00Z` becomes March 2nd instead of
 * being rejected (FIX-04B / C-AUD-06).  The day count is therefore computed arithmetic-
 * ally rather than with `Date.UTC`, whose years 0–99 are silently mapped to 1900–1999.
 */
const DATE_TIME_PATTERN =
  /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(Z|[+-]\d{2}:\d{2})$/

const MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] as const

function isLeapYear(year: number): boolean {
  return year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0)
}

function daysInMonth(year: number, month: number): number {
  if (month === 2 && isLeapYear(year)) return 29
  return MONTH_LENGTHS[month - 1]
}

function parseDateTime(value: unknown, path: string): string {
  if (typeof value !== 'string') throw invalidResponse(path, 'an RFC 3339 date-time')
  const match = DATE_TIME_PATTERN.exec(value)
  if (!match) throw invalidResponse(path, 'an RFC 3339 date-time')
  const [, year, month, day, hour, minute, second, zone] = match
  const numbers = [year, month, day, hour, minute, second].map(Number)
  const [y, mo, d, h, mi, s] = numbers
  if (mo < 1 || mo > 12) throw invalidResponse(path, 'a real calendar month')
  if (d < 1 || d > daysInMonth(y, mo)) throw invalidResponse(path, 'a real calendar day')
  if (h > 23 || mi > 59 || s > 59) {
    // Second 60 (a leap second) is not representable by the server's own datetime type,
    // so accepting it here would describe a value the API can never produce.
    throw invalidResponse(path, 'a real clock time')
  }
  if (zone !== 'Z') {
    const offsetHours = Number(zone.slice(1, 3))
    const offsetMinutes = Number(zone.slice(4, 6))
    if (offsetHours > 23 || offsetMinutes > 59) {
      throw invalidResponse(path, 'a real UTC offset')
    }
  }
  return value
}

export function reqDateTime(record: PlainRecord, key: string, path: string): string {
  return parseDateTime(present(record, key, path), `${path}.${key}`)
}

export function reqDateTimeOrNull(record: PlainRecord, key: string, path: string): string | null {
  const value = present(record, key, path)
  return value === null ? null : parseDateTime(value, `${path}.${key}`)
}

/**
 * `cost_amount` is a non-negative decimal, serialised as text by the backend
 * (`Decimal` → `"0"`, `"0.0012"`, `"1.50"`), or `null`.
 *
 * The text is kept verbatim — never coerced through a JS number, which would lose the
 * precision the server chose to send.  A plain number is still accepted for compatibility
 * with older payloads, but only when it is finite **and non-negative**: the earlier rule
 * checked `Number.isFinite` alone, so `-1` passed (FIX-04B / C-AUD-06).  Exponent notation
 * is rejected because Pydantic never writes a `Decimal` that way.
 */
const DECIMAL_TEXT = /^\+?\d+(?:\.\d+)?$/

export function reqCostAmount(record: PlainRecord, key: string, path: string): string | number | null {
  const value = present(record, key, path)
  if (value === null) return null
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || value < 0) {
      throw invalidResponse(`${path}.${key}`, 'a finite decimal >= 0 or null')
    }
    return value
  }
  if (typeof value === 'string' && DECIMAL_TEXT.test(value)) return value
  throw invalidResponse(`${path}.${key}`, 'a non-negative decimal string or null')
}

/**
 * `cost_currency` is an ISO-4217-shaped code (`^[A-Z]{3}$`) or `null`, and the backend
 * model validator requires it to be present exactly when `cost_amount` is.
 */
const CURRENCY_CODE = /^[A-Z]{3}$/

export function reqCostCurrency(
  record: PlainRecord,
  key: string,
  path: string,
): string | null {
  const value = present(record, key, path)
  if (value === null) return null
  if (typeof value !== 'string' || !CURRENCY_CODE.test(value)) {
    throw invalidResponse(`${path}.${key}`, 'a three-letter uppercase currency code or null')
  }
  return value
}

/**
 * A resource response carries its own identity.  A structurally valid payload that
 * describes a different object must never reach the view: it would silently render
 * one run's data under another run's selection.
 */
export function reqIdentity(
  record: PlainRecord,
  key: string,
  expected: string,
  path: string,
): string {
  const value = reqString(record, key, path)
  if (value !== expected) {
    throw invalidResponse(
      `${path}.${key}`,
      'the identity of the requested resource',
      `The Review API returned a response for a different resource: ${path}.${key} is not the object that was requested.`,
    )
  }
  return value
}
