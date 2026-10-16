/**
 * FIX-04B / C-AUD-06 — calendar dates and Decimal amounts.
 *
 * The audit found `apiPrimitives.ts` accepted `2026-02-30T12:00:00Z` because
 * `Date.parse` normalises an impossible day into March, and accepted `cost_amount: -1`
 * because the only numeric rule was `Number.isFinite`.  The backend contract these cases
 * are checked against (`contracts.py`, verified by running Pydantic directly):
 *
 *  - `occurred_at` is a timezone-aware `datetime`, serialised as `...T..:..:..Z` or with a
 *    `+HH:MM` offset;
 *  - `duration_ms` is `int >= 0`; `token_count` is `int >= 0 | None`;
 *  - `cost_amount` is `Decimal >= 0 | None`, serialised as a **string** that keeps the
 *    precision it was built with (`Decimal("0.0012")` → `"0.0012"`, `Decimal("1.50")` →
 *    `"1.50"`; Pydantic writes canonical decimal text, never exponent notation);
 *  - `cost_currency` is `str` matching `^[A-Z]{3}$` or `None`, and the model validator
 *    requires `cost_amount` and `cost_currency` to be present together or absent together.
 */

import { describe, expect, it } from 'vitest'

import { validateRun, validateTrace } from './apiContract'
import { makeRun, makeTrace } from './fix01cHarness'
import type { ReviewRun, TraceResponse } from './types'

function runWith(patch: Partial<ReviewRun>): ReviewRun {
  return makeRun('run-a', patch)
}

function runWithDate(value: string): ReviewRun {
  return runWith({ created_at: value })
}

function traceWithEvent(patch: Record<string, unknown>): TraceResponse {
  const payload = makeTrace('run-a', ['run_completed'])
  Object.assign(payload.trace.events[0], patch)
  return payload
}

describe('C-AUD-06 · the date must exist on the calendar', () => {
  it.each([
    ['a day the month does not have', '2026-02-30T12:00:00Z', false],
    ['April 31st', '2026-04-31T00:00:00Z', false],
    ['February 29th in a common year', '2025-02-29T00:00:00Z', false],
    ['February 29th in a leap year', '2024-02-29T00:00:00Z', true],
    ['February 29th in a century that is not a leap year', '1900-02-29T00:00:00Z', false],
    ['February 29th in a leap century', '2000-02-29T00:00:00Z', true],
    ['hour 24', '2026-01-01T24:00:00Z', false],
    ['minute 60', '2026-01-01T12:60:00Z', false],
    ['second 60', '2026-01-01T12:00:60Z', false],
    ['month 13', '2026-13-01T12:00:00Z', false],
    ['month 00', '2026-00-10T12:00:00Z', false],
    ['day 00', '2026-01-00T12:00:00Z', false],
    ['no timezone', '2026-01-01T12:00:00', false],
    ['a date without a time', '2026-01-01', false],
    ['a UTC time', '2026-01-01T12:00:00Z', true],
    ['a positive offset', '2026-01-01T12:00:00+08:00', true],
    ['a negative offset', '2026-01-01T12:00:00-05:30', true],
    ['an offset that crosses the day boundary', '2026-01-01T00:30:00+14:00', true],
    ['an out-of-range offset', '2026-01-01T12:00:00+25:00', false],
    ['an out-of-range offset minute', '2026-01-01T12:00:00+08:60', false],
    ['a fractional second', '2026-01-01T12:00:00.250Z', true],
    ['a two-digit year', '26-01-01T12:00:00Z', false],
    ['a 99 year without the Date.UTC two-digit quirk', '0099-02-28T12:00:00Z', true],
    ['a 99 year that is not a leap year', '0099-02-29T12:00:00Z', false],
  ])('%s (%s) is %s', (_label, value, accepted) => {
    if (accepted) {
      expect(() => validateRun(runWithDate(value))).not.toThrow()
    } else {
      expect(() => validateRun(runWithDate(value))).toThrow()
    }
  })

  it('applies the same calendar rule to nested timestamps', () => {
    expect(() => validateRun(runWith({ updated_at: '2026-02-30T12:00:00Z' }))).toThrow()
    expect(() =>
      validateTrace(traceWithEvent({ occurred_at: '2026-02-30T12:00:00Z' })),
    ).toThrow()
  })
})

describe('C-AUD-06 · amounts keep their contract and their precision', () => {
  it.each([
    ['a zero amount', '0', true],
    ['a fractional amount', '0.0012', true],
    ['a trailing-zero amount', '1.50', true],
    ['a large integer amount', '1000000', true],
    ['a negative amount', '-1', false],
    ['a negative fractional amount', '-0.5', false],
    ['an exponent amount', '1e5', false],
    ['a non-numeric amount', 'abc', false],
    ['an empty amount', '', false],
  ])('%s (%s) is %s', (_label, amount, accepted) => {
    const payload = traceWithEvent({ cost_amount: amount, cost_currency: 'CNY' })
    if (accepted) {
      expect(() => validateTrace(payload)).not.toThrow()
      // Precision is preserved: the value is never coerced through a JS number.
      expect(payload.trace.events[0].cost_amount).toBe(amount)
    } else {
      expect(() => validateTrace(payload)).toThrow()
    }
  })

  it.each([
    ['a negative number', -1, false],
    ['infinity', Number.POSITIVE_INFINITY, false],
    ['NaN', Number.NaN, false],
    ['a legal zero number', 0, true],
    ['a legal fractional number', 0.5, true],
  ])('legacy numeric compatibility: %s is %s', (_label, amount, accepted) => {
    const payload = traceWithEvent({ cost_amount: amount, cost_currency: 'CNY' })
    if (accepted) {
      expect(() => validateTrace(payload)).not.toThrow()
    } else {
      expect(() => validateTrace(payload)).toThrow()
    }
  })

  it.each([
    ['an amount without a currency', '0', null],
    ['a currency without an amount', null, 'CNY'],
  ])('rejects %s', (_label, amount, currency) => {
    const payload = traceWithEvent({ cost_amount: amount, cost_currency: currency })
    expect(() => validateTrace(payload)).toThrow()
  })

  it('accepts both being null', () => {
    const payload = traceWithEvent({ cost_amount: null, cost_currency: null })
    expect(() => validateTrace(payload)).not.toThrow()
  })

  it('rejects a currency the backend cannot serialise', () => {
    expect(() =>
      validateTrace(traceWithEvent({ cost_amount: '0', cost_currency: 'cny' })),
    ).toThrow()
    expect(() =>
      validateTrace(traceWithEvent({ cost_amount: '0', cost_currency: 'CN' })),
    ).toThrow()
  })

  it.each([
    ['a negative duration', { duration_ms: -1 }, false],
    ['a fractional duration', { duration_ms: 1.5 }, false],
    ['a zero duration', { duration_ms: 0 }, true],
    ['a negative token count', { token_count: -1 }, false],
    ['a fractional token count', { token_count: 2.5 }, false],
    ['a null token count', { token_count: null }, true],
    ['a zero token count', { token_count: 0 }, true],
  ])('%s is %s', (_label, patch, accepted) => {
    const payload = traceWithEvent(patch)
    if (accepted) {
      expect(() => validateTrace(payload)).not.toThrow()
    } else {
      expect(() => validateTrace(payload)).toThrow()
    }
  })
})
