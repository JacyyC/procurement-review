/**
 * Test support: read *every* argument of *every* console call, deeply.
 *
 * C-AUD-07 is about what the workbench writes to the console when a render fails, so a
 * check that only looks at strings, or that uses `JSON.stringify(error)`, would miss the
 * very leak it is supposed to find: `Error.message`, `stack` and `cause` are
 * non-enumerable, and `JSON.stringify(new Error('x'))` is `"{}"`.
 *
 * This module installs one recorder over the whole console surface and describes every
 * argument by walking its own property names (enumerable or not) recursively, including
 * `Map`/`Set` contents and `String(value)` (which is what an `Error` prints as).  It is
 * test support only: no product module imports it, so it is never part of a build.
 */

export interface ConsoleScan {
  /** One line per console call; already flattened to a single searchable string. */
  entries: string[]
  restore: () => void
}

const LEVELS = ['log', 'info', 'warn', 'error', 'debug', 'trace'] as const

/** Describe one console argument, including everything a naive check would miss. */
export function describeConsoleValue(value: unknown, depth = 0, seen = new Set<unknown>()): string {
  if (value === null || value === undefined) return String(value)
  const kind = typeof value
  if (kind === 'string') return value as string
  if (kind === 'number' || kind === 'boolean' || kind === 'bigint') return String(value)
  if (kind === 'symbol') return String(value)
  if (kind === 'function') return `[function ${(value as { name?: string }).name || 'anonymous'}]`
  if (depth > 6) return '[depth-limit]'
  const object = value as object
  if (seen.has(object)) return '[cycle]'
  seen.add(object)

  const parts: string[] = []
  try {
    for (const key of Object.getOwnPropertyNames(object)) {
      let nested: unknown
      try {
        nested = (object as Record<string, unknown>)[key]
      } catch {
        nested = '[unreadable]'
      }
      parts.push(`${key}=${describeConsoleValue(nested, depth + 1, seen)}`)
    }
  } catch {
    parts.push('[own-properties-unreadable]')
  }
  if (value instanceof Map) {
    for (const [mapKey, mapValue] of value) {
      parts.push(
        `entry(${describeConsoleValue(mapKey, depth + 1, seen)})=${describeConsoleValue(mapValue, depth + 1, seen)}`,
      )
    }
  }
  if (value instanceof Set) {
    for (const item of value) {
      parts.push(`item(${describeConsoleValue(item, depth + 1, seen)})`)
    }
  }
  try {
    parts.push(`String()=${String(value)}`)
  } catch {
    parts.push('String()=[threw]')
  }
  return parts.join(' | ')
}

/** Record every console call made while the scan is installed. */
export function scanConsole(): ConsoleScan {
  const entries: string[] = []
  const originals = new Map<string, (...args: unknown[]) => void>()
  for (const level of LEVELS) {
    const original = console[level] as (...args: unknown[]) => void
    originals.set(level, original)
    console[level] = ((...args: unknown[]) => {
      entries.push(`${level}: ${args.map((argument) => describeConsoleValue(argument)).join(' ')}`)
    }) as typeof console.log
  }
  return {
    entries,
    restore: () => {
      for (const [level, original] of originals) {
        console[level as (typeof LEVELS)[number]] = original as typeof console.log
      }
    },
  }
}

/** Whether any recorded call mentions this text. */
export function scanMentions(scan: ConsoleScan, needle: string): boolean {
  return scan.entries.some((entry) => entry.includes(needle))
}
