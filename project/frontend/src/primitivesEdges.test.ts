/** The narrow boundary helpers: every way a declared value can be the wrong shape. */
import { describe, expect, it } from 'vitest'
import {
  optionalInteger,
  reqBooleanOrNull,
  reqStringMap,
  reqStringOrNull,
} from './apiPrimitives'

describe('the boundary primitives', () => {
  it('reads a missing optional field as unrecorded and refuses a wrong type', () => {
    expect(optionalInteger({}, 'count', 'path', 0)).toBeNull()
    expect(optionalInteger({ count: null }, 'count', 'path', 0)).toBeNull()
    expect(optionalInteger({ count: 3 }, 'count', 'path', 0)).toBe(3)
    expect(() => optionalInteger({ count: 3.5 }, 'count', 'path', 0)).toThrowError(/count/)
    expect(() => optionalInteger({ count: '3' }, 'count', 'path', 0)).toThrowError(/count/)
  })

  it('reads a missing optional boolean as unrecorded and refuses anything else', () => {
    expect(reqBooleanOrNull({}, 'flag', 'path')).toBeNull()
    expect(reqBooleanOrNull({ flag: null }, 'flag', 'path')).toBeNull()
    expect(reqBooleanOrNull({ flag: false }, 'flag', 'path')).toBe(false)
    expect(() => reqBooleanOrNull({ flag: 'no' }, 'flag', 'path')).toThrowError(/flag/)
  })

  it('refuses a text that is neither a string nor null, and a non-string map value', () => {
    expect(reqStringOrNull({ name: null }, 'name', 'path')).toBeNull()
    expect(reqStringOrNull({ name: 'x' }, 'name', 'path')).toBe('x')
    expect(() => reqStringOrNull({ name: 7 }, 'name', 'path')).toThrowError(/name/)
    expect(reqStringMap({ map: { a: 'b' } }, 'map', 'path')).toEqual({ a: 'b' })
    expect(() => reqStringMap({ map: { a: 1 } }, 'map', 'path')).toThrowError(/map/)
    expect(() => reqStringMap({ map: 'x' }, 'map', 'path')).toThrowError(/map/)
  })
})
