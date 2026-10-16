// FIX-04B / C-AUD-07 console recorder (test asset, injected before any page script runs).
//
// A check that only looks at strings, or that stringifies with JSON.stringify, misses the
// leak it is meant to find: `Error.message`/`stack`/`cause` are non-enumerable, and
// `JSON.stringify(new Error('x'))` is `"{}"`.  This walks every console argument by its own
// property names — enumerable or not — including Map/Set contents, and records
// `String(value)` as well, because that is how an Error prints.
;(function installConsoleScan() {
  var CANARY = 'FIX04-PROBE-CANARY'
  var scan = []
  var pageErrors = []
  window.__consoleScan = scan
  window.__pageErrors = pageErrors
  window.__canary = CANARY

  function describe(value, depth, seen) {
    if (value === null || value === undefined) return String(value)
    var kind = typeof value
    if (kind === 'string') return value
    if (kind === 'number' || kind === 'boolean' || kind === 'bigint') return String(value)
    if (kind === 'symbol') return String(value)
    if (kind === 'function') return '[function ' + (value.name || 'anonymous') + ']'
    if (depth > 6) return '[depth-limit]'
    if (seen.indexOf(value) >= 0) return '[cycle]'
    seen.push(value)
    var parts = []
    try {
      var keys = Object.getOwnPropertyNames(value)
      for (var index = 0; index < keys.length; index += 1) {
        var nested
        try {
          nested = value[keys[index]]
        } catch {
          nested = '[unreadable]'
        }
        parts.push(keys[index] + '=' + describe(nested, depth + 1, seen))
      }
    } catch {
      parts.push('[own-properties-unreadable]')
    }
    try {
      if (value instanceof Map) {
        value.forEach(function (mapValue, mapKey) {
          parts.push('entry(' + describe(mapKey, depth + 1, seen) + ')=' + describe(mapValue, depth + 1, seen))
        })
      }
      if (value instanceof Set) {
        value.forEach(function (item) {
          parts.push('item(' + describe(item, depth + 1, seen) + ')')
        })
      }
    } catch {
      parts.push('[collections-unreadable]')
    }
    try {
      parts.push('String()=' + String(value))
    } catch {
      parts.push('String()=[threw]')
    }
    return parts.join(' | ')
  }

  var levels = ['log', 'info', 'warn', 'error', 'debug', 'trace']
  for (var index = 0; index < levels.length; index += 1) {
    ;(function wrap(level) {
      var original = console[level].bind(console)
      console[level] = function () {
        var args = Array.prototype.slice.call(arguments)
        var parts = []
        for (var position = 0; position < args.length; position += 1) {
          parts.push(describe(args[position], 0, []))
        }
        scan.push(level + ': ' + parts.join(' '))
        original.apply(null, args)
      }
    })(levels[index])
  }

  window.addEventListener('error', function (event) {
    pageErrors.push(String(event.message))
  })
  window.addEventListener('unhandledrejection', function (event) {
    pageErrors.push(String(event.reason))
  })
})()
