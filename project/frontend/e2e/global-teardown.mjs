/**
 * Playwright global teardown: drop the disposable review database created for this
 * run and delete the issued credential files.  It is deliberately independent from
 * the API webServer's own shutdown so a crashed server cannot leak a database.
 */

import { down } from './review-db.mjs'

export default async function globalTeardown() {
  try {
    if (down()) {
      console.log('[review-e2e] dropped the disposable review database')
    }
  } catch (error) {
    console.error(`[review-e2e] teardown warning: ${error.message}`)
  }
}
