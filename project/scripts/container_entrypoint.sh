#!/bin/sh
# Backend container entrypoint for the V1-04 delivery.
#
# The API must start with the least-privilege `review_runtime` DSN and never with the
# administrator DSN.  The DSN therefore lives in a protected named volume that the `api`
# service mounts read-only; it is exported into the process environment here and is
# never printed.
#
# An explicitly provided REVIEW_DATABASE_URL always wins (`docker-compose run -e ...`),
# which is how the delivery check probes the fail-closed paths.  When no DSN is
# available the API still starts and every business endpoint answers 503: there is no
# in-memory or anonymous fallback.

set -eu

dsn_file="${V104CS_RUNTIME_DSN_FILE:-/run/v104cs/runtime/runtime_dsn}"

if [ "${REVIEW_DATABASE_URL+set}" != "set" ] && [ -r "${dsn_file}" ]; then
    REVIEW_DATABASE_URL="$(cat "${dsn_file}")"
    export REVIEW_DATABASE_URL
fi

exec "$@"
