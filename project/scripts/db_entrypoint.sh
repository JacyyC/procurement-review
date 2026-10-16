#!/bin/sh
# PostgreSQL entrypoint wrapper for the V1-04 delivery.
#
# Job: generate the superuser password exactly once per `bootstrap-secrets` volume, then
# hand control to the untouched upstream entrypoint.
#
# Why a wrapper instead of an environment variable: the password must never live in
# docker-compose.yml, in an image layer, in a container environment (visible to
# `docker inspect`) or in a log line.  The upstream entrypoint already supports
# `POSTGRES_PASSWORD_FILE`; it re-executes itself through `gosu postgres`, so the file
# has to stay readable by the `postgres` user inside the container.
#
# Idempotent: re-running against an existing volume reuses the stored password and never
# rotates it, so an existing database keeps working across `down` / `up`.

set -eu

secret_dir="${V104CS_BOOTSTRAP_DIR:-/run/v104cs/bootstrap}"
password_file="${POSTGRES_PASSWORD_FILE:-${secret_dir}/postgres_password}"

postgres_uid="$(id -u postgres)"
postgres_gid="$(id -g postgres)"

# 0700, owned by the database user: only this container's postgres process and root
# (the one-shot `init` service) can reach anything inside.
install -d -m 0700 -o "${postgres_uid}" -g "${postgres_gid}" "${secret_dir}"

if [ ! -s "${password_file}" ]; then
    tmp="${password_file}.$$"
    umask 077
    head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > "${tmp}"
    printf '\n' >> "${tmp}"
    chown "${postgres_uid}:${postgres_gid}" "${tmp}"
    chmod 0600 "${tmp}"
    mv "${tmp}" "${password_file}"
    echo "v104cs: generated a new PostgreSQL superuser password at ${password_file}"
else
    echo "v104cs: reusing the existing PostgreSQL superuser password at ${password_file}"
fi

chown "${postgres_uid}:${postgres_gid}" "${password_file}"
chmod 0600 "${password_file}"

exec docker-entrypoint.sh "$@"
