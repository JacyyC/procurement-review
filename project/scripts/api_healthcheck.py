#!/usr/bin/env python3
"""Container readiness probe for the delivered API.

Two things are asserted, both of which a healthcheck is allowed to check and neither of
which mutates anything:

1. the runtime DSN is configured and the database answers ``SELECT 1``;
2. the connected role is the expected least-privilege role — it is not a superuser and
   it cannot create objects in the schema, so a deployment accidentally wired to the
   administrator DSN is reported unhealthy instead of silently running too privileged.

Only a one-line, secret-free reason is printed (healthcheck output is retained by
``docker inspect``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

DSN_FILE_ENV = "V104CS_RUNTIME_DSN_FILE"
EXPECTED_ROLE_ENV = "V104CS_EXPECTED_ROLE"
DEFAULT_DSN_FILE = "/run/v104cs/runtime/runtime_dsn"
DEFAULT_EXPECTED_ROLE = "review_runtime"
DATABASE_URL_ENV = "REVIEW_DATABASE_URL"

PROBE = (
    "SELECT current_user AS role_name, "
    "(SELECT rolsuper FROM pg_roles WHERE rolname = current_user) AS is_superuser, "
    "has_schema_privilege(current_user, 'public', 'CREATE') AS can_create"
)


def _database_url() -> str:
    explicit = os.environ.get(DATABASE_URL_ENV)
    if explicit:
        return explicit
    path = Path(os.environ.get(DSN_FILE_ENV, DEFAULT_DSN_FILE))
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


def main() -> int:
    database_url = _database_url()
    if not database_url:
        print("unavailable: no runtime DSN configured")
        return 1

    engine = create_engine(
        database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5}
    )
    try:
        with engine.connect() as connection:
            row = connection.execute(text(PROBE)).mappings().one()
    except Exception as error:  # noqa: BLE001 - a healthcheck never raises
        print(f"unavailable: {type(error).__name__}")
        return 1
    finally:
        engine.dispose()

    expected = os.environ.get(EXPECTED_ROLE_ENV, DEFAULT_EXPECTED_ROLE)
    if row["role_name"] != expected:
        print(f"unexpected role: {row['role_name']}")
        return 1
    if row["is_superuser"]:
        print("runtime role is a superuser")
        return 1
    if row["can_create"]:
        print("runtime role can create schema objects")
        return 1

    print("ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
