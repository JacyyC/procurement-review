#!/usr/bin/env python3
"""Print registry row counts and the demo-credential marker for delivery evidence.

Runs in a one-off container with the `init` service definition (bootstrap volume
mounted) so it can use the migrator DSN.  It reports counts only — no business payload,
no credential value, no DSN.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

BOOTSTRAP_DIR = Path(os.environ.get("V104CS_BOOTSTRAP_DIR", "/run/v104cs/bootstrap"))
TABLES = (
    "review_tenant",
    "review_subject",
    "review_subject_role",
    "review_case_registry",
    "review_case_grant",
    "review_document_registry",
    "review_case_document",
    "review_procurement_record",
    "review_retrieval_score",
    "review_credential",
    "review_seed_marker",
    "review_run",
    "review_receipt",
    "review_audit_event",
    "review_trace_event",
    "review_access_denial",
)


def main() -> int:
    dsn = (BOOTSTRAP_DIR / "migrator_dsn").read_text(encoding="utf-8").strip()
    engine = create_engine(dsn)
    counts: dict[str, int] = {}
    try:
        with engine.connect() as connection:
            for table in TABLES:
                counts[table] = int(
                    connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
                )
            counts["review_credential_active"] = int(
                connection.execute(
                    text(
                        "SELECT count(*) FROM review_credential "
                        "WHERE revoked_at IS NULL AND expires_at > now()"
                    )
                ).scalar_one()
            )
            migrations = tuple(
                connection.execute(
                    text("SELECT version FROM review_schema_migration ORDER BY version")
                ).scalars()
            )
    finally:
        engine.dispose()

    marker_path = BOOTSTRAP_DIR / "demo-credential.json"
    marker = None
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))

    json.dump(
        {
            "counts": counts,
            "applied_migrations": list(migrations),
            "credential_marker": marker,
        },
        sys.stdout,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
