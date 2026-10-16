#!/usr/bin/env python3
"""Least-privilege probe for the delivered database roles (ROLE-01).

Runs inside the backend image with the `init` service definition, so it can read the
generated role passwords from the protected volumes.  It answers one question with raw
counts: can the delivered runtime role reach anything it is not supposed to?

Every attempt below is a write against synthetic data in a local database.  The probe
expects each one to be *refused by PostgreSQL*; a statement that unexpectedly succeeds is
counted as a leak, not hidden.  The procurement fact checksum is re-read afterwards, and
the migrator role runs one create/drop pair as a positive control so the refusals cannot
be explained by a broken connection.

Output is one JSON document without secrets: role names, statement labels, SQLSTATE
codes and counts.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import psycopg

BOOTSTRAP_DIR = Path("/run/v104cs/bootstrap")
RUNTIME_DIR = Path("/run/v104cs/runtime")
DB_HOST = os.environ.get("V104CS_DB_HOST", "db")
DB_PORT = os.environ.get("V104CS_DB_PORT", "5432")
DB_NAME = os.environ.get("V104CS_DB_NAME", "review_workbench")

INSUFFICIENT_PRIVILEGE = "42501"

ROLE_MIGRATOR = "review_migrator"
ROLE_RUNTIME = "review_runtime"
ROLE_READONLY = "procurement_readonly"

CHECKSUM_SQL = (
    "SELECT md5(coalesce(string_agg(payload::text, '|' ORDER BY record_id), '')) "
    "FROM review_procurement_record"
)

# (label, statement) pairs the runtime role must never be able to execute.
RUNTIME_FORBIDDEN = (
    (
        "insert_procurement_fact",
        "INSERT INTO review_procurement_record (tenant_id, case_id, record_kind, "
        "record_id, dataset_version, as_of, payload) VALUES ('tenant-local-demo', "
        "'CASE-LOW-001', 'order', 'PO-PROBE', 'security-procurement.v1', "
        "DATE '2026-06-30', '{}'::jsonb)",
    ),
    (
        "update_procurement_fact",
        "UPDATE review_procurement_record SET payload = '{}'::jsonb",
    ),
    ("delete_procurement_fact", "DELETE FROM review_procurement_record"),
    ("truncate_procurement_fact", "TRUNCATE review_procurement_record"),
    (
        "insert_retrieval_score",
        "INSERT INTO review_retrieval_score (document_version, index_version, "
        "query_text, chunk_id, score) VALUES ('probe.v1', 'probe.v1', 'probe', "
        "'probe', 1.0)",
    ),
    ("create_table", "CREATE TABLE v104cs_forbidden_probe (id integer)"),
    (
        "alter_table",
        "ALTER TABLE review_procurement_record ADD COLUMN v104cs_probe text",
    ),
    ("drop_table", "DROP TABLE review_procurement_record"),
    ("create_extension", "CREATE EXTENSION IF NOT EXISTS vector"),
    (
        "insert_credential",
        "INSERT INTO review_credential (credential_id, tenant_id, subject_id, "
        "token_digest, digest_algorithm, issued_at, expires_at) VALUES "
        "('cred-probe', 'tenant-local-demo', 'subject-local-demo', "
        "'0000000000000000000000000000000000000000000000000000000000000000', "
        "'sha256', now(), now() + interval '1 day')",
    ),
    (
        "insert_seed_marker",
        "INSERT INTO review_seed_marker (fixture_version, checksum) "
        "VALUES ('probe.v1', 'probe')",
    ),
    # The runtime role holds the UPDATE privilege on two registry tables because the
    # accepted security layer takes row locks on them; the delivered guard trigger must
    # still refuse every real write.
    ("update_case_grant", "UPDATE review_case_grant SET revoked_at = now()"),
    ("delete_case_grant", "DELETE FROM review_case_grant"),
    (
        "update_authorization_revision",
        "UPDATE review_authorization_revision SET revision = revision + 1",
    ),
)

# Locking reads the accepted security layer performs on every human action.  They must
# keep working under least privilege: granting the privilege is what makes them legal,
# the guard trigger is what keeps it unusable for writes.
RUNTIME_LOCK_READS = (
    (
        "lock_case_grant",
        "SELECT revoked_at FROM review_case_grant WHERE tenant_id = 'tenant-local-demo' "
        "AND case_id = 'CASE-LOW-001' AND subject_id = 'subject-local-demo' FOR SHARE",
    ),
    (
        "lock_authorization_revision",
        "SELECT revision FROM review_authorization_revision "
        "WHERE tenant_id = 'tenant-local-demo' FOR SHARE",
    ),
)

# Privilege escalation needs an effect check rather than an error check: PostgreSQL
# answers a `GRANT` from a non-owner with a warning and grants nothing, so "no exception"
# must not be read as success.  The probe grants a privilege the read-only role does not
# hold and then asks the catalogue whether it actually took effect.
ESCALATION_STATEMENT = (
    "GRANT INSERT, UPDATE, DELETE ON review_run TO procurement_readonly"
)
ESCALATION_CHECK = (
    "SELECT has_table_privilege('procurement_readonly', 'review_run', 'INSERT')"
)

# The read-only role backs a protected stdio MCP startup context: it may read, never write.
READONLY_FORBIDDEN = (
    ("insert_procurement_fact", RUNTIME_FORBIDDEN[0][1]),
    ("create_table", RUNTIME_FORBIDDEN[5][1]),
    (
        "insert_run",
        "INSERT INTO review_run (run_id, tenant_id, case_id, case_version, status, "
        "revision, resume_count, created_by_subject, current_subject_id, "
        "request_checksum, document_version, dataset_version, index_version, "
        "rules_version, isolation, created_at, updated_at) VALUES "
        "('probe-run', 'tenant-local-demo', 'CASE-LOW-001', 'case-low-001.v1', "
        "'pending', 1, 0, 'subject-local-demo', 'subject-local-demo', 'probe', "
        "'probe', 'probe', 'probe', 'probe', '{}'::jsonb, now(), now())",
    ),
)


class ProbeError(RuntimeError):
    """Raised when the probe cannot run at all."""


@dataclass(slots=True)
class Attempt:
    label: str
    denied: bool
    sqlstate: str | None
    error_type: str | None


def _read_secret(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ProbeError(f"cannot read {path.name}: {error.strerror}") from error


def _dsn(user: str, password: str, database: str) -> str:
    return (
        f"postgresql://{user}:{quote(password, safe='')}@{DB_HOST}:{DB_PORT}/{database}"
    )


def _connect(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn, autocommit=True, connect_timeout=10)


def _attempt(conn: psycopg.Connection, label: str, statement: str) -> Attempt:
    try:
        conn.execute(statement)
    except psycopg.Error as error:
        return Attempt(
            label=label,
            denied=True,
            sqlstate=error.sqlstate,
            error_type=type(error).__name__,
        )
    return Attempt(label=label, denied=False, sqlstate=None, error_type=None)


def _escalation_probe(runtime_dsn: str, superuser_dsn: str) -> dict[str, object]:
    """Attempt to widen another role's privileges and verify the effective outcome."""

    notices: list[str] = []
    raised: str | None = None
    with _connect(runtime_dsn) as conn:
        conn.add_notice_handler(
            lambda diag: notices.append(str(diag.message_primary or ""))
        )
        try:
            conn.execute(ESCALATION_STATEMENT)
        except psycopg.Error as error:
            raised = error.sqlstate
    with _connect(superuser_dsn) as conn:
        effective = bool(conn.execute(ESCALATION_CHECK).fetchone()[0])
    return {
        "statement": ESCALATION_STATEMENT,
        "refused_with": raised,
        "notices": [item for item in notices if item],
        "readonly_insert_effective": effective,
    }


def _role_row(dsn: str) -> dict[str, object]:
    with _connect(dsn) as conn:
        row = conn.execute(
            "SELECT rolname, rolsuper, rolcreatedb, rolcreaterole, rolcanlogin "
            "FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
    return {
        "role": row[0],
        "superuser": bool(row[1]),
        "createdb": bool(row[2]),
        "createrole": bool(row[3]),
        "can_login": bool(row[4]),
    }


def _probe_role(
    dsn: str, forbidden: tuple[tuple[str, str], ...], *, select_probe: str
) -> dict[str, object]:
    attempts: list[Attempt] = []
    with _connect(dsn) as conn:
        read_ok = conn.execute(select_probe).fetchone() is not None
        checksum_before = conn.execute(CHECKSUM_SQL).fetchone()[0]
        for label, statement in forbidden:
            attempts.append(_attempt(conn, label, statement))
        checksum_after = conn.execute(CHECKSUM_SQL).fetchone()[0]
    return {
        "select_ok": bool(read_ok),
        "checksum_before": checksum_before,
        "checksum_after": checksum_after,
        "facts_unchanged": checksum_before == checksum_after,
        "blocked": [item.label for item in attempts if item.denied],
        "leaked": [item.label for item in attempts if not item.denied],
        "sqlstates": sorted(
            {item.sqlstate for item in attempts if item.denied and item.sqlstate}
        ),
        # A refusal must be a real privilege refusal, not a malformed statement
        # that merely failed to parse.
        "all_denials_are_insufficient_privilege": all(
            item.sqlstate == INSUFFICIENT_PRIVILEGE for item in attempts if item.denied
        ),
        "attempts": [
            {"label": item.label, "denied": item.denied, "sqlstate": item.sqlstate}
            for item in attempts
        ],
    }


def _lock_read_probe(dsn: str) -> list[dict[str, object]]:
    """Prove the accepted locking reads still work under least privilege."""

    results: list[dict[str, object]] = []
    with _connect(dsn) as conn:
        for label, statement in RUNTIME_LOCK_READS:
            attempt = _attempt(conn, label, statement)
            results.append(
                {
                    "label": label,
                    "ok": not attempt.denied,
                    "sqlstate": attempt.sqlstate,
                }
            )
    return results


def _registry_state(dsn: str) -> dict[str, object]:
    """Read the delivered registry guard and the privileges it guards, from the catalogue."""

    with _connect(dsn) as conn:
        guards = [
            row[0]
            for row in conn.execute(
                "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal "
                "AND tgname LIKE 'v104cs_%_readonly' ORDER BY tgname"
            ).fetchall()
        ]
        update_only = [
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' "
                "AND has_table_privilege('review_runtime', "
                "format('%I.%I', table_schema, table_name), 'UPDATE') "
                "AND NOT has_table_privilege('review_runtime', "
                "format('%I.%I', table_schema, table_name), 'INSERT') "
                "ORDER BY table_name"
            ).fetchall()
        ]
    return {
        "guards": guards,
        "runtime_update_only_tables": update_only,
        "guard_count": len(guards),
    }


def main() -> int:
    try:
        superuser_password = _read_secret(BOOTSTRAP_DIR / "postgres_password")
        migrator_password = _read_secret(BOOTSTRAP_DIR / "migrator_password")
        readonly_password = _read_secret(BOOTSTRAP_DIR / "procurement_password")
        runtime_password = _read_secret(RUNTIME_DIR / "runtime_password")
        database = DB_NAME
    except ProbeError as error:
        sys.stderr.write(f"role_probe: {error}\n")
        return 2

    runtime_dsn = _dsn(ROLE_RUNTIME, runtime_password, database)
    readonly_dsn = _dsn(ROLE_READONLY, readonly_password, database)
    migrator_dsn = _dsn(ROLE_MIGRATOR, migrator_password, database)

    runtime = _probe_role(
        runtime_dsn,
        RUNTIME_FORBIDDEN,
        select_probe="SELECT count(*) FROM review_case_registry",
    )
    readonly = _probe_role(
        readonly_dsn,
        READONLY_FORBIDDEN,
        select_probe="SELECT count(*) FROM review_procurement_record",
    )

    # Positive control: the migrator role owns DDL, so the refusals above are about
    # privilege and not about an unusable connection.
    with _connect(migrator_dsn) as conn:
        created = _attempt(
            conn, "create_table", "CREATE TABLE v104cs_positive_control (id integer)"
        )
        dropped = _attempt(conn, "drop_table", "DROP TABLE v104cs_positive_control")

    roles = [
        _role_row(runtime_dsn),
        _role_row(readonly_dsn),
        _role_row(migrator_dsn),
        _role_row(_dsn("postgres", superuser_password, database)),
    ]

    escalation = _escalation_probe(
        runtime_dsn, _dsn("postgres", superuser_password, database)
    )
    lock_reads = _lock_read_probe(runtime_dsn)
    registry = _registry_state(migrator_dsn)

    report = {
        "probe": "role-01",
        "database": database,
        "roles": roles,
        "runtime": runtime,
        "readonly": readonly,
        "escalation": escalation,
        "lock_reads": lock_reads,
        "registry": registry,
        "positive_control": {
            "create_table_denied": created.denied,
            "drop_table_denied": dropped.denied,
        },
        "counts": {
            "runtime_attempts": len(RUNTIME_FORBIDDEN),
            "runtime_blocked": len(runtime["blocked"]),
            "runtime_leaked": len(runtime["leaked"]),
            "readonly_attempts": len(READONLY_FORBIDDEN),
            "readonly_blocked": len(readonly["blocked"]),
            "readonly_leaked": len(readonly["leaked"]),
            "unauthorized_writes_succeeded": len(runtime["leaked"])
            + len(readonly["leaked"]),
            "escalation_effective": escalation["readonly_insert_effective"],
            "denials_are_insufficient_privilege": bool(
                runtime["all_denials_are_insufficient_privilege"]
                and readonly["all_denials_are_insufficient_privilege"]
            ),
            "procurement_facts_unchanged": bool(
                runtime["facts_unchanged"] and readonly["facts_unchanged"]
            ),
            "readonly_still_read_only": bool(
                readonly["select_ok"] and not escalation["readonly_insert_effective"]
            ),
            "lock_reads_ok": sum(1 for item in lock_reads if item["ok"]),
            "lock_reads_expected": len(RUNTIME_LOCK_READS),
            "registry_guards": registry["guard_count"],
        },
    }
    json.dump(report, sys.stdout, ensure_ascii=False, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    counts = report["counts"]
    clean = (
        counts["unauthorized_writes_succeeded"] == 0
        and counts["denials_are_insufficient_privilege"]
        and counts["procurement_facts_unchanged"]
        and counts["escalation_effective"] is False
        and counts["readonly_still_read_only"] is True
        and counts["lock_reads_ok"] == counts["lock_reads_expected"]
        and counts["registry_guards"] == counts["lock_reads_expected"]
        and report["positive_control"]["create_table_denied"] is False
        and report["positive_control"]["drop_table_denied"] is False
    )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
