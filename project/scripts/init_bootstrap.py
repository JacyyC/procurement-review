#!/usr/bin/env python3
"""One-shot delivery bootstrap for the V1-04 Compose workbench.

This script runs only in the `init` service.  It is the single place in the delivered
topology that holds a PostgreSQL superuser credential, and it exists to close the V1-03
retained limitation L-1: the delivered default is least privilege, not an administrator
DSN.

What it does, in order (Spec §5.1/§5.3):

1. creates the three database roles — ``review_migrator`` (DDL and seed, used only
   here), ``review_runtime`` (read/write on the review tables, read-only on procurement
   and retrieval facts), ``procurement_readonly`` (read-only, for a protected stdio MCP
   startup context);
2. narrows database-level access (``CONNECT`` is no longer granted to ``PUBLIC``);
3. runs the accepted ``review_admin migrate`` / ``seed --fixture security-fixture.v1``
   CLI with the *migrator* role, never with the superuser;
4. grants the runtime role exactly the table and sequence privileges the API uses, and
   nothing on procurement facts or DDL;
5. issues one demo credential into the protected host path and records only its
   identifier, subject and expiry.

Every step is idempotent: re-running it never duplicates seed data, never rotates a
still-valid demo credential and never rotates a stored role password.  Passwords are
generated locally into the protected volumes; no password, DSN or token is ever printed.
"""

from __future__ import annotations

import json
import os
import secrets
import stat
import subprocess
import sys
import urllib.parse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from psycopg import sql

ROLE_MIGRATOR = "review_migrator"
ROLE_RUNTIME = "review_runtime"
ROLE_READONLY = "procurement_readonly"
ALL_ROLES = (ROLE_MIGRATOR, ROLE_RUNTIME, ROLE_READONLY)

# Where each generated role password is stored.  Only the runtime password lives in the
# volume the API mounts; the migrator and read-only secrets stay in the bootstrap volume
# that no long-running service mounts.
PASSWORD_FILE_NAMES = {
    ROLE_MIGRATOR: "migrator_password",
    ROLE_RUNTIME: "runtime_password",
    ROLE_READONLY: "procurement_password",
}

# The runtime DSN is the only one a long-running service reads; it is written into the
# volume the API mounts.  Everything else stays in the bootstrap volume.
DSN_FILE_NAMES = {
    ROLE_MIGRATOR: "migrator_dsn",
    ROLE_RUNTIME: "runtime_dsn",
    ROLE_READONLY: "readonly_dsn",
}

SCHEMA = "public"

# The only tables the API may write.  Everything else — the authorization registry, the
# registered documents and the procurement/retrieval facts — stays read-only for the
# runtime role, so a compromised application process cannot rewrite facts or mint a
# credential.  Kept in sync with review_store/review_repository write targets.
RUNTIME_WRITABLE_TABLES = (
    "review_access_denial",
    "review_audit_event",
    "review_branch_checkpoint",
    "review_idempotency",
    "review_receipt",
    "review_run",
    "review_run_generation",
    "review_run_summary",
    "review_sequence_allocator",
    "review_trace_event",
    "review_worker_attempt",
)

# The accepted security layer reads its own authorization state with a row lock
# (`SELECT ... FOR SHARE` on the Case grant and on the tenant authorization revision),
# and PostgreSQL requires the UPDATE privilege for a locking read even when nothing is
# written.  Handing the runtime role that privilege would let a compromised application
# rewrite its own authorizations, so the delivery pairs the grant with a guard trigger:
# row locks do not fire triggers, while every real UPDATE/DELETE issued by the runtime
# role is refused with SQLSTATE 42501 — the same code PostgreSQL uses for a missing
# privilege, so the refusal is indistinguishable from an ungranted statement.
RUNTIME_LOCK_ONLY_TABLES = ("review_authorization_revision", "review_case_grant")

# Delivery-created guard for the tables above; versioned migrations stay untouched.
GUARD_FUNCTION = "v104cs_registry_readonly"
GUARD_TRIGGER_PREFIX = "v104cs_"
GUARD_TRIGGER_SUFFIX = "_readonly"

CREDENTIAL_FILE_NAME = "demo-operator.token"
MARKER_FILE_NAME = "demo-credential.json"

ADMIN_DSN_ENV = "REVIEW_ADMIN_DATABASE_URL"
DEFAULT_DEMO_SUBJECT = "subject-local-demo"
DEFAULT_TTL_SECONDS = 30 * 24 * 60 * 60

_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600


class BootstrapError(RuntimeError):
    """Raised for an explicit, safely reportable bootstrap failure."""


# --------------------------------------------------------------------------- config


class Config:
    """Bootstrap configuration; every value has a safe delivery default."""

    def __init__(self, environ: Mapping[str, str]) -> None:
        self.host = environ.get("V104CS_DB_HOST", "db")
        self.port = int(environ.get("V104CS_DB_PORT", "5432"))
        self.database = environ.get("V104CS_DB_NAME", "review_workbench")
        self.superuser = environ.get("V104CS_SUPERUSER", "postgres")
        self.bootstrap_dir = Path(
            environ.get("V104CS_BOOTSTRAP_DIR", "/run/v104cs/bootstrap")
        )
        self.runtime_dir = Path(
            environ.get("V104CS_RUNTIME_DIR", "/run/v104cs/runtime")
        )
        self.credential_dir = Path(
            environ.get("V104CS_CREDENTIAL_DIR", "/run/v104cs/delivery/credentials")
        )
        self.runtime_uid = int(environ.get("V104CS_RUNTIME_UID", "10001"))
        self.runtime_gid = int(environ.get("V104CS_RUNTIME_GID", "10001"))
        self.demo_subject = environ.get("V104CS_DEMO_SUBJECT", DEFAULT_DEMO_SUBJECT)
        self.ttl_seconds = int(
            environ.get("V104CS_CREDENTIAL_TTL_SECONDS", str(DEFAULT_TTL_SECONDS))
        )

    @property
    def postgres_password_file(self) -> Path:
        return self.bootstrap_dir / "postgres_password"

    def password_file(self, role: str) -> Path:
        directory = self.runtime_dir if role == ROLE_RUNTIME else self.bootstrap_dir
        return directory / PASSWORD_FILE_NAMES[role]

    def dsn_file(self, role: str) -> Path:
        directory = self.runtime_dir if role == ROLE_RUNTIME else self.bootstrap_dir
        return directory / DSN_FILE_NAMES[role]

    def dsn(
        self,
        role: str,
        password: str,
        *,
        database: str | None = None,
        sqlalchemy: bool = True,
    ) -> str:
        """Build a DSN.  The stored form is the SQLAlchemy one the app consumes."""

        driver = "postgresql+psycopg" if sqlalchemy else "postgresql"
        quoted = urllib.parse.quote(password, safe="")
        return (
            f"{driver}://{role}:{quoted}@{self.host}:{self.port}"
            f"/{database or self.database}"
        )


# ------------------------------------------------------------------- secret handling


def _ensure_directory(path: Path, mode: int, uid: int | None, gid: int | None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, mode)
    if uid is not None and gid is not None:
        os.chown(path, uid, gid)
    observed = stat.S_IMODE(path.stat().st_mode)
    if observed != mode:
        raise BootstrapError(f"{path} has mode {observed:04o}, expected {mode:04o}")


def _write_secret(
    path: Path, value: str, *, uid: int | None = None, gid: int | None = None
) -> None:
    """Write one protected file atomically; never logs the value."""

    _ensure_directory(path.parent, _DIRECTORY_MODE, None, None)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
    finally:
        os.chmod(temporary, _FILE_MODE)
    if uid is not None and gid is not None:
        os.chown(temporary, uid, gid)
    os.replace(temporary, path)
    observed = stat.S_IMODE(path.stat().st_mode)
    if observed != _FILE_MODE:
        raise BootstrapError(f"{path} has mode {observed:04o}, expected 0600")


def _read_secret(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise BootstrapError(f"cannot read {path}: {error.strerror}") from error
    if not value:
        raise BootstrapError(f"{path} is empty")
    return value


def _redact(text: str) -> str:
    """Strip anything that looks like a DSN from a message before reporting it."""

    return " ".join(
        "<redacted>" if "://" in token else token for token in text.split()
    ).strip()


# ------------------------------------------------------------------------ database


def _connect(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn, autocommit=True, connect_timeout=10)


def _role_exists(conn: psycopg.Connection, role: str) -> bool:
    row = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
    return row is not None


def ensure_role(conn: psycopg.Connection, config: Config, role: str) -> tuple[str, str]:
    """Return ``(action, password)``; an existing stored password is never rotated."""

    password_file = config.password_file(role)
    exists = _role_exists(conn, role)
    if exists and password_file.is_file() and password_file.stat().st_size > 0:
        return "reused", _read_secret(password_file)

    password = secrets.token_urlsafe(32)
    if exists:
        conn.execute(
            sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(password)
            )
        )
        action = "rotated"
    else:
        conn.execute(
            sql.SQL(
                "CREATE ROLE {} WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
                "NOREPLICATION NOBYPASSRLS PASSWORD {}"
            ).format(sql.Identifier(role), sql.Literal(password))
        )
        action = "created"

    uid = config.runtime_uid if role == ROLE_RUNTIME else None
    gid = config.runtime_gid if role == ROLE_RUNTIME else None
    _write_secret(password_file, password + "\n", uid=uid, gid=gid)
    return action, password


def _role_list(roles: Sequence[str]) -> sql.Composable:
    return sql.SQL(", ").join(sql.Identifier(role) for role in roles)


def configure_database(conn: psycopg.Connection, config: Config) -> None:
    database = sql.Identifier(config.database)
    conn.execute(
        sql.SQL("REVOKE CONNECT, TEMPORARY ON DATABASE {} FROM PUBLIC").format(database)
    )
    conn.execute(
        sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
            database, _role_list(ALL_ROLES)
        )
    )
    # `public` is not a shared scratch space: only the migrator may create objects.
    conn.execute(
        sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC").format(sql.Identifier(SCHEMA))
    )
    conn.execute(
        sql.SQL("GRANT USAGE, CREATE ON SCHEMA {} TO {}").format(
            sql.Identifier(SCHEMA), sql.Identifier(ROLE_MIGRATOR)
        )
    )
    conn.execute(
        sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
            sql.Identifier(SCHEMA), _role_list((ROLE_RUNTIME, ROLE_READONLY))
        )
    )


def grant_table_privileges(conn: psycopg.Connection) -> None:
    """Grant the runtime role exactly its working set, after migrations exist."""

    read_only = (ROLE_RUNTIME, ROLE_READONLY)
    conn.execute(
        sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA {} FROM {}").format(
            sql.Identifier(SCHEMA), _role_list(read_only)
        )
    )
    conn.execute(
        sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(
            sql.Identifier(SCHEMA), _role_list(read_only)
        )
    )
    conn.execute(
        sql.SQL("GRANT INSERT, UPDATE, DELETE ON {} TO {}").format(
            sql.SQL(", ").join(
                sql.Identifier(SCHEMA, table) for table in RUNTIME_WRITABLE_TABLES
            ),
            sql.Identifier(ROLE_RUNTIME),
        )
    )
    conn.execute(
        sql.SQL("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA {} TO {}").format(
            sql.Identifier(SCHEMA), sql.Identifier(ROLE_RUNTIME)
        )
    )
    # Locking reads need the UPDATE privilege; the guard below keeps it unusable.
    conn.execute(
        sql.SQL("GRANT UPDATE ON {} TO {}").format(
            sql.SQL(", ").join(
                sql.Identifier(SCHEMA, table) for table in RUNTIME_LOCK_ONLY_TABLES
            ),
            sql.Identifier(ROLE_RUNTIME),
        )
    )
    install_registry_guard(conn)


def install_registry_guard(conn: psycopg.Connection) -> None:
    """Refuse every real write to the authorization registry by the runtime role.

    Idempotent: the function is replaced and each trigger is recreated, so a repeated
    bootstrap leaves the same objects behind.
    """

    message = "the authorization registry is read-only for the review_runtime role"
    conn.execute(
        sql.SQL(
            "CREATE OR REPLACE FUNCTION {schema}.{function}() RETURNS trigger "
            "LANGUAGE plpgsql AS $guard$ BEGIN "
            "IF current_user = {role} THEN "
            "RAISE EXCEPTION {message} USING ERRCODE = '42501'; "
            "END IF; RETURN NULL; END $guard$"
        ).format(
            schema=sql.Identifier(SCHEMA),
            function=sql.Identifier(GUARD_FUNCTION),
            role=sql.Literal(ROLE_RUNTIME),
            message=sql.Literal(message),
        )
    )
    for table in RUNTIME_LOCK_ONLY_TABLES:
        trigger = f"{GUARD_TRIGGER_PREFIX}{table}{GUARD_TRIGGER_SUFFIX}"
        conn.execute(
            sql.SQL("DROP TRIGGER IF EXISTS {trigger} ON {table}").format(
                trigger=sql.Identifier(trigger),
                table=sql.Identifier(SCHEMA, table),
            )
        )
        conn.execute(
            sql.SQL(
                "CREATE TRIGGER {trigger} BEFORE UPDATE OR DELETE ON {table} "
                "FOR EACH STATEMENT EXECUTE FUNCTION {schema}.{function}()"
            ).format(
                trigger=sql.Identifier(trigger),
                table=sql.Identifier(SCHEMA, table),
                schema=sql.Identifier(SCHEMA),
                function=sql.Identifier(GUARD_FUNCTION),
            )
        )


def assert_runtime_boundary(conn: psycopg.Connection) -> dict[str, object]:
    """Re-read the delivered grants from the catalogue as a self-check."""

    rows = conn.execute(
        """
        SELECT table_name,
               has_table_privilege(%s, format('%%I.%%I', table_schema, table_name),
                                   'INSERT') AS can_insert,
               has_table_privilege(%s, format('%%I.%%I', table_schema, table_name),
                                   'DELETE') AS can_delete
        FROM information_schema.tables
        WHERE table_schema = %s
        ORDER BY table_name
        """,
        (ROLE_RUNTIME, ROLE_RUNTIME, SCHEMA),
    ).fetchall()
    writable = tuple(row[0] for row in rows if row[1] or row[2])
    forbidden = tuple(
        row[0]
        for row in rows
        if row[0] in ("review_procurement_record", "review_retrieval_score")
        and (row[1] or row[2])
    )
    create_on_schema = conn.execute(
        "SELECT has_schema_privilege(%s, %s, 'CREATE')", (ROLE_RUNTIME, SCHEMA)
    ).fetchone()[0]
    if forbidden or create_on_schema:
        raise BootstrapError(
            "the runtime role can write procurement facts or create schema objects"
        )
    update_only = tuple(
        row[0]
        for row in rows
        if row[0] not in RUNTIME_WRITABLE_TABLES
        and row[0] not in RUNTIME_LOCK_ONLY_TABLES
        and _can_update(conn, row[0])
    )
    if update_only:
        raise BootstrapError(
            f"the runtime role can update tables outside its working set: {update_only}"
        )
    guards = tuple(
        row[0]
        for row in conn.execute(
            "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal "
            "AND tgname LIKE %s ORDER BY tgname",
            (f"{GUARD_TRIGGER_PREFIX}%{GUARD_TRIGGER_SUFFIX}",),
        ).fetchall()
    )
    if len(guards) != len(RUNTIME_LOCK_ONLY_TABLES):
        raise BootstrapError(
            "the authorization-registry guard is not installed for every locked table"
        )
    return {
        "runtime_writable_tables": sorted(writable),
        "runtime_lock_only_tables": sorted(RUNTIME_LOCK_ONLY_TABLES),
        "procurement_fact_write_allowed": bool(forbidden),
        "runtime_schema_create_allowed": bool(create_on_schema),
        "registry_guards": sorted(guards),
    }


def _can_update(conn: psycopg.Connection, table: str) -> bool:
    return bool(
        conn.execute(
            "SELECT has_table_privilege(%s, %s, 'UPDATE')",
            (ROLE_RUNTIME, f"{SCHEMA}.{table}"),
        ).fetchone()[0]
    )


# --------------------------------------------------------------------- review_admin


def run_review_admin(config: Config, dsn: str, arguments: Sequence[str]) -> dict:
    environment = dict(os.environ)
    environment[ADMIN_DSN_ENV] = dsn
    completed = subprocess.run(
        [sys.executable, "-m", "procurement_review.review_admin", *arguments],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        detail = _redact(completed.stderr or completed.stdout)
        raise BootstrapError(
            f"review_admin {' '.join(arguments)} failed "
            f"(exit {completed.returncode}): {detail}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:  # pragma: no cover - defensive
        raise BootstrapError(
            f"review_admin {' '.join(arguments)} returned unreadable output"
        ) from error


# ----------------------------------------------------------------------- credential


def _credential_marker(config: Config) -> Path:
    return config.bootstrap_dir / MARKER_FILE_NAME


def _read_marker(config: Config) -> dict | None:
    path = _credential_marker(config)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _credential_is_usable(config: Config, password: str, credential_id: str) -> bool:
    from procurement_review.review_store import ReviewStore

    store = ReviewStore.from_url(config.dsn(ROLE_MIGRATOR, password))
    try:
        metadata = store.credential_metadata(credential_id)
    finally:
        store.close()
    if metadata is None:
        return False
    if metadata.get("revoked_at") is not None:
        return False
    expires_at = metadata.get("expires_at")
    return bool(expires_at and expires_at > datetime.now(UTC))


def _align_ownership(path: Path, reference: Path) -> dict[str, int | bool]:
    """Best-effort alignment with the bind-mount owner.

    On macOS the Docker file-sharing layer presents the host user; on Linux the
    container runs as root and would otherwise leave a root-owned file the operator
    cannot read.  A failure here is reported, never fatal.
    """

    try:
        stats = reference.stat()
    except OSError:
        return {"aligned": False}
    try:
        os.chown(path, stats.st_uid, stats.st_gid)
    except OSError:
        return {"aligned": False, "reference_uid": stats.st_uid}
    return {"aligned": True, "reference_uid": stats.st_uid}


def ensure_credential(
    config: Config, migrator_dsn: str, migrator_password: str
) -> tuple[dict, dict[str, object]]:
    """Issue the demo credential once; a still-valid one is never rotated."""

    token_path = config.credential_dir / CREDENTIAL_FILE_NAME
    marker = _read_marker(config)
    if marker is not None and token_path.is_file():
        credential_id = str(marker.get("credential_id", ""))
        if credential_id and _credential_is_usable(
            config, migrator_password, credential_id
        ):
            _align_ownership(token_path, config.credential_dir.parent)
            return (
                {
                    "action": "reused",
                    "credential_id": credential_id,
                    "subject_id": marker.get("subject_id"),
                    "expires_at": marker.get("expires_at"),
                    "token_file": str(token_path),
                    "token_printed": False,
                },
                _protection_report(config, token_path),
            )

    if token_path.exists():
        # A recognised stale token from a previous database would be unusable; replacing
        # it is the only way to keep the one-command start truthful.
        token_path.unlink()

    issued = run_review_admin(
        config,
        migrator_dsn,
        [
            "credentials",
            "issue",
            "--subject",
            config.demo_subject,
            "--ttl-seconds",
            str(config.ttl_seconds),
            "--output",
            str(token_path),
        ],
    )
    ownership = _align_ownership(token_path, config.credential_dir.parent)
    _write_secret(
        _credential_marker(config),
        json.dumps(
            {
                "subject_id": issued["subject_id"],
                "credential_id": issued["credential_id"],
                "issued_at": issued["issued_at"],
                "expires_at": issued["expires_at"],
                "token_file": str(token_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return (
        {
            "action": "issued",
            "credential_id": issued["credential_id"],
            "subject_id": issued["subject_id"],
            "expires_at": issued["expires_at"],
            "token_file": str(token_path),
            "token_printed": False,
            "ownership": ownership,
        },
        _protection_report(config, token_path),
    )


def _protection_report(config: Config, token_path: Path) -> dict[str, object]:
    directory_mode = stat.S_IMODE(config.credential_dir.stat().st_mode)
    file_mode = stat.S_IMODE(token_path.stat().st_mode)
    if directory_mode != _DIRECTORY_MODE or file_mode != _FILE_MODE:
        raise BootstrapError(
            "the demo credential is not protected "
            f"(dir {directory_mode:04o}, file {file_mode:04o})"
        )
    return {
        "credential_dir": str(config.credential_dir),
        "credential_dir_mode": f"{directory_mode:04o}",
        "token_file_mode": f"{file_mode:04o}",
    }


# ------------------------------------------------------------------------------ main


def bootstrap(config: Config) -> dict:
    superuser_password = _read_secret(config.postgres_password_file)
    superuser_dsn = config.dsn(
        config.superuser,
        superuser_password,
        database=config.database,
        sqlalchemy=False,
    )

    _ensure_directory(config.bootstrap_dir, _DIRECTORY_MODE, None, None)
    _ensure_directory(
        config.runtime_dir, _DIRECTORY_MODE, config.runtime_uid, config.runtime_gid
    )
    config.credential_dir.mkdir(parents=True, exist_ok=True)

    actions: dict[str, str] = {}
    passwords: dict[str, str] = {}
    with _connect(superuser_dsn) as admin:
        for role in ALL_ROLES:
            action, password = ensure_role(admin, config, role)
            actions[role] = action
            passwords[role] = password
        configure_database(admin, config)

    migrator_dsn = config.dsn(ROLE_MIGRATOR, passwords[ROLE_MIGRATOR])
    migration = run_review_admin(config, migrator_dsn, ["migrate"])
    seed = run_review_admin(
        config, migrator_dsn, ["seed", "--fixture", "security-fixture.v1"]
    )

    with _connect(superuser_dsn) as admin:
        grant_table_privileges(admin)
        boundary = assert_runtime_boundary(admin)

    credential, protection = ensure_credential(
        config, migrator_dsn, passwords[ROLE_MIGRATOR]
    )

    for role in ALL_ROLES:
        _write_secret(
            config.dsn_file(role),
            config.dsn(role, passwords[role]) + "\n",
            uid=config.runtime_uid if role == ROLE_RUNTIME else None,
            gid=config.runtime_gid if role == ROLE_RUNTIME else None,
        )

    runtime_dsn = config.dsn(ROLE_RUNTIME, passwords[ROLE_RUNTIME], sqlalchemy=False)
    with _connect(runtime_dsn) as runtime:
        runtime_row = runtime.execute(
            "SELECT current_user, "
            "(SELECT rolsuper FROM pg_roles WHERE rolname = current_user)"
        ).fetchone()

    return {
        "command": "init",
        "database": config.database,
        "roles": actions,
        "role_attributes": {
            "runtime_user": runtime_row[0],
            "runtime_is_superuser": bool(runtime_row[1]),
        },
        "migrate": migration,
        "seed": {
            key: seed.get(key)
            for key in (
                "fixture_version",
                "fixture_checksum",
                "plan_checksum",
                "tenants",
                "subjects",
                "cases",
                "documents",
                "retrieval_scores",
                "procurement_records",
                "already_present",
            )
        },
        "credential": credential,
        "protection": protection,
        "boundary": boundary,
        "runtime_dsn_file": str(config.dsn_file(ROLE_RUNTIME)),
        "migrator_dsn_file": str(config.dsn_file(ROLE_MIGRATOR)),
    }


def main() -> int:
    config = Config(os.environ)
    try:
        report = bootstrap(config)
    except BootstrapError as error:
        sys.stderr.write(f"v104cs init: {_redact(str(error))}\n")
        return 2
    except Exception as error:  # noqa: BLE001 - never echo a DSN or a password
        sys.stderr.write(f"v104cs init: {type(error).__name__}\n")
        return 1
    json.dump(report, sys.stdout, ensure_ascii=False, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
