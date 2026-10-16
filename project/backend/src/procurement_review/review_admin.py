"""Trusted local maintenance CLI for the V1-03 review runtime.

Every command here is a *local maintainer* operation.  It never prints a credential
value or a database URL: issued tokens are written to one protected file and only the
credential ID plus the safe file path are reported.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .authorization import generate_credential
from .mcp_startup import MCP_STARTUP_VERSION, STARTUP_FILE_ENV
from .review_schema import REVIEW_MIGRATIONS
from .review_store import ReviewStore
from .security_contracts import CREDENTIAL_DEFAULT_TTL_SECONDS, RuntimeScope
from .security_fixture import (
    SECURITY_FIXTURE_VERSION,
    build_seed_plan,
    fixture_checksum,
)

ADMIN_DATABASE_URL_ENV = "REVIEW_ADMIN_DATABASE_URL"
DEFAULT_CREDENTIAL_DIRECTORY = Path(".local/credentials")
DEFAULT_STARTUP_DIRECTORY = Path(".local/mcp-startup")
_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600


class AdminError(RuntimeError):
    """Raised for an explicit, safely reportable maintenance failure."""


def _database_url() -> str:
    value = os.environ.get(ADMIN_DATABASE_URL_ENV) or os.environ.get(
        "REVIEW_DATABASE_URL"
    )
    if not value:
        raise AdminError(
            f"{ADMIN_DATABASE_URL_ENV} (or REVIEW_DATABASE_URL) is required"
        )
    return value


def _store() -> ReviewStore:
    return ReviewStore.from_url(_database_url())


def _ensure_protected_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, _DIRECTORY_MODE)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != _DIRECTORY_MODE:
        raise AdminError(
            f"refusing to use {path}: directory mode is {mode:04o}, expected 0700"
        )


def _write_protected_file(path: Path, payload: str) -> None:
    _ensure_protected_directory(path.parent)
    if path.exists():
        raise AdminError(f"refusing to overwrite the existing protected file {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
    finally:
        os.chmod(path, _FILE_MODE)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != _FILE_MODE:
        raise AdminError(f"protected file {path} has mode {mode:04o}, expected 0600")


def _print(payload: dict[str, object]) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, sort_keys=True, indent=2)
    sys.stdout.write("\n")


def cmd_migrate(args: argparse.Namespace) -> int:
    store = _store()
    try:
        report = store.migrate()
        known = tuple(item.version for item in REVIEW_MIGRATIONS)
    finally:
        store.close()
    _print(
        {
            "command": "migrate",
            "applied": list(report.applied),
            "already_present": list(report.already_present),
            "known_versions": list(known),
        }
    )
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    if args.fixture != SECURITY_FIXTURE_VERSION:
        raise AdminError(f"unknown fixture {args.fixture}")
    plan = build_seed_plan()
    store = _store()
    try:
        report = store.seed(plan)
    finally:
        store.close()
    _print(
        {
            "command": "seed",
            "fixture_version": report.fixture_version,
            "fixture_checksum": fixture_checksum(),
            "plan_checksum": plan.checksum(),
            "tenants": report.tenants,
            "subjects": report.subjects,
            "cases": report.cases,
            "documents": report.documents,
            "retrieval_scores": report.retrieval_scores,
            "procurement_records": report.procurement_records,
            "already_present": report.already_present,
        }
    )
    return 0


def cmd_credentials_issue(args: argparse.Namespace) -> int:
    store = _store()
    try:
        tenant_id = args.tenant
        if tenant_id is None:
            for candidate in store.tenant_ids():
                if args.subject in store.subject_ids(candidate):
                    tenant_id = candidate
                    break
        if tenant_id is None:
            raise AdminError(f"unknown subject {args.subject}")
        if args.subject not in store.subject_ids(tenant_id):
            raise AdminError(f"unknown subject {args.subject}")
        token, credential_id, digest = generate_credential()
        issued_at = datetime.now(UTC)
        expires_at = issued_at + timedelta(seconds=args.ttl_seconds)
        store.issue_credential(
            credential_id=credential_id,
            tenant_id=tenant_id,
            subject_id=args.subject,
            token_digest=digest,
            issued_at=issued_at,
            expires_at=expires_at,
        )
    finally:
        store.close()
    output = (
        Path(args.output)
        if args.output
        else (DEFAULT_CREDENTIAL_DIRECTORY / f"{credential_id}.token")
    )
    _write_protected_file(output, token + "\n")
    _print(
        {
            "command": "credentials issue",
            "credential_id": credential_id,
            "subject_id": args.subject,
            "tenant_id": tenant_id,
            "issued_at": issued_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "token_file": str(output.resolve()),
            "token_printed": False,
        }
    )
    return 0


def cmd_credentials_revoke(args: argparse.Namespace) -> int:
    store = _store()
    try:
        revoked = store.revoke_credential(
            credential_id=args.credential_id,
            revoked_at=datetime.now(UTC),
            reason=args.reason,
        )
    finally:
        store.close()
    _print(
        {
            "command": "credentials revoke",
            "credential_id": args.credential_id,
            "revoked": revoked,
            "reason": args.reason,
        }
    )
    return 0


def cmd_subject_set_disabled(args: argparse.Namespace) -> int:
    store = _store()
    try:
        updated = store.set_subject_disabled(
            subject_id=args.subject, disabled=not args.enable
        )
    finally:
        store.close()
    _print(
        {
            "command": "subject disable" if not args.enable else "subject enable",
            "subject_id": args.subject,
            "tenants_updated": updated,
        }
    )
    return 0


def cmd_grant(args: argparse.Namespace) -> int:
    store = _store()
    try:
        changed = store.set_case_grant(
            tenant_id=args.tenant,
            case_id=args.case,
            subject_id=args.subject,
            granted=not args.revoke,
            occurred_at=datetime.now(UTC),
        )
    finally:
        store.close()
    _print(
        {
            "command": "grant revoke" if args.revoke else "grant add",
            "tenant_id": args.tenant,
            "case_id": args.case,
            "subject_id": args.subject,
            "changed": changed,
        }
    )
    return 0


def cmd_denials(args: argparse.Namespace) -> int:
    store = _store()
    try:
        if args.unanchored:
            rows = store.list_unanchored_denials(limit=args.limit)
        elif args.tenant and args.case:
            rows = store.list_case_denials(
                tenant_id=args.tenant, case_id=args.case, limit=args.limit
            )
        else:
            raise AdminError("provide --tenant/--case or --unanchored")
        payload = [
            {
                "denial_id": row.denial_id,
                "sequence": row.sequence,
                "occurred_at": row.occurred_at.isoformat(),
                "request_id": row.request_id,
                "operation": row.operation,
                "reason": row.reason,
                "actor_subject_id": row.actor_subject_id,
                "actor_tenant_id": row.actor_tenant_id,
                "case_id": row.case_id,
                "target_digest": row.target_digest,
            }
            for row in rows
        ]
    finally:
        store.close()
    _print({"command": "denials", "count": len(payload), "rows": payload})
    return 0


def cmd_startup_context(args: argparse.Namespace) -> int:
    """Write one protected stdio MCP startup context for an existing scope."""

    scope = RuntimeScope.model_validate_json(Path(args.scope_file).read_text("utf-8"))
    output = (
        Path(args.output)
        if args.output
        else (
            DEFAULT_STARTUP_DIRECTORY / f"{scope.case_id}-{secrets.token_hex(4)}.json"
        )
    )
    payload = json.dumps(
        {
            "startup_context_version": MCP_STARTUP_VERSION,
            "database_url": _database_url(),
            "scope": scope.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    _write_protected_file(output, payload + "\n")
    _print(
        {
            "command": "startup-context",
            "startup_file": str(output.resolve()),
            "startup_file_env": STARTUP_FILE_ENV,
            "case_id": scope.case_id,
            "tenant_id": scope.tenant_id,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review_admin",
        description="Local maintenance commands for the authenticated review runtime.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    migrate = sub.add_parser("migrate", help="create or upgrade the review schema")
    migrate.set_defaults(handler=cmd_migrate)

    seed = sub.add_parser("seed", help="load the synthetic registry fixture")
    seed.add_argument("--fixture", default=SECURITY_FIXTURE_VERSION)
    seed.set_defaults(handler=cmd_seed)

    credentials = sub.add_parser(
        "credentials", help="issue or revoke local credentials"
    )
    credential_sub = credentials.add_subparsers(
        dest="credentials_command", required=True
    )
    issue = credential_sub.add_parser("issue")
    issue.add_argument("--subject", required=True)
    issue.add_argument("--tenant", default=None)
    issue.add_argument(
        "--ttl-seconds", type=int, default=CREDENTIAL_DEFAULT_TTL_SECONDS
    )
    issue.add_argument("--output", default=None)
    issue.set_defaults(handler=cmd_credentials_issue)
    revoke = credential_sub.add_parser("revoke")
    revoke.add_argument("--credential-id", required=True)
    revoke.add_argument("--reason", default="local_maintenance_revocation")
    revoke.set_defaults(handler=cmd_credentials_revoke)

    subject = sub.add_parser("subject", help="disable or re-enable one subject")
    subject_sub = subject.add_subparsers(dest="subject_command", required=True)
    disable = subject_sub.add_parser("disable")
    disable.add_argument("--subject", required=True)
    disable.set_defaults(handler=cmd_subject_set_disabled, enable=False)
    enable = subject_sub.add_parser("enable")
    enable.add_argument("--subject", required=True)
    enable.set_defaults(handler=cmd_subject_set_disabled, enable=True)

    grant = sub.add_parser("grant", help="add or revoke one Case grant")
    grant.add_argument("--tenant", required=True)
    grant.add_argument("--case", required=True)
    grant.add_argument("--subject", required=True)
    grant.add_argument("--revoke", action="store_true")
    grant.set_defaults(handler=cmd_grant)

    denials = sub.add_parser("denials", help="read the trusted access-denial ledger")
    denials.add_argument("--tenant", default=None)
    denials.add_argument("--case", default=None)
    denials.add_argument("--unanchored", action="store_true")
    denials.add_argument("--limit", type=int, default=200)
    denials.set_defaults(handler=cmd_denials)

    startup = sub.add_parser(
        "startup-context",
        help="write one protected stdio MCP startup context",
    )
    startup.add_argument("--scope-file", required=True)
    startup.add_argument("--output", default=None)
    startup.set_defaults(handler=cmd_startup_context)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except AdminError as error:
        sys.stderr.write(f"review_admin: {error}\n")
        return 2
    except Exception as error:  # noqa: BLE001 - never echo a DSN or token
        sys.stderr.write(f"review_admin: {type(error).__name__}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
