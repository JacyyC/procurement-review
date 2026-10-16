#!/usr/bin/env python3
"""Local credential maintenance through the delivered `init` service definition.

The one-shot `init` container is the only place in the delivered topology that holds the
migrator credential, so every credential operation that needs it runs here.  The wrapper
adds the two things the delivery topology needs on top of the accepted
`review_admin credentials ...` CLI:

1. it reads the migrator DSN from the protected bootstrap volume instead of taking it on
   a command line, so no DSN ever reaches a shell history or a log;
2. it re-issues its own file on every run, so repeating an acceptance run does not fail
   on the "refuse to overwrite a protected file" rule of the CLI.

Subcommands mirror the accepted maintenance surface: `issue`, `revoke` and `denials`.
Only non-secret metadata is printed; the token itself stays in the `0700`/`0600` file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BOOTSTRAP_DIR = Path(os.environ.get("V104CS_BOOTSTRAP_DIR", "/run/v104cs/bootstrap"))
CREDENTIAL_DIR = Path(
    os.environ.get("V104CS_CREDENTIAL_DIR", "/run/v104cs/delivery/credentials")
)
ADMIN_DSN_ENV = "REVIEW_ADMIN_DATABASE_URL"
DEFAULT_TTL_SECONDS = 2 * 60 * 60


def _review_admin(arguments: list[str]) -> subprocess.CompletedProcess:
    dsn = (BOOTSTRAP_DIR / "migrator_dsn").read_text(encoding="utf-8").strip()
    return subprocess.run(
        [sys.executable, "-m", "procurement_review.review_admin", *arguments],
        capture_output=True,
        text=True,
        env={**os.environ, ADMIN_DSN_ENV: dsn},
        check=False,
    )


def _fail(completed: subprocess.CompletedProcess) -> int:
    detail = (completed.stderr or completed.stdout).strip()[:300]
    sys.stderr.write(
        f"credential_admin: review_admin exited {completed.returncode}: {detail}\n"
    )
    return 1


def issue(subject: str, name: str, ttl_seconds: int) -> int:
    CREDENTIAL_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(CREDENTIAL_DIR, 0o700)
    token_path = CREDENTIAL_DIR / f"{name}.token"
    if token_path.exists():
        token_path.unlink()
    completed = _review_admin(
        [
            "credentials",
            "issue",
            "--subject",
            subject,
            "--ttl-seconds",
            str(ttl_seconds),
            "--output",
            str(token_path),
        ]
    )
    if completed.returncode != 0:
        return _fail(completed)
    report = json.loads(completed.stdout)
    os.chmod(token_path, 0o600)
    json.dump(
        {
            "command": "credentials issue",
            "subject_id": report["subject_id"],
            "tenant_id": report["tenant_id"],
            "credential_id": report["credential_id"],
            "expires_at": report["expires_at"],
            "token_file": report["token_file"],
            "token_printed": False,
        },
        sys.stdout,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


def _passthrough(arguments: list[str]) -> int:
    completed = _review_admin(arguments)
    if completed.returncode != 0:
        return _fail(completed)
    sys.stdout.write(completed.stdout)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    issue_parser = sub.add_parser(
        "issue", help="issue one credential into the protected path"
    )
    issue_parser.add_argument("--subject", required=True)
    issue_parser.add_argument(
        "--name", required=True, help="file stem in the credential dir"
    )
    issue_parser.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS)

    revoke_parser = sub.add_parser("revoke", help="revoke one credential by id")
    revoke_parser.add_argument("--credential-id", required=True)
    revoke_parser.add_argument("--reason", default="local_maintenance_revocation")

    denials_parser = sub.add_parser(
        "denials", help="read the trusted access-denial ledger"
    )
    denials_parser.add_argument("--limit", type=int, default=50)
    denials_parser.add_argument("--tenant", default=None)
    denials_parser.add_argument("--case", default=None)
    denials_parser.add_argument(
        "--unanchored",
        action="store_true",
        help="read the maintenance ledger of denials without a resolvable Case",
    )

    arguments = parser.parse_args(argv)
    if arguments.command == "issue":
        return issue(arguments.subject, arguments.name, arguments.ttl_seconds)
    if arguments.command == "revoke":
        return _passthrough(
            [
                "credentials",
                "revoke",
                "--credential-id",
                arguments.credential_id,
                "--reason",
                arguments.reason,
            ]
        )
    command = ["denials", "--limit", str(arguments.limit)]
    if arguments.unanchored:
        command.append("--unanchored")
    if arguments.tenant:
        command += ["--tenant", arguments.tenant]
    if arguments.case:
        command += ["--case", arguments.case]
    return _passthrough(command)


if __name__ == "__main__":
    raise SystemExit(main())
