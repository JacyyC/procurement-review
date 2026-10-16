#!/usr/bin/env python3
"""Read the persisted run ledger through the runtime DSN (read-only).

`delivery_check.py` uses this probe to judge recovery and human-action evidence from the
database rather than from the transport projection: a Run's status, revision, resume
count, worker attempts, adopted checkpoints, generations, receipts and audit events are
read exactly as `procurement_review.security_recovery_check` reads them for the V1-03
matrix, so the delivery subset keeps the same counting rules.

It runs in the `api` service definition, which mounts the least-privilege runtime DSN
read-only and exports it as `REVIEW_DATABASE_URL`.  Only counters and identifiers are
printed: no business payload, no credential and no DSN.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from sqlalchemy import create_engine, text


def _runtime_dsn() -> str:
    value = os.environ.get("REVIEW_DATABASE_URL", "").strip()
    if not value:
        raise SystemExit("REVIEW_DATABASE_URL (the runtime DSN) is required")
    return value


def run_ledger(connection, run_id: str) -> dict[str, object]:
    attempts = connection.execute(
        text(
            "SELECT worker, attempt_no, outcome, abandoned, discarded_late "
            "FROM review_worker_attempt WHERE run_id = :r ORDER BY worker, attempt_no"
        ),
        {"r": run_id},
    ).all()
    checkpoints = connection.execute(
        text(
            "SELECT worker, generation, attempt_no FROM review_branch_checkpoint "
            "WHERE run_id = :r ORDER BY worker"
        ),
        {"r": run_id},
    ).all()
    run_row = connection.execute(
        text("SELECT status, revision, resume_count FROM review_run WHERE run_id = :r"),
        {"r": run_id},
    ).first()
    generations = connection.execute(
        text(
            "SELECT generation, status FROM review_run_generation "
            "WHERE run_id = :r ORDER BY generation"
        ),
        {"r": run_id},
    ).all()
    receipts = connection.execute(
        text("SELECT count(*) FROM review_receipt WHERE run_id = :r"), {"r": run_id}
    ).scalar_one()
    audits = connection.execute(
        text("SELECT count(*) FROM review_audit_event WHERE run_id = :r"), {"r": run_id}
    ).scalar_one()
    return {
        "run_id": run_id,
        "found": run_row is not None,
        "status": run_row.status if run_row is not None else None,
        "revision": int(run_row.revision) if run_row is not None else None,
        "resume_count": int(run_row.resume_count) if run_row is not None else None,
        "receipts": int(receipts),
        "audit_events": int(audits),
        "attempts": [
            {
                "worker": row.worker,
                "attempt_no": int(row.attempt_no),
                "outcome": row.outcome,
                "abandoned": bool(row.abandoned),
                "discarded_late": bool(row.discarded_late),
            }
            for row in attempts
        ],
        "checkpoints": [
            {
                "worker": row.worker,
                "generation": int(row.generation),
                "attempt_no": int(row.attempt_no),
            }
            for row in checkpoints
        ],
        "generations": [
            {"generation": int(row.generation), "status": row.status}
            for row in generations
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "denials"))
    parser.add_argument("--run-id", default=None)
    arguments = parser.parse_args(argv)
    if arguments.mode == "run" and not arguments.run_id:
        parser.error("--run-id is required for mode `run`")

    engine = create_engine(_runtime_dsn())
    try:
        with engine.connect() as connection:
            if arguments.mode == "run":
                payload: dict[str, object] = run_ledger(connection, arguments.run_id)
            else:
                payload = {
                    "denial_events": int(
                        connection.execute(
                            text("SELECT count(*) FROM review_access_denial")
                        ).scalar_one()
                    )
                }
    finally:
        engine.dispose()
    json.dump(payload, sys.stdout, ensure_ascii=False, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
