#!/usr/bin/env python3
"""Reconcile the versioned model-attempt ledgers of this repository, read-only.

Every batch writes one SQLite ledger beside its registration.  A ledger row is not a send
and a directory name is not a namespace, so this tool reports each ledger by its own
identity — the registration digest it was created under and the mode its run directory
recorded — and never adds a copied ledger to the original it copies.  Two consumers need
these numbers: the work item's accounting (what was really sent, by which batch) and the
audit trail of unknown usage, which must stay locatable instead of being filled with zero.

Real and offline ledgers cannot be told apart by their receipts: an offline mock writes the
resolved model id too.  The mode therefore comes from the run directory (``result.json``),
and a ledger whose mode is unknown is reported as ``unattributed`` rather than being
counted as real.

Usage:
    python3 project/scripts/model_attempt_reconciliation.py --root project/backend/.local --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

TERMINAL_STATES = {"committed", "failed", "cancelled", "remote_unknown"}
NAMESPACES = (
    ("v1-07c", "C"),
    ("v1-07f/live", "F"),
    ("v1-07f/v1-07g/probe", "G-probe"),
    ("v1-07g/paired", "G-paired"),
    ("v1-07g/paired-dry", "G-mock-paired"),
    ("v1-07g/probe-dryrun", "G-mock-probe"),
    ("v1-07g/probe-dryrun-diag", "G-mock-diag"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mode_of(directory: Path) -> str:
    """The mode the run directory itself recorded, or ``unattributed``."""

    result = directory / "result.json"
    if result.is_file():
        try:
            body = json.loads(result.read_text())
        except json.JSONDecodeError:
            return "unattributed"
        mode = body.get("mode")
        if isinstance(mode, str) and mode:
            return mode
    return "unattributed"


def _ledger_row(path: Path) -> dict | None:
    directory = path.parent
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return None
    try:
        tasks = connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        states = dict(
            connection.execute("SELECT state, COUNT(*) FROM attempts GROUP BY state")
        )
        row = connection.execute(
            "SELECT known_input_tokens, known_output_tokens, unknown_usage_attempts,"
            " reserved_requests FROM budget"
        ).fetchone()
    except sqlite3.DatabaseError:
        return None
    finally:
        connection.close()
    registration = directory / "registration.json"
    return {
        "path": str(path),
        "mode": _mode_of(directory),
        "registration_sha256": (
            _sha256(registration) if registration.is_file() else None
        ),
        "planned_tasks": int(tasks),
        "attempts": sum(states.values()),
        "states": states,
        "receipts": sum(
            count for state, count in states.items() if state in TERMINAL_STATES
        ),
        "committed": states.get("committed", 0),
        "failed": states.get("failed", 0),
        "unknown": states.get("remote_unknown", 0),
        "known_input_tokens": row[0],
        "known_output_tokens": row[1],
        "unknown_usage_attempts": row[2],
        "reserved_requests": row[3],
    }


def reconcile(root: Path) -> dict:
    """Every ledger under ``root``, grouped by namespace, with totals per mode."""

    namespaces: dict[str, dict] = {}
    for relative, label in NAMESPACES:
        base = root / relative
        if not base.is_dir():
            continue
        rows = []
        for ledger in sorted(base.rglob("ledger.sqlite3")):
            record = _ledger_row(ledger)
            if record is not None:
                rows.append(record)
        if rows:
            namespaces[label] = {
                "ledgers": rows,
                "totals": _totals(rows),
            }
    return {"root": str(root), "namespaces": namespaces, "grand_total": _totals(
        [row for entry in namespaces.values() for row in entry["ledgers"]]
    )}


def _totals(rows: list[dict]) -> dict:
    body: dict = {
        "ledgers": len(rows),
        "planned_tasks": 0,
        "attempts": 0,
        "receipts": 0,
        "committed": 0,
        "failed": 0,
        "unknown": 0,
        "known_input_tokens": 0,
        "known_output_tokens": 0,
        "unknown_usage_attempts": 0,
        "reserved_requests": 0,
        "by_mode": {},
    }
    for row in rows:
        body["planned_tasks"] += row["planned_tasks"]
        body["attempts"] += row["attempts"]
        body["receipts"] += row["receipts"]
        body["committed"] += row["committed"]
        body["failed"] += row["failed"]
        body["unknown"] += row["unknown"]
        body["known_input_tokens"] += row["known_input_tokens"] or 0
        body["known_output_tokens"] += row["known_output_tokens"] or 0
        body["unknown_usage_attempts"] += row["unknown_usage_attempts"] or 0
        body["reserved_requests"] += row["reserved_requests"] or 0
        mode = body["by_mode"].setdefault(
            row["mode"], {"ledgers": 0, "attempts": 0, "known_input_tokens": 0,
                          "known_output_tokens": 0, "unknown_usage_attempts": 0}
        )
        mode["ledgers"] += 1
        mode["attempts"] += row["attempts"]
        mode["known_input_tokens"] += row["known_input_tokens"] or 0
        mode["known_output_tokens"] += row["known_output_tokens"] or 0
        mode["unknown_usage_attempts"] += row["unknown_usage_attempts"] or 0
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    body = reconcile(args.root)
    if args.json:
        print(json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2))
        return
    for label, entry in body["namespaces"].items():
        totals = entry["totals"]
        print(
            f"{label:12s} ledgers={totals['ledgers']:3d} attempts={totals['attempts']:5d} "
            f"receipts={totals['receipts']:5d} unknown={totals['unknown']:3d} "
            f"known_input={totals['known_input_tokens']} "
            f"known_output={totals['known_output_tokens']} "
            f"unknown_usage={totals['unknown_usage_attempts']}"
        )
    grand = body["grand_total"]
    print(
        f"{'TOTAL':12s} ledgers={grand['ledgers']:3d} attempts={grand['attempts']:5d} "
        f"receipts={grand['receipts']:5d} unknown={grand['unknown']:3d}"
    )


if __name__ == "__main__":
    main()
