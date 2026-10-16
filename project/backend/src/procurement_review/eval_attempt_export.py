"""Byte-compatible atomic export of the evaluation ledger. No model calls."""

import json
from pathlib import Path

from .eval_attempt_ledger import LEDGER_SCHEMA_VERSION, _atomic_write


def export_ledger(ledger, output: Path) -> Path:
    connection = ledger._connect()
    try:
        attempts = []
        for row in connection.execute("SELECT * FROM attempts ORDER BY attempt_id"):
            attempts.append(
                {
                    "attempt_id": row["attempt_id"],
                    "task_key": row["task_key"],
                    "attempt_no": row["attempt_no"],
                    "request_sha256": row["request_sha256"],
                    "state": row["state"],
                    "reserved_input_tokens": row["reserved_input_tokens"],
                    "reserved_output_tokens": row["reserved_output_tokens"],
                    "receipt_sha256": row["receipt_sha256"],
                    "projection_sha256": row["projection_sha256"],
                    "receipt": json.loads(row["receipt_json"])
                    if row["receipt_json"]
                    else None,
                    "projection": json.loads(row["projection_json"])
                    if row["projection_json"]
                    else None,
                    "created_at": row["created_at"],
                    "sent_at": row["sent_at"],
                    "terminal_at": row["terminal_at"],
                }
            )
        document = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "manifest": ledger.manifest().model_dump(mode="json"),
            "attempts": attempts,
        }
    finally:
        connection.close()
    body = (
        json.dumps(
            document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(output, body)
    return output
