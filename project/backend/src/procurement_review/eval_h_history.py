"""Read only consumed input snapshots from a pinned, immutable SQLite archive."""

import json
import sqlite3
from pathlib import Path

from .gold_v3_contracts import payload_sha256


def load_source(path: Path, seen: dict):
    if path.suffix != ".sqlite3":
        from .eval_h_storage import read

        return read(path)
    ids = tuple(seen["case_ids"])
    fingerprints = tuple(seen["source_fingerprints"])
    if not ids or not fingerprints:
        raise ValueError("h_consumed_inventory_selection_empty")
    query = (
        "SELECT a.case_id, i.branch FROM attempt AS a "
        "JOIN input_snapshot AS i USING(run_id,worker,ordinal) "
        "WHERE a.case_id IN (" + ",".join("?" for _ in ids) + ") "
        "AND a.source_fingerprints IN (" + ",".join("?" for _ in fingerprints) + ")"
    )
    cases = {}
    with sqlite3.connect(
        path.absolute().as_uri() + "?mode=ro&immutable=1", uri=True
    ) as connection:
        for case_id, branch in connection.execute(query, (*ids, *fingerprints)):
            body = json.loads(branch)
            if type(body) is not dict or body.get("case_id") != case_id:
                raise ValueError("h_consumed_inventory_case_binding")
            cases[payload_sha256(body)] = body
    return {"consumed_input_snapshots": list(cases.values())}
