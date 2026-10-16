"""Quality-first DEV selection, performed once before formal evaluation."""

import json
from pathlib import Path

from .eval_attempt_ledger import LedgerError
from .eval_f_context import stable_write
from .eval_f_registration import LIVE_ROOT, registration
from .eval_f_lineage import diagnostic_directory
from .gold_v3_contracts import payload_sha256


def select(path, *, root=LIVE_ROOT):
    registered = json.loads(Path(path).read_text())
    if registered != registration(
        registered.get("diagnostic_predecessor"), registered["versions"][0]
    ):
        raise LedgerError("selection registration changed")
    digest = payload_sha256(registered)
    directory = Path(root) / digest
    selection_path = directory / "selection.json"
    if selection_path.exists():
        return json.loads(selection_path.read_text())
    choices = []
    for arm in ("baseline", "compact", "reasoning", "concurrency"):
        arm_directory = diagnostic_directory(root, registered, arm)
        result_path = arm_directory / "result.json"
        if not result_path.exists():
            raise LedgerError("registered ablations are incomplete")
        result = json.loads(result_path.read_text())
        if result["diagnostic_passed"]:
            import sqlite3

            with sqlite3.connect(arm_directory / "ledger.sqlite3") as conn:
                receipts = [
                    json.loads(row[0])
                    for row in conn.execute(
                        "SELECT receipt_json FROM attempts WHERE receipt_json IS NOT NULL"
                    )
                ]
            if any(r["usage_unknown"] for r in receipts):
                cost = None
            else:
                cost = sum(r["input_tokens"] + r["output_tokens"] for r in receipts)
            choices.append((result["combined"]["runner_p95_ms"], cost, arm))
    if not choices:
        raise LedgerError("no quality-qualified DEV candidate")
    latency, cost, arm = min(
        choices, key=lambda v: (v[0], float("inf") if v[1] is None else v[1], v[2])
    )
    selected = dict(
        registration_sha256=digest,
        arm=arm,
        runner_p95_ms=latency,
        known_token_sum=cost,
        criterion="quality first; then all-sample runner p95; known tokens break ties",
        validation_group=registered["versions"][0] + "-validation",
        formal_quality_pending=True,
        diagnostic_predecessor=registered.get("diagnostic_predecessor"),
    )
    stable_write(selection_path, selected)
    return selected
