"""Conservative early stop from settled first-attempt failures, without labels.

In-flight Cases may settle. Unstarted Cases remain missing planned observations,
never fabricated model failures. A durable stop prevents restart from resuming.
"""

import json
import sqlite3

from .eval_f_context import stable_write


def quality_stop(directory, candidate):
    path = directory / "quality-stop.json"
    if path.exists():
        return json.loads(path.read_text())
    ledger = directory / "ledger.sqlite3"
    if not ledger.exists():
        return None
    planned = {
        f"{cid}.{seed}.{worker}": seed
        for cid in candidate["case_ids"]
        for seed in candidate["repeat_seeds"]
        for worker in ("policy", "contract")
    }
    failures = {seed: [] for seed in candidate["repeat_seeds"]}
    with sqlite3.connect(ledger) as connection:
        for key, receipt_json in connection.execute(
            "SELECT task_key, receipt_json FROM attempts "
            "WHERE attempt_no=1 AND receipt_json IS NOT NULL"
        ):
            if key not in planned:
                continue
            receipt = json.loads(receipt_json)
            if receipt["outcome"] != "succeeded":
                failures[planned[key]].append(key)
    denominator = len(candidate["case_ids"]) * 2
    # Exact integer Q-01 >= 99/100, independently for each repeat.
    allowed = denominator // 100
    for seed, keys in failures.items():
        if len(keys) > allowed:
            record = dict(
                contract="v1-07f-quality-stop.v1",
                reason="first_pass_threshold_unreachable",
                repeat_seed=seed,
                planned_first_attempts=denominator,
                required_valid=denominator - allowed,
                settled_failed_first_attempts=sorted(keys),
                maximum_possible_valid=denominator - len(keys),
                action="stop_unstarted_cases_settle_active_cases",
                stage_acceptance=False,
            )
            stable_write(path, record)
            return record
    return None
