"""A failed fixed candidate stops before spending the rest of its planned calls."""

import asyncio
import json
import sqlite3

import httpx
import pytest

from procurement_review.eval_attempt_ledger import LedgerError
from procurement_review.eval_f_context import stable_write
from procurement_review.eval_f_registration import fixed_registration
from procurement_review.eval_f_run import execute, replay
from procurement_review.eval_f_stop import quality_stop
from v1_07f_support import response_for


def test_q01_uses_each_repeat_and_exact_planned_boundary(tmp_path):
    candidate = dict(case_ids=[f"case-{i}" for i in range(60)], repeat_seeds=[301, 302])
    with sqlite3.connect(tmp_path / "ledger.sqlite3") as db:
        db.execute(
            "CREATE TABLE attempts (task_key TEXT, attempt_no INT, receipt_json TEXT)"
        )
        db.executemany(
            "INSERT INTO attempts VALUES (?,1,?)",
            [
                (f"case-0.{seed}.policy", json.dumps(dict(outcome="failed")))
                for seed in (301, 302)
            ],
        )
        # One failure in each 120-request repeat still allows 119/120 (>=99%).
    assert quality_stop(tmp_path, candidate) is None
    with sqlite3.connect(tmp_path / "ledger.sqlite3") as db:
        db.execute(
            "INSERT INTO attempts VALUES (?,1,?)",
            ("case-0.302.contract", json.dumps(dict(outcome="failed"))),
        )
    stopped = quality_stop(tmp_path, candidate)
    assert stopped["repeat_seed"] == 302
    assert stopped["maximum_possible_valid"] == 118
    assert stopped["required_valid"] == 119
    assert stopped["planned_first_attempts"] == 120
    assert not stopped["stage_acceptance"]
    # Restart obeys the durable stop even if there are no newly read receipts.
    (tmp_path / "ledger.sqlite3").rename(tmp_path / "old.sqlite3")
    assert quality_stop(tmp_path, candidate) == stopped


@pytest.mark.parametrize("split", ["validation", "dev"])
def test_fixed_candidate_stops_and_never_restarts_missing_cases(tmp_path, split):
    async def scenario():
        root = tmp_path / ".local/live"
        path = tmp_path / "registration.json"
        registered = fixed_registration("gold-set.v3.2", split=split)
        candidate = split + "-concurrency"
        planned_cases = 120 if split == "validation" else 240
        stable_write(path, registered)
        calls = []
        first_pair = asyncio.Event()

        async def handler(req):
            calls.append(req)
            if len(calls) == 2:
                first_pair.set()
            await first_pair.wait()
            return response_for(req, body={"invalid": True})

        result = await execute(
            path,
            candidate,
            transport_override=httpx.MockTransport(handler),
            offline_root=root,
        )
        assert not result["diagnostic_passed"]
        # DEV permits two failed first sends, validation one; at most two
        # already-active Cases can finish after the threshold becomes unreachable.
        assert 2 <= len(calls) <= (8 if split == "validation" else 16)
        directory = next(root.glob("*/" + candidate))
        assert (directory / "quality-stop.json").exists()
        completed = len(list(directory.glob("V107-*.json")))
        assert completed == 2 if split == "validation" else 2 <= completed <= 4
        assert result["quality_stop"]["planned_first_attempts"] == planned_cases
        assert result["combined"]["counts"]["cases"] == planned_cases
        assert (
            result["combined"]["counts"]["missing_cases"] == planned_cases - completed
        )
        assert not result["stage_acceptance"]
        count = len(calls)
        await execute(
            path,
            candidate,
            transport_override=httpx.MockTransport(handler),
            offline_root=root,
        )
        assert len(calls) == count
        assert replay(directory)["combined"] == result["combined"]
        # A new implementation/config registration cannot consume this group again.
        from procurement_review.eval_f_validation import claim_validation

        if split == "validation":
            changed = dict(registered, revision=999)
            with pytest.raises(LedgerError, match="already consumed"):
                claim_validation(root, changed, candidate)
        else:
            assert not (root / "consumed-validation").exists()
            assert list(registered["candidates"]) == ["dev-concurrency"]

    asyncio.run(scenario())


def test_fixed_registration_has_no_dev_or_alternative_runs():
    registered = fixed_registration("gold-set.v3.2")
    assert list(registered["candidates"]) == ["validation-concurrency"]
    c = registered["candidates"]["validation-concurrency"]
    assert c["configuration"]["prompt_style"] == "baseline"
    assert c["configuration"]["thinking"] == "disabled"
    assert c["concurrency"] == 2 and c["max_attempts"] == 2
    assert c["max_requests"] == 480
    assert registered["reserve"]["dev_requests"] == 0
