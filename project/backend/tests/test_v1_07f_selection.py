"""Selection unit tests; qualified report inputs are fixtures, never live evidence."""

import json
import sqlite3

import pytest

from procurement_review.eval_attempt_ledger import LedgerError
from procurement_review.eval_f_context import stable_write
from procurement_review.eval_f_registration import prepare, registration
from procurement_review.eval_f_selection import select
from procurement_review.gold_v3_contracts import payload_sha256


def candidates(root, choices):
    path = root / "registration.json"
    prepare(path)
    directory = root / payload_sha256(registration())
    for arm, passed, milliseconds, tokens in choices:
        folder = directory / arm
        folder.mkdir(parents=True)
        stable_write(
            folder / "result.json",
            {
                "diagnostic_passed": passed,
                "combined": {"runner_p95_ms": milliseconds},
            },
        )
        # Minimal receipt reader fixture for selection, not a substitute B ledger.
        with sqlite3.connect(folder / "ledger.sqlite3") as conn:
            conn.execute("CREATE TABLE attempts (receipt_json TEXT)")
            conn.execute(
                "INSERT INTO attempts VALUES (?)",
                (
                    json.dumps(
                        {
                            "usage_unknown": tokens is None,
                            "input_tokens": tokens,
                            "output_tokens": 0 if tokens is not None else None,
                        }
                    ),
                ),
            )
    return path, directory


def test_selection_prioritizes_quality_then_latency_then_observed_cost(tmp_path):
    path, directory = candidates(
        tmp_path,
        [
            ("baseline", True, 2000, 200),
            ("compact", False, 1, 1),
            ("reasoning", True, 1000, None),
            ("concurrency", True, 1000, 150),
        ],
    )
    result = select(path, root=tmp_path)
    assert result["arm"] == "concurrency"
    assert result["known_token_sum"] == 150
    assert result["formal_quality_pending"]
    # Reports arriving later do not silently replace the locked choice.
    stable_write(
        directory / "baseline/result.json",
        {"diagnostic_passed": True, "combined": {"runner_p95_ms": 1}},
    )
    assert select(path, root=tmp_path) == result


def test_selection_retains_unknown_usage_when_it_is_the_faster_qualified_arm(tmp_path):
    path, _ = candidates(
        tmp_path,
        [
            ("baseline", True, 2000, 200),
            ("compact", True, 1500, 10),
            ("reasoning", True, 1000, None),
            ("concurrency", False, 1, 1),
        ],
    )
    result = select(path, root=tmp_path)
    assert result["arm"] == "reasoning" and result["known_token_sum"] is None


def test_selection_rejects_all_failed_or_incomplete_ablations(tmp_path):
    path, directory = candidates(
        tmp_path,
        [
            (arm, False, 1000, 100)
            for arm in ("baseline", "compact", "reasoning", "concurrency")
        ],
    )
    with pytest.raises(LedgerError, match="no quality-qualified"):
        select(path, root=tmp_path)
    (directory / "concurrency/result.json").rename(
        directory / "concurrency/incomplete.json"
    )
    with pytest.raises(LedgerError, match="incomplete"):
        select(path, root=tmp_path)
    assert not (directory / "selection.json").exists()


def test_selection_refuses_configuration_reinterpretation(tmp_path):
    path, _ = candidates(tmp_path, [])
    value = json.loads(path.read_text())
    value["candidates"]["baseline"]["concurrency"] = 9
    stable_write(path, value)
    with pytest.raises(LedgerError, match="registration changed"):
        select(path, root=tmp_path)
