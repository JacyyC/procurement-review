import json
from unittest.mock import patch

import pytest

from procurement_review.model_probe_v2 import registration, dev_requests, replay
from v1_07c_support import setup_driver, response
from test_v1_07c_attempts import run, export


def test_registration_is_zero_network_dev_only_and_remains_blocked():
    with patch(
        "httpx.AsyncClient.post", side_effect=AssertionError("network forbidden")
    ):
        plan = registration()
    assert plan["physical_request_limit"] == 234 <= plan["stage_ceiling"]
    assert plan["execution_status"] == "blocked_budget_unproven"
    assert plan["historical_unknown_output_reserve"] is None
    assert plan["new_attempt_unknown_reserve"] is None
    assert {t["case_id"] for c in plan["candidates"] for t in c["tasks"]} == set(
        plan["dev_case_ids"]
    )
    assert all(x.startswith("V107-D-") for x in plan["dev_case_ids"])
    assert len(dev_requests()) == 18
    for request in dev_requests():
        assert all(source.worker == request.worker for source in request.sources)


def test_source_bytes_preserved_including_trailing_newline():
    import hashlib

    for request in dev_requests():
        for source in request.sources:
            assert (
                hashlib.sha256(source.content.encode()).hexdigest()
                == source.identity.content_sha256
            )


def test_branch_replay_retains_failure_denominator_and_unknown_usage(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        return response({"bad": True}, usage=False) if len(calls) == 1 else response()

    driver, deadline = setup_driver(tmp_path, handler)
    result = run(driver, deadline)
    assert result.corrected_valid
    path = driver.ledger.export(tmp_path / ".local" / "replay.json")
    one = replay(path)
    assert one == replay(path)
    assert (
        one["first_attempts"],
        one["first_valid"],
        one["final_valid"],
        one["corrected_valid"],
    ) == (1, 0, 1, 1)
    assert one["unknown_usage"] == 1 and one["known_output"] == 41
    assert one["quality_acceptance"] is False


@pytest.mark.parametrize(
    "corruption", ["receipt", "projection", "missing_projection", "duplicate", "usage"]
)
def test_replay_rejects_corrupt_commit_or_usage(corruption, tmp_path):
    driver, deadline = setup_driver(tmp_path, lambda req: response())
    run(driver, deadline)
    document = export(driver, tmp_path)
    if corruption == "receipt":
        document["attempts"][0]["receipt"]["request_sha256"] = "f" * 64
    if corruption == "projection":
        document["attempts"][0]["projection"]["first_pass_valid"] = False
    if corruption == "missing_projection":
        document["attempts"][0]["projection"] = None
    if corruption == "duplicate":
        document["attempts"].append(document["attempts"][0])
    if corruption == "usage":
        document["manifest"]["budget"]["known_output_tokens"] = 0
    path = tmp_path / ".local" / "corrupt.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        replay(path)
