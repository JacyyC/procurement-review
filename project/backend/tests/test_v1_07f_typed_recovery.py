"""An interruption after a typed error must preserve the one correction slot."""

import asyncio
from datetime import UTC, datetime, timedelta
import json

import pytest

from procurement_review.eval_f_transport import PreciseAttemptDriver
from v1_07f_support import body_for_payload, response_for, setup


def test_restart_after_typed_failure_uses_remaining_correction_once(
    tmp_path, monkeypatch
):
    sent = []

    def handler(req):
        payload = json.loads(req.content)
        sent.append(payload)
        body = body_for_payload(payload)
        if len(sent) == 1:
            value = json.loads(body["findings"][0]["normalized_value"])
            value["applies"] = "true"
            body["findings"][0]["normalized_value"] = json.dumps(value)
        return response_for(req, body=body)

    driver, request, _ = setup(tmp_path, handler)
    ledger = driver.ledger
    deadline = datetime.now(UTC) + timedelta(seconds=60)
    original = ledger.commit_failure

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("process stopped after durable failure")

    monkeypatch.setattr(ledger, "commit_failure", interrupted)
    with pytest.raises(RuntimeError, match="process stopped"):
        asyncio.run(
            driver.run(
                "typed-recovery",
                request,
                deadline=deadline,
                reserve_input=5000,
                reserve_output=5000,
            )
        )
    monkeypatch.setattr(ledger, "commit_failure", original)
    # Recreate the driver as a new process does; no in-memory correction survives.
    resumed = PreciseAttemptDriver(ledger, driver.transport, driver.authority)
    result = asyncio.run(
        resumed.run(
            "typed-recovery",
            request,
            deadline=deadline,
            reserve_input=5000,
            reserve_output=5000,
        )
    )
    assert result.corrected_valid and not result.first_pass_valid
    assert result.model_request_count == 2 and len(sent) == 2
    assert "typed_o_01_applies_bool_type" in sent[1]["messages"][-1]["content"]
    assert (
        asyncio.run(
            resumed.run(
                "typed-recovery",
                request,
                deadline=deadline,
                reserve_input=5000,
                reserve_output=5000,
            )
        )
        == result
    )
    assert len(sent) == 2
