"""The complete Case boundary must reject results after cancellation too."""

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest

from procurement_review.eval_f_context import stable_write
from procurement_review.eval_f_registration import prepare
from procurement_review.eval_f_run import execute, FAuthority
from v1_07f_support import response_for


@pytest.mark.parametrize("event", ["cancelled", "generation", "deadline"])
def test_cancellation_after_committed_policy_cannot_publish_case_evidence(
    tmp_path, event, monkeypatch
):
    root = tmp_path / ".local/live"
    path = tmp_path / ".local/registration.json"
    prepare(path)
    sent = []
    expired = []
    original_init = FAuthority.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.now = (
            lambda: datetime(2099, 1, 1, tzinfo=UTC) if expired else datetime.now(UTC)
        )

    monkeypatch.setattr(FAuthority, "__init__", initialize)

    def transport(request):
        sent.append(request)
        body = json.loads(json.loads(request.content)["messages"][1]["content"])
        if body["worker"] == "contract":
            execution = next(root.glob("*/minimum/execution.json"))
            state = json.loads(execution.read_text())
            if event == "cancelled":
                state["cancelled"] = True
            elif event == "generation":
                state["generation"] += 1
            else:
                expired.append(True)
            stable_write(execution, state)
        return response_for(request)

    report = asyncio.run(
        execute(
            path,
            "minimum",
            transport_override=httpx.MockTransport(transport),
            offline_root=root,
        )
    )
    assert len(sent) == 2
    assert report["combined"]["counts"].get("obligation_correct", 0) == 0
    for p in root.glob("*/minimum/V107-*.json"):
        case = json.loads(p.read_text())
        assert case["decision"]["spans"] == []
        assert case["decision"]["business"]["observations"] == []
        assert case["decision"]["business"]["pipeline_failures"] == [
            "execution_generation_changed" if event == "generation" else event
        ]
    # Previously committed branch evidence remains in its original ledger.
    export = json.loads(next(root.glob("*/minimum/ledger-export.json")).read_text())
    assert export["attempts"][0]["projection"]["evidence_output"] is not None
