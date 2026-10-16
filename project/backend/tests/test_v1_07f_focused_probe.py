"""Failed repair probes stop new work and preserve their actual receipts."""

import asyncio
import json

import httpx
import pytest

from procurement_review.eval_attempt_ledger import LedgerError
from procurement_review.eval_f_budget import FStageBudget
from procurement_review.eval_f_context import stable_write
from procurement_review.eval_f_repair import preregistration, run
from v1_07f_support import response_for


def test_failed_pair_stops_new_cases_and_reopen_does_not_send(tmp_path):
    async def scenario():
        path = tmp_path / "registration.json"
        stable_write(path, preregistration())
        root = tmp_path / ".local/live"
        calls = []
        both_started = asyncio.Event()

        async def handler(req):
            calls.append(req)
            if len(calls) == 2:
                both_started.set()
            await both_started.wait()
            return response_for(req, body={"bad": True})

        transport = httpx.MockTransport(handler)
        result = await run(path, transport_override=transport, offline_root=root)
        assert result["failed"] and not result["stage_acceptance"]
        assert len(calls) == 2
        budget = FStageBudget(root).summary()
        assert budget["reserved_attempts"] == budget["receipts"] == 2
        assert budget["unreturned"] == 0
        assert (
            await run(path, transport_override=transport, offline_root=root) == result
        )
        assert len(calls) == 2
        export = json.loads(
            next(root.glob("*/focused-dev/ledger-export.json")).read_text()
        )
        assert all(a["attempt_no"] == 1 and a["receipt"] for a in export["attempts"])

    asyncio.run(scenario())


def test_focused_real_run_cannot_change_inventory_root(tmp_path):
    with pytest.raises(LedgerError, match="fixed live root"):
        asyncio.run(run(tmp_path / "missing", offline_root=tmp_path))


def test_focused_probe_rejects_configuration_change_before_transport(tmp_path):
    path = tmp_path / "registration.json"
    registered = preregistration()
    registered["configuration"]["thinking"] = None
    stable_write(path, registered)
    with pytest.raises(LedgerError, match="inputs changed"):
        asyncio.run(
            run(
                path,
                transport_override=httpx.MockTransport(lambda req: pytest.fail("send")),
                offline_root=tmp_path / ".local/live",
            )
        )
