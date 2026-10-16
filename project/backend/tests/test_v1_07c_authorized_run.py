import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from procurement_review.eval_attempt_ledger import (
    AttemptLedger,
    BudgetExceeded,
    LedgerError,
)
from procurement_review.model_attempt_v2 import BudgetProof
from procurement_review.model_probe_v2 import registration
from procurement_review.model_probe_run import (
    execute,
    require_stage_capacity,
    stage_observations,
)


def proof(**changes):
    fields = dict(
        evidence_sha256="a" * 64,
        cumulative_request_upper=7639,
        cumulative_input_upper=None,
        cumulative_output_upper=None,
        historical_unknown_input_reserve=None,
        historical_unknown_output_reserve=None,
        allocated_requests=240,
        allocated_input=None,
        allocated_output=None,
        limit_policy="user_no_balance_limit",
        authorization_basis="user-explicit-no-balance-limit",
    )
    fields.update(changes)
    return BudgetProof(**fields)


def test_unlimited_is_explicit_and_does_not_fill_unknown_usage():
    unlimited = proof()
    unlimited.require_capacity()
    assert unlimited.cumulative_output_upper is None
    assert unlimited.historical_unknown_output_reserve is None
    with pytest.raises(ValidationError):
        proof(authorization_basis=None)
    with pytest.raises(ValidationError):
        proof(allocated_requests=241)
    with pytest.raises(LedgerError):
        proof(
            limit_policy="cumulative_brakes", allocated_input=100, allocated_output=100
        ).require_capacity()


def test_unlimited_token_ledger_still_enforces_requests_and_recovers(tmp_path):
    path = tmp_path / ".local/ledger.sqlite3"
    ledger = AttemptLedger.create(
        path,
        run_id="v1-07-unlimited-test",
        gold_manifest_sha256="a" * 64,
        configuration_sha256="b" * 64,
        max_requests=1,
        max_input_tokens=None,
        max_output_tokens=None,
        stage_id="V1-07C",
    )
    ledger.register_task("one", "c" * 64)
    first = ledger.reserve_attempt(
        "one",
        request_sha256="d" * 64,
        reserve_input_tokens=300_000_000,
        reserve_output_tokens=30_000_000,
    )
    ledger.mark_sent(first.attempt_id)
    reopened = AttemptLedger(path)
    reopened.recover()
    assert reopened.budget_snapshot().unknown_usage_attempts == 1
    assert reopened.budget_snapshot().max_output_tokens is None
    assert reopened.manifest().budget.unknown_usage_attempts == 1
    # Recovery must also remain exportable under the already-frozen RunId schema.
    reopened.export(tmp_path / ".local/unlimited-export.json")
    ledger.register_task("two", "e" * 64)
    with pytest.raises(BudgetExceeded):
        ledger.reserve_attempt(
            "two",
            request_sha256="f" * 64,
            reserve_input_tokens=1,
            reserve_output_tokens=1,
        )


def mock_handler(calls):
    def handler(req):
        calls.append(req)
        visible = json.loads(json.loads(req.content)["messages"][1]["content"])
        findings = [
            dict(
                obligation_id=c["proposition"]["check_id"],
                status="supported",
                assertion="compliant",
                risk_level="none",
                normalized_value="明确约定",
                statement="仅用于传输测试",
                recommended_action="人工复核",
                evidence_refs=[visible["sources"][0]["ref"]],
                missing_information=[],
            )
            for c in visible["checks"]
        ]
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-flash-ga-260731",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "schema_version": "finding-envelope.v2",
                                    "findings": findings,
                                }
                            )
                        },
                    }
                ],
                "usage": None,
            },
        )

    return handler


def test_registered_real_runner_mock_replays_without_new_requests(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(registration(no_balance_limit_authorized=True)))
    root = tmp_path / ".local/live"
    calls = []
    transport = httpx.MockTransport(mock_handler(calls))
    result = asyncio.run(
        execute(path, "minimum:json_schema", root=root, transport_override=transport)
    )
    assert result["first_valid"] == 3 and result["unknown_usage"] == 3
    assert result["quality_acceptance"] is False
    asyncio.run(
        execute(path, "minimum:json_schema", root=root, transport_override=transport)
    )
    assert len(calls) == 3 and stage_observations(root)[0] == 3
    with pytest.raises(LedgerError):
        asyncio.run(
            execute(path, "screen:text", root=root, transport_override=transport)
        )
    require_stage_capacity(root, additional=237)
    with pytest.raises(LedgerError):
        require_stage_capacity(root, additional=238)
    path.write_text(json.dumps(registration()))
    with pytest.raises(LedgerError):
        asyncio.run(
            execute(path, "minimum:text", root=root, transport_override=transport)
        )
    assert len(calls) == 3


def test_stage_guard_runs_before_a_second_physical_request(tmp_path):
    from v1_07c_support import setup_driver
    from test_v1_07c_attempts import run

    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("remote usage remains unknown")

    driver, deadline = setup_driver(tmp_path, handler)
    admitted = []

    def limit():
        admitted.append(True)
        if len(admitted) > 1:
            raise LedgerError("stage limit reached by another branch")

    driver.authority.before_attempt = limit
    with pytest.raises(LedgerError, match="stage limit"):
        run(driver, deadline)
    assert len(calls) == 1
    assert driver.ledger.budget_snapshot().reserved_requests == 1
    assert driver.ledger.budget_snapshot().unknown_usage_attempts == 1


def test_stage_stop_survives_late_security_result_and_missing_stop_file(tmp_path):
    from v1_07c_support import answer, response, setup_driver
    from test_v1_07c_attempts import run

    def handler(req):
        driver.authority.cancel()
        bad = answer()
        bad["findings"][0]["normalized_value"] = "synthetic_secret_to_external_sink"
        return response(bad)

    driver, deadline = setup_driver(tmp_path, handler)
    assert run(driver, deadline).pipeline_failure
    with pytest.raises(LedgerError, match="safety"):
        require_stage_capacity(tmp_path / ".local", additional=1)
