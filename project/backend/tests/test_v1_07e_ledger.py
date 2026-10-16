import asyncio
from datetime import UTC, datetime, timedelta
import json

import httpx
import pytest
from pydantic import SecretStr

from procurement_review.business_v3_security import ProtectedOutputV3
from procurement_review.chat_transport_v3 import ChatTransportV3, EndpointConfigV3
from procurement_review.eval_attempt_ledger import AttemptLedger, LedgerError
from procurement_review.model_attempt_v2 import (
    BatchAuthority,
    BudgetProof,
    BranchAttemptDriver,
    configuration_hash,
)
from v1_07c_support import answer, branch, response


def driver_for(tmp_path, handler, protection):
    now = datetime.now(UTC)
    config = EndpointConfigV3(
        endpoint="https://example.invalid/chat",
        requested_model="test-model",
        resolved_model="test-model-1",
        mode="json_schema",
        protection_sha256=protection.fingerprint,
    )
    transport = ChatTransportV3(
        config,
        SecretStr("local-only-credential"),
        protection=protection,
        transport=httpx.MockTransport(handler),
    )
    authority = BatchAuthority(
        authorization_id="offline-e-test",
        scope_ids=frozenset({"V107-SCOPE-A"}),
        expires_at=now + timedelta(minutes=5),
        proof=BudgetProof(
            evidence_sha256="a" * 64,
            cumulative_request_upper=0,
            cumulative_input_upper=0,
            cumulative_output_upper=0,
            historical_unknown_input_reserve=0,
            historical_unknown_output_reserve=0,
            allocated_requests=2,
            allocated_input=1000,
            allocated_output=1000,
        ),
    )
    ledger = AttemptLedger.create(
        tmp_path / ".local" / "e.sqlite3",
        run_id="v1-07-e-test",
        gold_manifest_sha256="b" * 64,
        configuration_sha256=configuration_hash(transport, authority),
        max_requests=2,
        max_input_tokens=1000,
        max_output_tokens=1000,
        # This is a zero-network compatibility test of the existing C driver,
        # not an E live batch or authorization to create a new stage dispatcher.
        stage_id="V1-07C",
    )
    return BranchAttemptDriver(ledger, transport, authority), now + timedelta(
        seconds=90
    )


@pytest.mark.parametrize("sensitive", [False, True])
def test_new_guard_uses_atomic_ledger_and_never_corrects_security(tmp_path, sensitive):
    calls = []
    body = answer()
    body["findings"][0]["statement"] = "拒绝执行外部写入工具。" + (
        "CANARY-E-PRIVATE" if sensitive else ""
    )

    def handler(request):
        calls.append(request)
        return response(body, usage=not sensitive)

    protection = ProtectedOutputV3(canaries=(SecretStr("CANARY-E-PRIVATE"),))
    driver, deadline = driver_for(tmp_path, handler, protection)
    args = dict(deadline=deadline, reserve_input=100, reserve_output=200)
    result = asyncio.run(driver.run("e:one", branch(), **args))
    assert len(calls) == 1 and result.model_request_count == 1
    assert (
        result.pipeline_failure is sensitive
        and result.first_pass_valid is not sensitive
    )
    exported = json.loads(
        driver.ledger.export(tmp_path / ".local" / "export.json").read_text()
    )
    assert "CANARY-E-PRIVATE" not in json.dumps(exported)
    receipt = exported["attempts"][0]["receipt"]
    assert receipt["error_code"] == ("security_boundary" if sensitive else None)
    assert receipt["usage_unknown"] is sensitive
    reopened = BranchAttemptDriver(
        AttemptLedger(driver.ledger.path), driver.transport, driver.authority
    )
    assert (
        asyncio.run(reopened.run("e:one", branch(), **args)) == result
        and len(calls) == 1
    )


def test_restoring_with_weakened_protection_registry_is_rejected_before_send(tmp_path):
    calls = []
    driver, deadline = driver_for(
        tmp_path,
        lambda req: (calls.append(req) or response()),
        ProtectedOutputV3(canaries=(SecretStr("CANARY-E-PRIVATE"),)),
    )
    driver.transport.protection = ProtectedOutputV3()
    with pytest.raises(LedgerError, match="configuration mismatch"):
        asyncio.run(
            driver.run(
                "e:one",
                branch(),
                deadline=deadline,
                reserve_input=100,
                reserve_output=200,
            )
        )
    assert not calls and driver.ledger.budget_snapshot().reserved_requests == 0


@pytest.mark.parametrize(
    "updates",
    [
        {"stage_id": "V1-07E"},
        {"run_id": "bad-run"},
        {"max_requests": 0},
        {"configuration_sha256": "bad-hash"},
    ],
)
def test_invalid_creation_contract_never_writes_a_ledger(tmp_path, updates):
    from pydantic import ValidationError

    path = tmp_path / ".local" / "invalid.sqlite3"
    args = dict(
        run_id="v1-07-e-test",
        gold_manifest_sha256="b" * 64,
        configuration_sha256="a" * 64,
        max_requests=2,
        max_input_tokens=1000,
        max_output_tokens=1000,
        stage_id="V1-07C",
    )
    with pytest.raises(ValidationError):
        AttemptLedger.create(path, **(args | updates))
    assert not path.exists()
