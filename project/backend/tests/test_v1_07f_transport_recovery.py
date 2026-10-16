"""A transport stop allows only a reviewed single probe; failures stay stopped."""

import pytest

from procurement_review.eval_attempt_ledger import LedgerError
from procurement_review.eval_f_budget import FStageBudget
from procurement_review.eval_f_context import stable_write
from procurement_review.gold_v3_contracts import payload_sha256


def stopped(tmp_path, monkeypatch):
    budget = FStageBudget(tmp_path / ".local")
    rows = [
        {"error_code": "timeout", "outcome": "failed", "task_key": f"case.{n}"}
        for n in range(12)
    ]
    state = {"count": 12, "receipts": rows}
    monkeypatch.setattr(
        budget, "observations", lambda: (state["count"], state["receipts"])
    )
    config = {"thinking": "disabled", "resolved_model": "fixed"}
    permit = {
        "contract": "v1-07f-transport-recovery.v1",
        "status": "probe_pending",
        "max_requests": 1,
        "at_request_count": 12,
        "last_twelve_sha256": payload_sha256(rows),
        "configuration_sha256": payload_sha256(config),
        "review_evidence_sha256": "a" * 64,
    }
    return budget, state, config, permit


def test_stop_does_not_allow_configuration_change_without_review(tmp_path, monkeypatch):
    b, _, config, permit = stopped(tmp_path, monkeypatch)
    with pytest.raises(LedgerError, match="twelve consecutive"):
        b.require(configuration=config)
    stable_write(b.root / "transport-recovery.json", permit)
    with pytest.raises(LedgerError, match="does not apply"):
        b.require(configuration={"thinking": "changed"})
    b.require(configuration=config)


@pytest.mark.parametrize("error", ["timeout", "invalid_schema", None])
def test_probe_is_one_shot_even_if_failure_breaks_transport_sequence(
    tmp_path, monkeypatch, error
):
    b, state, config, permit = stopped(tmp_path, monkeypatch)
    stable_write(b.root / "transport-recovery.json", permit)
    b.require(configuration=config)
    state["count"] += 1
    state["receipts"].append(
        {"error_code": error, "outcome": "succeeded" if error is None else "failed"}
    )
    with pytest.raises(LedgerError, match="probe consumed"):
        b.require(configuration=config)


def test_review_release_requires_real_success_and_preserves_old_errors(
    tmp_path, monkeypatch
):
    b, state, config, permit = stopped(tmp_path, monkeypatch)
    permit.update(status="released", success_receipt_sha256="b" * 64)
    stable_write(b.root / "transport-recovery.json", permit)
    with pytest.raises(LedgerError, match="no actual successful"):
        b.require(configuration=config)
    receipt = {"error_code": None, "outcome": "succeeded", "task_key": "recovery"}
    state["count"] += 1
    state["receipts"].append(receipt)
    permit["success_receipt_sha256"] = payload_sha256(receipt)
    stable_write(b.root / "transport-recovery.json", permit)
    b.require(configuration=config)
    assert state["count"] == 13 and len(state["receipts"]) == 13
    assert all(r["error_code"] == "timeout" for r in state["receipts"][:12])


def test_probe_cannot_bypass_authentication_or_durable_stop(tmp_path, monkeypatch):
    b, state, config, permit = stopped(tmp_path, monkeypatch)
    state["receipts"].insert(0, {"error_code": "authentication"})
    stable_write(b.root / "transport-recovery.json", permit)
    with pytest.raises(LedgerError, match="non-retryable stop"):
        b.require(configuration=config)
    stable_write(b.root / "stopped.json", {"reason": "security"})
    with pytest.raises(LedgerError, match="durable stage stop"):
        b.require(configuration=config)
