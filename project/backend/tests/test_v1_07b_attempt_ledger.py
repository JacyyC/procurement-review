from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from procurement_review.eval_attempt_ledger import (
    AttemptLedger,
    BudgetExceeded,
    InjectedCrash,
    LedgerError,
)
from procurement_review.eval_v2_contracts import AttemptReceiptV1
from procurement_review.gold_v3_contracts import payload_sha256
from v1_07b_support import prediction_for, v3_bundle


def _ledger(tmp_path: Path, *, max_requests: int = 4) -> AttemptLedger:
    return AttemptLedger.create(
        tmp_path / ".local" / "ledger" / "attempts.sqlite3",
        run_id="v1-07-unit-ledger",
        gold_manifest_sha256="a" * 64,
        configuration_sha256="b" * 64,
        max_requests=max_requests,
        max_input_tokens=10_000,
        max_output_tokens=10_000,
    )


def _receipt(
    task_key: str,
    attempt_no: int,
    request_hash: str,
    *,
    outcome: str = "succeeded",
    usage_unknown: bool = False,
) -> AttemptReceiptV1:
    return AttemptReceiptV1(
        task_key=task_key,
        attempt_no=attempt_no,
        outcome=outcome,
        request_sha256=request_hash,
        requested_model_id="offline-test-model",
        resolved_model_id="offline-test-model-v1",
        input_tokens=None if usage_unknown else 11,
        output_tokens=None if usage_unknown else 17,
        request_bytes=128,
        response_bytes=None if usage_unknown else 256,
        finish_reason=None if outcome != "succeeded" else "stop",
        usage_unknown=usage_unknown,
        error_code=None if outcome == "succeeded" else "injected_failure",
    )


def _register_and_send(ledger: AttemptLedger, task_key: str = "task:one"):
    ledger.register_task(task_key, payload_sha256({"task": task_key}))
    request_hash = payload_sha256({"request": task_key, "attempt": 1})
    reserved = ledger.reserve_attempt(
        task_key,
        request_sha256=request_hash,
        reserve_input_tokens=100,
        reserve_output_tokens=200,
    )
    ledger.mark_sent(reserved.attempt_id)
    return reserved


def test_success_is_atomically_reusable_and_never_sent_twice(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    reserved = _register_and_send(ledger)
    truth = v3_bundle()[1].truths[0]
    projection = prediction_for(truth)
    ledger.commit_success(
        reserved.attempt_id,
        receipt=_receipt(
            reserved.task_key, reserved.attempt_no, reserved.request_sha256
        ),
        projection=projection,
    )
    reopened = AttemptLedger(ledger.path)
    assert reopened.next_action(reserved.task_key) == "reuse_committed"
    assert reopened.load_committed_projection(reserved.task_key) == projection
    with pytest.raises(LedgerError, match="already committed"):
        reopened.reserve_attempt(
            reserved.task_key,
            request_sha256="c" * 64,
            reserve_input_tokens=1,
            reserve_output_tokens=1,
        )
    assert reopened.audit_summary() == {
        "complete": True,
        "task_count": 1,
        "terminal_task_count": 1,
        "unknown_remote_attempts": 0,
    }
    assert stat.S_IMODE(ledger.path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "crash_boundary",
    [
        "before_atomic_commit",
        "after_receipt_write",
        "after_projection_write",
        "after_commit_marker_write",
        "before_transaction_commit",
    ],
)
def test_crash_inside_commit_rolls_back_and_sent_attempt_becomes_remote_unknown(
    tmp_path: Path, crash_boundary: str
) -> None:
    ledger = _ledger(tmp_path)
    reserved = _register_and_send(ledger)

    def crash(boundary: str) -> None:
        if boundary == crash_boundary:
            raise InjectedCrash(boundary)

    with pytest.raises(InjectedCrash):
        ledger.commit_success(
            reserved.attempt_id,
            receipt=_receipt(
                reserved.task_key, reserved.attempt_no, reserved.request_sha256
            ),
            projection=prediction_for(v3_bundle()[1].truths[0]),
            fault=crash,
        )
    reopened = AttemptLedger(ledger.path)
    assert reopened.next_action(reserved.task_key) == "recover_sent_attempt"
    summary = reopened.recover()
    assert summary.sent_marked_remote_unknown == 1
    assert reopened.next_action(reserved.task_key) == "blocked_remote_unknown"
    budget = reopened.budget_snapshot()
    assert budget.unknown_usage_attempts == 1
    assert budget.known_input_tokens == budget.known_output_tokens == 0
    with pytest.raises(LedgerError, match="unknown remote result"):
        reopened.reserve_attempt(
            reserved.task_key,
            request_sha256="c" * 64,
            reserve_input_tokens=1,
            reserve_output_tokens=1,
        )
    exported = reopened.export(tmp_path / ".local" / "ledger" / "export.json")
    attempt = json.loads(exported.read_text(encoding="utf-8"))["attempts"][0]
    assert attempt["receipt"]["outcome"] == "remote_unknown"
    assert attempt["receipt"]["input_tokens"] is None
    assert attempt["receipt"]["output_tokens"] is None


def test_crash_after_commit_keeps_the_complete_projection(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    reserved = _register_and_send(ledger)
    projection = prediction_for(v3_bundle()[1].truths[0])

    def crash(boundary: str) -> None:
        if boundary == "after_atomic_commit":
            raise InjectedCrash(boundary)

    with pytest.raises(InjectedCrash):
        ledger.commit_success(
            reserved.attempt_id,
            receipt=_receipt(
                reserved.task_key, reserved.attempt_no, reserved.request_sha256
            ),
            projection=projection,
            fault=crash,
        )
    reopened = AttemptLedger(ledger.path)
    assert reopened.next_action(reserved.task_key) == "reuse_committed"
    assert reopened.recover().sent_marked_remote_unknown == 0
    assert reopened.load_committed_projection(reserved.task_key) == projection
    assert reopened.budget_snapshot().known_output_tokens == 17


def test_reserved_attempt_survives_restart_without_becoming_remote_unknown(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    ledger.register_task("task:reserved", payload_sha256({"task": "reserved"}))
    first = ledger.reserve_attempt(
        "task:reserved",
        request_sha256="d" * 64,
        reserve_input_tokens=10,
        reserve_output_tokens=20,
    )
    reopened = AttemptLedger(ledger.path)
    assert reopened.recover().reserved_ready_to_send == 1
    assert reopened.next_action("task:reserved") == "send_reserved"
    assert (
        reopened.reserve_attempt(
            "task:reserved",
            request_sha256="d" * 64,
            reserve_input_tokens=10,
            reserve_output_tokens=20,
        )
        == first
    )


def test_budget_reservations_are_monotonic_and_two_attempt_limit_is_shared(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path, max_requests=2)
    ledger.register_task("task:retry", payload_sha256({"task": "retry"}))
    snapshots = [ledger.budget_snapshot()]
    for attempt_no in (1, 2):
        request_hash = payload_sha256({"attempt": attempt_no})
        reserved = ledger.reserve_attempt(
            "task:retry",
            request_sha256=request_hash,
            reserve_input_tokens=100,
            reserve_output_tokens=200,
        )
        ledger.mark_sent(reserved.attempt_id)
        ledger.commit_failure(
            reserved.attempt_id,
            _receipt("task:retry", attempt_no, request_hash, outcome="failed"),
        )
        snapshots.append(ledger.budget_snapshot())
    assert [item.reserved_requests for item in snapshots] == [0, 1, 2]
    assert [item.reserved_input_tokens for item in snapshots] == [0, 100, 200]
    assert [item.reserved_output_tokens for item in snapshots] == [0, 200, 400]
    with pytest.raises(LedgerError, match="physical attempt budget exhausted"):
        ledger.reserve_attempt(
            "task:retry",
            request_sha256="f" * 64,
            reserve_input_tokens=1,
            reserve_output_tokens=1,
        )


def test_budget_is_reserved_before_send_and_fails_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, max_requests=1)
    ledger.register_task("task:first", payload_sha256({"task": "first"}))
    ledger.register_task("task:second", payload_sha256({"task": "second"}))
    ledger.reserve_attempt(
        "task:first",
        request_sha256="1" * 64,
        reserve_input_tokens=10,
        reserve_output_tokens=10,
    )
    with pytest.raises(BudgetExceeded, match="request budget"):
        ledger.reserve_attempt(
            "task:second",
            request_sha256="2" * 64,
            reserve_input_tokens=10,
            reserve_output_tokens=10,
        )


def test_receipt_identity_and_unknown_usage_cannot_be_forged(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    reserved = _register_and_send(ledger)
    with pytest.raises(LedgerError, match="request hash"):
        ledger.commit_success(
            reserved.attempt_id,
            receipt=_receipt(reserved.task_key, reserved.attempt_no, "f" * 64),
            projection=prediction_for(v3_bundle()[1].truths[0]),
        )
    assert ledger.next_action(reserved.task_key) == "recover_sent_attempt"

    with pytest.raises(ValidationError, match="unknown usage must remain null"):
        AttemptReceiptV1(
            task_key="task:unknown",
            attempt_no=1,
            outcome="remote_unknown",
            request_sha256="a" * 64,
            input_tokens=0,
            output_tokens=0,
            usage_unknown=True,
            error_code="remote_unknown",
        )

    with pytest.raises(ValidationError, match="known usage requires both token values"):
        AttemptReceiptV1(
            task_key="task:known-gap",
            attempt_no=1,
            outcome="failed",
            request_sha256="b" * 64,
            input_tokens=None,
            output_tokens=0,
            usage_unknown=False,
            error_code="injected_failure",
        )

    with pytest.raises(
        ValidationError, match="non-successful receipt requires an error code"
    ):
        AttemptReceiptV1(
            task_key="task:missing-error",
            attempt_no=1,
            outcome="cancelled",
            request_sha256="c" * 64,
            input_tokens=0,
            output_tokens=0,
            usage_unknown=False,
        )
