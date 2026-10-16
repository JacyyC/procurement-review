"""Admission counterexamples discovered while connecting the C attempt driver."""

import pytest

from procurement_review.eval_attempt_ledger import BudgetExceeded, LedgerError
from test_v1_07b_attempt_ledger import _ledger, _receipt, _register_and_send


def test_actual_overrun_does_not_consume_another_unknown_reservation(tmp_path):
    ledger = _ledger(tmp_path)
    unknown = _register_and_send(ledger, "task:unknown")
    ledger.commit_failure(
        unknown.attempt_id,
        _receipt(
            unknown.task_key,
            1,
            unknown.request_sha256,
            outcome="failed",
            usage_unknown=True,
        ),
    )
    known = _register_and_send(ledger, "task:known")
    receipt = _receipt(known.task_key, 1, known.request_sha256, outcome="failed")
    ledger.commit_failure(
        known.attempt_id, receipt.model_copy(update={"output_tokens": 9900})
    )
    ledger.register_task("task:third", "c" * 64)
    # 9900 actual + 200 unknown reservation already exceeds 10000.
    with pytest.raises(BudgetExceeded):
        ledger.reserve_attempt(
            "task:third",
            request_sha256="d" * 64,
            reserve_input_tokens=1,
            reserve_output_tokens=1,
        )


def test_reserved_request_cannot_change_after_restart(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.register_task("task:reserved", "c" * 64)
    ledger.reserve_attempt(
        "task:reserved",
        request_sha256="d" * 64,
        reserve_input_tokens=10,
        reserve_output_tokens=20,
    )
    with pytest.raises(LedgerError, match="reservation"):
        ledger.reserve_attempt(
            "task:reserved",
            request_sha256="e" * 64,
            reserve_input_tokens=10,
            reserve_output_tokens=20,
        )


@pytest.mark.parametrize("limit", [0, 3, True])
def test_caller_cannot_expand_the_frozen_attempt_limit(tmp_path, limit):
    ledger = _ledger(tmp_path)
    ledger.register_task("task:limited", "c" * 64)
    with pytest.raises(LedgerError):
        ledger.reserve_attempt(
            "task:limited",
            request_sha256="d" * 64,
            reserve_input_tokens=10,
            reserve_output_tokens=20,
            max_attempts=limit,
        )
