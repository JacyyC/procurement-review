"""Cumulative limits can be removed only without losing history or stop gates."""

import pytest

from procurement_review.eval_attempt_ledger import LedgerError
from procurement_review.eval_f_budget import FStageBudget
from procurement_review.eval_f_context import stable_write


def test_unlimited_cumulative_requests_preserve_usage_and_unknown(
    tmp_path, monkeypatch
):
    budget = FStageBudget(tmp_path / ".local")
    receipts = [
        {"input_tokens": 17, "output_tokens": 11, "usage_unknown": False},
        {"input_tokens": None, "output_tokens": None, "usage_unknown": True},
    ]
    monkeypatch.setattr(budget, "observations", lambda: (9001, receipts))
    budget.require(diagnostic=True)
    summary = budget.summary()
    assert summary["reserved_attempts"] == 9001
    assert summary["known_input"] == 17 and summary["known_output"] == 11
    assert summary["unknown"] == 1 and summary["unreturned"] == 8999
    assert summary["d_e_f_remaining"] is None
    assert summary["g_remaining"] is None and summary["item_remaining"] is None
    stable_write(budget.root / "stopped.json", {"reason": "foundation_defect"})
    with pytest.raises(LedgerError, match="durable stage stop"):
        budget.require()


@pytest.mark.parametrize(
    "error", ["model_identity_drift", "security_boundary", "authentication"]
)
def test_unlimited_does_not_remove_nonretryable_stop(tmp_path, monkeypatch, error):
    budget = FStageBudget(tmp_path / ".local")
    monkeypatch.setattr(budget, "observations", lambda: (9001, [{"error_code": error}]))
    with pytest.raises(LedgerError, match="non-retryable stop"):
        budget.require()
