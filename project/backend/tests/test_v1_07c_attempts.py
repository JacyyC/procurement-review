import asyncio
import json
from datetime import timedelta

import httpx
import pytest

from procurement_review.chat_transport_v2 import request_payload
from procurement_review.eval_attempt_ledger import (
    AttemptLedger,
    LedgerError,
    BudgetExceeded,
)
from procurement_review.model_attempt_v2 import AdmissionDenied, BranchAttemptDriver
from v1_07c_support import branch, answer, response, setup_driver


def run(driver, deadline, *, task="task:one", request=None):
    return asyncio.run(
        driver.run(
            task,
            request or branch(),
            deadline=deadline,
            reserve_input=100,
            reserve_output=200,
        )
    )


def export(driver, tmp_path):
    return json.loads(
        driver.ledger.export(tmp_path / ".local" / "export.json").read_text()
    )


def test_first_attempt_only_screen_never_corrects_and_cannot_reset_on_recovery(
    tmp_path,
):
    calls = []
    driver, deadline = setup_driver(
        tmp_path, lambda req: (calls.append(req) or response({"invalid": True}))
    )
    args = dict(deadline=deadline, reserve_input=100, reserve_output=200)
    result = asyncio.run(driver.run("screen", branch(), max_attempts=1, **args))
    assert result.pipeline_failure and not result.first_pass_valid
    assert result.model_request_count == 1 and len(calls) == 1
    with pytest.raises(LedgerError):
        asyncio.run(driver.run("screen", branch(), max_attempts=2, **args))
    assert len(calls) == 1


def test_one_correction_revalidates_and_records_first_and_final_separately(tmp_path):
    sent = []

    def handler(req):
        sent.append(json.loads(req.content))
        return (
            response({"wrong": "an untrusted original answer"})
            if len(sent) == 1
            else response()
        )

    driver, deadline = setup_driver(tmp_path, handler)
    result = run(driver, deadline)
    assert result.first_pass_valid is False and result.corrected_valid is True
    assert result.model_request_count == 2 and not result.pipeline_failure
    assert len(sent) == 2
    assert "untrusted original answer" not in json.dumps(sent[1])
    attempts = export(driver, tmp_path)["attempts"]
    assert [x["receipt"]["error_code"] for x in attempts] == ["invalid_schema", None]
    assert attempts[0]["receipt"]["finish_reason"] == "stop"
    assert attempts[1]["receipt"]["retry_reason"] == "invalid_schema"
    assert driver.ledger.budget_snapshot().known_output_tokens == 82
    # A new driver and process-style reopen must reuse the complete projection.
    again = BranchAttemptDriver(
        AttemptLedger(driver.ledger.path), driver.transport, driver.authority
    )
    assert run(again, deadline) == result and len(sent) == 2
    assert export(again, tmp_path) == export(driver, tmp_path)


@pytest.mark.parametrize(
    "failure_sequence",
    [
        ("schema", "transport"),
        ("transport", "schema"),
        ("schema", "schema"),
        ("transport", "transport"),
    ],
)
def test_correction_and_transport_share_two_attempts(failure_sequence, tmp_path):
    count = 0

    def handler(req):
        nonlocal count
        failure = failure_sequence[count]
        count += 1
        if failure == "transport":
            raise httpx.ReadError("private exception text")
        return response({"bad": True})

    driver, deadline = setup_driver(tmp_path, handler)
    result = run(driver, deadline)
    assert count == 2 and result.pipeline_failure and not result.corrected_valid
    assert result.claims == ()
    assert run(driver, deadline) == result and count == 2
    text = json.dumps(export(driver, tmp_path))
    assert "private exception text" not in text


@pytest.mark.parametrize(
    "failure",
    [
        "citation",
        "check",
        "credential",
        "canary",
        "refusal",
        "tools",
        "model",
        "authentication",
        "unsupported",
        "length",
    ],
)
def test_security_identity_and_refusal_never_prompted_to_try_again(failure, tmp_path):
    count = 0

    def handler(req):
        nonlocal count
        count += 1
        body = answer()
        if failure == "citation":
            body["findings"][0]["evidence_refs"] = ["ref-" + "f" * 64]
        if failure == "check":
            body["findings"][0]["obligation_id"] = "O-05"
        if failure == "credential":
            body["findings"][0]["statement"] = "test-credential-not-real"
        if failure == "canary":
            body["findings"][0]["statement"] = "SYNTHETIC_SECRET_TO_EXTERNAL_SINK"
        return response(
            body,
            model="other-model" if failure == "model" else "test-model-1",
            finish="tool_calls"
            if failure == "tools"
            else "length"
            if failure == "length"
            else "stop",
            refusal="refused" if failure == "refusal" else None,
            status=401
            if failure == "authentication"
            else 400
            if failure == "unsupported"
            else 200,
        )

    driver, deadline = setup_driver(tmp_path, handler)
    result = run(driver, deadline)
    assert count == 1 and result.pipeline_failure and result.claims == ()
    text = json.dumps(export(driver, tmp_path))
    assert (
        "test-credential-not-real" not in text
        and "SYNTHETIC_SECRET_TO_EXTERNAL_SINK" not in text
    )
    if failure == "length":
        assert (
            export(driver, tmp_path)["attempts"][0]["receipt"]["error_code"]
            == "length_truncated"
        )


def test_unknown_is_valid_without_correction(tmp_path):
    count = []
    driver, deadline = setup_driver(
        tmp_path, lambda req: (count.append(req) or response(answer(state="unknown")))
    )
    result = run(driver, deadline)
    assert result.first_pass_valid and not result.corrected_valid
    assert result.claims[0].normalized_value == "UNKNOWN" and len(count) == 1


def test_every_retry_reenters_cancellation_gate(tmp_path):
    def handler(req):
        driver.authority.cancel()
        return response({"bad": True})

    driver, deadline = setup_driver(tmp_path, handler)
    result = run(driver, deadline)
    assert result.model_request_count == 1 and result.pipeline_failure
    assert driver.ledger.budget_snapshot().known_output_tokens == 41
    assert (
        export(driver, tmp_path)["attempts"][0]["receipt"]["error_code"]
        == "late_result_discarded"
    )


def test_deadline_elapsed_drops_late_success_but_keeps_actual_usage(tmp_path):
    from datetime import UTC, datetime

    current = [datetime.now(UTC)]

    def handler(req):
        current[0] += timedelta(seconds=100)
        return response()

    driver, deadline = setup_driver(tmp_path, handler, now=lambda: current[0])
    result = run(driver, deadline)
    assert result.pipeline_failure and result.claims == ()
    assert driver.ledger.budget_snapshot().known_input_tokens == 31


@pytest.mark.parametrize(
    "denial", ["cancel", "scope", "deadline", "unknown_budget", "cumulative_budget"]
)
def test_pre_admission_denial_sends_zero_requests(denial, tmp_path):
    sent = []
    driver, deadline = setup_driver(
        tmp_path, lambda req: (sent.append(req) or response())
    )
    if denial == "cancel":
        driver.authority.cancel()
    if denial == "scope":
        driver.authority.scope_ids = frozenset({"V107-OTHER"})
    if denial == "deadline":
        deadline = driver.authority.now() - timedelta(seconds=1)
    if denial == "unknown_budget":
        driver.authority.proof = driver.authority.proof.model_copy(
            update={"historical_unknown_output_reserve": None}
        )
    if denial == "cumulative_budget":
        driver.authority.proof = driver.authority.proof.model_copy(
            update={"cumulative_output_upper": 24_000_000}
        )
    with pytest.raises((AdmissionDenied, LedgerError)):
        run(driver, deadline)
    assert sent == [] and driver.ledger.budget_snapshot().reserved_requests == 0


def test_retry_budget_denial_keeps_first_failure_and_never_zero_fills(tmp_path):
    count = []

    def handler(req):
        count.append(req)
        raise httpx.ReadTimeout("unknown remote use")

    driver, deadline = setup_driver(tmp_path, handler, allocated=1)
    with pytest.raises(BudgetExceeded):
        run(driver, deadline)
    assert len(count) == 1
    assert driver.ledger.budget_snapshot().unknown_usage_attempts == 1
    assert export(driver, tmp_path)["attempts"][0]["receipt"]["output_tokens"] is None


def test_crash_recovery_never_resends_inflight_unknown(tmp_path):
    count = []
    driver, deadline = setup_driver(
        tmp_path, lambda req: (count.append(req) or response())
    )
    driver.ledger.register_task("task:crash", "d" * 64)
    attempt = driver.ledger.reserve_attempt(
        "task:crash",
        request_sha256="c" * 64,
        reserve_input_tokens=100,
        reserve_output_tokens=200,
    )
    driver.ledger.mark_sent(attempt.attempt_id)
    reopened = AttemptLedger(driver.ledger.path)
    assert reopened.recover().sent_marked_remote_unknown == 1
    with pytest.raises(LedgerError):
        run(driver, deadline, task="task:crash")
    assert count == [] and reopened.budget_snapshot().reserved_requests == 1


def test_modes_only_change_declared_output_parameter(tmp_path):
    driver, _ = setup_driver(tmp_path, lambda req: response())
    base = driver.transport.config
    data = {
        mode: request_payload(base.model_copy(update={"mode": mode}), branch())
        for mode in ("json_schema", "json_object", "text")
    }
    assert len({json.dumps(p["messages"], sort_keys=True) for p in data.values()}) == 1
    assert "response_format" not in data["text"]
    assert data["json_object"]["response_format"] == {"type": "json_object"}
    for payload in data.values():
        assert not set(payload) & {
            "tools",
            "thinking",
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
        }


def test_missing_usage_is_unknown_and_not_discarded(tmp_path):
    driver, deadline = setup_driver(tmp_path, lambda req: response(usage=False))
    result = run(driver, deadline)
    assert result.first_pass_valid
    assert driver.ledger.budget_snapshot().unknown_usage_attempts == 1
    assert export(driver, tmp_path)["attempts"][0]["receipt"]["input_tokens"] is None


def test_recovery_rejects_configuration_or_deadline_drift(tmp_path):
    driver, deadline = setup_driver(tmp_path, lambda req: response())
    run(driver, deadline)
    with pytest.raises(LedgerError):
        run(driver, deadline + timedelta(seconds=1))
    driver.transport.config = driver.transport.config.model_copy(
        update={"mode": "text"}
    )
    with pytest.raises(LedgerError):
        BranchAttemptDriver(driver.ledger, driver.transport, driver.authority)


def test_second_reserved_correction_recovers_identical_request_and_budget(tmp_path):
    from procurement_review.eval_attempt_ledger import InjectedCrash

    calls = []

    def handler(req):
        calls.append(json.loads(req.content))
        return response({"bad": True}) if len(calls) == 1 else response()

    driver, deadline = setup_driver(tmp_path, handler)
    original = driver.ledger.mark_sent

    def crash_on_second(attempt_id):
        if driver.ledger.budget_snapshot().reserved_requests == 2:
            raise InjectedCrash("before_second_send")
        original(attempt_id)

    driver.ledger.mark_sent = crash_on_second
    with pytest.raises(InjectedCrash):
        run(driver, deadline)
    reopened = AttemptLedger(driver.ledger.path)
    assert reopened.recover().reserved_ready_to_send == 1
    again = BranchAttemptDriver(reopened, driver.transport, driver.authority)
    result = run(again, deadline)
    assert result.corrected_valid and result.model_request_count == 2
    assert reopened.budget_snapshot().reserved_requests == 2 and len(calls) == 2
    assert "invalid_schema" in calls[1]["messages"][0]["content"]


def test_correction_rechecks_citation_boundary_and_stops(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            return response({"bad": True})
        body = answer()
        body["findings"][0]["evidence_refs"] = ["ref-" + "f" * 64]
        return response(body)

    driver, deadline = setup_driver(tmp_path, handler)
    result = run(driver, deadline)
    assert result.pipeline_failure and result.claims == () and len(calls) == 2
    assert (
        export(driver, tmp_path)["attempts"][1]["receipt"]["error_code"]
        == "citation_outside_branch"
    )


def test_retry_after_is_persisted_and_excess_delay_prevents_retry(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(
            429, headers={"Retry-After": "300"}, json={"error": "do not echo"}
        )

    driver, deadline = setup_driver(tmp_path, handler)
    result = run(driver, deadline)
    assert result.pipeline_failure and len(calls) == 1
    receipt = export(driver, tmp_path)["attempts"][0]["receipt"]
    assert receipt["retry_not_before"] is not None and receipt["usage_unknown"]


def test_retry_rechecks_configuration_before_second_request(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        driver.transport.config = driver.transport.config.model_copy(
            update={"mode": "text"}
        )
        return response({"bad": True})

    driver, deadline = setup_driver(tmp_path, handler)
    with pytest.raises(LedgerError, match="configuration"):
        run(driver, deadline)
    assert len(calls) == 1 and driver.ledger.budget_snapshot().reserved_requests == 1
