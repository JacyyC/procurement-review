"""V1-06 AC-08: structured runtime logging on the service path.

Asserts the key migrations emit log events (run created / branch committed /
summary committed / human action / API lifecycle / access denials including the
reserved-field guard) and that every emitted record stays free of the existing
sensitive patterns — events carry stable IDs and status codes only.
"""

from __future__ import annotations

import contextlib
import logging

import pytest

from procurement_review.field_guard import SENSITIVE_VALUE_PATTERNS

from v1_03_support import DEMO_SUBJECT, SecurityHarness, wait_for_status

API = "/api/v2"
API_VERSION_V2 = "review-api.v2"
LOW_CASE = "CASE-LOW-001"
LOW_VERSION = "case-low-001.v1"


@contextlib.contextmanager
def package_logs(caplog: pytest.LogCaptureFixture):
    """Capture package events.

    The package logger deliberately does not propagate (one deterministic line per
    event), so the caplog handler is attached to it instead of relying on root.
    """

    logger = logging.getLogger("procurement_review")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="procurement_review"):
            yield
    finally:
        logger.removeHandler(caplog.handler)


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def _assert_records_clean(caplog: pytest.LogCaptureFixture, *raw_secrets: str) -> None:
    for record in caplog.records:
        message = record.getMessage()
        for category, pattern in SENSITIVE_VALUE_PATTERNS:
            assert not pattern.search(message), (
                f"log record matched sensitive pattern {category}: {message!r}"
            )
        for secret in raw_secrets:
            assert secret not in message


def test_run_lifecycle_and_action_events_logged(
    fresh_harness: SecurityHarness,
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = fresh_harness.issue(DEMO_SUBJECT)
    headers = {"Authorization": f"Bearer {token}"}
    create_key = "log-lifecycle-create"
    with package_logs(caplog):
        with fresh_harness.client() as client:
            created = client.post(
                f"{API}/reviews",
                headers={**headers, "Idempotency-Key": create_key},
                json={
                    "api_version": API_VERSION_V2,
                    "case_id": LOW_CASE,
                    "case_version": LOW_VERSION,
                },
            )
            assert created.status_code == 202, created.text
            run_id = str(created.json()["run_id"])
            body = wait_for_status(client, run_id, headers, "pending_confirmation")
            confirmed = client.post(
                f"{API}/reviews/{run_id}/confirm",
                headers={**headers, "Idempotency-Key": "log-lifecycle-confirm"},
                json={
                    "api_version": API_VERSION_V2,
                    "expected_version": body["version"],
                    "expected_summary_checksum": body["summary_checksum"],
                },
            )
            assert confirmed.status_code == 200, confirmed.text

    messages = _messages(caplog)
    assert any("api startup: security_store=ready" in item for item in messages)
    assert any(
        f"run created: run_id={run_id} case_id={LOW_CASE}" in item for item in messages
    )
    assert sum("run branch committed:" in item for item in messages) >= 1
    assert any(f"run summary committed: run_id={run_id}" in item for item in messages)
    assert any(
        f"human action applied: run_id={run_id} action=confirm status=confirmed" in item
        for item in messages
    )
    _assert_records_clean(caplog, create_key, token)


def test_access_denial_events_logged(
    fresh_harness: SecurityHarness,
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = fresh_harness.issue(DEMO_SUBJECT)
    with package_logs(caplog):
        with fresh_harness.client() as client:
            anonymous = client.get(f"{API}/reviews")
            assert anonymous.status_code == 401, anonymous.text
            reserved = client.post(
                f"{API}/reviews",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Idempotency-Key": "log-reserved",
                },
                json={
                    "api_version": API_VERSION_V2,
                    "case_id": LOW_CASE,
                    "case_version": LOW_VERSION,
                    "actor": "subject-forged",
                },
            )
            assert reserved.status_code == 422, reserved.text
            assert reserved.json()["error"]["code"] == "reserved_field_rejected"

    messages = _messages(caplog)
    assert any(
        "access denied: operation=review_list reason=missing_credential" in item
        for item in messages
    )
    assert any(
        "access denied: operation=review_create reason=reserved_field_rejected" in item
        for item in messages
    )
    _assert_records_clean(caplog, token)


def test_field_guard_rejection_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from procurement_review.review_repository import FieldGuardError, guard_payload

    dsn = "postgresql://review_owner:Sup3rSecretPassphrase@127.0.0.1:5432/review"
    with package_logs(caplog):
        with pytest.raises(FieldGuardError):
            guard_payload({"reasoning": dsn}, location="trace_output")

    messages = _messages(caplog)
    assert any(
        "field guard rejected: location=trace_output categories=dsn_with_secret" in item
        for item in messages
    ), messages
    # Neither the matched value nor the scanned key name reaches the log.
    _assert_records_clean(caplog, dsn, "reasoning")


def test_logging_level_honours_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from procurement_review import logging_setup

    monkeypatch.setenv(logging_setup.REVIEW_LOG_LEVEL_ENV, "WARNING")
    logging_setup.configure_logging()
    assert logging.getLogger(logging_setup.PACKAGE_LOGGER_NAME).level == logging.WARNING
    monkeypatch.delenv(logging_setup.REVIEW_LOG_LEVEL_ENV)
    logging_setup.configure_logging()
    assert logging.getLogger(logging_setup.PACKAGE_LOGGER_NAME).level == logging.INFO
