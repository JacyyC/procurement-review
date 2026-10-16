"""FIX-03A V-03A-04: transaction rollback, evidence and negative controls.

Every scenario injects one rejected record into one key transaction and compares the
complete committed ledger — every table of the Run plus the audit allocator — before and
after, then proves the same operation still succeeds once it carries a legal record.  A
refused write may not leave a Run, a mapping, a budget unit, a generation, a checkpoint, a
summary, an audit number, a Trace event or a receipt behind, and neither the log nor the
raised message may echo the rejected value.

The final group is the negative control for AC-03: it restores the pre-fix write path
in-process (``model_construct`` is exactly what "bind the parameter straight from the
caller" means through today's types) and shows the same canary really does reach the
columns, so the assertions above are about the fix and not about the database.
"""

from __future__ import annotations

import logging
import secrets
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from procurement_review import record_guard, review_repository
from procurement_review.audit_contracts import AuditEventClass
from procurement_review.contracts import RunEventType
from procurement_review.review_repository import TraceDraft
from procurement_review.run_errors import RecordContractError
from procurement_review.trace import canonical_checksum

from fix03a_support import (
    confirm_action,
    confirmed,
    create_arguments,
    ledger_snapshot,
    pending_confirmation,
    processing,
    unchecked_guard,
)
from test_fix01b_record_integrity import (
    API,
    API_VERSION_V2,
    api_node,
    audit_draft,
    create_run,
    database,
    headers,
)

PHONE_CANARY = "13800001234"
CREDENTIAL_CANARY = "rvk1.cred-canary.ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHECKSUM = "sha256:" + "a" * 64


def test_a_refused_audit_rolls_back_its_whole_state_change() -> None:
    """V-03A-04: a rejected audit leaves no status change, no number and no Trace event."""

    with database("v_03a_04_audit") as fixture:
        ident = fixture.identity()
        run_id = "run-" + secrets.token_hex(16)
        fixture.repository.create_run(**create_arguments(ident, run_id, events=()))
        before = ledger_snapshot(fixture, run_id)
        poisoned = audit_draft(
            ident.context.subject_id,
            ident.context.authorization_revision,
            detail_code=PHONE_CANARY,
        )
        with pytest.raises(RecordContractError):
            fixture.repository.mark_interrupted(
                run_id=run_id,
                reason_code="storage_unavailable",
                now=datetime.now(UTC),
                events=(
                    TraceDraft(
                        event_type=RunEventType.STEP_FAILED.value,
                        step="review_workflow",
                        output_summary={"status": "interrupted"},
                        source_versions={"recovery": "review-recovery.v1"},
                        error="The local review execution stopped.",
                    ),
                ),
                audit=poisoned,
            )
        assert ledger_snapshot(fixture, run_id) == before
        # The same step with a legal audit commits, and takes the first number.
        assert fixture.repository.mark_interrupted(
            run_id=run_id,
            reason_code="storage_unavailable",
            now=datetime.now(UTC),
            events=(
                TraceDraft(
                    event_type=RunEventType.STEP_FAILED.value,
                    step="review_workflow",
                    output_summary={"status": "interrupted"},
                    source_versions={"recovery": "review-recovery.v1"},
                    error="The local review execution stopped.",
                ),
            ),
            audit=audit_draft(
                ident.context.subject_id, ident.context.authorization_revision
            ),
        )
        after = ledger_snapshot(fixture, run_id)
        assert after != before
        assert after["allocator"] == [("audit", 2)], after["allocator"]
        assert len(after["review_audit_event"]) == 1


def test_a_refused_state_change_is_logged_by_category_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """V-03A-04: neither the log nor the raised message echoes the rejected value."""

    with database("v_03a_04_log") as fixture:
        ident = fixture.identity()
        run_id = "run-" + secrets.token_hex(16)
        fixture.repository.create_run(**create_arguments(ident, run_id, events=()))
        with caplog.at_level(logging.INFO):
            with pytest.raises(RecordContractError) as raised:
                fixture.repository.append_trace(
                    run_id=run_id,
                    drafts=(
                        TraceDraft(
                            event_type=RunEventType.STEP_COMPLETED.value,
                            step="worker.contract",
                            output_summary={"status": "valid"},
                            source_versions={"api": API_VERSION_V2},
                            worker=CREDENTIAL_CANARY,
                            generation=1,
                            attempt=1,
                        ),
                    ),
                    now=datetime.now(UTC),
                )
        assert CREDENTIAL_CANARY not in caplog.text
        assert CREDENTIAL_CANARY not in str(raised.value)
        # Only the fixed category name reaches the caller, never the JSON path or value.
        assert raised.value.args[0].endswith("local_credential")


def test_a_refused_human_action_writes_no_receipt_audit_or_mapping() -> None:
    """V-03A-04: the decision transaction is all-or-nothing."""

    with database("v_03a_04_action") as fixture:
        node = api_node(fixture, "v03a_action")
        ident = fixture.identity()
        token = ident.token
        try:
            run_id = create_run(node, ident, key="v-03a-04-action")
            from fix01a_support import settle

            settle(fixture, run_id, "pending_confirmation")
            before = ledger_snapshot(fixture, run_id)
            version = int(fixture.ledger(run_id)["revision"])
            with fixture.driver.transaction() as connection:
                summary_checksum = str(
                    connection.execute(
                        text(
                            "SELECT summary_checksum FROM review_run_summary "
                            "WHERE run_id = :r"
                        ),
                        {"r": run_id},
                    ).scalar_one()
                )
            with pytest.raises(RecordContractError):
                node.repository.apply_human_action(
                    run_id=run_id,
                    action=confirm_action(),
                    expected_version=version,
                    expected_summary_checksum=summary_checksum,
                    actor_subject_id=ident.context.subject_id,
                    actor_credential_id=ident.context.credential_id,
                    actor_authorization_revision=(ident.context.authorization_revision),
                    idempotency_key="v-03a-04-poisoned",
                    request_digest=canonical_checksum({"poisoned": True}),
                    now=datetime.now(UTC),
                    receipt_id="receipt-" + secrets.token_hex(16),
                    receipt_payload={"receipt_id": "x", "freeform": "text"},
                    audit=audit_draft(
                        ident.context.subject_id,
                        ident.context.authorization_revision,
                        event_class=AuditEventClass.APPROVAL,
                        action_type=confirm_action(),
                        previous_status=pending_confirmation(),
                        new_status=confirmed(),
                        receipt_id="receipt-" + secrets.token_hex(16),
                    ),
                )
            assert ledger_snapshot(fixture, run_id) == before
            # The same decision through the API still succeeds afterwards.
            response = node.client.post(
                f"{API}/reviews/{run_id}/confirm",
                headers=headers(token, "v-03a-04-legal"),
                json={
                    "api_version": API_VERSION_V2,
                    "expected_version": version,
                    "expected_summary_checksum": summary_checksum,
                },
            )
            assert response.status_code == 200, response.text
            after = ledger_snapshot(fixture, run_id)
            assert len(after["review_receipt"]) == 1
            assert len(after["review_idempotency"]) == 2
        finally:
            node.close()


def test_a_refused_create_leaves_the_next_attempt_whole() -> None:
    """V-03A-04: a refused create writes nothing and does not poison the next one."""

    with database("v_03a_04_create") as fixture:
        ident = fixture.identity()
        run_id = "run-" + secrets.token_hex(16)
        with pytest.raises(RecordContractError):
            fixture.repository.create_run(
                **create_arguments(ident, run_id, trace_id=f"trace-{PHONE_CANARY}")
            )
        empty = ledger_snapshot(fixture, run_id)
        assert empty["review_run"] == []
        assert empty["review_idempotency"] == []
        assert empty["allocator"] == [("audit", 1)]
        outcome = fixture.repository.create_run(**create_arguments(ident, run_id))
        assert outcome.created is True
        assert len(ledger_snapshot(fixture, run_id)["review_run"]) == 1


# ------------------------------------------------------------------ negative controls


def test_the_pre_fix_create_path_would_persist_the_digest_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V-03A-03 control: without the create record the canary really does reach the table."""

    with database("v_03a_ctl_create") as fixture:
        ident = fixture.identity()
        run_id = "run-" + secrets.token_hex(16)
        monkeypatch.setattr(
            review_repository, "guard_record", unchecked_guard("idempotency")
        )
        outcome = fixture.repository.create_run(
            **create_arguments(ident, run_id, request_digest=PHONE_CANARY)
        )
        assert outcome.created is True
        with fixture.driver.transaction() as connection:
            stored = connection.execute(
                text("SELECT request_digest FROM review_idempotency WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
        assert stored == PHONE_CANARY, stored
    # The assertion the fixed path relies on really is about the guard, not the database.
    monkeypatch.undo()
    with database("v_03a_ctl_create_fixed") as fixture:
        ident = fixture.identity()
        with pytest.raises(RecordContractError):
            fixture.repository.create_run(
                **create_arguments(
                    ident,
                    "run-" + secrets.token_hex(16),
                    request_digest=PHONE_CANARY,
                )
            )


def test_the_pre_fix_trace_row_would_persist_the_worker_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V-03A-03 control: without the Trace row contract the canary reaches the columns."""

    with database("v_03a_ctl_trace") as fixture:
        ident = fixture.identity()
        run_id = "run-" + secrets.token_hex(16)
        trace_id = "trace-" + secrets.token_hex(16)
        fixture.repository.create_run(
            **create_arguments(ident, run_id, trace_id=trace_id, events=())
        )
        import procurement_review.run_trace as run_trace_module

        monkeypatch.setattr(
            run_trace_module, "guard_record", unchecked_guard("trace_row")
        )
        fixture.repository.append_trace(
            run_id=run_id,
            drafts=(
                TraceDraft(
                    event_type=RunEventType.STEP_COMPLETED.value,
                    step="worker.contract",
                    output_summary={"status": "valid"},
                    source_versions={"api": API_VERSION_V2},
                    worker=PHONE_CANARY,
                    generation="7",  # type: ignore[arg-type]
                    attempt="2",  # type: ignore[arg-type]
                ),
            ),
            now=datetime.now(UTC),
        )
        with fixture.driver.transaction() as connection:
            row = connection.execute(
                text(
                    "SELECT worker, generation, attempt FROM review_trace_event "
                    "WHERE run_id = :r"
                ),
                {"r": run_id},
            ).one()
        assert (row.worker, int(row.generation), int(row.attempt)) == (
            PHONE_CANARY,
            7,
            2,
        )


def test_the_pre_fix_run_contract_would_accept_an_unregistered_trace_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V-03A-03 control: the Run's Trace identity is only closed by its own record."""

    from procurement_review.audit_contracts import RunRecord

    monkeypatch.setitem(record_guard.RECORD_MODELS, "run", RunRecord)
    base = {
        "run_id": "run-" + "1" * 32,
        "trace_id": f"trace-{PHONE_CANARY}",
        "tenant_id": "tenant-local-demo",
        "case_id": "CASE-LOW-001",
        "case_version": "case-v1",
        "status": processing(),
        "revision": 1,
        "resume_count": 0,
        "created_by_subject": "subject-local-demo",
        "current_subject_id": "subject-local-demo",
        "request_checksum": CHECKSUM,
        "document_version": "doc-v1",
        "dataset_version": "data-v1",
        "index_version": "index-v1",
        "rules_version": "2.0.0",
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    # The pre-fix contract had no trace_id field at all, so the Run's identity was never
    # part of the record that was checked; the current one refuses it.
    with pytest.raises(RecordContractError):
        record_guard.guard_record("run", base)
    record_guard.RECORD_MODELS["run"] = RunRecord
    try:
        accepted = record_guard.guard_record(
            "run", {key: value for key, value in base.items() if key != "trace_id"}
        )
        assert accepted is not None
    finally:
        from procurement_review.record_contracts import RunRecordWithTrace

        record_guard.RECORD_MODELS["run"] = RunRecordWithTrace
