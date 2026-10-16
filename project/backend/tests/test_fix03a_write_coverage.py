"""FIX-03A V-03A-03: full write coverage of every persisted column.

V-03A-03 (AC-03) starts from the persisted schema itself: every column of every table a
state transaction writes is enumerated from ``information_schema`` and has to have an
owner — a field of a registered closed record, a registered JSON payload, or an explicitly
named derived value.  A new column without an owner fails the check instead of being
written unchecked.  It then drives the three gaps the FIX-01 audit proved (the create
idempotency digest, a Run's Trace identity on the no-event path, and the Trace
generation/attempt/worker columns) plus the same root cause on the neighbouring entries.

The V-03A-04 rollback half and its negative controls live in
``test_fix03a_transactions``; the shared ledger snapshot lives in ``fix03a_support``.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import text

from procurement_review import record_guard
from procurement_review.contracts import RunEventType
from procurement_review.review_repository import TraceDraft
from procurement_review.run_errors import RecordContractError
from procurement_review.trace import canonical_checksum

from fix03a_support import (
    create_arguments,
    ledger_snapshot,
    processing,
)
from test_fix01b_record_integrity import API_VERSION_V2, database

PHONE_CANARY = "13800001234"
CREDENTIAL_CANARY = "rvk1.cred-canary.ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHECKSUM = "sha256:" + "a" * 64

#: Which closed record owns each persisted table, and which columns are values derived
#: inside the writing statement rather than taken from a record.
TABLE_SCALAR_RECORDS: dict[str, str] = {
    "review_run": "run",
    "review_run_summary": "summary",
    "review_run_generation": "generation",
    "review_worker_attempt": "attempt",
    "review_idempotency": "idempotency",
    "review_receipt": "receipt",
    "review_trace_event": "trace_row",
    "review_audit_event": "audit",
    "review_branch_checkpoint": "checkpoint",
}
#: The JSON columns and the record kind whose serialized projection fills them.
TABLE_JSON_RECORDS: dict[tuple[str, str], str] = {
    ("review_run", "isolation"): "isolation",
    ("review_branch_checkpoint", "result_payload"): "checkpoint_payload",
    ("review_run_summary", "summary_payload"): "summary",
    ("review_run_summary", "execution_payload"): "execution",
    ("review_receipt", "receipt_payload"): "receipt_payload",
    ("review_trace_event", "event_payload"): "trace",
    ("review_audit_event", "event_payload"): "audit",
}
#: Record fields that carry a contract version literal rather than a stored column.
CONTRACT_VERSION_FIELDS = {
    "schema_version": "the shared contract version literal",
    "run_record_version": "the Run record version literal",
    "checkpoint_version": "the checkpoint record version literal",
    "audit_schema_version": "the audit record version literal",
}
#: Columns a statement derives or the database assigns, with the reason they have no
#: record field of their own.
DERIVED_COLUMNS: dict[tuple[str, str], str] = {
    ("review_run", "revision"): "incremented inside the same UPDATE statement",
    ("review_run", "updated_at"): "the writing statement's own clock",
    (
        "review_run_summary",
        "summary_checksum",
    ): "computed from the checked summary record",
    (
        "review_run_summary",
        "human_gate_status",
    ): "the RunStatusPatch the same step applies",
    ("review_run_summary", "committed_revision"): "the Run revision read FOR UPDATE",
    ("review_run_summary", "created_at"): "the writing statement's own clock",
    ("review_trace_event", "sequence"): "PostgreSQL bigserial identity",
    ("review_trace_event", "occurred_at"): "RunEvent.occurred_at of the checked row",
    ("review_audit_event", "sequence"): "allocated inside the audit transaction",
    (
        "review_branch_checkpoint",
        "adopted_at",
    ): "the checkpoint transaction's own clock",
}


def _owned_fields(kind: str) -> set[str]:
    """Every field name the given record kind can own, including nested payloads."""

    if kind == "trace_row":
        from procurement_review.contracts import RunEvent
        from procurement_review.record_contracts import TraceRowRecord

        return set(TraceRowRecord.model_fields) | set(RunEvent.model_fields)
    return set(record_guard.RECORD_MODELS[kind].model_fields)


def _observed_columns(fixture: Any, table: str) -> list[str]:
    with fixture.driver.transaction() as connection:
        return [
            str(row[0])
            for row in connection.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = :t ORDER BY ordinal_position"
                ),
                {"t": table},
            ).all()
        ]


# ------------------------------------------------------------------ V-03A-03 coverage


def test_every_persisted_column_has_a_registered_owner() -> None:
    """V-03A-03: no column of a written table is left without a closed record."""

    with database("v_03a_03_columns") as fixture:
        tables = sorted(TABLE_SCALAR_RECORDS)
        observed: dict[str, list[str]] = {
            table: _observed_columns(fixture, table) for table in tables
        }
        assert observed["review_run"], "the schema was not migrated"
        for table, columns in observed.items():
            scalar_kind = TABLE_SCALAR_RECORDS.get(table)
            for column in columns:
                json_kind = TABLE_JSON_RECORDS.get((table, column))
                if json_kind is not None:
                    assert json_kind in record_guard.RECORD_MODELS, (table, column)
                    continue
                if (table, column) in DERIVED_COLUMNS:
                    continue
                assert scalar_kind is not None, f"{table}.{column} has no owner"
                assert column in _owned_fields(scalar_kind), (
                    f"{table}.{column} is not a field of the {scalar_kind} record"
                )
        # The three JSON payloads that belong to a second kind on the same table.
        assert "isolation" in record_guard.RECORD_MODELS
        assert {
            "summary_payload",
            "execution_payload",
        } <= set(observed["review_run_summary"])
        # And the reverse direction: every declared field of a scalar record is written by
        # some statement, so a contract cannot drift into fiction.
        written = {column for columns in observed.values() for column in columns} | {
            "sequence",
            "adopted_at",
        }
        payload_kinds = set(TABLE_JSON_RECORDS.values()) | {
            "trace_row",
            "checkpoint",
            # The v0 read-side version of the isolation column owns no column of its own.
            "isolation_v0",
        }
        for kind, model in sorted(record_guard.RECORD_MODELS.items()):
            if kind in payload_kinds:
                continue
            for name in model.model_fields:
                assert name in written or name in CONTRACT_VERSION_FIELDS, (
                    f"the {kind} record declares {name}, which no table stores"
                )


def test_the_create_idempotency_digest_cannot_be_a_canary() -> None:
    """V-03A-03: the create mapping is a closed record, not a raw parameter."""

    with database("v_03a_03_create") as fixture:
        ident = fixture.identity()
        run_id = "run-" + secrets.token_hex(16)
        with pytest.raises(RecordContractError):
            fixture.repository.create_run(
                **create_arguments(ident, run_id, request_digest=PHONE_CANARY)
            )
        with fixture.driver.transaction() as connection:
            runs = connection.execute(
                text("SELECT count(*) FROM review_run")
            ).scalar_one()
            mappings = connection.execute(
                text("SELECT count(*) FROM review_idempotency")
            ).scalar_one()
        assert (int(runs), int(mappings)) == (0, 0)
        # The same call with a computed digest succeeds and stores it verbatim.
        digest = canonical_checksum({"case_id": ident.registration.case_id})
        outcome = fixture.repository.create_run(
            **create_arguments(ident, run_id, request_digest=digest)
        )
        assert outcome.created is True
        with fixture.driver.transaction() as connection:
            stored = connection.execute(
                text("SELECT request_digest FROM review_idempotency WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
        assert stored == digest


def test_the_no_event_run_trace_identity_must_be_a_registered_format() -> None:
    """V-03A-03: the Run record owns the Trace identity on the events-free path too."""

    with database("v_03a_03_trace_id") as fixture:
        ident = fixture.identity()
        run_id = "run-" + secrets.token_hex(16)
        for candidate in (f"trace-{PHONE_CANARY}", PHONE_CANARY, "trace-X"):
            with pytest.raises(RecordContractError):
                fixture.repository.create_run(
                    **create_arguments(ident, run_id, trace_id=candidate, events=())
                )
        with fixture.driver.transaction() as connection:
            assert (
                int(
                    connection.execute(
                        text("SELECT count(*) FROM review_run")
                    ).scalar_one()
                )
                == 0
            )
        trace_id = "trace-" + secrets.token_hex(16)
        outcome = fixture.repository.create_run(
            **create_arguments(ident, run_id, trace_id=trace_id, events=())
        )
        assert outcome.created is True
        assert outcome.run.persisted_trace_id == trace_id
        assert outcome.run.trace_event_count == 0
        with fixture.driver.transaction() as connection:
            assert (
                connection.execute(
                    text("SELECT trace_id FROM review_run WHERE run_id = :r"),
                    {"r": run_id},
                ).scalar_one()
                == trace_id
            )


def test_the_trace_auxiliary_columns_cannot_bypass_their_contract() -> None:
    """V-03A-03: the Trace row is one closed record, not an event plus three free columns."""

    with database("v_03a_03_trace_row") as fixture:
        ident = fixture.identity()
        run_id = "run-" + secrets.token_hex(16)
        trace_id = "trace-" + secrets.token_hex(16)
        fixture.repository.create_run(
            **create_arguments(ident, run_id, trace_id=trace_id, events=())
        )
        before = ledger_snapshot(fixture, run_id)
        for draft in (
            TraceDraft(
                event_type=RunEventType.STEP_COMPLETED.value,
                step="worker.contract",
                output_summary={"status": "valid"},
                source_versions={"api": API_VERSION_V2},
                worker=PHONE_CANARY,
                generation=1,
                attempt=1,
            ),
            TraceDraft(
                event_type=RunEventType.STEP_COMPLETED.value,
                step="worker.contract",
                output_summary={"status": "valid"},
                source_versions={"api": API_VERSION_V2},
                worker="contract",
                generation="7",  # type: ignore[arg-type]
                attempt=1,
            ),
            TraceDraft(
                event_type=RunEventType.STEP_COMPLETED.value,
                step="worker.contract",
                output_summary={"status": "valid"},
                source_versions={"api": API_VERSION_V2},
                worker="contract",
                generation=1,
                attempt=0,
            ),
            # An attempt number without its Worker is not a partial row.
            TraceDraft(
                event_type=RunEventType.STEP_COMPLETED.value,
                step="worker.contract",
                output_summary={"status": "valid"},
                source_versions={"api": API_VERSION_V2},
                attempt=1,
            ),
        ):
            with pytest.raises(RecordContractError):
                fixture.repository.append_trace(
                    run_id=run_id, drafts=(draft,), now=datetime.now(UTC)
                )
            assert ledger_snapshot(fixture, run_id) == before
        # A legal worker event persists, and every stored column is the checked value.
        created = fixture.repository.append_trace(
            run_id=run_id,
            drafts=(
                TraceDraft(
                    event_type=RunEventType.STEP_COMPLETED.value,
                    step="worker.contract",
                    output_summary={"status": "valid"},
                    source_versions={"api": API_VERSION_V2},
                    worker="contract",
                    generation=1,
                    attempt=1,
                ),
            ),
            now=datetime.now(UTC),
        )
        assert len(created) == 1
        with fixture.driver.transaction() as connection:
            row = connection.execute(
                text(
                    "SELECT event_ordinal, worker, generation, attempt, step, "
                    "input_summary, output_summary, source_versions, event_payload "
                    "FROM review_trace_event WHERE run_id = :r"
                ),
                {"r": run_id},
            ).one()
        assert (row.worker, int(row.generation), int(row.attempt)) == ("contract", 1, 1)
        payload = row.event_payload
        assert payload["output_summary"] == row.output_summary
        assert payload["input_summary"] == row.input_summary
        assert payload["source_versions"] == row.source_versions
        assert payload["event_id"] == f"event-{run_id}-{int(row.event_ordinal):04d}"
        assert payload["step"] == row.step


def test_a_summary_or_checkpoint_cannot_persist_a_poisoned_payload() -> None:
    """V-03A-03: the stored JSON is the projection of the record the guard accepted."""

    with database("v_03a_03_payloads") as fixture:
        ident = fixture.identity()
        run_id = "run-" + secrets.token_hex(16)
        fixture.repository.create_run(**create_arguments(ident, run_id, events=()))
        before = ledger_snapshot(fixture, run_id)
        for payload in (
            {"run_id": run_id, "customer_notes": "free text"},
            {"run_id": run_id, "findings": [{"unknown": 1}]},
        ):
            with pytest.raises(RecordContractError):
                fixture.repository.commit_summary(
                    run_id=run_id,
                    generation=1,
                    summary_payload=payload,
                    execution_payload={"schema_version": "1.0"},
                    summary_checksum=canonical_checksum(payload),
                    gate_status=processing(),
                    now=datetime.now(UTC),
                )
            assert ledger_snapshot(fixture, run_id) == before
        for result_payload in (
            {"input_checksum": CHECKSUM, "contract_checksum": CHECKSUM, "notes": "x"},
            {"input_checksum": CHECKSUM, "contract_checksum": CHECKSUM, "result": {}},
        ):
            with pytest.raises(RecordContractError):
                fixture.repository.commit_branch(
                    run_id=run_id,
                    generation=1,
                    worker="contract",
                    attempt_no=1,
                    input_checksum=CHECKSUM,
                    contract_checksum=CHECKSUM,
                    result_payload=result_payload,
                    now=datetime.now(UTC),
                )
            assert ledger_snapshot(fixture, run_id) == before


def test_a_generation_status_is_a_closed_word() -> None:
    """V-03A-03: the generation close is a closed update type, not free text."""

    with database("v_03a_03_close") as fixture:
        ident = fixture.identity()
        run_id = "run-" + secrets.token_hex(16)
        fixture.repository.create_run(**create_arguments(ident, run_id, events=()))
        before = ledger_snapshot(fixture, run_id)
        for status in (PHONE_CANARY, "finished", "ACTIVE", ""):
            with pytest.raises(RecordContractError):
                fixture.repository.close_generation(
                    run_id=run_id, generation=1, status=status, now=datetime.now(UTC)
                )
            assert ledger_snapshot(fixture, run_id) == before
        fixture.repository.close_generation(
            run_id=run_id, generation=1, status="completed", now=datetime.now(UTC)
        )
        with fixture.driver.transaction() as connection:
            assert (
                connection.execute(
                    text("SELECT status FROM review_run_generation WHERE run_id = :r"),
                    {"r": run_id},
                ).scalar_one()
                == "completed"
            )


def test_a_heartbeat_is_a_closed_update_type() -> None:
    """V-03A-03: the lease renewal is a closed update, not two free parameters."""

    with database("v_03a_03_heartbeat") as fixture:
        ident = fixture.identity()
        run_id = "run-" + secrets.token_hex(16)
        created = fixture.repository.create_run(
            **create_arguments(ident, run_id, events=())
        )
        owner = created.run.generation.owner_id
        before_lease = created.run.generation.lease_expires_at
        # The renewal's own type refuses anything that is not a timestamp.
        for poisoned in (
            {"heartbeat_at": "now", "lease_expires_at": datetime.now(UTC)},
            {"heartbeat_at": datetime.now(UTC), "lease_expires_at": PHONE_CANARY},
            {"heartbeat_at": datetime.now(UTC)},
            {
                "heartbeat_at": datetime.now(UTC),
                "lease_expires_at": datetime.now(UTC),
                "extra": 1,
            },
        ):
            with pytest.raises(RecordContractError):
                record_guard.guard_record(
                    "generation_heartbeat", poisoned, location="generation_heartbeat"
                )
        later = datetime.now(UTC)
        assert (
            fixture.repository.heartbeat(
                run_id=run_id,
                generation=1,
                owner_id=owner,
                now=later,
                lease_seconds=5_000.0,
            )
            is True
        )
        with fixture.driver.transaction() as connection:
            row = connection.execute(
                text(
                    "SELECT heartbeat_at, lease_expires_at FROM review_run_generation "
                    "WHERE run_id = :r AND generation = 1"
                ),
                {"r": run_id},
            ).one()
        assert row.heartbeat_at == later
        assert row.lease_expires_at > before_lease
        assert row.lease_expires_at == later + timedelta(seconds=5_000.0)
        # A renewal for another owner matches no row and changes nothing.
        snapshot = ledger_snapshot(fixture, run_id)
        assert (
            fixture.repository.heartbeat(
                run_id=run_id,
                generation=1,
                owner_id="owner-" + "0" * 32,
                now=later,
                lease_seconds=120.0,
            )
            is False
        )
        assert ledger_snapshot(fixture, run_id) == snapshot
