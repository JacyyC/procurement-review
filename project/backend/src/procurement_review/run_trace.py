"""One Run's Trace: its persisted identity, its ordinal sequence and its envelope.

A Run and its events share exactly one committed Trace identity: the identity is written
with the Run row and falls back to the Run's own earliest event for records committed
before the column existed.  Nothing here ever derives an identity from the Run ID, and a
Run with neither answers with an explicit incomplete-record failure.

Appending is serialized by the Run row lock, so the per-Run ordinal (and the event
identity built from it) stay unique and ordered by commit even when two connections write
at the same time.  Every complete event passes the closed, type-aware record guard before
it is persisted.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from .contracts import RedactionStatus, RunEvent, RunEventType
from .record_contracts import TraceRowRecord
from .record_guard import guard_record, validate_stored_model
from .run_errors import RecordContractError, RecordIncompleteError, RunNotFoundError
from .run_records import PersistedRun, TraceDraft, _json
from .trace import canonical_checksum, TraceEnvelope


def lock_run(connection: Any, run_id: str) -> None:
    """Take the Run row lock that serializes this Run's Trace appenders."""

    locked = connection.execute(
        text("SELECT run_id FROM review_run WHERE run_id = :r FOR UPDATE"),
        {"r": run_id},
    ).first()
    if locked is None:
        raise RunNotFoundError(run_id)


def trace_identity(connection: Any, run_id: str) -> str:
    """The committed Trace identity of one Run, never derived from the Run ID.

    The Run record owns the identity; a Run committed before the identity column existed
    answers with the identity its own earliest event already carries.  A Run with neither
    is incomplete and stops here instead of being given a new identity.
    """

    row = connection.execute(
        text("SELECT trace_id FROM review_run WHERE run_id = :r"),
        {"r": run_id},
    ).first()
    if row is not None and row.trace_id:
        return str(row.trace_id)
    event = connection.execute(
        text(
            "SELECT trace_id FROM review_trace_event WHERE run_id = :r "
            "ORDER BY sequence LIMIT 1"
        ),
        {"r": run_id},
    ).first()
    if event is not None and event.trace_id:
        return str(event.trace_id)
    raise RecordIncompleteError(f"run {run_id} has no persisted trace identity")


def identity_for(run: PersistedRun) -> str:
    """Read back the one persisted Trace identity the Run and its events share.

    A Run without a committed identity, or events that disagree with it, is reported as
    an incomplete/corrupt record: the reader never invents another identity.
    """

    identity = run.persisted_trace_id
    if identity is None:
        raise RecordIncompleteError(
            f"run {run.header.run_id} has no persisted trace identity"
        )
    for event in run.trace_events:
        recorded = (
            event.get("trace_id")
            if isinstance(event, dict)
            else getattr(event, "trace_id", None)
        )
        if recorded is not None and str(recorded) != identity:
            raise RecordContractError(
                f"run {run.header.run_id} carries events of another trace identity"
            )
    return identity


def append_trace(
    connection: Any,
    *,
    run_id: str,
    trace_id: str,
    drafts: Sequence[TraceDraft],
    now: datetime,
) -> tuple[Any, ...]:
    """Persist the given drafts as the next events of one Run's Trace."""

    created: list[Any] = []
    if not drafts:
        return ()
    ordinal = int(
        connection.execute(
            text(
                "SELECT coalesce(max(event_ordinal), 0) FROM review_trace_event "
                "WHERE run_id = :r"
            ),
            {"r": run_id},
        ).scalar_one()
    )
    last_time = connection.execute(
        text("SELECT max(occurred_at) FROM review_trace_event WHERE run_id = :r"),
        {"r": run_id},
    ).scalar()
    for draft in drafts:
        ordinal += 1
        occurred_at = now
        if last_time is not None and occurred_at <= last_time:
            occurred_at = last_time + timedelta(microseconds=1)
        last_time = occurred_at
        input_summary = dict(draft.input_summary)
        output_summary = dict(draft.output_summary)
        source_versions = dict(draft.source_versions or {"trace": "trace.v1"})
        event = RunEvent(
            event_id=f"event-{run_id}-{ordinal:04d}",
            run_id=run_id,
            trace_id=trace_id,
            event_type=RunEventType(draft.event_type),
            step=draft.step,
            occurred_at=occurred_at,
            input_summary=input_summary,
            output_summary=output_summary,
            input_checksum=canonical_checksum(input_summary),
            output_checksum=canonical_checksum(output_summary),
            source_versions=source_versions,
            duration_ms=max(0, draft.duration_ms),
            token_count=0,
            cost_amount=Decimal("0"),
            cost_currency="CNY",
            redaction_status=RedactionStatus.NOT_REQUIRED,
            error=draft.error,
        )
        # The complete row — relation columns and JSON payload together — is screened by
        # its own closed type, and every bound value below is projected from the accepted
        # record: the summaries, the registered version streams, the error path and the
        # generation/attempt/worker columns all come from one checked value, and a
        # failure rolls this transaction back.
        row = guard_record(
            "trace_row",
            {
                "event_ordinal": ordinal,
                "generation": draft.generation,
                "attempt": draft.attempt,
                "worker": draft.worker,
                "event": event,
            },
            location="trace",
        )
        assert isinstance(row, TraceRowRecord)
        parameters = row.insert_parameters()
        connection.execute(
            text(
                """
                INSERT INTO review_trace_event (
                    event_ordinal, event_id, trace_id, run_id, event_type, step,
                    occurred_at, input_summary, output_summary, input_checksum,
                    output_checksum, source_versions, duration_ms, error,
                    generation, attempt, worker, event_payload
                ) VALUES (
                    :ordinal, :event_id, :trace_id, :run_id, :event_type, :step,
                    :occurred_at, CAST(:input AS jsonb), CAST(:output AS jsonb),
                    :input_checksum, :output_checksum, CAST(:versions AS jsonb),
                    :duration, :error, :generation, :attempt, :worker,
                    CAST(:payload AS jsonb)
                )
                """
            ),
            {
                "ordinal": parameters["ordinal"],
                "event_id": parameters["event_id"],
                "trace_id": parameters["trace_id"],
                "run_id": parameters["run_id"],
                "event_type": parameters["event_type"],
                "step": parameters["step"],
                "occurred_at": parameters["occurred_at"],
                "input": _json(parameters["input"]),
                "output": _json(parameters["output"]),
                "input_checksum": parameters["input_checksum"],
                "output_checksum": parameters["output_checksum"],
                "versions": _json(parameters["versions"]),
                "duration": parameters["duration"],
                "error": parameters["error"],
                "generation": parameters["generation"],
                "attempt": parameters["attempt"],
                "worker": parameters["worker"],
                "payload": _json(parameters["payload"]),
            },
        )
        created.append(row.event)
    return tuple(created)


def envelope(run: PersistedRun, *, complete: bool) -> TraceEnvelope:
    """The committed Trace envelope of one Run, or an explicit incomplete record."""

    identity = identity_for(run)
    events = tuple(validate_stored_model(RunEvent, item) for item in run.trace_events)
    if not events:
        # The envelope contract covers at least one committed event; a Run without any is
        # reported as incomplete rather than answered with an empty envelope or a freshly
        # invented identity.
        raise RecordIncompleteError(
            f"run {run.header.run_id} has no committed trace event"
        )
    return TraceEnvelope(
        trace_id=identity,
        run_id=run.header.run_id,
        events=events,
        complete=complete,
    )


__all__ = [
    "append_trace",
    "envelope",
    "identity_for",
    "lock_run",
    "trace_identity",
]
