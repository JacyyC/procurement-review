"""The committed read model of one persisted review run.

These are the shapes the database has already agreed on: the Run header, its current
generation, checkpoints, attempts, receipt and events.  ``load_run`` assembles them on a
connection the caller already holds, so a state transaction and a read share one
projection instead of keeping two.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from sqlalchemy import text

from .audit_contracts import AuditEventClass
from .contracts import ApprovalActionType, ReviewLifecycleStatus
from .record_rows import (
    NOT_JSON,
    TRACE_ROW_VERSIONS,
    RunRows,
    assess_run,
    checkpoint_rows,
    json_value,
    row_mapping,
    trace_event_from_columns,
    trace_row_version,
)


TERMINAL_STATUSES = (
    ReviewLifecycleStatus.CONFIRMED,
    ReviewLifecycleStatus.REJECTED,
    ReviewLifecycleStatus.CANCELLED,
    ReviewLifecycleStatus.FAILED,
)
PENDING_STATUSES = (
    ReviewLifecycleStatus.PENDING_CONFIRMATION,
    ReviewLifecycleStatus.PENDING_HUMAN_REVIEW,
)
ACTIVE_GENERATION_STATUS = "active"
CLOSED_GENERATION_STATUSES = ("completed", "cancelled", "superseded", "expired")


def _json(value: object) -> str:
    """Serialize one persisted payload deterministically."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


@dataclass(frozen=True, slots=True)
class RunHeader:
    run_id: str
    tenant_id: str
    case_id: str
    case_version: str
    #: ``None`` when the column declares no lifecycle the contract knows: a damaged record
    #: is refused before any of these fields is answered with.
    status: ReviewLifecycleStatus | None
    revision: int
    resume_count: int
    created_by_subject: str
    current_subject_id: str
    request_checksum: str
    document_version: str
    dataset_version: str
    index_version: str
    rules_version: str
    isolation: Any
    error_code: str | None
    error_message: str | None
    created_at: datetime | None
    updated_at: datetime | None
    trace_id: str | None = None


@dataclass(frozen=True, slots=True)
class RunGeneration:
    run_id: str
    generation: int
    subject_id: str
    credential_id: str
    authorization_revision: int
    owner_id: str
    status: str
    heartbeat_at: datetime
    lease_expires_at: datetime
    deadline_at: datetime
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RunAuthorizationIdentity:
    """What an access decision needs before any business record is read.

    Only the write-once columns that name the object: the reader parses no payload, runs no
    guard and projects nothing, so a damaged record cannot answer an authorization question.
    """

    run_id: str
    tenant_id: str
    case_id: str


@dataclass(frozen=True, slots=True)
class GenerationIdentity:
    """Whom one generation was authorized for, plus its current lifecycle state.

    Read without row locks: the identity columns are write-once.  ``subject_id`` and
    ``credential_id`` are ``None`` when the asked-for generation row does not exist.
    """

    tenant_id: str
    case_id: str
    run_status: str
    subject_id: str | None
    credential_id: str | None
    authorization_revision: int
    generation_status: str | None
    lease_expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    run_id: str
    generation: int
    worker: str
    attempt_no: int
    started_at: datetime
    finished_at: datetime | None
    outcome: str | None
    first_error: str | None
    last_error: str | None
    abandoned: bool
    discarded_late: bool


@dataclass(frozen=True, slots=True)
class PersistedRun:
    """Complete read model of one Run, assembled from the committed tables."""

    header: RunHeader
    generation: RunGeneration | None
    summary_payload: Mapping[str, Any] | None
    summary_checksum: str | None
    execution_payload: Mapping[str, Any] | None
    receipt_payload: Mapping[str, Any] | None
    checkpoints: Mapping[str, Mapping[str, Any]]
    attempts: tuple[AttemptRecord, ...]
    #: Every committed row behind this projection, with all of its columns and the verdict
    #: the complete records earned.  A reader refuses a Run from here, so nothing a
    #: component did not need can escape its contract, and nothing it did need is checked
    #: only after it has been trimmed.
    raw: RunRows
    trace_events: tuple[Any, ...]
    audit_events: tuple[Any, ...]
    trace_event_count: int = 0
    audit_event_count: int = 0
    audit_sequence_anomalies: tuple[str, ...] = ()

    @property
    def is_processing(self) -> bool:
        return self.header.status is ReviewLifecycleStatus.PROCESSING

    @property
    def persisted_trace_id(self) -> str | None:
        """The committed Trace identity of this Run, or ``None`` when it has none.

        The Run record owns the identity; a Run committed before the identity was
        persisted still answers with the identity its own earliest event carries.  No
        reader ever derives one from the Run ID, and a record with neither is reported
        as incomplete instead of being given a new identity.
        """

        if self.header.trace_id:
            return self.header.trace_id
        for event in self.trace_events:
            if isinstance(event, Mapping):
                value = event.get("trace_id")
            else:
                value = getattr(event, "trace_id", None)
            if value:
                return str(value)
        return None


@dataclass(frozen=True, slots=True)
class CreateOutcome:
    run: PersistedRun
    created: bool


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    run: PersistedRun
    replay: bool


# The persisted identity of one idempotent operation.  A replay is only ever a result
# kind: nothing is written for it.
IDEMPOTENT_APPLIED = "applied"
IDEMPOTENT_FAILED = "failed"
IDEMPOTENT_REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class TraceDraft:
    """One execution event to persist; it never carries request or model payloads."""

    event_type: str
    step: str
    input_summary: Mapping[str, Any] = field(default_factory=dict)
    output_summary: Mapping[str, Any] = field(default_factory=dict)
    source_versions: Mapping[str, str] = field(default_factory=dict)
    duration_ms: int = 0
    error: str | None = None
    generation: int | None = None
    attempt: int | None = None
    worker: str | None = None


@dataclass(frozen=True, slots=True)
class AuditDraft:
    """One persisted audit event; the raw idempotency key never appears here."""

    audit_id: str
    event_class: AuditEventClass
    occurred_at: datetime
    action_type: ApprovalActionType | None = None
    actor: str | None = None
    authorization_revision: int | None = None
    previous_status: ReviewLifecycleStatus | None = None
    new_status: ReviewLifecycleStatus | None = None
    previous_revision: int | None = None
    new_revision: int | None = None
    result_checksum: str | None = None
    idempotency_digest: str | None = None
    receipt_id: str | None = None
    detail_code: str | None = None


@dataclass(frozen=True, slots=True)
class ResumeOutcome:
    """What one resume transaction decided; it is the only resume record.

    ``replay`` and ``failed`` carry no new generation: the committed idempotency row is
    the whole answer, so a repeated request cannot consume a second budget unit, open a
    second generation or duplicate the audit and Trace events.
    """

    run: PersistedRun
    kind: str
    generation: int | None = None
    failure_code: str | None = None

    @property
    def schedules(self) -> bool:
        """Only a newly applied resume dispatches work."""

        return self.kind == IDEMPOTENT_APPLIED


@dataclass(frozen=True, slots=True)
class BranchCommit:
    adopted: bool
    reason: str


def load_run(
    connection: Any, run_id: str, *, with_events: bool = True
) -> PersistedRun | None:
    """Assemble one committed Run, and the complete record set it is made of.

    Every row arrives with its whole column set, so the guard sees each record as it was
    stored instead of as the fields a reader happened to need.  Building the model is total:
    a damaged value marks the record as refused, it never raises out of a read and never
    becomes a default.
    """

    row = connection.execute(
        text(
            """
            SELECT r.*,
                   coalesce(
                       r.trace_id,
                       (SELECT t.trace_id FROM review_trace_event t
                         WHERE t.run_id = r.run_id ORDER BY t.sequence LIMIT 1)
                   ) AS resolved_trace_id,
                   (SELECT count(*) FROM review_trace_event t
                     WHERE t.run_id = r.run_id) AS trace_event_count,
                   (SELECT count(*) FROM review_audit_event a
                     WHERE a.run_id = r.run_id) AS audit_event_count
            FROM review_run r WHERE r.run_id = :run_id FOR SHARE OF r
            """
        ),
        {"run_id": run_id},
    ).first()
    if row is None:
        return None
    mapping = row_mapping(row)
    # Three values come from this statement rather than from the row: the counts size the
    # event tables, and the resolved identity is one declared compatibility read — a Run
    # committed before the column existed answers with the identity of its own earliest
    # event, and that is the identity the record is judged with.
    resolved_trace_id = mapping.pop("resolved_trace_id", None)
    trace_event_count = mapping.pop("trace_event_count", 0)
    audit_event_count = mapping.pop("audit_event_count", 0)
    mapping["trace_id"] = resolved_trace_id
    rows = RunRows(
        run=mapping,
        generation=_one(
            connection.execute(
                text(
                    "SELECT * FROM review_run_generation WHERE run_id = :run_id "
                    "ORDER BY generation DESC LIMIT 1"
                ),
                {"run_id": run_id},
            ).first()
        ),
        attempts=_many(
            connection.execute(
                text(
                    "SELECT * FROM review_worker_attempt WHERE run_id = :run_id "
                    "ORDER BY generation, worker, attempt_no"
                ),
                {"run_id": run_id},
            ).all()
        ),
        checkpoints=_many(
            connection.execute(
                text(
                    "SELECT * FROM review_branch_checkpoint WHERE run_id = :run_id "
                    "ORDER BY worker"
                ),
                {"run_id": run_id},
            ).all()
        ),
        summary=_one(
            connection.execute(
                text("SELECT * FROM review_run_summary WHERE run_id = :run_id"),
                {"run_id": run_id},
            ).first()
        ),
        receipt=_one(
            connection.execute(
                text("SELECT * FROM review_receipt WHERE run_id = :run_id"),
                {"run_id": run_id},
            ).first()
        ),
        idempotency=_many(
            connection.execute(
                text(
                    "SELECT * FROM review_idempotency WHERE run_id = :run_id "
                    "ORDER BY operation_family, key_digest"
                ),
                {"run_id": run_id},
            ).all()
        ),
        trace_rows=(
            _many(
                connection.execute(
                    text(
                        "SELECT * FROM review_trace_event WHERE run_id = :run_id "
                        "ORDER BY sequence"
                    ),
                    {"run_id": run_id},
                ).all()
            )
            if with_events
            else ()
        ),
        audit_rows=(
            _many(
                connection.execute(
                    text(
                        "SELECT * FROM review_audit_event WHERE run_id = :run_id "
                        "ORDER BY sequence"
                    ),
                    {"run_id": run_id},
                ).all()
            )
            if with_events
            else ()
        ),
        events_loaded=with_events,
    )
    rows = replace(rows, refusal=assess_run(rows))
    header = RunHeader(
        run_id=_text(mapping.get("run_id")),
        tenant_id=_text(mapping.get("tenant_id")),
        case_id=_text(mapping.get("case_id")),
        case_version=_text(mapping.get("case_version")),
        status=_status(mapping.get("status")),
        revision=_int(mapping.get("revision")),
        resume_count=_int(mapping.get("resume_count")),
        created_by_subject=_text(mapping.get("created_by_subject")),
        current_subject_id=_text(mapping.get("current_subject_id")),
        request_checksum=_text(mapping.get("request_checksum")),
        document_version=_text(mapping.get("document_version")),
        dataset_version=_text(mapping.get("dataset_version")),
        index_version=_text(mapping.get("index_version")),
        rules_version=_text(mapping.get("rules_version")),
        isolation=_json_value(mapping.get("isolation")),
        error_code=mapping.get("error_code"),
        error_message=mapping.get("error_message"),
        created_at=mapping.get("created_at"),
        updated_at=mapping.get("updated_at"),
        # The Run record owns the identity; a Run committed before the column existed
        # answers with the identity its own earliest event carries, whether or not the
        # caller asked for the events (a list view must answer the same way).
        trace_id=mapping.get("trace_id"),
    )
    generation = rows.generation
    return PersistedRun(
        header=header,
        generation=(
            None
            if generation is None
            else RunGeneration(
                run_id=_text(generation.get("run_id")),
                generation=_int(generation.get("generation")),
                subject_id=_text(generation.get("subject_id")),
                credential_id=_text(generation.get("credential_id")),
                authorization_revision=_int(generation.get("authorization_revision")),
                owner_id=_text(generation.get("owner_id")),
                status=_text(generation.get("status")),
                heartbeat_at=generation.get("heartbeat_at"),
                lease_expires_at=generation.get("lease_expires_at"),
                deadline_at=generation.get("deadline_at"),
                created_at=generation.get("created_at"),
            )
        ),
        summary_payload=(
            None
            if rows.summary is None
            else _json_value(rows.summary.get("summary_payload"))
        ),
        summary_checksum=(
            None if rows.summary is None else rows.summary.get("summary_checksum")
        ),
        execution_payload=(
            None
            if rows.summary is None
            else _json_value(rows.summary.get("execution_payload"))
        ),
        receipt_payload=(
            None
            if rows.receipt is None
            else _json_value(rows.receipt.get("receipt_payload"))
        ),
        checkpoints=checkpoint_rows(run_id=run_id, rows=rows.checkpoints),
        attempts=tuple(
            AttemptRecord(
                run_id=_text(item.get("run_id")),
                generation=_int(item.get("generation")),
                worker=_text(item.get("worker")),
                attempt_no=_int(item.get("attempt_no")),
                started_at=item.get("started_at"),
                finished_at=item.get("finished_at"),
                outcome=item.get("outcome"),
                first_error=item.get("first_error"),
                last_error=item.get("last_error"),
                abandoned=bool(item.get("abandoned")),
                discarded_late=bool(item.get("discarded_late")),
            )
            for item in rows.attempts
        ),
        trace_events=rows.traces(),
        audit_events=rows.audits(),
        trace_event_count=_int(trace_event_count),
        audit_event_count=_int(audit_event_count),
        audit_sequence_anomalies=rows.anomalies(),
        raw=rows,
    )


def _one(row: Any) -> Mapping[str, Any] | None:
    return None if row is None else row_mapping(row)


def _many(rows: Sequence[Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(row_mapping(row) for row in rows)


def _int(value: Any) -> int:
    """The integer a numeric column holds, or ``0`` for a value that is not one.

    A non-number is a damaged record, and the closed type reports it; reading it as ``0``
    here only keeps the assembly total so a failure never arrives as an unhandled error.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else "" if value is None else str(value)


def _status(value: Any) -> ReviewLifecycleStatus | None:
    """The lifecycle a status column declares, or ``None`` when it declares none."""

    if isinstance(value, ReviewLifecycleStatus):
        return value
    try:
        return ReviewLifecycleStatus(value)
    except ValueError:
        return None


def _json_value(value: Any) -> Any:
    """The value one JSON column holds, or ``None`` when it holds no JSON at all."""

    loaded = json_value(value)
    return None if loaded is NOT_JSON else loaded


__all__ = [
    "ACTIVE_GENERATION_STATUS",
    "AttemptRecord",
    "AuditDraft",
    "ActionOutcome",
    "BranchCommit",
    "CLOSED_GENERATION_STATUSES",
    "CreateOutcome",
    "GenerationIdentity",
    "TraceDraft",
    "IDEMPOTENT_APPLIED",
    "IDEMPOTENT_FAILED",
    "IDEMPOTENT_REPLAY",
    "PENDING_STATUSES",
    "PersistedRun",
    "RunAuthorizationIdentity",
    "RunGeneration",
    "RunHeader",
    "RunRows",
    "TERMINAL_STATUSES",
    "TRACE_ROW_VERSIONS",
    "load_run",
    "trace_event_from_columns",
    "trace_row_version",
]
