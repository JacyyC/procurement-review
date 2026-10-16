"""The declared inventory of every record the review runtime persists (FIX-01B rework).

This is data, not logic: one entry per persisted class, naming the table it lives in, the
closed contract kind that judges it, the relation columns it owns and the JSON column that is
a record of its own.  Two consumers read the same declaration on purpose —

* :mod:`.record_rows` judges every committed row through it, and
* :mod:`.field_scan` reports the offline inventory through it, class by class.

Keeping the list here means a column can be named in exactly one place, so an online reader
and an offline scan cannot drift into checking different halves of the same row.  A class
that is not declared here is not judged by either of them, which is why the scan test reads
the schema itself and fails when a column has no owner.
"""

from __future__ import annotations

from dataclasses import dataclass


#: How one persisted class is stored.  ``columns`` are the relation columns the class owns,
#: ``payload_column`` names the JSON column that is a record of its own, and ``shape`` says
#: which rule reads the row.
@dataclass(frozen=True, slots=True)
class ScanClass:
    name: str
    table: str
    kind: str
    columns: tuple[str, ...]
    payload_column: str | None = None
    #: ``row`` judges the declared columns as one record; ``payload`` judges the JSON column
    #: as the whole record; ``trace_row``/``audit_row`` judge a row that is more than one
    #: record and has a versioned read shape of its own.
    shape: str = "row"
    #: The operation that guarantees at least one row for a scanned Run, or ``None`` when
    #: the class is genuinely optional.
    zero_row_reason: str | None = None
    #: An extra predicate a caller may add; the coverage check still counts every row in
    #: scope, so a predicate that hides rows is reported as a coverage problem.
    row_filter: str = ""


#: Every class the product persists, in the order a reader judges them.
RECORD_SCAN_PLAN: tuple[ScanClass, ...] = (
    ScanClass(
        name="run",
        table="review_run",
        kind="run",
        columns=(
            "run_id",
            "tenant_id",
            "case_id",
            "case_version",
            "status",
            "revision",
            "resume_count",
            "created_by_subject",
            "current_subject_id",
            "request_checksum",
            "document_version",
            "dataset_version",
            "index_version",
            "rules_version",
            "created_at",
            "updated_at",
            "error_code",
            "error_message",
            "trace_id",
        ),
    ),
    ScanClass(
        name="isolation",
        table="review_run",
        kind="isolation",
        columns=("isolation",),
        payload_column="isolation",
        shape="payload",
    ),
    ScanClass(
        name="generation",
        table="review_run_generation",
        kind="generation",
        columns=(
            "run_id",
            "generation",
            "subject_id",
            "credential_id",
            "authorization_revision",
            "owner_id",
            "status",
            "heartbeat_at",
            "lease_expires_at",
            "deadline_at",
            "created_at",
        ),
        zero_row_reason="every created Run opens one execution generation",
    ),
    ScanClass(
        name="attempt",
        table="review_worker_attempt",
        kind="attempt",
        columns=(
            "run_id",
            "generation",
            "worker",
            "attempt_no",
            "started_at",
            "finished_at",
            "outcome",
            "first_error",
            "last_error",
            "abandoned",
            "discarded_late",
        ),
        zero_row_reason="a completed Run books one attempt per Worker",
    ),
    ScanClass(
        name="checkpoint",
        table="review_branch_checkpoint",
        kind="checkpoint_payload",
        columns=("result_payload",),
        payload_column="result_payload",
        shape="payload",
        zero_row_reason="a completed Run adopts one checkpoint per Worker",
    ),
    ScanClass(
        name="checkpoint_columns",
        table="review_branch_checkpoint",
        kind="checkpoint",
        columns=(
            "run_id",
            "worker",
            "generation",
            "attempt_no",
            "input_checksum",
            "contract_checksum",
            "result_checksum",
        ),
        zero_row_reason="a completed Run adopts one checkpoint per Worker",
    ),
    ScanClass(
        name="summary_columns",
        table="review_run_summary",
        kind="run_summary",
        columns=(
            "run_id",
            "summary_checksum",
            "human_gate_status",
            "committed_revision",
        ),
        zero_row_reason="a Run that reached the human gate commits one summary",
    ),
    ScanClass(
        name="summary",
        table="review_run_summary",
        kind="summary",
        columns=("summary_payload",),
        payload_column="summary_payload",
        shape="payload",
        zero_row_reason="a Run that reached the human gate commits one summary",
    ),
    ScanClass(
        name="execution",
        table="review_run_summary",
        kind="execution",
        columns=("execution_payload",),
        payload_column="execution_payload",
        shape="payload",
        zero_row_reason="a Run that reached the human gate commits one execution",
    ),
    ScanClass(
        name="idempotency",
        table="review_idempotency",
        kind="idempotency",
        columns=(
            "tenant_id",
            "subject_id",
            "operation_family",
            "key_digest",
            "request_digest",
            "run_id",
            "action",
            "receipt_id",
            "first_revision",
            "outcome",
            "failure_code",
            "created_at",
        ),
        zero_row_reason="every accepted operation claims its key first",
    ),
    ScanClass(
        name="receipt",
        table="review_receipt",
        kind="receipt",
        columns=(
            "receipt_id",
            "run_id",
            "case_id",
            "human_actor",
            "decision",
            "reviewed_recommendation",
            "recorded_at",
            "idempotency_digest",
        ),
        zero_row_reason="only a confirm or reject records a receipt",
    ),
    ScanClass(
        name="receipt_payload",
        table="review_receipt",
        kind="receipt_payload",
        columns=("receipt_payload",),
        payload_column="receipt_payload",
        shape="payload",
        zero_row_reason="only a confirm or reject records a receipt",
    ),
    ScanClass(
        name="trace",
        table="review_trace_event",
        kind="trace",
        columns=(
            "sequence",
            "event_ordinal",
            "event_id",
            "trace_id",
            "run_id",
            "event_type",
            "step",
            "occurred_at",
            "input_summary",
            "output_summary",
            "input_checksum",
            "output_checksum",
            "source_versions",
            "duration_ms",
            "error",
            "generation",
            "attempt",
            "worker",
            "event_payload",
        ),
        payload_column="event_payload",
        shape="trace_row",
        zero_row_reason="every created Run commits its first Trace event",
    ),
    ScanClass(
        name="audit",
        table="review_audit_event",
        kind="audit",
        columns=(
            "sequence",
            "audit_id",
            "run_id",
            "case_id",
            "event_class",
            "action_type",
            "actor",
            "authorization_revision",
            "previous_status",
            "new_status",
            "previous_revision",
            "new_revision",
            "result_checksum",
            "idempotency_digest",
            "receipt_id",
            "api_version",
            "occurred_at",
            "event_payload",
        ),
        payload_column="event_payload",
        shape="audit_row",
        zero_row_reason="a confirmed Run records one approval audit event",
    ),
)

#: Every table the plan reads; a missing one is an explicit failure.
SCANNED_TABLES = tuple(sorted({item.table for item in RECORD_SCAN_PLAN}))

#: Columns a statement derives or the database assigns.  They carry no record of their own,
#: and the column-coverage check accepts them by this declaration alone.
DERIVED_COLUMNS: dict[str, dict[str, str]] = {
    "review_branch_checkpoint": {
        "adopted_at": "the checkpoint transaction's own clock"
    },
    "review_run_summary": {"created_at": "the summary transaction's own clock"},
    "review_trace_event": {"sequence": "PostgreSQL bigserial identity"},
    "review_audit_event": {"sequence": "allocated inside the audit transaction"},
}

#: The classes stored in one table, in plan order.
CLASSES_BY_TABLE: dict[str, tuple[ScanClass, ...]] = {
    table: tuple(item for item in RECORD_SCAN_PLAN if item.table == table)
    for table in SCANNED_TABLES
}
