"""Every committed row as the complete record it is, judged before it is projected.

The read side used to judge what it had already decided to keep: a Trace row was reduced to
its JSON payload, a checkpoint to one of its keys, an audit row to whatever happened to be a
mapping.  Damage that lived in a relation column, in a field the reader did not need, or in
the *shape* of a stored payload therefore passed.  This module owns the opposite rule
(B-AUD2-02/03):

* the inventory of persisted classes is declared once, as data, in
  :mod:`.record_inventory`, and it is the same inventory the offline scan reports on;
* one row is judged from its own columns **and** its own JSON payload, as the complete
  records its closed types describe, before any projection is built from it;
* a relation column that disagrees with the payload beside it is a finding, because a reader
  that answers from one half would answer with a record nobody committed;
* a legacy shape is allowed only where a version is *declared*: the ``v0`` Trace row is the
  row whose payload is exactly ``{}`` and whose own columns satisfy the full event contract.
  Nothing else can downgrade itself into an older shape, and the adaptation happens after
  the original content has been read, never by dropping a field.

The judging functions here are total: a damaged row produces fixed category names, it never
raises a ``KeyError``, a ``ValueError`` or a validation error into its caller.  Refusing a
record is the caller's decision, and the refusal it reports carries the record class and the
categories — never a stored value (AC-04.2).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from . import record_guard
from .contracts import RedactionStatus
from .record_inventory import CLASSES_BY_TABLE, ScanClass


#: What a reader calls the record of one table.  These are the fixed names a refusal and an
#: internal log may carry; they never contain a stored value or a JSON path.
REFUSAL_NAMES: dict[str, str] = {
    "review_run": "review_run",
    "review_run_generation": "review_run",
    "review_worker_attempt": "review_run",
    "review_branch_checkpoint": "review_checkpoint",
    "review_run_summary": "review_run",
    "review_idempotency": "review_run",
    "review_receipt": "review_run",
    "review_trace_event": "review_trace",
    "review_audit_event": "review_audit",
}


@dataclass(frozen=True, slots=True)
class Refusal:
    """One committed record that no longer satisfies the contract its readers enforce."""

    record: str
    categories: tuple[str, ...]


# ------------------------------------------------------------------------ row values


def row_mapping(row: Any) -> dict[str, Any]:
    """One database row as ``{column: value}``, with nothing dropped."""

    if isinstance(row, Mapping):
        return dict(row)
    mapping = getattr(row, "_mapping", None)
    if mapping is None:
        raise TypeError("a persisted row has no column mapping")
    return dict(mapping)


def json_value(value: Any) -> Any:
    """The value one JSON column holds, or a marker when it holds no JSON at all.

    A driver hands ``jsonb`` back already parsed; a text column or a hand-written fixture
    may still carry the document.  A value that is neither is not a record, and judging it
    is the contract's job, not this one's.
    """

    if isinstance(value, (str, bytes, bytearray)):
        try:
            return json.loads(value)
        except ValueError:
            return NOT_JSON
    return value


#: The marker one column carries when its content is not a JSON document at all.
NOT_JSON: Any = object()


def _findings(kind: str, payload: Any, *, location: str) -> tuple[str, ...]:
    """The fixed categories one record earns from its closed type and its value rules."""

    if payload is NOT_JSON:
        return ("record_contract",)
    try:
        # Reached through the module so one scan rule, and one substituted rule, drives
        # both the online reader and the offline inventory.
        report = record_guard.scan_persisted(payload, kind=kind, location=location)
    except Exception as error:  # noqa: BLE001 - judging a row never raises
        return (f"scan_error:{type(error).__name__}",)
    return tuple(sorted({item.category for item in report.findings}))


def columns_of(item: ScanClass, row: Mapping[str, Any]) -> dict[str, Any]:
    """The columns one class owns, taken by name so a missing column is a finding."""

    return {column: row.get(column) for column in item.columns}


# ------------------------------------------------------------------- version decisions


#: The declared versions of one committed ``review_trace_event`` row.  ``v1`` is the shape
#: this build writes: the JSON payload is the whole record.  ``v0`` is the shape a database
#: migrated from an earlier schema carries for rows committed before the payload column was
#: populated; the relation columns are the committed authority there.
TRACE_ROW_VERSIONS = ("trace-row.v0", "trace-row.v1")

#: The redaction status a ``v0`` row is adapted with: the column did not exist, and every
#: Trace this product wrote in that era recorded no redaction need.
TRACE_ROW_V0_REDACTION = RedactionStatus.NOT_REQUIRED


def trace_row_version(payload: Any) -> str:
    """Which declared version one stored Trace row carries.

    The selection is the whole recorded shape, never one key: only an *empty* object is the
    legacy row, and it is still required to satisfy the full event contract from its own
    columns below.  A payload with any content — including one that has lost a required key,
    or that is not an object at all — is judged as ``v1`` and fails closed.  No damaged row
    can downgrade itself into the compatible shape (AC-03.4, AC-04.1).
    """

    if isinstance(payload, Mapping) and not payload:
        return "trace-row.v0"
    return "trace-row.v1"


def trace_event_from_columns(row: Mapping[str, Any]) -> dict[str, Any]:
    """The complete Trace event one stored row's relation columns carry.

    The columns are what the write path binds from the checked record, so they are the row's
    own authority.  Rebuilding is only an *adaptation after* the content has been read: every
    field is taken by name, a missing column arrives as ``None`` and is refused by the
    contract instead of being defaulted away.
    """

    return {
        "event_id": row.get("event_id"),
        "run_id": row.get("run_id"),
        "trace_id": row.get("trace_id"),
        "event_type": row.get("event_type"),
        "step": row.get("step"),
        "occurred_at": row.get("occurred_at"),
        "input_summary": json_value(row.get("input_summary")),
        "output_summary": json_value(row.get("output_summary")),
        "input_checksum": row.get("input_checksum"),
        "output_checksum": row.get("output_checksum"),
        "source_versions": json_value(row.get("source_versions")),
        "duration_ms": row.get("duration_ms"),
        "redaction_status": TRACE_ROW_V0_REDACTION,
        "error": row.get("error"),
    }


def isolation_kind(isolation: Any) -> str:
    """Which declared version of the Case input one stored Run carries.

    The selection is the recorded fact itself: an empty column is the ``v0`` shape a database
    migrated from an earlier schema carries, and anything present has to satisfy the complete
    ``v1`` record.  A populated but damaged input therefore cannot downgrade itself.
    """

    return (
        "isolation_v0"
        if isinstance(isolation, Mapping) and not isolation
        else "isolation"
    )


# ---------------------------------------------------------------- column / payload ties

#: The relation columns of one audit row and the payload key each mirrors.
AUDIT_COLUMN_KEYS: tuple[tuple[str, str], ...] = (
    ("audit_id", "audit_id"),
    ("run_id", "run_id"),
    ("case_id", "case_id"),
    ("event_class", "event_class"),
    ("action_type", "action_type"),
    ("actor", "actor"),
    ("authorization_revision", "authorization_revision"),
    ("previous_status", "previous_status"),
    ("new_status", "new_status"),
    ("previous_revision", "previous_revision"),
    ("new_revision", "new_revision"),
    ("result_checksum", "result_checksum"),
    ("idempotency_digest", "idempotency_digest"),
    ("receipt_id", "receipt_id"),
    ("api_version", "api_version"),
    ("occurred_at", "occurred_at"),
)

#: The relation columns of one Trace row and the payload key each mirrors.  ``sequence`` and
#: ``event_ordinal`` are assigned by the row itself, and ``generation``/``attempt``/``worker``
#: are part of the row record rather than of the event, so none of them ties to the payload.
TRACE_COLUMN_KEYS: tuple[tuple[str, str], ...] = (
    ("event_id", "event_id"),
    ("trace_id", "trace_id"),
    ("run_id", "run_id"),
    ("event_type", "event_type"),
    ("step", "step"),
    ("occurred_at", "occurred_at"),
    ("input_summary", "input_summary"),
    ("output_summary", "output_summary"),
    ("input_checksum", "input_checksum"),
    ("output_checksum", "output_checksum"),
    ("source_versions", "source_versions"),
    ("duration_ms", "duration_ms"),
    ("error", "error"),
)

#: The category a reader reports when a relation column disagrees with its own payload.  The
#: audit sequence column is deliberately excluded from the tie: a legacy row whose JSON
#: sequence disagrees with its committed number is the declared anomaly the projection names.
COLUMN_MISMATCH = "scan_column_mismatch"


#: The marker for a value that names no moment at all.
_NOT_MOMENT: Any = object()


def _moment(value: Any) -> Any:
    """The moment one value names, in the one form both halves of a row can be compared in.

    A ``timestamptz`` column arrives as a datetime and the same instant in a JSON payload is
    a string, which this product's own serializer writes with a trailing ``Z``.  Both are
    read as the instant they name; anything that names none stays as it is.
    """

    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return _NOT_MOMENT
    return _NOT_MOMENT


def _same(actual: Any, expected: Any) -> bool:
    """Whether one column still says what the payload beside it says."""

    if isinstance(actual, bool) or isinstance(expected, bool):
        # A boolean is never an interchangeable spelling of a number or a string here.
        return actual is expected
    if isinstance(actual, datetime) or isinstance(expected, datetime):
        left, right = _moment(actual), _moment(expected)
        return left is not _NOT_MOMENT and left == right
    if isinstance(actual, (Mapping, list)) or isinstance(expected, (Mapping, list)):
        return json.dumps(actual, sort_keys=True, default=str) == json.dumps(
            expected, sort_keys=True, default=str
        )
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected and type(actual) is type(expected)
    return str(actual) == str(expected)


def _tie_findings(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    ties: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    # The payload arrives parsed, so its inner values are compared as they are: a string in
    # the document is the string the column has to say, never another document to re-read.
    mismatched = [
        column for column, key in ties if not _same(row.get(column), payload.get(key))
    ]
    return (COLUMN_MISMATCH,) if mismatched else ()


# ------------------------------------------------------------------------- row rules


@dataclass(frozen=True, slots=True)
class TraceRowView:
    """What one committed Trace row is: the event a reader answers with, and its findings."""

    event: Mapping[str, Any] | None
    categories: tuple[str, ...]


def trace_row_view(row: Mapping[str, Any]) -> TraceRowView:
    """Judge one committed Trace row and read the event it actually carries.

    Both halves are persisted facts.  A ``v1`` row is its payload, which additionally has to
    agree with every column that mirrors it; a ``v0`` row is the complete event rebuilt from
    its own columns.  The row record — the ordinal, the generation, the attempt and the
    Worker beside the event — is judged either way, so a value that only ever existed in a
    relation column is seen by the same rules as one that was written into the payload.
    """

    payload = json_value(row.get("event_payload"))
    if trace_row_version(payload) == "trace-row.v0":
        event: Mapping[str, Any] | None = trace_event_from_columns(row)
        categories = _findings("trace", event, location="trace")
    else:
        event = dict(payload) if isinstance(payload, Mapping) else None
        categories = _findings("trace", payload, location="trace")
        if event is not None:
            categories += _tie_findings(row, payload, TRACE_COLUMN_KEYS)
    categories += _findings(
        "trace_row", _trace_row_record(row, payload, event), location="trace"
    )
    return TraceRowView(event=event, categories=tuple(sorted(set(categories))))


def _trace_row_record(
    row: Mapping[str, Any],
    payload: Any,
    event: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The whole ``review_trace_event`` row: its own columns plus the event it carries."""

    return {
        "event_ordinal": row.get("event_ordinal"),
        "generation": row.get("generation"),
        "attempt": row.get("attempt"),
        "worker": row.get("worker"),
        "event": event if event is not None else payload,
    }


@dataclass(frozen=True, slots=True)
class AuditRowView:
    """One committed audit row: its projected event, its legacy anomaly and its findings."""

    event: Mapping[str, Any] | None
    anomaly: str | None
    categories: tuple[str, ...]


def audit_row_view(row: Mapping[str, Any]) -> AuditRowView:
    """Judge one committed audit row and project the sequence it is read with.

    The committed row is the authority for the sequence and the JSON payload is history,
    which is never rewritten.  A payload that is not an object, or whose sequence is not the
    integer its contract declares, is a damaged record and is reported as one — it is never
    skipped, because a silently missing audit event is its own kind of false answer.
    """

    payload = json_value(row.get("event_payload"))
    if not isinstance(payload, Mapping):
        return AuditRowView(event=None, anomaly=None, categories=("record_contract",))
    committed = row.get("sequence")
    stored = payload.get("sequence")
    if not isinstance(committed, int) or isinstance(committed, bool):
        return AuditRowView(event=None, anomaly=None, categories=("record_contract",))
    projected = dict(payload)
    anomaly: str | None = None
    if isinstance(stored, bool) or not isinstance(stored, int):
        return AuditRowView(event=None, anomaly=None, categories=("record_contract",))
    if stored != committed:
        anomaly = str(payload.get("audit_id", ""))
        projected["sequence"] = committed
    categories = _findings("audit", projected, location="audit")
    categories += _tie_findings(row, payload, AUDIT_COLUMN_KEYS)
    return AuditRowView(
        event=projected, anomaly=anomaly, categories=tuple(sorted(set(categories)))
    )


def class_findings(item: ScanClass, row: Mapping[str, Any]) -> tuple[str, ...]:
    """Judge one row as one persisted class, through the rules that class owns."""

    if item.shape == "trace_row":
        return trace_row_view(row).categories
    if item.shape == "audit_row":
        return audit_row_view(row).categories
    if item.shape == "payload":
        assert item.payload_column is not None
        payload = json_value(row.get(item.payload_column))
        kind = isolation_kind(payload) if item.kind == "isolation" else item.kind
        return _findings(kind, payload, location=item.name)
    return _findings(item.kind, columns_of(item, row), location=item.name)


def row_findings(table: str, row: Mapping[str, Any]) -> tuple[str, ...]:
    """Judge one row as every class that is stored in it."""

    categories: set[str] = set()
    for item in CLASSES_BY_TABLE.get(table, ()):
        categories.update(class_findings(item, row))
    return tuple(sorted(categories))


# --------------------------------------------------------------------- the whole Run


@dataclass(frozen=True, slots=True)
class RunRows:
    """Every committed row of one Run, each as its own complete raw record.

    The reader holds this before it holds any projection, so a field a component did not
    think it needed cannot escape the contract that governs it.
    """

    run: Mapping[str, Any]
    generation: Mapping[str, Any] | None
    attempts: tuple[Mapping[str, Any], ...]
    checkpoints: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any] | None
    receipt: Mapping[str, Any] | None
    idempotency: tuple[Mapping[str, Any], ...]
    trace_rows: tuple[Mapping[str, Any], ...]
    audit_rows: tuple[Mapping[str, Any], ...]
    events_loaded: bool
    refusal: Refusal | None = None

    def traces(self) -> tuple[Mapping[str, Any], ...]:
        """The event of each committed Trace row, in sequence order."""

        return tuple(
            item
            for item in (trace_row_view(row).event for row in self.trace_rows)
            if item is not None
        )

    def audits(self) -> tuple[Mapping[str, Any], ...]:
        """The projected event of each committed audit row, in sequence order."""

        return tuple(
            item
            for item in (audit_row_view(row).event for row in self.audit_rows)
            if item is not None
        )

    def anomalies(self) -> tuple[str, ...]:
        """The audit events whose stored sequence disagrees with their committed number."""

        return tuple(
            item
            for item in (audit_row_view(row).anomaly for row in self.audit_rows)
            if item
        )


def assess_run(rows: RunRows) -> Refusal | None:
    """Judge every record one committed Run is made of, or name the one that fails.

    The order is the order a reader assembles the Run in, so the first named class is a
    deterministic answer rather than whichever query happened to finish first.
    """

    ordered: list[tuple[str, Mapping[str, Any] | None]] = [
        ("review_run", rows.run),
        ("review_run_generation", rows.generation),
        *[("review_worker_attempt", item) for item in rows.attempts],
        *[("review_branch_checkpoint", item) for item in rows.checkpoints],
        ("review_run_summary", rows.summary),
        ("review_receipt", rows.receipt),
        *[("review_idempotency", item) for item in rows.idempotency],
        *[("review_trace_event", item) for item in rows.trace_rows],
        *[("review_audit_event", item) for item in rows.audit_rows],
    ]
    for table, row in ordered:
        if row is None:
            continue
        categories = row_findings(table, row)
        if categories:
            return Refusal(
                record=REFUSAL_NAMES.get(table, table), categories=categories
            )
    return None


def checkpoint_rows(
    *, run_id: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The reusable identity of each adopted branch, read from its complete row.

    The trimmed view is what a recovery reuses; it is projected from a row the caller has
    already judged, and it keeps the payload exactly as it was stored.
    """

    return {
        str(row.get("worker")): {
            "input_checksum": row.get("input_checksum"),
            "contract_checksum": row.get("contract_checksum"),
            "result": _result_of(row),
        }
        for row in rows
    }


def _result_of(row: Mapping[str, Any]) -> Any:
    payload = json_value(row.get("result_payload"))
    return payload.get("result") if isinstance(payload, Mapping) else None


__all__ = [
    "AUDIT_COLUMN_KEYS",
    "COLUMN_MISMATCH",
    "REFUSAL_NAMES",
    "TRACE_COLUMN_KEYS",
    "TRACE_ROW_V0_REDACTION",
    "TRACE_ROW_VERSIONS",
    "AuditRowView",
    "NOT_JSON",
    "Refusal",
    "RunRows",
    "TraceRowView",
    "assess_run",
    "audit_row_view",
    "checkpoint_rows",
    "class_findings",
    "columns_of",
    "isolation_kind",
    "json_value",
    "row_findings",
    "row_mapping",
    "trace_event_from_columns",
    "trace_row_view",
]
