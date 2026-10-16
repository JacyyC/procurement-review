"""The guard every persisted record passes, and the scan its readers reuse.

The product persists a bounded set of records: the Run header and its isolation input,
the execution generation, the Worker attempt ledger, adopted branch checkpoints, the
fan-in summary and execution, the idempotency ledger, the local receipt, Trace rows and
approval/rejection audit events.  Each one has exactly one closed type in
:mod:`.record_contracts`, and this module applies it to the *complete* record — top-level
and nested, mappings and arrays, identifiers and error paths — before any of it is
written.

Three rules make this more than a key filter:

* an unknown key is rejected recursively, and a registered key that appears in the wrong
  place is not automatically legal: its value still has to satisfy the declared type;
* the value of a registered open-mapping key is judged by the type that key owns, so a
  string count, a numeric status, an object inside an array or an unregistered word are
  all refused at any depth;
* a value is screened by the declared type of its slot.  A digest or a product-minted
  identity is trusted by type *and* registered format (``run-<32 hex>``,
  ``sha256:<64 hex>``, ...), so a legitimate identifier that happens to contain a
  phone-shaped digit run is not a false positive, while a phone number pasted into an
  identifier slot, or text pretending to be an identity, is still rejected.

The read side reuses :func:`scan_persisted` and :func:`guard_record`, so a record that
was persisted by an older build is judged by the rule the current build applies.

Guard failures never quote the offending value: only the record kind and the fixed
category names reach the log.  A failure inside a state transaction rolls that
transaction back, so a rejected record leaves no partial write behind.
"""

from __future__ import annotations

import json
import logging
import types
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter

from .api_contracts import ReceiptViewV2
from .audit_contracts import (
    ApprovalAuditEventV2,
    CheckpointRecord,
    RunSummaryRecord,
)
from .contracts import (
    Checksum,
    Identifier,
    NonEmptyText,
    ReviewSummary,
    RunEvent,
)
from .field_guard import (
    FORBIDDEN_KEY_PARTS,
    ScanFinding,
    ScanReport,
    scan_typed_value,
)
from .record_contracts import (
    BUSINESS_FACT_RESULT_KEYS,
    ExecutionRecord,
    BUSINESS_FACT_RESULT_RULES,
    BUSINESS_FACT_TOOL_KINDS,
    MAPPING_KEY_REGISTRIES,
    RUN_ISOLATION_KEYS,
    SOURCE_VERSION_STREAMS,
    TRACE_SUMMARY_KEYS,
    AttemptAbandonPatch,
    AttemptCompletionPatch,
    AttemptDiscardPatch,
    AttemptOutcomePatch,
    CheckpointPayload,
    GenerationHeartbeatPatch,
    GenerationRecord,
    GenerationStatusPatch,
    IdempotencyRecord,
    ReceiptRecord,
    RunIsolationRecord,
    RunIsolationRecordV0,
    RunRecordWithTrace,
    RunResumePatch,
    RunStatusPatch,
    TraceRowRecord,
    WorkerAttemptRecord,
    business_fact_namespace,
    check_mapping,
    contract_for,
)
from .run_errors import RecordContractError

logger = logging.getLogger("procurement_review.record_guard")

RECORD_GUARD_VERSION = "review-record-guard.v2"

#: Every record kind the guard knows, and the closed type that defines it.  ``trace`` and
#: ``audit`` reuse the persisted product contracts they are stored as; ``trace_row`` is
#: the complete Trace row, and the ``*_patch`` kinds are the closed update value sets.
RECORD_MODELS: dict[str, type[BaseModel]] = {
    "run": RunRecordWithTrace,
    "isolation": RunIsolationRecord,
    "isolation_v0": RunIsolationRecordV0,
    "generation": GenerationRecord,
    "generation_status": GenerationStatusPatch,
    "generation_heartbeat": GenerationHeartbeatPatch,
    "attempt": WorkerAttemptRecord,
    "attempt_outcome": AttemptOutcomePatch,
    "attempt_completion": AttemptCompletionPatch,
    "attempt_abandon": AttemptAbandonPatch,
    "attempt_discard": AttemptDiscardPatch,
    "run_status": RunStatusPatch,
    "run_resume": RunResumePatch,
    "checkpoint": CheckpointRecord,
    "checkpoint_payload": CheckpointPayload,
    "summary": ReviewSummary,
    "run_summary": RunSummaryRecord,
    "execution": ExecutionRecord,
    "idempotency": IdempotencyRecord,
    "receipt": ReceiptRecord,
    "receipt_payload": ReceiptViewV2,
    "trace": RunEvent,
    "trace_row": TraceRowRecord,
    "audit": ApprovalAuditEventV2,
}


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = [item for item in get_args(annotation) if item is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _sequence_item_annotation(annotation: Any) -> Any:
    args = get_args(_unwrap_optional(annotation))
    if not args:
        return Any
    return args[0]


def _mapping_value_annotation(annotation: Any) -> Any:
    args = get_args(_unwrap_optional(annotation))
    if len(args) == 2:
        return args[1]
    return Any


def _annotation_kind(annotation: Any) -> str:
    """Classify one declared leaf type: identifier, checksum, text or any.

    ``Any`` and an unclassifiable type are **not** identity slots: a value whose declared
    type is unknown is screened as free text, so no annotation grants an exemption by
    accident (AC-02).
    """

    resolved = _unwrap_optional(annotation)
    if resolved is Any:
        return "text"
    if getattr(resolved, "__name__", "") in {"JsonValue", "JsonValueType"}:
        return "text"
    if resolved is str:
        return "text"
    if resolved == Identifier:
        return "identifier"
    if resolved == Checksum:
        return "checksum"
    if resolved == NonEmptyText:
        return "text"
    try:
        schema = TypeAdapter(resolved).json_schema()
    except Exception:  # noqa: BLE001 - an unclassifiable type is screened as text
        return "text"
    if not isinstance(schema, Mapping) or schema.get("type") != "string":
        return "text"
    pattern = schema.get("pattern")
    if pattern in (r"^sha256:[0-9a-f]{64}$", r"^[0-9a-f]{64}$"):
        # A bare-hex digest slot makes the same claim as the prefixed one: a digit run
        # inside a hash is not free text read as a phone number.
        return "checksum"
    if pattern == r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$":
        return "identifier"
    return "identifier" if schema.get("format") == "evidence-locator" else "text"


def _plain_value(value: Any) -> Any:
    """Expand one Python value into the raw values it is made of.

    A model built by ``model_construct`` or changed by ``model_copy`` is not revalidated
    by Pydantic, so the closed contract is re-run on the values it actually carries.
    """

    if isinstance(value, BaseModel):
        return {
            name: _plain_value(getattr(value, name, None))
            for name in type(value).model_fields
        }
    if isinstance(value, Mapping):
        return {key: _plain_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_plain_value(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_plain_value(child) for child in value)
    return value


def to_json_document(payload: Any) -> Any:
    """The JSON document one stored value says it is.

    A database hands back typed values (a ``timestamptz`` arrives as a ``datetime``); the
    declared JSON representation is what a record is judged by, and anything without one
    fails closed instead of being coerced into a record.
    """

    if payload is None or isinstance(payload, (bool, int, str)):
        return payload
    if isinstance(payload, float):
        if payload != payload or payload in (float("inf"), float("-inf")):
            raise RecordContractError("a stored record carries a non-finite number")
        return payload
    if isinstance(payload, Mapping):
        return {str(key): to_json_document(child) for key, child in payload.items()}
    if isinstance(payload, (list, tuple, set, frozenset)):
        return [to_json_document(child) for child in payload]
    if isinstance(payload, Enum):
        return to_json_document(payload.value)
    if isinstance(payload, (datetime, date, time)):
        return payload.isoformat()
    if isinstance(payload, Decimal):
        return str(payload)
    raise RecordContractError("a stored record is not a JSON document")


def validate_memory_record(kind: str, record: Any) -> BaseModel:
    """Judge one record built in memory by its closed type, with no conversion."""

    model = _model_for(kind)
    payload = _plain_value(record)
    if not isinstance(payload, Mapping):
        raise RecordContractError("the record is not a structured record")
    return model.model_validate(payload, strict=True)


def validate_stored_model(model: type[BaseModel], payload: Any) -> BaseModel:
    """Judge one serialized document against one closed model, in strict JSON mode.

    The only entry that accepts a JSON spelling of a date, an enum word or an array, and
    nothing else: a value not stored as its declared type stays illegal.
    """

    document = to_json_document(payload)
    return model.model_validate_json(
        json.dumps(
            document, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ),
        strict=True,
    )


def validate_stored_record(kind: str, payload: Any) -> BaseModel:
    """Judge one stored record of a known kind by strict *JSON* semantics."""

    return validate_stored_model(_model_for(kind), payload)


def _model_for(kind: str) -> type[BaseModel]:
    model = RECORD_MODELS.get(kind)
    if model is None:
        raise RecordContractError(f"unknown record kind: {kind}")
    return model


def guard_record(
    kind: str,
    record: Any,
    *,
    location: str | None = None,
    strict: bool = True,
) -> BaseModel:
    """Validate and screen one complete persisted record, or raise.

    Every write entry point calls this before it persists anything; the returned model is
    the structured projection of the record that was accepted, and it is the only value a
    writer may serialize.  ``strict`` selects which of the two strict entries judges the
    input: ``True`` is an in-memory record (``model_validate`` with ``strict=True``) and
    ``False`` is a payload that already arrived serialized (strict JSON).  Neither one
    converts a value between types: the closed key space and every value rule are enforced
    either way (AC-04.1).
    """

    site = location or kind
    # Resolved here so an unregistered kind is a caller error, not a refused record.
    _model_for(kind)
    try:
        if strict:
            validated = validate_memory_record(kind, record)
        else:
            validated = validate_stored_record(kind, record)
    except Exception as error:  # noqa: BLE001 - the guard fails closed
        _log_rejection(site, ["record_contract"])
        raise RecordContractError(
            f"the {site} record violates its closed contract: {type(error).__name__}",
            ("record_contract",),
        ) from error
    report = scan_record(validated, kind=kind, location=site)
    if report.findings:
        categories = sorted({item.category for item in report.findings})
        _log_rejection(site, categories)
        raise RecordContractError(
            f"the {site} record violates the field guard: " + ", ".join(categories),
            tuple(categories),
        )
    return validated


def scan_record(
    record: BaseModel, *, kind: str | None = None, location: str | None = None
) -> ScanReport:
    """Screen every leaf of a validated record by the type its slot declares."""

    report = ScanReport()
    _walk_model(
        record,
        path="$",
        kind=kind,
        location=location or kind or "record",
        report=report,
    )
    return report


def scan_persisted(payload: Any, *, kind: str, location: str) -> ScanReport:
    """Screen one already-persisted payload in the namespace of its record kind.

    The offline auditor and the read paths use this instead of the raw pattern scanner so
    a stored record is judged by the same typed rules the write path applied.  A payload
    that no longer satisfies its closed type is reported as a contract finding rather
    than raising, so the caller can count it instead of guessing.
    """

    model = RECORD_MODELS.get(kind)
    if model is None:
        report = ScanReport()
        report.scanned_values += 1
        report.findings.append(
            ScanFinding(
                category="unknown_record_kind",
                location=f"{location}:$",
                detail="no closed contract is registered for this record kind",
            )
        )
        return report
    try:
        validated = validate_stored_record(kind, payload)
    except Exception as error:  # noqa: BLE001 - the auditor counts instead of raising
        report = ScanReport()
        report.scanned_values += 1
        report.findings.append(
            ScanFinding(
                category="record_contract",
                location=f"{location}:$",
                detail=f"the stored record violates its closed contract "
                f"({type(error).__name__})",
            )
        )
        return report
    return scan_record(validated, kind=kind, location=location)


def _walk_model(
    model: BaseModel,
    *,
    path: str,
    kind: str | None,
    location: str,
    report: ScanReport,
) -> None:
    for name, field in type(model).model_fields.items():
        value = getattr(model, name, None)
        # ``FieldInfo.annotation`` drops the ``Annotated`` metadata, so the declared
        # identifier/checksum constraints have to be rebuilt before they can classify a
        # slot; otherwise every identifier would look like free text.
        annotation = (
            field.rebuild_annotation()
            if hasattr(field, "rebuild_annotation")
            else field.annotation
        )
        _walk_value(
            value,
            annotation=annotation,
            path=f"{path}.{name}",
            kind=kind,
            location=location,
            report=report,
        )


def _walk_value(
    value: Any,
    *,
    annotation: Any,
    path: str,
    kind: str | None,
    location: str,
    report: ScanReport,
) -> None:
    if value is None:
        return
    if isinstance(value, BaseModel):
        _walk_model(
            value,
            path=path,
            kind=_nested_kind(value, kind),
            location=location,
            report=report,
        )
        return
    if isinstance(value, Mapping):
        contract = contract_for(kind or "", _leaf_name(path))
        if contract is not None:
            check_mapping(
                value,
                contract,
                path=path,
                location=location,
                report=report,
                forbidden_key=_forbidden_key,
            )
            return
        if kind is not None and kind.startswith("business_fact:"):
            # A business fact whose tool carries no registered result shape is an
            # unregistered record, not a new fact: unknown types fail closed.
            report.scanned_values += 1
            report.findings.append(
                ScanFinding(
                    category="unregistered_shape",
                    location=f"{location}:{path}",
                    detail="the business fact's tool has no registered result shape",
                )
            )
            return
        for key, child in value.items():
            report.scanned_values += 1
            if _forbidden_key(str(key)):
                report.findings.append(
                    ScanFinding(
                        category="forbidden_key",
                        location=f"{location}:{path}.{key}",
                        detail="key carries a forbidden part",
                    )
                )
                continue
            _walk_value(
                child,
                annotation=_mapping_value_annotation(annotation),
                path=f"{path}.{key}",
                kind=kind,
                location=location,
                report=report,
            )
        return
    if isinstance(value, (list, tuple)):
        item_annotation = _sequence_item_annotation(annotation)
        for index, child in enumerate(value):
            _walk_value(
                child,
                annotation=item_annotation,
                path=f"{path}[{index}]",
                kind=kind,
                location=location,
                report=report,
            )
        return
    if not isinstance(value, str):
        # Numbers, booleans and timestamps are typed by the closed contract; they carry
        # no free text to screen.
        report.scanned_values += 1
        return
    report.scanned_values += 1
    outcome = scan_typed_value(
        value, kind=_annotation_kind(annotation), location=location, path=path
    )
    report.scanned_values += outcome.scanned_values
    report.findings.extend(outcome.findings)


def _leaf_name(path: str) -> str:
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def _nested_kind(model: BaseModel, outer: str | None) -> str:
    """The registry namespace of a nested model, so its open mappings resolve."""

    name = type(model).__name__
    if name == "RunEvent":
        return "trace"
    if name == "BusinessFact":
        return business_fact_namespace(str(getattr(model, "tool_name", "")))
    if name == "RunIsolationRecord":
        return "isolation"
    return outer or name


def _forbidden_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in FORBIDDEN_KEY_PARTS)


def _log_rejection(location: str, categories: Sequence[str]) -> None:
    # Only the record kind and the fixed category names reach the log: never the value
    # and never the JSON path (AC-04.2).
    logger.info(
        "record guard rejected: location=%s categories=%s",
        location,
        ",".join(categories),
    )


__all__ = [
    "BUSINESS_FACT_RESULT_KEYS",
    "BUSINESS_FACT_RESULT_RULES",
    "BUSINESS_FACT_TOOL_KINDS",
    "MAPPING_KEY_REGISTRIES",
    "RECORD_GUARD_VERSION",
    "RECORD_MODELS",
    "RUN_ISOLATION_KEYS",
    "SOURCE_VERSION_STREAMS",
    "TRACE_SUMMARY_KEYS",
    "AttemptAbandonPatch",
    "AttemptCompletionPatch",
    "AttemptDiscardPatch",
    "AttemptOutcomePatch",
    "CheckpointPayload",
    "GenerationHeartbeatPatch",
    "GenerationRecord",
    "GenerationStatusPatch",
    "IdempotencyRecord",
    "ReceiptRecord",
    "RunIsolationRecord",
    "RunRecordWithTrace",
    "RunResumePatch",
    "RunStatusPatch",
    "TraceRowRecord",
    "WorkerAttemptRecord",
    "guard_record",
    "scan_persisted",
    "scan_record",
    "to_json_document",
    "validate_memory_record",
    "validate_stored_model",
    "validate_stored_record",
]
