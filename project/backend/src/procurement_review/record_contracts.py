"""Closed record types and the registered key spaces of their open mappings (FIX-03A).

This module owns the closed **type** of each persisted record — one Pydantic model per
record and per closed update, so a column can never be written from an undeclared value —
and the registries that decide which open mapping a record may carry and which key names it
may use.  The value rules those keys own live in :mod:`.value_rules`, and the result shapes
of the read-only procurement projection live in :mod:`.business_fact_contracts`.

The registries are deliberately spelled out as data: a new summary key is a one-line
addition here, and the offline scanner picks it up without a second definition.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from .audit_contracts import RunRecord
from .business_fact_contracts import (
    BUSINESS_FACT_RESULT_KEYS,
    BUSINESS_FACT_RESULT_RULES,
    BUSINESS_FACT_TOOL_KINDS,
    BUSINESS_FACT_TOOLS,
    business_fact_namespace,
)
from .contracts import (
    ApprovalActionType,
    CaseType,
    Checksum,
    Identifier,
    RecommendationStatus,
    ReviewLifecycleStatus,
    RiskLevel,
    RunEvent,
    WorkerKind,
)
from .field_guard import ScanFinding, ScanReport
from .precise_run_contracts import (
    ExecutionPayload,
    ExecutionRecord,
    PreciseCheckpointPayloadV1,
    PreciseRunSnapshotV1,
)
from .value_rules import (
    MappingContract,
    ValueRule,
    check_rule,
    count,
    flag,
    identity,
    text,
    text_list,
    token_list,
    word,
)
from .worker_contracts import (
    WorkerBranchResult,
    WorkerFailureCode,
    WorkerOutcome,
)

RECORD_CONTRACT_VERSION = "review-record-contract.v1"


def _vocabulary(enum_type: Any) -> frozenset[str]:
    return frozenset(str(item.value) for item in enum_type)


#: The lifecycle words a Trace summary may report, plus the two step-local words the
#: pipeline uses for a completed validation step.
TRACE_STATUS_VOCABULARY = _vocabulary(ReviewLifecycleStatus) | {"valid", "ok"}
#: Reasons a product step may record.  They are registered words, not free text.
TRACE_REASON_VOCABULARY = frozenset({"lease_expired"})
TRACE_EXECUTION_MODES = frozenset({"test_only"})

#: The error categories one persisted model attempt may carry.  They are the transport's
#: fixed vocabulary, so a receipt reports *which* failure happened instead of free text.
TRACE_ATTEMPT_ERRORS = frozenset(
    {
        "account_quota_exhausted",
        "attempt_budget_exhausted",
        "authentication",
        "check_outside_branch",
        "citation_outside_branch",
        "citation_proposition_unsupported",
        "citation_field_value_unsupported",
        "credential_in_request",
        "deadline_exhausted",
        "duplicate_citation",
        "evidence_envelope_invalid",
        "http_error",
        "invalid_json",
        "invalid_schema",
        "length_truncated",
        "model_identity_drift",
        "rate_limited",
        "required_check_missing_or_duplicate",
        "response_envelope_invalid",
        "safety_refusal",
        "security_boundary",
        "server_error",
        "timeout",
        "tool_call_forbidden",
        "transport",
        "unregistered_provider_error",
        "unsupported_request",
    }
)
#: The finish reasons one stored receipt may report, exactly as the wire declares them.
TRACE_FINISH_REASONS = frozenset(
    {"stop", "length", "content_filter", "tool_calls", "other"}
)
#: The format diagnostics one stored receipt may report.
TRACE_FORMAT_DIAGNOSTICS = frozenset(
    {
        "valid_json",
        "json_syntax",
        "non_json_text",
        "markdown_fence",
        "duplicate_key",
        "invalid_number",
    }
)
#: The correction kinds a stored receipt may name for the attempt that follows it.
TRACE_CORRECTION_KINDS = frozenset({"contract", "citation", "transport", "other"})


TRACE_SUMMARY_RULES: dict[str, ValueRule] = {
    "action": word(_vocabulary(ApprovalActionType)),
    "attempt_error": word(TRACE_ATTEMPT_ERRORS, nullable=True),
    #: The physical attempt this receipt belongs to.  The Trace row carries it as its own
    #: column, but the event payload is what a reader of ``trace_events`` sees, so the
    #: receipt states it too and a Run's metering can pair a row with its own receipt
    #: instead of guessing from order.
    "attempt_no": count(),
    "business_fact_count": count(),
    "business_fact_ids": text_list(),
    "case_id": text(max_length=160),
    "case_type": word(_vocabulary(CaseType)),
    "conflict_count": count(),
    "evidence_ref_count": count(),
    "evidence_ref_ids": text_list(),
    "execution_mode": word(TRACE_EXECUTION_MODES),
    "failure_code": word(_vocabulary(WorkerFailureCode)),
    "failure_count": count(),
    "finding_count": count(),
    "finish_reason": word(TRACE_FINISH_REASONS, nullable=True),
    "format_diagnostic": word(TRACE_FORMAT_DIAGNOSTICS, nullable=True),
    "format_diagnostics": token_list(),
    "format_fields": text_list(),
    "highest_risk": word(_vocabulary(RiskLevel)),
    "http_status": count(nullable=True),
    "input_usage": count(nullable=True),
    "latency_ms": count(),
    "missing_count": count(),
    #: The kind of correction the following attempt of the same branch is sent for, so the
    #: Run's metering can split a citation correction from a contract correction.
    "next_correction_kind": word(TRACE_CORRECTION_KINDS, nullable=True),
    "outcome": word(_vocabulary(WorkerOutcome)),
    "output_usage": count(nullable=True),
    "reason": word(TRACE_REASON_VOCABULARY, nullable=True),
    "receipt_model": identity(nullable=True),
    "recommendation": word(_vocabulary(RecommendationStatus)),
    "request_sha256": identity(nullable=True),
    "response_bytes": count(nullable=True),
    "scenario": text(max_length=160),
    "status": word(TRACE_STATUS_VOCABULARY),
    "usage_unknown": flag(),
    "worker": word(_vocabulary(WorkerKind)),
}

#: The version streams a Trace event may declare.  Values are registered version strings.
SOURCE_VERSION_STREAMS = frozenset(
    {
        "api",
        "approval_audit",
        "checkpoint",
        "dataset",
        "document",
        "execution",
        "index",
        "recovery",
        "rules",
        "snapshot",
        "trace",
        "worker_result",
    }
)
SOURCE_VERSION_RULES: dict[str, ValueRule] = {
    stream: text(max_length=160) for stream in sorted(SOURCE_VERSION_STREAMS)
}

#: Compatibility spellings used by callers and documentation.
TRACE_SUMMARY_KEYS = frozenset(TRACE_SUMMARY_RULES)
RUN_ISOLATION_KEYS = frozenset(
    {
        "allowed_order_ids",
        "allowed_supplier_ids",
        "allowed_transaction_ids",
        "case_document_ids",
        "case_type",
        "execution_snapshot",
        "purchase_order_id",
        "scenario",
        "shared_document_ids",
    }
)

TRACE_SUMMARY_CONTRACT = MappingContract("trace_summary", TRACE_SUMMARY_RULES)
SOURCE_VERSION_CONTRACT = MappingContract("source_versions", SOURCE_VERSION_RULES)

#: The registered key space of every open mapping a persisted record may carry, keyed by
#: the record namespace and the mapping's field name.  A business fact resolves its
#: namespace from its own registered tool name, so its result is closed per tool.
MAPPING_KEY_REGISTRIES: dict[tuple[str, str], MappingContract] = {
    ("trace", "input_summary"): TRACE_SUMMARY_CONTRACT,
    ("trace", "output_summary"): TRACE_SUMMARY_CONTRACT,
    ("trace", "source_versions"): SOURCE_VERSION_CONTRACT,
    **{
        (f"business_fact:{kind}", "result"): MappingContract(
            f"business_fact_result:{kind}", rules
        )
        for kind, rules in BUSINESS_FACT_RESULT_RULES.items()
    },
}


def contract_for(namespace: str, field: str) -> MappingContract | None:
    """The registered contract of one open mapping, or ``None`` when there is none."""

    return MAPPING_KEY_REGISTRIES.get((namespace, field))


def _finding(
    report: ScanReport, category: str, location: str, path: str, detail: str
) -> None:
    report.findings.append(
        ScanFinding(category=category, location=f"{location}:{path}", detail=detail)
    )


def check_mapping(
    mapping: Mapping[Any, Any],
    contract: MappingContract,
    *,
    path: str,
    location: str,
    report: ScanReport,
    forbidden_key: Any,
) -> None:
    """Judge one registered open mapping key by key, then value by value.

    ``forbidden_key`` is the shared key-name predicate of the field guard, passed in so
    this module and the record guard cannot drift apart.
    """

    for key, value in mapping.items():
        name = str(key)
        report.scanned_values += 1
        if name not in contract.fields:
            _finding(
                report,
                "unregistered_key",
                location,
                f"{path}.{name}",
                f"key is outside the {contract.name} key space",
            )
            continue
        if forbidden_key(name):
            _finding(
                report,
                "forbidden_key",
                location,
                f"{path}.{name}",
                "key carries a forbidden part",
            )
            continue
        check_rule(
            value,
            contract.fields[name],
            path=f"{path}.{name}",
            location=location,
            report=report,
        )


class _Record(BaseModel):
    """The common validation policy of every persisted record type."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        str_strip_whitespace=True,
        revalidate_instances="always",
    )


class RunIsolationRecord(_Record):
    """The Case input a Run was created with; its field list is the registry.

    ``execution_snapshot`` is the frozen execution configuration of a Run created by a
    line that declares one; a Run created by a line without a snapshot simply carries
    ``None``, and the version the snapshot declares is checked by its own model.
    """

    case_type: Literal["purchase_order", "contract_change"]
    scenario: Identifier
    purchase_order_id: Identifier
    case_document_ids: tuple[Identifier, ...] = ()
    shared_document_ids: tuple[Identifier, ...] = ()
    allowed_order_ids: tuple[Identifier, ...] = ()
    allowed_supplier_ids: tuple[Identifier, ...] = ()
    allowed_transaction_ids: tuple[Identifier, ...] = ()
    execution_snapshot: PreciseRunSnapshotV1 | None = None


class RunIsolationRecordV0(_Record):
    """The declared ``v0`` shape of a Run's Case input.

    A database migrated from an earlier schema can carry Run rows committed before the
    isolation column held the Case input; those rows answer with an empty mapping.  The
    version is selected by exactly that fact — an empty column — never by a fallback from
    a populated one, so a present but damaged input can never be downgraded into it.  The
    key space stays closed, and every value that *is* present is still judged by the v1
    rule for its key.
    """

    case_type: Literal["purchase_order", "contract_change"] | None = None
    scenario: Identifier | None = None
    purchase_order_id: Identifier | None = None
    case_document_ids: tuple[Identifier, ...] = ()
    shared_document_ids: tuple[Identifier, ...] = ()
    allowed_order_ids: tuple[Identifier, ...] = ()
    allowed_supplier_ids: tuple[Identifier, ...] = ()
    allowed_transaction_ids: tuple[Identifier, ...] = ()


class GenerationRecord(_Record):
    """One execution generation: the inserted row and every later close of it.

    The insert always writes ``active``; the same closed type is what a reader applies to
    the committed row, so the status is the whole registered lifecycle of a generation.
    """

    run_id: Identifier
    generation: Annotated[int, Field(strict=True, ge=1)]
    subject_id: Identifier
    credential_id: Identifier
    authorization_revision: Annotated[int, Field(strict=True, ge=1)]
    owner_id: Identifier
    status: Literal["active", "completed", "cancelled", "superseded", "expired"]
    heartbeat_at: datetime
    lease_expires_at: datetime
    deadline_at: datetime
    created_at: datetime


class WorkerAttemptRecord(_Record):
    run_id: Identifier
    generation: Annotated[int, Field(strict=True, ge=1)]
    worker: Identifier
    attempt_no: Annotated[int, Field(strict=True, ge=1)]
    started_at: datetime
    finished_at: datetime | None = None
    outcome: Identifier | None = None
    first_error: Identifier | None = None
    last_error: Identifier | None = None
    abandoned: bool = False
    discarded_late: bool = False


class CheckpointPayloadV1(_Record):
    """The frozen checkpoint payload: identity digests plus the adopted Worker result."""

    input_checksum: Checksum
    contract_checksum: Checksum
    result: WorkerBranchResult


class CheckpointPayloadRecord(
    RootModel[CheckpointPayloadV1 | PreciseCheckpointPayloadV1]
):
    """One stored checkpoint payload, judged by the shape it actually declares.

    The frozen line's payload carries no version key and the precise line's payload
    declares ``precise-branch-commit.v1``; the two are disjoint, so a damaged payload
    fails both alternatives and is refused rather than downgraded into the older shape.
    """


CheckpointPayload = CheckpointPayloadRecord


class IdempotencyRecord(_Record):
    tenant_id: Identifier
    subject_id: Identifier
    operation_family: Literal["create", "resume", "human_action"]
    key_digest: Checksum
    request_digest: Checksum
    run_id: Identifier | None = None
    action: Identifier | None = None
    receipt_id: Identifier | None = None
    first_revision: Annotated[int, Field(strict=True, ge=1)] | None = None
    outcome: Literal["applied", "failed"] | None = None
    failure_code: Identifier | None = None
    created_at: datetime


class ReceiptRecord(_Record):
    receipt_id: Identifier
    run_id: Identifier
    case_id: Identifier
    human_actor: Identifier
    decision: Identifier
    reviewed_recommendation: Identifier
    recorded_at: datetime
    idempotency_digest: Checksum


class RunRecordWithTrace(RunRecord):
    """The persisted Run row: the Run record plus the Trace identity it owns.

    The Run table carries ``trace_id``, so the Run's own closed type has to carry it too:
    a Run created without events still persists exactly one identity, and that value must
    satisfy the registered Trace formats before it is written.
    """

    trace_id: Identifier

    @model_validator(mode="after")
    def require_registered_trace_identity(self) -> RunRecordWithTrace:
        from .field_guard import trusted_value_kind

        if trusted_value_kind(self.trace_id) != "trace_id":
            raise ValueError(
                "the Run's trace identity is outside its registered format"
            )
        return self


class RunStatusPatch(_Record):
    """The closed value set of a Run status/error update.

    Every status change writes exactly these values; the revision is derived inside the
    same statement and is typed by the row itself.
    """

    status: ReviewLifecycleStatus
    error_code: Identifier | None = None
    error_message: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    current_subject_id: Identifier | None = None


class RunResumePatch(_Record):
    """The closed value set of the resume counter/owner update."""

    current_subject_id: Identifier
    updated_at: datetime


class GenerationStatusPatch(_Record):
    """The closed value set of a generation status update."""

    status: Literal["active", "completed", "cancelled", "superseded", "expired"]


class GenerationHeartbeatPatch(_Record):
    """The closed value set of a lease renewal."""

    heartbeat_at: datetime
    lease_expires_at: datetime


class AttemptOutcomePatch(_Record):
    """The closed value set of an attempt closing update."""

    outcome: Identifier
    error_code: Identifier | None = None
    abandoned: bool = False
    discarded_late: bool = False


class AttemptCompletionPatch(_Record):
    """The closed value set of an attempt completed by an adopted checkpoint."""

    finished_at: datetime
    outcome: Literal["completed"]
    last_error: None = None


class AttemptAbandonPatch(_Record):
    """The closed value set of an attempt abandoned with its Run or generation."""

    finished_at: datetime
    outcome: Literal["abandoned"]
    abandoned: Literal[True] = True


class AttemptDiscardPatch(_Record):
    """The closed value set of an attempt superseded by a later one."""

    discarded_late: Literal[True] = True


class TraceRowRecord(_Record):
    """The complete persisted ``review_trace_event`` row.

    The relation columns and the JSON payload come from this one record: the event is the
    payload contract, and the three identity columns beside it are declared here instead
    of being bound straight from the caller's draft.
    """

    event_ordinal: Annotated[int, Field(strict=True, ge=1)]
    generation: Annotated[int, Field(strict=True, ge=1)] | None = None
    attempt: Annotated[int, Field(strict=True, ge=1)] | None = None
    worker: Identifier | None = None
    event: RunEvent

    @model_validator(mode="after")
    def enforce_row_identity(self) -> TraceRowRecord:
        if self.attempt is not None and self.worker is None:
            raise ValueError("a Trace attempt number requires its Worker")
        if self.worker is not None and self.generation is None:
            raise ValueError("a Trace Worker requires its generation")
        expected = f"event-{self.event.run_id}-{self.event_ordinal:04d}"
        if self.event.event_id != expected:
            raise ValueError("the Trace event identity disagrees with its ordinal")
        return self

    def insert_parameters(self) -> dict[str, Any]:
        """Every bound value of the row, projected once from the checked record."""

        event = self.event.model_dump(mode="json")
        return {
            "ordinal": self.event_ordinal,
            "event_id": event["event_id"],
            "trace_id": event["trace_id"],
            "run_id": event["run_id"],
            "event_type": event["event_type"],
            "step": event["step"],
            # The relation column keeps its typed value; only the JSON columns take the
            # JSON projection of the same checked record.
            "occurred_at": self.event.occurred_at,
            "input": event["input_summary"],
            "output": event["output_summary"],
            "input_checksum": event["input_checksum"],
            "output_checksum": event["output_checksum"],
            "versions": event["source_versions"],
            "duration": event["duration_ms"],
            "error": event["error"],
            "generation": self.generation,
            "attempt": self.attempt,
            "worker": self.worker,
            "payload": event,
        }


__all__ = [
    "ExecutionPayload",
    "ExecutionRecord",
    "AttemptAbandonPatch",
    "AttemptCompletionPatch",
    "AttemptDiscardPatch",
    "AttemptOutcomePatch",
    "BUSINESS_FACT_RESULT_KEYS",
    "BUSINESS_FACT_RESULT_RULES",
    "BUSINESS_FACT_TOOL_KINDS",
    "BUSINESS_FACT_TOOLS",
    "CheckpointPayload",
    "GenerationHeartbeatPatch",
    "GenerationRecord",
    "GenerationStatusPatch",
    "IdempotencyRecord",
    "MAPPING_KEY_REGISTRIES",
    "MappingContract",
    "RECORD_CONTRACT_VERSION",
    "RUN_ISOLATION_KEYS",
    "ReceiptRecord",
    "RunIsolationRecord",
    "RunIsolationRecordV0",
    "RunRecordWithTrace",
    "RunResumePatch",
    "RunStatusPatch",
    "SOURCE_VERSION_CONTRACT",
    "SOURCE_VERSION_RULES",
    "SOURCE_VERSION_STREAMS",
    "TRACE_SUMMARY_CONTRACT",
    "TRACE_SUMMARY_KEYS",
    "TRACE_SUMMARY_RULES",
    "TRACE_STATUS_VOCABULARY",
    "TraceRowRecord",
    "ValueRule",
    "WorkerAttemptRecord",
    "business_fact_namespace",
    "check_mapping",
    "contract_for",
]
