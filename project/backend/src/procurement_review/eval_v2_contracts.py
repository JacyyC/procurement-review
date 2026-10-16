"""Versioned evaluation contracts for V1-07 attempts and semantic projections."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .gold_v3_contracts import (
    CaseId,
    EvidenceIdentity,
    HexSha256,
    ObligationId,
    ObligationState,
    PropositionIdentity,
    SecurityEventType,
    V3Recommendation,
    canonical_json_bytes,
    payload_sha256,
)

EVAL_MANIFEST_V2 = "eval-run-manifest.v2"
EVAL_LEDGER_V1 = "eval-attempt-ledger.v1"
EVAL_PROJECTION_V2 = "eval-projection.v2"
EVAL_METRICS_V1 = "v1-07-eval-metrics.v1"

TaskKey = Annotated[str, Field(pattern=r"^[a-zA-Z0-9_.:-]{3,192}$")]
RunId = Annotated[str, Field(pattern=r"^v1-07-[a-z0-9][a-z0-9-]{2,95}$")]
ReasonCode = Annotated[str, Field(pattern=r"^[a-z0-9_.:-]+$", max_length=96)]

#: One bounded location token of a refused answer: ``name`` or ``name=value`` where the
#: value is a registered word, a declared field name, a basic JSON type or a bounded
#: number.  It can locate a refusal without quoting anything the model wrote.
DiagnosticToken = Annotated[
    str, Field(pattern=r"^[a-z_]{2,24}(=[A-Za-z0-9_.\[\]-]{1,32})?$", max_length=64)
]

_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[a-z0-9._-]{12,}"),
    re.compile(r"(?i)\bsk-[a-z0-9_-]{12,}"),
    re.compile(r"(?i)(api[_-]?key|authorization|password|secret)\s*[:=]"),
    re.compile(r"SYNTHETIC_SECRET_TO_EXTERNAL_SINK"),
)


class EvalV2Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    @model_validator(mode="after")
    def serialized_values_are_secret_free(self) -> Self:
        _assert_no_secret_tree(self.model_dump(mode="python"))
        return self


class EvalTopology(StrEnum):
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    ORACLE = "independent_oracle"


class ForbiddenEvent(StrEnum):
    EXTERNAL_WRITE = "external_write"
    CROSS_SCOPE_READ = "cross_scope_read"
    SECRET_LEAK = "secret_leak"
    UNAUTHORIZED_TOOL = "unauthorized_tool"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class RetrievalQueryVariant(StrEnum):
    ORIGINAL = "original"
    WITHOUT_CASE_ID = "without_case_id"


class AttemptState(StrEnum):
    RESERVED = "reserved"
    SENT = "sent"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REMOTE_UNKNOWN = "remote_unknown"
    COMMITTED = "committed"


class TaskState(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED_UNKNOWN = "blocked_unknown"
    COMMITTED = "committed"


class FaultTriggerProof(EvalV2Model):
    plan_version: Literal["eval-fault-plan.v1"] = "eval-fault-plan.v1"
    fault_kind: Annotated[str, Field(pattern=r"^[a-z_]+$", max_length=48)]
    injection_layer: Annotated[str, Field(pattern=r"^[a-z_]+$", max_length=48)]
    trigger_id: HexSha256
    detail_code: ReasonCode
    observed: Literal[True] = True


class SemanticClaimProjection(EvalV2Model):
    obligation_id: ObligationId
    state: ObligationState
    proposition: PropositionIdentity
    normalized_value: Annotated[str, Field(min_length=1, max_length=256)]
    evidence: tuple[EvidenceIdentity, ...] = Field(max_length=4)
    reason_codes: tuple[ReasonCode, ...] = Field(max_length=8)
    statement_sha256: HexSha256
    recommended_action_sha256: HexSha256

    @model_validator(mode="after")
    def supported_claim_has_evidence(self) -> Self:
        if self.state in {ObligationState.SUPPORTED, ObligationState.VIOLATED}:
            if not self.evidence:
                raise ValueError("decidable projected claim requires evidence identity")
        _assert_no_secret(self.normalized_value)
        return self


class ConflictProjection(EvalV2Model):
    pair_id: HexSha256
    relation: Literal["must_equal"] = "must_equal"
    left: PropositionIdentity
    right: PropositionIdentity
    left_value: Annotated[str, Field(min_length=1, max_length=256)]
    right_value: Annotated[str, Field(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def values_are_bounded(self) -> Self:
        _assert_no_secret(self.left_value)
        _assert_no_secret(self.right_value)
        if self.left == self.right or self.left_value == self.right_value:
            raise ValueError(
                "a projected conflict needs distinct propositions and values"
            )
        canonical = sorted(
            (self.left.model_dump(mode="json"), self.right.model_dump(mode="json")),
            key=canonical_json_bytes,
        )
        expected = payload_sha256(
            {"left": canonical[0], "relation": self.relation, "right": canonical[1]}
        )
        if self.pair_id != expected:
            raise ValueError("projected conflict pair id mismatch")
        return self


class SecurityEventProjection(EvalV2Model):
    event_type: SecurityEventType
    source_id: Annotated[str, Field(pattern=r"^V107-[A-Z0-9-]+$")]
    location: Annotated[str, Field(min_length=1, max_length=160)]

    @model_validator(mode="after")
    def location_has_no_secret(self) -> Self:
        _assert_no_secret(self.location)
        return self


class RetrievalObservation(EvalV2Model):
    """Bounded evidence identities for one pre-registered retrieval slice."""

    query_variant: RetrievalQueryVariant
    evidence_domain: Literal["policy", "contract"]
    required_evidence: tuple[EvidenceIdentity, ...] = Field(max_length=16)
    top_five: tuple[EvidenceIdentity, ...] = Field(max_length=5)
    negative_sample: bool = False
    wrong_scope_count: Annotated[int, Field(ge=0)] = 0
    wrong_version_count: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def slice_is_consistent(self) -> Self:
        if self.negative_sample and self.required_evidence:
            raise ValueError("negative retrieval samples cannot require evidence")
        if not self.negative_sample and not self.required_evidence:
            raise ValueError("positive retrieval samples require evidence identities")
        if len(self.required_evidence) != len(set(self.required_evidence)):
            raise ValueError("required retrieval evidence must be unique")
        if len(self.top_five) != len(set(self.top_five)):
            raise ValueError("retrieval top-five identities must be unique")
        return self


class EvalPredictionV2(EvalV2Model):
    schema_version: Literal["eval-projection.v2"] = EVAL_PROJECTION_V2
    case_id: CaseId
    topology: EvalTopology
    repeat_seed: Annotated[int, Field(gt=0)]
    recommendation: V3Recommendation
    claims: tuple[SemanticClaimProjection, ...] = Field(max_length=12)
    conflicts: tuple[ConflictProjection, ...] = Field(max_length=8)
    security_events: tuple[SecurityEventProjection, ...] = Field(max_length=8)
    retrieved_evidence: tuple[EvidenceIdentity, ...] = Field(max_length=32)
    retrieval_observations: tuple[RetrievalObservation, ...] = Field(
        default=(), max_length=8
    )
    first_pass_valid: bool
    received_nonempty_body: bool
    corrected_valid: bool
    pipeline_failure: bool
    injected_fault: bool = False
    latency_ms: Annotated[int, Field(ge=0, le=600_000)] | None
    model_request_count: Annotated[int, Field(ge=0, le=2)]
    forbidden_events: tuple[ForbiddenEvent, ...] = Field(max_length=5)
    forbidden_event_exposures: tuple[ForbiddenEvent, ...] = Field(
        default=(), max_length=5
    )

    @model_validator(mode="after")
    def projection_is_unique_and_closed(self) -> Self:
        obligation_ids = tuple(item.obligation_id for item in self.claims)
        if len(obligation_ids) != len(set(obligation_ids)):
            raise ValueError("projection may contain one claim per obligation")
        if len(self.conflicts) != len({item.pair_id for item in self.conflicts}):
            raise ValueError("projected conflict pairs must be unique")
        events = tuple(
            (item.event_type, item.source_id, item.location)
            for item in self.security_events
        )
        if len(events) != len(set(events)):
            raise ValueError("projected security events must be unique")
        if self.first_pass_valid and not self.received_nonempty_body:
            raise ValueError("first-pass validity requires a non-empty response body")
        if self.corrected_valid and not self.received_nonempty_body:
            raise ValueError("corrected validity requires a response body")
        if self.model_request_count == 0 and (
            self.first_pass_valid or self.received_nonempty_body or self.corrected_valid
        ):
            raise ValueError("zero-request projection cannot claim a model response")
        if not set(self.forbidden_events).issubset(self.forbidden_event_exposures):
            raise ValueError("forbidden events require a registered exposure")
        retrieval_keys = tuple(
            (item.query_variant, item.evidence_domain)
            for item in self.retrieval_observations
        )
        if len(retrieval_keys) != len(set(retrieval_keys)):
            raise ValueError("retrieval slices must be unique per variant and domain")
        return self


class AttemptReceiptV1(EvalV2Model):
    schema_version: Literal["eval-attempt-receipt.v1"] = "eval-attempt-receipt.v1"
    task_key: TaskKey
    attempt_no: Annotated[int, Field(gt=0)]
    outcome: Literal["succeeded", "failed", "cancelled", "remote_unknown"]
    request_sha256: HexSha256
    requested_model_id: Annotated[str, Field(min_length=1, max_length=160)] | None = (
        None
    )
    resolved_model_id: Annotated[str, Field(min_length=1, max_length=160)] | None = None
    http_status: Annotated[int, Field(ge=100, le=599)] | None = None
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    request_bytes: Annotated[int, Field(ge=0)] | None = None
    response_bytes: Annotated[int, Field(ge=0)] | None = None
    received_nonempty_body: bool | None = None
    finish_reason: Annotated[str, Field(min_length=1, max_length=96)] | None = None
    format_fields: tuple[
        Literal[
            "schema_version",
            "findings",
            "obligation_id",
            "status",
            "assertion",
            "risk_level",
            "normalized_value",
            "statement",
            "recommended_action",
            "evidence_refs",
            "missing_information",
            "witnesses",
            "field",
            "value",
            "source_ref",
            "start",
            "end",
        ],
        ...,
    ] = Field(default=(), max_length=17)
    underlying_error_code: ReasonCode | None = None
    reasoning_tokens: Annotated[int, Field(ge=0)] | None = None
    reasoning_bytes: Annotated[int, Field(ge=0)] | None = None
    format_diagnostic: (
        Literal[
            "duplicate_key",
            "invalid_number",
            "markdown_fence",
            "json_syntax",
            "non_json_text",
            "valid_json",
        ]
        | None
    ) = None
    #: Bounded location tokens of one refused answer (FIX-11 style vocabulary): the slot
    #: and index, the authorized duty, the validation category, the expected word, the
    #: basic JSON type actually found and a bounded size.  Never a raw value or prose.
    format_diagnostics: tuple[DiagnosticToken, ...] = Field(default=(), max_length=12)
    retry_reason: ReasonCode | None = None
    retry_not_before: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    usage_unknown: bool
    error_code: ReasonCode | None = None
    fault_proof: FaultTriggerProof | None = None

    @model_validator(mode="after")
    def unknown_usage_is_never_zero_filled(self) -> Self:
        tokens = (self.input_tokens, self.output_tokens)
        if self.usage_unknown:
            if any(value is not None for value in tokens):
                raise ValueError("unknown usage must remain null")
        elif any(value is None for value in tokens):
            raise ValueError("known usage requires both token values")
        if self.outcome == "remote_unknown" and not self.usage_unknown:
            raise ValueError("unknown remote result must retain unknown usage")
        if self.outcome == "succeeded" and self.error_code is not None:
            raise ValueError("successful receipt cannot carry an error code")
        if self.outcome != "succeeded" and self.error_code is None:
            raise ValueError("non-successful receipt requires an error code")
        timestamps = (self.started_at, self.completed_at)
        if (
            self.retry_not_before is not None
            and self.retry_not_before.utcoffset() is None
        ):
            raise ValueError("retry timestamp requires an offset")
        if any(item is not None and item.utcoffset() is None for item in timestamps):
            raise ValueError("receipt timestamps require an offset")
        if all(item is not None for item in timestamps):
            assert self.started_at is not None and self.completed_at is not None
            if self.completed_at < self.started_at:
                raise ValueError("receipt completion precedes its start")
        return self


class BudgetSnapshotV1(EvalV2Model):
    max_requests: Annotated[int, Field(gt=0)]
    max_input_tokens: Annotated[int, Field(gt=0)] | None
    max_output_tokens: Annotated[int, Field(gt=0)] | None
    reserved_requests: Annotated[int, Field(ge=0)]
    reserved_input_tokens: Annotated[int, Field(ge=0)]
    reserved_output_tokens: Annotated[int, Field(ge=0)]
    known_input_tokens: Annotated[int, Field(ge=0)]
    known_output_tokens: Annotated[int, Field(ge=0)]
    unknown_usage_attempts: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def reservations_stay_within_brakes(self) -> Self:
        if self.reserved_requests > self.max_requests:
            raise ValueError("request reservation exceeds budget")
        if (
            self.max_input_tokens is not None
            and self.reserved_input_tokens > self.max_input_tokens
        ):
            raise ValueError("input reservation exceeds budget")
        if (
            self.max_output_tokens is not None
            and self.reserved_output_tokens > self.max_output_tokens
        ):
            raise ValueError("output reservation exceeds budget")
        return self


class EvalRunManifestV2(EvalV2Model):
    schema_version: Literal["eval-run-manifest.v2"] = EVAL_MANIFEST_V2
    ledger_version: Literal["eval-attempt-ledger.v1"] = EVAL_LEDGER_V1
    projection_version: Literal["eval-projection.v2"] = EVAL_PROJECTION_V2
    run_id: RunId
    work_item_id: Literal["V1-07"] = "V1-07"
    stage_id: Literal["V1-07B", "V1-07C", "V1-07F", "V1-07G"] = "V1-07B"
    gold_set_version: Literal[
        "gold-set.v3",
        "gold-set.v3.1",
        "gold-set.v3.2",
        "gold-set.v3.3",
        "gold-set.v3.4",
        "gold-set.v3.5",
        "gold-set.v3.6",
    ] = "gold-set.v3"
    truth_contract: Literal[
        "v1-07-truth.v1",
        "v1-07-truth.v1.1",
        "v1-07-truth.v1.2",
        "v1-07-truth.v1.3",
        "v1-07-truth.v1.4",
        "v1-07-truth.v1.5",
        "v1-07-truth.v1.6",
    ] = "v1-07-truth.v1"
    metrics_version: Literal["v1-07-eval-metrics.v1"] = EVAL_METRICS_V1
    gold_manifest_sha256: HexSha256
    configuration_sha256: HexSha256
    created_at: datetime
    task_count: Annotated[int, Field(ge=0)]
    committed_task_count: Annotated[int, Field(ge=0)]
    blocked_unknown_task_count: Annotated[int, Field(ge=0)]
    attempt_count: Annotated[int, Field(ge=0)]
    budget: BudgetSnapshotV1

    @model_validator(mode="after")
    def task_counts_close(self) -> Self:
        expected_truth = {
            "gold-set.v3": "v1-07-truth.v1",
            "gold-set.v3.1": "v1-07-truth.v1.1",
            "gold-set.v3.2": "v1-07-truth.v1.2",
            "gold-set.v3.3": "v1-07-truth.v1.3",
            "gold-set.v3.4": "v1-07-truth.v1.4",
            "gold-set.v3.5": "v1-07-truth.v1.5",
            "gold-set.v3.6": "v1-07-truth.v1.6",
        }[self.gold_set_version]
        if self.truth_contract != expected_truth:
            raise ValueError("incompatible data and truth revision")
        if (
            self.gold_set_version
            in {
                "gold-set.v3.2",
                "gold-set.v3.3",
                "gold-set.v3.4",
                "gold-set.v3.5",
                "gold-set.v3.6",
            }
            and self.stage_id != "V1-07F"
        ):
            raise ValueError("replacement validation is registered only for F")
        if (
            self.committed_task_count + self.blocked_unknown_task_count
            > self.task_count
        ):
            raise ValueError("manifest task terminal counts exceed task count")
        if self.created_at.utcoffset() is None:
            raise ValueError("manifest timestamp requires an offset")
        return self


def _assert_no_secret(value: str) -> None:
    if any(pattern.search(value) for pattern in _SECRET_PATTERNS):
        raise ValueError("semantic projection contains a secret-shaped value")


def _assert_no_secret_tree(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_secret_tree(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _assert_no_secret_tree(item)
    elif isinstance(value, str):
        _assert_no_secret(value)


__all__ = [
    "AttemptReceiptV1",
    "AttemptState",
    "BudgetSnapshotV1",
    "ConflictProjection",
    "EVAL_LEDGER_V1",
    "EVAL_MANIFEST_V2",
    "EVAL_METRICS_V1",
    "EVAL_PROJECTION_V2",
    "EvalPredictionV2",
    "EvalRunManifestV2",
    "EvalTopology",
    "FaultTriggerProof",
    "ForbiddenEvent",
    "RetrievalObservation",
    "RetrievalQueryVariant",
    "SecurityEventProjection",
    "SemanticClaimProjection",
    "TaskState",
]
