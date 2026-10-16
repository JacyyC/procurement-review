"""Versioned transport contracts for the local Review API."""

from __future__ import annotations

from enum import StrEnum
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .execution_profile import execution_mode_value
from .contracts import (
    ApprovalActionType,
    ApprovalAuditEvent,
    BusinessFact,
    CaseType,
    Checksum,
    ConfirmationReceipt,
    EvidenceRef,
    HumanDecision,
    Identifier,
    RecommendationStatus,
    ReviewLifecycleStatus,
    ReviewRequest,
    ReviewSummary,
    SimulationStatus,
    WorkerKind,
)
from .audit_contracts import ApprovalAuditEventV2
from .precise_run_contracts import ExecutionPayload
from .trace import TraceEnvelope

API_VERSION = "review-api.v1"
API_VERSION_V2 = "review-api.v2"
EXECUTION_MODE = "test_only"

#: Responses narrate the profile the process actually runs under: a real-mode workbench
#: must never be answered with a synthetic-mode label.  A request model never carries it.
ExecutionMode = Literal["test_only", "real"]
DEFAULT_DOCUMENT_VERSION = "retrieval-fixture-v1"
DEFAULT_DATASET_VERSION = "synthetic-procurement.v1"
DEFAULT_INDEX_VERSION = "retrieval-index-v1"
DEFAULT_PURCHASE_ORDER_ID = "PO-001"
DEFAULT_INITIATOR = "reviewer-demo"


class ReviewScenario(StrEnum):
    """Predeclared local cases used to exercise success and failure paths."""

    LOW = "low"
    HIGH = "high"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"
    PARTIAL_FAILURE = "partial_failure"
    ALL_FAILURE = "all_failure"
    VERSION_MISMATCH = "version_mismatch"
    CANCELLABLE = "cancellable"
    RUNTIME_FAILURE = "runtime_failure"


class APIContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
        populate_by_name=True,
    )

    api_version: Literal["review-api.v1"] = API_VERSION


class ReviewCreateRequest(APIContract):
    """Create-and-execute input; all defaults point to local synthetic fixtures."""

    case_id: Identifier
    case_type: CaseType = CaseType.PURCHASE_ORDER
    document_version: Identifier = DEFAULT_DOCUMENT_VERSION
    purchase_order_id: Identifier = DEFAULT_PURCHASE_ORDER_ID
    dataset_version: Identifier = DEFAULT_DATASET_VERSION
    initiator: Identifier = DEFAULT_INITIATOR
    idempotency_key: Identifier | None = Field(
        default=None,
        validation_alias=AliasChoices("idempotency_key", "request_id"),
    )
    scenario: ReviewScenario = ReviewScenario.LOW
    index_version: Identifier = DEFAULT_INDEX_VERSION
    policy_query: Annotated[str, Field(min_length=1, max_length=2_000)] = (
        "competitive bidding threshold"
    )
    contract_query: Annotated[str, Field(min_length=1, max_length=2_000)] = (
        "payment terms termination rights"
    )
    policy_model_fixture_id: Identifier | None = None
    contract_model_fixture_id: Identifier | None = None
    procurement_model_fixture_id: Identifier | None = None

    @field_validator("policy_query", "contract_query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        return " ".join(value.split())

    def domain_request(self) -> ReviewRequest:
        return ReviewRequest(
            case_id=self.case_id,
            case_type=self.case_type,
            document_version=self.document_version,
            purchase_order_id=self.purchase_order_id,
            dataset_version=self.dataset_version,
            initiator=self.initiator,
        )


class HumanActionRequest(APIContract):
    """Explicit human action body for confirm, reject, or cancel endpoints."""

    human_actor: Identifier = Field(
        validation_alias=AliasChoices("human_actor", "actor"),
    )
    idempotency_key: Identifier | None = Field(
        default=None,
        validation_alias=AliasChoices("idempotency_key", "request_id"),
    )
    expected_version: Annotated[int, Field(strict=True, ge=1)] | None = Field(
        default=None,
        validation_alias=AliasChoices("expected_version", "version", "revision"),
    )


class BranchFailure(APIContract):
    worker: WorkerKind
    code: Identifier
    detail: Annotated[str, Field(min_length=1, max_length=500)]


class ReviewRunResponse(APIContract):
    """Stable API view combining a run, its suggestion, and local metadata."""

    run_id: Identifier
    trace_id: Identifier
    status: ReviewLifecycleStatus
    version: Annotated[int, Field(strict=True, ge=1)]
    revision: Annotated[int, Field(strict=True, ge=1)]
    execution_mode: ExecutionMode = Field(default_factory=execution_mode_value)
    request: ReviewRequest
    summary: ReviewSummary | None = None
    summary_checksum: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] | None = (
        None
    )
    execution: ExecutionPayload | None = None
    branch_failures: tuple[BranchFailure, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    business_facts: tuple[BusinessFact, ...] = ()
    receipt: ConfirmationReceipt | None = None
    created_at: datetime
    updated_at: datetime
    trace_event_count: Annotated[int, Field(ge=0)] = 0
    audit_event_count: Annotated[int, Field(ge=0)] = 0
    error_code: Identifier | None = None
    error_message: Annotated[str, Field(min_length=1, max_length=500)] | None = None

    @model_validator(mode="after")
    def versions_and_terminal_fields_are_consistent(self) -> ReviewRunResponse:
        if self.version != self.revision:
            raise ValueError("version and revision must match")
        if (
            self.status
            in {
                ReviewLifecycleStatus.CONFIRMED,
                ReviewLifecycleStatus.REJECTED,
            }
            and self.receipt is None
        ):
            raise ValueError("confirmed and rejected runs require a receipt")
        if (
            self.status
            not in {
                ReviewLifecycleStatus.CONFIRMED,
                ReviewLifecycleStatus.REJECTED,
            }
            and self.receipt is not None
        ):
            raise ValueError("only confirmed or rejected runs may expose a receipt")
        if self.status is ReviewLifecycleStatus.FAILED:
            if self.error_code is None or self.error_message is None:
                raise ValueError("failed runs require an error contract")
        elif self.error_code is not None or self.error_message is not None:
            raise ValueError("only failed runs may expose a run error")
        if self.summary is not None:
            if (
                self.summary.run_id != self.run_id
                or self.summary.case_id != self.request.case_id
            ):
                raise ValueError("summary identity must match the review response")
            if self.summary_checksum is None:
                raise ValueError("a review response with a summary needs its checksum")
        elif self.summary_checksum is not None:
            raise ValueError("only reviews with a summary may expose its checksum")
        if self.execution is not None:
            if (
                self.execution.summary.run_id != self.run_id
                or self.execution.summary.case_id != self.request.case_id
            ):
                raise ValueError("execution identity must match the review response")
        if self.receipt is not None:
            if (
                self.receipt.run_id != self.run_id
                or self.receipt.case_id != self.request.case_id
            ):
                raise ValueError("receipt identity must match the review response")
        return self


class ReviewListResponse(APIContract):
    runs: tuple[ReviewRunResponse, ...] = ()


class TraceResponse(APIContract):
    trace: TraceEnvelope


class AuditResponse(APIContract):
    audit_schema_version: Literal["approval-audit.v1"] = "approval-audit.v1"
    run_id: Identifier
    events: tuple[ApprovalAuditEvent, ...] = ()


class CaseOption(APIContract):
    case_id: Identifier
    label: Annotated[str, Field(min_length=1, max_length=200)]
    case_type: CaseType
    purchase_order_id: Identifier
    scenario: ReviewScenario
    description: Annotated[str, Field(min_length=1, max_length=500)]


class CaseListResponse(APIContract):
    cases: tuple[CaseOption, ...]
    dataset_version: Identifier = DEFAULT_DATASET_VERSION
    document_version: Identifier = DEFAULT_DOCUMENT_VERSION
    index_version: Identifier = DEFAULT_INDEX_VERSION
    execution_mode: ExecutionMode = Field(default_factory=execution_mode_value)


class HealthResponse(APIContract):
    service: Literal["review-api"] = "review-api"
    status: Literal["ok"] = "ok"
    execution_mode: ExecutionMode = Field(default_factory=execution_mode_value)
    api_version: Literal["review-api.v1"] = API_VERSION


class ErrorBody(BaseModel):
    """Stable error body; it deliberately carries no API version or input echo."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    code: Identifier
    message: Annotated[str, Field(min_length=1, max_length=500)]
    request_id: Identifier
    details: dict[str, object] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    error: ErrorBody


# --------------------------------------------------------------------------- v2
#
# `review-api.v2` is the authenticated product transport.  It carries no client
# supplied authority field: identity, tenant, Case input versions and the effective
# scope are all assembled on the server from the verified credential and the Case
# registry.  The shared domain, rule and evaluation contracts are unchanged.


class APIContractV2(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
        populate_by_name=True,
    )

    api_version: Literal["review-api.v2"] = API_VERSION_V2


class ReviewCreateRequestV2(APIContractV2):
    """Authenticated create input: one granted Case and its pinned version."""

    case_id: Identifier
    case_version: Identifier
    idempotency_key: Identifier | None = Field(
        default=None,
        validation_alias=AliasChoices("idempotency_key", "request_id"),
    )


class HumanActionRequestV2(APIContractV2):
    """Explicit human action; the actor comes from the verified credential."""

    idempotency_key: Identifier | None = Field(
        default=None,
        validation_alias=AliasChoices("idempotency_key", "request_id"),
    )
    expected_version: Annotated[int, Field(strict=True, ge=1)] | None = Field(
        default=None,
        validation_alias=AliasChoices("expected_version", "version", "revision"),
    )
    expected_summary_checksum: (
        Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")] | None
    ) = None


class ResumeRequestV2(APIContractV2):
    """Explicit resume input for one interrupted Run."""

    idempotency_key: Identifier | None = Field(
        default=None,
        validation_alias=AliasChoices("idempotency_key", "request_id"),
    )
    expected_version: Annotated[int, Field(strict=True, ge=1)] | None = None


class SessionResponse(APIContractV2):
    """The caller's own verified identity; never a credential or a digest."""

    subject_id: Identifier
    tenant_id: Identifier
    credential_id: Identifier
    authorization_revision: Annotated[int, Field(strict=True, ge=1)]
    roles: tuple[str, ...]
    scopes: tuple[str, ...]
    issued_at: datetime
    expires_at: datetime


class CaseOptionV2(APIContractV2):
    case_id: Identifier
    case_version: Identifier
    label: Annotated[str, Field(min_length=1, max_length=200)]
    case_type: CaseType
    scenario: ReviewScenario
    document_version: Identifier
    dataset_version: Identifier
    index_version: Identifier
    rules_version: Identifier


class CaseListResponseV2(APIContractV2):
    cases: tuple[CaseOptionV2, ...]
    execution_mode: ExecutionMode = Field(default_factory=execution_mode_value)


class HealthResponseV2(APIContractV2):
    service: Literal["review-api"] = "review-api"
    status: Literal["ok"] = "ok"
    execution_mode: ExecutionMode = Field(default_factory=execution_mode_value)
    security_store: Literal["ready", "not_configured"] = "not_configured"


class ReceiptViewV2(APIContractV2):
    """Product v2 projection of a local simulated receipt.

    The raw idempotency key stays a server-side record; the transport exposes only its
    digest, so a client cannot replay or correlate another caller's key.
    """

    receipt_id: Identifier
    run_id: Identifier
    case_id: Identifier
    human_actor: Identifier
    decision: HumanDecision
    reviewed_recommendation: RecommendationStatus
    recorded_at: datetime
    idempotency_key_digest: Checksum
    simulation_status: SimulationStatus = SimulationStatus.LOCAL_SIMULATION_RECORDED
    simulation_mode: Literal["local_test_only"] = "local_test_only"
    external_write_performed: Literal[False] = False


class AuditEventViewV2(APIContractV2):
    """Product v2 projection of one approval-audit event (raw keys removed)."""

    audit_schema_version: Literal["approval-audit.v1"] = "approval-audit.v1"
    audit_id: Identifier
    run_id: Identifier
    case_id: Identifier
    object_type: Literal["review"] = "review"
    object_id: Identifier
    action_type: ApprovalActionType
    actor: Identifier
    occurred_at: datetime
    # The transport reports the API version that actually recorded the action.  The
    # frozen ``approval-audit.v1`` domain record keeps its own historical literal; the
    # persisted ``approval-audit.v2`` record contract arrives with V1-03B.
    api_version: Literal["review-api.v2"] = API_VERSION_V2
    idempotency_key_digest: Checksum
    previous_status: ReviewLifecycleStatus
    new_status: ReviewLifecycleStatus
    receipt_id: Identifier | None = None
    simulation_only: Literal[True] = True
    external_write_performed: Literal[False] = False


class RecoveryView(BaseModel):
    """Recovery facts the UI needs to decide whether it may offer a resume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resume_count: Annotated[int, Field(strict=True, ge=0)] = 0
    resume_limit: Annotated[int, Field(strict=True, ge=0)] = 0
    resumable: bool = False
    generation: Annotated[int, Field(strict=True, ge=0)] = 0
    lease_expires_at: datetime | None = None
    adopted_workers: tuple[Identifier, ...] = ()
    attempt_totals: dict[str, int] = Field(default_factory=dict)
    interrupted_reason: Identifier | None = None


class ReviewRunResponseV2(BaseModel):
    """Authenticated transport view of one Run, with the v2 receipt projection."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    run_id: Identifier
    trace_id: Identifier
    status: ReviewLifecycleStatus
    version: Annotated[int, Field(strict=True, ge=1)]
    revision: Annotated[int, Field(strict=True, ge=1)]
    execution_mode: ExecutionMode = Field(default_factory=execution_mode_value)
    request: ReviewRequest
    summary: ReviewSummary | None = None
    summary_checksum: Checksum | None = None
    execution: ExecutionPayload | None = None
    branch_failures: tuple[BranchFailure, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    business_facts: tuple[BusinessFact, ...] = ()
    receipt: ReceiptViewV2 | None = None
    created_at: datetime
    updated_at: datetime
    trace_event_count: Annotated[int, Field(ge=0)] = 0
    audit_event_count: Annotated[int, Field(ge=0)] = 0
    error_code: Identifier | None = None
    error_message: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    recovery: RecoveryView | None = None


class ReviewListResponseV2(APIContractV2):
    runs: tuple[ReviewRunResponseV2, ...] = ()


class AuditResponseV2(APIContractV2):
    audit_schema_version: Literal["approval-audit.v2"] = "approval-audit.v2"
    run_id: Identifier
    events: tuple[ApprovalAuditEventV2, ...] = ()
    # The sequence every event carries is the one its committed row holds.  Events
    # committed before the database allocator existed have their committed number
    # projected here and are named in the anomaly list; the stored payload is history and
    # is never rewritten.  A reader can therefore tell corrected history from consistent
    # records without guessing.
    sequence_projection_version: Literal["approval-audit.sequence-projection.v1"] = (
        "approval-audit.sequence-projection.v1"
    )
    legacy_sequence_anomalies: tuple[Identifier, ...] = ()
