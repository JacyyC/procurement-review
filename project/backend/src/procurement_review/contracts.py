"""Strict, versioned domain contracts for procurement review.

These contracts describe evidence and recommendations.  They deliberately do
not contain an approval action or production credential field.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
import json
import re
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

CONTRACT_SCHEMA_VERSION = "1.0"
Checksum = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Identifier = Annotated[
    str,
    Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"),
]
NonEmptyText = Annotated[str, Field(min_length=1, max_length=4_000)]


class StrictContract(BaseModel):
    """Common validation policy for all public contracts."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
        # An instance handed to a parent is revalidated, so a value that was changed after
        # construction cannot travel inside an otherwise legal record (AC-04.1).
        revalidate_instances="always",
    )

    schema_version: Literal["1.0"] = CONTRACT_SCHEMA_VERSION


class CaseType(str, Enum):
    PURCHASE_ORDER = "purchase_order"
    CONTRACT_CHANGE = "contract_change"


class SourceKind(str, Enum):
    PUBLIC_DOCUMENT = "public_document"
    CASE_DOCUMENT = "case_document"


class FactStatus(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    UPSTREAM_FAILURE = "upstream_failure"


class WorkerKind(str, Enum):
    POLICY = "policy"
    CONTRACT = "contract"
    PROCUREMENT_DATA = "procurement_data"


class RiskCategory(str, Enum):
    POLICY_COMPLIANCE = "policy_compliance"
    CONTRACT_TERMS = "contract_terms"
    PROCUREMENT_ANOMALY = "procurement_anomaly"
    SUPPLIER_INTEGRITY = "supplier_integrity"
    FINANCIAL_EXPOSURE = "financial_exposure"
    DATA_QUALITY = "data_quality"


class RiskLevel(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingAssertion(str, Enum):
    RISK_PRESENT = "risk_present"
    RISK_ABSENT = "risk_absent"
    UNKNOWN = "unknown"


class FindingStatus(str, Enum):
    SUPPORTED = "supported"
    UNKNOWN = "unknown"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UPSTREAM_FAILURE = "upstream_failure"
    INVALID_INPUT = "invalid_input"


class RecommendationStatus(str, Enum):
    RECOMMEND_APPROVE = "recommend_approve"
    RECOMMEND_CONDITIONAL = "recommend_conditional"
    RECOMMEND_REJECT = "recommend_reject"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class HumanReviewStatus(str, Enum):
    PENDING_CONFIRMATION = "pending_confirmation"
    REQUIRED = "required"
    COMPLETED = "completed"


class ReviewLifecycleStatus(str, Enum):
    """Product-level Run lifecycle, kept separate from recommendation semantics."""

    PROCESSING = "processing"
    PENDING_CONFIRMATION = "pending_confirmation"
    PENDING_HUMAN_REVIEW = "pending_human_review"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    FAILED = "failed"
    # V1-03 product state: the execution stopped without a committed result and needs an
    # explicit, authorized resume.  It is not a recommendation and carries no receipt.
    INTERRUPTED = "interrupted"

    # A readable compatibility alias for callers that use the shorter wording.
    PENDING_HUMAN = PENDING_HUMAN_REVIEW


class ApprovalActionType(str, Enum):
    """Explicit local human actions recorded by the approval audit."""

    CONFIRM = "confirm"
    REJECT = "reject"
    CANCEL = "cancel"


class ConflictReason(str, Enum):
    OPPOSING_ASSERTIONS = "opposing_assertions"
    CATEGORY_MISMATCH = "category_mismatch"


# Rules 2.0 records every cause it generates under one of these namespaces so the three
# R-2 outcomes stay mechanically separable. `coverage` is a health signal and never gates a
# recommendation; `pipeline` and `evidence` do, and only in the direction rules v2 declares.
PIPELINE_CAUSE_PREFIX = "pipeline:"
EVIDENCE_CAUSE_PREFIX = "evidence:"
COVERAGE_CAUSE_PREFIX = "coverage:"


class CauseClass(str, Enum):
    PIPELINE = "pipeline"
    EVIDENCE = "evidence"
    COVERAGE = "coverage"
    CONTENT_NOTE = "content_note"


# A model may author any text in ``missing_information``, so a namespace only counts when the
# whole string has the shape these rules emit: no spaces, identifier characters only.
_NAMESPACED_CAUSE_PATTERN = re.compile(
    rf"^(?:{PIPELINE_CAUSE_PREFIX}|{EVIDENCE_CAUSE_PREFIX}|{COVERAGE_CAUSE_PREFIX})"
    r"[A-Za-z0-9][A-Za-z0-9_.:/\[\]-]*$"
)
_CAUSE_CLASS_BY_PREFIX = {
    PIPELINE_CAUSE_PREFIX: CauseClass.PIPELINE,
    EVIDENCE_CAUSE_PREFIX: CauseClass.EVIDENCE,
    COVERAGE_CAUSE_PREFIX: CauseClass.COVERAGE,
}


def namespaced_cause_class(code: str) -> CauseClass | None:
    """The registered cause class of a namespaced code, or ``None`` when it is prose."""

    if not _NAMESPACED_CAUSE_PATTERN.fullmatch(code):
        return None
    for prefix, cause_class in _CAUSE_CLASS_BY_PREFIX.items():
        if code.startswith(prefix):
            return cause_class
    return None


class EscalationClass(str, Enum):
    """The mutually exclusive outcome a summary escalated for, per R-2."""

    CONTENT = "content"
    PIPELINE_INCOMPLETE = "pipeline_incomplete"
    CONFLICT = "conflict"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"


class HumanDecision(str, Enum):
    CONFIRM_RECOMMENDATION = "confirm_recommendation"
    REJECT_RECOMMENDATION = "reject_recommendation"


class SimulationStatus(str, Enum):
    LOCAL_SIMULATION_RECORDED = "local_simulation_recorded"


class RunEventType(str, Enum):
    RUN_STARTED = "run_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_CANCELLED = "step_cancelled"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    CONFIRMATION_RECORDED = "confirmation_recorded"
    HUMAN_ACTION_RECORDED = "human_action_recorded"


class RedactionStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    REDACTED = "redacted"


class ReviewRequest(StrictContract):
    case_id: Identifier
    case_type: CaseType
    document_version: Identifier
    purchase_order_id: Identifier
    dataset_version: Identifier
    initiator: Identifier


class EvidenceRef(StrictContract):
    source_kind: SourceKind
    source_id: Identifier
    version: Identifier
    page_number: Annotated[int, Field(ge=1)] | None = None
    section: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    chunk_id: Identifier | None = None
    content_checksum: Checksum

    @model_validator(mode="after")
    def require_stable_locator(self) -> EvidenceRef:
        if self.page_number is None and self.section is None and self.chunk_id is None:
            raise ValueError("evidence must include page_number, section, or chunk_id")
        return self


class BusinessFact(StrictContract):
    tool_name: Identifier
    record_id: Identifier
    dataset_version: Identifier
    as_of: datetime
    status: FactStatus
    result: dict[str, JsonValue] = Field(default_factory=dict)
    failure_reason: NonEmptyText | None = None
    read_only: Literal[True] = True

    @field_validator("as_of")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        return value

    @model_validator(mode="after")
    def result_matches_status(self) -> BusinessFact:
        if self.status is FactStatus.KNOWN:
            if not self.result:
                raise ValueError("known facts require a non-empty result")
            if self.failure_reason is not None:
                raise ValueError("known facts cannot carry a failure_reason")
        else:
            if self.result:
                raise ValueError("unknown or failed facts cannot carry a result")
            if self.failure_reason is None:
                raise ValueError("unknown or failed facts require a failure_reason")
        return self


class Finding(StrictContract):
    worker: WorkerKind
    finding_key: Identifier
    category: RiskCategory
    risk_level: RiskLevel
    assertion: FindingAssertion
    statement: NonEmptyText
    status: FindingStatus
    evidence_refs: tuple[EvidenceRef, ...] = ()
    business_fact_refs: tuple[BusinessFact, ...] = ()
    missing_information: tuple[NonEmptyText, ...] = ()
    recommended_action: NonEmptyText

    @model_validator(mode="after")
    def enforce_evidence_semantics(self) -> Finding:
        if self.status is FindingStatus.SUPPORTED:
            if self.assertion is FindingAssertion.UNKNOWN:
                raise ValueError("supported findings require a decidable assertion")
            if self.risk_level is RiskLevel.UNKNOWN:
                raise ValueError("supported findings require a known risk level")
            if not self.evidence_refs and not self.business_fact_refs:
                raise ValueError(
                    "supported findings require evidence or a business fact"
                )
            if any(
                fact.status is not FactStatus.KNOWN for fact in self.business_fact_refs
            ):
                raise ValueError(
                    "supported findings may reference only known business facts"
                )
        else:
            if self.assertion is not FindingAssertion.UNKNOWN:
                raise ValueError("unresolved findings must use the UNKNOWN assertion")
            if self.risk_level is not RiskLevel.UNKNOWN:
                raise ValueError("unresolved findings must use the UNKNOWN risk level")
            if not self.missing_information:
                raise ValueError(
                    "unresolved findings must identify missing information"
                )
        return self


class FindingConflict(StrictContract):
    finding_key: Identifier
    reason: ConflictReason
    assertions: tuple[FindingAssertion, ...]
    categories: tuple[RiskCategory, ...]
    statements: tuple[NonEmptyText, ...]

    @model_validator(mode="after")
    def require_conflicting_values(self) -> FindingConflict:
        if self.reason is ConflictReason.OPPOSING_ASSERTIONS:
            required = {FindingAssertion.RISK_PRESENT, FindingAssertion.RISK_ABSENT}
            if not required.issubset(self.assertions):
                raise ValueError(
                    "opposing assertion conflict requires present and absent"
                )
        elif len(set(self.categories)) < 2:
            raise ValueError("category mismatch requires at least two categories")
        return self


class ReviewSummary(StrictContract):
    run_id: Identifier
    case_id: Identifier
    rules_version: Identifier
    findings: tuple[Finding, ...]
    highest_risk: RiskLevel
    recommendation: RecommendationStatus
    human_review_status: HumanReviewStatus
    conflicts: tuple[FindingConflict, ...] = ()
    missing_information: tuple[NonEmptyText, ...] = ()
    failure_reasons: tuple[NonEmptyText, ...] = ()
    is_final_decision: Literal[False] = False

    @model_validator(mode="after")
    def recommendation_is_fail_closed(self) -> ReviewSummary:
        """Enforce the gate semantics of the rules version this summary declares."""

        major = self.rules_version.split(".", maxsplit=1)[0]
        if major == "1":
            return self._enforce_rules_v1_gate_semantics()
        if major == "2":
            return self._enforce_rules_v2_gate_semantics()
        if major == "3":
            return self._enforce_rules_v3_gate_semantics()
        raise ValueError("summary rules_version is not a known aggregation contract")

    def _enforce_rules_v1_gate_semantics(self) -> ReviewSummary:
        has_unresolved_findings = any(
            finding.status is not FindingStatus.SUPPORTED for finding in self.findings
        )
        if has_unresolved_findings and not self.missing_information:
            raise ValueError(
                "unresolved findings must be surfaced as missing information"
            )
        unresolved = (
            bool(self.missing_information or self.failure_reasons)
            or has_unresolved_findings
        )
        if self.conflicts:
            if self.recommendation is not RecommendationStatus.NEEDS_HUMAN_REVIEW:
                raise ValueError("conflicts require needs_human_review")
            if self.human_review_status is not HumanReviewStatus.REQUIRED:
                raise ValueError("conflicts require human review")
        elif unresolved:
            if self.recommendation is not RecommendationStatus.INSUFFICIENT_INFORMATION:
                raise ValueError("gaps or failures require insufficient_information")
        elif self.recommendation in {
            RecommendationStatus.INSUFFICIENT_INFORMATION,
            RecommendationStatus.NEEDS_HUMAN_REVIEW,
        }:
            raise ValueError("escalated recommendations require a recorded cause")
        if self.recommendation in {
            RecommendationStatus.RECOMMEND_APPROVE,
            RecommendationStatus.RECOMMEND_CONDITIONAL,
            RecommendationStatus.RECOMMEND_REJECT,
        }:
            if (
                not self.findings
                or unresolved
                or self.conflicts
                or self.highest_risk is RiskLevel.UNKNOWN
            ):
                raise ValueError(
                    "ordinary recommendations require complete, decidable input"
                )
            if self.human_review_status is not HumanReviewStatus.PENDING_CONFIRMATION:
                raise ValueError(
                    "ordinary recommendations remain pending human confirmation"
                )
        elif self.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION:
            if not unresolved:
                raise ValueError(
                    "insufficient_information requires a recorded gap or failure"
                )
            if self.human_review_status is not HumanReviewStatus.REQUIRED:
                raise ValueError("insufficient information requires human review")
        if self.human_review_status is HumanReviewStatus.COMPLETED:
            raise ValueError("a summary cannot claim that human review is completed")
        return self

    def _enforce_rules_v2_gate_semantics(self) -> ReviewSummary:
        """Rules 2.0 keeps the fail-closed floor but splits gaps by their direction.

        A supported high risk may not be erased by another worker's inconclusive branch,
        while an approval still requires every dimension to be positively supported.
        """

        causes = (*self.missing_information, *self.failure_reasons)
        cause_classes = {code: namespaced_cause_class(code) for code in causes}
        pipeline_gap = any(
            cause_class is CauseClass.PIPELINE for cause_class in cause_classes.values()
        )
        evidence_gap = any(
            cause_class is CauseClass.EVIDENCE for cause_class in cause_classes.values()
        )
        has_unresolved_findings = any(
            finding.status is not FindingStatus.SUPPORTED for finding in self.findings
        )
        if has_unresolved_findings and not self.missing_information:
            raise ValueError(
                "unresolved findings must be surfaced as missing information"
            )
        if self.human_review_status is HumanReviewStatus.COMPLETED:
            raise ValueError("a summary cannot claim that human review is completed")
        if self.conflicts:
            if self.recommendation is not RecommendationStatus.NEEDS_HUMAN_REVIEW:
                raise ValueError("conflicts require needs_human_review")
            if self.human_review_status is not HumanReviewStatus.REQUIRED:
                raise ValueError("conflicts require human review")
            return self
        if self.recommendation in {
            RecommendationStatus.RECOMMEND_APPROVE,
            RecommendationStatus.RECOMMEND_CONDITIONAL,
        }:
            if (
                not self.findings
                or pipeline_gap
                or evidence_gap
                or has_unresolved_findings
                or self.highest_risk is RiskLevel.UNKNOWN
            ):
                raise ValueError(
                    "approval requires complete, decidable support without a recorded gap"
                )
            if self.human_review_status is not HumanReviewStatus.PENDING_CONFIRMATION:
                raise ValueError(
                    "ordinary recommendations remain pending human confirmation"
                )
            return self
        if self.recommendation is RecommendationStatus.RECOMMEND_REJECT:
            if (
                not self.findings
                or pipeline_gap
                or self.highest_risk is RiskLevel.UNKNOWN
            ):
                raise ValueError(
                    "a rejection requires a decidable finding and no pipeline failure"
                )
            if self.human_review_status is not HumanReviewStatus.PENDING_CONFIRMATION:
                raise ValueError(
                    "ordinary recommendations remain pending human confirmation"
                )
            return self
        if not causes:
            raise ValueError("escalated recommendations require a recorded cause")
        if self.human_review_status is not HumanReviewStatus.REQUIRED:
            raise ValueError("an escalated recommendation requires human review")
        return self

    def _enforce_rules_v3_gate_semantics(self) -> ReviewSummary:
        """Rules 3.1 keeps the fail-closed floor of the v2 aggregation.

        The conditional rule set computes its own decision; this validator only
        refuses a summary whose recorded causes do not support its recommendation,
        so a precise Run cannot commit a gate state its own findings contradict.
        """

        causes = (*self.missing_information, *self.failure_reasons)
        has_unresolved_findings = any(
            finding.status is not FindingStatus.SUPPORTED for finding in self.findings
        )
        if has_unresolved_findings and not self.missing_information:
            raise ValueError(
                "unresolved findings must be surfaced as missing information"
            )
        if self.human_review_status is HumanReviewStatus.COMPLETED:
            raise ValueError("a summary cannot claim that human review is completed")
        if self.conflicts:
            if self.recommendation is not RecommendationStatus.NEEDS_HUMAN_REVIEW:
                raise ValueError("conflicts require needs_human_review")
            if self.human_review_status is not HumanReviewStatus.REQUIRED:
                raise ValueError("conflicts require human review")
            return self
        if self.recommendation in {
            RecommendationStatus.RECOMMEND_APPROVE,
            RecommendationStatus.RECOMMEND_CONDITIONAL,
        }:
            if (
                not self.findings
                or causes
                or has_unresolved_findings
                or self.highest_risk is RiskLevel.UNKNOWN
            ):
                raise ValueError(
                    "approval requires complete, decidable support without a recorded gap"
                )
            if self.human_review_status is not HumanReviewStatus.PENDING_CONFIRMATION:
                raise ValueError(
                    "ordinary recommendations remain pending human confirmation"
                )
            return self
        if self.recommendation is RecommendationStatus.RECOMMEND_REJECT:
            if (
                not self.findings
                or self.failure_reasons
                or self.highest_risk is RiskLevel.UNKNOWN
            ):
                raise ValueError(
                    "a rejection requires a decidable finding and no pipeline failure"
                )
            if self.human_review_status is not HumanReviewStatus.PENDING_CONFIRMATION:
                raise ValueError(
                    "ordinary recommendations remain pending human confirmation"
                )
            return self
        if not causes:
            raise ValueError("escalated recommendations require a recorded cause")
        if self.human_review_status is not HumanReviewStatus.REQUIRED:
            raise ValueError("an escalated recommendation requires human review")
        return self


class ConfirmationReceipt(StrictContract):
    receipt_id: Identifier
    run_id: Identifier
    case_id: Identifier
    human_actor: Identifier
    decision: HumanDecision
    reviewed_recommendation: RecommendationStatus
    recorded_at: datetime
    idempotency_key: Identifier
    simulation_status: SimulationStatus = SimulationStatus.LOCAL_SIMULATION_RECORDED
    simulation_mode: Literal["local_test_only"] = "local_test_only"
    external_write_performed: Literal[False] = False

    @field_validator("recorded_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware")
        return value


class ApprovalAuditEvent(StrictContract):
    """A human action record; it is intentionally not a Trace event."""

    audit_schema_version: Literal["approval-audit.v1"] = "approval-audit.v1"
    audit_id: Identifier
    run_id: Identifier
    case_id: Identifier
    object_type: Literal["review"] = "review"
    object_id: Identifier
    action_type: ApprovalActionType
    actor: Identifier
    occurred_at: datetime
    api_version: Literal["review-api.v1"] = "review-api.v1"
    idempotency_key: Identifier
    previous_status: ReviewLifecycleStatus
    new_status: ReviewLifecycleStatus
    receipt_id: Identifier | None = None
    simulation_only: Literal[True] = True
    external_write_performed: Literal[False] = False

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def enforce_action_semantics(self) -> ApprovalAuditEvent:
        if self.object_id != self.run_id:
            raise ValueError("approval audit object_id must equal run_id")
        if self.previous_status is self.new_status:
            raise ValueError("approval audit action must change lifecycle status")
        if self.action_type is ApprovalActionType.CONFIRM:
            if self.previous_status not in {
                ReviewLifecycleStatus.PENDING_CONFIRMATION,
                ReviewLifecycleStatus.PENDING_HUMAN_REVIEW,
            }:
                raise ValueError("confirm action requires a pending review status")
            if self.new_status is not ReviewLifecycleStatus.CONFIRMED:
                raise ValueError("confirm action must produce confirmed status")
            if self.receipt_id is None:
                raise ValueError("confirm action must reference a receipt")
        elif self.action_type is ApprovalActionType.REJECT:
            if self.previous_status not in {
                ReviewLifecycleStatus.PENDING_CONFIRMATION,
                ReviewLifecycleStatus.PENDING_HUMAN_REVIEW,
            }:
                raise ValueError("reject action requires a pending review status")
            if self.new_status is not ReviewLifecycleStatus.REJECTED:
                raise ValueError("reject action must produce rejected status")
            if self.receipt_id is None:
                raise ValueError("reject action must reference a receipt")
        else:
            if self.previous_status not in {
                ReviewLifecycleStatus.PROCESSING,
                ReviewLifecycleStatus.PENDING_CONFIRMATION,
                ReviewLifecycleStatus.PENDING_HUMAN_REVIEW,
            }:
                raise ValueError("cancel action requires a cancellable review status")
            if self.new_status is not ReviewLifecycleStatus.CANCELLED:
                raise ValueError("cancel action must produce cancelled status")
            if self.receipt_id is not None:
                raise ValueError("cancel action cannot reference a receipt")
        return self


_SENSITIVE_KEY_PARTS = ("api_key", "credential", "password", "secret", "token")
_TRACE_CONTENT_KEY_PARTS = ("prompt", "reasoning", "chain_of_thought", "thought")
_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?i)\b(?:api[_ -]?key|password|token|secret)\s*[:=]"),
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
)


def _contains_sensitive_key(value: JsonValue) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                return True
            if _contains_sensitive_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def contains_secret_value(value: str) -> bool:
    """Return whether a string carries a credential-shaped value."""

    return any(pattern.search(value) for pattern in _SENSITIVE_VALUE_PATTERNS)


def _contains_forbidden_trace_content(value: JsonValue) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            if any(part in normalized for part in _TRACE_CONTENT_KEY_PARTS):
                return True
            if _contains_forbidden_trace_content(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_trace_content(item) for item in value)
    elif isinstance(value, str):
        return contains_secret_value(value)
    return False


class RunEvent(StrictContract):
    event_id: Identifier
    run_id: Identifier
    trace_id: Identifier
    event_type: RunEventType
    step: Identifier
    occurred_at: datetime
    input_summary: dict[str, JsonValue] = Field(default_factory=dict)
    output_summary: dict[str, JsonValue] = Field(default_factory=dict)
    input_checksum: Checksum
    output_checksum: Checksum
    source_versions: dict[str, Identifier]
    duration_ms: Annotated[int, Field(ge=0)]
    token_count: Annotated[int, Field(ge=0)] | None = None
    cost_amount: Annotated[Decimal, Field(ge=0)] | None = None
    cost_currency: (
        Annotated[str, Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")] | None
    ) = None
    redaction_status: RedactionStatus
    error: NonEmptyText | None = None

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def enforce_event_safety(self) -> RunEvent:
        if not self.source_versions:
            raise ValueError("source_versions cannot be empty")
        if (self.cost_amount is None) != (self.cost_currency is None):
            raise ValueError("cost_amount and cost_currency must be provided together")
        if (
            self.event_type
            in {
                RunEventType.STEP_FAILED,
                RunEventType.RUN_FAILED,
            }
            and self.error is None
        ):
            raise ValueError("failed events require an error summary")
        if (
            _contains_sensitive_key(self.input_summary)
            or _contains_sensitive_key(self.output_summary)
            or _contains_sensitive_key(self.source_versions)
        ):
            raise ValueError(
                "event summaries cannot contain credential or secret fields"
            )
        if (
            _contains_forbidden_trace_content(self.input_summary)
            or _contains_forbidden_trace_content(self.output_summary)
            or _contains_forbidden_trace_content(self.source_versions)
        ):
            raise ValueError(
                "event summaries cannot contain prompts, reasoning, or secret values"
            )
        if self.error is not None and _contains_forbidden_trace_content(self.error):
            raise ValueError("event errors cannot contain reasoning or secret values")
        for summary_name, summary in (
            ("input_summary", self.input_summary),
            ("output_summary", self.output_summary),
        ):
            encoded = json.dumps(
                summary,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            if len(encoded) > 8_192:
                raise ValueError(f"{summary_name} exceeds the trace size limit")
        return self
