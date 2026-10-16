from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from procurement_review.contracts import (
    ApprovalActionType,
    ApprovalAuditEvent,
    BusinessFact,
    CaseType,
    ConfirmationReceipt,
    EvidenceRef,
    FactStatus,
    Finding,
    FindingAssertion,
    FindingStatus,
    HumanDecision,
    HumanReviewStatus,
    RecommendationStatus,
    ReviewLifecycleStatus,
    RedactionStatus,
    ReviewRequest,
    ReviewSummary,
    RiskCategory,
    RiskLevel,
    RunEvent,
    RunEventType,
    SourceKind,
    WorkerKind,
)

CHECKSUM = "sha256:" + "a" * 64
NOW = datetime(2026, 8, 30, 8, 0, tzinfo=UTC)


def evidence() -> EvidenceRef:
    return EvidenceRef(
        source_kind=SourceKind.PUBLIC_DOCUMENT,
        source_id="policy-001",
        version="2026-01",
        section="Article-4",
        content_checksum=CHECKSUM,
    )


def fact() -> BusinessFact:
    return BusinessFact(
        tool_name="get_purchase_order",
        record_id="po-001",
        dataset_version="fixture-v1",
        as_of=NOW,
        status=FactStatus.KNOWN,
        result={"amount": 1000, "currency": "CNY"},
    )


def test_review_request_is_strict_versioned_and_forbids_credentials() -> None:
    request = ReviewRequest(
        case_id="case-001",
        case_type=CaseType.PURCHASE_ORDER,
        document_version="case-doc-v1",
        purchase_order_id="po-001",
        dataset_version="fixture-v1",
        initiator="reviewer-001",
    )

    assert request.schema_version == "1.0"
    with pytest.raises(ValidationError):
        ReviewRequest.model_validate(
            {
                **request.model_dump(),
                "case_type": CaseType.PURCHASE_ORDER,
                "api_key": "must-not-enter-contract",
            }
        )
    with pytest.raises(ValidationError):
        ReviewRequest(
            case_id="case-001",
            case_type="purchase_order",  # type: ignore[arg-type]
            document_version="case-doc-v1",
            purchase_order_id="po-001",
            dataset_version="fixture-v1",
            initiator="reviewer-001",
        )


def test_evidence_requires_a_stable_locator_and_checksum() -> None:
    assert evidence().section == "Article-4"
    with pytest.raises(ValidationError):
        EvidenceRef(
            source_kind=SourceKind.PUBLIC_DOCUMENT,
            source_id="policy-001",
            version="2026-01",
            content_checksum=CHECKSUM,
        )
    with pytest.raises(ValidationError):
        EvidenceRef(
            source_kind=SourceKind.PUBLIC_DOCUMENT,
            source_id="policy-001",
            version="2026-01",
            section="Article-4",
            content_checksum="not-a-checksum",
        )


def test_business_fact_is_read_only_and_explicit_about_unknown_results() -> None:
    assert fact().read_only is True
    unknown = BusinessFact(
        tool_name="get_purchase_order",
        record_id="po-missing",
        dataset_version="fixture-v1",
        as_of=NOW,
        status=FactStatus.UNKNOWN,
        failure_reason="record not found",
    )
    assert unknown.result == {}
    with pytest.raises(ValidationError):
        BusinessFact(
            tool_name="get_purchase_order",
            record_id="po-missing",
            dataset_version="fixture-v1",
            as_of=NOW,
            status=FactStatus.UNKNOWN,
            result={"invented": True},
            failure_reason="record not found",
        )
    with pytest.raises(ValidationError):
        BusinessFact(
            tool_name="get_purchase_order",
            record_id="po-001",
            dataset_version="fixture-v1",
            as_of=NOW.replace(tzinfo=None),
            status=FactStatus.KNOWN,
            result={"amount": 1000},
        )


def test_supported_finding_requires_real_evidence() -> None:
    supported = Finding(
        worker=WorkerKind.POLICY,
        finding_key="policy.required_policy_evidence",
        category=RiskCategory.POLICY_COMPLIANCE,
        risk_level=RiskLevel.LOW,
        assertion=FindingAssertion.RISK_ABSENT,
        statement="Applicable policy evidence is present.",
        status=FindingStatus.SUPPORTED,
        evidence_refs=(evidence(),),
        recommended_action="Continue review.",
    )
    assert supported.status is FindingStatus.SUPPORTED
    with pytest.raises(ValidationError):
        Finding(
            worker=WorkerKind.POLICY,
            finding_key="policy.required_policy_evidence",
            category=RiskCategory.POLICY_COMPLIANCE,
            risk_level=RiskLevel.LOW,
            assertion=FindingAssertion.RISK_ABSENT,
            statement="Unsupported claim.",
            status=FindingStatus.SUPPORTED,
            recommended_action="Continue review.",
        )


@pytest.mark.parametrize(
    "status",
    [
        FindingStatus.UNKNOWN,
        FindingStatus.INSUFFICIENT_EVIDENCE,
        FindingStatus.UPSTREAM_FAILURE,
        FindingStatus.INVALID_INPUT,
    ],
)
def test_unresolved_findings_are_explicit_unknown(status: FindingStatus) -> None:
    finding = Finding(
        worker=WorkerKind.CONTRACT,
        finding_key="contract.required_terms",
        category=RiskCategory.CONTRACT_TERMS,
        risk_level=RiskLevel.UNKNOWN,
        assertion=FindingAssertion.UNKNOWN,
        statement="The contract term cannot be determined.",
        status=status,
        missing_information=("signed contract text",),
        recommended_action="Request the signed contract.",
    )
    assert finding.assertion is FindingAssertion.UNKNOWN
    with pytest.raises(ValidationError):
        Finding(
            **{
                **finding.model_dump(exclude={"schema_version"}),
                "assertion": FindingAssertion.RISK_ABSENT,
            }
        )


def test_confirmation_receipt_is_local_only() -> None:
    receipt = ConfirmationReceipt(
        receipt_id="receipt-001",
        run_id="run-001",
        case_id="case-001",
        human_actor="reviewer-001",
        decision=HumanDecision.CONFIRM_RECOMMENDATION,
        reviewed_recommendation=RecommendationStatus.RECOMMEND_CONDITIONAL,
        recorded_at=NOW,
        idempotency_key="confirm-case-001-v1",
    )
    assert receipt.external_write_performed is False
    with pytest.raises(ValidationError):
        ConfirmationReceipt(
            **{
                **receipt.model_dump(
                    exclude={"schema_version", "external_write_performed"}
                ),
                "external_write_performed": True,
            }
        )


def test_approval_audit_rejects_impossible_state_transitions() -> None:
    valid = ApprovalAuditEvent(
        audit_id="audit-001",
        run_id="run-001",
        case_id="case-001",
        object_id="run-001",
        action_type=ApprovalActionType.CANCEL,
        actor="reviewer-001",
        occurred_at=NOW,
        api_version="review-api.v1",
        idempotency_key="cancel-run-001-v1",
        previous_status=ReviewLifecycleStatus.PROCESSING,
        new_status=ReviewLifecycleStatus.CANCELLED,
    )
    assert valid.external_write_performed is False
    with pytest.raises(ValidationError):
        ApprovalAuditEvent(
            **{
                **valid.model_dump(
                    exclude={"schema_version", "previous_status", "action_type"}
                ),
                "action_type": ApprovalActionType.CONFIRM,
                "previous_status": ReviewLifecycleStatus.PROCESSING,
                "new_status": ReviewLifecycleStatus.CONFIRMED,
                "receipt_id": "receipt-001",
            }
        )

    with pytest.raises(ValidationError):
        ApprovalAuditEvent(
            **{
                **valid.model_dump(exclude={"schema_version"}),
                "api_version": "review-api.v999",
            }
        )


def test_summary_cannot_hide_missing_information_behind_a_normal_recommendation() -> (
    None
):
    with pytest.raises(ValidationError):
        ReviewSummary(
            run_id="run-001",
            case_id="case-001",
            rules_version="1.0.0",
            findings=(),
            highest_risk=RiskLevel.LOW,
            recommendation=RecommendationStatus.RECOMMEND_CONDITIONAL,
            human_review_status=HumanReviewStatus.PENDING_CONFIRMATION,
            missing_information=("signed contract",),
        )


def test_summary_cannot_recommend_approval_without_findings() -> None:
    with pytest.raises(ValidationError):
        ReviewSummary(
            run_id="run-empty",
            case_id="case-empty",
            rules_version="1.0.0",
            findings=(),
            highest_risk=RiskLevel.LOW,
            recommendation=RecommendationStatus.RECOMMEND_APPROVE,
            human_review_status=HumanReviewStatus.PENDING_CONFIRMATION,
        )


def test_summary_must_surface_unresolved_findings() -> None:
    unresolved = Finding(
        worker=WorkerKind.CONTRACT,
        finding_key="contract.required_terms",
        category=RiskCategory.CONTRACT_TERMS,
        risk_level=RiskLevel.UNKNOWN,
        assertion=FindingAssertion.UNKNOWN,
        statement="Signed terms are unavailable.",
        status=FindingStatus.INSUFFICIENT_EVIDENCE,
        missing_information=("signed contract",),
        recommended_action="Request the signed contract.",
    )
    with pytest.raises(ValidationError):
        ReviewSummary(
            run_id="run-unresolved",
            case_id="case-unresolved",
            rules_version="1.0.0",
            findings=(unresolved,),
            highest_risk=RiskLevel.UNKNOWN,
            recommendation=RecommendationStatus.INSUFFICIENT_INFORMATION,
            human_review_status=HumanReviewStatus.REQUIRED,
        )


def test_run_event_tracks_cost_and_rejects_secret_fields() -> None:
    event = RunEvent(
        event_id="event-001",
        run_id="run-001",
        trace_id="trace-001",
        event_type=RunEventType.STEP_COMPLETED,
        step="policy-review",
        occurred_at=NOW,
        input_summary={"case_id": "case-001"},
        output_summary={"finding_count": 1},
        input_checksum=CHECKSUM,
        output_checksum=CHECKSUM,
        source_versions={"rules": "1.0.0"},
        duration_ms=12,
        token_count=0,
        cost_amount=Decimal(0),
        cost_currency="CNY",
        redaction_status=RedactionStatus.NOT_REQUIRED,
    )
    assert event.cost_amount == Decimal(0)
    with pytest.raises(ValidationError):
        RunEvent(
            **{
                **event.model_dump(exclude={"schema_version", "input_summary"}),
                "input_summary": {"nested": {"api-token": "secret"}},
            }
        )

    with pytest.raises(ValidationError):
        RunEvent(
            **{
                **event.model_dump(exclude={"schema_version", "input_summary"}),
                "input_summary": {"prompt": "unnecessary model context"},
            }
        )

    with pytest.raises(ValidationError):
        RunEvent(
            **{
                **event.model_dump(exclude={"schema_version", "input_summary"}),
                "input_summary": {"note": "x" * 9_000},
            }
        )

    with pytest.raises(ValidationError):
        RunEvent(
            **{
                **event.model_dump(exclude={"schema_version", "error"}),
                "error": "secret=unnecessary-value",
            }
        )
