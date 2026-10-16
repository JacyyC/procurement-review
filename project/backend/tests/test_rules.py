from datetime import UTC, datetime

import pytest

from procurement_review.contracts import (
    BusinessFact,
    EvidenceRef,
    FactStatus,
    Finding,
    FindingAssertion,
    FindingStatus,
    HumanReviewStatus,
    RecommendationStatus,
    RiskCategory,
    RiskLevel,
    SourceKind,
    WorkerKind,
)
from procurement_review.rules import (
    REQUIRED_FINDING_KEYS,
    RULE_SET,
    RULES_VERSION,
    aggregate_findings,
    deduplicate_findings,
    merge_risk,
)

CHECKSUM = "sha256:" + "b" * 64
NOW = datetime(2026, 8, 30, 8, 0, tzinfo=UTC)


def evidence(source_id: str) -> EvidenceRef:
    return EvidenceRef(
        source_kind=SourceKind.PUBLIC_DOCUMENT,
        source_id=source_id,
        version="v1",
        chunk_id=f"chunk-{source_id}",
        content_checksum=CHECKSUM,
    )


def business_fact() -> BusinessFact:
    return BusinessFact(
        tool_name="get_purchase_order",
        record_id="po-001",
        dataset_version="fixture-v1",
        as_of=NOW,
        status=FactStatus.KNOWN,
        result={"amount": 1200},
    )


def supported_finding(
    key: str,
    category: RiskCategory,
    worker: WorkerKind,
    *,
    risk: RiskLevel = RiskLevel.LOW,
    assertion: FindingAssertion = FindingAssertion.RISK_ABSENT,
    statement: str | None = None,
    use_fact: bool = False,
) -> Finding:
    return Finding(
        worker=worker,
        finding_key=key,
        category=category,
        risk_level=risk,
        assertion=assertion,
        statement=statement or f"Supported result for {key}.",
        status=FindingStatus.SUPPORTED,
        evidence_refs=() if use_fact else (evidence(key),),
        business_fact_refs=(business_fact(),) if use_fact else (),
        recommended_action="Continue deterministic review.",
    )


def complete_low_risk_findings() -> tuple[Finding, ...]:
    return (
        supported_finding(
            REQUIRED_FINDING_KEYS[0], RiskCategory.POLICY_COMPLIANCE, WorkerKind.POLICY
        ),
        supported_finding(
            REQUIRED_FINDING_KEYS[1], RiskCategory.CONTRACT_TERMS, WorkerKind.CONTRACT
        ),
        supported_finding(
            REQUIRED_FINDING_KEYS[2],
            RiskCategory.PROCUREMENT_ANOMALY,
            WorkerKind.PROCUREMENT_DATA,
            use_fact=True,
        ),
    )


def test_versioned_rule_table_covers_fixed_domains() -> None:
    assert RULES_VERSION == "1.0.0"
    assert RULE_SET.schema_version == "1.0"
    assert tuple(RULE_SET.risk_categories) == tuple(RiskCategory)
    assert set(RULE_SET.risk_order) == set(RiskLevel)
    assert set(RULE_SET.required_finding_keys) == set(
        RULE_SET.required_finding_contracts
    )


def test_merge_risk_uses_fixed_highest_risk_order() -> None:
    assert merge_risk([]) is RiskLevel.UNKNOWN
    assert (
        merge_risk([RiskLevel.MEDIUM, RiskLevel.CRITICAL, RiskLevel.LOW])
        is RiskLevel.CRITICAL
    )


def test_deduplication_is_order_independent_and_merges_highest_risk() -> None:
    low = supported_finding(
        "supplier.duplicate-check",
        RiskCategory.SUPPLIER_INTEGRITY,
        WorkerKind.POLICY,
        risk=RiskLevel.LOW,
        assertion=FindingAssertion.RISK_PRESENT,
        statement="Duplicate supplier signal.",
    )
    high = supported_finding(
        "supplier.duplicate-check",
        RiskCategory.SUPPLIER_INTEGRITY,
        WorkerKind.POLICY,
        risk=RiskLevel.HIGH,
        assertion=FindingAssertion.RISK_PRESENT,
        statement="Corroborated duplicate supplier signal.",
    )

    forward = deduplicate_findings((low, high))
    reverse = deduplicate_findings((high, low))

    assert forward == reverse
    assert len(forward) == 1
    assert forward[0].risk_level is RiskLevel.HIGH
    assert forward[0].statement == high.statement
    assert len(forward[0].evidence_refs) == 1


def test_deduplication_canonicalizes_business_fact_mapping_key_order() -> None:
    first_fact = business_fact().model_copy(
        update={"result": {"amount": 1200, "currency": "CNY"}}
    )
    second_fact = business_fact().model_copy(
        update={"result": {"currency": "CNY", "amount": 1200}}
    )
    base = supported_finding(
        "procurement.mapping-order",
        RiskCategory.PROCUREMENT_ANOMALY,
        WorkerKind.PROCUREMENT_DATA,
        use_fact=True,
    )
    first = base.model_copy(update={"business_fact_refs": (first_fact,)})
    second = base.model_copy(update={"business_fact_refs": (second_fact,)})

    forward = deduplicate_findings((first, second))
    reverse = deduplicate_findings((second, first))

    assert forward == reverse
    assert len(forward) == 1
    assert len(forward[0].business_fact_refs) == 1


def test_complete_low_risk_input_is_only_a_suggestion_to_approve() -> None:
    summary = aggregate_findings(
        run_id="run-approve",
        case_id="case-approve",
        findings=reversed(complete_low_risk_findings()),
    )

    assert summary.recommendation is RecommendationStatus.RECOMMEND_APPROVE
    assert summary.human_review_status is HumanReviewStatus.PENDING_CONFIRMATION
    assert summary.is_final_decision is False
    assert summary.missing_information == ()
    assert summary.failure_reasons == ()


def test_highest_risk_deterministically_controls_complete_recommendation() -> None:
    findings = list(complete_low_risk_findings())
    findings.append(
        supported_finding(
            "financial.exposure",
            RiskCategory.FINANCIAL_EXPOSURE,
            WorkerKind.PROCUREMENT_DATA,
            risk=RiskLevel.HIGH,
            assertion=FindingAssertion.RISK_PRESENT,
        )
    )
    summary = aggregate_findings(
        run_id="run-risk",
        case_id="case-risk",
        findings=findings,
    )
    assert summary.highest_risk is RiskLevel.HIGH
    assert summary.recommendation is RecommendationStatus.RECOMMEND_REJECT


def test_missing_required_finding_fails_closed() -> None:
    summary = aggregate_findings(
        run_id="run-missing",
        case_id="case-missing",
        findings=complete_low_risk_findings()[:2],
    )

    assert summary.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION
    assert summary.human_review_status is HumanReviewStatus.REQUIRED
    assert any(
        item.startswith("missing_required_finding:")
        for item in summary.missing_information
    )


def test_caller_cannot_disable_fixed_required_findings() -> None:
    with pytest.raises(TypeError):
        aggregate_findings(
            run_id="run-bypass",
            case_id="case-bypass",
            findings=(
                supported_finding(
                    "caller.supplied-only",
                    RiskCategory.DATA_QUALITY,
                    WorkerKind.POLICY,
                ),
            ),
            required_finding_keys=(),  # type: ignore[call-arg]
        )


def test_wrong_critical_evidence_kind_fails_closed() -> None:
    findings = list(complete_low_risk_findings())
    findings[0] = supported_finding(
        REQUIRED_FINDING_KEYS[0],
        RiskCategory.POLICY_COMPLIANCE,
        WorkerKind.POLICY,
        use_fact=True,
    )
    summary = aggregate_findings(
        run_id="run-wrong-evidence",
        case_id="case-wrong-evidence",
        findings=findings,
    )
    assert summary.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION
    assert (
        f"critical_evidence_missing:{REQUIRED_FINDING_KEYS[0]}:evidence_ref"
        in summary.missing_information
    )


def test_wrong_worker_for_required_key_fails_closed() -> None:
    findings = list(complete_low_risk_findings())
    findings[0] = supported_finding(
        REQUIRED_FINDING_KEYS[0],
        RiskCategory.POLICY_COMPLIANCE,
        WorkerKind.PROCUREMENT_DATA,
    )
    summary = aggregate_findings(
        run_id="run-wrong-worker",
        case_id="case-wrong-worker",
        findings=findings,
    )
    assert summary.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION
    assert (
        f"required_finding_source_mismatch:{REQUIRED_FINDING_KEYS[0]}"
        in summary.missing_information
    )


def test_wrong_category_for_required_key_fails_closed() -> None:
    findings = list(complete_low_risk_findings())
    findings[0] = supported_finding(
        REQUIRED_FINDING_KEYS[0],
        RiskCategory.DATA_QUALITY,
        WorkerKind.POLICY,
    )
    summary = aggregate_findings(
        run_id="run-wrong-category",
        case_id="case-wrong-category",
        findings=findings,
    )
    assert summary.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION
    assert (
        f"required_finding_source_mismatch:{REQUIRED_FINDING_KEYS[0]}"
        in summary.missing_information
    )


def test_unknown_and_upstream_failure_both_fail_closed() -> None:
    findings = list(complete_low_risk_findings())
    findings[1] = Finding(
        worker=WorkerKind.CONTRACT,
        finding_key=REQUIRED_FINDING_KEYS[1],
        category=RiskCategory.CONTRACT_TERMS,
        risk_level=RiskLevel.UNKNOWN,
        assertion=FindingAssertion.UNKNOWN,
        statement="Signed terms are unavailable.",
        status=FindingStatus.INSUFFICIENT_EVIDENCE,
        missing_information=("signed contract",),
        recommended_action="Request signed contract.",
    )
    summary = aggregate_findings(
        run_id="run-unknown",
        case_id="case-unknown",
        findings=findings,
        upstream_failures=("procurement MCP timeout",),
    )
    assert summary.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION
    assert "procurement MCP timeout" in summary.failure_reasons
    assert "signed contract" in summary.missing_information


@pytest.mark.parametrize(
    "status",
    (
        FindingStatus.UNKNOWN,
        FindingStatus.INSUFFICIENT_EVIDENCE,
        FindingStatus.UPSTREAM_FAILURE,
        FindingStatus.INVALID_INPUT,
    ),
)
def test_every_unresolved_finding_status_fails_closed(
    status: FindingStatus,
) -> None:
    findings = list(complete_low_risk_findings())
    findings[1] = Finding(
        worker=WorkerKind.CONTRACT,
        finding_key=REQUIRED_FINDING_KEYS[1],
        category=RiskCategory.CONTRACT_TERMS,
        risk_level=RiskLevel.UNKNOWN,
        assertion=FindingAssertion.UNKNOWN,
        statement="The contract term cannot be determined.",
        status=status,
        missing_information=("signed contract",),
        recommended_action="Request the signed contract.",
    )

    summary = aggregate_findings(
        run_id=f"run-{status.value}",
        case_id="case-unresolved",
        findings=findings,
    )

    assert summary.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION
    assert summary.human_review_status is HumanReviewStatus.REQUIRED


def test_invalid_raw_finding_fails_closed_instead_of_disappearing() -> None:
    summary = aggregate_findings(
        run_id="run-invalid",
        case_id="case-invalid",
        findings=(*complete_low_risk_findings(), {"unexpected": "shape"}),
    )
    assert summary.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION
    assert summary.failure_reasons == ("invalid_finding_input",)


def test_opposite_assertions_are_preserved_and_escalated() -> None:
    findings = list(complete_low_risk_findings())
    findings.extend(
        (
            supported_finding(
                "contract.auto-renewal",
                RiskCategory.CONTRACT_TERMS,
                WorkerKind.CONTRACT,
                assertion=FindingAssertion.RISK_PRESENT,
                statement="The contract contains auto-renewal.",
            ),
            supported_finding(
                "contract.auto-renewal",
                RiskCategory.CONTRACT_TERMS,
                WorkerKind.POLICY,
                assertion=FindingAssertion.RISK_ABSENT,
                statement="The contract does not contain auto-renewal.",
            ),
        )
    )

    summary = aggregate_findings(
        run_id="run-conflict",
        case_id="case-conflict",
        findings=findings,
    )

    assert summary.recommendation is RecommendationStatus.NEEDS_HUMAN_REVIEW
    assert summary.human_review_status is HumanReviewStatus.REQUIRED
    assert summary.highest_risk is RiskLevel.HIGH
    assert len(summary.conflicts) == 1
    conflicting_findings = [
        item for item in summary.findings if item.finding_key == "contract.auto-renewal"
    ]
    assert len(conflicting_findings) == 2


def test_category_mismatch_is_preserved_and_escalated() -> None:
    findings = list(complete_low_risk_findings())
    findings.extend(
        (
            supported_finding(
                "shared.control",
                RiskCategory.DATA_QUALITY,
                WorkerKind.POLICY,
            ),
            supported_finding(
                "shared.control",
                RiskCategory.FINANCIAL_EXPOSURE,
                WorkerKind.POLICY,
            ),
        )
    )

    summary = aggregate_findings(
        run_id="run-category-conflict",
        case_id="case-category-conflict",
        findings=findings,
    )

    assert summary.recommendation is RecommendationStatus.NEEDS_HUMAN_REVIEW
    assert summary.human_review_status is HumanReviewStatus.REQUIRED
    assert len(summary.conflicts) == 1
    assert summary.conflicts[0].reason.value == "category_mismatch"


def test_output_is_identical_for_same_findings_in_any_order() -> None:
    findings = complete_low_risk_findings()
    forward = aggregate_findings(
        run_id="run-001", case_id="case-001", findings=findings
    )
    reverse = aggregate_findings(
        run_id="run-001", case_id="case-001", findings=reversed(findings)
    )
    assert forward.model_dump_json() == reverse.model_dump_json()
