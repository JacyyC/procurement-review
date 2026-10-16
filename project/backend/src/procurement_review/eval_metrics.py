"""Independent scoring of V1-02C eval runs against frozen Gold Labels.

Labels are loaded by the caller only at score time, after every model call has
finished, and no label field ever reaches a model request. Both topologies are
validated per worker and passed through the production deterministic aggregator, so
the only remaining difference is how many model calls produced the findings.

Every ratio is reported with its raw numerator, denominator and Wilson 95% interval.
The topology difference is a paired bootstrap over cases: repeat seeds are averaged
inside a case first, so repeats never masquerade as independent samples.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from .contracts import (
    BusinessFact,
    EvidenceRef,
    EscalationClass,
    Finding,
    FindingStatus,
    HumanReviewStatus,
    RecommendationStatus,
    RiskCategory,
    RiskLevel,
    WorkerKind,
)
from .eval_contract import (
    EvalProjection,
    ProjectedBranch,
    ProjectedRef,
    percentile,
)
from .gold_corpus import (
    GoldLexicalCorpus,
    gold_required_document_ids,
    mask_case_identifiers,
    ndcg_at_k,
    query_fingerprint,
    recall_at_k,
    reciprocal_rank,
)
from .gold_set import GoldCase, GoldLabel
from .review_graph import aggregate_worker_results
from .rules import RULE_SET, RuleSet, escalation_class, is_generated_reason_code
from .worker_contracts import WorkerBranchResult, WorkerFailureCode, WorkerOutcome
from .workers import failed_worker_result, validate_worker_findings

METRICS_VERSION = "v1-02c-eval-metrics.v1"
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260901
RECALL_K = 5

_ESCALATED = {
    RecommendationStatus.INSUFFICIENT_INFORMATION,
    RecommendationStatus.NEEDS_HUMAN_REVIEW,
}
_HIGH_RISK = {RiskLevel.HIGH, RiskLevel.CRITICAL}
WORKER_KINDS = tuple(WorkerKind)
_WORKERS = tuple(WorkerKind)
_REQUIRED_BY_WORKER: dict[WorkerKind, tuple[str, RiskCategory]] = {
    contract.worker: (finding_key, contract.category)
    for finding_key, contract in RULE_SET.required_finding_contracts.items()
}
if set(_REQUIRED_BY_WORKER) != set(WorkerKind):  # pragma: no cover - import guard
    raise RuntimeError("rules must define exactly one required finding per worker")

# Frozen quality gates from Evidence §23.4. A negative threshold means "lower is
# better"; -1 means the goal is exactly zero.
FROZEN_THRESHOLDS = {
    "recall_at_5": 0.95,
    "citation_support": 0.98,
    "unsupported_citation": 0.02,
    "structured_output_valid": 0.99,
    "required_finding_coverage": 0.98,
    "recommendation_accuracy": 0.90,
    "unknown_recall": 0.95,
    "conflict_recall": 0.95,
    "high_risk_miss": 0.0,
    "duplicate_query": 0.15,
}


class Statistic(BaseModel):
    """A ratio with the raw counts and interval that produced it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    threshold: float | None = None
    direction: Literal[">=", "<="] = ">="
    meets_threshold: bool | None = None
    detail: str | None = Field(default=None, max_length=300)

    @classmethod
    def of(
        cls,
        numerator: int,
        denominator: int,
        *,
        threshold: float | None = None,
        direction: Literal[">=", "<="] = ">=",
        detail: str | None = None,
    ) -> Self:
        value = numerator / denominator if denominator else None
        low = high = None
        if denominator:
            low, high = wilson_interval(numerator, denominator)
        meets = None
        if value is not None and threshold is not None:
            meets = value >= threshold if direction == ">=" else value <= threshold
        return cls(
            numerator=numerator,
            denominator=denominator,
            value=value,
            ci_low=low,
            ci_high=high,
            threshold=threshold,
            direction=direction,
            meets_threshold=meets,
            detail=detail,
        )


class MacroF1(BaseModel):
    """Macro-averaged F1 over the five Demo Policy recommendations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float | None = None
    class_count: int = Field(ge=0)
    per_class: dict[str, float] = Field(default_factory=dict)
    threshold: float | None = None


class PairedDelta(BaseModel):
    """Per-case paired difference between the two topologies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(min_length=1, max_length=80)
    case_count: int = Field(ge=0)
    single_agent_mean: float | None = None
    multi_agent_mean: float | None = None
    delta: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    favors: Literal["single_agent", "multi_agent", "indistinguishable"]
    detail: str | None = Field(default=None, max_length=300)


def _z_score(confidence: float) -> float:
    if confidence == 0.95:
        return 1.959963984540054
    return _inverse_normal_cdf(1 - (1 - confidence) / 2)


def _inverse_normal_cdf(probability: float) -> float:
    """Acklam's rational approximation of the inverse normal CDF."""

    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615364129522377e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    low, high = 0.02425, 1 - 0.02425
    if probability < low:
        q = math.sqrt(-2 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if probability > high:
        q = math.sqrt(-2 * math.log(1 - probability))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = probability - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def wilson_interval(
    numerator: int, denominator: int, *, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if denominator <= 0:
        return (0.0, 0.0)
    z = _z_score(confidence)
    rate = numerator / denominator
    scale = z * z / denominator
    center = (rate + scale / 2) / (1 + scale)
    spread = (z / (1 + scale)) * math.sqrt(
        rate * (1 - rate) / denominator + scale / (4 * denominator)
    )
    return (max(0.0, center - spread), min(1.0, center + spread))


def paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for the mean of per-case paired differences."""

    if not deltas:
        return None
    rng = random.Random(seed)
    size = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    )
    tail = (1 - confidence) / 2
    low_index = max(0, min(len(means) - 1, math.floor(tail * len(means))))
    high_index = max(0, min(len(means) - 1, math.ceil((1 - tail) * len(means)) - 1))
    return (means[low_index], means[high_index])


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _resolve_evidence(
    refs: Sequence[ProjectedRef], available: Mapping[str, EvidenceRef]
) -> tuple[EvidenceRef, ...]:
    resolved: list[EvidenceRef] = []
    for ref in refs:
        item = available.get(ref.source_id)
        if item is None or item.version != ref.version:
            raise ValueError("projected reference is outside its Gold Case")
        resolved.append(item)
    return tuple(resolved)


def _resolve_facts(
    refs: Sequence[ProjectedRef], available: Mapping[str, BusinessFact]
) -> tuple[BusinessFact, ...]:
    resolved: list[BusinessFact] = []
    for ref in refs:
        item = available.get(ref.source_id)
        if item is None or item.dataset_version != ref.version:
            raise ValueError("projected business fact is outside its Gold Case")
        resolved.append(item)
    return tuple(resolved)


def rehydrate_findings(
    projections: Sequence[Any],
    *,
    case: GoldCase,
    worker: WorkerKind,
) -> tuple[Finding, ...]:
    """Rebuild Findings for one worker from projected identities only.

    Model narrative text is never recovered: statement and recommended action become
    stable hashes, which is why reported conflict samples carry keys instead of text.
    """

    evidence_by_id = {
        context.evidence_ref.source_id: context.evidence_ref
        for context in case.model_input.evidence_contexts
    }
    facts_by_id = {fact.record_id: fact for fact in case.model_input.business_facts}
    rebuilt: list[Finding] = []
    for item in projections:
        rebuilt.append(
            Finding(
                worker=worker,
                finding_key=item.finding_key,
                category=item.category,
                risk_level=item.risk_level,
                assertion=item.assertion,
                status=item.status,
                statement=f"statement:{item.statement_hash[:24]}",
                recommended_action=f"action:{item.recommended_action_hash[:24]}",
                evidence_refs=_resolve_evidence(item.evidence_refs, evidence_by_id),
                business_fact_refs=_resolve_facts(item.business_fact_refs, facts_by_id),
                missing_information=tuple(item.missing_information),
            )
        )
    return tuple(rebuilt)


def _worker_allowlist(case: GoldCase, worker: WorkerKind) -> tuple[str, ...]:
    return next(
        item.allowed_ref_ids for item in case.allowed_context if item.worker is worker
    )


_FAILURE_CODES = {
    "timeout": WorkerFailureCode.TIMEOUT,
    "output_schema": WorkerFailureCode.INVALID_MODEL_OUTPUT,
    "finding_schema_invalid": WorkerFailureCode.INVALID_MODEL_OUTPUT,
    "invalid_json": WorkerFailureCode.INVALID_MODEL_OUTPUT,
    "evidence_reference_out_of_context": WorkerFailureCode.INVALID_MODEL_OUTPUT,
    "security_boundary": WorkerFailureCode.MODEL_FAILURE,
    "model_output_security_boundary": WorkerFailureCode.MODEL_FAILURE,
}


def _branch_failure_code(error_code: str | None) -> WorkerFailureCode:
    if error_code is None:
        return WorkerFailureCode.MODEL_FAILURE
    return _FAILURE_CODES.get(error_code, WorkerFailureCode.MODEL_FAILURE)


@dataclass(frozen=True, slots=True)
class CaseObservation:
    """One ``(case, seed, topology)`` result after the deterministic aggregation."""

    case_id: str
    split: str
    case_family_id: str
    seed: int
    topology: Literal["single_agent", "multi_agent"]
    recommendation: RecommendationStatus
    highest_risk: RiskLevel
    human_review_required: bool
    conflict_keys: tuple[str, ...]
    missing_information: tuple[str, ...]
    failure_reasons: tuple[str, ...]
    findings: tuple[Finding, ...]
    branch_outcomes: Mapping[str, str]
    failed_workers: tuple[str, ...]
    escalation_class: EscalationClass | None = None


def observations_from_projection(
    projection: EvalProjection,
    cases: Mapping[str, GoldCase],
    *,
    rule_set: RuleSet = RULE_SET,
) -> tuple[CaseObservation, ...]:
    """Rebuild per-case end-to-end results for either topology under one contract."""

    grouped: dict[str, list[ProjectedBranch]] = defaultdict(list)
    for branch in projection.branches:
        grouped[branch.case_id].append(branch)

    observations: list[CaseObservation] = []
    for case_id in sorted(grouped):
        case = cases.get(case_id)
        if case is None:
            raise ValueError("projected case is not part of the frozen Gold Set")
        branches = grouped[case_id]
        results: list[WorkerBranchResult] = []
        outcomes: dict[str, str] = {}
        failed: list[str] = []
        for worker in _WORKERS:
            branch, omitted_code = _select_branch(projection, branches, worker)
            outcomes[worker.value] = (
                "not_recorded"
                if branch is None
                else _outcome_of(branch, worker, omitted_code)
            )
            result = _worker_branch_result(
                case=case,
                worker=worker,
                branch=branch,
                omitted_code=omitted_code,
            )
            results.append(result)
            if result.outcome is WorkerOutcome.FAILED:
                failed.append(worker.value)
        summary = aggregate_worker_results(
            run_id=f"{projection.run_id}:{case_id}",
            case_id=case_id,
            branch_results=results,
            rule_set=rule_set,
        )
        observations.append(
            CaseObservation(
                case_id=case_id,
                split=case.split.value,
                case_family_id=case.case_family_id,
                seed=projection.repeat_seed,
                topology=projection.topology,
                recommendation=summary.recommendation,
                highest_risk=summary.highest_risk,
                human_review_required=(
                    summary.human_review_status is HumanReviewStatus.REQUIRED
                ),
                conflict_keys=tuple(
                    conflict.finding_key for conflict in summary.conflicts
                ),
                missing_information=summary.missing_information,
                failure_reasons=summary.failure_reasons,
                findings=summary.findings,
                branch_outcomes=outcomes,
                failed_workers=tuple(failed),
                escalation_class=(
                    None
                    if rule_set.gate_mode != "directional"
                    else escalation_class(summary)
                ),
            )
        )
    return tuple(observations)


def _select_branch(
    projection: EvalProjection,
    branches: Sequence[ProjectedBranch],
    worker: WorkerKind,
) -> tuple[ProjectedBranch | None, str | None]:
    """Pick the branch feeding one worker, plus the code when a branch is missing.

    A multi-agent case that skipped an empty-context branch has no record for that
    worker at all; a single-agent call always covers all three workers or fails.
    """

    if projection.topology == "multi_agent":
        for branch in branches:
            if branch.worker is worker:
                return branch, None
        return None, "skipped_empty_context"
    return branches[0], None


def _outcome_of(
    branch: ProjectedBranch, worker: WorkerKind, omitted_code: str | None
) -> str:
    if omitted_code:
        return omitted_code
    if branch is None:
        return "not_recorded"
    if branch.outcome != "succeeded":
        return branch.outcome
    if worker.value not in {finding.worker.value for finding in branch.findings}:
        return "worker_findings_omitted"
    return "succeeded"


def _worker_branch_result(
    *,
    case: GoldCase,
    worker: WorkerKind,
    branch: ProjectedBranch | None,
    omitted_code: str | None,
) -> WorkerBranchResult:
    if branch is None:
        return failed_worker_result(
            worker, WorkerFailureCode.NO_EVIDENCE, omitted_code or "branch_not_recorded"
        )
    if branch.outcome != "succeeded":
        return failed_worker_result(
            worker,
            _branch_failure_code(branch.error_code),
            branch.error_code or "model_call_failed",
        )
    worker_findings = tuple(
        finding for finding in branch.findings if finding.worker is worker
    )
    if not worker_findings:
        return failed_worker_result(
            worker,
            WorkerFailureCode.INVALID_MODEL_OUTPUT,
            "required_finding_omitted",
        )
    allowlist = set(_worker_allowlist(case, worker))
    permitted_evidence: tuple[EvidenceRef, ...] = ()
    permitted_facts: tuple[BusinessFact, ...] = ()
    if worker in {WorkerKind.POLICY, WorkerKind.CONTRACT}:
        permitted_evidence = tuple(
            context.evidence_ref
            for context in case.model_input.evidence_contexts
            if context.evidence_ref.source_id in allowlist
        )
    else:
        permitted_facts = tuple(
            fact
            for fact in case.model_input.business_facts
            if fact.record_id in allowlist
        )
    rehydrated = rehydrate_findings(worker_findings, case=case, worker=worker)
    try:
        findings = validate_worker_findings(
            rehydrated,
            worker=worker,
            evidence_refs=permitted_evidence,
            business_facts=permitted_facts,
        )
    except (TypeError, ValueError):
        return failed_worker_result(
            worker,
            WorkerFailureCode.INVALID_MODEL_OUTPUT,
            "model_output_invalid",
        )
    return WorkerBranchResult(
        worker=worker, outcome=WorkerOutcome.COMPLETED, findings=findings
    )


def _acceptable(label: GoldLabel, predicted: RecommendationStatus) -> bool:
    return predicted in {
        label.expected_recommendation,
        *label.acceptable_alternatives,
    }


def _supported_required_findings(findings: Sequence[Finding]) -> int:
    covered = 0
    for worker in _WORKERS:
        required_key, required_category = _REQUIRED_BY_WORKER[worker]
        if any(
            finding.worker is worker
            and finding.finding_key == required_key
            and finding.category is required_category
            and finding.status is FindingStatus.SUPPORTED
            for finding in findings
        ):
            covered += 1
    return covered


def _citations(findings: Sequence[Finding]) -> tuple[str, ...]:
    cited: list[str] = []
    for finding in findings:
        cited.extend(ref.source_id for ref in finding.evidence_refs)
        cited.extend(fact.record_id for fact in finding.business_fact_refs)
    return tuple(cited)


def quality_metrics(
    observations: Sequence[CaseObservation],
    cases: Mapping[str, GoldCase],
    labels: Mapping[str, GoldLabel],
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compute the frozen quality metrics for one topology's observations."""

    limits = dict(FROZEN_THRESHOLDS, **(thresholds or {}))
    scored = [
        item
        for item in observations
        if item.case_id in labels and item.case_id in cases
    ]
    recommendation_hits = 0
    unknown_hits = unknown_total = 0
    conflict_hits = conflict_total = 0
    high_risk_hits = high_risk_total = 0
    human_review_total = 0
    citation_supported = citation_total = 0
    citation_outside = 0
    finding_hits = 0
    confusion: Counter[str] = Counter()
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    worker_hits = {worker.value: 0 for worker in _WORKERS}
    branch_outcomes: Counter[str] = Counter()
    failure_codes: Counter[str] = Counter()

    for item in scored:
        label = labels[item.case_id]
        case = cases[item.case_id]
        if _acceptable(label, item.recommendation):
            recommendation_hits += 1
        expected = label.expected_recommendation.value
        predicted = item.recommendation.value
        confusion[f"{expected}->{predicted}"] += 1
        per_class[expected][1] += 1
        per_class[predicted][2] += 1
        if _acceptable(label, item.recommendation):
            per_class[expected][0] += 1

        if label.expected_recommendation in _ESCALATED:
            unknown_total += 1
            unknown_hits += int(item.recommendation in _ESCALATED)
        if label.expected_conflicts:
            conflict_total += 1
            conflict_hits += int(bool(item.conflict_keys))
        if (
            label.expected_recommendation is RecommendationStatus.RECOMMEND_REJECT
            or label.expected_risk_level in _HIGH_RISK
        ):
            high_risk_total += 1
            high_risk_hits += int(
                item.recommendation is RecommendationStatus.RECOMMEND_APPROVE
            )
        human_review_total += int(item.human_review_required)

        allowed = {
            ref_id for entry in case.allowed_context for ref_id in entry.allowed_ref_ids
        }
        gold_refs = {ref.source_id for ref in label.required_evidence_refs} | {
            fact.record_id for fact in label.required_business_fact_refs
        }
        for ref_id in _citations(item.findings):
            citation_total += 1
            citation_supported += int(ref_id in gold_refs)
            citation_outside += int(ref_id not in allowed)

        covered = _supported_required_findings(item.findings)
        finding_hits += covered
        for worker in _WORKERS:
            required_key, required_category = _REQUIRED_BY_WORKER[worker]
            if any(
                finding.worker is worker
                and finding.finding_key == required_key
                and finding.category is required_category
                and finding.status is FindingStatus.SUPPORTED
                for finding in item.findings
            ):
                worker_hits[worker.value] += 1
        branch_outcomes.update(item.branch_outcomes.values())
        failure_codes.update(f"{code}" for code in item.failure_reasons[:4])

    count = len(scored)
    observations_per_worker = count * len(_WORKERS)
    return {
        "metrics_version": METRICS_VERSION,
        "observation_count": count,
        "recommendation_accuracy": Statistic.of(
            recommendation_hits, count, threshold=limits["recommendation_accuracy"]
        ),
        "recommendation_macro_f1": macro_f1(per_class, limits),
        "confusion_matrix": dict(sorted(confusion.items())),
        "unknown_recall": Statistic.of(
            unknown_hits, unknown_total, threshold=limits["unknown_recall"]
        ),
        "conflict_recall": Statistic.of(
            conflict_hits, conflict_total, threshold=limits["conflict_recall"]
        ),
        "high_risk_miss_rate": Statistic.of(
            high_risk_hits,
            high_risk_total,
            threshold=limits["high_risk_miss"],
            direction="<=",
        ),
        "citation_support_precision": Statistic.of(
            citation_supported, citation_total, threshold=limits["citation_support"]
        ),
        "unsupported_citation_rate": Statistic.of(
            citation_total - citation_supported,
            citation_total,
            threshold=limits["unsupported_citation"],
            direction="<=",
        ),
        "citation_outside_allowed_context": Statistic.of(
            citation_outside, citation_total
        ),
        "required_finding_coverage": Statistic.of(
            finding_hits,
            observations_per_worker,
            threshold=limits["required_finding_coverage"],
        ),
        "required_finding_coverage_by_worker": {
            name: Statistic.of(hits, count)
            for name, hits in sorted(worker_hits.items())
        },
        "human_takeover_rate": Statistic.of(human_review_total, count),
        "branch_outcome_counts": dict(sorted(branch_outcomes.items())),
        "failure_reason_examples": dict(sorted(failure_codes.items())),
        "stratified_by_family": _slices(scored, labels, key="case_family_id"),
        "stratified_by_split": _slices(scored, labels, key="split"),
    }


def _slices(
    observations: Sequence[CaseObservation],
    labels: Mapping[str, GoldLabel],
    *,
    key: str,
) -> dict[str, Any]:
    grouped: dict[str, list[CaseObservation]] = defaultdict(list)
    for item in observations:
        grouped[str(getattr(item, key))].append(item)
    report: dict[str, Any] = {}
    for name, items in sorted(grouped.items()):
        exact = sum(
            1
            for item in items
            if item.case_id in labels
            and _acceptable(labels[item.case_id], item.recommendation)
        )
        report[name] = {
            "observations": len(items),
            "recommendation_accuracy": Statistic.of(exact, len(items)),
            "escalated": sum(1 for item in items if item.recommendation in _ESCALATED),
            "succeeded_branches": sum(
                1
                for item in items
                for value in item.branch_outcomes.values()
                if value == "succeeded"
            ),
        }
    return report


def macro_f1(
    per_class: Mapping[str, Sequence[int]],
    limits: Mapping[str, float] | None = None,
) -> MacroF1:
    scores: dict[str, float] = {}
    for name in sorted({*per_class, *(item.value for item in RecommendationStatus)}):
        hits, support, predicted = per_class.get(name, (0, 0, 0))
        precision = hits / predicted if predicted else 0.0
        recall = hits / support if support else 0.0
        if precision + recall == 0:
            scores[name] = 0.0
            continue
        scores[name] = 2 * precision * recall / (precision + recall)
    present = [name for name in scores if per_class.get(name, (0, 0, 0))[1]]
    value = sum(scores[name] for name in present) / len(present) if present else None
    resolved = dict(FROZEN_THRESHOLDS, **(limits or {}))
    return MacroF1(
        value=value,
        class_count=len(present),
        per_class={name: round(score, 6) for name, score in scores.items()},
        threshold=resolved["recommendation_accuracy"],
    )


def structured_output_stats(
    manifests: Sequence[Any],
) -> dict[str, Any]:
    """Structured-output validity and cost metrics straight from segment manifests."""

    calls = sum(item.model_call_count for item in manifests)
    succeeded = sum(item.succeeded_count for item in manifests)
    failed = sum(item.failed_count for item in manifests)
    skipped = sum(item.skipped_empty_context_count for item in manifests)
    audited = sum(item.branch_count for item in manifests)
    planned = sum(
        item.case_count * (len(WORKER_KINDS) if item.topology == "multi_agent" else 1)
        for item in manifests
    )
    error_codes: Counter[str] = Counter()
    for item in manifests:
        error_codes.update(item.error_code_counts)
    all_latencies = [
        call.latency_ms
        for item in manifests
        for call in item.calls
        if call.disposition == "model_call" and call.latency_ms is not None
    ]
    http_counts: Counter[str] = Counter()
    for item in manifests:
        http_counts.update(
            {status: count for status, count in item.http_status_counts.items()}
        )
    return {
        "model_call_count": calls,
        "succeeded_count": succeeded,
        "failed_count": failed,
        "skipped_empty_context_count": skipped,
        "structured_output_valid_rate": Statistic.of(
            succeeded, calls, threshold=FROZEN_THRESHOLDS["structured_output_valid"]
        ),
        "run_record_completeness": Statistic.of(audited, planned, threshold=1.0),
        "error_code_counts": dict(sorted(error_codes.items())),
        "input_tokens": sum(item.usage.input_tokens for item in manifests),
        "output_tokens": sum(item.usage.output_tokens for item in manifests),
        "total_tokens": sum(item.usage.total_tokens for item in manifests),
        "retry_total": sum(item.retry_total for item in manifests),
        "latency_ms": {
            "observed_count": len(all_latencies),
            "p50": percentile(all_latencies, 0.50),
            "p95": percentile(all_latencies, 0.95),
            "max": max(all_latencies) if all_latencies else None,
        },
        "http_status_counts": dict(sorted(http_counts.items())),
        "budget": {
            "warning_reached": any(item.budget.warning_reached for item in manifests),
            "exceeded": any(item.budget.exceeded for item in manifests),
            "consumed_requests": sum(
                item.budget.consumed_requests for item in manifests
            ),
            "consumed_output_tokens": sum(
                item.budget.consumed_output_tokens for item in manifests
            ),
        },
    }


def retrieval_report(
    corpora: Iterable[tuple[str, GoldLexicalCorpus]],
    cases: Sequence[GoldCase],
    labels: Mapping[str, GoldLabel],
    *,
    k: int = RECALL_K,
) -> dict[str, Any]:
    """Score the lexical index against Gold required documents for both query variants."""

    report: dict[str, Any] = {}
    by_split: dict[str, list[GoldCase]] = defaultdict(list)
    for case in cases:
        by_split[case.split.value].append(case)
    for name, corpus in corpora:
        fingerprints: Counter[str] = Counter()
        rows = {
            variant: _retrieval_variant(
                corpus, by_split[corpus.split.value], labels, k, variant
            )
            for variant in ("raw_query", "masked_case_number_query")
        }
        for case in by_split[corpus.split.value]:
            if case.case_id not in labels or not case.model_input.evidence_contexts:
                continue
            if not gold_required_document_ids(labels[case.case_id]):
                continue
            fingerprints[query_fingerprint(case.question)] += 1
        total = sum(fingerprints.values())
        distinct = len(fingerprints)
        report[name] = {
            "split": corpus.split.value,
            "indexed_document_count": corpus.document_count,
            "index_checksum": corpus.index_checksum,
            "vector_mode": corpus.vector_mode,
            "variants": rows,
            "duplicate_query_rate": Statistic.of(
                total - distinct,
                total,
                threshold=FROZEN_THRESHOLDS["duplicate_query"],
                direction="<=",
            ),
        }
    return report


def _retrieval_variant(
    corpus: GoldLexicalCorpus,
    cases: Sequence[GoldCase],
    labels: Mapping[str, GoldLabel],
    k: int,
    variant: str,
) -> dict[str, Any]:
    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    gold_sizes: list[int] = []
    for case in cases:
        label = labels.get(case.case_id)
        if label is None or not case.model_input.evidence_contexts:
            continue
        required = gold_required_document_ids(label)
        if not required:
            continue
        gold_sizes.append(len(required))
        query = (
            case.question
            if variant == "raw_query"
            else mask_case_identifiers(case.question)
        )
        ranked = corpus.ranked_source_ids(query, limit=max(k, 20))
        recalls.append(recall_at_k(ranked, required, k=k))
        mrrs.append(reciprocal_rank(ranked, required))
        ndcgs.append(ndcg_at_k(ranked, required))
    hit_cases = sum(1 for value in recalls if value > 0)
    found = sum(
        round(value * size) for value, size in zip(recalls, gold_sizes, strict=True)
    )
    return {
        "case_count": len(recalls),
        f"recall_at_{k}": Statistic.of(
            found,
            sum(gold_sizes),
            threshold=FROZEN_THRESHOLDS["recall_at_5"],
            detail="Gold required documents found in the top-k, over all required documents",
        ),
        f"case_hit_rate_at_{k}": Statistic.of(hit_cases, len(recalls)),
        "recall_mean": _mean(recalls),
        "mrr": _mean(mrrs),
        "ndcg_at_10": _mean(ndcgs),
    }


_INDICATORS: dict[str, Callable[[CaseObservation, GoldLabel], float]] = {
    "recommendation_exact": lambda item, label: float(
        _acceptable(label, item.recommendation)
    ),
    "escalated_to_human": lambda item, label: float(item.recommendation in _ESCALATED),
    "conflict_detected": lambda item, label: float(bool(item.conflict_keys)),
    "required_finding_coverage": lambda item, label: (
        _supported_required_findings(item.findings) / len(_WORKERS)
    ),
    "high_risk_missed": lambda item, label: float(
        label.expected_recommendation is RecommendationStatus.RECOMMEND_REJECT
        and item.recommendation is RecommendationStatus.RECOMMEND_APPROVE
    ),
    "citation_support_precision": lambda item, label: (
        sum(
            1
            for ref_id in _citations(item.findings)
            if ref_id
            in {ref.source_id for ref in label.required_evidence_refs}
            | {fact.record_id for fact in label.required_business_fact_refs}
        )
        / len(_citations(item.findings))
        if _citations(item.findings)
        else 0.0
    ),
}


def paired_topology_deltas(
    single: Sequence[CaseObservation],
    multi: Sequence[CaseObservation],
    labels: Mapping[str, GoldLabel],
) -> tuple[PairedDelta, ...]:
    """Paired per-case differences, averaging repeat seeds inside each case."""

    def per_case(metric: str, items: Sequence[CaseObservation]) -> dict[str, float]:
        buckets: dict[str, list[float]] = defaultdict(list)
        indicator = _INDICATORS[metric]
        for item in items:
            label = labels.get(item.case_id)
            if label is not None:
                buckets[item.case_id].append(indicator(item, label))
        return {
            case_id: sum(values) / len(values)
            for case_id, values in buckets.items()
            if values
        }

    deltas: list[PairedDelta] = []
    for metric in _INDICATORS:
        single_values = per_case(metric, single)
        multi_values = per_case(metric, multi)
        shared = sorted(set(single_values) & set(multi_values))
        differences = [single_values[key] - multi_values[key] for key in shared]
        interval = paired_bootstrap_ci(differences)
        delta = _mean(differences)
        favors: Literal["single_agent", "multi_agent", "indistinguishable"] = (
            "indistinguishable"
        )
        if interval is not None:
            if interval[0] > 0:
                favors = "single_agent"
            elif interval[1] < 0:
                favors = "multi_agent"
        deltas.append(
            PairedDelta(
                metric=metric,
                case_count=len(shared),
                single_agent_mean=_mean([single_values[key] for key in shared]),
                multi_agent_mean=_mean([multi_values[key] for key in shared]),
                delta=delta,
                ci_low=None if interval is None else interval[0],
                ci_high=None if interval is None else interval[1],
                favors=favors,
                detail=(
                    "repeat seeds are averaged inside each case, so case_count is "
                    "the effective sample behind this interval"
                ),
            )
        )
    return tuple(deltas)


def _redact_reason(value: str) -> str:
    """Keep the reason codes this repository emits; digest anything the model authored."""

    if is_generated_reason_code(value):
        return value
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def error_samples(
    observations: Sequence[CaseObservation],
    labels: Mapping[str, GoldLabel],
    *,
    limit: int = 60,
) -> tuple[dict[str, Any], ...]:
    """Bounded wrong-answer samples: identities, codes and labels only, no model prose."""

    samples: list[dict[str, Any]] = []
    for item in observations:
        label = labels.get(item.case_id)
        if label is None:
            continue
        correct = _acceptable(label, item.recommendation) and not item.failed_workers
        if correct:
            continue
        samples.append(
            {
                "case_id": item.case_id,
                "family": item.case_family_id,
                "split": item.split,
                "seed": item.seed,
                "topology": item.topology,
                "expected_recommendation": label.expected_recommendation.value,
                "predicted_recommendation": item.recommendation.value,
                "acceptable_alternatives": [
                    value.value for value in label.acceptable_alternatives
                ],
                "expected_risk_level": label.expected_risk_level.value,
                "predicted_risk_level": item.highest_risk.value,
                "rule_ids": list(label.rule_ids),
                "expected_missing_information": list(
                    label.expected_missing_information
                )[:6],
                "expected_conflicts": [
                    conflict.conflict_code for conflict in label.expected_conflicts
                ],
                "detected_conflicts": list(item.conflict_keys),
                "failed_workers": list(item.failed_workers),
                "branch_outcomes": dict(sorted(item.branch_outcomes.items())),
                "missing_information": [
                    _redact_reason(item_text)
                    for item_text in item.missing_information[:12]
                ],
                "failure_reasons": list(item.failure_reasons)[:12],
            }
        )
        if len(samples) >= limit:
            break
    return tuple(samples)
