"""Versioned, deterministic and model-free aggregation rules."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .contracts import (
    COVERAGE_CAUSE_PREFIX,
    CONTRACT_SCHEMA_VERSION,
    EVIDENCE_CAUSE_PREFIX,
    PIPELINE_CAUSE_PREFIX,
    CauseClass,
    ConflictReason,
    EscalationClass,
    Finding,
    FindingAssertion,
    FindingConflict,
    FindingStatus,
    HumanReviewStatus,
    RecommendationStatus,
    ReviewSummary,
    RiskCategory,
    RiskLevel,
    WorkerKind,
    namespaced_cause_class,
)


# Reason codes this repository generates itself. The distinction matters twice: the AC-09
# audit classifies retained strings with it, and the derived report keeps only these while
# replacing model-authored text with a digest.
GENERATED_REASON_PREFIXES = (
    "missing_required_finding:",
    "unresolved_finding:",
    "required_finding_source_mismatch:",
    "critical_evidence_missing:",
    "finding_failure:",
    "worker_failure:",
    "upstream_failure:",
    "no_valid_findings",
    "invalid_finding_input",
    "incomplete_recommendation",
    "no_supported_finding:",
)

# The rules 2.0 namespaces are shorter and therefore much likelier to collide with prose a
# model authored, so only the strict shape contracts declares counts as generated.


def is_generated_reason_code(value: str) -> bool:
    """Whether a reason string came from these rules rather than from the model."""

    if namespaced_cause_class(value) is not None:
        return True
    return any(
        value == prefix.rstrip(":") or value.startswith(prefix)
        for prefix in GENERATED_REASON_PREFIXES
    )


def classify_cause(code: str) -> CauseClass:
    """Which R-2 outcome class a recorded cause belongs to under rules 2.0.

    Anything this repository generated but did not namespace is treated as a pipeline
    cause, because an unlabelled generated code can only come from a failing branch.
    """

    registered = namespaced_cause_class(code)
    if registered is not None:
        return registered
    if is_generated_reason_code(code):
        return CauseClass.PIPELINE
    return CauseClass.CONTENT_NOTE


RulesVersion = Annotated[str, Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")]
_ConflictAnchor = Literal["finding_key", "evidence_ref", "business_fact"]
_EscalationCause = Literal["pipeline", "conflict", "evidence"]


class _RequiredFindingContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    worker: WorkerKind
    category: RiskCategory
    evidence_kind: Literal["evidence_ref", "business_fact"]


class RuleSet(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["1.0"]
    rules_version: RulesVersion
    risk_categories: tuple[RiskCategory, ...]
    risk_order: tuple[RiskLevel, ...]
    required_finding_keys: tuple[str, ...]
    required_finding_contracts: dict[str, _RequiredFindingContract]
    fail_closed_finding_statuses: tuple[FindingStatus, ...]
    recommendation_by_highest_risk: dict[RiskLevel, RecommendationStatus]
    conflict_assertions: tuple[FindingAssertion, FindingAssertion]
    conflict_minimum_risk: RiskLevel
    conflict_recommendation: Literal[RecommendationStatus.NEEDS_HUMAN_REVIEW]
    incomplete_recommendation: Literal[RecommendationStatus.INSUFFICIENT_INFORMATION]
    # Rules 1.0 leaves every one of these at its default; the defaults are the frozen v1
    # semantics, so an older resource cannot acquire new behaviour by omission.
    gate_mode: Literal["all_gaps", "directional"] = "all_gaps"
    conflict_anchors: tuple[_ConflictAnchor, ...] = ("finding_key",)
    coverage_required_recommendations: tuple[RecommendationStatus, ...] = ()
    escalation_precedence: tuple[_EscalationCause, ...] = ()

    @model_validator(mode="after")
    def validate_complete_rule_domain(self) -> RuleSet:
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ValueError("rule and contract schema versions differ")
        if tuple(self.risk_categories) != tuple(RiskCategory):
            raise ValueError("risk_categories must enumerate the fixed contract order")
        if set(self.risk_order) != set(RiskLevel) or len(self.risk_order) != len(
            RiskLevel
        ):
            raise ValueError("risk_order must contain every risk level exactly once")
        if self.risk_order[0] is not RiskLevel.UNKNOWN:
            raise ValueError("UNKNOWN must be the lowest item in risk_order")
        if len(set(self.required_finding_keys)) != len(self.required_finding_keys):
            raise ValueError("required_finding_keys must be unique")
        if set(self.required_finding_contracts) != set(self.required_finding_keys):
            raise ValueError(
                "every required finding must define its source and evidence contract"
            )
        expected_failures = set(FindingStatus) - {FindingStatus.SUPPORTED}
        if set(self.fail_closed_finding_statuses) != expected_failures:
            raise ValueError("every unresolved finding status must fail closed")
        if set(self.recommendation_by_highest_risk) != set(RiskLevel):
            raise ValueError("every risk level must map to a recommendation")
        if set(self.conflict_assertions) != {
            FindingAssertion.RISK_PRESENT,
            FindingAssertion.RISK_ABSENT,
        }:
            raise ValueError("conflict assertions must be risk_present and risk_absent")
        self._validate_gate_mode()
        return self

    def _validate_gate_mode(self) -> None:
        major = self.rules_version.split(".", maxsplit=1)[0]
        if self.gate_mode == "all_gaps":
            if major != "1":
                raise ValueError("only rules 1.x may use the all_gaps gate mode")
            frozen_defaults = (
                self.conflict_anchors == ("finding_key",)
                and not self.escalation_precedence
                and not self.coverage_required_recommendations
            )
            if not frozen_defaults:
                raise ValueError(
                    "rules 1.x must keep the frozen conflict and gate defaults"
                )
            return
        if major != "2":
            raise ValueError("only rules 2.x may use the directional gate mode")
        if len(set(self.conflict_anchors)) != len(self.conflict_anchors):
            raise ValueError("conflict_anchors must be unique")
        precedence = self.escalation_precedence
        if set(precedence) != {"pipeline", "conflict", "evidence"} or len(
            set(precedence)
        ) != len(precedence):
            raise ValueError(
                "escalation_precedence must order each R-2 cause class once"
            )
        allowed = {
            RecommendationStatus.RECOMMEND_APPROVE,
            RecommendationStatus.RECOMMEND_CONDITIONAL,
        }
        if not set(self.coverage_required_recommendations) <= allowed:
            raise ValueError(
                "only an approval or conditional recommendation may require full coverage"
            )


def _load_rule_set(version_file: str) -> RuleSet:
    path = Path(__file__).parent / "resources" / "rules" / f"{version_file}.json"
    return RuleSet.model_validate_json(path.read_text(encoding="utf-8"))


def _assert_model_vocabulary_unchanged(rule_set: RuleSet, frozen: RuleSet) -> None:
    """Keep the M2 promise: a revised aggregate may not change what the model is asked.

    ``chat_adapter.required_findings_for`` builds the outbound vocabulary from the frozen
    rules 1.0 table, so a rule set that renamed or re-bound a required finding would
    silently turn an aggregation-only revision into a model-visible change.
    """

    if rule_set.required_finding_keys != frozen.required_finding_keys:
        raise RuntimeError(
            f"rules {rule_set.rules_version} must keep the frozen required finding keys"
        )
    if rule_set.required_finding_contracts != frozen.required_finding_contracts:
        raise RuntimeError(
            f"rules {rule_set.rules_version} must keep the frozen required finding sources"
        )


RULE_SET = _load_rule_set("v1")
RULE_SET_V2 = _load_rule_set("v2")
_assert_model_vocabulary_unchanged(RULE_SET_V2, RULE_SET)

RULE_SETS: Mapping[str, RuleSet] = MappingProxyType(
    {rule_set.rules_version: rule_set for rule_set in (RULE_SET, RULE_SET_V2)}
)
if len(RULE_SETS) != 2:  # pragma: no cover - import guard
    raise RuntimeError("every loaded rule set must have a distinct version")

RULES_VERSION = RULE_SET.rules_version
RULES_VERSION_V2 = RULE_SET_V2.rules_version
REQUIRED_FINDING_KEYS = RULE_SET.required_finding_keys


def rule_set_for_version(rules_version: str) -> RuleSet:
    """Look a loaded rule set up by its declared version; unknown versions fail closed."""

    try:
        return RULE_SETS[rules_version]
    except KeyError:
        raise ValueError(f"unknown rules version {rules_version!r}") from None


_RISK_RANK = {level: rank for rank, level in enumerate(RULE_SET.risk_order)}
_REQUIRED_FINDING_CONTRACTS = MappingProxyType(
    dict(RULE_SET.required_finding_contracts)
)
_RECOMMENDATION_BY_HIGHEST_RISK = MappingProxyType(
    dict(RULE_SET.recommendation_by_highest_risk)
)
_STATUS_RANK = {
    FindingStatus.SUPPORTED: 0,
    FindingStatus.UNKNOWN: 1,
    FindingStatus.INSUFFICIENT_EVIDENCE: 2,
    FindingStatus.UPSTREAM_FAILURE: 3,
    FindingStatus.INVALID_INPUT: 4,
}


def _unique_sorted(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({value.strip() for value in values if value.strip()}))


def _model_sort_key(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _unique_models[ModelT: BaseModel](
    models: Iterable[ModelT],
) -> tuple[ModelT, ...]:
    by_serialized_value = {_model_sort_key(model): model for model in models}
    return tuple(by_serialized_value[key] for key in sorted(by_serialized_value))


def merge_risk(levels: Iterable[RiskLevel]) -> RiskLevel:
    """Return the highest risk deterministically; empty input is UNKNOWN."""

    return max(levels, key=_RISK_RANK.__getitem__, default=RiskLevel.UNKNOWN)


def deduplicate_findings(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    """Merge same-key/same-assertion findings without depending on input order."""

    groups: dict[
        tuple[str, FindingAssertion, RiskCategory, WorkerKind], list[Finding]
    ] = defaultdict(list)
    for finding in findings:
        if not isinstance(finding, Finding):
            raise TypeError("deduplicate_findings accepts validated Finding objects")
        groups[
            (finding.finding_key, finding.assertion, finding.category, finding.worker)
        ].append(finding)

    merged: list[Finding] = []
    for group_key in sorted(
        groups,
        key=lambda key: tuple(
            part.value if hasattr(part, "value") else part for part in key
        ),
    ):
        group = groups[group_key]
        representative = min(
            group,
            key=lambda item: (
                -_RISK_RANK[item.risk_level],
                -_STATUS_RANK[item.status],
                item.worker.value,
                item.statement.casefold(),
                item.recommended_action.casefold(),
                _model_sort_key(item),
            ),
        )
        evidence_refs = _unique_models(
            item for finding in group for item in finding.evidence_refs
        )
        business_facts = _unique_models(
            item for finding in group for item in finding.business_fact_refs
        )
        missing_information = _unique_sorted(
            item for finding in group for item in finding.missing_information
        )
        merged.append(
            Finding(
                worker=representative.worker,
                finding_key=representative.finding_key,
                category=representative.category,
                risk_level=merge_risk(item.risk_level for item in group),
                assertion=representative.assertion,
                statement=representative.statement,
                status=max(group, key=lambda item: _STATUS_RANK[item.status]).status,
                evidence_refs=evidence_refs,
                business_fact_refs=business_facts,
                missing_information=missing_information,
                recommended_action=representative.recommended_action,
            )
        )
    return tuple(merged)


def _find_conflicts(findings: tuple[Finding, ...]) -> tuple[FindingConflict, ...]:
    by_key: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        by_key[finding.finding_key].append(finding)

    conflicts: list[FindingConflict] = []
    for finding_key in sorted(by_key):
        group = by_key[finding_key]
        assertions = tuple(
            sorted({item.assertion for item in group}, key=lambda item: item.value)
        )
        categories = tuple(
            sorted({item.category for item in group}, key=lambda item: item.value)
        )
        statements = _unique_sorted(item.statement for item in group)
        if set(RULE_SET.conflict_assertions).issubset(assertions):
            conflicts.append(
                FindingConflict(
                    finding_key=finding_key,
                    reason=ConflictReason.OPPOSING_ASSERTIONS,
                    assertions=assertions,
                    categories=categories,
                    statements=statements,
                )
            )
        elif len(categories) > 1:
            conflicts.append(
                FindingConflict(
                    finding_key=finding_key,
                    reason=ConflictReason.CATEGORY_MISMATCH,
                    assertions=assertions,
                    categories=categories,
                    statements=statements,
                )
            )
    return tuple(conflicts)


def _validated_findings(
    findings: Iterable[Finding | Mapping[str, object]],
) -> tuple[tuple[Finding, ...], bool]:
    validated: list[Finding] = []
    invalid = False
    try:
        for finding in findings:
            if isinstance(finding, Finding):
                validated.append(finding)
                continue
            try:
                validated.append(Finding.model_validate(finding))
            except (TypeError, ValidationError):
                invalid = True
    except (TypeError, ValueError, RuntimeError):
        invalid = True
    return tuple(validated), invalid


_CONFLICT_SUBJECT_SLUG = re.compile(r"[^A-Za-z0-9._:/-]+")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")


def _conflict_subject(anchor: str, subject: str) -> str:
    """A stable, contract-valid identifier naming what two findings disagreed about."""

    readable = subject if anchor == "finding_key" else f"conflict:{anchor}:{subject}"
    if len(readable) <= 160 and _IDENTIFIER_PATTERN.fullmatch(readable):
        return readable
    digest = hashlib.sha256(subject.encode("utf-8")).hexdigest()[:16]
    return f"conflict:{anchor}:{digest}"


def _finding_subjects(anchor: str, finding: Finding) -> tuple[str, ...]:
    """The subjects this Finding speaks about, under one of the registered anchors."""

    if anchor == "finding_key":
        return (finding.finding_key,)
    if anchor == "evidence_ref":
        return tuple(f"{ref.source_id}/{ref.version}" for ref in finding.evidence_refs)
    return tuple(
        f"{fact.record_id}/{fact.dataset_version}"
        for fact in finding.business_fact_refs
    )


def find_conflicts_v2(
    findings: tuple[Finding, ...], rule_set: RuleSet
) -> tuple[FindingConflict, ...]:
    """Detect opposing assertions on every anchor the rule set registers (R-3).

    Unlike rules 1.0, the subject may be a shared reference rather than a shared key, so a
    single worker that reports the same document as both risky and not risky is a conflict
    even when no other worker touched it. Worker isolation is never widened here: the
    grouping reads only Findings that already passed their own worker boundary.
    """

    conflicts: list[FindingConflict] = []
    reported: set[frozenset[int]] = set()
    for anchor in rule_set.conflict_anchors:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, finding in enumerate(findings):
            for subject in _finding_subjects(anchor, finding):
                groups[subject].append(index)
        for subject in sorted(groups):
            members = frozenset(groups[subject])
            if len(members) < 2 or members in reported:
                continue
            group = [findings[index] for index in sorted(members)]
            assertions = tuple(
                sorted({item.assertion for item in group}, key=lambda item: item.value)
            )
            categories = tuple(
                sorted({item.category for item in group}, key=lambda item: item.value)
            )
            opposing = set(rule_set.conflict_assertions).issubset(assertions)
            category_mismatch = anchor == "finding_key" and len(categories) > 1
            if not opposing and not category_mismatch:
                continue
            reported.add(members)
            conflicts.append(
                FindingConflict(
                    finding_key=_conflict_subject(
                        anchor,
                        _CONFLICT_SUBJECT_SLUG.sub("-", subject).strip("-"),
                    ),
                    reason=(
                        ConflictReason.OPPOSING_ASSERTIONS
                        if opposing
                        else ConflictReason.CATEGORY_MISMATCH
                    ),
                    assertions=assertions,
                    categories=categories,
                    statements=_unique_sorted(item.statement for item in group),
                )
            )
    return tuple(conflicts)


def _cause_prefix(cause_class: CauseClass) -> str:
    return {
        CauseClass.PIPELINE: PIPELINE_CAUSE_PREFIX,
        CauseClass.EVIDENCE: EVIDENCE_CAUSE_PREFIX,
        CauseClass.COVERAGE: COVERAGE_CAUSE_PREFIX,
        CauseClass.CONTENT_NOTE: "",
    }[cause_class]


def namespaced_cause(code: str, cause_class: CauseClass) -> str:
    """Record one cause under the namespace of its class; content notes stay verbatim."""

    prefix = _cause_prefix(cause_class)
    if not prefix:
        return code
    if namespaced_cause_class(code) is cause_class:
        return code
    return f"{prefix}{code}"


class _GapProfile:
    """Which R-2 cause classes a set of recorded causes belongs to."""

    __slots__ = ("coverage", "evidence", "pipeline")

    def __init__(self, causes: Iterable[str]) -> None:
        classes = {classify_cause(code) for code in causes}
        self.pipeline = CauseClass.PIPELINE in classes
        self.evidence = CauseClass.EVIDENCE in classes
        self.coverage = CauseClass.COVERAGE in classes


_ESCALATION_BY_CAUSE = {
    "pipeline": EscalationClass.PIPELINE_INCOMPLETE,
    "conflict": EscalationClass.CONFLICT,
    "evidence": EscalationClass.EVIDENCE_INSUFFICIENT,
}
_CONTENT_RECOMMENDATIONS = frozenset(
    {
        RecommendationStatus.RECOMMEND_APPROVE,
        RecommendationStatus.RECOMMEND_CONDITIONAL,
        RecommendationStatus.RECOMMEND_REJECT,
    }
)


def rule_set_of(summary: ReviewSummary) -> RuleSet:
    return rule_set_for_version(summary.rules_version)


def escalation_class(summary: ReviewSummary) -> EscalationClass:
    """Derive the single R-2 outcome class of a rules 2.0 summary.

    The class is read back from the recorded causes instead of being stored, so a summary
    cannot claim a content recommendation while carrying a gating cause: the contract
    rejects that combination before it can be observed here.
    """

    rule_set = rule_set_of(summary)
    if rule_set.gate_mode != "directional":
        raise ValueError("escalation classes require a directional rule set")
    if summary.recommendation in _CONTENT_RECOMMENDATIONS:
        return EscalationClass.CONTENT
    gaps = _GapProfile((*summary.missing_information, *summary.failure_reasons))
    signals = {
        "pipeline": gaps.pipeline,
        "conflict": bool(summary.conflicts),
        "evidence": gaps.evidence,
    }
    for cause in rule_set.escalation_precedence:
        if signals[cause]:
            return _ESCALATION_BY_CAUSE[cause]
    raise ValueError("an escalated rules 2.0 summary must record its escalation cause")


def aggregate_findings(
    *,
    run_id: str,
    case_id: str,
    findings: Iterable[Finding | Mapping[str, object]],
    upstream_failures: Iterable[str] = (),
    input_errors: Iterable[str] = (),
    rule_set: RuleSet = RULE_SET,
) -> ReviewSummary:
    """Aggregate findings deterministically under the explicitly selected contract.

    The default rule set is the frozen rules 1.0 table, so every historical caller and every
    replay of a batch recorded under it keeps producing the same summary byte for byte.
    """

    if rule_set.gate_mode == "all_gaps":
        return _aggregate_rules_v1(
            run_id=run_id,
            case_id=case_id,
            findings=findings,
            upstream_failures=upstream_failures,
            input_errors=input_errors,
        )
    return _aggregate_rules_v2(
        run_id=run_id,
        case_id=case_id,
        findings=findings,
        upstream_failures=upstream_failures,
        input_errors=input_errors,
        rule_set=rule_set,
    )


def _aggregate_rules_v1(
    *,
    run_id: str,
    case_id: str,
    findings: Iterable[Finding | Mapping[str, object]],
    upstream_failures: Iterable[str] = (),
    input_errors: Iterable[str] = (),
) -> ReviewSummary:
    """Aggregate findings deterministically and fail closed on every uncertainty.

    Invalid raw finding objects are not allowed to disappear silently: they are
    recorded as an input failure and force an insufficient-information result.
    """

    validated, has_invalid_finding = _validated_findings(findings)
    deduplicated = deduplicate_findings(validated)
    required_keys = RULE_SET.required_finding_keys

    present_keys = {finding.finding_key for finding in deduplicated}
    missing_information: list[str] = [
        f"missing_required_finding:{key}"
        for key in required_keys
        if key not in present_keys
    ]
    if not deduplicated:
        missing_information.append("no_valid_findings")

    for finding in deduplicated:
        missing_information.extend(finding.missing_information)
        if finding.status in RULE_SET.fail_closed_finding_statuses:
            missing_information.append(
                f"unresolved_finding:{finding.finding_key}:{finding.status.value}"
            )

    for finding_key, finding_contract in sorted(_REQUIRED_FINDING_CONTRACTS.items()):
        if finding_key not in required_keys or finding_key not in present_keys:
            continue
        key_findings = [
            finding
            for finding in deduplicated
            if finding.finding_key == finding_key
            and finding.worker is finding_contract.worker
            and finding.category is finding_contract.category
        ]
        if not key_findings:
            missing_information.append(
                f"required_finding_source_mismatch:{finding_key}"
            )
            continue
        has_required_evidence = any(
            finding.evidence_refs
            if finding_contract.evidence_kind == "evidence_ref"
            else finding.business_fact_refs
            for finding in key_findings
            if finding.status is FindingStatus.SUPPORTED
        )
        if not has_required_evidence:
            missing_information.append(
                f"critical_evidence_missing:{finding_key}:{finding_contract.evidence_kind}"
            )

    failure_reasons = list(upstream_failures)
    failure_reasons.extend(input_errors)
    if has_invalid_finding:
        failure_reasons.append("invalid_finding_input")
    for finding in deduplicated:
        if finding.status in {
            FindingStatus.UPSTREAM_FAILURE,
            FindingStatus.INVALID_INPUT,
        }:
            failure_reasons.append(
                f"finding_failure:{finding.finding_key}:{finding.status.value}"
            )

    conflicts = _find_conflicts(deduplicated)
    highest_risk = merge_risk(finding.risk_level for finding in deduplicated)
    if (
        conflicts
        and _RISK_RANK[highest_risk] < _RISK_RANK[RULE_SET.conflict_minimum_risk]
    ):
        highest_risk = RULE_SET.conflict_minimum_risk

    normalized_missing = _unique_sorted(missing_information)
    normalized_failures = _unique_sorted(failure_reasons)
    if conflicts:
        recommendation = RecommendationStatus(RULE_SET.conflict_recommendation)
        human_review_status = HumanReviewStatus.REQUIRED
    elif normalized_missing or normalized_failures:
        recommendation = RecommendationStatus(RULE_SET.incomplete_recommendation)
        human_review_status = HumanReviewStatus.REQUIRED
    else:
        recommendation = _RECOMMENDATION_BY_HIGHEST_RISK[highest_risk]
        human_review_status = HumanReviewStatus.PENDING_CONFIRMATION

    return ReviewSummary(
        run_id=run_id,
        case_id=case_id,
        rules_version=RULE_SET.rules_version,
        findings=deduplicated,
        highest_risk=highest_risk,
        recommendation=recommendation,
        human_review_status=human_review_status,
        conflicts=conflicts,
        missing_information=normalized_missing,
        failure_reasons=normalized_failures,
    )


def _aggregate_rules_v2(
    *,
    run_id: str,
    case_id: str,
    findings: Iterable[Finding | Mapping[str, object]],
    upstream_failures: Iterable[str],
    input_errors: Iterable[str],
    rule_set: RuleSet,
) -> ReviewSummary:
    """Split the frozen gate into the four R-2 outcome classes.

    A detected conflict always escalates for adjudication (R-4, and the frozen summary
    invariant that a conflict means ``needs_human_review``). Otherwise only ``pipeline`` and
    ``evidence`` causes gate a recommendation, and an evidence gap never blocks a rejection a
    supported high risk already justifies: approval is the one direction where missing
    evidence is dangerous. Pipeline and worker-boundary failures keep closing their branch
    before this function is reached, and stay recorded in ``failure_reasons`` regardless of
    which class the summary ends up in.
    """

    validated, has_invalid_finding = _validated_findings(findings)
    deduplicated = deduplicate_findings(validated)
    rank = {level: index for index, level in enumerate(rule_set.risk_order)}
    pipeline: list[str] = []
    evidence: list[str] = []
    coverage: list[str] = []
    notes: list[str] = []
    buckets = {
        CauseClass.PIPELINE: pipeline,
        CauseClass.EVIDENCE: evidence,
        CauseClass.COVERAGE: coverage,
        CauseClass.CONTENT_NOTE: notes,
    }

    def record(code: str, cause_class: CauseClass) -> None:
        """File one cause under its class namespace, or prove why it could not be filed."""

        if cause_class is CauseClass.CONTENT_NOTE:
            notes.append(code)
            return
        namespaced = namespaced_cause(code, cause_class)
        if namespaced_cause_class(namespaced) is cause_class:
            buckets[cause_class].append(namespaced)
            return
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
        pipeline.append(f"{PIPELINE_CAUSE_PREFIX}unlisted_cause:{digest}")
        notes.append(code)

    for code in (*upstream_failures, *input_errors):
        # The parameter itself carries the class: a caller-declared failure is a pipeline
        # failure whatever wording the caller used.
        record(code, CauseClass.PIPELINE)
    if has_invalid_finding:
        record("invalid_finding_input", CauseClass.PIPELINE)
    if not deduplicated:
        record("no_valid_findings", CauseClass.PIPELINE)

    for finding in deduplicated:
        for code in finding.missing_information:
            record(code, classify_cause(code))
        if finding.status in {
            FindingStatus.UPSTREAM_FAILURE,
            FindingStatus.INVALID_INPUT,
        }:
            record(
                f"finding_failure:{finding.finding_key}:{finding.status.value}",
                CauseClass.PIPELINE,
            )
        elif finding.status in rule_set.fail_closed_finding_statuses:
            record(
                f"unresolved_finding:{finding.finding_key}:{finding.status.value}",
                CauseClass.EVIDENCE,
            )

    covered: set[str] = set()
    for finding_key in rule_set.required_finding_keys:
        contract = rule_set.required_finding_contracts[finding_key]
        key_findings = [
            finding for finding in deduplicated if finding.finding_key == finding_key
        ]
        if not key_findings:
            record(f"missing_required_finding:{finding_key}", CauseClass.COVERAGE)
            continue
        matching = [
            finding
            for finding in key_findings
            if finding.worker is contract.worker
            and finding.category is contract.category
        ]
        if not matching:
            record(
                f"required_finding_source_mismatch:{finding_key}",
                CauseClass.COVERAGE,
            )
            continue
        supported = [
            finding for finding in matching if finding.status is FindingStatus.SUPPORTED
        ]
        if not supported:
            record(f"unsupported_required_finding:{finding_key}", CauseClass.COVERAGE)
            continue
        has_required_evidence = any(
            finding.evidence_refs
            if contract.evidence_kind == "evidence_ref"
            else finding.business_fact_refs
            for finding in supported
        )
        if not has_required_evidence:
            record(
                f"critical_evidence_missing:{finding_key}:{contract.evidence_kind}",
                CauseClass.COVERAGE,
            )
            continue
        covered.add(finding_key)
    missing_coverage = [
        key for key in rule_set.required_finding_keys if key not in covered
    ]

    conflicts = find_conflicts_v2(deduplicated, rule_set)
    highest_risk = max(
        (finding.risk_level for finding in deduplicated),
        key=rank.__getitem__,
        default=RiskLevel.UNKNOWN,
    )
    if conflicts and rank[highest_risk] < rank[rule_set.conflict_minimum_risk]:
        highest_risk = rule_set.conflict_minimum_risk

    if conflicts:
        recommendation = RecommendationStatus(rule_set.conflict_recommendation)
    elif pipeline:
        recommendation = RecommendationStatus(rule_set.incomplete_recommendation)
    else:
        recommendation = rule_set.recommendation_by_highest_risk[highest_risk]
        if (
            recommendation in rule_set.coverage_required_recommendations
            and missing_coverage
        ):
            for finding_key in missing_coverage:
                record(
                    f"coverage_incomplete:{finding_key}",
                    CauseClass.EVIDENCE,
                )
            recommendation = RecommendationStatus(rule_set.incomplete_recommendation)
        elif recommendation not in _CONTENT_RECOMMENDATIONS and not evidence:
            record("no_supported_finding", CauseClass.EVIDENCE)

    normalized_missing = _unique_sorted((*evidence, *coverage, *notes))
    normalized_failures = _unique_sorted(pipeline)
    human_review_status = (
        HumanReviewStatus.PENDING_CONFIRMATION
        if recommendation in _CONTENT_RECOMMENDATIONS
        else HumanReviewStatus.REQUIRED
    )
    return ReviewSummary(
        run_id=run_id,
        case_id=case_id,
        rules_version=rule_set.rules_version,
        findings=deduplicated,
        highest_risk=highest_risk,
        recommendation=recommendation,
        human_review_status=human_review_status,
        conflicts=conflicts,
        missing_information=normalized_missing,
        failure_reasons=normalized_failures,
    )
