from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import pytest

from procurement_review.gold_v3 import (
    GoldV3Error,
    build_gold_v3,
    jaccard_5gram,
    load_bundle,
    validate_gold_v3,
)
from procurement_review.gold_v3_contracts import (
    EXPECTED_RECOMMENDATION_COUNTS,
    GoldCaseV3,
    GoldCasesV3File,
    GoldTruthV1File,
    ObligationState,
    TruthRecordV1,
    V3Recommendation,
    V3Split,
    payload_sha256,
)
from procurement_review.gold_v3_oracle import derive_truth, replace_fact


def _rehash_case(case: GoldCaseV3, **changes: object) -> GoldCaseV3:
    changed = case.model_copy(update=changes)
    values = changed.model_dump(mode="python", exclude={"source_checksum"})
    checksum_view = changed.model_dump(mode="json", exclude={"source_checksum"})
    return GoldCaseV3(**values, source_checksum=payload_sha256(checksum_view))


def _cases_file(cases: tuple[GoldCaseV3, ...]) -> GoldCasesV3File:
    return GoldCasesV3File(
        cases=cases,
        payload_sha256=payload_sha256(
            {"cases": [item.model_dump(mode="json") for item in cases]}
        ),
    )


def _truth_file_for_changed_case(
    original: GoldTruthV1File, changed_case: GoldCaseV3
) -> GoldTruthV1File:
    truths = list(original.truths)
    index = next(
        index
        for index, item in enumerate(truths)
        if item.case_id == changed_case.case_id
    )
    changed = truths[index].model_copy(
        update={"case_source_checksum": changed_case.source_checksum}
    )
    values = changed.model_dump(mode="python", exclude={"truth_checksum"})
    checksum_view = changed.model_dump(mode="json", exclude={"truth_checksum"})
    truths[index] = TruthRecordV1(
        **values, truth_checksum=payload_sha256(checksum_view)
    )
    return GoldTruthV1File(
        truths=tuple(truths),
        payload_sha256=payload_sha256(
            {"truths": [item.model_dump(mode="json") for item in truths]}
        ),
    )


def test_committed_bundle_is_reproducible_and_has_frozen_quotas() -> None:
    committed = load_bundle()
    assert committed == build_gold_v3()
    cases, truth, manifest = committed
    assert len(cases.cases) == len(truth.truths) == 180
    assert manifest.split_counts == {V3Split.DEV: 120, V3Split.VALIDATION: 60}
    split_by_case = {item.case_id: item.split for item in cases.cases}
    for split, expected in EXPECTED_RECOMMENDATION_COUNTS.items():
        observed = Counter(
            item.expected_recommendation
            for item in truth.truths
            if split_by_case[item.case_id] is split
        )
        assert observed == expected
    assert manifest.high_risk_counts == {V3Split.DEV: 36, V3Split.VALIDATION: 18}
    assert manifest.conflict_counts == {V3Split.DEV: 18, V3Split.VALIDATION: 9}
    assert manifest.security_counts == {V3Split.DEV: 18, V3Split.VALIDATION: 9}
    assert {
        split: {item.language for item in cases.cases if item.split is split}
        for split in V3Split
    } == {V3Split.DEV: {"zh-CN"}, V3Split.VALIDATION: {"en-US"}}


def test_visible_cases_are_answer_free_and_truth_is_separate() -> None:
    cases, truths, _ = load_bundle()
    encoded = json.dumps(cases.model_dump(mode="json"), ensure_ascii=False).lower()
    for forbidden in (
        "expected_recommendation",
        "truth_checksum",
        "adjudication_source",
        "recommend_approve",
        "recommend_reject",
        "needs_human_review",
    ):
        assert forbidden not in encoded
    assert {item.case_id for item in cases.cases} == {
        item.case_id for item in truths.truths
    }
    assert all(item.required_evidence for item in truths.truths)


def test_independent_oracle_does_not_import_production_rules_or_aggregator() -> None:
    source_path = (
        Path(__file__).parents[1] / "src" / "procurement_review" / "gold_v3_oracle.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any("rules" in name or "aggreg" in name for name in imports)


def test_single_field_metamorphs_change_only_the_expected_business_semantics() -> None:
    cases, truths, _ = load_bundle()
    truth_by_id = {item.case_id: item for item in truths.truths}
    base = next(
        item
        for item in cases.cases
        if truth_by_id[item.case_id].expected_recommendation is V3Recommendation.APPROVE
    )

    blocked = derive_truth(replace_fact(base, procurement_status="blocked"))
    assert blocked.expected_recommendation is V3Recommendation.REJECT
    assert blocked.obligations[2].state is ObligationState.VIOLATED

    missing = derive_truth(replace_fact(base, competitive_quote_count=None))
    assert missing.expected_recommendation is V3Recommendation.INSUFFICIENT
    assert missing.obligations[3].state is ObligationState.UNKNOWN

    violated = derive_truth(replace_fact(base, competitive_quote_count=2))
    assert violated.expected_recommendation is V3Recommendation.CONDITIONAL
    assert violated.obligations[3].state is ObligationState.VIOLATED

    conflict = derive_truth(replace_fact(base, invoice_currency="USD"))
    assert conflict.expected_recommendation is V3Recommendation.HUMAN
    assert len(conflict.conflicts) == 1


@pytest.mark.parametrize(
    "field",
    ["generation_family_id", "near_duplicate_group_id", "entity_root_id"],
)
def test_cross_split_group_leak_is_rejected(field: str) -> None:
    cases, truths, manifest = load_bundle()
    dev = next(item for item in cases.cases if item.split is V3Split.DEV)
    validation_index = next(
        index
        for index, item in enumerate(cases.cases)
        if item.split is V3Split.VALIDATION
    )
    changed = list(cases.cases)
    changed_case = _rehash_case(
        changed[validation_index], **{field: getattr(dev, field)}
    )
    changed[validation_index] = changed_case
    changed_truths = _truth_file_for_changed_case(truths, changed_case)
    with pytest.raises(GoldV3Error, match=f"{field} crosses splits"):
        validate_gold_v3(_cases_file(tuple(changed)), changed_truths, manifest)


def test_declared_near_duplicate_groups_equal_jaccard_components() -> None:
    cases, _, manifest = load_bundle()
    assert manifest.near_duplicate_group_count == 12
    texts = {
        case.case_id: "\n".join(
            (
                case.model_input.question,
                case.model_input.policy_document.content,
                case.model_input.contract_document.content,
            )
        )
        for case in cases.cases
    }
    for index, left in enumerate(cases.cases):
        for right in cases.cases[index + 1 :]:
            edge = jaccard_5gram(texts[left.case_id], texts[right.case_id]) >= 0.82
            same_group = left.near_duplicate_group_id == right.near_duplicate_group_id
            assert edge is same_group


def test_near_duplicate_edge_cannot_span_declared_groups() -> None:
    cases, truths, manifest = load_bundle()
    changed = list(cases.cases)
    changed_case = _rehash_case(
        changed[0], near_duplicate_group_id=changed[1].near_duplicate_group_id
    )
    changed[0] = changed_case
    changed_truths = _truth_file_for_changed_case(truths, changed_case)
    with pytest.raises(
        GoldV3Error,
        match="(identical template|near-duplicate component) spans .*groups",
    ):
        validate_gold_v3(_cases_file(tuple(changed)), changed_truths, manifest)


def test_stage_b_bundle_contains_no_formal_blind_case() -> None:
    cases, _, _ = load_bundle()
    assert {item.split for item in cases.cases} == {
        V3Split.DEV,
        V3Split.VALIDATION,
    }


def test_manifest_summary_tampering_is_rejected_even_when_its_checksum_is_rebuilt() -> (
    None
):
    cases, truths, manifest = load_bundle()
    values = manifest.model_dump(mode="python", exclude={"manifest_sha256"})
    counts = dict(values["high_risk_counts"])
    counts[V3Split.DEV] = counts[V3Split.DEV] - 1
    values["high_risk_counts"] = counts
    view = manifest.model_construct(**values, manifest_sha256="0" * 64).model_dump(
        mode="json", exclude={"manifest_sha256"}
    )
    tampered = manifest.__class__(**values, manifest_sha256=payload_sha256(view))
    with pytest.raises(GoldV3Error, match="manifest high-risk counts mismatch"):
        validate_gold_v3(cases, truths, tampered)
