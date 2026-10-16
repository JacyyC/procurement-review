"""Independently authored counterexamples for the B source-sufficiency defect."""

import hashlib
import json
from pathlib import Path
import pytest
from procurement_review.gold_v3_contracts import (
    GoldCasesV3File,
    ObligationState,
    payload_sha256,
)
from procurement_review.gold_v3_oracle import derive_truth as old_truth
from procurement_review.gold_v31 import revised_case, build_bundle
from procurement_review.gold_v31_contracts import (
    GoldCaseV31,
    ModelInputV31,
    EvidenceDocumentV31,
)
from procurement_review.gold_v31_oracle import derive_truth, source_sufficiency


def old_case():
    path = (
        Path(__file__).parents[1]
        / "src/procurement_review/resources/gold_set/cases.v3.json"
    )
    return GoldCasesV3File.model_validate_json(path.read_bytes()).cases[0]


def change_contract(case, text):
    doc = EvidenceDocumentV31(
        source_id=case.model_input.contract_document.source_id,
        section=case.model_input.contract_document.section,
        content=text,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    inp = ModelInputV31(
        question=case.model_input.question,
        policy_document=case.model_input.policy_document,
        contract_document=doc,
        business_facts=case.model_input.business_facts,
    )
    values = case.model_dump(mode="json", exclude={"source_checksum"})
    values["model_input"] = inp.model_dump(mode="json")
    values["source_checksum"] = payload_sha256(values)
    return GoldCaseV31.model_validate_json(json.dumps(values))


def test_old_boolean_and_generic_title_are_not_visible_terms():
    original = old_case()
    # This records the defect without rewriting the frozen old oracle or data.
    assert old_truth(original).obligations[8].state is ObligationState.SUPPORTED
    assert source_sufficiency(original) == {"O-05": False, "O-08": False, "O-09": False}
    fixed = derive_truth(original)
    assert all(fixed.obligations[i].state is ObligationState.UNKNOWN for i in (4, 7, 8))


@pytest.mark.parametrize(
    "remove,obligation",
    [
        ("买方", "O-05"),
        ("到货数量逐项相符且完好率100%", "O-08"),
        ("卖方免费修复或更换缺陷交付物", "O-08"),
        ("100000.00 CNY", "O-09"),
        ("双方书面签署后变更生效", "O-09"),
    ],
)
def test_missing_text_is_unknown_even_with_unchanged_positive_procurement_flags(
    remove, obligation
):
    base = revised_case(old_case())
    changed = change_contract(
        base, base.model_input.contract_document.content.replace(remove, "仅有标题")
    )
    result = derive_truth(changed)
    assert (
        next(o for o in result.obligations if o.obligation_id == obligation).state
        is ObligationState.UNKNOWN
    )
    assert changed.model_input.business_facts.liability_cap_present is True


def test_parallel_revision_preserves_frozen_quotas_and_replay_resources():
    root = Path(__file__).parents[1] / "src/procurement_review/resources/gold_set"
    paths = [
        root / name for name in ("cases.v3.json", "truth.v1.json", "manifest.v3.json")
    ]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    cases, truth, manifest = build_bundle()
    assert manifest.split_counts == {"dev": 120, "validation": 60}
    assert manifest.recommendation_counts["dev"]["recommend_approve"] == 24
    assert manifest.recommendation_counts["validation"]["recommend_approve"] == 12
    assert all(all(source_sufficiency(c).values()) for c in cases.cases)
    assert len(truth.truths) == 180 and all(
        c.split.value != "blind" for c in cases.cases
    )
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_revised_truth_and_ledger_have_explicit_non_mixable_lineage(tmp_path):
    from procurement_review.eval_v2_report import build_report_v4
    from procurement_review.eval_attempt_ledger import AttemptLedger
    from procurement_review.eval_v2_contracts import EvalRunManifestV2
    from pydantic import ValidationError

    corrected = derive_truth(revised_case(old_case()))
    report = build_report_v4((corrected,), ())
    assert report["gold_set_version"] == "gold-set.v3.1"
    assert report["truth_contract"] == "v1-07-truth.v1.1"
    with pytest.raises(ValueError, match="mixed"):
        build_report_v4((corrected, old_truth(old_case())), ())
    ledger = AttemptLedger.create(
        tmp_path / ".local/new.sqlite3",
        run_id="v1-07-revised-test",
        gold_manifest_sha256="a" * 64,
        configuration_sha256="b" * 64,
        max_requests=2,
        max_input_tokens=None,
        max_output_tokens=None,
        stage_id="V1-07C",
        gold_set_version="gold-set.v3.1",
    )
    record = ledger.manifest().model_dump(mode="json")
    assert record["truth_contract"] == "v1-07-truth.v1.1"
    record["truth_contract"] = "v1-07-truth.v1"
    with pytest.raises(ValidationError):
        EvalRunManifestV2.model_validate_json(json.dumps(record))


def test_recovery_snapshot_retains_blocked_remote_unknown_count(tmp_path):
    from procurement_review.eval_attempt_ledger import AttemptLedger

    ledger = AttemptLedger.create(
        tmp_path / ".local/unknown.sqlite3",
        run_id="v1-07-recovery-revision",
        gold_manifest_sha256="a" * 64,
        configuration_sha256="b" * 64,
        max_requests=2,
        max_input_tokens=None,
        max_output_tokens=None,
        stage_id="V1-07C",
        gold_set_version="gold-set.v3.1",
    )
    ledger.register_task("task:source", "c" * 64)
    attempt = ledger.reserve_attempt(
        "task:source",
        request_sha256="d" * 64,
        reserve_input_tokens=100,
        reserve_output_tokens=100,
    )
    ledger.mark_sent(attempt.attempt_id)
    ledger.recover()
    assert ledger.manifest().blocked_unknown_task_count == 1
    assert ledger.manifest().budget.unknown_usage_attempts == 1
