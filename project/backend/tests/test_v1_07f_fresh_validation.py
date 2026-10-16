"""Fresh data isolation and truth checks; no real requests and no tuning here."""

import hashlib
import json

import pytest
from pydantic import ValidationError

from procurement_review.eval_f_context import data_directory, load_cases
from procurement_review.eval_f_registration import fixed_registration
from procurement_review.eval_f_score import load_truth, score_batch
from procurement_review.eval_f_validation import claim_validation
from procurement_review.gold_v32_contracts import CasesFileV32
from procurement_review.gold_v33 import bundle
from procurement_review.gold_v33_oracle import derive_truth


def test_v33_materialization_quotas_originals_and_dev_remain_fixed():
    cases, truth, manifest, audit = bundle()
    assert manifest.split_counts == {"dev": 120, "validation": 60}
    assert audit["prior_validation_max_jaccard"] < 0.82
    assert audit["network_calls"] == audit["formal_blind"] == 0
    for name, model in zip(
        ("cases.json", "truth.json", "manifest.json"), (cases, truth, manifest)
    ):
        assert json.loads(
            (data_directory("gold-set.v3.3") / name).read_text()
        ) == model.model_dump(mode="json")
    old = {c.case_id: c for c in load_cases(version="gold-set.v3.2")}
    old_truth = load_truth("gold-set.v3.2")
    for c, t in zip(cases.cases, truth.truths):
        if c.split.value == "dev":
            before = old[c.case_id].model_input
            assert c.model_input.question == before.question
            assert c.model_input.business_facts == before.business_facts
            assert (
                c.model_input.policy_document.content == before.policy_document.content
            )
            assert (
                c.model_input.contract_document.content
                == before.contract_document.content
            )
            assert (
                t.expected_recommendation
                == old_truth[c.case_id].expected_recommendation
            )
    expected_hashes = {
        "cases.json": "65806f87f0c006790b8758802f53ec46f2550258d9068d1af528f80d6ca4d6b5",
        "manifest.json": "6ab9244dc1f9d0c3f87116ab1eedcaeeb8f229462d47b85a6d7b1c7fbe158876",
        "truth.json": "e6ca8fdce882bc0da15dec014f54a17b3bdb30e8ed33cfd52f858827ba85cc4b",
    }
    for name, digest in expected_hashes.items():
        assert (
            hashlib.sha256(
                (data_directory("gold-set.v3.2") / name).read_bytes()
            ).hexdigest()
            == digest
        )


def test_v33_requires_its_own_truth_and_does_not_reopen_old_groups(tmp_path):
    with pytest.raises(ValidationError):
        CasesFileV32.model_validate_json(
            (data_directory("gold-set.v3.3") / "cases.json").read_bytes()
        )
    for version in ("gold-set.v3.1", "gold-set.v3.2", "gold-set.v3.3"):
        claim_validation(
            tmp_path, fixed_registration(version), "validation-concurrency"
        )
    assert len(list((tmp_path / "consumed-validation").glob("*.json"))) == 3
    ids = [c.case_id for c in load_cases("validation", "gold-set.v3.3")]
    with pytest.raises(KeyError):
        score_batch([], ids, [301], "gold-set.v3.2")
    report = score_batch([], ids, [301, 302], "gold-set.v3.3")
    assert report["combined"]["counts"]["missing_cases"] == 120
    assert not report["diagnostic_passed"]


@pytest.mark.parametrize(
    "text,duty",
    [
        ("Acceptance standard:", "O-08"),
        ("Warranty responsibility:", "O-08"),
        ("Damages:", "O-09"),
        ("Liability cap:", "O-09"),
    ],
)
def test_fresh_truth_requires_actual_terms_not_a_title(text, duty):
    c = load_cases("validation", "gold-set.v3.3")[0]
    d = c.model_input.contract_document
    missing = "\n".join(
        line.split(text)[0] + text if text in line else line
        for line in d.content.splitlines()
    )
    altered = c.model_copy(
        update={
            "model_input": c.model_input.model_copy(
                update={"contract_document": d.model_copy(update={"content": missing})}
            )
        }
    )
    expected = derive_truth(altered)
    assert (
        next(o for o in expected.obligations if o.obligation_id == duty).state.value
        == "unknown"
    )
