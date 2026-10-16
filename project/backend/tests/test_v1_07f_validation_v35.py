"""Replacement group isolation, immutable prior evidence and independent truth."""

import hashlib
import json
import pytest
from pydantic import ValidationError
from procurement_review.eval_f_context import data_directory, load_cases
from procurement_review.eval_f_registration import fixed_registration
from procurement_review.eval_f_score import load_truth, score_batch
from procurement_review.eval_f_validation import claim_validation
from procurement_review.eval_v2_contracts import EvalRunManifestV2
from procurement_review.gold_v35 import bundle
from procurement_review.gold_v35_oracle import derive_truth


def test_new_group_is_frozen_disjoint_and_keeps_dev_and_quotas():
    cases, truth, manifest, audit = bundle()
    assert manifest.split_counts == {"dev": 120, "validation": 60}
    assert audit["prior_validation_max_jaccard"] < 0.82
    assert audit["network_calls"] == audit["formal_blind"] == 0
    for name, model in zip(
        ("cases.json", "truth.json", "manifest.json"), (cases, truth, manifest)
    ):
        assert json.loads(
            (data_directory("gold-set.v3.5") / name).read_text()
        ) == model.model_dump(mode="json")
    old = {c.case_id: c for c in load_cases(version="gold-set.v3.4")}
    for c in cases.cases:
        if c.split.value == "dev":
            assert (
                c.model_input.business_facts
                == old[c.case_id].model_input.business_facts
            )
            assert (
                c.model_input.policy_document.content
                == old[c.case_id].model_input.policy_document.content
            )
            assert (
                c.model_input.contract_document.content
                == old[c.case_id].model_input.contract_document.content
            )
    assert {
        k.value: v for k, v in manifest.recommendation_counts["validation"].items()
    } == {
        "recommend_approve": 12,
        "recommend_conditional": 9,
        "recommend_reject": 12,
        "insufficient_information": 9,
        "needs_human_review": 18,
    }
    expected = {
        "cases.json": "157180e41caf4e12ab2cbcaa4309c7db371e516502652e485c52530b57cfbdbc",
        "manifest.json": "fb33a2a6a1b0f3b54327b0feb029509fd9e414a87eeeecfa3a9d052743b66284",
        "truth.json": "b9482068265445d7819e06e2602cf308c2586f02bbf1c875d2ea89c7341e0f57",
    }
    for name, digest in expected.items():
        assert (
            hashlib.sha256(
                (data_directory("gold-set.v3.4") / name).read_bytes()
            ).hexdigest()
            == digest
        )


def test_new_group_does_not_reopen_old_groups_and_wrong_truth_cannot_score(tmp_path):
    for version in (
        "gold-set.v3.1",
        "gold-set.v3.2",
        "gold-set.v3.3",
        "gold-set.v3.4",
        "gold-set.v3.5",
    ):
        claim_validation(
            tmp_path, fixed_registration(version), "validation-concurrency"
        )
    assert len(list((tmp_path / "consumed-validation").glob("*.json"))) == 5
    ids = [c.case_id for c in load_cases("validation", "gold-set.v3.5")]
    with pytest.raises(KeyError):
        score_batch([], ids, [301], "gold-set.v3.4")
    assert (
        score_batch([], ids, [301, 302], "gold-set.v3.5")["combined"]["counts"][
            "missing_cases"
        ]
        == 120
    )
    assert all(
        derive_truth(c) == load_truth("gold-set.v3.5")[c.case_id]
        for c in load_cases("validation", "gold-set.v3.5")
    )


@pytest.mark.parametrize(
    "label,duty",
    [
        ("Acceptance standard:", "O-08"),
        ("Warranty responsibility:", "O-08"),
        ("Damages:", "O-09"),
        ("Liability cap:", "O-09"),
    ],
)
def test_heading_without_terms_cannot_supply_truth(label, duty):
    c = load_cases("validation", "gold-set.v3.5")[0]
    d = c.model_input.contract_document
    text = "\n".join(
        line.split(label)[0] + label if label in line else line
        for line in d.content.splitlines()
    )
    c = c.model_copy(
        update={
            "model_input": c.model_input.model_copy(
                update={"contract_document": d.model_copy(update={"content": text})}
            )
        }
    )
    assert (
        next(
            o for o in derive_truth(c).obligations if o.obligation_id == duty
        ).state.value
        == "unknown"
    )


def test_c_cannot_masquerade_as_new_f_dataset(tmp_path):
    from procurement_review.eval_attempt_ledger import AttemptLedger

    manifest = AttemptLedger.create(
        tmp_path / ".local" / "ledger.sqlite3",
        run_id="v1-07-new-group",
        gold_manifest_sha256="a" * 64,
        configuration_sha256="b" * 64,
        max_requests=480,
        max_input_tokens=None,
        max_output_tokens=None,
        gold_set_version="gold-set.v3.5",
        stage_id="V1-07F",
    ).manifest()
    with pytest.raises(ValidationError):
        EvalRunManifestV2.model_validate(manifest.model_dump() | {"stage_id": "V1-07C"})
