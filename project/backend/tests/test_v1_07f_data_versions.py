"""Version and group isolation, independent source truth, and immutable old data."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from procurement_review.eval_attempt_ledger import LedgerError
from procurement_review.eval_f_context import data_directory, load_cases, stable_write
from procurement_review.eval_f_registration import registration
from procurement_review.eval_f_score import load_truth, score_batch
from procurement_review.eval_f_validation import claim_validation
from procurement_review.gold_v31_contracts import CasesFileV31
from procurement_review.gold_v32 import bundle
from procurement_review.gold_v32_oracle import derive_truth


def test_new_bundle_retains_dev_content_and_semantics_and_separates_validation():
    cases, truth, manifest, audit = bundle()
    old_dev = {c.case_id: c for c in load_cases()}
    old_truth = load_truth()
    for case, expected in zip(cases.cases, truth.truths):
        if case.case_id in old_dev:
            old = old_dev[case.case_id]
            assert case.model_input.question == old.model_input.question
            assert case.model_input.business_facts == old.model_input.business_facts
            assert (
                case.model_input.policy_document.content
                == old.model_input.policy_document.content
            )
            assert (
                case.model_input.contract_document.content
                == old.model_input.contract_document.content
            )
            assert (
                expected.expected_recommendation
                == old_truth[case.case_id].expected_recommendation
            )
            assert [o.state for o in expected.obligations] == [
                o.state for o in old_truth[case.case_id].obligations
            ]
    assert sum(manifest.split_counts.values()) == 180
    assert audit["prior_validation_max_jaccard"] < 0.82
    assert audit["formal_blind"] == audit["network_calls"] == 0
    assert not (
        {c.case_id for c in load_cases("validation")}
        & {c.case_id for c in load_cases("validation", "gold-set.v3.2")}
    )
    for name, value in zip(
        ("cases.json", "truth.json", "manifest.json"), (cases, truth, manifest)
    ):
        assert json.loads(
            (data_directory("gold-set.v3.2") / name).read_text()
        ) == value.model_dump(mode="json")


def test_new_source_truth_does_not_accept_missing_contract():
    # Verify oracle behavior; never feed validation text into a model or tune a recognizer here.
    case = load_cases("validation", "gold-set.v3.2")[0]
    doc = case.model_input.contract_document.model_copy(
        update={"content": "Contract terms unavailable"}
    )
    missing = case.model_copy(
        update={
            "model_input": case.model_input.model_copy(
                update={"contract_document": doc}
            )
        }
    )
    outcome = derive_truth(missing)
    assert all(
        o.state.value == "unknown"
        for o in outcome.obligations
        if o.obligation_id in {"O-05", "O-08", "O-09"}
    )
    assert outcome.expected_recommendation.value != "recommend_approve"


def test_cross_version_scoring_never_guesses_dataset():
    with pytest.raises(ValidationError):
        CasesFileV31.model_validate_json(
            (data_directory("gold-set.v3.2") / "cases.json").read_bytes()
        )
    with pytest.raises(KeyError):
        score_batch([], [load_cases("validation", "gold-set.v3.2")[0].case_id], [1])
    with pytest.raises(ValueError, match="unsupported"):
        load_cases(version="../../other")
    report = score_batch(
        [], [load_cases("validation", "gold-set.v3.2")[0].case_id], [1], "gold-set.v3.2"
    )
    assert not report["diagnostic_passed"]


def test_legacy_group_stays_consumed_and_new_group_is_one_use(tmp_path):
    prior = registration()
    old = dict(
        registration="old-registration",
        candidate="validation-baseline",
        case_ids=prior["validation"],
    )
    stable_write(tmp_path / "validation-consumed.json", old)
    before = (tmp_path / "validation-consumed.json").read_bytes()
    with pytest.raises(LedgerError, match="already consumed"):
        claim_validation(tmp_path, prior, "validation-baseline")
    new = registration(data_version="gold-set.v3.2")
    claim_validation(tmp_path, new, "validation-baseline")
    claim_validation(tmp_path, new, "validation-baseline")  # same execution can resume
    assert (tmp_path / "validation-consumed.json").read_bytes() == before
    with pytest.raises(LedgerError, match="already consumed"):
        claim_validation(tmp_path, new, "validation-compact")
    assert len(list((tmp_path / "consumed-validation").glob("*.json"))) == 1


def test_independent_truth_has_no_production_dependency():
    import procurement_review.gold_v32_oracle as oracle

    text = Path(oracle.__file__).read_text()
    assert "business_v3" not in text


def test_new_dataset_runs_and_replays_with_its_own_ledger_contract(tmp_path):
    import asyncio
    import httpx
    from procurement_review.eval_f_registration import prepare
    from procurement_review.eval_f_run import execute, replay
    from procurement_review.eval_attempt_ledger import AttemptLedger
    from procurement_review.gold_v3_contracts import payload_sha256
    from v1_07f_support import response_for

    path = tmp_path / "registration.json"
    prepare(path, data_version="gold-set.v3.2")
    root = tmp_path / ".local/live"
    result = asyncio.run(
        execute(
            path,
            "minimum",
            transport_override=httpx.MockTransport(response_for),
            offline_root=root,
        )
    )
    registered = json.loads(path.read_text())
    folder = root / payload_sha256(registered) / "minimum"
    manifest = AttemptLedger(folder / "ledger.sqlite3").manifest()
    assert manifest.gold_set_version == "gold-set.v3.2"
    assert manifest.truth_contract == "v1-07-truth.v1.2"
    assert result["diagnostic_passed"]
    assert replay(folder)["repeats"] == result["repeats"]
