"""Independent H preparation assertions at contract, PG, CLI and consumer boundaries."""

import asyncio
import copy
import json
import os
import secrets
from pathlib import Path
from collections import Counter

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from procurement_review.eval_h_contracts import HPlan
from procurement_review.eval_h_fixture import make_disposable
from procurement_review.eval_h_seal import HSeal
from procurement_review.eval_h_storage import write
from procurement_review.eval_h_runner import run, setup
from procurement_review.eval_h_mock import mock_transport
from procurement_review.eval_h_consumption import completion_proof, observation_failures
from procurement_review.eval_h_journal import Journal
from procurement_review.eval_h_score import rebuild as score, repeat_score
from procurement_review.eval_h_report import rebuild as report
from procurement_review.eval_h_audit import audit, freeze, replay, verify_freeze
from procurement_review.eval_h_material import seed_plan


def sealed(tmp_path, count=2, suffix=None):
    suffix = suffix or secrets.token_hex(5)
    value = make_disposable(
        tmp_path / ".local", count, batch_id="disposable-h-test-" + suffix
    )
    seal = HSeal(Path(value["root"]))
    seal.generate(
        seal.directory / "private/recipe.json",
        seal.directory / "private/seed.bin",
        seal.directory / "private/registry.json",
    )
    seal.seal()
    return seal


def test_formal_plan_capacity_without_generation(tmp_path):
    seal = sealed(tmp_path)
    body = seal.plan().model_dump(mode="json")
    body.update(
        mode="formal",
        batch_id="formal-h-offline-plan",
        case_ids=[f"V107-B-{index:04d}" for index in range(200)],
    )
    plan = HPlan.model_validate_json(json.dumps(body), strict=True)
    assert len(set(plan.observation_keys())) == 800
    assert Counter(key[2] for key in plan.observation_keys()) == {
        "multi_agent": 400,
        "single_agent": 400,
    }
    assert len(set(plan.task_keys())) == 1200
    assert plan.capacity() == {
        "cases": 200,
        "repeats": 2,
        "multi_observations": 400,
        "single_observations": 400,
        "first_sends": 1200,
        "physical_capacity": 2400,
        "max_per_task": 2,
    }
    body["case_ids"] = body["case_ids"][:199]
    with pytest.raises(ValueError, match="200_cases"):
        HPlan.model_validate_json(json.dumps(body), strict=True)


def test_business_chunk_locator_keeps_typed_digest_and_text_boundaries():
    from procurement_review.business_v3_types import BusinessEvidenceIdentity
    from procurement_review.record_guard import scan_record
    from pydantic import BaseModel

    digest = "110101199003070000" + "a" * 46
    locator = "chunk-" + digest

    def evidence(value):
        return BusinessEvidenceIdentity(
            source_id="source-h-safe",
            version="h.v1",
            locator=value,
            content_sha256="a" * 64,
        )

    assert scan_record(evidence(locator)).clean
    for value in ("110101199003070000", "chunk-110101199003070000", locator + "-text"):
        assert {item.category for item in scan_record(evidence(value)).findings} == {
            "national_id_pii"
        }

    class FreeText(BaseModel):
        value: str

    assert not scan_record(FreeText(value=locator)).clean


def test_seal_refuses_manual_completion_and_runner_never_opens_label(
    tmp_path, monkeypatch
):
    seal = sealed(tmp_path)
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.name in {"labels.json", "seed.bin", "commitment.key", "recipe.json"}:
            pytest.fail("runner accessor opened private generation material")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "open", guarded)
        assert len(seal.runner_cases()) == 2
        from procurement_review.eval_h_cli import preflight

        assert preflight(seal.directory, runner=True)["ready_for_formal"] is False
    with pytest.raises(ValueError, match="run_incomplete"):
        seal.complete()
    with pytest.raises(ValueError, match="expected:run_complete"):
        seal.release()
    assert seal.record()["state"] == "sealed"
    records = seed_plan(seal.plan(), seal.runner_cases()).records
    assert Counter(record.kind for record in records) == {
        "order": 2,
        "supplier": 2,
        "transaction": 2,
    }


@pytest.fixture(scope="module")
def pg_batch(tmp_path_factory):
    directory = tmp_path_factory.mktemp("h-pg")
    seal = sealed(directory, 12)
    database = "v107h_" + seal.plan().batch_id.replace("-", "_")
    base = os.environ.get("TEST_POSTGRES_DSN")
    assert base, "TEST_POSTGRES_DSN is required; H PostgreSQL checks never skip"
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{database}"'))
    url = make_url(base).set(database=database).render_as_string(hide_password=False)
    setup(seal.directory, url)
    requests = []
    try:
        original = Path.open

        def no_labels(path, *args, **kwargs):
            if path.name in {
                "labels.json",
                "seed.bin",
                "commitment.key",
                "recipe.json",
            }:
                pytest.fail("actual runner opened private generation material")
            return original(path, *args, **kwargs)

        with pytest.MonkeyPatch.context() as guard:
            guard.setattr(Path, "open", no_labels)
            result = asyncio.run(
                run(
                    seal.directory,
                    transport_override=mock_transport("material", requests),
                )
            )
        assert result["committed"] == 48
        yield seal, requests
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE "{database}" WITH (FORCE)'))
        admin.dispose()


def test_pg_full_consumer_chain_and_no_resend(pg_batch, tmp_path):
    seal, requests = pg_batch
    proof = completion_proof(seal.directory)
    assert proof["complete"], proof["failures"]
    assert proof["by_arm"] == {"multi_agent": 24, "single_agent": 24}
    assert len(requests) == 72
    assert all(
        "truth" not in json.dumps(body).lower()
        and "label" not in json.dumps(body).lower()
        for body in requests
    )
    second = []
    asyncio.run(
        run(seal.directory, transport_override=mock_transport("material", second))
    )
    assert second == []
    seal.complete()
    assert len(seal.release()) == 12
    scored = score(seal.directory)
    for seed in ("101", "202"):
        for arm in ("multi_agent", "single_agent"):
            item = scored["repeats"][seed][arm]
            assert item["q01"]["first"]["value"] == 1
            assert item["q02"]["all"]["value"] == 1, item["semantic_verdicts"]
            assert item["metrics"]["q03_recommendation"]["value"] == 1
            assert all(value["value"] == 1 for value in item["q08"].values())
    write(seal.directory / "score.json", scored)
    seal.scored(seal.directory / "score.json")
    rebuilt = report(seal.directory)
    assert rebuilt["schema_version"] == "eval-report.v6"
    assert rebuilt["Q-12"]["complete"] is True
    assert rebuilt["stage_acceptance"] is False
    assert rebuilt["safety"]["verified"] is False
    write(seal.directory / "result.json", rebuilt)
    assert replay(seal.directory)["verified"]
    assert audit(seal.directory)["audit_passed"]
    frozen = tmp_path / "freeze.json"
    freeze(seal.directory, frozen)
    assert verify_freeze(seal.directory, frozen)["verified"]


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("model", "attempt_model_identity"),
        ("bool_token", "attempt_usage_type"),
        ("source", "retrieval_source_identity:policy:original"),
        ("early_adoption", "adoption_before_request"),
        ("checkpoint", "product_checkpoint_projection:spans"),
    ],
)
def test_independent_product_consumption_counterexamples(pg_batch, mutation, expected):
    seal, _requests = pg_batch
    case = seal.runner_cases()[0]
    body = copy.deepcopy(
        Journal(seal.directory).committed(case.case_id, 101, "multi_agent")
    )
    if mutation == "model":
        body["attempts"][0]["receipt"]["receipt_model"] = "other-model"
    elif mutation == "bool_token":
        body["attempts"][0]["receipt"]["input_usage"] = True
    elif mutation == "source":
        body["retrieval"]["policy"]["original"][0]["content_sha256"] = "0" * 64
    elif mutation == "early_adoption":
        body["binding"]["checkpoints"][0]["adopted_at"] = "2020-01-01T00:00:00+00:00"
    else:
        cp = body["binding"]["checkpoints"][0]
        cp["result_payload"]["result"]["spans"] = []
    errors = observation_failures(seal.plan(), case, body)
    assert any(expected in error for error in errors), errors


def test_failed_and_missing_samples_keep_quality_denominators(pg_batch):
    seal, _ = pg_batch
    plan, cases = seal.plan(), seal.runner_cases()
    labels = tuple(
        __import__(
            "procurement_review.eval_h_fixture", fromlist=["fixture_label"]
        ).fixture_label(case)
        for case in __import__(
            "procurement_review.gold_v38_dev_quality", fromlist=["dev_quality_cases"]
        ).dev_quality_cases()
    )
    result = repeat_score(plan, cases, labels, [], 101)
    for arm, branches in (("multi_agent", 2), ("single_agent", 1)):
        item = result[arm]
        assert item["q01"]["first"]["denominator"] == 12 * branches
        assert item["q01"]["first"]["numerator"] == 0
        assert item["q02"]["all"]["denominator"] == 144
        assert item["metrics"]["q03_recommendation"]["value"] == 0
        assert item["q09"]["all_sample_p95_ms"] is None
        assert item["metrics"]["q09_pipeline"] == {
            "non_expected_pipeline_failures": 12,
            "planned_runs": 12,
        }
        assert item["gates"]["Q-09"] is False
        assert all(
            value["denominator"] == 12 and value["numerator"] == 0
            for value in item["q08"].values()
        )


@pytest.mark.parametrize("arm", ["multi_agent", "single_agent"])
@pytest.mark.parametrize("mutation", ["omit", "invent"])
def test_native_pipeline_failure_cannot_be_dropped_or_forged(pg_batch, arm, mutation):
    seal, _ = pg_batch
    case = seal.runner_cases()[0]
    body = copy.deepcopy(Journal(seal.directory).committed(case.case_id, 101, arm))
    assert observation_failures(seal.plan(), case, body) == []
    if mutation == "omit":
        body["row"].pop("pipeline_failures")
    else:
        body["row"]["pipeline_failures"] = ["transport"]
    errors = observation_failures(seal.plan(), case, body)
    assert any(error.endswith("projection:pipeline_failures") for error in errors), (
        errors
    )


def test_attempt_unknown_is_preserved_and_not_resent(tmp_path):
    seal = sealed(tmp_path)
    journal = Journal(seal.directory)
    journal.reserve(seal.plan().case_ids[0], 101, "single_agent", "joint", "a" * 64)
    with pytest.raises(ValueError, match="unsettled_attempt"):
        journal.reserve(seal.plan().case_ids[0], 101, "single_agent", "joint", "b" * 64)
    journal.recover_unknown(seal.plan().case_ids[0], 101, "single_agent")
    receipt = journal.attempts(seal.plan().case_ids[0], 101, "single_agent")[0][
        "receipt"
    ]
    assert receipt["usage_unknown"] is True and receipt["input_tokens"] is None
    assert receipt["error_code"] == "interrupted_unknown"
