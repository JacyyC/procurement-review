"""V1-08B: the offline C probe shape (Spec P-03) — registered, sized and measured.

The C probe cannot be the historical two-arm two-repeat grid: that shape answers a
comparison question on 48–72 first sends, while the frozen probe answers the capability
question on 16–24.  These tests pin the probe's own version, arithmetic and identity and
then measure one offline batch to show the registered counts are what actually leaves:
the product arm alone, two model branches per Case, one initial observation per Case.
Nothing here sends to a supplier, opens a label or claims a capability.
"""

import asyncio
import hashlib
import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from procurement_review.eval_h_consumption import completion_proof
from procurement_review.eval_h_contracts import (
    C_PROBE_SEED,
    H_PLAN_V2,
    H_PROBE,
    HPlanV2,
    HProbePlan,
    PROBE_GENERATION_DEADLINE_SECONDS,
    load_h_plan,
    plan_generation_deadline,
)
from procurement_review.eval_h_fixture import make_disposable
from procurement_review.eval_h_mock import mock_transport
from procurement_review.eval_h_runner import run, setup
from procurement_review.eval_h_seal import HSeal, plan_digest
from procurement_review.precise_dev_paired import paired_configuration


def probe_batch(tmp_path, suffix, count, **overrides):
    values = dict(
        plan_version=H_PROBE,
        mode="probe",
        thinking="enabled",
        generation_deadline_seconds=PROBE_GENERATION_DEADLINE_SECONDS,
        rate_limit_window_seconds=60,
        rate_limit_burst_threshold=8,
    )
    values.update(overrides)
    value = make_disposable(
        tmp_path / ".local", count, batch_id="probe-h-" + suffix, **values
    )
    return HSeal(Path(value["root"]))


def plan_base(batch_id, *, case_ids, repeat_seeds, deadline, **overrides):
    values = dict(
        batch_id=batch_id,
        mode="probe",
        candidate_sha256="a" * 64,
        consumer_sha256="b" * 64,
        product_sha256="c" * 64,
        generator_spec_sha256="d" * 64,
        seed_sha256="e" * 64,
        exclusion_sha256="f" * 64,
        case_ids=case_ids,
        repeat_seeds=repeat_seeds,
        configuration=paired_configuration().model_copy(update={"thinking": "enabled"}),
        generation_deadline_seconds=deadline,
        rate_limit_window_seconds=60,
        rate_limit_burst_threshold=8,
        document_version="doc.v1",
        dataset_version="data.v1",
        index_version="index.v1",
        tenant_id="tenant",
        subject_id="subject",
        authorization="explicit",
        lifecycle_evidence_sha256="1" * 64,
    )
    values.update(overrides)
    return values


def role_of(payload):
    user = json.loads(
        next(
            message["content"]
            for message in payload["messages"]
            if message["role"] == "user"
        )
    )
    return "joint" if "tasks" in user else user["worker"]


def test_probe_plan_arithmetic_matches_the_frozen_c_shape(tmp_path):
    """8–12 Cases → 16–24 first sends and at most 48 physical attempts, product arm only."""

    expected = {8: (16, 32), 12: (24, 48)}
    for count, (first_sends, physical) in expected.items():
        seal = probe_batch(tmp_path, f"arith-{count}", count)
        plan = seal.plan()
        assert isinstance(plan, HProbePlan)
        assert plan.schema_version == H_PROBE
        assert plan_generation_deadline(plan) == PROBE_GENERATION_DEADLINE_SECONDS
        capacity = plan.capacity()
        assert capacity == {
            "cases": count,
            "repeats": 1,
            "multi_observations": count,
            "single_observations": 0,
            "first_sends": first_sends,
            "physical_capacity": physical,
            "max_per_task": 2,
        }
        assert plan.arms() == ("multi_agent",)
        keys = plan.observation_keys()
        assert len(keys) == count
        assert {key[2] for key in keys} == {"multi_agent"}
        assert all(key[1] == C_PROBE_SEED for key in keys)
        # Two model branches per Case and never a joint task: the single Agent arm and the
        # second repeat are not part of the probe.
        assert len(plan.task_keys()) == count * 2
        assert {branch for _case, _seed, _arm, branch in plan.task_keys()} == {
            "policy",
            "contract",
        }
        # The forbidden grid at the probe's own case counts: the two-arm two-repeat shape
        # would plan 48–72 first sends, which is exactly what the probe refuses to be.
        grid = HPlanV2(
            **plan_base(
                f"probe-h-grid-{count}",
                case_ids=plan.case_ids,
                repeat_seeds=(101, 202),
                deadline=120.0,
            )
        )
        assert grid.capacity()["first_sends"] == count * 6
        assert grid.capacity()["first_sends"] not in range(16, 25)


def test_probe_registration_refuses_grid_or_unregistered_parameters(tmp_path):
    """A probe without its own registered parameters is refused, never defaulted."""

    with pytest.raises(ValueError, match="h_fixture_probe_parameters_required"):
        probe_batch(tmp_path, "missing", 8, thinking=None)
    with pytest.raises(ValueError, match="h_fixture_probe_shape"):
        probe_batch(tmp_path, "count-seven", 7)
    with pytest.raises(ValueError, match="h_fixture_count"):
        probe_batch(tmp_path, "count-thirteen", 13)
    with pytest.raises(ValueError, match="h_probe_deadline_required"):
        probe_batch(tmp_path, "short-deadline", 8, generation_deadline_seconds=120.0)
    with pytest.raises(ValueError, match="h_probe_thinking_candidate_required"):
        probe_batch(tmp_path, "disabled-thinking", 8, thinking="disabled")
    # A V1-08 grid batch still requires its own parameters; the probe version is the only
    # one that may register a single observation per Case.
    with pytest.raises(ValueError, match="h_fixture_v2_parameters_required"):
        make_disposable(
            tmp_path / ".local",
            8,
            batch_id="disposable-h-still-v2",
            plan_version=H_PLAN_V2,
            thinking="enabled",
            generation_deadline_seconds=180.0,
        )

    cases = probe_batch(tmp_path, "read-back", 8).plan().case_ids
    with pytest.raises(
        ValidationError, match="h_probe_single_registered_seed_required"
    ):
        HProbePlan(
            **plan_base(
                "probe-h-two-seeds",
                case_ids=cases,
                repeat_seeds=(C_PROBE_SEED, 202),
                deadline=PROBE_GENERATION_DEADLINE_SECONDS,
            )
        )
    with pytest.raises(ValidationError, match="h_probe_deadline_required"):
        HProbePlan(
            **plan_base(
                "probe-h-240",
                case_ids=cases,
                repeat_seeds=(C_PROBE_SEED,),
                deadline=240.0,
            )
        )
    with pytest.raises(ValidationError):
        HProbePlan(
            **plan_base(
                "probe-h-window",
                case_ids=cases,
                repeat_seeds=(C_PROBE_SEED,),
                deadline=PROBE_GENERATION_DEADLINE_SECONDS,
                rate_limit_window_seconds=0,
            )
        )
    with pytest.raises(ValidationError):
        HProbePlan(
            **plan_base(
                "probe-h-thinking",
                case_ids=cases,
                repeat_seeds=(C_PROBE_SEED,),
                deadline=PROBE_GENERATION_DEADLINE_SECONDS,
                configuration=paired_configuration().model_copy(
                    update={"thinking": "disabled"}
                ),
            )
        )
    with pytest.raises(ValueError, match="h_plan_version_unknown"):
        load_h_plan(json.dumps({"schema_version": "v1-08d-plan.v1"}).encode())


def test_probe_identity_covers_thinking_deadline_and_stop_parameters(tmp_path):
    """The registered thinking, deadline and stop parameters are part of the identity."""

    baseline = probe_batch(tmp_path, "identity-base", 8).plan()
    assert baseline.stop_policy() == type(baseline.stop_policy())(60, 8)
    assert baseline.configuration.thinking == "enabled"
    assert baseline.mode == "probe"
    assert baseline.stage_id == "V1-08C"
    assert baseline.generation_deadline_seconds == PROBE_GENERATION_DEADLINE_SECONDS
    # The stop parameters are registered, not defaults: two plans that differ only in the
    # dense-429 window or threshold are two different identities.
    other_window = HProbePlan(
        **plan_base(
            baseline.batch_id,
            case_ids=baseline.case_ids,
            repeat_seeds=(C_PROBE_SEED,),
            deadline=PROBE_GENERATION_DEADLINE_SECONDS,
            rate_limit_window_seconds=30,
        )
    )
    assert plan_digest(other_window) != plan_digest(baseline)
    assert other_window.stop_policy() != baseline.stop_policy()
    # The shape itself is part of the identity: the same content under the grid version is
    # a different plan, so a probe can never be read as one of its observations.
    grid = HPlanV2(
        **plan_base(
            baseline.batch_id,
            case_ids=baseline.case_ids,
            repeat_seeds=(101, 202),
            deadline=PROBE_GENERATION_DEADLINE_SECONDS,
        )
    )
    assert plan_digest(grid) != plan_digest(baseline)


@contextmanager
def runnable(tmp_path, suffix, count, **overrides):
    """One sealed probe batch with its dedicated database."""

    seal = probe_batch(tmp_path, suffix, count, **overrides)
    seal.generate(
        seal.directory / "private/recipe.json",
        seal.directory / "private/seed.bin",
        seal.directory / "private/registry.json",
    )
    seal.seal()
    base = os.environ["TEST_POSTGRES_DSN"]
    name = "v107h_" + seal.plan().batch_id.replace("-", "_")
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(base).set(database=name).render_as_string(hide_password=False)
    try:
        setup(seal.directory, url)
        yield seal, url
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.mark.parametrize("count,first_sends", [(8, 16), (12, 24)])
def test_probe_offline_run_measures_only_the_product_arm(
    tmp_path, count, first_sends, monkeypatch
):
    """One offline probe batch sends exactly ``Cases × 2`` and records only that.

    A probe batch runs under the product's real execution profile — its Runs freeze
    ``mode=real`` — so the process profile is set the same way, exactly as the historical
    formal batch and any real probe must be composed.
    """

    monkeypatch.setenv("REVIEW_EXECUTION_MODE", "real")
    with runnable(tmp_path, f"run-{count}", count) as (seal, _url):
        requests = []
        result = asyncio.run(
            run(
                seal.directory,
                transport_override=mock_transport("material", requests),
            )
        )
        plan = seal.plan()
        assert result["stopped"] is False
        assert result["committed"] == count
        # Exactly the registered first sends left, and both model branches of every Case
        # are among them: one request per branch, no retry and no joint arm.
        assert len(requests) == first_sends
        roles = [role_of(payload) for payload in requests]
        assert roles.count("policy") == count
        assert roles.count("contract") == count
        assert "joint" not in roles
        observations = {
            path.name: json.loads(path.read_bytes())
            for path in sorted((seal.directory / "observations").glob("*.json"))
        }
        assert len(observations) == count
        assert {body["arm"] for body in observations.values()} == {"multi_agent"}
        assert {body["repeat_seed"] for body in observations.values()} == {C_PROBE_SEED}
        for body in observations.values():
            # The Run's own frozen snapshot carries the registered configuration, so the
            # probe's identity covers thinking, model, prompt/schema/wire and deadline.
            snapshot = body["binding"]["snapshot"]
            assert snapshot["thinking"] == "enabled"
            assert snapshot["generation_deadline_seconds"] == 180.0
            assert snapshot["requested_model"] == plan.configuration.requested_model
            assert snapshot["resolved_model"] == plan.configuration.resolved_model
            assert snapshot["wire_version"] and snapshot["prompt_version"]
            assert snapshot["schema_sha256"] and snapshot["implementation_sha256"]
            assert sorted(item["branch"] for item in body["attempts"]) == [
                "contract",
                "policy",
            ]
            assert len(body["attempts"]) <= 2


def observations_of(seal):
    return {
        path.name: json.loads(path.read_bytes())
        for path in sorted((seal.directory / "observations").glob("*.json"))
    }


def rate_limited_wire(sent):
    """A wire whose every answer is a 429; ``sent`` counts the physical sends."""

    def respond(request):
        sent.append(role_of(json.loads(request.content)))
        return httpx.Response(429, json={"error": "slow down"})

    return httpx.MockTransport(respond)


def test_probe_batch_stops_at_its_own_registered_threshold(tmp_path, monkeypatch):
    """The probe's registered window/threshold really ends sending at the threshold.

    A probe whose policy was not loaded ran the whole grid: every 429 stayed an ordinary
    retry and all sixteen tasks spent their retries.  The assertions are on the physical
    sends of the real runner.  The bound is the admission boundary the contract states:
    the two Cases the semaphore has admitted may still be winding down their in-flight
    attempts when the threshold is reached, and nothing beyond them may start.
    """

    monkeypatch.setenv("REVIEW_EXECUTION_MODE", "real")
    with runnable(tmp_path, "stop-policy", 8) as (seal, _url):
        sent = []
        result = asyncio.run(
            run(seal.directory, transport_override=rate_limited_wire(sent))
        )
        assert result["stop_policy"] == {
            "rate_limit_window_seconds": 60,
            "rate_limit_burst_threshold": 8,
        }
        assert result["stopped"] is True
        assert result["stop_reason"] == "rate_limited_dense"
        assert result["stop_summary"]["trigger"] == {
            "window_seconds": 60,
            "threshold": 8,
            "observed": 8,
        }
        # The unloaded policy sent 32 times; the registered threshold bounds the sends to
        # itself plus at most one attempt per already-admitted Case.
        assert 8 <= len(sent) <= 10, sent
        assert result["committed"] <= 2, observations_of(seal).keys()
        proof = completion_proof(seal.directory)
        assert proof["complete"] is False
        assert "batch_stopped_record:rate_limited_dense" in proof["failures"]
        missing = [
            failure
            for failure in proof["failures"]
            if failure.startswith("missing_observation")
        ]
        assert len(missing) == 8 - result["committed"] >= 6, proof["failures"]
        for body in observations_of(seal).values():
            # The admitted attempts were settled, and every branch stayed inside its own
            # shared budget instead of the queue restarting underneath the stop.
            assert all(item["status"] == "settled" for item in body["attempts"])
            counted = Counter(item["branch"] for item in body["attempts"])
            assert set(counted) <= {"policy", "contract"}
            assert all(count <= 2 for count in counted.values())


def test_probe_stop_gate_keeps_the_next_branch_and_queued_cases_unsent(
    tmp_path, monkeypatch
):
    """At the threshold the next branch is gated, not sent, and queued Cases never start."""

    monkeypatch.setenv("REVIEW_EXECUTION_MODE", "real")
    with runnable(tmp_path, "stop-gate", 8, rate_limit_burst_threshold=2) as (
        seal,
        _url,
    ):
        sent = []
        result = asyncio.run(
            run(seal.directory, transport_override=rate_limited_wire(sent))
        )
        assert result["stop_reason"] == "rate_limited_dense"
        assert result["stop_summary"]["trigger"]["threshold"] == 2
        # Every answer was a 429, so no policy branch could complete and no contract
        # branch may start at all; the two admitted Cases bound the in-flight sends.
        assert 2 <= len(sent) <= 4, sent
        assert set(sent) == {"policy"}, sent
        assert result["committed"] <= 2, observations_of(seal).keys()
        for body in observations_of(seal).values():
            assert all(item["status"] == "settled" for item in body["attempts"])
            counted = Counter(item["branch"] for item in body["attempts"])
            assert set(counted) <= {"policy"}
            assert all(count <= 2 for count in counted.values())
            assert body["row"]["error_code"] in {
                "batch_stopped",
                "review_execution_failed",
            }


def test_probe_retries_inside_the_registered_window_and_finishes(tmp_path, monkeypatch):
    """Below the threshold a 429 is retried once per task and the probe completes."""

    monkeypatch.setenv("REVIEW_EXECUTION_MODE", "real")
    with runnable(tmp_path, "retry-429", 8, rate_limit_burst_threshold=64) as (
        seal,
        _url,
    ):
        material = mock_transport("material")
        sent, counts = [], {}

        def respond(request):
            payload = json.loads(request.content)
            sent.append(role_of(payload))
            # One key per logical task: a transport retry repeats the same request body.
            key = hashlib.sha256(request.content).hexdigest()
            counts[key] = counts.get(key, 0) + 1
            if counts[key] == 1:
                return httpx.Response(
                    429, headers={"retry-after": "0"}, json={"error": "slow down"}
                )
            return material.handler(request)

        result = asyncio.run(
            run(seal.directory, transport_override=httpx.MockTransport(respond))
        )
        assert result["stopped"] is False
        assert result["stop_reason"] is None
        # Every one of the sixteen tasks was rate-limited exactly once and retried inside
        # its own shared two-attempt budget; none of them reached the registered threshold.
        assert len(counts) == 16, sorted(counts.values())
        assert set(counts.values()) == {2}, counts
        assert result["stop_summary"]["attempts_noted"] == 16
        assert (
            sum(
                1
                for note in result["stop_summary"]["notes"]
                if note["code"] == "rate_limited"
            )
            == 16
        ), result["stop_summary"]["notes"]
        # Eight Cases, two branches each, every task rate-limited once and retried once.
        assert len(sent) == 32, sent
        assert sent.count("policy") == 16 and sent.count("contract") == 16
        assert result["committed"] == 8
        proof = completion_proof(seal.directory)
        assert proof["complete"], proof["failures"]
        assert proof["observed"] == proof["planned"] == 8
