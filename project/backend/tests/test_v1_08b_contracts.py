"""V1-08B: the trusted failure projection, the stop contract and the V1-08 plan.

These tests pin the independent expectations of the repair:

* a Run that ended without a summary keeps its place only while its own provenance —
  frozen snapshot, lifecycle event, closed generation, attempt ledger and timing — still
  proves that is what happened, and every form of damage is refused by name;
* an interrupted Run is a terminal, measurable outcome, and only the registered
  interruption reasons let the batch continue;
* the stop contract reaches the senders: a stopped batch starts no new physical attempt
  (including a queued case and a branch's next attempt), while an attempt already in
  flight keeps its settlement;
* a V1-08 batch registers its thinking value and one accepted Run deadline, both arms
  execute the same configuration, and a request that omits or oversteps them is refused.
"""

import asyncio
import copy
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from procurement_review.eval_h_consumption import (
    completion_proof,
    observation_failures,
)
from procurement_review.eval_h_contracts import (
    H_PLAN_V2,
    HPlanV2,
    load_h_plan,
    plan_generation_deadline,
)
from procurement_review.eval_h_fixture import make_disposable
from procurement_review.eval_h_joint import execute
from procurement_review.eval_h_journal import Journal
from procurement_review.eval_h_mock import mock_transport
from procurement_review.eval_h_runner import assembled, run, setup
from procurement_review.eval_h_seal import HSeal
from procurement_review.eval_h_storage import read
from procurement_review.run_stop import (
    BatchStopController,
    BatchStoppedSignal,
    StopPolicy,
    stop_level,
)


@contextmanager
def batch(tmp_path, suffix, *, count=2, contract=None):
    """One generated, sealed disposable batch; ``contract`` selects the V1-08 plan."""

    value = make_disposable(
        tmp_path / ".local",
        count,
        batch_id="disposable-h-" + suffix,
        **(contract or {}),
    )
    seal = HSeal(Path(value["root"]))
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


V2 = {
    "plan_version": H_PLAN_V2,
    "thinking": "enabled",
    "generation_deadline_seconds": 120.0,
    "rate_limit_window_seconds": 10,
    "rate_limit_burst_threshold": 2,
}


def observations(seal):
    return {
        path.name: json.loads(path.read_bytes())
        for path in sorted((seal.directory / "observations").glob("*.json"))
    }


def case_of(seal, case_id):
    return next(item for item in seal.runner_cases() if item.case_id == case_id)


def test_registered_failures_must_keep_their_provenance(tmp_path):
    class Once:
        enabled = True
        fired = False

        def barrier(self, site):
            if site == "during_summary_commit" and not self.fired:
                self.fired = True
                raise RuntimeError("controlled summary failure")

    with batch(tmp_path, "failed-run-projection") as (seal, _):
        asyncio.run(
            run(
                seal.directory,
                transport_override=mock_transport("material"),
                fault=Once(),
            )
        )
        plan = seal.plan()
        proof = completion_proof(seal.directory)
        # The failed Run is an outcome, not a damaged record: it keeps its place.
        assert proof["complete"], proof["failures"]
        assert len(proof["product_outcomes"]) == 1, proof["product_outcomes"]
        name = proof["product_outcomes"][0].split(":", 1)[0]
        assert proof["product_outcomes"] == [
            f"{name}:product_run_failed:review_execution_failed"
        ]
        body = observations(seal)[name]
        assert body["row"]["status"] == "failed"

        def damaged(mutate):
            changed = copy.deepcopy(body)
            mutate(changed)
            return observation_failures(
                plan, case_of(seal, changed["case_id"]), changed
            )

        # Erasing the failure's provenance is refused by name.
        assert damaged(
            lambda item: item["binding"].update(
                snapshot=None, events=[], attempts=[], checkpoints=[], generations=[]
            )
        )
        # Contradicting the recorded status is refused by name.
        assert "product_failed_status_binding" in damaged(
            lambda item: item["binding"].update(status="processing")
        )
        # A fabricated (zeroed) duration no longer matches the persisted timestamps.
        assert "product_failed_elapsed" in damaged(
            lambda item: item.update(elapsed_ms=0)
        )
        # Deleting the recorded attempts while keeping their projection disagrees.
        assert "product_failed_attempt_ledger" in damaged(
            lambda item: item["binding"].update(attempts=[])
        )
        # An unregistered failure code cannot be read as a measured outcome.
        assert any(
            error.startswith("product_failed_code_unregistered")
            for error in damaged(
                lambda item: (
                    item["row"].update(error_code="something_new"),
                    item["binding"].update(error_code="something_new"),
                )
            )
        )
        # A successful record relabelled as an empty failure loses its lifecycle event
        # and its generation closure, and is refused instead of being accepted.
        success = next(
            body
            for name, body in observations(seal).items()
            if name.endswith("single_agent.json")
        )
        changed = copy.deepcopy(success)
        changed["row"].update(status="failed", error_code="review_execution_failed")
        changed["binding"].update(status="failed", error_code="review_execution_failed")
        assert observation_failures(plan, case_of(seal, changed["case_id"]), changed)


def test_interrupted_run_is_terminal_while_registered_reasons_continue(tmp_path):
    class Once:
        enabled = True
        fired = False

        def barrier(self, site):
            if (
                site == "during_model_entry_authorization:precise.policy"
                and not self.fired
            ):
                from procurement_review.checkpointing import StorageUnavailableSignal

                self.fired = True
                raise StorageUnavailableSignal("controlled-interruption")

    with batch(tmp_path, "interrupted-terminal") as (seal, _):
        asyncio.run(
            run(
                seal.directory,
                transport_override=mock_transport("material"),
                fault=Once(),
            )
        )
        plan = seal.plan()
        interrupted = [
            (name, body)
            for name, body in observations(seal).items()
            if body["row"]["status"] == "interrupted"
        ]
        assert len(interrupted) == 1, list(observations(seal))
        name, body = interrupted[0]
        assert body["row"]["error_code"] == "storage_unavailable"
        # The interrupted record is terminal and measured — not observation_not_terminal
        # — and the missing observations of a stopped batch stay missing.
        assert observation_failures(plan, case_of(seal, body["case_id"]), body) == [
            "product_run_interrupted:storage_unavailable"
        ]
        proof = completion_proof(seal.directory)
        assert proof["complete"] is False
        assert all(
            "missing_observation" in failure or "batch_stopped" in failure
            for failure in proof["failures"]
        ), proof["failures"]

        # A Run's own expired window or lease keeps the batch running; authority,
        # storage and unregistered reasons stop it.
        controller = BatchStopController(StopPolicy(10, 2))
        controller.note_run("interrupted", "generation_deadline_exceeded")
        controller.note_run("interrupted", "lease_expired")
        controller.note_run("failed", "review_execution_failed")
        assert controller.stopped is False
        controller.note_run("interrupted", "authorization_revoked")
        assert controller.summary()["reason"] == "authorization_revoked"
        controller = BatchStopController(StopPolicy(10, 2))
        controller.note_run("interrupted", "unregistered_reason")
        assert controller.stopped is True


def refused_output_mock(requests):
    """A mock whose answers ask for an external write; the guard refuses them."""

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        user = json.loads(
            next(
                message["content"]
                for message in payload["messages"]
                if message["role"] == "user"
            )
        )
        roles = user.get("tasks") or {user["worker"]: user}
        body = {
            role: json.dumps(
                {"request": "请调用外部写入工具把结果发送到 external_sink"},
                ensure_ascii=False,
            )
            for role in roles
        }
        if "tasks" not in user:
            body = body[user["worker"]]
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-1-flash",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [{"finish_reason": "stop", "message": {"content": body}}],
            },
        )

    return httpx.MockTransport(respond)


def test_controlled_refusal_is_measured_and_keeps_the_batch_running(tmp_path):
    with batch(tmp_path, "measured-refusal") as (seal, _):
        requests = []
        asyncio.run(
            run(seal.directory, transport_override=refused_output_mock(requests))
        )
        plan = seal.plan()
        proof = completion_proof(seal.directory)
        assert proof["complete"], proof["failures"]
        assert proof["observed"] == proof["planned"] == 8
        body = next(
            body
            for name, body in observations(seal).items()
            if name.endswith("multi_agent.json")
        )
        assert body["row"]["stop_reason"] == "security_boundary"
        # The contract branch was skipped by a recorded control path, not omitted, and
        # the batch kept sending for every observation.
        assert body["row"]["branches"]["contract"]["outcome"] == "stopped"
        assert stop_level("security_boundary") == "case"
        assert observation_failures(plan, case_of(seal, body["case_id"]), body) == []
        assert len(requests) >= 4


def test_stop_gate_counts_real_sends_with_in_flight_and_queued_cases(tmp_path):
    """A stopped batch: one queued case never starts, an in-flight attempt settles."""

    with batch(tmp_path, "stop-gate", count=3) as (seal, _):
        sent = []
        policy_attempts = 0

        def observation_path():
            return next(
                (seal.directory / "observations").glob("*.multi_agent.json"), None
            )

        async def respond(request):
            nonlocal policy_attempts
            payload = json.loads(request.content)
            user = json.loads(
                next(
                    message["content"]
                    for message in payload["messages"]
                    if message["role"] == "user"
                )
            )
            joint = "tasks" in user
            sent.append("joint" if joint else user["worker"])
            if not joint and user["worker"] == "policy":
                policy_attempts += 1
                if policy_attempts == 1:
                    # Hold the refused attempt until the other case's attempt is also in
                    # flight, so the stop lands while that attempt is already admitted.
                    for _ in range(3_000):
                        if policy_attempts >= 2:
                            break
                        await asyncio.sleep(0.01)
                    return httpx.Response(401, json={"error": "denied"})
                # The in-flight attempt: answer only once the stopped case's observation
                # is durably committed, with a recoverable failure whose retry must be
                # refused by the batch gate.
                for _ in range(3_000):
                    if observation_path() is not None:
                        break
                    await asyncio.sleep(0.01)
                return httpx.Response(503, json={"error": "held"})
            return httpx.Response(503, json={"error": "held"})

        result = asyncio.run(
            run(seal.directory, transport_override=httpx.MockTransport(respond))
        )
        assert result["stopped"] is True
        assert result["stop_reason"] == "authentication"
        # Exactly two physical sends happened in this seed: the refused first attempt and
        # the already-admitted in-flight attempt.  The queued case never started, neither
        # case sent a contract branch, no retry left the gate, and no joint arm ran.
        assert sent == ["policy", "policy"], sent
        bodies = observations(seal)
        assert len(bodies) == 2, list(bodies)
        in_flight = [
            body for body in bodies.values() if body["row"]["status"] == "interrupted"
        ]
        refused = [
            body for body in bodies.values() if body["row"]["status"] != "interrupted"
        ]
        assert refused and refused[0]["row"]["stop_reason"] == "authentication"
        # The in-flight attempt kept its settlement and its Run was interrupted by the
        # batch stop instead of being cancelled or left unsettled.
        assert in_flight and in_flight[0]["row"]["error_code"] == "batch_stopped"
        assert len(in_flight[0]["attempts"]) == 1
        assert in_flight[0]["attempts"][0]["status"] == "settled"


def test_v2_batch_registers_thinking_and_deadline_for_both_arms(tmp_path):
    with batch(tmp_path, "v2-contract", contract=V2) as (seal, url):
        requests = []
        asyncio.run(
            run(
                seal.directory,
                transport_override=mock_transport("material", requests),
            )
        )
        plan = seal.plan()
        assert isinstance(plan, HPlanV2)
        assert plan_generation_deadline(plan) == 120.0
        assert plan.stop_policy() == StopPolicy(10, 2)
        proof = completion_proof(seal.directory)
        assert proof["complete"], proof["failures"]
        bodies = observations(seal)
        assert len(bodies) == 8
        product_bodies = [
            body for body in bodies.values() if body["arm"] == "multi_agent"
        ]
        assert len(product_bodies) == 4
        for body in product_bodies:
            assert body["row"]["status"] in {
                "pending_confirmation",
                "pending_human_review",
            }, body["row"]
            snapshot = body["binding"]["snapshot"]
            assert snapshot["thinking"] == "enabled"
            assert snapshot["generation_deadline_seconds"] == 120.0
        # Both arms execute the registered thinking configuration on the wire.
        thinking = [payload["thinking"]["type"] for payload in requests]
        assert thinking and set(thinking) == {"enabled"}
        # The authorization scope carries the registered Run deadline.
        from procurement_review.eval_h_product import admission
        from procurement_review.review_store import ReviewStore

        store = ReviewStore.from_url(url)
        try:
            state = read(seal.directory / "private/runtime.json")
            _context, _registration, scope = admission(
                store, state["token"], plan, plan.case_ids[0]
            )
            window = (scope.deadline_at - datetime.now(UTC)).total_seconds()
            assert 119 <= window <= 121, window
        finally:
            store.close()


def test_v2_registration_refuses_missing_or_unaccepted_parameters(tmp_path):
    with pytest.raises(ValueError, match="h_fixture_v2_parameters_required"):
        make_disposable(
            tmp_path / ".local",
            1,
            batch_id="disposable-h-missing",
            plan_version=H_PLAN_V2,
            thinking="enabled",
        )
    with pytest.raises(ValueError, match="h_fixture_contract_parameter_without_v2"):
        make_disposable(
            tmp_path / ".local",
            1,
            batch_id="disposable-h-v1-extra",
            thinking="enabled",
        )
    from procurement_review.precise_dev_paired import paired_configuration

    base = dict(
        mode="disposable",
        batch_id="disposable-h-bad",
        candidate_sha256="a" * 64,
        consumer_sha256="b" * 64,
        product_sha256="c" * 64,
        generator_spec_sha256="d" * 64,
        seed_sha256="e" * 64,
        exclusion_sha256="f" * 64,
        case_ids=("V107-D-9001",),
        repeat_seeds=(101, 202),
        document_version="doc.v1",
        dataset_version="data.v1",
        index_version="index.v1",
        tenant_id="tenant",
        subject_id="subject",
        authorization="explicit",
        lifecycle_evidence_sha256="1" * 64,
        rate_limit_window_seconds=10,
        rate_limit_burst_threshold=2,
    )
    enabled = paired_configuration().model_copy(update={"thinking": "enabled"})
    with pytest.raises(ValueError, match="h_generation_deadline_not_accepted"):
        HPlanV2(configuration=enabled, generation_deadline_seconds=240.0, **base)
    with pytest.raises(ValueError, match="h_fair_model_parameters"):
        HPlanV2(
            configuration=paired_configuration().model_copy(update={"thinking": None}),
            generation_deadline_seconds=120.0,
            **base,
        )
    with pytest.raises(ValueError):
        HPlanV2(
            configuration=enabled,
            generation_deadline_seconds=120.0,
            **{**base, "rate_limit_window_seconds": 0},
        )
    with pytest.raises(ValueError, match="h_plan_version_unknown"):
        load_h_plan(json.dumps({"schema_version": "v1-09h-plan.v1"}).encode())


def test_joint_branch_never_sends_after_its_budget_or_the_gate(tmp_path):
    """The joint branch is capped by the deadline and gated by the batch stop."""

    with batch(tmp_path, "joint-gate") as (seal, _):
        from procurement_review.review_store import ReviewStore

        state = read(seal.directory / "private/runtime.json")
        store = ReviewStore.from_url(state["database_url"])
        journal = Journal(seal.directory)
        try:
            plan = seal.plan()
            case = seal.runner_cases()[0]
            context, facts, inputs, _retrieval, admit = asyncio.run(
                assembled(store, state["token"], plan, case)
            )
            responses = []

            async def scenario(stop):
                def counter(request):
                    responses.append(request)
                    return httpx.Response(500)

                from procurement_review.precise_run_transport import (
                    PreciseRunChatTransport,
                )

                transport = PreciseRunChatTransport(
                    plan.configuration.model_copy(update={"seed": 101}),
                    SecretStr("h-offline-only"),
                    catalog=None,
                    transport=httpx.MockTransport(counter),
                )
                return await execute(
                    journal,
                    plan,
                    case,
                    101,
                    transport,
                    inputs,
                    facts,
                    admit,
                    stop=stop,
                    deadline_seconds=0.0,
                )

            grounded, attempts = asyncio.run(scenario(None))
            assert grounded is None and attempts == [] and responses == []

            controller = BatchStopController(StopPolicy(10, 2))
            controller.stop("authentication")
            with pytest.raises(BatchStoppedSignal):
                asyncio.run(scenario(controller))
            assert responses == []
        finally:
            store.close()


def test_dense_rate_limit_stops_the_batch_inside_the_shared_budget(tmp_path):
    """Two 429s inside the registered window stop the batch after two real sends."""

    contract = {**V2, "rate_limit_window_seconds": 60, "rate_limit_burst_threshold": 2}
    with batch(tmp_path, "dense-429", count=1, contract=contract) as (seal, _):
        sent = []

        def respond(request):
            payload = json.loads(request.content)
            user = json.loads(
                next(
                    message["content"]
                    for message in payload["messages"]
                    if message["role"] == "user"
                )
            )
            sent.append("joint" if "tasks" in user else user["worker"])
            return httpx.Response(429, json={"error": "slow down"})

        result = asyncio.run(
            run(seal.directory, transport_override=httpx.MockTransport(respond))
        )
        assert result["stopped"] is True
        assert result["stop_reason"] == "rate_limited_dense"
        assert result["stop_summary"]["trigger"] == {
            "window_seconds": 60,
            "threshold": 2,
            "observed": 2,
        }
        # The shared two-attempt budget bounds the physical sends: the retry happened,
        # the next branch was gated, and the queued seed never started.
        assert sent == ["policy", "policy"], sent
        bodies = observations(seal)
        assert len(bodies) == 1, list(bodies)
        body = next(iter(bodies.values()))
        assert body["row"]["status"] == "interrupted"
        assert body["row"]["error_code"] == "batch_stopped"
        assert [item["status"] for item in body["attempts"]] == [
            "settled",
            "settled",
        ]
        assert body["attempts"][1]["effective_error"] == "rate_limited"


def test_retry_backoff_spends_the_remaining_deadline(tmp_path):
    """A bounded retry delay comes out of the Run's own remaining time."""

    import time as clock

    from procurement_review.precise_run_runner import _spend_backoff

    class Scope:
        deadline_at = datetime.now(UTC) + timedelta(seconds=0.05)

    before = clock.monotonic()
    asyncio.run(
        _spend_backoff(10.0, Scope.deadline_at, Scope(), lambda: datetime.now(UTC))
    )
    spent = clock.monotonic() - before
    # The ten-second delay was clamped to the remaining fifty milliseconds.
    assert 0.03 <= spent < 1.0, spent

    with batch(tmp_path, "backoff-deadline") as (seal, _):
        from procurement_review.review_store import ReviewStore

        state = read(seal.directory / "private/runtime.json")
        store = ReviewStore.from_url(state["database_url"])
        journal = Journal(seal.directory)
        try:
            plan = seal.plan()
            case = seal.runner_cases()[0]
            _context, facts, inputs, _retrieval, admit = asyncio.run(
                assembled(store, state["token"], plan, case)
            )
            responses = []

            async def scenario():
                from procurement_review.precise_run_transport import (
                    PreciseRunChatTransport,
                )

                def respond(request):
                    responses.append(request)
                    return httpx.Response(
                        429, headers={"retry-after": "5"}, json={"error": "slow"}
                    )

                transport = PreciseRunChatTransport(
                    plan.configuration.model_copy(update={"seed": 101}),
                    SecretStr("h-offline-only"),
                    catalog=None,
                    transport=httpx.MockTransport(respond),
                )
                return await execute(
                    journal,
                    plan,
                    case,
                    101,
                    transport,
                    inputs,
                    facts,
                    admit,
                    deadline_seconds=0.05,
                )

            grounded, attempts = asyncio.run(scenario())
            # The one retry the budget allowed was spent waiting out the deadline: no
            # second physical attempt left, and the wait did not extend the Run.
            assert len(responses) == 1
            assert [item["status"] for item in attempts] == ["settled"]
        finally:
            store.close()


def test_stop_classification_follows_the_frozen_priority_order(tmp_path):
    """Every Spec P-05 class is reachable, ordered and recorded, not just a field check."""

    from procurement_review.run_stop import (
        ACCOUNT_QUOTA,
        ACCOUNT_QUOTA_EXHAUSTED,
        CONTROLLED_REFUSAL,
        RECOVERABLE,
        TRUST_BOUNDARY,
        UNRECEIPTED,
        UNREGISTERED,
        UNREGISTERED_PROVIDER_ERROR,
        attempt_priority,
        priority_of,
        run_priority,
    )

    # Rank 1 — authority, credential, identity and drift end the batch.
    for code in ("model_identity_drift", "credential_in_request", "authentication"):
        controller = BatchStopController(StopPolicy(10, 2))
        controller.note_attempt(code, datetime.now(UTC))
        assert controller.stopped is True, code
        assert controller.priority == TRUST_BOUNDARY
    authority = BatchStopController(StopPolicy(10, 2))
    authority.note_run("interrupted", "authorization_revoked")
    assert authority.priority == TRUST_BOUNDARY

    # Rank 2 — an unregistered attempt outcome stops the batch under one fixed code while
    # the observed code itself stays in the protected note.
    unknown = BatchStopController(StopPolicy(10, 2))
    unknown.note_attempt("a_brand_new_attempt_error", datetime.now(UTC))
    assert unknown.stopped is True
    assert unknown.reason == UNREGISTERED_PROVIDER_ERROR
    assert unknown.trigger["attempt_error"] == "a_brand_new_attempt_error"
    assert unknown.notes[0]["priority"] == UNREGISTERED
    unregistered_reason = BatchStopController(StopPolicy(10, 2))
    unregistered_reason.note_stop_reason("a_brand_new_stop_reason")
    assert unregistered_reason.stopped is True
    assert unregistered_reason.priority == UNREGISTERED
    unregistered_failure = BatchStopController(StopPolicy(10, 2))
    unregistered_failure.note_run("failed", "a_brand_new_failure_code")
    assert unregistered_failure.reason == UNREGISTERED_PROVIDER_ERROR

    # Rank 3 — no trusted receipt and unsettled persistence stop the batch.
    for event in (
        lambda item: item.note_attempt("interrupted_unknown", datetime.now(UTC)),
        lambda item: item.note_run("interrupted", "storage_unavailable"),
    ):
        receipted = BatchStopController(StopPolicy(10, 2))
        event(receipted)
        assert receipted.stopped is True
        assert receipted.priority == UNRECEIPTED

    # Rank 4 — an account/quota refusal has its own class: it stops immediately and never
    # counts as a dense-429 event or waits for the window.
    quota = BatchStopController(StopPolicy(60, 64))
    quota.note_attempt(ACCOUNT_QUOTA_EXHAUSTED, datetime.now(UTC))
    assert quota.stopped is True
    assert quota.priority == ACCOUNT_QUOTA
    assert quota.summary()["rate_limited_events"] == 0

    # Rank 5 — a dense window counts only the registered 429 events inside it: a stale
    # event leaves the window, two events inside it stop the batch.
    dense = BatchStopController(StopPolicy(1, 2))
    dense.note_attempt("rate_limited", datetime.now(UTC) - timedelta(seconds=30))
    dense.note_attempt("rate_limited", datetime.now(UTC))
    assert dense.stopped is False
    dense.note_attempt("rate_limited", datetime.now(UTC))
    assert dense.stopped is True
    assert dense.summary()["trigger"]["observed"] == 2

    # Ranks 6 to 8 — recoverable transport failures, locatable unknown usage and the
    # registered per-branch outcomes are recorded and never stop the batch.
    measured = BatchStopController(StopPolicy(60, 64))
    measured.note_attempt("security_boundary", datetime.now(UTC))
    measured.note_attempt("invalid_schema", datetime.now(UTC))
    measured.note_attempt("unsupported_request", datetime.now(UTC))
    measured.note_attempt("deadline_exhausted", datetime.now(UTC))
    measured.note_attempt("record_guard_refused:record_contract", datetime.now(UTC))
    measured.note_attempt("rate_limited", datetime.now(UTC))
    measured.note_attempt("timeout", datetime.now(UTC))
    measured.note_attempt("transport", datetime.now(UTC))
    measured.note_attempt("server_error", datetime.now(UTC))
    measured.note_attempt(None, datetime.now(UTC))
    measured.note_run("interrupted", "generation_deadline_exceeded")
    measured.note_run("interrupted", "lease_expired")
    measured.note_run("failed", "review_execution_failed")
    measured.note_stop_reason("security_boundary")
    assert measured.stopped is False
    assert measured.summary()["rate_limited_events"] == 1
    assert measured.summary()["attempts_noted"] == 10
    assert measured.notes and {note["rank"] for note in measured.notes} <= {6, 8}
    for code in ("rate_limited", "timeout", "transport", "server_error"):
        assert attempt_priority(code) == RECOVERABLE, code
    for code in ("security_boundary", "invalid_schema", "unsupported_request"):
        assert attempt_priority(code) == CONTROLLED_REFUSAL, code
    assert run_priority("interrupted", "lease_expired") == RECOVERABLE
    assert run_priority("interrupted", "authorization_revoked") == TRUST_BOUNDARY
    assert run_priority("failed", "review_execution_failed") is None

    # One event may match several conditions: the most severe level decides the recorded
    # stop while every reason stays in the notes, and a later, less severe stop never
    # rewrites the reason.
    mixed = BatchStopController(StopPolicy(60, 2))
    mixed.note_run("interrupted", "storage_unavailable")
    assert mixed.reason == "storage_unavailable"
    mixed.note_attempt("authentication", datetime.now(UTC))
    assert mixed.reason == "authentication"
    assert mixed.priority == TRUST_BOUNDARY
    mixed.note_attempt("rate_limited", datetime.now(UTC))
    mixed.note_attempt("rate_limited", datetime.now(UTC))
    assert mixed.reason == "authentication"
    assert mixed.priority == TRUST_BOUNDARY
    assert {
        "storage_unavailable",
        "authentication",
        "rate_limited",
        "rate_limited_dense",
    } <= {note["code"] for note in mixed.notes}
    assert priority_of("rate_limited_dense") == "rate_limited_dense"
    assert priority_of(ACCOUNT_QUOTA_EXHAUSTED) == ACCOUNT_QUOTA


def test_accepted_deadline_candidates_drive_score_and_report(tmp_path):
    """Both accepted candidates score with their own gate; a legacy plan keeps 120/90."""

    from procurement_review.eval_h_score import deadline_gates, rebuild as score
    from procurement_review.eval_h_report import rebuild as report
    from procurement_review.eval_h_storage import write

    assert deadline_gates(object()) == (120000, 90000)
    for deadline, thinking, gates in (
        (120.0, "enabled", (120000, 90000)),
        (180.0, "disabled", (180000, 150000)),
    ):
        contract = {
            **V2,
            "thinking": thinking,
            "generation_deadline_seconds": deadline,
        }
        suffix = f"candidate-{int(deadline)}"
        with batch(tmp_path, suffix, count=1, contract=contract) as (seal, _):
            asyncio.run(
                run(seal.directory, transport_override=mock_transport("material"))
            )
            plan = seal.plan()
            assert deadline_gates(plan) == gates
            assert seal.complete()["state"] == "run_complete"
            seal.release()
            value = score(seal.directory)
            write(seal.directory / "score.json", value)
            seal.scored(seal.directory / "score.json")
            rebuilt = report(seal.directory)
            body = next(iter(value["repeats"].values()))["multi_agent"]
            assert body["q09"]["deadline_ms"] == gates[0]
            assert body["q09"]["censored"] == 0
            assert rebuilt["stage_acceptance"] is False


def test_reasoning_metadata_is_kept_without_the_reasoning_text(tmp_path):
    """The supplier's reasoning metadata is measured; the reasoning text is dropped."""

    from procurement_review.chat_transport_v2 import AttemptResult, decode_response
    from procurement_review.model_wire_v2 import (
        BranchRequestV2,
        CheckSlot,
        SourceSlice,
    )
    from procurement_review.gold_v3_contracts import PropositionIdentity
    from procurement_review.business_v3_types import BusinessEvidenceIdentity

    from datetime import date

    content = "参考正文"
    request = BranchRequestV2(
        case_id="V107-D-9001",
        worker="policy",
        scope_id="V107-SCOPE-1",
        question="question",
        checks=(
            CheckSlot(
                proposition=PropositionIdentity(
                    scope_id="V107-SCOPE-1",
                    subject_type="purchase_order",
                    subject_id="V107-PO-1",
                    check_id="O-01",
                    field_path="policy.applicability",
                    effective_at=date(2026, 6, 30),
                    source_version="doc.v1",
                ),
                instruction="instruction",
            ),
        ),
        sources=(
            SourceSlice(
                scope_id="V107-SCOPE-1",
                worker="policy",
                identity=BusinessEvidenceIdentity(
                    source_id="V107-SRC-1",
                    version="doc.v1",
                    locator="chunk-1",
                    content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                ),
                content=content,
            ),
        ),
    )
    from procurement_review.precise_dev_paired import paired_configuration

    config = paired_configuration()

    def response(message, usage):
        return httpx.Response(
            200,
            json={
                "model": config.resolved_model,
                "usage": usage,
                "choices": [{"finish_reason": "stop", "message": message}],
            },
        )

    secret_reasoning = "内部推理正文-should-never-be-retained"
    measured = decode_response(
        response(
            {"content": "{", "reasoning_content": secret_reasoning},
            {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "completion_tokens_details": {"reasoning_tokens": 25},
            },
        ),
        request,
        config,
        "",
    )
    assert measured.reasoning_tokens == 25
    assert measured.reasoning_bytes == len(secret_reasoning.encode())
    assert secret_reasoning not in repr(measured)
    # A response without the field records "not provided" instead of zero.
    silent = decode_response(
        response({"content": "{"}, {"prompt_tokens": 100, "completion_tokens": 40}),
        request,
        config,
        "",
    )
    assert silent.reasoning_tokens is None
    assert silent.reasoning_bytes is None
    assert isinstance(measured, AttemptResult) and isinstance(silent, AttemptResult)


def test_v2_batch_stores_no_reasoning_text_anywhere(tmp_path):
    """A thinking answer never leaks its reasoning into the recorded observation."""

    secret = "REASONING-正文-leak-canary"

    def respond(request):
        payload = json.loads(request.content)
        user = json.loads(
            next(
                message["content"]
                for message in payload["messages"]
                if message["role"] == "user"
            )
        )
        roles = user.get("tasks") or {user["worker"]: user}
        body = {role: "{" for role in roles}
        if "tasks" not in user:
            body = body[user["worker"]]
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-1-flash",
                "usage": {
                    "prompt_tokens": 700,
                    "completion_tokens": 260,
                    "completion_tokens_details": {"reasoning_tokens": 120},
                },
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": body, "reasoning_content": secret},
                    }
                ],
            },
        )

    with batch(tmp_path, "reasoning-canary", count=1, contract=V2) as (seal, _):
        asyncio.run(
            run(seal.directory, transport_override=httpx.MockTransport(respond))
        )
        for path in (seal.directory / "observations").glob("*.json"):
            assert secret not in path.read_text(encoding="utf-8")


def test_v2_resume_inherits_the_deadline_and_reuses_the_committed_branch(tmp_path):
    """An interrupted V1-08 Run resumes under its own registered snapshot."""

    class Once:
        enabled = True
        fired = False

        def barrier(self, site):
            if (
                site == "during_model_entry_authorization:precise.contract"
                and not self.fired
            ):
                from procurement_review.checkpointing import StorageUnavailableSignal

                self.fired = True
                raise StorageUnavailableSignal("h-v2-resume-outage")

    from procurement_review.eval_h_product import multi
    from procurement_review.eval_h_storage import read as read_json
    from procurement_review.precise_run_transport import PreciseRunChatTransport
    from procurement_review.review_store import ReviewStore

    with batch(tmp_path, "v2-resume", count=1, contract=V2) as (seal, url):
        state = read_json(seal.directory / "private/runtime.json")
        store = ReviewStore.from_url(url)
        requests = []
        transport = PreciseRunChatTransport(
            seal.plan().configuration.model_copy(update={"seed": 101}),
            SecretStr("h-offline-only"),
            catalog=None,
            transport=mock_transport("material", requests),
        )
        case = seal.runner_cases()[0]
        try:
            _row, binding = asyncio.run(
                multi(
                    store,
                    state["token"],
                    seal.plan(),
                    case,
                    101,
                    transport,
                    resume=True,
                    fault=Once(),
                )
            )
        finally:
            store.close()
        assert binding["resume_count"] == 1
        assert binding["execution"]["metering"]["reused_branches"] == 1
        assert len(binding["snapshot"]) > 0
        assert binding["snapshot"]["generation_deadline_seconds"] == 120.0
        assert binding["snapshot"]["thinking"] == "enabled"
        # The committed branch was not resent: one policy send before and after resume.
        assert binding["execution"]["metering"]["first_sends"] == 1, binding[
            "execution"
        ]["metering"]
        # The resumed generation carries the registered window, not a fresh default.
        with create_engine(url).connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT generation, EXTRACT(EPOCH FROM"
                    " (deadline_at - created_at)) AS window"
                    " FROM review_run_generation ORDER BY generation"
                )
            ).all()
        assert len(rows) == 2
        assert all(abs(float(row.window) - 120.0) < 0.5 for row in rows), rows


def test_frozen_snapshot_refuses_a_later_thinking_or_deadline_change(tmp_path):
    """Once a Run freezes its configuration, another thinking/deadline is refused."""

    from types import SimpleNamespace

    from procurement_review.checkpointing import IncompatibleCheckpointError
    from procurement_review.precise_run_scope import (
        execution_snapshot,
        require_frozen_snapshot,
    )

    with batch(tmp_path, "v2-frozen", count=1, contract=V2) as (seal, _):
        plan = seal.plan()
        registration = SimpleNamespace(
            rules_version="3.1.0",
            document_version=plan.document_version,
            dataset_version=plan.dataset_version,
            index_version=plan.index_version,
        )

        def snapshot(config, deadline):
            return execution_snapshot(
                mode="test_only",
                config=config,
                registration=registration,
                branch_timeout_seconds=90.0,
                max_attempts_per_branch=2,
                generation_deadline_seconds=deadline,
            )

        frozen = snapshot(plan.configuration, 120.0)
        run = SimpleNamespace(
            header=SimpleNamespace(
                isolation={"execution_snapshot": frozen.model_dump(mode="json")}
            ),
            generation=None,
        )
        assert require_frozen_snapshot(run, frozen) == frozen
        unchanged = snapshot(plan.configuration, 120.0)
        assert require_frozen_snapshot(run, unchanged) == frozen
        for changed in (
            snapshot(
                plan.configuration.model_copy(update={"thinking": "disabled"}), 120.0
            ),
            snapshot(plan.configuration, 180.0),
        ):
            with pytest.raises(
                IncompatibleCheckpointError, match="execution snapshot mismatch"
            ):
                require_frozen_snapshot(run, changed)


@pytest.mark.parametrize(
    "reason",
    [
        "something_new",
        "account_quota_exhausted",
        "unregistered_provider_error",
        "storage_unavailable",
    ],
)
def test_batch_level_stop_reasons_are_untrusted_in_the_consumer(tmp_path, reason):
    with batch(tmp_path, "batch-level-stop") as (seal, _):
        asyncio.run(run(seal.directory, transport_override=mock_transport("material")))
        plan = seal.plan()
        _name, body = next(iter(sorted(observations(seal).items())))
        changed = copy.deepcopy(body)
        changed["row"]["stop_reason"] = reason
        assert "batch_stopped_trust_boundary" in observation_failures(
            plan, case_of(seal, changed["case_id"]), changed
        )


def test_provider_refusals_classify_from_the_fixed_code_field_only():
    """The refusal vocabulary is read from the fixed code/category field and never echoed."""

    from procurement_review.chat_transport_v2 import decode_response
    from procurement_review.precise_dev_paired import paired_configuration

    config = paired_configuration()

    def attempt(status, body=None, headers=None):
        response = (
            httpx.Response(status, json=body, headers=headers)
            if body is not None
            else httpx.Response(status)
        )
        return decode_response(response, None, config, "")

    # The historical account exhaustion arrives as a 400 carrying the subscription code;
    # it has its own fixed classification, is never retryable and never waits for the
    # dense-429 window.
    quota = attempt(
        400,
        {"error": {"code": "InvalidSubscription", "message": "account 2108278017 ..."}},
    )
    assert quota.error_code == "account_quota_exhausted"
    assert quota.retryable is False
    assert "InvalidSubscription" not in repr(quota)
    assert (
        attempt(429, {"error": {"code": "insufficient_quota"}}).error_code
        == "account_quota_exhausted"
    )
    # An ordinary 429 stays the registered, retryable classification whatever body it
    # carries, so a described rate limit is still retried inside the shared budget.
    plain = attempt(429, {"error": {"code": "rate_limit_exceeded"}})
    assert plain.error_code == "rate_limited" and plain.retryable is True
    # A registered refusal status whose fixed code field names something outside the
    # vocabulary cannot be classified from what may be read; it stops the batch under the
    # versioned unregistered code instead of passing as an ordinary business failure.
    unknown_code = attempt(400, {"error": {"code": "SomethingBrandNew"}})
    assert unknown_code.error_code == "unregistered_provider_error"
    assert unknown_code.retryable is False
    # Without a readable code field the registered refusal keeps its own classification.
    assert (
        attempt(400, {"error": {"message": "no fixed code here"}}).error_code
        == "unsupported_request"
    )
    assert attempt(400, {"error": {}}).error_code == "unsupported_request"
    # An unknown error status is unregistered content by definition.
    assert attempt(418).error_code == "unregistered_provider_error"
    assert attempt(302).error_code == "unregistered_provider_error"
    # Registered statuses keep their classifications.
    assert attempt(401).error_code == "authentication"
    assert attempt(403).error_code == "authentication"
    assert attempt(503).error_code == "server_error" and attempt(503).retryable is True
    assert attempt(501).error_code == "server_error" and attempt(501).retryable is False


def quota_wire(code="InvalidSubscription", status=400):
    """One mock answering every request with an account/subscription refusal."""

    sent = []

    def respond(request):
        payload = json.loads(request.content)
        user = json.loads(
            next(
                message["content"]
                for message in payload["messages"]
                if message["role"] == "user"
            )
        )
        sent.append("joint" if "tasks" in user else user["worker"])
        return httpx.Response(
            status, json={"error": {"code": code, "message": "not classified here"}}
        )

    return sent, httpx.MockTransport(respond)


def test_account_quota_refusal_stops_the_batch_without_a_dense_window(tmp_path):
    """The quota refusal is not retried and never waits for the registered 429 window."""

    contract = {**V2, "rate_limit_window_seconds": 60, "rate_limit_burst_threshold": 64}
    with batch(tmp_path, "quota-stop", count=1, contract=contract) as (seal, _):
        sent, transport = quota_wire()
        result = asyncio.run(run(seal.directory, transport_override=transport))
        assert result["stopped"] is True
        assert result["stop_reason"] == "account_quota_exhausted"
        assert result["stop_summary"]["priority"] == "account_quota"
        # Exactly one physical send left the batch: no retry, no contract branch, no
        # joint arm, and the queued seed never started.
        assert sent == ["policy"], sent
        bodies = observations(seal)
        assert len(bodies) == 1, list(bodies)
        body = next(iter(bodies.values()))
        assert body["row"]["stop_reason"] == "account_quota_exhausted"
        assert body["row"]["branches"]["contract"]["outcome"] == "stopped"
        assert [item["status"] for item in body["attempts"]] == ["settled"]
        plan = seal.plan()
        assert "batch_stopped_trust_boundary" in observation_failures(
            plan, case_of(seal, body["case_id"]), body
        )
        proof = completion_proof(seal.directory)
        assert proof["complete"] is False
        assert all(
            "missing_observation" in failure or "batch_stopped" in failure
            for failure in proof["failures"]
        ), proof["failures"]


@pytest.mark.parametrize(
    "wire",
    ["unknown-code", "unknown-status"],
)
def test_unknown_provider_error_stops_the_batch(tmp_path, wire):
    """An unrecognized refusal code or status is unregistered content, not a plain failure."""

    contract = {**V2, "rate_limit_window_seconds": 60, "rate_limit_burst_threshold": 64}
    with batch(tmp_path, "unknown-provider", count=1, contract=contract) as (seal, _):
        sent, transport = quota_wire(
            code="BrandNewProviderThing" if wire == "unknown-code" else "no_code",
            status=400 if wire == "unknown-code" else 418,
        )
        result = asyncio.run(run(seal.directory, transport_override=transport))
        assert result["stopped"] is True
        assert result["stop_reason"] == "unregistered_provider_error"
        assert result["stop_summary"]["priority"] == "unregistered"
        assert sent == ["policy"], sent
        bodies = observations(seal)
        assert len(bodies) == 1, list(bodies)
        body = next(iter(bodies.values()))
        assert body["row"]["stop_reason"] == "unregistered_provider_error"
        assert body["row"]["branches"]["contract"]["outcome"] == "stopped"
        assert "batch_stopped_trust_boundary" in observation_failures(
            seal.plan(), case_of(seal, body["case_id"]), body
        )


def test_joint_arm_account_quota_stops_the_batch_without_a_retry(tmp_path):
    """The single Agent arm reaches the same classification and is not retried either."""

    contract = {**V2, "rate_limit_window_seconds": 60, "rate_limit_burst_threshold": 64}
    with batch(tmp_path, "joint-quota", count=1, contract=contract) as (seal, _):
        material = mock_transport("material")
        sent, refusals = [], 0

        def respond(request):
            nonlocal refusals
            payload = json.loads(request.content)
            user = json.loads(
                next(
                    message["content"]
                    for message in payload["messages"]
                    if message["role"] == "user"
                )
            )
            if "tasks" in user:
                sent.append("joint")
                refusals += 1
                return httpx.Response(
                    400, json={"error": {"code": "InvalidSubscription"}}
                )
            sent.append(user["worker"])
            return material.handler(request)

        result = asyncio.run(
            run(seal.directory, transport_override=httpx.MockTransport(respond))
        )
        assert result["stopped"] is True
        assert result["stop_reason"] == "account_quota_exhausted"
        assert result["stop_summary"]["priority"] == "account_quota"
        # The joint attempt was sent once and never retried, and the queued seed never
        # started.
        assert sent == ["policy", "contract", "joint"], sent
        assert refusals == 1
        bodies = observations(seal)
        assert len(bodies) == 2, list(bodies)
        assert {body["arm"] for body in bodies.values()} == {
            "multi_agent",
            "single_agent",
        }
        assert all(body["repeat_seed"] == 101 for body in bodies.values())
        assert any(
            note["code"] == "account_quota_exhausted"
            for note in result["stop_summary"]["notes"]
        )


@pytest.mark.parametrize(
    "code,recordable",
    [("http_error", True), ("a_brand_new_attempt_error", False)],
)
def test_unregistered_attempt_code_blocks_the_next_physical_send(
    tmp_path, code, recordable
):
    """An unclassified attempt outcome reaching the controller stops the very next send.

    Two shapes of the same defect are covered: a code the record vocabulary still knows
    but the frozen stop classification does not (the historical unnamed ``http_error``),
    and a code no vocabulary knows at all.  Either way the controller records the versioned
    unregistered classification with the observed code in its note, and the second physical
    send of the Run never leaves.
    """

    from procurement_review.chat_transport_v2 import AttemptResult
    from procurement_review.eval_h_product import multi
    from procurement_review.eval_h_storage import read as read_json
    from procurement_review.precise_run_transport import PreciseRunChatTransport
    from procurement_review.review_store import ReviewStore

    class Rewriting(PreciseRunChatTransport):
        def decode(self, response, request, credential):
            return AttemptResult(error_code=code)

    with batch(tmp_path, "unknown-attempt", count=1, contract=V2) as (seal, url):
        state = read_json(seal.directory / "private/runtime.json")
        store = ReviewStore.from_url(url)
        plan = seal.plan()
        controller = BatchStopController(plan.stop_policy())
        sent = []

        def respond(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={})

        transport = Rewriting(
            plan.configuration.model_copy(update={"seed": 101}),
            SecretStr("h-offline-only"),
            catalog=None,
            transport=httpx.MockTransport(respond),
        )
        case = seal.runner_cases()[0]
        try:
            _row, binding = asyncio.run(
                multi(
                    store,
                    state["token"],
                    plan,
                    case,
                    101,
                    transport,
                    stop=controller,
                )
            )
        finally:
            store.close()
        assert len(sent) == 1, len(sent)
        assert controller.stopped is True
        assert controller.reason == "unregistered_provider_error"
        assert controller.priority == "unregistered"
        assert controller.trigger["attempt_error"] == code
        # The contract branch never left: only the policy attempt is in the ledger and
        # nothing was committed.
        assert [item["worker"] for item in binding["attempts"]] == ["policy"]
        assert binding["checkpoints"] == []
        if recordable:
            # The failed attempt settled with its receipt; the Run then records the batch
            # stop at the next gate instead of starting a second send.
            assert binding["status"] == "interrupted"
            assert binding["error_code"] == "batch_stopped"
            assert binding["attempts"][0]["outcome"] == "failed"
            assert binding["execution"] == {}
        else:
            # A code the record vocabulary cannot persist is refused by the guard: the
            # execution fails closed, and the attempt it already sent stays in the ledger
            # as an unsettled send — never rewritten as one that never happened, and never
            # silently dropped.  The consumer reads an unsettled attempt as a batch-level
            # record gap, so the batch cannot be scored from it.
            assert binding["status"] == "failed"
            assert binding["error_code"] == "review_execution_failed"
            assert binding["attempts"][0]["finished_at"] is None
            assert binding["attempts"][0]["abandoned"] is False


def test_ordinary_429_retries_inside_the_shared_budget_and_the_batch_continues(
    tmp_path,
):
    """A 429 below the dense threshold is retried per branch, never a third time."""

    contract = {**V2, "rate_limit_window_seconds": 60, "rate_limit_burst_threshold": 64}
    with batch(tmp_path, "retry-429", count=1, contract=contract) as (seal, _):
        material = mock_transport("material")
        sent, counts = [], {}

        def respond(request):
            payload = json.loads(request.content)
            user = json.loads(
                next(
                    message["content"]
                    for message in payload["messages"]
                    if message["role"] == "user"
                )
            )
            role = "joint" if "tasks" in user else user["worker"]
            key = (payload.get("seed"), role)
            sent.append(role)
            counts[key] = counts.get(key, 0) + 1
            if counts[key] == 1:
                return httpx.Response(
                    429, headers={"retry-after": "0"}, json={"error": "slow down"}
                )
            return material.handler(request)

        result = asyncio.run(
            run(seal.directory, transport_override=httpx.MockTransport(respond))
        )
        # Two seeds of one Case, three model tasks each: every task was rate-limited once
        # and retried exactly once inside its own shared two-attempt budget.
        assert counts == {
            (101, "policy"): 2,
            (101, "contract"): 2,
            (101, "joint"): 2,
            (202, "policy"): 2,
            (202, "contract"): 2,
            (202, "joint"): 2,
        }, counts
        assert len(sent) == 12, sent
        assert result["stopped"] is False
        assert result["stop_reason"] is None
        proof = completion_proof(seal.directory)
        assert proof["complete"], proof["failures"]
        assert proof["observed"] == proof["planned"] == 4
        assert all(
            len([item for item in body["attempts"] if item["branch"] == "policy"]) == 2
            for body in observations(seal).values()
            if body["arm"] == "multi_agent"
        )
