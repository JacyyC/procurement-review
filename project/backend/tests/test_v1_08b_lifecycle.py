"""V1-08B: the 120- and 180-second lifecycle readings on the isolated product path.

Every scenario runs one real product Run (``ReviewService`` + ``PreciseScopedRunner``)
against an isolated PostgreSQL and a Mock transport, at each of the two accepted Run
deadlines, and reads the counts back from the product's own ledger:

* cancel while a request is in flight: the received answer is never adopted, the unsent
  branch is never sent and the sent attempt keeps its place as unresolved;
* both revocation orders — revoked before the response arrives, and received before the
  revocation — refuse the result without pretending the send never happened;
* the generation deadline and the execution lease each interrupt the Run under a held
  response, with zero adoption and no second send;
* recovery reuses the committed branch, inherits the registered window and the spent
  shared attempt quota, and a changed thinking value or deadline is refused by the frozen
  snapshot instead of re-running under new settings;
* identical idempotency keys add nothing: no new Run, generation or physical send.

No long sleeps: the in-flight window is held by an event and the clocks are injected.
"""

import asyncio
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

from procurement_review.api_contracts import (
    HumanActionRequestV2,
    ReviewCreateRequestV2,
    ResumeRequestV2,
)
from procurement_review.contracts import ApprovalActionType
from procurement_review.eval_h_contracts import H_PLAN_V2
from procurement_review.eval_h_fixture import make_disposable
from procurement_review.eval_h_mock import mock_transport
from procurement_review.eval_h_product import admission, source_snapshot
from procurement_review.eval_h_runner import setup
from procurement_review.eval_h_seal import HSeal
from procurement_review.eval_h_storage import read
from procurement_review.precise_run_runner import PreciseScopedRunner
from procurement_review.precise_run_transport import PreciseRunChatTransport
from procurement_review.review_repository import ReviewRepository
from procurement_review.review_service import ReviewService
from procurement_review.review_store import ReviewStore

DEADLINES = (120.0, 180.0)


@contextmanager
def batch(tmp_path, suffix, *, deadline, count=1):
    """One sealed V1-08 disposable batch and its dedicated database."""

    value = make_disposable(
        tmp_path / ".local",
        count,
        batch_id="disposable-h-" + suffix,
        plan_version=H_PLAN_V2,
        thinking="enabled",
        generation_deadline_seconds=deadline,
        rate_limit_window_seconds=60,
        rate_limit_burst_threshold=8,
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


class Clock:
    """One injected, timezone-aware clock the test moves by hand."""

    def __init__(self):
        self.value = datetime.now(UTC)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value = self.value + timedelta(seconds=seconds)


def held_material():
    """A Mock wire that holds every request until released, then answers valid material."""

    material = mock_transport("material")
    sent, hold = [], asyncio.Event()

    async def respond(request):
        payload = json.loads(request.content)
        user = json.loads(
            next(
                message["content"]
                for message in payload["messages"]
                if message["role"] == "user"
            )
        )
        sent.append("joint" if "tasks" in user else user["worker"])
        await hold.wait()
        return material.handler(request)

    return sent, hold, httpx.MockTransport(respond)


def answering_material(sent):
    """A Mock wire that appends the worker it answers and returns valid material."""

    material = mock_transport("material")

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
        return material.handler(request)

    return httpx.MockTransport(respond)


def first_credential(url):
    with create_engine(url).connect() as connection:
        return str(
            connection.execute(
                text("SELECT credential_id FROM review_credential LIMIT 1")
            ).scalar_one()
        )


class Harness:
    """One service, repository and transport bound to one sealed batch."""

    def __init__(
        self,
        seal,
        url,
        *,
        deadline,
        transport,
        fault=None,
        clock=None,
        lease_seconds=15.0,
        heartbeat_seconds=600.0,
    ):
        self.url = url
        self.plan = seal.plan()
        self.case = seal.runner_cases()[0]
        self.state = read(seal.directory / "private/runtime.json")
        self.store = ReviewStore.from_url(url)
        self.context, self.registration, self.scope = admission(
            self.store,
            self.state["token"],
            self.plan,
            self.case.case_id,
            deadline_seconds=deadline,
        )
        self.transport = transport
        self.repository = ReviewRepository(self.store, fault=fault)
        self.service = ReviewService(
            repository=self.repository,
            runner=PreciseScopedRunner(
                store=self.store,
                transport=transport,
                generation_deadline_seconds=deadline,
                mode="test_only",
            ),
            clock=clock,
            lease_seconds=lease_seconds,
            heartbeat_seconds=heartbeat_seconds,
            generation_deadline_seconds=deadline,
            fault=fault,
        )

    async def create(self, key="lifecycle-create"):
        response, created = await self.service.create_review(
            ReviewCreateRequestV2(
                api_version="review-api.v2",
                case_id=self.case.case_id,
                case_version=self.registration.case_version,
            ),
            registration=self.registration,
            scope=self.scope,
            context=self.context,
            idempotency_key=key,
        )
        return response.run_id, created

    def cancel(self, run_id, key="lifecycle-cancel"):
        detail = self.service.get_review(run_id, context=self.context)
        return self.service.apply_action(
            run_id=run_id,
            action=ApprovalActionType.CANCEL,
            request=HumanActionRequestV2(
                expected_version=detail.revision, idempotency_key=key
            ),
            context=self.context,
        )

    async def resume(self, run_id, key="lifecycle-resume"):
        detail = self.service.get_review(run_id, context=self.context)
        return await self.service.resume_review(
            run_id,
            ResumeRequestV2(
                api_version="review-api.v2", expected_version=detail.revision
            ),
            context=self.context,
            idempotency_key=key,
        )

    def binding(self, run_id):
        return source_snapshot(self.store, run_id)

    def generations(self, run_id):
        """``(generation, window_seconds)`` of every generation row of one Run."""

        with create_engine(self.url).connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT generation, EXTRACT(EPOCH FROM"
                    " (deadline_at - created_at)) AS window"
                    " FROM review_run_generation WHERE run_id=:r ORDER BY generation"
                ),
                {"r": run_id},
            ).all()
        return [(int(row.generation), float(row.window)) for row in rows]

    async def wait_for(self, condition, timeout=30.0):
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if condition():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("the expected state never arrived")

    async def wait_for_settled(self, run_id, timeout=30.0):
        await self.wait_for(
            lambda: self.service.get_review(run_id, context=self.context).status.value
            in {
                "pending_confirmation",
                "pending_human_review",
                "failed",
                "cancelled",
                "interrupted",
            },
            timeout,
        )

    def close(self):
        self.store.close()


class RaiseOnce:
    """A test-only fault that raises one controlled signal at one barrier."""

    enabled = True

    def __init__(self, site, signal):
        self.site = site
        self.signal = signal
        self.fired = False

    def barrier(self, site):
        if site == self.site and not self.fired:
            self.fired = True
            raise self.signal("h-lifecycle-controlled-fault")


class CountEntries:
    """A test-only fault that raises once the named entry is seen the Nth time."""

    enabled = True

    def __init__(self, site, count, signal):
        self.site = site
        self.count = count
        self.signal = signal
        self.seen = 0
        self.fired = False

    def barrier(self, site):
        if site != self.site:
            return
        self.seen += 1
        if self.seen >= self.count and not self.fired:
            self.fired = True
            raise self.signal("h-lifecycle-controlled-fault")


class RevokeOnAdmission:
    """A test-only fault that revokes the credential at the result-admission barrier."""

    enabled = True

    def __init__(self, url, site="after_result_admission:precise.policy.answer"):
        self.url = url
        self.site = site
        self.fired = False

    def barrier(self, site):
        if site != self.site or self.fired:
            return
        self.fired = True
        store = ReviewStore.from_url(self.url)
        try:
            assert store.revoke_credential(
                credential_id=first_credential(self.url),
                revoked_at=datetime.now(UTC),
                reason="h-lifecycle-revocation",
            )
        finally:
            store.close()


def storage_signal():
    from procurement_review.checkpointing import StorageUnavailableSignal

    return StorageUnavailableSignal


def transport_for(seal, wire):
    return PreciseRunChatTransport(
        seal.plan().configuration.model_copy(update={"seed": 101}),
        SecretStr("h-offline-only"),
        catalog=None,
        transport=wire,
    )


@pytest.mark.parametrize("deadline", DEADLINES)
def test_cancel_in_flight_never_adopts_and_never_sends_the_next_branch(
    tmp_path, deadline
):
    with batch(tmp_path, f"cancel-{int(deadline)}", deadline=deadline) as (seal, url):
        sent, hold, wire = held_material()
        harness = Harness(
            seal, url, deadline=deadline, transport=transport_for(seal, wire)
        )
        try:

            async def scenario():
                run_id, created = await harness.create()
                assert created is True
                await harness.wait_for(lambda: bool(sent))
                assert sent == ["policy"], sent
                cancelled = harness.cancel(run_id)
                assert cancelled.status.value == "cancelled"
                # The answer that was already received arrives after the cancel and is
                # never adopted; nothing new is sent for the never-sent contract branch.
                hold.set()
                await asyncio.sleep(0.25)
                return run_id

            run_id = asyncio.run(scenario())
            binding = harness.binding(run_id)
            assert sent == ["policy"], sent
            assert binding["status"] == "cancelled"
            assert binding["checkpoints"] == []
            assert binding["summary"] is None
            assert binding["execution"] == {}
            # The received request keeps its place as an unresolved attempt instead of
            # being rewritten as one that never happened.
            assert len(binding["attempts"]) == 1
            assert binding["attempts"][0]["worker"] == "policy"
            assert binding["attempts"][0]["abandoned"] is True
        finally:
            hold.set()
            asyncio.run(harness.service.shutdown())
            harness.close()


@pytest.mark.parametrize("deadline", DEADLINES)
@pytest.mark.parametrize("order", ["revoke-first", "received-first"])
def test_revocation_refuses_the_result_in_both_orders(tmp_path, deadline, order):
    with batch(tmp_path, f"revoke-{int(deadline)}-{order}", deadline=deadline) as (
        seal,
        url,
    ):
        sent, hold, wire = held_material()
        fault = RevokeOnAdmission(url) if order == "received-first" else None
        harness = Harness(
            seal,
            url,
            deadline=deadline,
            transport=transport_for(seal, wire),
            fault=fault,
        )
        try:

            async def scenario():
                run_id, _created = await harness.create()
                await harness.wait_for(lambda: bool(sent))
                assert sent == ["policy"], sent
                if order == "revoke-first":
                    store = ReviewStore.from_url(url)
                    try:
                        assert store.revoke_credential(
                            credential_id=first_credential(url),
                            revoked_at=datetime.now(UTC),
                            reason="h-lifecycle-revocation",
                        )
                    finally:
                        store.close()
                hold.set()
                await harness.wait_for_settled(run_id)
                return run_id

            run_id = asyncio.run(scenario())
            binding = harness.binding(run_id)
            assert binding["status"] == "interrupted"
            assert binding["error_code"] == "authorization_revoked"
            # Zero adoption: neither branch committed, no summary, no second send.
            assert sent == ["policy"], sent
            assert binding["checkpoints"] == []
            assert binding["summary"] is None
            assert len(binding["attempts"]) == 1
            assert binding["attempts"][0]["abandoned"] is True
            if order == "received-first":
                assert fault.fired is True
        finally:
            hold.set()
            asyncio.run(harness.service.shutdown())
            harness.close()


@pytest.mark.parametrize("deadline", DEADLINES)
def test_generation_deadline_expiry_adopts_nothing(tmp_path, deadline):
    with batch(tmp_path, f"deadline-{int(deadline)}", deadline=deadline) as (seal, url):
        sent, hold, wire = held_material()
        clock = Clock()
        harness = Harness(
            seal,
            url,
            deadline=deadline,
            transport=transport_for(seal, wire),
            clock=clock,
        )
        try:

            async def scenario():
                run_id, _created = await harness.create()
                await harness.wait_for(lambda: bool(sent))
                clock.advance(deadline + 1.0)
                hold.set()
                await harness.wait_for_settled(run_id)
                return run_id

            run_id = asyncio.run(scenario())
            binding = harness.binding(run_id)
            assert binding["status"] == "interrupted"
            assert binding["error_code"] == "generation_deadline_exceeded"
            assert sent == ["policy"], sent
            assert binding["checkpoints"] == []
            assert binding["summary"] is None
            assert len(binding["attempts"]) == 1
        finally:
            hold.set()
            asyncio.run(harness.service.shutdown())
            harness.close()


@pytest.mark.parametrize("deadline", DEADLINES)
def test_lease_expiry_interrupts_before_any_adoption(tmp_path, deadline):
    with batch(tmp_path, f"lease-{int(deadline)}", deadline=deadline) as (seal, url):
        sent, hold, wire = held_material()
        clock = Clock()
        harness = Harness(
            seal,
            url,
            deadline=deadline,
            transport=transport_for(seal, wire),
            clock=clock,
            lease_seconds=10.0,
        )
        try:

            async def scenario():
                run_id, _created = await harness.create()
                await harness.wait_for(lambda: bool(sent))
                clock.advance(11.0)
                hold.set()
                await harness.wait_for_settled(run_id)
                return run_id

            run_id = asyncio.run(scenario())
            binding = harness.binding(run_id)
            assert binding["status"] == "interrupted"
            assert binding["error_code"] == "lease_expired"
            assert sent == ["policy"], sent
            assert binding["checkpoints"] == []
            assert binding["summary"] is None
        finally:
            hold.set()
            asyncio.run(harness.service.shutdown())
            harness.close()


@pytest.mark.parametrize("deadline", DEADLINES)
def test_resume_reuses_the_committed_branch_and_inherits_the_window(tmp_path, deadline):
    with batch(tmp_path, f"resume-{int(deadline)}", deadline=deadline) as (seal, url):
        fault = RaiseOnce(
            "during_model_entry_authorization:precise.contract", storage_signal()
        )
        sent = []
        harness = Harness(
            seal,
            url,
            deadline=deadline,
            transport=transport_for(seal, answering_material(sent)),
            fault=fault,
        )
        try:

            async def scenario():
                run_id, _created = await harness.create()
                await harness.wait_for_settled(run_id)
                binding = harness.binding(run_id)
                assert binding["status"] == "interrupted"
                assert binding["error_code"] == "storage_unavailable"
                assert sent == ["policy"], sent
                # The interrupted generation spent one contract attempt slot; the ledger
                # is what the next execution reads its remaining quota from.
                assert harness.repository.attempt_totals(run_id) == {
                    "policy": 1,
                    "contract": 1,
                }
                assert (
                    harness.service._runner.remaining_attempts(
                        harness.repository, run_id, "contract"
                    )
                    == 1
                )
                _response, scheduled = await harness.resume(
                    run_id, key="lifecycle-resume"
                )
                assert scheduled is True
                await harness.wait_for_settled(run_id)
                return run_id

            run_id = asyncio.run(scenario())
            binding = harness.binding(run_id)
            assert binding["status"] in {"pending_confirmation", "pending_human_review"}
            assert binding["resume_count"] == 1
            # The committed branch was not resent; only the remaining one was.
            assert sent == ["policy", "contract"], sent
            assert binding["execution"]["metering"]["reused_branches"] == 1
            assert binding["execution"]["policy"]["outcome"] == "completed"
            assert binding["execution"]["contract"]["outcome"] == "completed"
            assert harness.repository.attempt_totals(run_id) == {
                "policy": 1,
                "contract": 2,
            }
            # Both generations carry the registered window, not a fresh default.
            windows = harness.generations(run_id)
            assert len(windows) == 2
            assert all(abs(window - deadline) < 0.5 for _generation, window in windows)
        finally:
            asyncio.run(harness.service.shutdown())
            harness.close()


@pytest.mark.parametrize("deadline", DEADLINES)
def test_resume_inherits_the_spent_shared_attempt_quota(tmp_path, deadline):
    with batch(tmp_path, f"quota-{int(deadline)}", deadline=deadline) as (seal, url):
        fault = CountEntries(
            "during_model_entry_authorization:precise.contract", 2, storage_signal()
        )
        material = mock_transport("material")
        sent = []

        def invalid_body():
            return httpx.Response(
                200,
                json={
                    "model": "deepseek-v4-1-flash",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "choices": [{"finish_reason": "stop", "message": {"content": "{"}}],
                },
            )

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
            sent.append(role)
            if role == "contract" and sent.count("contract") == 1:
                # The contract branch's first attempt is a correctable format failure, so
                # it spends one of the two shared slots and retries; the retry is admitted
                # and then interrupted before it can be sent.
                return invalid_body()
            return material.handler(request)

        harness = Harness(
            seal,
            url,
            deadline=deadline,
            transport=transport_for(seal, httpx.MockTransport(respond)),
            fault=fault,
        )
        try:

            async def scenario():
                run_id, _created = await harness.create()
                await harness.wait_for_settled(run_id)
                assert harness.binding(run_id)["status"] == "interrupted"
                # The first contract attempt was a correctable format failure; the second
                # was admitted and then interrupted, so both slots are spent.
                assert sent == ["policy", "contract"], sent
                assert harness.repository.attempt_totals(run_id)["contract"] == 2
                _response, scheduled = await harness.resume(
                    run_id, key="lifecycle-quota-resume"
                )
                assert scheduled is True
                await harness.wait_for_settled(run_id)
                return run_id

            run_id = asyncio.run(scenario())
            binding = harness.binding(run_id)
            # The resumed generation got no fresh budget: the contract branch had none of
            # its two shared attempts left, so it failed without a new physical send while
            # the committed policy branch was reused.
            assert sent == ["policy", "contract"], sent
            assert binding["status"] in {"pending_confirmation", "pending_human_review"}
            assert binding["execution"]["contract"]["outcome"] == "no_attempts_left"
            assert binding["execution"]["policy"]["outcome"] == "completed"
            assert harness.repository.attempt_totals(run_id) == {
                "policy": 1,
                "contract": 2,
            }
            assert len(harness.generations(run_id)) == 2
        finally:
            asyncio.run(harness.service.shutdown())
            harness.close()


@pytest.mark.parametrize("deadline", DEADLINES)
@pytest.mark.parametrize("change", ["thinking", "deadline"])
def test_a_changed_snapshot_is_refused_on_resume(tmp_path, deadline, change):
    with batch(tmp_path, f"frozen-{int(deadline)}-{change}", deadline=deadline) as (
        seal,
        url,
    ):
        fault = RaiseOnce(
            "during_model_entry_authorization:precise.contract", storage_signal()
        )
        sent = []
        harness = Harness(
            seal,
            url,
            deadline=deadline,
            transport=transport_for(seal, answering_material(sent)),
            fault=fault,
        )
        try:

            async def scenario():
                run_id, _created = await harness.create()
                await harness.wait_for_settled(run_id)
                assert harness.binding(run_id)["status"] == "interrupted"
                # The resumed execution is built with a runtime that no longer matches the
                # Run's frozen snapshot: a different thinking value, or the other accepted
                # deadline.  The refusal must happen before any new send.
                config = harness.plan.configuration.model_copy(update={"seed": 101})
                if change == "thinking":
                    config = config.model_copy(update={"thinking": "disabled"})
                changed_deadline = 180.0 if deadline == 120.0 else 120.0
                harness.service._runner = PreciseScopedRunner(
                    store=harness.store,
                    transport=PreciseRunChatTransport(
                        config,
                        SecretStr("h-offline-only"),
                        catalog=None,
                        transport=answering_material(sent),
                    ),
                    generation_deadline_seconds=(
                        changed_deadline if change == "deadline" else deadline
                    ),
                    mode="test_only",
                )
                if change == "deadline":
                    harness.service._generation_deadline_seconds = changed_deadline
                _response, scheduled = await harness.resume(
                    run_id, key="lifecycle-frozen-resume"
                )
                assert scheduled is True
                await harness.wait_for_settled(run_id)
                return run_id

            run_id = asyncio.run(scenario())
            binding = harness.binding(run_id)
            assert binding["status"] == "failed"
            assert binding["error_code"] == "checkpoint_incompatible"
            assert sent == ["policy"], sent
            assert binding["summary"] is None
        finally:
            asyncio.run(harness.service.shutdown())
            harness.close()


@pytest.mark.parametrize("deadline", DEADLINES)
def test_identical_idempotency_keys_add_nothing(tmp_path, deadline):
    with batch(tmp_path, f"replay-{int(deadline)}", deadline=deadline) as (seal, url):
        sent = []
        harness = Harness(
            seal,
            url,
            deadline=deadline,
            transport=transport_for(seal, answering_material(sent)),
        )
        try:

            async def scenario():
                run_id, created = await harness.create(key="lifecycle-same-key")
                assert created is True
                await harness.wait_for_settled(run_id)
                first = (len(sent), len(harness.generations(run_id)))
                again_id, created_again = await harness.create(key="lifecycle-same-key")
                assert again_id == run_id
                assert created_again is False
                assert (len(sent), len(harness.generations(run_id))) == first
                return run_id, first

            run_id, first = asyncio.run(scenario())
            binding = harness.binding(run_id)
            assert binding["status"] in {"pending_confirmation", "pending_human_review"}
            assert binding["resume_count"] == 0
            assert first == (2, 1)
            assert sent == ["policy", "contract"], sent
        finally:
            asyncio.run(harness.service.shutdown())
            harness.close()
