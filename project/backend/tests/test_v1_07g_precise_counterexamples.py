"""V-G-02: the precise Run's own interleavings, on its own product path.

The precise line replaced the model adapter and the branch lifecycle, so the FIX-era
counterexamples that were measured against the previous runner are re-measured here against
this one: a credential or a Case grant revoked between two branches, a Run cancelled between
two branches, a generation superseded while its answer is still in flight, a generation
deadline that expires before the answer can be adopted, a stored commit that no longer
satisfies its record, a repeated idempotency key, and a process rebuild that must continue
from persisted state alone.

Every test states its own interleaving point and asserts the exact class of the outcome: an
authority that lapsed is an explicit stop with its own reason, a late or superseded answer is
never adopted, and a damaged record is refused rather than reinterpreted.  No assertion is
satisfied by catching a bare ``Exception``, because a controlled stop and an incidental
failure are not the same evidence.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))

from procurement_review.api_contracts import (  # noqa: E402
    HumanActionRequestV2,
    ResumeRequestV2,
)
from procurement_review.authorization import (  # noqa: E402
    AuthorizationService,
    generate_credential,
)
from procurement_review.checkpointing import IncompatibleCheckpointError  # noqa: E402
from procurement_review.contracts import (  # noqa: E402
    ApprovalActionType,
    ReviewLifecycleStatus,
)
from procurement_review.eval_f_registration import configuration  # noqa: E402
from procurement_review.execution_guards import ExecutionInvalidated  # noqa: E402
from procurement_review.gold_v37_conditional import conditional_cases  # noqa: E402
from procurement_review.precise_run_runner import PreciseScopedRunner  # noqa: E402
from procurement_review.precise_run_state import (  # noqa: E402
    adopted_branch_commit,
    attempt_ledger_rows,
)
from procurement_review.precise_run_transport import (  # noqa: E402
    PreciseRunChatTransport,
)
from procurement_review.review_repository import ReviewRepository  # noqa: E402
from procurement_review.review_service import ReviewService  # noqa: E402
from procurement_review.review_store import ReviewStore  # noqa: E402
from procurement_review.security_contracts import SecurityOperation  # noqa: E402
from test_v1_07g_precise_run import (  # noqa: E402
    SUBJECT,
    TENANT,
    _contract_body,
    _plan,
    _policy_body,
    _scope_and_registration,
)
from v1_03_support import create_isolated_database, drop_database  # noqa: E402

CASE_INDEX = 4


# --------------------------------------------------------------- the hooked environment


@dataclass(slots=True)
class Environment:
    store: ReviewStore
    database_url: str
    database: str
    authorization: AuthorizationService
    repository: ReviewRepository
    service: ReviewService
    case: Any
    requests: list
    hook_box: dict


def _environment(*, case=None, **runner_kwargs) -> Iterator[Environment]:
    """The precise product chain, with one interleaving point before each answer.

    ``case`` overrides the authored material, so a test can register a Case whose text
    legitimately carries a value the stored-record guard must refuse.
    """

    database_url, database = create_isolated_database()
    store = ReviewStore.from_url(database_url)
    store.migrate()
    case = case if case is not None else conditional_cases()[CASE_INDEX]
    store.seed(_plan(case))
    authorization = AuthorizationService(store=store)
    repository = ReviewRepository(store)
    requests: list = []
    hook_box: dict = {"hook": None}
    config = configuration()

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        user = json.loads(payload["messages"][1]["content"])
        worker = user["worker"]
        source = user["sources"][0]
        hook = hook_box["hook"]
        if hook is not None:
            # The interleaving happens while the answer exists only at the endpoint:
            # nothing the runner holds has been decoded or committed yet.
            hook(worker)
        body = (
            _policy_body(source["content"])
            if worker == "policy"
            else _contract_body(source["content"])
        )
        for finding in body["findings"]:
            finding["evidence_refs"] = [source["ref"]]
        for witness in body["witnesses"]:
            witness["source_ref"] = source["ref"]
        return httpx.Response(
            200,
            json={
                "model": config.resolved_model,
                "usage": {"prompt_tokens": 700, "completion_tokens": 260},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"schema_version": "finding-evidence.v3", **body},
                                ensure_ascii=False,
                            )
                        },
                    }
                ],
            },
        )

    transport = PreciseRunChatTransport(
        config,
        SecretStr("offline-only"),
        catalog=None,
        transport=httpx.MockTransport(respond),
    )
    service = ReviewService(
        repository=repository,
        runner=PreciseScopedRunner(store=store, transport=transport, **runner_kwargs),
    )
    try:
        yield Environment(
            store=store,
            database_url=database_url,
            database=database,
            authorization=authorization,
            repository=repository,
            service=service,
            case=case,
            requests=requests,
            hook_box=hook_box,
        )
    finally:
        # Every execution in this module ran on the client's own portal loop; the loop
        # is gone by teardown, so only the store and the database remain to release.
        store.close()
        drop_database(database)


@pytest.fixture()
def environment() -> Iterator[Environment]:
    yield from _environment()


def _issue(environment: Environment) -> tuple[str, str]:
    """One credential for the registered subject, returned with its id."""

    token, credential_id, digest = generate_credential()
    issued_at = datetime.now(UTC)
    environment.store.issue_credential(
        credential_id=credential_id,
        tenant_id=TENANT,
        subject_id=SUBJECT,
        token_digest=digest,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(seconds=3_600),
    )
    return token, credential_id


def _context(environment: Environment, token: str, operation: SecurityOperation):
    return environment.authorization.resolve(
        token, request_id="g-counterexample", operation=operation
    )


def _client(environment: Environment):
    from fastapi.testclient import TestClient

    from procurement_review.api import create_app

    return TestClient(
        create_app(
            store=environment.store,
            authorization=environment.authorization,
            repository=environment.repository,
            service=environment.service,
        ),
        base_url="http://127.0.0.1",
    )


def _create(
    client, environment: Environment, token: str, key: str = "g-counterexample"
):
    return client.post(
        "/api/v2/reviews",
        headers={"Authorization": f"Bearer {token}", "Idempotency-Key": key},
        json={
            "api_version": "review-api.v2",
            "case_id": environment.case.case_id,
            "case_version": f"{environment.case.case_id.lower()}.v1",
        },
    )


def _await_status(
    environment: Environment,
    run_id: str,
    statuses: set[str],
    *,
    timeout: float = 30.0,
) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = environment.repository.load_run(run_id)
        if run is not None and run.header.status.value in statuses:
            return run
        time.sleep(0.05)
    raise AssertionError(f"the Run never reached {sorted(statuses)}")


def _checkpoint_workers(environment: Environment, run_id: str) -> list[str]:
    with environment.store.transaction() as connection:
        return [
            str(row[0])
            for row in connection.execute(
                text(
                    "SELECT worker FROM review_branch_checkpoint WHERE run_id = :r"
                    " ORDER BY worker"
                ),
                {"r": run_id},
            ).all()
        ]


# ------------------------------------------------------- a lapsed authority, mid-flight


@pytest.mark.parametrize("lapse", ["credential_revoked", "case_grant_revoked"])
def test_a_lapse_while_the_first_answer_is_in_flight_stops_everything(
    environment, lapse
):
    """The first answer exists only at the endpoint when the authority lapses.

    The interleaving point is the first receipt itself, so the answer decoded afterwards
    was produced under an authority that no longer exists.  It must not be adopted, and
    the second branch must never be asked.
    """

    token, credential_id = _issue(environment)

    def hook(worker: str) -> None:
        if worker != "policy":
            return
        now = datetime.now(UTC)
        if lapse == "credential_revoked":
            assert environment.store.revoke_credential(
                credential_id=credential_id, revoked_at=now, reason="g-counterexample"
            )
        else:
            assert environment.store.set_case_grant(
                tenant_id=TENANT,
                case_id=environment.case.case_id,
                subject_id=SUBJECT,
                granted=False,
                occurred_at=now,
            )

    environment.hook_box["hook"] = hook
    with _client(environment) as client:
        created = _create(client, environment, token)
        assert created.status_code in {200, 201, 202}, created.text
        run_id = created.json()["run_id"]
        run = _await_status(environment, run_id, {"interrupted", "failed", "cancelled"})
    # One physical send exists — the one whose answer was in flight — and the second
    # branch was never asked.
    assert len(environment.requests) == 1
    assert run.header.status is ReviewLifecycleStatus.INTERRUPTED
    assert run.header.error_code == "authorization_revoked"
    # The in-flight answer was decoded but never adopted: no branch is committed.
    assert _checkpoint_workers(environment, run_id) == []
    rows = attempt_ledger_rows(environment.repository, run_id)
    # Only the refused answer's own attempt exists: the stop happened at its commit, so
    # the second branch was never even reserved.
    assert [(row["worker"], row["attempt_no"]) for row in rows] == [("policy", 1)]
    # The settled receipt of the refused answer is kept as what it is — a received
    # answer whose adoption never happened: the reservation was closed by the
    # interruption as abandoned.
    policy = rows[0]
    assert policy["finished_at"] is not None and policy["outcome"] == "abandoned"
    # A fresh credential may read the Run back, and the audit names the stop reason.  A
    # revoked Case grant is a different fact: the Case itself is no longer reachable for
    # that subject, so the product answers 404 rather than a partial view.
    reader, _reader_id = _issue(environment)
    with _client(environment) as client:
        body = client.get(
            f"/api/v2/reviews/{run_id}", headers={"Authorization": f"Bearer {reader}"}
        )
    if lapse == "credential_revoked":
        assert body.status_code == 200, body.text
        assert body.json()["status"] == "interrupted"
    else:
        assert body.status_code == 404, body.text
        assert body.json()["error"]["code"] == "resource_not_found"
    detail_codes = {
        event.detail_code for event in environment.repository.audit_events(run_id)
    }
    assert "authorization_revoked" in detail_codes, detail_codes


@pytest.mark.parametrize(
    "lapse,reason",
    [
        # A revocation locks and bumps the authorization revision, and the revision is
        # the guard's single linearization point, so it is the first fact a running
        # generation observes.  The Run-level reason for any authority lapse is the same
        # "authorization_revoked" (asserted by the API-level test above).
        ("credential_revoked", "authorization_revision_changed"),
        ("case_grant_revoked", "authorization_revision_changed"),
        # A credential that expires moves no revision, so its own reason stays visible.
        ("credential_expired", "credential_expired"),
    ],
)
def test_a_committed_branch_survives_a_lapse_and_the_next_attempt_is_refused(
    environment, lapse, reason
):
    """After a lapse the next execution reuses the committed branch and sends nothing."""

    calls: list = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        payload = json.loads(request.content)
        user = json.loads(payload["messages"][1]["content"])
        worker = user["worker"]
        if worker == "contract":
            raise httpx.ConnectError("the endpoint is unreachable")
        source = user["sources"][0]
        body = _policy_body(source["content"])
        for finding in body["findings"]:
            finding["evidence_refs"] = [source["ref"]]
        for witness in body["witnesses"]:
            witness["source_ref"] = source["ref"]
        return httpx.Response(
            200,
            json={
                "model": configuration().resolved_model,
                "usage": {"prompt_tokens": 700, "completion_tokens": 260},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"schema_version": "finding-evidence.v3", **body},
                                ensure_ascii=False,
                            )
                        },
                    }
                ],
            },
        )

    transport = PreciseRunChatTransport(
        configuration(),
        SecretStr("offline-only"),
        catalog=None,
        transport=httpx.MockTransport(respond),
    )
    runner = PreciseScopedRunner(store=environment.store, transport=transport)
    run_id, run, scope, registration, credential_id = _open_run(environment, runner)
    first = _execute(runner, environment, run_id, run, scope, registration)
    assert first.policy.outcome.value == "completed"
    assert first.contract.outcome.value == "failed"
    assert first.contract.failure_code == "transport"
    assert _checkpoint_workers(environment, run_id) == ["policy"]

    now = datetime.now(UTC)
    if lapse == "credential_revoked":
        assert environment.store.revoke_credential(
            credential_id=credential_id, revoked_at=now, reason="g-counterexample"
        )
    elif lapse == "case_grant_revoked":
        assert environment.store.set_case_grant(
            tenant_id=TENANT,
            case_id=environment.case.case_id,
            subject_id=SUBJECT,
            granted=False,
            occurred_at=now,
        )
    else:
        with environment.store.transaction() as connection:
            connection.execute(
                text(
                    "UPDATE review_credential SET expires_at = :at"
                    " WHERE credential_id = :c"
                ),
                {"at": now - timedelta(seconds=5), "c": credential_id},
            )
    sent_before = len(calls)
    second = PreciseScopedRunner(store=environment.store, transport=transport)
    with pytest.raises(ExecutionInvalidated) as caught:
        _execute(second, environment, run_id, run, scope, registration)
    assert caught.value.reason == reason
    # Nothing was sent, and the committed branch is still there.
    assert len(calls) == sent_before
    assert _checkpoint_workers(environment, run_id) == ["policy"]


def test_a_cancelled_run_stops_the_second_branch_before_any_send(environment):
    """A human cancellation while the first answer is in flight is terminal."""

    token, _credential_id = _issue(environment)
    run_ids: list[str] = []

    def hook(worker: str) -> None:
        if worker != "policy" or not run_ids:
            return
        run_id = run_ids[0]
        current = environment.service.get_review(
            run_id,
            context=_context(environment, token, SecurityOperation.REVIEW_CANCEL),
        )
        environment.service.apply_action(
            run_id=run_id,
            action=ApprovalActionType.CANCEL,
            request=HumanActionRequestV2(
                expected_version=current.revision,
                idempotency_key="g-counterexample-cancel",
            ),
            context=_context(environment, token, SecurityOperation.REVIEW_CANCEL),
        )

    environment.hook_box["hook"] = hook
    with _client(environment) as client:
        created = _create(client, environment, token)
        assert created.status_code in {200, 201, 202}, created.text
        run_id = created.json()["run_id"]
        run_ids.append(run_id)
        run = _await_status(environment, run_id, {"cancelled"})
    assert run.header.status is ReviewLifecycleStatus.CANCELLED
    assert len(environment.requests) == 1
    # The cancelled Run adopts nothing, and its second branch was never asked.
    assert _checkpoint_workers(environment, run_id) == []


# -------------------------------------------- late and superseded answers, never adopted


class _SupersedingTransport:
    """A transport that lets another generation take the Run while an answer is in flight."""

    def __init__(
        self, environment: Environment, calls: list, takeover: Callable[[], None]
    ):
        config = configuration()

        def respond(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            payload = json.loads(request.content)
            user = json.loads(payload["messages"][1]["content"])
            worker = user["worker"]
            source = user["sources"][0]
            if worker == "policy":
                takeover()
            body = (
                _policy_body(source["content"])
                if worker == "policy"
                else _contract_body(source["content"])
            )
            for finding in body["findings"]:
                finding["evidence_refs"] = [source["ref"]]
            for witness in body["witnesses"]:
                witness["source_ref"] = source["ref"]
            return httpx.Response(
                200,
                json={
                    "model": config.resolved_model,
                    "usage": {"prompt_tokens": 700, "completion_tokens": 260},
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {"schema_version": "finding-evidence.v3", **body},
                                    ensure_ascii=False,
                                )
                            },
                        }
                    ],
                },
            )

        self.transport = PreciseRunChatTransport(
            config,
            SecretStr("offline-only"),
            catalog=None,
            transport=httpx.MockTransport(respond),
        )


def _open_run(
    environment: Environment,
    runner: PreciseScopedRunner,
    *,
    lease_seconds: float = 600.0,
    deadline_seconds: float = 30.0,
):
    """Create one Run with generation 1 directly on the repository, as the service does."""

    import secrets

    from procurement_review.precise_run_state import run_isolation
    from procurement_review.trace import canonical_checksum

    token, credential_id = _issue(environment)
    context = _context(environment, token, SecurityOperation.REVIEW_CREATE)
    registration = environment.authorization.require_case(
        context,
        environment.case.case_id,
        request_id="g-counterexample",
        operation=SecurityOperation.REVIEW_CREATE,
    )
    run_id = f"run-{secrets.token_hex(16)}"
    digest = canonical_checksum({"create": run_id})
    environment.repository.create_run(
        run_id=run_id,
        trace_id=f"trace-{secrets.token_hex(16)}",
        tenant_id=registration.tenant_id,
        case_id=registration.case_id,
        case_version=registration.case_version,
        subject_id=context.subject_id,
        credential_id=context.credential_id,
        authorization_revision=context.authorization_revision,
        request_checksum=digest,
        document_version=registration.document_version,
        dataset_version=registration.dataset_version,
        index_version=registration.index_version,
        rules_version=registration.rules_version,
        isolation=run_isolation(registration, runner),
        idempotency_key=f"g-counterexample-{secrets.token_hex(4)}",
        request_digest=digest,
        owner_id="owner-" + secrets.token_hex(16),
        now=datetime.now(UTC),
        lease_seconds=lease_seconds,
        deadline_seconds=deadline_seconds,
        events=(),
    )
    run = environment.repository.load_run(run_id)
    scope, registration = _scope_and_registration(environment, generation=1)
    return run_id, run, scope, registration, credential_id


def _execute(runner, environment, run_id, run, scope, registration, *, clock=None):
    return asyncio.run(
        runner.run(
            run_id=run_id,
            scope=scope,
            registration=registration,
            generation=run.generation.generation,
            repository=environment.repository,
            deadline_at=run.generation.deadline_at,
            fault=None,
            heartbeat=None,
            owner_id=run.generation.owner_id,
            now=clock or (lambda: datetime.now(UTC)),
        )
    )


def test_a_superseded_generation_never_adopts_its_late_answer(environment):
    """A takeover lands while the answer is in flight; the old generation must stop."""

    calls: list = []
    holder: dict = {}

    def takeover() -> None:
        with environment.store.transaction() as connection:
            environment.repository._open_generation_locked(
                connection,
                run_id=holder["run_id"],
                subject_id=SUBJECT,
                credential_id=holder["credential_id"],
                authorization_revision=1,
                owner_id="owner-takeover",
                now=datetime.now(UTC),
                lease_seconds=600.0,
                deadline_seconds=30.0,
            )

    wire = _SupersedingTransport(environment, calls, takeover)
    runner = PreciseScopedRunner(store=environment.store, transport=wire.transport)
    run_id, run, scope, registration, credential_id = _open_run(environment, runner)
    holder.update(run_id=run_id, credential_id=credential_id)

    with pytest.raises(ExecutionInvalidated) as caught:
        _execute(runner, environment, run_id, run, scope, registration)
    # The stop carries the reason that actually happened, never a checkpoint story.
    assert caught.value.reason == "superseded_generation"
    assert len(calls) == 1
    # The answer of the superseded generation was refused at the consumption admission,
    # so it was never decoded, never grounded and never adopted: no branch is committed
    # and the reservation has no settlement.
    assert _checkpoint_workers(environment, run_id) == []
    rows = attempt_ledger_rows(environment.repository, run_id)
    assert len(rows) == 1
    assert rows[0]["worker"] == "policy" and rows[0]["finished_at"] is None

    # The newer generation re-asks the branch it never adopted and finishes normally.
    live = environment.repository.load_run(run_id)
    assert live.generation.generation == 2
    scope2, registration2 = _scope_and_registration(environment, generation=2)
    resumed_calls: list = []
    resumed_wire = _SupersedingTransport(environment, resumed_calls, lambda: None)
    resumed = PreciseScopedRunner(
        store=environment.store, transport=resumed_wire.transport
    )
    execution = _execute(resumed, environment, run_id, live, scope2, registration2)
    assert execution.policy.outcome == "completed"
    assert execution.contract.outcome == "completed"
    assert len(resumed_calls) == 2
    assert _checkpoint_workers(environment, run_id) == ["contract", "policy"]
    with environment.store.transaction() as connection:
        stores = {
            str(row[0]): int(row[1])
            for row in connection.execute(
                text(
                    "SELECT worker, generation FROM review_branch_checkpoint"
                    " WHERE run_id = :r"
                ),
                {"r": run_id},
            ).all()
        }
    assert stores == {"policy": 2, "contract": 2}


def test_a_commit_refused_by_a_superseded_generation_is_an_explicit_stop(environment):
    """The commit-time refusal: the reason is the guard's, and the ledger says so.

    The in-flight answer is judged at the consumption admission first, so this rule is
    exercised where it is reached — at the commit itself, with the real repository
    refusing it because another generation now owns the Run.
    """

    from procurement_review.gold_v3_contracts import payload_sha256
    from procurement_review.precise_run_contracts import (
        PreciseBranchCommitV1,
    )
    from procurement_review.precise_branch_records import resume_refused_commit
    from procurement_review.precise_run_state import commit_precise_branch

    wire = _SupersedingTransport(environment, [], lambda: None)
    runner = PreciseScopedRunner(store=environment.store, transport=wire.transport)
    run_id, run, _scope, _registration, credential_id = _open_run(environment, runner)
    environment.repository.begin_attempt(
        run_id=run_id,
        generation=1,
        worker="policy",
        attempt_no=1,
        now=datetime.now(UTC),
        owner_id=run.generation.owner_id,
    )
    # A second generation takes the Run over before the first one's commit lands.
    with environment.store.transaction() as connection:
        environment.repository._open_generation_locked(
            connection,
            run_id=run_id,
            subject_id=SUBJECT,
            credential_id=credential_id,
            authorization_revision=1,
            owner_id="owner-takeover",
            now=datetime.now(UTC),
            lease_seconds=600.0,
            deadline_seconds=30.0,
        )
    request_sha = payload_sha256({"request": "superseded-commit"})
    refused = commit_precise_branch(
        environment.repository,
        run_id=run_id,
        generation=1,
        worker="policy",
        attempt_no=1,
        input_checksum="sha256:" + request_sha,
        contract_checksum="sha256:" + payload_sha256({"contract": "superseded"}),
        payload=PreciseBranchCommitV1(
            worker="policy",
            duties=(_unknown_duty(),),
            spans=(),
            document_failures=(),
            support_failures=(),
            security=(),
            blocked=False,
            request_sha256=request_sha,
            answer_sha256=payload_sha256({"answer": "superseded"}),
            attempts=(),
            evidence_refs=(),
        ),
        now=lambda: datetime.now(UTC),
        owner_id=run.generation.owner_id,
    )
    assert refused.adopted is False and refused.reason == "superseded_generation"
    with pytest.raises(ExecutionInvalidated) as caught:
        resume_refused_commit(
            refused,
            repository=environment.repository,
            run_id=run_id,
            worker="policy",
            request=_policy_request(environment),
            request_sha=request_sha,
        )
    assert caught.value.reason == "superseded_generation"
    assert _checkpoint_workers(environment, run_id) == []
    # The attempt the refused commit belonged to is recorded as settled but not adopted.
    rows = attempt_ledger_rows(environment.repository, run_id)
    superseded = next(row for row in rows if row["worker"] == "policy")
    assert superseded["discarded_late"] is True


def test_a_duplicate_commit_of_an_adopted_branch_reuses_it_instead_of_failing(
    environment,
):
    """A commit that lost the race adopts the committed branch; the execution goes on."""

    from procurement_review.gold_v3_contracts import payload_sha256
    from procurement_review.precise_run_contracts import (
        PreciseAdoptedDutyV1,
        PreciseBranchCommitV1,
    )
    from procurement_review.precise_branch_records import resume_refused_commit
    from procurement_review.precise_run_state import commit_precise_branch

    wire = _SupersedingTransport(environment, [], lambda: None)
    runner = PreciseScopedRunner(store=environment.store, transport=wire.transport)
    run_id, run, _scope, _registration, _credential_id = _open_run(environment, runner)
    request = _policy_request(environment)
    request_sha = payload_sha256({"request": "duplicate-commit"})
    # A committed branch answers every check of its request; an unknown duty is a
    # complete and legal answer for the obligations the material does not carry.
    duties = tuple(
        PreciseAdoptedDutyV1(
            obligation_id=check.proposition.check_id,
            status="unknown",
            risk_level="unknown",
            value="UNKNOWN",
            evidence_refs=(),
        )
        for check in request.checks
    )
    # The branch another attempt of this Run already committed: an unknown duty, which
    # is a complete and legal answer on its own.
    adopted = commit_precise_branch(
        environment.repository,
        run_id=run_id,
        generation=1,
        worker="policy",
        attempt_no=1,
        input_checksum="sha256:" + request_sha,
        contract_checksum="sha256:" + payload_sha256({"contract": "duplicate"}),
        payload=PreciseBranchCommitV1(
            worker="policy",
            duties=duties,
            spans=(),
            document_failures=(),
            support_failures=(),
            security=(),
            blocked=False,
            request_sha256=request_sha,
            answer_sha256=payload_sha256({"answer": "committed-first"}),
            attempts=(),
            evidence_refs=(),
        ),
        now=lambda: datetime.now(UTC),
        owner_id=run.generation.owner_id,
    )
    assert adopted.adopted is True
    environment.repository.begin_attempt(
        run_id=run_id,
        generation=1,
        worker="policy",
        attempt_no=2,
        now=datetime.now(UTC),
        owner_id=run.generation.owner_id,
    )
    replay = commit_precise_branch(
        environment.repository,
        run_id=run_id,
        generation=1,
        worker="policy",
        attempt_no=1,
        input_checksum="sha256:" + request_sha,
        contract_checksum="sha256:" + payload_sha256({"contract": "duplicate"}),
        payload=PreciseBranchCommitV1(
            worker="policy",
            duties=duties,
            spans=(),
            document_failures=(),
            support_failures=(),
            security=(),
            blocked=False,
            request_sha256=request_sha,
            answer_sha256=payload_sha256({"answer": "late-copy"}),
            attempts=(),
            evidence_refs=(),
        ),
        now=lambda: datetime.now(UTC),
        owner_id=run.generation.owner_id,
    )
    assert replay.adopted is False and replay.reason == "superseded_attempt"
    outcome = resume_refused_commit(
        replay,
        repository=environment.repository,
        run_id=run_id,
        worker="policy",
        request=request,
        request_sha=request_sha,
    )
    # The execution continues with the branch the Run actually adopted: not its own
    # answer, not a failure.
    assert outcome.record.outcome.value == "completed"
    assert outcome.record.answer_sha256 == payload_sha256({"answer": "committed-first"})
    assert [item.obligation_id for item in outcome.record.reported] == [
        check.proposition.check_id for check in request.checks
    ]
    assert all(item.normalized is None for item in outcome.record.reported)


def _unknown_duty():
    """One legal duty: an obligation the material does not answer."""

    from procurement_review.precise_run_contracts import PreciseAdoptedDutyV1

    return PreciseAdoptedDutyV1(
        obligation_id="O-01",
        status="unknown",
        risk_level="unknown",
        value="UNKNOWN",
        evidence_refs=(),
    )


def _policy_request(environment: Environment):
    """One real policy request, built the way the runner builds it.

    The commit-time rules do not depend on retrieved sources: an all-unknown committed
    branch carries no citations, so an empty source set is a complete input for it.
    """

    from procurement_review.business_v31_documents import document_request_v31
    from procurement_review.business_v3_types import BusinessContextV3
    from procurement_review.gold_v37_conditional import as_gold_case
    from procurement_review.precise_run_scope import PRECISE_QUESTIONS

    gold = as_gold_case(environment.case)
    facts = gold.model_input.business_facts
    return document_request_v31(
        BusinessContextV3(
            scope_id=facts.scope_id,
            purchase_order_id=str(facts.purchase_order_id),
            supplier_id=str(facts.supplier_id),
            contract_id=str(facts.contract_reference),
            as_of=gold.as_of,
            dataset_version=facts.source_version,
        ),
        case_id=environment.case.case_id,
        worker="policy",
        question=PRECISE_QUESTIONS["policy"],
        sources=(),
    )


def test_an_answer_that_arrives_after_the_generation_deadline_is_not_adopted(
    environment,
):
    """The clock moves past the deadline while the answer is in flight."""

    calls: list = []
    clock = {"now": datetime.now(UTC)}
    wire = _SupersedingTransport(
        environment,
        calls,
        lambda: clock.update(now=clock["now"] + timedelta(seconds=45)),
    )
    runner = PreciseScopedRunner(store=environment.store, transport=wire.transport)
    run_id, run, scope, registration, _credential_id = _open_run(
        environment, runner, deadline_seconds=30.0
    )
    with pytest.raises(ExecutionInvalidated) as caught:
        _execute(
            runner,
            environment,
            run_id,
            run,
            scope,
            registration,
            clock=lambda: clock["now"],
        )
    assert caught.value.reason == "generation_deadline_exceeded"
    assert len(calls) == 1
    assert _checkpoint_workers(environment, run_id) == []
    # The answer was refused at the consumption admission: it was physically received and
    # then never decoded, so the reservation has no settlement at all.
    rows = attempt_ledger_rows(environment.repository, run_id)
    assert len(rows) == 1 and rows[0]["worker"] == "policy"
    assert rows[0]["finished_at"] is None


# -------------------------------------------------------- a stored commit that is damaged


def test_a_committed_branch_that_no_longer_satisfies_its_record_stops_the_execution(
    environment,
):
    """A rewritten payload is refused, never reinterpreted as this execution's answer."""

    calls: list = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        payload = json.loads(request.content)
        user = json.loads(payload["messages"][1]["content"])
        worker = user["worker"]
        source = user["sources"][0]
        if worker == "contract" and len(calls) == 2:
            raise RuntimeError("the executing process died mid-send")
        body = (
            _policy_body(source["content"])
            if worker == "policy"
            else _contract_body(source["content"])
        )
        for finding in body["findings"]:
            finding["evidence_refs"] = [source["ref"]]
        for witness in body["witnesses"]:
            witness["source_ref"] = source["ref"]
        return httpx.Response(
            200,
            json={
                "model": configuration().resolved_model,
                "usage": {"prompt_tokens": 700, "completion_tokens": 260},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"schema_version": "finding-evidence.v3", **body},
                                ensure_ascii=False,
                            )
                        },
                    }
                ],
            },
        )

    transport = PreciseRunChatTransport(
        configuration(),
        SecretStr("offline-only"),
        catalog=None,
        transport=httpx.MockTransport(respond),
    )
    runner = PreciseScopedRunner(store=environment.store, transport=transport)
    run_id, run, scope, registration, _credential_id = _open_run(environment, runner)
    with pytest.raises(RuntimeError):
        _execute(runner, environment, run_id, run, scope, registration)
    assert _checkpoint_workers(environment, run_id) == ["policy"]

    with environment.store.transaction() as connection:
        connection.execute(
            text(
                "UPDATE review_branch_checkpoint SET result_payload = jsonb_set("
                "result_payload, '{result,duties,0,status}', '\"frozen\"') "
                "WHERE run_id = :r AND worker = 'policy'"
            ),
            {"r": run_id},
        )
    sent_before = len(calls)
    resumed = PreciseScopedRunner(store=environment.store, transport=transport)
    with pytest.raises(IncompatibleCheckpointError):
        _execute(resumed, environment, run_id, run, scope, registration)
    # The refusal happened before anything was sent, and the damaged row was not repaired.
    assert len(calls) == sent_before
    with pytest.raises(IncompatibleCheckpointError):
        adopted_branch_commit(environment.repository, run_id, "policy")


# ------------------------------------------------------ repeated keys and the local receipt


def test_a_repeated_key_never_starts_a_second_run_or_a_second_receipt(environment):
    """Same-key create and same-key confirm both replay; nothing is done twice."""

    token, _credential_id = _issue(environment)
    with _client(environment) as client:
        first = _create(client, environment, token, key="g-counterexample-replay")
        assert first.status_code in {200, 201, 202}, first.text
        run_id = first.json()["run_id"]
        second = _create(client, environment, token, key="g-counterexample-replay")
        assert second.status_code in {200, 201, 202}, second.text
        assert second.json()["run_id"] == run_id
        replay_epoch = datetime.now(UTC)
        body = _await_run(client, token, run_id, environment)
        assert body["status"] == "pending_confirmation", body
        # The replay answered from the stored record: exactly one execution, two sends.
        assert len(environment.requests) == 2
        assert environment.repository.attempt_totals(run_id) == {
            "policy": 1,
            "contract": 1,
        }
        # A zero-attempt second execution would have been visible as earlier receipts.
        assert replay_epoch <= datetime.now(UTC)

        confirm = {
            "api_version": "review-api.v2",
            "expected_version": body["revision"],
            "expected_summary_checksum": body["summary_checksum"],
            "idempotency_key": "g-counterexample-confirm",
        }
        accepted = client.post(
            f"/api/v2/reviews/{run_id}/confirm",
            headers={"Authorization": f"Bearer {token}"},
            json=confirm,
        )
        assert accepted.status_code == 200, accepted.text
        replayed = client.post(
            f"/api/v2/reviews/{run_id}/confirm",
            headers={"Authorization": f"Bearer {token}"},
            json=confirm,
        )
        assert replayed.status_code == 200, replayed.text
        assert replayed.json()["status"] == "confirmed"
        assert (
            replayed.json()["receipt"]["receipt_id"]
            == accepted.json()["receipt"]["receipt_id"]
        )
        assert accepted.json()["receipt"]["run_id"] == run_id
        assert accepted.json()["receipt"]["case_id"] == environment.case.case_id
        # The local receipt is stored once, and the audit names one confirmation.
        with environment.store.transaction() as connection:
            receipts = connection.execute(
                text("SELECT count(*) FROM review_receipt WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
            confirmations = connection.execute(
                text(
                    "SELECT count(*) FROM review_audit_event WHERE run_id = :r"
                    " AND action_type = 'confirm'"
                ),
                {"r": run_id},
            ).scalar_one()
        assert int(receipts) == 1
        assert int(confirmations) == 1
        # A different key on the now-decided Run cannot silently re-run the action.
        stale = client.post(
            f"/api/v2/reviews/{run_id}/confirm",
            headers={"Authorization": f"Bearer {token}"},
            json={**confirm, "idempotency_key": "g-counterexample-confirm-2"},
        )
        assert stale.status_code == 409, stale.text
    assert len(environment.requests) == 2


def _await_run(
    client, token: str, run_id: str, environment: Environment, *, attempts: int = 200
):
    for _ in range(attempts):
        response = client.get(
            f"/api/v2/reviews/{run_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in {
            "pending_confirmation",
            "pending_human_review",
            "failed",
            "cancelled",
            "interrupted",
        }:
            return body
        time.sleep(0.05)
    raise AssertionError("the Run never reached a gate state")


# ---------------------------------------------------------------- the process rebuild


class _Loop:
    """One event loop in its own thread, so a service call can drive its own task."""

    def __init__(self) -> None:
        import threading

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def start(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def close(self) -> None:
        import contextlib

        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
        with contextlib.suppress(Exception):
            self.loop.close()


def _service_over(url: str, case: Any):
    """One service on a fresh engine over the same database: a rebuilt process."""

    store = ReviewStore.from_url(url)
    requests: list = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        user = json.loads(payload["messages"][1]["content"])
        worker = user["worker"]
        source = user["sources"][0]
        body = (
            _policy_body(source["content"])
            if worker == "policy"
            else _contract_body(source["content"])
        )
        for finding in body["findings"]:
            finding["evidence_refs"] = [source["ref"]]
        for witness in body["witnesses"]:
            witness["source_ref"] = source["ref"]
        return httpx.Response(
            200,
            json={
                "model": configuration().resolved_model,
                "usage": {"prompt_tokens": 700, "completion_tokens": 260},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"schema_version": "finding-evidence.v3", **body},
                                ensure_ascii=False,
                            )
                        },
                    }
                ],
            },
        )

    transport = PreciseRunChatTransport(
        configuration(),
        SecretStr("offline-only"),
        catalog=None,
        transport=httpx.MockTransport(respond),
    )
    runner = PreciseScopedRunner(store=store, transport=transport)
    repository = ReviewRepository(store)
    authorization = AuthorizationService(store=store)
    service = ReviewService(repository=repository, runner=runner)
    return store, repository, authorization, service, requests, runner


def test_a_process_rebuild_resumes_from_persisted_state(environment):
    """The first process dies mid-send; a rebuilt process continues from the ledger."""

    calls: list = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        payload = json.loads(request.content)
        user = json.loads(payload["messages"][1]["content"])
        worker = user["worker"]
        source = user["sources"][0]
        if worker == "contract":
            raise RuntimeError("the executing process died mid-send")
        body = _policy_body(source["content"])
        for finding in body["findings"]:
            finding["evidence_refs"] = [source["ref"]]
        for witness in body["witnesses"]:
            witness["source_ref"] = source["ref"]
        return httpx.Response(
            200,
            json={
                "model": configuration().resolved_model,
                "usage": {"prompt_tokens": 700, "completion_tokens": 260},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"schema_version": "finding-evidence.v3", **body},
                                ensure_ascii=False,
                            )
                        },
                    }
                ],
            },
        )

    transport = PreciseRunChatTransport(
        configuration(),
        SecretStr("offline-only"),
        catalog=None,
        transport=httpx.MockTransport(respond),
    )
    runner = PreciseScopedRunner(store=environment.store, transport=transport)
    run_id, run, scope, registration, credential_id = _open_run(
        environment, runner, lease_seconds=1.0
    )
    with pytest.raises(RuntimeError):
        _execute(runner, environment, run_id, run, scope, registration)
    assert _checkpoint_workers(environment, run_id) == ["policy"]
    environment.store.close()

    # A rebuilt process: a new engine, a new repository, a new service, no in-memory
    # state of the previous execution anywhere.
    store, repository, authorization, service, rebuilt_calls, rebuilt_runner = (
        _service_over(environment.database_url, environment.case)
    )
    loop = _Loop()
    try:
        deadline = time.monotonic() + 30.0
        interrupted: tuple[str, ...] = ()
        while time.monotonic() < deadline:
            interrupted = repository.scan_expired_leases(
                now=datetime.now(UTC),
                events_factory=lambda _run_id: (),
                audit_factory=lambda _run_id: None,
            )
            if interrupted:
                break
            time.sleep(0.1)
        assert interrupted == (run_id,)
        token, credential_id2, digest = generate_credential()
        issued_at = datetime.now(UTC)
        store.issue_credential(
            credential_id=credential_id2,
            tenant_id=TENANT,
            subject_id=SUBJECT,
            token_digest=digest,
            issued_at=issued_at,
            expires_at=issued_at + timedelta(seconds=3_600),
        )
        context = authorization.resolve(
            token, request_id="g-rebuild", operation=SecurityOperation.REVIEW_RESUME
        )
        authorization.require_case(
            context,
            environment.case.case_id,
            request_id="g-rebuild",
            operation=SecurityOperation.REVIEW_RESUME,
        )
        current = repository.load_run(run_id)
        handle = loop.start(
            service.resume_review(
                run_id,
                ResumeRequestV2(
                    expected_version=current.header.revision,
                    idempotency_key="g-counterexample-rebuild",
                ),
                context=context,
                idempotency_key="g-counterexample-rebuild",
            )
        )
        handle.result(timeout=120)
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            live = repository.load_run(run_id)
            if live.header.status.value in {
                "pending_confirmation",
                "failed",
                "interrupted",
            }:
                break
            time.sleep(0.1)
        final = repository.load_run(run_id)
        loop.start(service.shutdown()).result(timeout=30)
    finally:
        loop.close()
        store.close()
    # The rebuilt process sent only the branch that was never committed, and the Run
    # finished under its second generation with both branches adopted.
    assert len(rebuilt_calls) == 1
    assert len(calls) == 2
    assert final.header.status.value == "pending_confirmation", final.header
    assert _checkpoint_workers_over(repository, run_id) == ["contract", "policy"]
    assert repository.attempt_totals(run_id) == {"policy": 1, "contract": 2}
    assert rebuilt_runner.branch_telemetry.reused == 1
    from procurement_review.precise_run_contracts import ExecutionRecord

    record = ExecutionRecord.model_validate_json(
        json.dumps(final.execution_payload), strict=True
    ).root
    # The ledger spans both processes: the contract branch was sent once before the
    # rebuild and once after it, and the policy branch was never sent twice.  Only the
    # attempts whose own receipt survived prove a send, so the interrupted attempt is
    # reported as unresolved rather than as a first send.
    assert record.metering.physical_attempts == 3
    assert record.metering.first_sends == 1
    assert record.metering.proven_sends == 2
    assert record.metering.unresolved_attempts == 1
    assert record.metering.corrections == 1
    assert record.metering.usage_complete is False
    assert record.metering.explicit_recoveries == 1
    assert record.policy.outcome.value == "completed"
    assert record.contract.outcome.value == "completed"
    # The branch sent by the dead process has no receipt at all: its usage is unknown
    # rather than zero, and the interruption closed its reservation.
    assert record.metering.unknown_usage_attempts >= 1


def _checkpoint_workers_over(repository: ReviewRepository, run_id: str) -> list[str]:
    with repository.store.transaction() as connection:
        return [
            str(row[0])
            for row in connection.execute(
                text(
                    "SELECT worker FROM review_branch_checkpoint WHERE run_id = :r"
                    " ORDER BY worker"
                ),
                {"r": run_id},
            ).all()
        ]


# -------------------------------------------- the stored-record guard on a real branch


def _stored_checkpoint_payload(environment: Environment, run_id: str, worker: str):
    with environment.store.transaction() as connection:
        row = connection.execute(
            text(
                "SELECT result_payload FROM review_branch_checkpoint"
                " WHERE run_id = :r AND worker = :w"
            ),
            {"r": run_id, "w": worker},
        ).first()
    payload = row[0]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return dict(payload)


def test_a_digest_slot_is_not_read_as_a_phone_number(environment):
    """A hash that happens to contain a phone-shaped digit run is still a digest."""

    from procurement_review.record_guard import guard_record
    from procurement_review.run_errors import RecordContractError

    calls: list = []
    wire = _SupersedingTransport(environment, calls, lambda: None)
    runner = PreciseScopedRunner(store=environment.store, transport=wire.transport)
    run_id, run, scope, registration, _credential_id = _open_run(environment, runner)
    _execute(runner, environment, run_id, run, scope, registration)
    payload = _stored_checkpoint_payload(environment, run_id, "policy")
    # 64 hex characters, with 13800138000 inside: a real digest slot may contain that
    # run, and refusing it would refuse a record the product itself wrote.
    payload["result"]["spans"][0]["span_sha256"] = "13" + "987654321" + "a" * 53
    accepted = guard_record(
        "checkpoint_payload", payload, location="probe", strict=False
    )
    assert accepted is not None
    payload["result"]["duties"][0]["value"] = "13800138000"
    with pytest.raises(RecordContractError):
        guard_record("checkpoint_payload", payload, location="probe", strict=False)


def test_a_phone_shaped_value_in_a_branch_output_is_a_security_stop():
    """The branch's own output inspection catches it before anything is persisted."""

    case = conditional_cases()[CASE_INDEX]
    # The material itself names the counterparty with a phone-shaped identifier: the
    # branch may ground it, and the product may not persist it.
    hostile = case.model_copy(
        update={
            "contract_text": case.contract_text.replace(
                str(case.facts["counterparty_reference"]), "13800138000"
            )
        }
    )
    generated = _environment(case=hostile)
    try:
        environment = next(generated)
        calls: list = []
        wire = _SupersedingTransport(environment, calls, lambda: None)
        runner = PreciseScopedRunner(store=environment.store, transport=wire.transport)
        run_id, run, scope, registration, _credential_id = _open_run(
            environment, runner
        )
        execution = _execute(runner, environment, run_id, run, scope, registration)
        assert execution.policy.outcome.value == "completed"
        # The contract branch named a phone-shaped party; its own output inspection
        # escalated, so the branch ends with that code and the generation stops sending
        # instead of persisting the value or retrying it.
        assert execution.contract.outcome.value == "failed"
        assert execution.contract.failure_code == "security_boundary"
        assert execution.stop_reason == "security_boundary"
        assert _checkpoint_workers(environment, run_id) == ["policy"]
        assert "security_boundary" in execution.summary.failure_reasons
        # Nothing was persisted for the refused branch: no partial write survived.
        with environment.store.transaction() as connection:
            stored = connection.execute(
                text(
                    "SELECT count(*) FROM review_branch_checkpoint"
                    " WHERE run_id = :r AND worker = 'contract'"
                ),
                {"r": run_id},
            ).scalar_one()
        assert int(stored) == 0
    finally:
        generated.close()


def test_a_commit_the_record_guard_refuses_names_its_category(environment):
    """A refused payload is refused before any write, and the code names the category."""

    from procurement_review.gold_v3_contracts import payload_sha256
    from procurement_review.precise_branch_records import guard_failure_code
    from procurement_review.precise_run_contracts import (
        PreciseAdoptedDutyV1,
        PreciseBranchCommitV1,
    )
    from procurement_review.precise_run_state import commit_precise_branch
    from procurement_review.run_errors import RecordContractError

    wire = _SupersedingTransport(environment, [], lambda: None)
    runner = PreciseScopedRunner(store=environment.store, transport=wire.transport)
    run_id, run, _scope, _registration, _credential_id = _open_run(environment, runner)
    environment.repository.begin_attempt(
        run_id=run_id,
        generation=1,
        worker="policy",
        attempt_no=1,
        now=datetime.now(UTC),
        owner_id=run.generation.owner_id,
    )
    request_sha = payload_sha256({"request": "guard"})
    with pytest.raises(RecordContractError) as caught:
        commit_precise_branch(
            environment.repository,
            run_id=run_id,
            generation=1,
            worker="policy",
            attempt_no=1,
            input_checksum="sha256:" + request_sha,
            contract_checksum="sha256:" + payload_sha256({"contract": "guard"}),
            payload=PreciseBranchCommitV1(
                worker="policy",
                duties=(
                    PreciseAdoptedDutyV1(
                        obligation_id="O-01",
                        status="supported",
                        risk_level="low",
                        value="13800138000",
                        evidence_refs=(),
                    ),
                ),
                spans=(),
                document_failures=(),
                support_failures=(),
                security=(),
                blocked=False,
                request_sha256=request_sha,
                answer_sha256=payload_sha256({"answer": "guard"}),
                attempts=(),
                evidence_refs=(),
            ),
            now=lambda: datetime.now(UTC),
            owner_id=run.generation.owner_id,
        )
    assert "cn_mobile_pii" in caught.value.categories
    assert guard_failure_code(caught.value.categories) == (
        "record_guard_refused:cn_mobile_pii"
    )
    # The refusal happened before any write: no checkpoint row exists.
    assert _checkpoint_workers(environment, run_id) == []
