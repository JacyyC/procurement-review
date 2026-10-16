"""A refused output must escalate to a human path, not read as a plain pipeline failure.

The 8002 observation was that a branch stopped by the output boundary ended as
``insufficient_information``: the boundary refusal became an ordinary pipeline failure and
no trusted security fact reached the decision.  These cases pin the two directions of the
rule: an output refusal carries a fixed category and a location identity and escalates the
recommendation to ``needs_human_review``, while a benign refusal in the same position keeps
its own code and does not escalate by itself.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from procurement_review.api import create_app  # noqa: E402
from procurement_review.authorization import (  # noqa: E402
    AuthorizationService,
    generate_credential,
)
from procurement_review.eval_f_registration import configuration  # noqa: E402
from procurement_review.gold_v37_conditional import conditional_cases  # noqa: E402
from procurement_review.precise_run_runner import PreciseScopedRunner  # noqa: E402
from procurement_review.precise_run_transport import PreciseRunChatTransport  # noqa: E402
from procurement_review.review_repository import ReviewRepository  # noqa: E402
from procurement_review.review_service import ReviewService  # noqa: E402
from procurement_review.review_store import ReviewStore  # noqa: E402
from test_v1_07g_precise_run import (  # noqa: E402
    SUBJECT,
    TENANT,
    _contract_body,
    _plan,
    _policy_body,
)
from v1_03_support import create_isolated_database, drop_database  # noqa: E402

CANARY = "synthetic_secret_placeholder"
BENIGN = "该请求超出只读审查范围，我不提供该内容。"


@dataclass(slots=True)
class Environment:
    store: ReviewStore
    authorization: AuthorizationService
    repository: ReviewRepository
    service: ReviewService
    case: object
    requests: list


def _handler(requests: list, statement: str) -> httpx.MockTransport:
    """One Mock wire whose answer carries ``statement`` in the branch's own findings."""

    config = configuration()

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
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
            finding["status"] = "unknown"
            finding["normalized_value"] = None
            finding["missing_information"] = [statement]
        body["witnesses"] = []
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

    return httpx.MockTransport(respond)


def _environment(statement: str, case_index: int = 0) -> Iterator[Environment]:
    database_url, database = create_isolated_database()
    store = ReviewStore.from_url(database_url)
    store.migrate()
    case = conditional_cases()[case_index]
    store.seed(_plan(case))
    authorization = AuthorizationService(store=store)
    repository = ReviewRepository(store)
    requests: list = []
    service = ReviewService(
        repository=repository,
        runner=PreciseScopedRunner(
            store=store, transport=_transport(requests, statement)
        ),
    )
    try:
        yield Environment(
            store=store,
            authorization=authorization,
            repository=repository,
            service=service,
            case=case,
            requests=requests,
        )
    finally:
        store.close()
        drop_database(database)


def _transport(requests: list, statement: str) -> PreciseRunChatTransport:
    return PreciseRunChatTransport(
        configuration(),
        SecretStr("offline-only"),
        catalog=None,
        transport=_handler(requests, statement),
    )


def _run(environment: Environment) -> dict:
    from fastapi.testclient import TestClient

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
    application = create_app(
        store=environment.store,
        authorization=environment.authorization,
        repository=environment.repository,
        service=environment.service,
    )
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(application, base_url="http://127.0.0.1") as client:
        created = client.post(
            "/api/v2/reviews",
            headers={
                **headers,
                "Idempotency-Key": f"escalation-{environment.case.case_id}",
            },
            json={
                "api_version": "review-api.v2",
                "case_id": environment.case.case_id,
                "case_version": f"{environment.case.case_id.lower()}.v1",
            },
        )
        assert created.status_code in {200, 201, 202}, created.text
        run_id = created.json()["run_id"]
        for _ in range(200):
            body = client.get(f"/api/v2/reviews/{run_id}", headers=headers).json()
            if body["status"] in {
                "pending_confirmation",
                "pending_human_review",
                "failed",
                "cancelled",
            }:
                return body
            import time

            time.sleep(0.05)
    raise AssertionError("the Run never reached a terminal state")


@pytest.fixture()
def canary_environment() -> Iterator[Environment]:
    yield from _environment(CANARY)


@pytest.fixture()
def benign_environment() -> Iterator[Environment]:
    yield from _environment(BENIGN)


def test_a_refused_output_escalates_with_a_trusted_security_event(canary_environment):
    body = _run(canary_environment)
    assert body["status"] == "pending_human_review", body
    execution = body["execution"]
    assert execution["execution_version"] == "review-execution.v2"
    # The decision itself is a human one, and it names the fixed category that caused it.
    assert execution["decision"]["recommendation"] == "needs_human_review"
    assert "canary_output_blocked" in execution["decision"]["security_events"]
    events = {event["code"]: event for event in execution["security"]}
    assert "canary_output_blocked" in events
    # The event is a trusted, locatable fact: origin and a hashed location, no payload.
    blocked = events["canary_output_blocked"]
    assert blocked["origin"] == "output"
    assert len(blocked["location_sha256"]) == 64
    assert CANARY not in json.dumps(execution, ensure_ascii=False)
    # The refused branch keeps its own control code instead of a generic failure.
    codes = {execution[worker]["failure_code"] for worker in ("policy", "contract")}
    assert "security_boundary" in codes
    assert execution["metering"]["unknown_usage_attempts"] == 0


def test_a_benign_refusal_keeps_its_own_code_and_does_not_escalate_by_itself(
    benign_environment,
):
    body = _run(benign_environment)
    execution = body["execution"]
    # A refusal the boundary reads as benign is not a security event: the Case is judged on
    # what its branches did report, and the refusal never becomes a hidden approval.
    security_codes = {event["code"] for event in execution["security"]}
    assert "benign_refusal" not in security_codes
    assert "canary_output_blocked" not in security_codes
    assert execution["decision"]["recommendation"] != "recommend_approve"


def test_a_contract_correction_reaches_its_decision_and_persists_its_diagnostic():
    """A real correction must survive persistence; its tokens are facts, not prose."""

    requests: list = []
    seen = {"count": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
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
        if worker == "policy" and seen["count"] == 0:
            # One declared state combination the contract forbids: a decided duty without
            # its citation.  The branch may spend one bounded correction on it.
            body["findings"][0]["evidence_refs"] = []
        seen["count"] += 1
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

    database_url, database = create_isolated_database()
    store = ReviewStore.from_url(database_url)
    store.migrate()
    case = conditional_cases()[0]
    store.seed(_plan(case))
    repository = ReviewRepository(store)
    service = ReviewService(
        repository=repository,
        runner=PreciseScopedRunner(
            store=store,
            transport=PreciseRunChatTransport(
                configuration(),
                SecretStr("offline-only"),
                catalog=None,
                transport=httpx.MockTransport(respond),
            ),
        ),
    )
    try:
        environment = Environment(
            store=store,
            authorization=AuthorizationService(store=store),
            repository=repository,
            service=service,
            case=case,
            requests=requests,
        )
        body = _run(environment)
        assert body["status"] == "pending_confirmation", body
        metering = body["execution"]["metering"]
        # The shared budget covered one contract correction, and the first attempt is not
        # remembered as a first-pass success.
        assert metering["contract_corrections"] == 1
        assert metering["citation_corrections"] == 0
        assert metering["proven_sends"] == 3
        assert metering["first_sends"] == 2
        # The refused attempt's own bounded location is persisted on its trace receipt.
        run = repository.load_run(body["run_id"])
        receipts = [
            dict(event["output_summary"])
            for event in run.trace_events
            if dict(event).get("step") == "model_attempt"
        ]
        refused = [
            item for item in receipts if item.get("attempt_error") == "invalid_schema"
        ]
        assert refused
        assert "category=value_rule" in refused[0]["format_diagnostics"]
        assert "rule=decided_value_or_citation" in refused[0]["format_diagnostics"]
        assert refused[0]["next_correction_kind"] == "contract"
    finally:
        store.close()
        drop_database(database)
