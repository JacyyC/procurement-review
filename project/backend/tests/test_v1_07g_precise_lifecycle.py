"""V1-07G precise line: the v3 answer contract and the execution lifecycle.

Two things are exercised here that the happy-path chain cannot show:

* the ``finding-evidence.v3`` answer contract — the mechanical fields the server derives,
  the value bound that the richest registered duty actually needs, and the refusals that
  name one location instead of a union's worth of noise;
* the lifecycle of one Run — a committed branch reused without a second send, metering
  recomputed from the persistent ledger across generations, a drifted runtime stopped
  before anything is sent, an explicit stop path that keeps the original code, and a
  branch that never answered ending in a controlled state instead of an exception.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from procurement_review.business_v3_security import ProtectedOutputV3  # noqa: E402
from procurement_review.business_v31_documents import (  # noqa: E402
    document_request_v31,
)
from procurement_review.business_v3_types import BusinessContextV3  # noqa: E402
from procurement_review.checkpointing import (  # noqa: E402
    IncompatibleCheckpointError,
)
from procurement_review.eval_f_context import located_sources  # noqa: E402
from procurement_review.gold_v37_conditional import (  # noqa: E402
    as_gold_case,
    conditional_cases,
)
from procurement_review.model_wire_v2 import WireIssue  # noqa: E402
from procurement_review.precise_answer_v3 import (  # noqa: E402
    PRECISE_VALUE_LIMIT,
    parse_precise_answer,
)
from procurement_review.precise_run_state import run_isolation  # noqa: E402
from procurement_review.security_contracts import SecurityOperation  # noqa: E402
from procurement_review.trace import canonical_checksum  # noqa: E402
from test_v1_07g_precise_run import (  # noqa: E402
    _contract_body,
    _environment,
    _policy_body,
    _scope_and_registration,
    _token,
)

CASE_INDEX = 0


# --------------------------------------------------------------- the v3 answer contract


def _request(worker: str = "policy"):
    case = conditional_cases()[CASE_INDEX]
    gold = as_gold_case(case)
    facts = gold.model_input.business_facts
    context = BusinessContextV3(
        scope_id=facts.scope_id,
        purchase_order_id=str(facts.purchase_order_id),
        supplier_id=str(facts.supplier_id),
        contract_id=str(facts.contract_reference),
        as_of=gold.as_of,
        dataset_version=facts.source_version,
    )
    sources = tuple(
        item.source for item in located_sources(gold) if item.source.worker == worker
    )
    return (
        document_request_v31(
            context,
            case_id=case.case_id,
            worker=worker,
            question=case.question,
            sources=sources,
        ),
        sources,
    )


def _parse(body: dict, worker: str = "policy"):
    request, sources = _request(worker)
    return (
        request,
        sources,
        parse_precise_answer(
            json.dumps(body, ensure_ascii=False),
            request,
            protection=ProtectedOutputV3(),
        ),
    )


def _refusal(body: dict, worker: str = "policy") -> WireIssue:
    request, _sources = _request(worker)
    with pytest.raises(WireIssue) as caught:
        parse_precise_answer(
            json.dumps(body, ensure_ascii=False),
            request,
            protection=ProtectedOutputV3(),
        )
    return caught.value


def _rich_rule() -> dict:
    """The O-04 policy rule of the measured refusal: its canonical text exceeds 256."""

    return {
        "minimum_quotes": 3,
        "applies_to": ["purchase", "service"],
        "excluded_categories": ["lease"],
        "exclusion_basis": "本政策的竞争性报价要求不适用于租赁类交易，须以租赁方式执行的采购不适用竞争性报价要求。",
        "exceptions": ["single_source"],
        "exception_requires": ["written_justification", "approval_reference"],
        "exception_basis": "登记例外：单一来源采购经书面说明与审批授权后不适用竞争性报价要求。",
    }


def _rich_answer() -> dict:
    """A v3 answer whose O-04 value carries the full conditional rule."""

    case = conditional_cases()[CASE_INDEX]
    source = located_sources(as_gold_case(case))
    policy_source = next(
        item.source for item in source if item.source.worker == "policy"
    )
    content = policy_source.content
    rule_value = _rich_rule()
    canonical = json.dumps(rule_value, ensure_ascii=False, separators=(",", ":"))
    assert len(canonical) > 256, len(canonical)
    line = next(
        (
            item
            for item in content.split("\n")
            if "报价规则" in item or "不适用竞争性报价要求" in item
        ),
        content.split("\n")[0],
    )
    start = content.index(line)
    findings = [
        {
            "obligation_id": "O-01",
            "status": "supported",
            "normalized_value": {"applies": True, "basis": "2026", "exempt": []},
            "evidence_refs": [policy_source.ref],
            "statement": "原文提取",
            "recommended_action": "人工核对来源",
        },
        {
            "obligation_id": "O-02",
            "status": "unknown",
            "missing_information": ["专项条款的具体值未在授权材料中给出"],
            "statement": "材料不足",
            "recommended_action": "补充所缺证据",
        },
        {
            "obligation_id": "O-04",
            "status": "supported",
            "normalized_value": rule_value,
            "evidence_refs": [policy_source.ref],
            "statement": "原文提取",
            "recommended_action": "人工核对来源",
        },
    ]
    witnesses = [
        {
            "obligation_id": "O-04",
            "field": "applies_to",
            "value": ["purchase", "service"],
            "source_ref": policy_source.ref,
            "start": start,
            "end": start + len(line),
        }
    ]
    return {
        "schema_version": "finding-evidence.v3",
        "findings": findings,
        "witnesses": witnesses,
    }


def test_the_v3_wire_accepts_the_value_the_frozen_bound_refused():
    """The measured refusal: the policy O-04 rule does not fit the frozen 256 bound."""

    body = _rich_answer()
    _request_obj, _sources, (document, witnesses) = _parse(body)
    assert "O-04" in document.values
    rule = document.values["O-04"]
    assert rule.minimum_quotes == 3
    assert rule.excluded_categories == ("lease",)
    canonical = document.findings[2].finding.normalized_value
    assert 256 < len(canonical) <= PRECISE_VALUE_LIMIT
    # The list witness is stored in the canonical form the grounding comparison uses.
    assert witnesses[0].value == '["purchase","service"]'


def test_the_v3_wire_derives_the_mechanical_fields():
    """assertion, the severity of the non-violated states and the unknown literal."""

    body = _rich_answer()
    _request_obj, _sources, (document, _witnesses) = _parse(body)
    by_duty = {f.finding.obligation_id: f.finding for f in document.findings}
    assert by_duty["O-01"].assertion == "compliant"
    assert by_duty["O-01"].risk_level == "none"
    assert by_duty["O-01"].missing_information == ()
    assert by_duty["O-02"].assertion == "unknown"
    assert by_duty["O-02"].risk_level == "unknown"
    assert by_duty["O-02"].normalized_value == "UNKNOWN"
    assert by_duty["O-02"].evidence_refs == ()


def test_a_violation_without_its_severity_is_refused_by_name():
    body = _rich_answer()
    body["findings"][2]["status"] = "violated"
    refusal = _refusal(body)
    assert refusal.code == "invalid_schema" and refusal.correctable
    assert refusal.fields == ("risk_level",)
    assert "path=risk_level" in refusal.details
    assert "category=required_missing" in refusal.details
    # One refused field, one root cause: no union's worth of alternatives.
    assert sum(1 for token in refusal.details if token.startswith("path=")) == 1


def test_the_unknown_state_cannot_carry_a_value_or_a_citation():
    body = _rich_answer()
    body["findings"][1]["normalized_value"] = {"applies": True}
    refusal = _refusal(body)
    assert refusal.fields == ("normalized_value",)
    assert "path=normalized_value" in refusal.details
    assert "expected=absent" in refusal.details


def test_an_unstorable_witness_is_refused_with_its_location():
    body = _rich_answer()
    body["witnesses"][0]["value"] = ""
    refusal = _refusal(body)
    assert refusal.fields == ("value",)
    assert "slot=witnesses" in refusal.details
    assert "index=0" in refusal.details
    assert "category=size_out_of_range" in refusal.details
    assert "size=0" in refusal.details


def test_a_citation_outside_the_branch_is_never_correctable():
    body = _rich_answer()
    body["findings"][2]["evidence_refs"] = ["ref-" + "0" * 64]
    refusal = _refusal(body)
    assert refusal.code == "citation_outside_branch"
    assert refusal.correctable is False


# ------------------------------------------------------------------ run lifecycle


def _open_run(environment, runner, *, deadline_seconds: float = 30.0):
    """Create a Run through the product's own create_run, with the runner's snapshot."""

    import secrets
    from datetime import UTC, datetime

    token = _token(environment)
    context = environment.authorization.resolve(
        token, request_id="g-lifecycle", operation=SecurityOperation.REVIEW_CREATE
    )
    registration = environment.authorization.require_case(
        context,
        environment.case.case_id,
        request_id="g-lifecycle",
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
        idempotency_key=f"g-lifecycle-{secrets.token_hex(4)}",
        request_digest=digest,
        owner_id="owner-" + secrets.token_hex(16),
        now=datetime.now(UTC),
        lease_seconds=600.0,
        deadline_seconds=deadline_seconds,
        events=(),
    )
    run = environment.repository.load_run(run_id)
    scope, registration = _scope_and_registration(environment, generation=1)
    return run_id, run, scope, registration


def _execute(runner, environment, run_id, run, scope, registration):
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
            now=lambda: __import__("datetime").datetime.now(__import__("datetime").UTC),
        )
    )


def _answer_handler(calls: list, fail_on: int = 0, mutate=None):
    """One Mock transport that answers the v3 wire from its own material.

    ``mutate(worker, body, call_index)`` may change the answer before it is sent, so a
    test can model one specific delivery error without touching the product path.
    """

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        payload = json.loads(request.content)
        user = json.loads(payload["messages"][1]["content"])
        worker = user["worker"]
        source = user["sources"][0]
        if fail_on and len(calls) == fail_on:
            raise RuntimeError("the execution was interrupted mid-send")
        body = (
            _policy_body(source["content"])
            if worker == "policy"
            else _contract_body(source["content"])
        )
        if mutate is not None:
            mutate(worker, body, len(calls))
        for finding in body["findings"]:
            finding["evidence_refs"] = [source["ref"]]
        for witness in body["witnesses"]:
            witness["source_ref"] = source["ref"]
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-flash-ga-260731",
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

    return respond


def _runner(
    environment,
    calls: list,
    *,
    mode="test_only",
    fail_on: int = 0,
    mutate=None,
    **kwargs,
):
    from procurement_review.eval_f_registration import configuration
    from procurement_review.precise_run_runner import PreciseScopedRunner
    from procurement_review.precise_run_transport import PreciseRunChatTransport

    transport = PreciseRunChatTransport(
        configuration(),
        SecretStr("offline-only"),
        catalog=None,
        transport=httpx.MockTransport(
            _answer_handler(calls, fail_on=fail_on, mutate=mutate)
        ),
    )
    return PreciseScopedRunner(
        store=environment.store, transport=transport, mode=mode, **kwargs
    )


@pytest.fixture()
def environment():
    yield from _environment()


def test_a_committed_branch_is_reused_and_the_metering_covers_both_generations(
    environment,
):
    """The interruption case: policy committed, contract interrupted, resume continues."""

    calls: list = []
    runner = _runner(environment, calls, fail_on=2)
    run_id, run, scope, registration = _open_run(environment, runner)
    with pytest.raises(RuntimeError):
        _execute(runner, environment, run_id, run, scope, registration)
    assert environment.repository.attempt_totals(run_id) == {"policy": 1, "contract": 1}

    resumed = _runner(environment, calls)
    execution = _execute(resumed, environment, run_id, run, scope, registration)
    # The policy branch was reused: only the contract branch reached the endpoint again.
    assert len(calls) == 3
    assert execution.policy.outcome == "completed"
    assert execution.contract.outcome == "completed"
    assert execution.policy.attempts[0].attempt_no == 1
    assert execution.contract.attempts[-1].attempt_no == 2
    metering = execution.metering
    assert metering.physical_attempts == 3
    # Only a receipt proves a request was sent and answered: the interrupted attempt of the
    # contract branch left no receipt, so it is reported as an unresolved send instead of
    # being counted as a first send that provably happened.
    assert metering.first_sends == 1
    assert metering.proven_sends == 2
    # The interrupted attempt was never closed by any path, so it is an open row whose
    # send state is undecidable: it is reported as unreturned, never as a cheap attempt.
    assert metering.open_attempts == 1
    assert metering.unresolved_attempts == 0
    assert metering.corrections == 1
    assert metering.other_extra_attempts == 1
    assert metering.reused_branches == 1
    assert metering.unreturned_attempts == 1
    # The interrupted attempt never wrote a receipt: its usage is unknown, so the known
    # total is the lower bound of the two settled attempts.
    assert metering.input_tokens_known == 1400
    assert metering.output_tokens_known is not None
    assert metering.usage_complete is False
    assert metering.unknown_usage_attempts == 1
    # Every attempt's receipt is durable, including the interrupted one.
    from procurement_review.precise_run_state import attempt_receipts

    receipts = attempt_receipts(environment.repository, run_id)
    assert len(receipts) == 2
    assert {item["worker"] for item in receipts} == {"policy", "contract"}


def test_a_drifted_runtime_stops_before_anything_is_sent(environment):
    calls: list = []
    runner = _runner(environment, calls)
    run_id, run, scope, registration = _open_run(environment, runner)
    drifted = _runner(environment, calls)
    drifted._transport.config = drifted._transport.config.model_copy(
        update={"temperature": 0.1}
    )
    with pytest.raises(IncompatibleCheckpointError) as caught:
        _execute(drifted, environment, run_id, run, scope, registration)
    assert "sampling_parameters" in str(caught.value)
    assert calls == []
    assert environment.repository.attempt_totals(run_id) == {}


def test_an_identity_drift_stops_the_remaining_branch_with_its_own_code(environment):
    calls: list = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "model": "some-other-model",
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "{}"},
                    }
                ],
            },
        )

    from procurement_review.eval_f_registration import configuration
    from procurement_review.precise_run_runner import PreciseScopedRunner
    from procurement_review.precise_run_transport import PreciseRunChatTransport

    transport = PreciseRunChatTransport(
        configuration(),
        SecretStr("offline-only"),
        catalog=None,
        transport=httpx.MockTransport(respond),
    )
    runner = PreciseScopedRunner(store=environment.store, transport=transport)
    run_id, run, scope, registration = _open_run(environment, runner)
    execution = _execute(runner, environment, run_id, run, scope, registration)
    assert len(calls) == 1
    assert execution.policy.outcome == "failed"
    assert execution.policy.failure_code == "model_identity_drift"
    assert execution.stop_reason == "model_identity_drift"
    assert execution.contract.outcome == "stopped"
    assert execution.contract.attempts == ()
    assert execution.metering.physical_attempts == 1
    assert "transport" in execution.summary.failure_reasons


def test_a_branch_that_never_answered_ends_in_a_controlled_state(environment):
    calls: list = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        payload = json.loads(request.content)
        user = json.loads(payload["messages"][1]["content"])
        if user["worker"] == "policy":
            return httpx.Response(
                200,
                json={
                    "model": "deepseek-v4-flash-ga-260731",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10},
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "not json"}}
                    ],
                },
            )
        source = user["sources"][0]
        body = _contract_body(source["content"])
        for finding in body["findings"]:
            finding["evidence_refs"] = [source["ref"]]
        for witness in body["witnesses"]:
            witness["source_ref"] = source["ref"]
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-flash-ga-260731",
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

    from procurement_review.eval_f_registration import configuration
    from procurement_review.precise_run_runner import PreciseScopedRunner
    from procurement_review.precise_run_transport import PreciseRunChatTransport

    transport = PreciseRunChatTransport(
        configuration(),
        SecretStr("offline-only"),
        catalog=None,
        transport=httpx.MockTransport(respond),
    )
    runner = PreciseScopedRunner(store=environment.store, transport=transport)
    run_id, run, scope, registration = _open_run(environment, runner)
    execution = _execute(runner, environment, run_id, run, scope, registration)
    assert execution.policy.outcome == "failed"
    assert execution.policy.failure_code == "invalid_json"
    assert execution.policy.reported == ()
    assert execution.contract.outcome == "completed"
    assert execution.stop_reason is None
    assert execution.metering.physical_attempts == 3
    assert execution.metering.corrections == 1
    assert "evidence_envelope_invalid" in execution.summary.failure_reasons


# ------------------------------------------- a citation the source boundary refused


def _bad_special(worker: str, body: dict, call_index: int) -> None:
    """Make the O-02 `special` field cite a value its own clause does not carry."""

    if worker != "policy" or call_index != 1:
        return
    paraphrase = "专项条款按登记口径执行"
    for finding in body["findings"]:
        if finding["obligation_id"] == "O-02":
            finding["normalized_value"]["special"] = paraphrase
    for witness in body["witnesses"]:
        if witness["obligation_id"] == "O-02" and witness["field"] == "special":
            witness["value"] = paraphrase


def test_a_citation_the_source_refuses_is_corrected_once_and_recovered(environment):
    """One unsupported citation is a correctable delivery error, not a lost branch."""

    calls: list = []
    runner = _runner(environment, calls, mutate=_bad_special)
    run_id, run, scope, registration = _open_run(environment, runner)
    execution = _execute(runner, environment, run_id, run, scope, registration)
    # The first answer was received and refused by the source boundary; the second
    # attempt carried one bounded correction and passed.
    policy_calls = [
        json.loads(json.loads(request.content)["messages"][1]["content"])
        for request in calls
        if json.loads(json.loads(request.content)["messages"][1]["content"])["worker"]
        == "policy"
    ]
    sent = [
        json.loads(request.content)["messages"]
        for request in calls
        if json.loads(json.loads(request.content)["messages"][1]["content"])["worker"]
        == "policy"
    ]
    assert len(sent) == 2
    correction = sent[1][-1]["content"]
    assert "不被其来源支持" in correction
    assert "special" in correction
    assert "category=" in correction
    # The refused value never reaches the record as if it had been supported.
    assert execution.policy.outcome == "completed"
    assert "O-02" in {item.obligation_id for item in execution.policy.reported}
    assert execution.policy.support_failures == ()
    assert execution.policy.attempts[0].error_code in {
        "citation_proposition_unsupported",
        "citation_field_value_unsupported",
    }
    assert execution.policy.attempts[0].correctable is True
    assert execution.policy.attempts[1].attempt_no == 2
    assert execution.metering.corrections == 1
    assert execution.metering.physical_attempts == 3
    assert len(policy_calls) == 2


def test_a_branch_whose_every_duty_was_refused_still_fails_closed(environment):
    """Nothing survived the source check, so the branch is a failure, not an empty success."""

    calls: list = []

    def break_every_witness(worker: str, body: dict, call_index: int) -> None:
        for finding in body["findings"]:
            finding["normalized_value"] = {
                key: "not-in-the-source" if isinstance(value, str) else value
                for key, value in finding["normalized_value"].items()
            }
        for witness in body["witnesses"]:
            if isinstance(witness["value"], str):
                witness["value"] = "not-in-the-source"

    runner = _runner(environment, calls, mutate=break_every_witness)
    run_id, run, scope, registration = _open_run(environment, runner)
    execution = _execute(runner, environment, run_id, run, scope, registration)
    assert execution.policy.outcome == "failed"
    assert execution.policy.support_failures
    assert "document_model_failed" in execution.summary.failure_reasons
    # Nothing of the refused branch reaches the decision: its duties are unresolved.
    duties = {row.obligation_id: row.state for row in execution.decision.findings}
    assert {
        duty for duty in ("O-01", "O-02", "O-04") if duties[duty] != "unknown"
    } == set()


def test_the_run_attributes_a_correction_to_its_own_branch(environment):
    """A citation correction in one branch is not charged to the other branch's send.

    The two document branches number their attempts independently, so the Run's metering
    must pair each attempt with the predecessor of its own branch.  Before that fix, the
    whole attempt table was sorted by attempt number and adjacent rows were paired across
    branches, which read this Run's one citation correction as an unlabelled extra attempt
    and reported ``citation_corrections`` as zero.
    """

    calls: list = []
    runner = _runner(environment, calls, mutate=_bad_special)
    run_id, run, scope, registration = _open_run(environment, runner)
    execution = _execute(runner, environment, run_id, run, scope, registration)
    metering = execution.metering
    # policy: two attempts (one citation refusal then a corrected answer); contract: one.
    assert metering.physical_attempts == 3
    assert metering.proven_sends == 3
    assert metering.first_sends == 2
    assert metering.citation_corrections == 1
    assert metering.contract_corrections == 0
    assert metering.transport_retries == 0
    assert metering.other_extra_attempts == 0
    assert metering.corrections == 1
    workers = [
        json.loads(json.loads(request.content)["messages"][1]["content"])["worker"]
        for request in calls
    ]
    assert (
        workers == ["policy", "policy", "contract"]
        or workers == ["policy", "contract", "policy"]
        or sorted(workers) == ["contract", "policy", "policy"]
    )
    # The same reading comes back from the persisted rows, independently of row order.
    from procurement_review.precise_run_state import (
        attempt_ledger_rows,
        attempt_receipts,
        metering_from_ledger,
    )

    rows = list(attempt_ledger_rows(environment.repository, run_id))
    receipts = list(attempt_receipts(environment.repository, run_id))
    recomputed = metering_from_ledger(
        list(reversed(rows)),
        receipts=receipts,
        requested_model=metering.requested_model,
        resolved_models=metering.resolved_models,
    )
    assert recomputed.citation_corrections == 1
    assert recomputed.other_extra_attempts == 0
    assert recomputed.contract_corrections == 0
    assert recomputed.proven_sends == 3
