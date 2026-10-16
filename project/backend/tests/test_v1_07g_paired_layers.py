"""The paired diagnostic's validation layers, correction budget and diagnostics.

The diagnostic exists to say *where* a branch lost its answer, so each case below is a
negative control for one way that could be misread: a refused citation must not be
recorded as a first-pass success, one shared correction must be able to rescue a citation
refusal without rewriting the first attempt, a witness outside a duty's own dimensions
must cost that duty and nothing else, and a refused answer must name the rule it broke.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from pydantic import SecretStr

from procurement_review.business_v31_documents import document_request_v31
from procurement_review.business_v31_review import ground_document_v31_located
from procurement_review.precise_answer_v3 import parse_precise_answer
from procurement_review.business_v3_security import ProtectedOutputV3
from procurement_review.business_v3_support import SourceCatalogV3
from procurement_review.eval_attempt_ledger import AttemptLedger
from procurement_review.eval_f_context import context_for, located_sources, stable_write
from procurement_review.eval_f_contracts import FBudgetProof
from procurement_review.eval_f_registration import configuration
from procurement_review.eval_f_run import FAuthority
from procurement_review.gold_v37_conditional import as_gold_case, conditional_cases
from procurement_review.model_attempt_v2 import configuration_hash
from procurement_review.model_wire_v2 import WireIssue
from procurement_review.precise_dev_paired_run import PairedDriver, PairedProjectionV2
from procurement_review.precise_run_transport import PreciseRunChatTransport
from test_v1_07g_precise_run import _contract_body, _policy_body

CASE = conditional_cases()[0]


def _request(case, worker):
    gold = as_gold_case(case)
    located = located_sources(gold)
    context = context_for(gold)
    sources = tuple(item.source for item in located)
    request = document_request_v31(
        context,
        case_id=case.case_id,
        worker=worker,
        question=case.question,
        sources=tuple(item for item in sources if item.worker == worker),
    )
    return request, SourceCatalogV3(located)


def _filled(body: dict, ref: str) -> dict:
    """One authored answer bound to the single authorized source of this branch."""

    for finding in body["findings"]:
        finding["evidence_refs"] = [ref]
    for witness in body["witnesses"]:
        witness["source_ref"] = ref
    return body


def _answer(content: str, worker: str, ref: str) -> dict:
    builder = _policy_body if worker == "policy" else _contract_body
    return _filled(builder(content), ref)


def _reply(body: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": configuration().resolved_model,
            "usage": {"prompt_tokens": 900, "completion_tokens": 300},
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


def _driver(tmp_path: Path, handler):
    root = tmp_path / ".local" / "paired-layers"
    root.mkdir(parents=True, exist_ok=True)
    execution = root / "execution.json"
    if not execution.exists():
        stable_write(execution, dict(generation=1, cancelled=False))
    proof = FBudgetProof(
        evidence_sha256="c" * 64,
        cumulative_request_upper=2796,
        cumulative_input_upper=None,
        cumulative_output_upper=None,
        historical_unknown_input_reserve=None,
        historical_unknown_output_reserve=None,
        allocated_requests=8,
        allocated_input=None,
        allocated_output=None,
        limit_policy="user_no_balance_limit",
        authorization_basis="user-explicit-no-balance-limit",
    )
    authority = FAuthority(
        authorization_id="test-paired-layers",
        scope_ids=frozenset([context_for(as_gold_case(CASE)).scope_id]),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        proof=proof,
        execution_path=execution,
        generation=1,
    )
    transport = PreciseRunChatTransport(
        configuration(),
        SecretStr("offline-only"),
        catalog=None,
        transport=httpx.MockTransport(handler),
    )
    ledger_path = root / "ledger.sqlite3"
    if ledger_path.exists():
        ledger_path.unlink()
    ledger = AttemptLedger.create(
        ledger_path,
        run_id="v1-07-g-paired-layers",
        gold_manifest_sha256="d" * 64,
        configuration_sha256=configuration_hash(transport, authority),
        max_requests=8,
        max_input_tokens=None,
        max_output_tokens=None,
        stage_id="V1-07G",
        gold_set_version="gold-set.v3.1",
    )
    return PairedDriver(ledger, transport, authority)


def _run_contract(handler, tmp_path: Path) -> PairedProjectionV2:
    driver = _driver(tmp_path, handler)
    request, catalog = _request(CASE, "contract")
    task_key = f"{CASE.case_id}.own.oracle.contract"
    driver.register(task_key, request, catalog)
    driver.note_rank(task_key, 0)
    return asyncio.run(
        driver.run(
            task_key,
            request,
            deadline=datetime.now(UTC) + timedelta(seconds=110),
            reserve_input=1000,
            reserve_output=4000,
            max_attempts=2,
        )
    )


def test_a_refused_citation_is_never_a_first_pass_success(tmp_path: Path):
    """The regression that made a grounding failure look like a clean first attempt."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        user = json.loads(payload["messages"][1]["content"])
        source = user["sources"][0]
        body = _answer(source["content"], user["worker"], source["ref"])
        # The warranty value the source does not carry: that field cannot be supported, and
        # the branch has no attempt left after the correction it earns here.
        for witness in body["witnesses"]:
            if witness["field"] == "warranty":
                witness["value"] = 900
        return _reply(body)

    projection = _run_contract(handler, tmp_path)
    assert isinstance(projection, PairedProjectionV2)
    assert projection.citations_valid is False
    assert projection.first_pass_valid is False
    assert projection.corrected_valid is False
    assert projection.qualified is False
    assert projection.committed is False
    assert projection.pipeline_failure is True
    assert projection.model_request_count == 2
    assert "citation_field_value_unsupported" in projection.grounding_failures
    assert any(
        token.startswith("category=") for token in projection.grounding_locations
    )
    assert any(token == "obligation=O-08" for token in projection.grounding_locations)


def test_one_shared_correction_rescues_a_citation_refusal_without_rewriting_attempt_one(
    tmp_path: Path,
):
    """A corrected answer proves the correction worked; it never proves the first one did."""

    prompts: list[str] = []
    seen = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        user = json.loads(payload["messages"][1]["content"])
        source = user["sources"][0]
        prompts.append(str(payload["messages"][-1]["content"]))
        body = _answer(source["content"], user["worker"], source["ref"])
        if seen["count"] == 0:
            for witness in body["witnesses"]:
                if witness["field"] == "warranty":
                    witness["value"] = 900
        seen["count"] += 1
        return _reply(body)

    projection = _run_contract(handler, tmp_path)
    assert projection.model_request_count == 2
    assert projection.first_pass_valid is False
    assert projection.corrected_valid is True
    assert projection.qualified is True
    assert projection.committed is True
    assert projection.pipeline_failure is False
    assert projection.correction_kinds == ("citation",)
    # The one correction names the refused field and its fixed category, never a value.
    assert "citation_field_value_unsupported" in prompts[-1]
    assert "field=warranty" in prompts[-1]
    assert "900" not in prompts[-1]


def test_a_witness_outside_a_duty_costs_that_duty_and_not_the_whole_document():
    """The rules 3.1.0 grounding of the branch whose rule object carried a stray field."""

    request, catalog = _request(CASE, "policy")
    source = request.sources[0]
    body = _answer(source.content, "policy", source.ref)
    # A dimension the policy branch's O-01 duty does not declare.  The frozen behaviour
    # discarded every duty of the document, which is the loss this line must not repeat.
    body["witnesses"].append(
        {
            "obligation_id": "O-01",
            "field": "quote_threshold",
            "value": 3,
            "source_ref": source.ref,
            "start": 0,
            "end": 1,
        }
    )
    document, witnesses = parse_precise_answer(
        json.dumps(
            {"schema_version": "finding-evidence.v3", **body}, ensure_ascii=False
        ),
        request,
        protection=ProtectedOutputV3(),
    )
    grounded, locations = ground_document_v31_located(
        request, document, witnesses, catalog
    )
    assert "O-01" not in grounded.document.values
    assert {"O-02", "O-04"} <= set(grounded.document.values)
    assert "citation_field_outside_task" in grounded.support_failures
    assert "obligation=O-01" in locations
    assert "field=quote_threshold" in locations
    # Other duties of the same branch survive, so the document-level failure is not raised.
    assert "document_citation_invalid" not in grounded.document.failures


def test_a_document_whose_every_duty_was_refused_still_fails_closed():
    """The document-level code keeps its meaning: it is what an unusable document says."""

    request, catalog = _request(CASE, "contract")
    source = request.sources[0]
    body = _answer(source.content, "contract", source.ref)
    for witness in body["witnesses"]:
        witness["value"] = "not-the-source-value"
    document, witnesses = parse_precise_answer(
        json.dumps(
            {"schema_version": "finding-evidence.v3", **body}, ensure_ascii=False
        ),
        request,
        protection=ProtectedOutputV3(),
    )
    grounded, locations = ground_document_v31_located(
        request, document, witnesses, catalog
    )
    assert grounded.document.values == {}
    assert "document_citation_invalid" in grounded.document.failures
    assert grounded.support_failures
    assert locations


def test_the_value_rule_refusal_names_the_rule_and_the_field_it_is_about():
    """What the one correction is asked to redo has to be readable from the receipt."""

    request, _catalog = _request(CASE, "policy")
    source = request.sources[0]
    body = _answer(source.content, "policy", source.ref)
    for finding in body["findings"]:
        finding["evidence_refs"] = []
    issue = None
    try:
        parse_precise_answer(
            json.dumps(
                {"schema_version": "finding-evidence.v3", **body}, ensure_ascii=False
            ),
            request,
            protection=ProtectedOutputV3(),
        )
    except WireIssue as error:
        issue = error
    assert issue is not None
    tokens = list(issue.details)
    assert "category=value_rule" in tokens
    assert "expected=rule" in tokens
    assert "rule=decided_value_or_citation" in tokens
    assert "path=evidence_refs" in tokens
    assert "duty=O-01" in tokens
    assert "slot=findings" in tokens
