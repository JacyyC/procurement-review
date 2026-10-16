from __future__ import annotations

import ast
import asyncio
import json
import inspect
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr, ValidationError

import test_retrieval as retrieval_support
from procurement_review import workers as workers_module
from procurement_review.chat_adapter import (
    ChatAdapterConfig,
    OpenAICompatibleChatAdapter,
)
from procurement_review.contracts import (
    EvidenceRef,
    Finding,
    FindingAssertion,
    FindingStatus,
    RiskCategory,
    RiskLevel,
    SourceKind,
    WorkerKind,
)
from procurement_review.model_adapters import DeterministicFakeModelAdapter
from procurement_review.retrieval import RetrievalPort, RetrievalRequest
from procurement_review.model_invocation import OfflineModelInvocation
from procurement_review.result_admission import OFFLINE_RESULT_CONSUMPTION
from procurement_review.worker_contracts import (
    ContractWorkerInput,
    PolicyWorkerInput,
    ProcurementWorkerInput,
    WorkerFailureCode,
    WorkerModelRequest,
    WorkerOutcome,
)
from procurement_review.workers import ContractWorker, PolicyWorker


def fixture_retriever():
    chunks = tuple(
        retrieval_support._chunk(document["source_id"], document["content"])
        for document in retrieval_support.RETRIEVAL_FIXTURE["documents"]
    )
    return retrieval_support._retriever(chunks)


class RecordingRetrieval:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.requests: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest):
        self.requests.append(request)
        return self.delegate.retrieve(request)


class _ReceivedCall:
    """The handle of one recorded call; only its result is asynchronous."""

    __slots__ = ("_model", "_request")

    def __init__(self, model: RecordingModel, request: WorkerModelRequest) -> None:
        self._model = model
        self._request = request

    async def result(self) -> object:
        return await self._model.deliver(self._request)


class RecordingModel:
    """A test model adapter with both a synchronous receive and the coroutine entry."""

    adapter_version = "recording-test-model-v1"
    fixture_version = "worker-model-fixture-v1"
    test_only = True

    def __init__(self, delegate=None, output: object | None = None) -> None:
        self.delegate = delegate or DeterministicFakeModelAdapter()
        self.output = output
        self.requests: list[WorkerModelRequest] = []

    def start(self, request: WorkerModelRequest) -> _ReceivedCall:
        """Record the synchronous receipt, exactly as a real adapter with one would."""

        self.requests.append(request)
        return _ReceivedCall(self, request)

    async def generate(self, request: WorkerModelRequest) -> object:
        return await self.start(request).result()

    async def deliver(self, request: WorkerModelRequest) -> object:
        if self.output is not None:
            return self.output
        return await self.delegate.generate(request)


class RaisingRetrieval:
    def retrieve(self, request: RetrievalRequest):
        raise RuntimeError("injected test failure")


class NonTestModel(RecordingModel):
    test_only = False


def policy_input(**updates) -> PolicyWorkerInput:
    values = {
        "case_id": "CASE-001",
        "query": "competitive bidding threshold",
        "index_version": retrieval_support.INDEX_VERSION,
        "model_fixture_id": "policy-low",
    }
    values.update(updates)
    return PolicyWorkerInput(**values)


def contract_input(**updates) -> ContractWorkerInput:
    values = {
        "case_id": "CASE-001",
        "query": "payment terms termination rights",
        "index_version": retrieval_support.INDEX_VERSION,
        "model_fixture_id": "contract-low",
    }
    values.update(updates)
    return ContractWorkerInput(**values)


def test_worker_inputs_are_versioned_minimal_and_forbid_private_context() -> None:
    assert set(PolicyWorkerInput.model_fields) == {
        "schema_version",
        "input_version",
        "case_id",
        "query",
        "index_version",
        "model_fixture_id",
    }
    assert set(ContractWorkerInput.model_fields) == set(PolicyWorkerInput.model_fields)
    assert set(ProcurementWorkerInput.model_fields) == {
        "schema_version",
        "input_version",
        "case_id",
        "purchase_order_id",
        "dataset_version",
        "model_fixture_id",
    }
    assert policy_input().input_version == "worker-input.v1"
    assert contract_input().input_version == "worker-input.v1"

    with pytest.raises(ValidationError):
        PolicyWorkerInput(
            case_id="CASE-001",
            query="policy",
            index_version="retrieval-index-v1",
            model_fixture_id="policy-low",
            procurement_context={"purchase_order_id": "PO-001"},
        )
    with pytest.raises(ValidationError):
        ProcurementWorkerInput(
            case_id="CASE-001",
            purchase_order_id="PO-001",
            dataset_version="synthetic-procurement.v1",
            model_fixture_id="procurement-low",
            retrieved_contract="private context",
        )


def test_policy_and_contract_use_only_retrieval_and_isolate_model_context() -> None:
    async def exercise() -> tuple:
        policy_retrieval = RecordingRetrieval(fixture_retriever())
        contract_retrieval = RecordingRetrieval(fixture_retriever())
        policy_model = RecordingModel()
        contract_model = RecordingModel()
        policy = PolicyWorker(
            retrieval=policy_retrieval,
            model=policy_model,
            consumption=OFFLINE_RESULT_CONSUMPTION,
            invocation=OfflineModelInvocation(receiver=policy_model),
        )
        contract = ContractWorker(
            retrieval=contract_retrieval,
            model=contract_model,
            consumption=OFFLINE_RESULT_CONSUMPTION,
            invocation=OfflineModelInvocation(receiver=contract_model),
        )
        results = await asyncio.gather(
            policy.review(policy_input()),
            contract.review(contract_input()),
        )
        return (
            *results,
            policy_retrieval,
            contract_retrieval,
            policy_model,
            contract_model,
        )

    (
        policy_result,
        contract_result,
        policy_retrieval,
        contract_retrieval,
        policy_model,
        contract_model,
    ) = asyncio.run(exercise())

    assert isinstance(policy_retrieval, RetrievalPort)
    assert isinstance(contract_retrieval, RetrievalPort)
    assert policy_retrieval.requests == [
        RetrievalRequest(
            query=policy_input().query,
            index_version=retrieval_support.INDEX_VERSION,
            limit=10,
            minimum_results=1,
        )
    ]
    assert contract_retrieval.requests[0].query == contract_input().query
    assert policy_result.outcome is WorkerOutcome.COMPLETED
    assert contract_result.outcome is WorkerOutcome.COMPLETED
    assert policy_result.findings[0].worker is WorkerKind.POLICY
    assert contract_result.findings[0].worker is WorkerKind.CONTRACT
    assert policy_result.findings[0].evidence_refs[0].source_id == (
        "fixture-competition"
    )
    assert contract_result.findings[0].evidence_refs[0].source_id == (
        "fixture-contract"
    )

    policy_context = policy_model.requests[0]
    contract_context = contract_model.requests[0]
    assert policy_context.worker is WorkerKind.POLICY
    assert contract_context.worker is WorkerKind.CONTRACT
    assert policy_context.business_facts == ()
    assert contract_context.business_facts == ()
    assert {
        item.evidence_ref.source_id for item in policy_context.evidence_contexts
    } == {
        "fixture-competition",
        "fixture-sanctions",
    }
    assert {
        item.evidence_ref.source_id for item in contract_context.evidence_contexts
    } == {"fixture-contract"}


@pytest.mark.parametrize(
    ("worker_request", "expected_code"),
    (
        (
            policy_input(query="unrepresented phrase"),
            WorkerFailureCode.NO_EVIDENCE,
        ),
        (
            policy_input(index_version="retrieval-index-v999"),
            WorkerFailureCode.VERSION_MISMATCH,
        ),
    ),
)
def test_document_worker_fails_closed_without_evidence_or_on_version_mismatch(
    worker_request: PolicyWorkerInput,
    expected_code: WorkerFailureCode,
) -> None:
    model = RecordingModel()
    worker = PolicyWorker(
        retrieval=fixture_retriever(),
        model=model,
        consumption=OFFLINE_RESULT_CONSUMPTION,
        invocation=OfflineModelInvocation(receiver=model),
    )

    result = asyncio.run(worker.review(worker_request))

    assert result.outcome is WorkerOutcome.FAILED
    assert result.failure_code is expected_code
    assert result.findings[0].status is not FindingStatus.SUPPORTED
    assert result.findings[0].evidence_refs == ()
    assert model.requests == []


def test_document_worker_fails_closed_on_port_failure_and_invalid_fake_output() -> None:
    port_failure_model = RecordingModel()
    port_failure = asyncio.run(
        PolicyWorker(
            retrieval=RaisingRetrieval(),
            model=port_failure_model,
            consumption=OFFLINE_RESULT_CONSUMPTION,
            invocation=OfflineModelInvocation(receiver=port_failure_model),
        ).review(policy_input())
    )
    invalid_output_model = RecordingModel(output={"not": "a Finding tuple"})
    invalid_output = asyncio.run(
        PolicyWorker(
            retrieval=fixture_retriever(),
            model=invalid_output_model,
            consumption=OFFLINE_RESULT_CONSUMPTION,
            invocation=OfflineModelInvocation(receiver=invalid_output_model),
        ).review(policy_input())
    )
    out_of_context_model = RecordingModel()
    out_of_context = asyncio.run(
        PolicyWorker(
            retrieval=fixture_retriever(),
            model=out_of_context_model,
            consumption=OFFLINE_RESULT_CONSUMPTION,
            invocation=OfflineModelInvocation(receiver=out_of_context_model),
        ).review(policy_input(query="sanctions supplier"))
    )

    assert port_failure.failure_code is WorkerFailureCode.PORT_FAILURE
    assert invalid_output.failure_code is WorkerFailureCode.INVALID_MODEL_OUTPUT
    assert out_of_context.failure_code is WorkerFailureCode.INVALID_MODEL_OUTPUT
    assert all(
        result.findings[0].status is not FindingStatus.SUPPORTED
        for result in (port_failure, invalid_output, out_of_context)
    )


def _real_chat_config(**updates) -> ChatAdapterConfig:
    values = {
        "base_url": "https://model.example",
        "model_name": "requested-model",
        "api_key": SecretStr("unit-test-key-never-real"),
        "timeout_seconds": 0.01,
        "max_retries": 0,
    }
    values.update(updates)
    return ChatAdapterConfig(**values)


def test_versioned_real_chat_adapter_runs_through_policy_worker() -> None:
    captured: list[dict] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        captured.append(body)
        context = json.loads(body["messages"][1]["content"])
        reference = EvidenceRef(
            source_kind=SourceKind.CASE_DOCUMENT,
            source_id=context["evidence_contexts"][0]["ref_id"],
            version=context["evidence_contexts"][0]["version"],
            section=context["evidence_contexts"][0].get("section"),
            content_checksum="sha256:" + "1" * 64,
        )
        finding = Finding(
            worker=WorkerKind.POLICY,
            finding_key="policy.required_policy_evidence",
            category=RiskCategory.POLICY_COMPLIANCE,
            risk_level=RiskLevel.LOW,
            assertion=FindingAssertion.RISK_ABSENT,
            statement="The synthetic policy context supports the required finding.",
            status=FindingStatus.SUPPORTED,
            evidence_refs=(reference,),
            recommended_action="Continue with deterministic review.",
        )
        finding_payload = finding.model_dump(mode="json")
        finding_payload["evidence_refs"] = [
            reference.source_id for reference in finding.evidence_refs
        ]
        finding_payload["business_fact_refs"] = [
            fact.record_id for fact in finding.business_fact_refs
        ]
        content = json.dumps({"findings": [finding_payload]}, ensure_ascii=False)
        return httpx.Response(
            200,
            json={
                "id": "worker-integration-request",
                "model": "resolved-model-v1",
                "choices": [{"message": {"content": content}}],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 17,
                    "total_tokens": 28,
                },
            },
        )

    adapter = OpenAICompatibleChatAdapter(
        _real_chat_config(),
        transport=httpx.MockTransport(handler),
    )

    async def exercise():
        try:
            return await PolicyWorker(
                retrieval=fixture_retriever(),
                model=adapter,
                consumption=OFFLINE_RESULT_CONSUMPTION,
                invocation=OfflineModelInvocation(receiver=adapter),
            ).review(policy_input())
        finally:
            await adapter.aclose()

    result = asyncio.run(exercise())

    assert result.outcome is WorkerOutcome.COMPLETED
    assert result.findings[0].finding_key == "policy.required_policy_evidence"
    assert len(captured) == 1
    assert "tools" not in captured[0]
    assert "gold_label" not in json.dumps(captured[0], ensure_ascii=False)
    assert adapter.records[0].requested_model_id == "requested-model"
    assert adapter.records[0].resolved_model_id == "resolved-model-v1"


def test_real_chat_adapter_failures_reach_worker_as_unresolved_and_retrieval_blocks_call() -> (
    None
):
    def invalid_handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "invalid-worker-response",
                "model": "resolved-model-v1",
                "choices": [{"message": {"content": "not-json"}}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    adapter = OpenAICompatibleChatAdapter(
        _real_chat_config(),
        transport=httpx.MockTransport(invalid_handler),
    )

    async def exercise_invalid():
        try:
            return await PolicyWorker(
                retrieval=fixture_retriever(),
                model=adapter,
                consumption=OFFLINE_RESULT_CONSUMPTION,
                invocation=OfflineModelInvocation(receiver=adapter),
            ).review(policy_input())
        finally:
            await adapter.aclose()

    invalid_result = asyncio.run(exercise_invalid())
    assert invalid_result.outcome is WorkerOutcome.FAILED
    assert invalid_result.failure_code is WorkerFailureCode.INVALID_MODEL_OUTPUT
    assert invalid_result.findings[0].status is not FindingStatus.SUPPORTED
    assert adapter.records[-1].error_code == "invalid_json"

    calls = 0

    def blocked_handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    blocked_adapter = OpenAICompatibleChatAdapter(
        _real_chat_config(),
        transport=httpx.MockTransport(blocked_handler),
    )

    async def exercise_blocked():
        try:
            return await PolicyWorker(
                retrieval=RaisingRetrieval(),
                model=blocked_adapter,
                consumption=OFFLINE_RESULT_CONSUMPTION,
                invocation=OfflineModelInvocation(receiver=adapter),
            ).review(policy_input())
        finally:
            await blocked_adapter.aclose()

    blocked_result = asyncio.run(exercise_blocked())
    assert blocked_result.failure_code is WorkerFailureCode.PORT_FAILURE
    assert calls == 0
    assert blocked_adapter.records == ()


def test_workers_reject_any_model_not_explicitly_marked_test_only() -> None:
    policy_model = NonTestModel()
    with pytest.raises(ValueError, match="test_only"):
        PolicyWorker(
            retrieval=fixture_retriever(),
            model=policy_model,
            consumption=OFFLINE_RESULT_CONSUMPTION,
            invocation=OfflineModelInvocation(receiver=policy_model),
        )
    contract_model = NonTestModel()
    with pytest.raises(ValueError, match="test_only"):
        ContractWorker(
            retrieval=fixture_retriever(),
            model=contract_model,
            consumption=OFFLINE_RESULT_CONSUMPTION,
            invocation=OfflineModelInvocation(receiver=contract_model),
        )


def test_worker_module_has_no_procurement_store_database_or_sql_import() -> None:
    source = Path(inspect.getfile(workers_module)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    forbidden_modules = {
        "procurement_review.procurement_store",
        ".procurement_store",
        "sqlalchemy",
        "psycopg",
    }
    assert imported_modules.isdisjoint(forbidden_modules)
    forbidden_names = {
        "PostgresProcurementStore",
        "ProcurementReader",
        "create_engine",
    }
    assert not {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }.intersection(forbidden_names)

    policy_parameters = set(inspect.signature(PolicyWorker).parameters)
    contract_parameters = set(inspect.signature(ContractWorker).parameters)
    procurement_parameters = set(
        inspect.signature(workers_module.ProcurementAnomalyWorker).parameters
    )
    # The construction surface is closed and every capability is mandatory: a Worker
    # without a result-consumption policy or without a model-call boundary is not
    # constructible, so neither can be dropped by accident.
    assert policy_parameters == {"retrieval", "model", "consumption", "invocation"}
    assert contract_parameters == {"retrieval", "model", "consumption", "invocation"}
    assert procurement_parameters == {"mcp", "model", "consumption", "invocation"}
