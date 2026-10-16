from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone

import httpx
import pytest
from pydantic import SecretStr

from procurement_review.chat_adapter import (
    EMBEDDING_STATUS,
    RERANK_STATUS,
    ChatAdapterConfig,
    ChatAdapterError,
    ChatCallRecord,
    ChatErrorCode,
    OpenAICompatibleChatAdapter,
)
from procurement_review.contracts import (
    BusinessFact,
    EvidenceRef,
    FactStatus,
    Finding,
    FindingAssertion,
    FindingStatus,
    RiskCategory,
    RiskLevel,
    SourceKind,
    WorkerKind,
)
from procurement_review.worker_contracts import (
    EvidenceContext,
    SingleAgentModelRequest,
    WorkerModelRequest,
)


TEST_KEY = "unit-test-key-never-real"


def config(**updates) -> ChatAdapterConfig:
    values = {
        "base_url": "https://model.example/api",
        "model_name": "requested-model",
        "api_key": SecretStr(TEST_KEY),
        "temperature": 0.0,
        "top_p": 1.0,
        "timeout_seconds": 0.01,
        "max_retries": 1,
    }
    values.update(updates)
    return ChatAdapterConfig(**values)


def policy_request(
    *,
    content: str = "Synthetic policy context.",
    source_id: str = "DOC-SYNTHETIC-POLICY-001",
    question: str | None = None,
) -> WorkerModelRequest:
    reference = EvidenceRef(
        source_kind=SourceKind.CASE_DOCUMENT,
        source_id=source_id,
        version="synthetic-gold-document.v1",
        section="synthetic-policy",
        content_checksum="sha256:" + "1" * 64,
    )
    return WorkerModelRequest(
        worker=WorkerKind.POLICY,
        case_id="GOLD-0001",
        model_fixture_id="unused-for-real-adapter",
        question=question,
        evidence_contexts=(EvidenceContext(evidence_ref=reference, content=content),),
    )


def procurement_request() -> WorkerModelRequest:
    fact = BusinessFact(
        tool_name="procurement.get_supplier",
        record_id="SYN-GOLD-SUP-0001",
        dataset_version="synthetic-gold-procurement.v1",
        as_of=datetime(2026, 6, 30, tzinfo=timezone.utc),
        status=FactStatus.KNOWN,
        result={"supplier_id": "SYN-GOLD-SUP-0001", "registry_status": "active"},
    )
    return WorkerModelRequest(
        worker=WorkerKind.PROCUREMENT_DATA,
        case_id="GOLD-0001",
        model_fixture_id="unused-for-real-adapter",
        business_facts=(fact,),
    )


def finding_for(request: WorkerModelRequest, *, reference=None) -> Finding:
    if request.worker is WorkerKind.POLICY:
        return Finding(
            worker=WorkerKind.POLICY,
            finding_key="policy.required_policy_evidence",
            category=RiskCategory.POLICY_COMPLIANCE,
            risk_level=RiskLevel.LOW,
            assertion=FindingAssertion.RISK_ABSENT,
            statement="The synthetic policy context supports the required finding.",
            status=FindingStatus.SUPPORTED,
            evidence_refs=(reference or request.evidence_contexts[0].evidence_ref,),
            recommended_action="Continue with deterministic review.",
        )
    return Finding(
        worker=WorkerKind.PROCUREMENT_DATA,
        finding_key="procurement.required_business_facts",
        category=RiskCategory.PROCUREMENT_ANOMALY,
        risk_level=RiskLevel.LOW,
        assertion=FindingAssertion.RISK_ABSENT,
        statement="The synthetic procurement fact supports the required finding.",
        status=FindingStatus.SUPPORTED,
        business_fact_refs=(request.business_facts[0],),
        recommended_action="Continue with deterministic review.",
    )


def response_for(
    request: WorkerModelRequest,
    *,
    finding: Finding | None = None,
    request_id: str = "remote-request-001",
    model: str = "resolved-model-v1",
    usage: dict[str, int] | None = None,
) -> dict:
    selected = finding or finding_for(request)
    finding_payload = selected.model_dump(mode="json")
    finding_payload["evidence_refs"] = [
        reference.source_id for reference in selected.evidence_refs
    ]
    finding_payload["business_fact_refs"] = [
        fact.record_id for fact in selected.business_fact_refs
    ]
    content = json.dumps(
        {"findings": [finding_payload]},
        ensure_ascii=False,
    )
    return {
        "id": request_id,
        "model": model,
        "choices": [{"message": {"content": content}}],
        "usage": usage
        or {"prompt_tokens": 11, "completion_tokens": 17, "total_tokens": 28},
    }


def run(adapter: OpenAICompatibleChatAdapter, request: WorkerModelRequest):
    async def invoke():
        try:
            return await adapter.generate(request)
        finally:
            await adapter.aclose()

    return asyncio.run(invoke())


def test_config_reads_only_named_environment_values_and_masks_key() -> None:
    loaded = ChatAdapterConfig.from_env(
        {
            "MODEL_BASE_URL": "https://model.example/base",
            "MODEL_NAME": "env-model",
            "MODEL_API_KEY": TEST_KEY,
            "MODEL_CHAT_PATH": "/chat/completions",
            "MODEL_TEMPERATURE": "0",
            "MODEL_TOP_P": "1",
            "MODEL_SEED": "20260901",
            "MODEL_MAX_OUTPUT_TOKENS": "1536",
            "MODEL_TIMEOUT_SECONDS": "30",
            "MODEL_MAX_RETRIES": "1",
        }
    )
    assert loaded.endpoint_url == "https://model.example/base/chat/completions"
    assert loaded.model_name == "env-model"
    assert TEST_KEY not in repr(loaded)


def test_success_records_metadata_and_sends_no_tools_or_gold_label() -> None:
    request = policy_request(question="Review this synthetic case.")
    captured: list[tuple[httpx.Request, dict]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        captured.append((http_request, body))
        return httpx.Response(
            200,
            json=response_for(request),
            headers={"x-request-id": "header-request-ignored-by-body"},
        )

    records = []
    adapter = OpenAICompatibleChatAdapter(
        config(),
        transport=httpx.MockTransport(handler),
        sleep=_no_sleep,
        audit_sink=records.append,
    )
    findings = run(adapter, request)

    assert len(findings) == 1
    assert findings[0].worker is WorkerKind.POLICY
    assert len(records) == 1
    record = records[0]
    assert record.requested_model_id == "requested-model"
    assert record.resolved_model_id == "resolved-model-v1"
    assert record.request_id_hash == hashlib.sha256(b"remote-request-001").hexdigest()
    assert (record.input_tokens, record.output_tokens, record.total_tokens) == (
        11,
        17,
        28,
    )
    assert record.http_status == 200
    assert record.error_code is None
    assert record.redaction_status == "redacted"
    sent_request, body = captured[0]
    assert sent_request.headers["authorization"] == f"Bearer {TEST_KEY}"
    assert "tools" not in body
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "expected_" not in json.dumps(body)
    assert "gold_label" not in json.dumps(body)
    assert "checksum" not in json.dumps(body).lower()
    assert "model_fixture_id" not in json.dumps(body)
    assert json.loads(body["messages"][1]["content"])["question"] == (
        "Review this synthetic case."
    )
    assert TEST_KEY not in json.dumps(record.model_dump(mode="json"))


def test_invalid_json_and_schema_fail_closed_without_response_body() -> None:
    request = policy_request()
    bodies = (
        {"findings": "not-a-list"},
        {"findings": [{"worker": "policy"}]},
    )
    expected = (ChatErrorCode.INVALID_JSON, ChatErrorCode.OUTPUT_SCHEMA)
    for body, code in zip(bodies, expected, strict=True):
        content = "not-json" if body["findings"] == "not-a-list" else json.dumps(body)

        def handler(http_request: httpx.Request, content=content) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "remote-request-invalid",
                    "model": "resolved-model-v1",
                    "choices": [{"message": {"content": content}}],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        adapter = OpenAICompatibleChatAdapter(
            config(), transport=httpx.MockTransport(handler), sleep=_no_sleep
        )
        with pytest.raises(ChatAdapterError) as raised:
            run(adapter, request)
        assert raised.value.code is code
        assert "not-json" not in str(raised.value)
        assert TEST_KEY not in str(raised.value)
        assert adapter.records[-1].error_code == code


def test_timeout_429_and_5xx_are_fail_closed_with_at_most_one_retry() -> None:
    request = policy_request()

    def timeout_handler(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private transport detail", request=http_request)

    timeout_adapter = OpenAICompatibleChatAdapter(
        config(), transport=httpx.MockTransport(timeout_handler), sleep=_no_sleep
    )
    with pytest.raises(ChatAdapterError) as timeout_error:
        run(timeout_adapter, request)
    assert timeout_error.value.code is ChatErrorCode.TIMEOUT
    assert timeout_adapter.records[-1].retry_count == 0

    calls = 0

    def rate_limit_then_success(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"detail": "do-not-record"})
        return httpx.Response(200, json=response_for(request))

    retry_adapter = OpenAICompatibleChatAdapter(
        config(),
        transport=httpx.MockTransport(rate_limit_then_success),
        sleep=_no_sleep,
    )
    assert len(run(retry_adapter, request)) == 1
    assert calls == 2
    assert retry_adapter.records[-1].retry_count == 1

    server_calls = 0

    def server_failure(http_request: httpx.Request) -> httpx.Response:
        nonlocal server_calls
        server_calls += 1
        return httpx.Response(503, json={"secret": TEST_KEY})

    server_adapter = OpenAICompatibleChatAdapter(
        config(), transport=httpx.MockTransport(server_failure), sleep=_no_sleep
    )
    with pytest.raises(ChatAdapterError) as server_error:
        run(server_adapter, request)
    assert server_error.value.code is ChatErrorCode.SERVER_ERROR
    assert server_calls == 2
    assert server_adapter.records[-1].retry_count == 1
    assert TEST_KEY not in str(server_error.value)


def test_missing_usage_model_request_id_and_out_of_context_citation_fail_closed() -> (
    None
):
    request = policy_request()
    cases: list[Callable[[dict], None]] = []

    def missing_usage(payload: dict) -> None:
        payload.pop("usage")

    def missing_model(payload: dict) -> None:
        payload.pop("model")

    def missing_id(payload: dict) -> None:
        payload.pop("id")

    cases.extend((missing_usage, missing_model, missing_id))
    expected = (
        ChatErrorCode.MISSING_USAGE,
        ChatErrorCode.MISSING_MODEL,
        ChatErrorCode.MISSING_REQUEST_ID,
    )
    for mutate, code in zip(cases, expected, strict=True):
        payload = response_for(request)
        mutate(payload)

        def handler(http_request: httpx.Request, payload=payload) -> httpx.Response:
            return httpx.Response(200, json=payload)

        adapter = OpenAICompatibleChatAdapter(
            config(), transport=httpx.MockTransport(handler), sleep=_no_sleep
        )
        with pytest.raises(ChatAdapterError) as raised:
            run(adapter, request)
        assert raised.value.code is code

    other_ref = EvidenceRef(
        source_kind=SourceKind.CASE_DOCUMENT,
        source_id="DOC-OUTSIDE-CONTEXT",
        version="synthetic-gold-document.v1",
        section="synthetic-policy",
        content_checksum="sha256:" + "2" * 64,
    )
    payload = response_for(request, finding=finding_for(request, reference=other_ref))

    def outside_handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    outside_adapter = OpenAICompatibleChatAdapter(
        config(), transport=httpx.MockTransport(outside_handler), sleep=_no_sleep
    )
    with pytest.raises(ChatAdapterError) as outside_error:
        run(outside_adapter, request)
    assert outside_error.value.code is ChatErrorCode.OUTPUT_SCHEMA
    assert (
        outside_adapter.records[-1].error_detail == "evidence_reference_out_of_context"
    )


def test_model_tool_call_is_rejected_even_when_message_has_json_content() -> None:
    request = policy_request()
    payload = response_for(request)
    payload["choices"][0]["finish_reason"] = "tool_calls"
    payload["choices"][0]["message"]["tool_calls"] = [
        {"id": "call-1", "type": "function", "function": {"name": "write"}}
    ]

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    adapter = OpenAICompatibleChatAdapter(
        config(), transport=httpx.MockTransport(handler), sleep=_no_sleep
    )
    with pytest.raises(ChatAdapterError) as raised:
        run(adapter, request)

    assert raised.value.code is ChatErrorCode.OUTPUT_SCHEMA
    assert raised.value.record is not None
    assert raised.value.record.error_detail == "model_tool_call_not_allowed"


def test_duplicate_model_json_keys_are_rejected_without_retaining_content() -> None:
    request = policy_request()
    duplicate_content = '{"findings": [], "findings": []}'

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "duplicate-response",
                "model": "resolved-model-v1",
                "choices": [{"message": {"content": duplicate_content}}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    adapter = OpenAICompatibleChatAdapter(
        config(), transport=httpx.MockTransport(handler), sleep=_no_sleep
    )
    with pytest.raises(ChatAdapterError) as raised:
        run(adapter, request)

    assert raised.value.code is ChatErrorCode.OUTPUT_SCHEMA
    assert raised.value.record is not None
    assert raised.value.record.error_detail == "model_content_json_duplicate_key"
    assert duplicate_content not in str(raised.value)


def test_duplicate_response_envelope_keys_are_rejected() -> None:
    request = policy_request()
    valid = response_for(request)
    encoded = json.dumps(valid, ensure_ascii=False)
    duplicate = encoded[:-1] + ',"model":"duplicate"}'

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=duplicate.encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    adapter = OpenAICompatibleChatAdapter(
        config(), transport=httpx.MockTransport(handler), sleep=_no_sleep
    )
    with pytest.raises(ChatAdapterError) as raised:
        run(adapter, request)

    assert raised.value.code is ChatErrorCode.OUTPUT_SCHEMA
    assert raised.value.record is not None
    assert raised.value.record.error_detail == "response_json_duplicate_key"


@pytest.mark.parametrize("marker", ("synthetic_secret_placeholder", "external_sink"))
def test_sensitive_or_external_output_is_fail_closed(marker: str) -> None:
    request = policy_request()
    finding = finding_for(request)
    finding_payload = finding.model_dump(mode="json")
    finding_payload["statement"] = f"Synthetic security marker: {marker}"
    finding_payload["evidence_refs"] = [
        reference.source_id for reference in finding.evidence_refs
    ]
    finding_payload["business_fact_refs"] = []
    content = json.dumps({"findings": [finding_payload]}, ensure_ascii=False)

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "security-marker-response",
                "model": "resolved-model-v1",
                "choices": [{"message": {"content": content}}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    adapter = OpenAICompatibleChatAdapter(
        config(), transport=httpx.MockTransport(handler), sleep=_no_sleep
    )
    with pytest.raises(ChatAdapterError) as raised:
        run(adapter, request)
    assert raised.value.code is ChatErrorCode.SECURITY_BOUNDARY
    assert adapter.records[-1].error_code == ChatErrorCode.SECURITY_BOUNDARY


def test_credential_context_is_rejected_and_embedding_rerank_are_not_supported() -> (
    None
):
    request = policy_request(content=f"Synthetic text {TEST_KEY}")
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    adapter = OpenAICompatibleChatAdapter(
        config(), transport=httpx.MockTransport(handler), sleep=_no_sleep
    )
    with pytest.raises(ChatAdapterError) as raised:
        run(adapter, request)
    assert raised.value.code is ChatErrorCode.SECURITY_BOUNDARY
    assert calls == 0
    assert not adapter.records
    assert adapter.capabilities["embedding"] == EMBEDDING_STATUS == "not_supported"
    assert adapter.capabilities["rerank"] == RERANK_STATUS == "not_supported"


def test_http_error_records_only_a_bounded_server_reason() -> None:
    request = policy_request()

    def structured_error(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": f"invalid value for seed with {TEST_KEY} and Bearer leaked",
                    "type": "invalid_request_error",
                    "param": "seed",
                    "code": "invalid_parameter",
                }
            },
        )

    adapter = OpenAICompatibleChatAdapter(
        config(), transport=httpx.MockTransport(structured_error), sleep=_no_sleep
    )
    with pytest.raises(ChatAdapterError) as raised:
        run(adapter, request)
    record = adapter.records[-1]
    assert raised.value.code is ChatErrorCode.HTTP_ERROR
    assert record.error_reason == "invalid_request_error/invalid_parameter/seed"
    encoded = json.dumps(record.model_dump(mode="json"))
    assert TEST_KEY not in encoded
    assert "Bearer" not in encoded
    assert "invalid value for" not in encoded


def test_free_text_or_oversized_error_bodies_record_no_reason() -> None:
    request = policy_request()
    bodies: list[Callable[[httpx.Request], httpx.Response]] = [
        lambda _r: httpx.Response(
            400, json={"error": {"message": f"seed rejected: {TEST_KEY}"}}
        ),
        lambda _r: httpx.Response(400, text="<html>400 Bad Request</html>"),
        lambda _r: httpx.Response(400, json={"error": {"message": "x" * 8_192}}),
        lambda _r: httpx.Response(
            400, json={"error": {"type": "unsupported 中文 " + TEST_KEY}}
        ),
    ]
    for responder in bodies:
        adapter = OpenAICompatibleChatAdapter(
            config(), transport=httpx.MockTransport(responder), sleep=_no_sleep
        )
        with pytest.raises(ChatAdapterError):
            run(adapter, request)
        record = adapter.records[-1]
        assert record.error_reason is None
        assert record.error_code == ChatErrorCode.HTTP_ERROR.value
        assert TEST_KEY not in json.dumps(record.model_dump(mode="json"))


async def _no_sleep(delay: float) -> None:
    return None


def single_request(
    *, workers: tuple[WorkerKind, ...] = tuple(WorkerKind)
) -> SingleAgentModelRequest:
    """The union context a three-worker review would see, in one request."""

    policy = EvidenceRef(
        source_kind=SourceKind.CASE_DOCUMENT,
        source_id="DOC-GOLD-POLICY-0001",
        version="synthetic-gold-document.v1",
        section="synthetic-policy",
        content_checksum="sha256:" + "1" * 64,
    )
    contract = EvidenceRef(
        source_kind=SourceKind.CASE_DOCUMENT,
        source_id="DOC-GOLD-CONTRACT-0001",
        version="synthetic-gold-document.v1",
        section="synthetic-contract",
        content_checksum="sha256:" + "2" * 64,
    )
    fact = BusinessFact(
        tool_name="procurement.get_purchase_order",
        record_id="SYN-GOLD-PO-0001",
        dataset_version="synthetic-gold-procurement.v1",
        as_of=datetime(2026, 6, 30, tzinfo=timezone.utc),
        status=FactStatus.KNOWN,
        result={"purchase_order_id": "SYN-GOLD-PO-0001"},
    )
    return SingleAgentModelRequest(
        case_id="GOLD-0001",
        model_fixture_id="unused-for-real-adapter",
        question="请依据演示政策审查合成采购 Case 0001。",
        workers=workers,
        evidence_contexts=(
            EvidenceContext(evidence_ref=policy, content="Synthetic policy text."),
            EvidenceContext(evidence_ref=contract, content="Synthetic contract text."),
        ),
        business_facts=(fact,),
    )


def wire_finding(
    worker: WorkerKind,
    *,
    evidence: tuple[str, ...] = (),
    facts: tuple[str, ...] = (),
) -> dict:
    return {
        "schema_version": "1.0",
        "worker": worker.value,
        "finding_key": f"{worker.value}.required",
        "category": RiskCategory.POLICY_COMPLIANCE.value,
        "risk_level": RiskLevel.LOW.value,
        "assertion": FindingAssertion.RISK_ABSENT.value,
        "status": FindingStatus.SUPPORTED.value,
        "statement": "The union context supports this finding.",
        "recommended_action": "Continue with deterministic review.",
        "evidence_refs": list(evidence),
        "business_fact_refs": list(facts),
    }


def single_agent_adapter(
    findings: list[dict],
    *,
    include_required_findings: bool = True,
    captured: list[dict] | None = None,
) -> OpenAICompatibleChatAdapter:
    def handler(_request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(json.loads(_request.content))
        return httpx.Response(
            200,
            json={
                "id": "single-agent-probe",
                "model": "resolved-single-agent-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"findings": findings}, ensure_ascii=False
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 20,
                    "total_tokens": 60,
                },
            },
        )

    return OpenAICompatibleChatAdapter(
        config(include_required_findings=include_required_findings),
        transport=httpx.MockTransport(handler),
        sleep=_no_sleep,
    )


def test_single_agent_request_rejects_an_empty_union_context() -> None:
    with pytest.raises(ValueError, match="model-visible context"):
        SingleAgentModelRequest(
            case_id="GOLD-0002",
            model_fixture_id="unused-for-real-adapter",
            question="请审查合成采购 Case 0002。",
            workers=tuple(WorkerKind),
        )


def test_single_agent_accepts_findings_from_every_worker_in_one_call() -> None:
    request = single_request()
    adapter = single_agent_adapter(
        [
            wire_finding(WorkerKind.POLICY, evidence=("DOC-GOLD-POLICY-0001",)),
            wire_finding(WorkerKind.CONTRACT, evidence=("DOC-GOLD-CONTRACT-0001",)),
            wire_finding(WorkerKind.PROCUREMENT_DATA, facts=("SYN-GOLD-PO-0001",)),
        ]
    )
    findings = asyncio.run(adapter.generate(request))
    assert isinstance(findings, tuple) and len(findings) == 3
    assert {finding.worker for finding in findings} == set(WorkerKind)
    record = adapter.records[-1]
    assert record.agent_topology == "single_agent_branch"
    assert record.worker is None
    assert record.resolved_model_id == "resolved-single-agent-model"


def test_single_agent_still_enforces_citation_and_worker_exclusivity() -> None:
    cases = [
        [wire_finding(WorkerKind.POLICY, facts=("SYN-GOLD-PO-0001",))],
        [wire_finding(WorkerKind.PROCUREMENT_DATA, evidence=("DOC-GOLD-POLICY-0001",))],
        [
            {
                **wire_finding(WorkerKind.POLICY, evidence=("DOC-GOLD-POLICY-0001",)),
                "worker": "not_a_worker",
            }
        ],
    ]
    for findings in cases:
        adapter = single_agent_adapter(findings)
        with pytest.raises(ChatAdapterError):
            asyncio.run(adapter.generate(single_request()))
        assert adapter.records[-1].error_code in {
            ChatErrorCode.OUTPUT_SCHEMA.value,
            ChatErrorCode.INVALID_JSON.value,
        }


def test_single_agent_rejects_a_reference_outside_the_union_context() -> None:
    adapter = single_agent_adapter(
        [wire_finding(WorkerKind.POLICY, evidence=("DOC-GOLD-POLICY-9999",))]
    )
    with pytest.raises(ChatAdapterError) as raised:
        asyncio.run(adapter.generate(single_request()))
    assert raised.value.detail == "evidence_reference_out_of_context"


def test_worker_branch_requests_keep_their_existing_single_worker_boundary() -> None:
    adapter = single_agent_adapter(
        [wire_finding(WorkerKind.CONTRACT, evidence=("DOC-GOLD-CONTRACT-0001",))]
    )
    with pytest.raises(ChatAdapterError) as raised:
        asyncio.run(adapter.generate(policy_request()))
    assert raised.value.detail == "finding_worker_boundary_violation"
    assert adapter.records[-1].agent_topology == "worker_branch"
    assert adapter.records[-1].worker is WorkerKind.POLICY


def test_required_finding_vocabulary_is_opt_in_and_worker_scoped() -> None:
    captured: list[dict] = []
    adapter = single_agent_adapter(
        [wire_finding(WorkerKind.POLICY, evidence=("DOC-GOLD-POLICY-0001",))],
        captured=captured,
    )
    asyncio.run(adapter.generate(single_request()))
    context = json.loads(captured[0]["messages"][1]["content"])
    assert len(context["required_findings"]) == 3
    assert context["agent_topology"] == "single_agent"
    assert context["workers"] == [worker.value for worker in WorkerKind]

    captured.clear()
    quiet = single_agent_adapter(
        [wire_finding(WorkerKind.POLICY, evidence=("DOC-GOLD-POLICY-0001",))],
        include_required_findings=False,
        captured=captured,
    )
    asyncio.run(quiet.generate(single_request()))
    assert "required_findings" not in json.loads(captured[0]["messages"][1]["content"])

    captured.clear()
    worker_adapter = single_agent_adapter(
        [wire_finding(WorkerKind.POLICY, evidence=("DOC-GOLD-POLICY-0001",))],
        captured=captured,
    )
    with pytest.raises(ChatAdapterError):
        asyncio.run(worker_adapter.generate(policy_request()))
    worker_context = json.loads(captured[0]["messages"][1]["content"])
    assert [item["worker"] for item in worker_context["required_findings"]] == [
        WorkerKind.POLICY.value
    ]


def test_call_record_cannot_claim_a_worker_for_a_single_agent_call() -> None:
    with pytest.raises(ValueError, match="single-agent records must not claim"):
        ChatCallRecord(
            case_id="GOLD-0001",
            agent_topology="single_agent_branch",
            worker=WorkerKind.POLICY,
            requested_model_id="requested-model",
            latency_ms=5,
            retry_count=0,
        )
    with pytest.raises(ValueError, match="worker-branch records require a worker"):
        ChatCallRecord(
            case_id="GOLD-0001",
            requested_model_id="requested-model",
            latency_ms=5,
            retry_count=0,
        )
