"""Independent authored source/answer fixtures for the C wire boundary."""

import hashlib
import json
from datetime import UTC, datetime, timedelta, date

import httpx
from pydantic import SecretStr

from procurement_review.chat_transport_v2 import ChatTransportV2, EndpointConfigV2
from procurement_review.eval_attempt_ledger import AttemptLedger
from procurement_review.gold_v3_contracts import EvidenceIdentity, PropositionIdentity
from procurement_review.model_wire_v2 import BranchRequestV2, CheckSlot, SourceSlice
from procurement_review.model_attempt_v2 import (
    BatchAuthority,
    BudgetProof,
    BranchAttemptDriver,
    configuration_hash,
)


def branch(worker="policy"):
    content = "本采购政策适用于当前采购，2026年全年有效。"
    source = SourceSlice(
        scope_id="V107-SCOPE-A",
        worker=worker,
        identity=EvidenceIdentity(
            source_id="V107-DOC-A",
            version="3.0",
            locator="chunk:one/section:scope",
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        ),
        content=content,
    )
    proposition = PropositionIdentity(
        scope_id="V107-SCOPE-A",
        subject_type="purchase_order",
        subject_id="V107-ORDER-A",
        check_id="O-01",
        field_path="policy.applicability",
        effective_at=date(2026, 6, 30),
        source_version="3.0",
    )
    return BranchRequestV2(
        case_id="V107-D-0001",
        worker=worker,
        scope_id="V107-SCOPE-A",
        question="核对采购的政策适用性",
        checks=(
            CheckSlot(proposition=proposition, instruction="提取适用性及有效时点"),
        ),
        sources=(source,),
    )


def answer(request=None, state="supported"):
    request = request or branch()
    fields = {
        "supported": ("compliant", "none", "适用于当前采购", []),
        "violated": ("non_compliant", "high", "明确禁止", []),
        "unknown": ("unknown", "unknown", "UNKNOWN", ["缺少适用时点"]),
        "not_applicable": ("not_applicable", "none", "有明确例外", []),
    }[state]
    return {
        "schema_version": "finding-envelope.v2",
        "findings": [
            {
                "obligation_id": "O-01",
                "status": state,
                "assertion": fields[0],
                "risk_level": fields[1],
                "normalized_value": fields[2],
                "missing_information": fields[3],
                "evidence_refs": [] if state == "unknown" else [request.sources[0].ref],
                "statement": "证据说明当前事项",
                "recommended_action": "审阅来源后由人工决定",
            }
        ],
    }


def response(
    body=None,
    *,
    finish="stop",
    model="test-model-1",
    status=200,
    usage=True,
    refusal=None,
):
    return httpx.Response(
        status,
        json={
            "id": "request-test",
            "model": model,
            "usage": {"prompt_tokens": 31, "completion_tokens": 41} if usage else None,
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {
                        "content": json.dumps(body or answer(), ensure_ascii=False),
                        "refusal": refusal,
                    },
                }
            ],
        },
    )


def setup_driver(tmp_path, handler, *, now=None, allocated=4, reserve_budget=10000):
    config = EndpointConfigV2(
        endpoint="https://example.invalid/v1/chat/completions",
        requested_model="test-model",
        resolved_model="test-model-1",
        mode="json_schema",
    )
    transport = ChatTransportV2(
        config,
        SecretStr("test-credential-not-real"),
        transport=httpx.MockTransport(handler),
    )
    clock = now or (lambda: datetime.now(UTC))
    proof = BudgetProof(
        evidence_sha256="a" * 64,
        cumulative_request_upper=0,
        cumulative_input_upper=0,
        cumulative_output_upper=0,
        historical_unknown_input_reserve=0,
        historical_unknown_output_reserve=0,
        allocated_requests=allocated,
        allocated_input=reserve_budget,
        allocated_output=reserve_budget,
    )
    authority = BatchAuthority(
        authorization_id="offline-test-only",
        scope_ids=frozenset({"V107-SCOPE-A"}),
        expires_at=clock() + timedelta(hours=1),
        proof=proof,
        now=clock,
    )
    ledger = AttemptLedger.create(
        tmp_path / ".local" / "c" / "ledger.sqlite3",
        run_id="v1-07-c-test",
        gold_manifest_sha256="b" * 64,
        configuration_sha256=configuration_hash(transport, authority),
        max_requests=allocated,
        max_input_tokens=reserve_budget,
        max_output_tokens=reserve_budget,
        stage_id="V1-07C",
    )
    return BranchAttemptDriver(ledger, transport, authority), clock() + timedelta(
        seconds=90
    )
