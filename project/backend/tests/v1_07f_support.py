"""Offline transport fixtures only. Never accepted as real quality evidence."""

import json
import re
from datetime import UTC, datetime, timedelta

import httpx
from pydantic import SecretStr

from procurement_review.business_v3_security import ProtectedOutputV3
from procurement_review.business_v3_support import SourceCatalogV3
from procurement_review.eval_f_context import (
    stable_write,
    load_cases,
    located_sources,
    context_for,
)
from procurement_review.eval_f_contracts import FBudgetProof
from procurement_review.eval_f_registration import configuration
from procurement_review.eval_f_run import FAuthority
from procurement_review.eval_f_transport import (
    PreciseAttemptDriver,
    PreciseChatTransport,
)
from procurement_review.eval_attempt_ledger import AttemptLedger
from procurement_review.model_attempt_v2 import configuration_hash
from procurement_review.business_v3_documents import document_request_v3


def body_for_payload(payload):
    user = json.loads(payload["messages"][1]["content"])
    text = user["sources"][0]["content"]
    ref = user["sources"][0]["ref"]
    if user["worker"] == "policy":
        values = {
            "O-01": {"applies": True, "basis": "2026", "exempt": []},
            "O-02": {
                "special": "交付不超过 15 日、验收不超过 7 日、质保不少于 90 日",
                "quotes": 3,
                "delivery": 15,
                "acceptance": 7,
                "warranty": 90,
                "supplier": "all_qualified",
                "money": "equal_invoice_capped_tx",
                "dates": "booked_ge_ordered",
            },
        }
    else:

        def grab(pattern):
            return re.search(pattern, text)[1]

        values = {
            "O-05": {
                "buyer": grab(r"买方 (\S+?)[；。]"),
                "seller": grab(r"卖方 (\S+?)[；。]"),
                "subject": grab(r"标的：([^。]+)"),
                "order": grab(r"订单 (V107-PO-[A-Z]-\d+)"),
                "contract": grab(r"合同 (V107-CONTRACT-[A-Z]-\d+)"),
            },
            "O-06": {"amount": grab(r"价款记录：订单 (\d+\.\d+)"), "currency": "CNY"},
            "O-08": {
                "delivery": int(grab(r"交付 (\d+) 日")),
                "acceptance": 5,
                "warranty": int(grab(r"质保 (\d+) 日")),
                "standard": "到货数量逐项相符且完好率100%",
                "responsibility": "卖方免费修复或更换缺陷交付物",
            },
            "O-09": {
                "written": True,
                "damages": "逾期交付每天按订单价款0.1%赔偿",
                "cap": "100000.00 CNY",
                "other": "本例仅适用上述变更、违约和责任上限事项，不增加分包、知识产权或争议程序检查",
            },
        }
    findings = []
    witnesses = []
    for key, value in values.items():
        findings.append(
            dict(
                obligation_id=key,
                status="supported",
                assertion="compliant",
                risk_level="none",
                normalized_value=json.dumps(
                    value, ensure_ascii=False, separators=(",", ":")
                ),
                evidence_refs=[ref],
                missing_information=[],
                statement="原文提取",
                recommended_action="人工核对来源",
            )
        )
        for field, v in value.items():
            witnesses.append(
                dict(
                    obligation_id=key,
                    field=field,
                    value=json.dumps(v, separators=(",", ":"))
                    if isinstance(v, list)
                    else v,
                    source_ref=ref,
                    start=0,
                    end=len(text),
                )
            )
    return dict(
        schema_version="finding-evidence.v1", findings=findings, witnesses=witnesses
    )


def response_for(req, *, body=None, status=200, model="deepseek-v4-flash-ga-260731"):
    payload = json.loads(req.content)
    return httpx.Response(
        status,
        json=dict(
            model=model,
            usage=dict(prompt_tokens=100, completion_tokens=200),
            choices=[
                dict(
                    finish_reason="stop",
                    message=dict(
                        content=json.dumps(
                            body if body is not None else body_for_payload(payload),
                            ensure_ascii=False,
                        )
                    ),
                )
            ],
        ),
    )


def setup(tmp_path, handler):
    root = tmp_path / ".local"
    execution = root / "execution.json"
    stable_write(execution, dict(generation=1, cancelled=False))
    case = load_cases()[0]
    located = located_sources(case)
    proof = FBudgetProof(
        evidence_sha256="a" * 64,
        cumulative_request_upper=7871,
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
        authorization_id="test-f",
        scope_ids=frozenset([context_for(case).scope_id]),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        proof=proof,
        execution_path=execution,
        generation=1,
    )
    transport = PreciseChatTransport(
        configuration(),
        SecretStr("offline-test-key"),
        protection=ProtectedOutputV3(),
        catalog=SourceCatalogV3(located),
        transport=httpx.MockTransport(handler),
    )
    ledger = AttemptLedger.create(
        root / "ledger.sqlite3",
        run_id="v1-07-f-test",
        gold_manifest_sha256="b" * 64,
        configuration_sha256=configuration_hash(transport, authority),
        max_requests=8,
        max_input_tokens=None,
        max_output_tokens=None,
        stage_id="V1-07F",
        gold_set_version="gold-set.v3.1",
    )
    driver = PreciseAttemptDriver(ledger, transport, authority)
    request = document_request_v3(
        context_for(case),
        case_id=case.case_id,
        worker="policy",
        question=case.model_input.question,
        sources=tuple(s.source for s in located if s.source.worker == "policy"),
    )
    return driver, request, datetime.now(UTC) + timedelta(seconds=120)
