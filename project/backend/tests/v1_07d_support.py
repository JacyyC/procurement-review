"""Hand-authored source facts, separate from Gold and from production rules."""

from datetime import UTC, date, datetime
import hashlib
import json

from procurement_review.business_v3_types import (
    BusinessContextV3,
    BusinessEvidenceIdentity,
)
from procurement_review.business_v3_documents import (
    document_request_v3,
    decode_document_values,
)
from procurement_review.business_v3_procurement import review_procurement_v3
from procurement_review.business_v3_aggregate import aggregate_business_v3
from procurement_review.contracts import BusinessFact, FactStatus
from procurement_review.model_wire_v2 import SourceSlice, parse_findings
from procurement_review.worker_contracts import (
    ProcurementReadResult,
    ProcurementReadStatus,
)


def context():
    return BusinessContextV3(
        scope_id="SCOPE-A",
        purchase_order_id="ORDER-A",
        supplier_id="SELLER-A",
        contract_id="CONTRACT-A",
        as_of=date(2026, 6, 30),
        dataset_version="business-test.v1",
    )


VALUES = {
    "O-01": {"applies": True, "basis": "2026年度华东设备采购", "exempt": []},
    "O-02": {
        "special": "交付15日/验收7日/质保90日",
        "quotes": 3,
        "delivery": 15,
        "acceptance": 7,
        "warranty": 90,
        "supplier": "all_qualified",
        "money": "equal_invoice_capped_tx",
        "dates": "booked_ge_ordered",
    },
    "O-05": {
        "buyer": "BUYER-A",
        "seller": "SELLER-A",
        "subject": "备件",
        "order": "ORDER-A",
        "contract": "CONTRACT-A",
    },
    "O-06": {"amount": "100.10", "currency": "CNY"},
    "O-08": {
        "delivery": 10,
        "acceptance": 5,
        "warranty": 120,
        "standard": "数量匹配且100%完好",
        "responsibility": "卖方免费修复更换",
    },
    "O-09": {
        "written": True,
        "damages": "逾期每日订单价款0.1%",
        "cap": "100000.00 CNY",
        "other": "本次只审书面变更、赔付与上限",
    },
}


def sources(worker, ctx=None):
    ctx = ctx or context()
    content = (
        "2026年度华东设备采购：供应商登记active、采购approved、税号存在、尽调verified、筛查clear。"
        "报价至少3；订单与发票等额，交易累计不超过订单，均同币；记账不早于下单。"
        "交付不超过15日，验收不超过7日，质保不少于90日。"
        if worker == "policy"
        else "买方BUYER-A，卖方SELLER-A，标的备件，订单ORDER-A，合同CONTRACT-A，价款100.10 CNY。"
        "交付10日、验收5日、质保120日，标准数量匹配且100%完好，卖方免费修复更换。"
        "变更须双方书面签署，逾期每日订单价款0.1%，累计赔偿上限100000.00 CNY。本次只审书面变更、赔付与上限。"
    )
    return (
        SourceSlice(
            scope_id=ctx.scope_id,
            worker=worker,
            identity=BusinessEvidenceIdentity(
                source_id=f"DOC-{worker}",
                version="document.v1",
                locator="section-1",
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            ),
            content=content,
        ),
    )


def response_body(request, *, changes=None, unknown=()):
    values = {key: dict(value) for key, value in VALUES.items()}
    for key, fields in (changes or {}).items():
        values[key].update(fields)
    rows = []
    for check in request.checks:
        key = check.proposition.check_id
        absent = key in unknown
        exempt = key == "O-01" and not values[key]["applies"]
        rows.append(
            dict(
                obligation_id=key,
                status="unknown"
                if absent
                else "not_applicable"
                if exempt
                else "supported",
                assertion="unknown"
                if absent
                else "not_applicable"
                if exempt
                else "compliant",
                risk_level="unknown" if absent else "none",
                normalized_value="UNKNOWN"
                if absent
                else json.dumps(values[key], ensure_ascii=False, separators=(",", ":")),
                evidence_refs=[] if absent else [request.sources[0].ref],
                missing_information=["对应原文未提供"] if absent else [],
                statement="按所给原文提取",
                recommended_action="人工核对来源",
            )
        )
    return {"schema_version": "finding-envelope.v2", "findings": rows}


def document(worker, *, changes=None, unknown=(), ctx=None, question="请审查采购"):
    ctx = ctx or context()
    request = document_request_v3(
        ctx,
        case_id="V107-D-0001",
        worker=worker,
        question=question,
        sources=sources(worker, ctx),
    )
    body = response_body(request, changes=changes, unknown=unknown)
    return decode_document_values(request, parse_findings(json.dumps(body), request))


def mcp_result(*, order=None, supplier=None, transactions=None):
    order_data = dict(
        purchase_order_id="ORDER-A",
        supplier_id="SELLER-A",
        amount_minor=10010,
        currency="CNY",
        invoice_amount_minor=10010,
        invoice_currency="CNY",
        competitive_quote_count=3,
        contract_reference="CONTRACT-A",
        ordered_on="2026-02-28",
    )
    supplier_data = dict(
        supplier_id="SELLER-A",
        registry_status="active",
        procurement_status="approved",
        tax_identifier="SYN-TAX-EXISTS",
        due_diligence_status="verified",
        sanctions_screening="clear",
    )
    tx_data = dict(
        transaction_id="TX-A",
        supplier_id="SELLER-A",
        purchase_order_id="ORDER-A",
        amount_minor=10010,
        currency="CNY",
        booked_on="2026-03-01",
        counterparty_reference="BUYER-A",
        status="settled",
    )
    order_data.update(order or {})
    supplier_data.update(supplier or {})
    txs = [tx_data | change for change in (transactions or [{}])]
    facts = [
        BusinessFact(
            tool_name=tool,
            record_id=record,
            dataset_version="business-test.v1",
            as_of=datetime(2026, 6, 30, tzinfo=UTC),
            status=FactStatus.KNOWN,
            result=data,
        )
        for tool, record, data in [
            ("get_purchase_order", "ORDER-A", order_data),
            ("get_supplier", "SELLER-A", supplier_data),
            *[("list_historical_transactions", tx["transaction_id"], tx) for tx in txs],
        ]
    ]
    return ProcurementReadResult(
        status=ProcurementReadStatus.KNOWN,
        requested_dataset_version="business-test.v1",
        facts=tuple(facts),
    )


def evaluate(
    *, order=None, supplier=None, transactions=None, changes=None, unknown=(), ctx=None
):
    ctx = ctx or context()
    policy = document("policy", changes=changes, unknown=unknown, ctx=ctx)
    contract = document("contract", changes=changes, unknown=unknown, ctx=ctx)
    procurement = review_procurement_v3(
        ctx,
        mcp_result(order=order, supplier=supplier, transactions=transactions),
        authorized_scope_id=ctx.scope_id,
    )
    return aggregate_business_v3(ctx, policy, contract, procurement)


def states(result):
    return {r.obligation_id: r.state for r in result.findings}
