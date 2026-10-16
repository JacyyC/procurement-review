"""Hand-authored evidence spans; no Gold/truth imports or generated expectations."""

import hashlib
import json

from procurement_review.business_v3_documents import document_request_v3
from procurement_review.business_v3_evidence import (
    parse_evidence_v3,
    ground_document_v3,
)
from procurement_review.business_v3_security import ProtectedOutputV3
from procurement_review.business_v3_support import LocatedSourceV3, SourceCatalogV3
from procurement_review.business_v3_types import BusinessEvidenceIdentity
from procurement_review.model_wire_v2 import SourceSlice
from v1_07d_support import context, response_body

POLICY = (
    "适用于2026年度华东设备采购。登记active、采购approved、税号存在、尽调verified、筛查clear。"
    "报价至少3。订单与发票等额，交易累计不超过订单，均同币。记账不早于下单。"
    "交付不超过15日，验收不超过7日，质保不少于90日。专项：交付15日/验收7日/质保90日。"
)
CONTRACT = (
    "买方BUYER-A，卖方SELLER-A，标的备件，订单ORDER-A，合同CONTRACT-A，价款100.10 CNY。"
    "交付10日、验收5日、质保120日。标准：数量匹配且100%完好，质保责任：卖方免费修复更换。"
    "变更须双方书面签署。逾期每日订单价款0.1%，累计赔偿上限100000.00 CNY。"
    "本次只审书面变更、赔付与上限。"
)


def located(
    worker,
    content=None,
    *,
    source_id=None,
    version="document.v1",
    locator="chunk-1",
    subject_id=None,
    effective_at=None,
):
    ctx = context()
    content = content or (POLICY if worker == "policy" else CONTRACT)
    source = SourceSlice(
        scope_id=ctx.scope_id,
        worker=worker,
        identity=BusinessEvidenceIdentity(
            source_id=source_id or f"DOC-{worker}",
            version=version,
            locator=locator,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        ),
        content=content,
    )
    return LocatedSourceV3(
        source,
        "purchase_order",
        subject_id or ctx.purchase_order_id,
        effective_at or ctx.as_of,
    )


def request(worker, entries=None):
    entries = entries or (located(worker),)
    return document_request_v3(
        context(),
        case_id="V107-D-0001",
        worker=worker,
        question="请核对交付、验收、质保、书面变更、赔付与上限",
        sources=tuple(e.source for e in entries),
    )


def evidence_body(req, *, changes=None, unknown=()):
    body = response_body(req, changes=changes, unknown=unknown)
    body["schema_version"] = "finding-evidence.v1"
    body["witnesses"] = []
    for row in body["findings"]:
        if row["status"] == "unknown":
            continue
        values = json.loads(row["normalized_value"])
        for name, value in values.items():
            body["witnesses"].append(
                dict(
                    obligation_id=row["obligation_id"],
                    field=name,
                    value=json.dumps(value, separators=(",", ":"))
                    if isinstance(value, list)
                    else value,
                    source_ref=req.sources[0].ref,
                    start=0,
                    end=len(req.sources[0].content),
                )
            )
    return body


def grounded(worker, *, entries=None, body=None, changes=None, unknown=()):
    entries = entries or (located(worker),)
    req = request(worker, entries)
    body = body or evidence_body(req, changes=changes, unknown=unknown)
    doc, witnesses = parse_evidence_v3(
        json.dumps(body), req, protection=ProtectedOutputV3()
    )
    return ground_document_v3(req, doc, witnesses, SourceCatalogV3(entries))
