"""Isolated document tasks and strict typed values inside finding-envelope.v2.

normalized_value stays a JSON string on the C wire. Here its JSON object is
validated against an explicit per-duty schema; prose is never mined for numbers.
"""

import json
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping

from pydantic import ValidationError

from .business_v3_types import (
    VALUE_MODELS,
    BusinessPropositionIdentity as PropositionIdentity,
)
from .model_wire_v2 import (
    BranchRequestV2,
    CheckSlot,
    ValidatedFindingV2,
    strict_json,
)


@dataclass(frozen=True)
class DocumentValuesV3:
    worker: str
    values: Mapping
    findings: tuple[ValidatedFindingV2, ...]
    failures: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def evidence(self, obligation):
        return tuple(
            e
            for f in self.findings
            if f.finding.obligation_id == obligation
            for e in f.evidence
        )


def _compact_schema(model):
    schema = model.model_json_schema()

    def strip(value):
        if isinstance(value, dict):
            return {
                k: strip(v)
                for k, v in value.items()
                if k not in {"title", "description"}
            }
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return strip(schema)


def document_request_v3(context, *, case_id, worker, question, sources):
    if worker not in VALUE_MODELS:
        raise ValueError("procurement does not have a model task")
    checks = []
    for obligation in VALUE_MODELS[worker]:
        checks.append(
            CheckSlot(
                proposition=PropositionIdentity(
                    scope_id=context.scope_id,
                    subject_type="purchase_order",
                    subject_id=context.purchase_order_id,
                    check_id=obligation,
                    field_path=f"{worker}.{obligation.lower().replace('-', '_')}",
                    effective_at=context.as_of,
                    source_version="typed-business-values.v1",
                ),
                instruction="提取本项具体条款，normalized_value须为下述Schema对象的JSON字符串；缺实际材料即unknown，不用其它分支材料补齐；不得判整个Case。",
            )
        )
    schemas = {k: _compact_schema(v) for k, v in VALUE_MODELS[worker].items()}
    task = (
        question
        + "\n本分支仅作文本提取；其它分支各自核对采购事实或条款，聚合层才跨域比较。"
        "未知值不得猜测，明确不适用须basis；只取已声明适用的条款，不新增隐藏批准条件。"
        "O-01的applies按本Case范围/日期判定；O-02的special须回答问题点名的专项具体值/清单/期限或明确豁免依据，泛称有条款无效。"
        "all_qualified仅当原文要求登记active、采购approved、税号存在、尽调verified、筛查clear；"
        "equal_invoice_capped_tx仅当原文要求订单发票同币等额且交易累计不超订单；booked_ge_ordered表示记账不早于下单。"
        "O-05提取双方标的引用；O-06提取价款币种；O-08提取时限、标准责任；O-09提取书面变更、赔付和上限，other覆盖问题/政策声明适用的分包、知识产权、争议等内容或明确不适用依据。"
        "数字天数不代表标准/责任完整。只有实际原文支持提取时supported；明确违反本分支条款义务时violated；缺实际材料unknown。只有O-01可not_applicable且applies=false，exempt逐项列明原文豁免的义务，不适用不得自动豁免其它事项；其它豁免留明确文字依据。"
        "normalized_value用紧凑JSON且总长≤256字符，每个文本值≤64字符；只省略JSON结构分隔符空格，字符串内原文空格与标点不得省略或重排。"
        "专项三个时限的special可规范为delivery<=数;acceptance<=数;warranty>=数，仍须原文逐项支持数值和方向并引用包含三项的精确片段；不得从示例填数。"
        "只依据授权原文填以下JSON对象，再将其序列化为normalized_value字符串："
        + json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
    )
    return BranchRequestV2(
        case_id=case_id,
        worker=worker,
        scope_id=context.scope_id,
        question=task,
        checks=tuple(checks),
        sources=tuple(sources),
    )


def decode_document_values(request, findings, *, registry=None):
    """Judge one document answer against the versioned value registry in force.

    ``registry`` is a ``{worker: {obligation: model}}`` mapping; the frozen 3.0.0
    registry is the default, so a versioned rule set supplies its own instead of
    changing what the frozen line accepts.
    """

    models = (VALUE_MODELS if registry is None else registry)[request.worker]
    expected = {c.proposition.check_id: c.proposition for c in request.checks}
    allowed = {s.ref: s.identity for s in request.sources}
    values, errors = {}, []
    if len(findings) != len(expected) or {
        f.finding.obligation_id for f in findings
    } != set(expected):
        return DocumentValuesV3(request.worker, {}, (), ("document_duty_incomplete",))
    for observed in findings:
        item = observed.finding
        if (
            observed.worker != request.worker
            or observed.proposition != expected[item.obligation_id]
            or observed.evidence
            != tuple(allowed.get(ref) for ref in item.evidence_refs)
        ):
            return DocumentValuesV3(
                request.worker, {}, (), ("document_boundary_invalid",)
            )
        if item.status == "unknown":
            continue
        try:
            value = strict_json(item.normalized_value)
            typed = models[item.obligation_id].model_validate_json(
                json.dumps(value), strict=True
            )
            if item.obligation_id == "O-01":
                if item.status != ("supported" if typed.applies else "not_applicable"):
                    raise ValueError("applicability state mismatch")
            elif item.status == "not_applicable":
                raise ValueError("explicit applicability is required")
            values[item.obligation_id] = typed
        except (ValueError, ValidationError):
            errors.append("document_typed_value_invalid")
    return DocumentValuesV3(
        request.worker, values, tuple(findings), tuple(sorted(set(errors)))
    )
