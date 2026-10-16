"""Explicit finding-evidence.v1 Chat transport, with the registered E guard."""

from dataclasses import dataclass
import json
from pydantic import ValidationError

from .business_v3_evidence import (
    EvidenceEnvelopeV3,
    parse_evidence_v3,
    ground_document_v3,
)
from .business_v3_precise_workers import EvidenceTaskV3
from .business_v3_support import SourceCatalogV3, SupportIssue
from .business_v3_types import VALUE_MODELS
from .business_v3_security import SecurityEventV3, event
from .chat_transport_v2 import AttemptResult, decode_response, request_payload
from .chat_transport_v3 import ChatTransportV3
from .eval_f_contracts import PreciseBranchProjection, RejectedWitness
from .eval_f_value_diagnostics import typed_error_codes, validate_typed_diagnostics
from .model_attempt_v2 import BranchAttemptDriver, _history
from .model_wire_v2 import WireIssue, response_schema, strict_json, parse_findings


@dataclass(frozen=True)
class PreciseAttemptResult(AttemptResult):
    evidence_output: EvidenceEnvelopeV3 | None = None
    security_events: tuple[SecurityEventV3, ...] = ()
    rejected_witnesses: tuple[RejectedWitness, ...] = ()


def evidence_payload(config, request, correction=None, correction_fields=()):
    # Reuse HTTP parameters only; C's correction field vocabulary is not F's.
    payload = request_payload(config, request)
    schema = EvidenceEnvelopeV3.model_json_schema()
    branch_schema = response_schema(request)
    schema["$defs"].update(branch_schema["$defs"])
    schema["properties"]["findings"] = branch_schema["properties"]["findings"]
    schema["$defs"]["WitnessV3"]["properties"]["source_ref"]["enum"] = [
        s.ref for s in request.sources
    ]
    schema["$defs"]["WitnessV3"]["properties"]["field"]["enum"] = sorted(
        {
            name
            for model in VALUE_MODELS[request.worker].values()
            for name in model.model_fields
        }
    )
    schema["$defs"]["WitnessV3"]["properties"]["obligation_id"]["enum"] = list(
        VALUE_MODELS[request.worker]
    )
    if not request.sources:
        schema["properties"]["witnesses"]["maxItems"] = 0
    if config.prompt_style == "compact":

        def compact(value):
            if isinstance(value, dict):
                return {
                    k: compact(v)
                    for k, v in value.items()
                    if k not in {"title", "description"}
                }
            return [compact(v) for v in value] if isinstance(value, list) else value

        schema = compact(schema)
    # A new protocol requires its own examples and schema, never C's envelope.
    system = (
        "你是只读采购审查的隔离文档提取器。只输出一个严格JSON对象；禁止围栏、注释和解释正文。"
        "每个checks恰好一个finding；四态组合：supported/compliant/none，violated/non_compliant/已知风险，"
        "unknown/unknown/unknown，not_applicable/not_applicable/none。"
        "所有字段必填，normalized_value是最多256字符的紧凑JSON字符串，unknown必须字面量UNKNOWN。"
        "supported/violated/not_applicable需要授权evidence_refs且missing_information=[]；unknown必须说明真实缺失。"
        "statement只写“原文提取”或“材料不足”；recommended_action只写“人工核对来源”或“补充所缺证据”。"
        + EvidenceTaskV3(request).instructions
        + "每个字段都须有witness；start/end必须是完整子句边界，可引用包含全部支持内容的完整原文区间。"
        "不能按其它字段、空标题、假设、否定或疑问文字证明字段。值必须保持原文及比较方向。"
        '注意两层编码：O-01的normalized_value内exempt是JSON数组；对应witness的value必须是JSON字符串，例如"[]"，不能输出数组。'
        "start/end直接复制sources给出的span_options；不要自行估算字符位置。可选整段或完整原文，但须实际支持该字段。"
        "start/end必须是JSON整数，不能带引号或填原文。basis只取原文中连续的最短日期依据（可直接抄原文四位年份），不要拼接地区、品类和年份；applies仍根据完整范围、时点和例外判定。"
        "每个JSON键都要独立witness，包括applies、basis、exempt和special，空数组也不能省略witness。"
        "witness.value须与normalized_value里对应键严格一致（相同类型、文字、空格）；只有exempt数组须紧凑序列化为字符串。"
        "basis/standard/responsibility/damages/cap/other这些文本值逐字复制原文片段，不改写措辞、标点或空格。"
        "special可逐字复制充分原文，或仅对交付/验收/质保三个时限使用任务说明的短规范表达；三项方向与数值必须都由所引原文支持。"
        "键名已表达字段含义：buyer/seller/order/contract只填对应标识；subject/standard/responsibility取标题冒号后的正文；cap只取数值及币种，例如100000.00 CNY。不要把字段标题重复填入值。"
        'O-06的amount只填两位小数金额字符串，例如"782.61"，不能附带币种、空格或标题；currency另填币种代码，例如CNY。amount的witness.value也只能是同一个纯金额字符串。O-09的cap是另一文本字段，仍须同时包含金额和币种。'
        "other必须保留原文的正面适用事项和任何排除其它事项的限定；不得只取否定/排除的后半句。"
        "standard和responsibility必须保留标题后完整实质条款，不得仅取开头词或省略费用/责任限定；cap必须包含完整数值与币种。"
        "damages必须逐字取赔偿标题后的完整实质条款，保留触发条件、比例或金额、计算基数及时间单位；不得只取数字起始的后半段。"
    )
    user = {
        "worker": request.worker,
        "question": request.question,
        "checks": [
            {"obligation_id": c.proposition.check_id, "instruction": c.instruction}
            for c in request.checks
        ],
        # No case/family/split/label/expected outcome enters model metadata.
        "sources": [
            {
                "ref": s.ref,
                "content": s.content,
                "unicode_length": len(s.content),
                "span_options": source_spans(s.content),
            }
            for s in request.sources
        ],
    }
    if config.prompt_style == "baseline" or config.mode != "json_schema":
        system += "\n输出Schema：" + json.dumps(
            schema, ensure_ascii=False, separators=(",", ":")
        )
    payload["messages"] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(user, ensure_ascii=False, separators=(",", ":")),
        },
    ]
    if correction:
        payload["messages"].append(
            {
                "role": "system",
                "content": "上次输出合同无效；请重做，错误码："
                + correction
                + "；固定字段："
                + ",".join(correction_fields),
            }
        )
    if config.mode == "json_schema":
        payload["response_format"]["json_schema"].update(
            name="finding_evidence_v1", schema=schema
        )
    return payload


def source_spans(content):
    offsets = [{"start": 0, "end": len(content), "text": content}]
    start = 0
    for paragraph in content.splitlines(keepends=True):
        end = start + len(paragraph.rstrip("\r\n"))
        option = {"start": start, "end": end, "text": content[start:end]}
        if end > start and option not in offsets:
            offsets.append(option)
        start += len(paragraph)
    return offsets


class PreciseChatTransport(ChatTransportV3):
    def __init__(self, *args, catalog: SourceCatalogV3, **kwargs):
        super().__init__(*args, **kwargs)
        self.catalog = catalog

    def decode(self, response, request, credential):
        outputs = []
        events = []
        rejected = []

        def guard(value, credential=""):
            events.extend(
                self.protection.inspect(
                    value, credential=credential, location=request.worker
                )
            )
            self.protection.assert_safe(value, credential)

        def parse(content, branch):
            try:
                raw = strict_json(content)
            except ValueError:
                raise WireIssue("invalid_json", correctable=True) from None
            guard(raw, credential)
            validate_typed_diagnostics(raw, branch.worker)
            if isinstance(raw, dict) and "findings" in raw:
                parse_findings(
                    json.dumps(
                        {
                            "schema_version": "finding-envelope.v2",
                            "findings": raw["findings"],
                        }
                    ),
                    branch,
                )
            try:
                envelope = EvidenceEnvelopeV3.model_validate_json(json.dumps(raw))
            except ValidationError as exc:
                allowed = {
                    "schema_version",
                    "findings",
                    "witnesses",
                    "obligation_id",
                    "field",
                    "value",
                    "source_ref",
                    "start",
                    "end",
                }
                fields = {
                    next(
                        (part for part in reversed(e["loc"]) if part in allowed),
                        "witnesses",
                    )
                    for e in exc.errors(
                        include_input=False, include_context=False, include_url=False
                    )
                }
                raise WireIssue(
                    "invalid_schema", correctable=True, fields=tuple(sorted(fields))
                ) from None
            document, witnesses = parse_evidence_v3(
                content, branch, protection=self.protection
            )
            if any(
                f.finding.statement not in {"原文提取", "材料不足"}
                or f.finding.recommended_action not in {"人工核对来源", "补充所缺证据"}
                for f in document.findings
            ):
                raise WireIssue(
                    "invalid_schema",
                    correctable=True,
                    fields=("statement", "recommended_action"),
                )
            if document.failures:
                raise WireIssue(
                    "invalid_schema", correctable=True, fields=("normalized_value",)
                )
            grounded = ground_document_v3(branch, document, witnesses, self.catalog)
            if grounded.support_failures:
                for witness in witnesses:
                    if (
                        witness.obligation_id not in VALUE_MODELS[branch.worker]
                        or witness.field
                        not in VALUE_MODELS[branch.worker][
                            witness.obligation_id
                        ].model_fields
                    ):
                        continue
                    try:
                        self.catalog.verify(branch, witness)
                    except SupportIssue as exc:
                        if len(rejected) < 16:
                            rejected.append(
                                RejectedWitness(
                                    obligation_id=witness.obligation_id,
                                    field=witness.field,
                                    value=witness.value,
                                    code=exc.code,
                                )
                            )
                for key, typed in document.values.items():
                    for name, value in typed.model_dump().items():
                        if isinstance(value, tuple):
                            value = json.dumps(value, separators=(",", ":"))
                        if not any(
                            w.obligation_id == key
                            and w.field == name
                            and type(w.value) is type(value)
                            and w.value == value
                            for w in witnesses
                        ):
                            # Names come from trusted typed schemas, never arbitrary model text.
                            raise WireIssue(
                                "citation_value_mismatch_"
                                + key.lower().replace("-", "_")
                                + "_"
                                + name
                            )
                raise WireIssue(grounded.support_failures[0])
            outputs.append(envelope)
            return document.findings

        result = decode_response(
            response,
            request,
            self.config,
            credential,
            output_guard=guard,
            parse_output=parse,
        )
        if result.error_code == "tool_call_forbidden":
            events.append(event("tool_call_blocked", "transport", request.worker))
        return PreciseAttemptResult(
            **vars(result),
            evidence_output=outputs[0] if outputs else None,
            security_events=tuple(
                sorted(set(events), key=lambda e: (e.code, e.location_sha256))
            ),
            rejected_witnesses=tuple(rejected),
        )


class PreciseAttemptDriver(BranchAttemptDriver):
    stage_id = "V1-07F"
    projection_type = PreciseBranchProjection

    def correctable_code(self, code):
        return super().correctable_code(code) or code in typed_error_codes()

    def make_payload(self, request, correction, correction_fields):
        return evidence_payload(
            self.transport.config, request, correction, correction_fields
        )

    def make_projection(self, observed, success, *, task_key, **fields):
        first_body = (
            observed.received_body if fields["model_request_count"] == 1 else None
        )
        if fields["model_request_count"] > 1:
            history = _history(self.ledger, task_key)
            if history and history[0]["receipt_json"]:
                first_body = json.loads(history[0]["receipt_json"]).get(
                    "received_nonempty_body"
                )
        projection = PreciseBranchProjection(
            **fields,
            security_events=getattr(observed, "security_events", ()),
            rejected_witnesses=getattr(observed, "rejected_witnesses", ()),
            first_received_nonempty_body=first_body,
            evidence_output=getattr(observed, "evidence_output", None)
            if success
            else None,
        )
        # Canonicalize inherited identity models at the durable wire boundary.
        # A reopened checkpoint must have exactly the same types and values.
        return PreciseBranchProjection.model_validate_json(projection.model_dump_json())

    def discarded_projection(self, projection):
        return (
            super()
            .discarded_projection(projection)
            .model_copy(
                update={
                    "evidence_output": None,
                    "security_events": (),
                    "rejected_witnesses": (),
                }
            )
        )
