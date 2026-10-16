"""The product wire for rules 3.1.0: versioned schema in, grounded envelope out.

The frozen product wire asks for ``finding-envelope.v2`` under the 3.0.0 value
registry.  This transport is its versioned sibling: the same HTTP contract and the
same bounded response handling, but the model sees the ``finding-evidence.v2``
schema, the 3.1.0 duty registry and the conditional field vocabularies.  One call
builds one payload and performs one physical send; corrections are separate sends
the runner admits again, so no retry happens inside this object.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pydantic import SecretStr, ValidationError

from .business_v31_documents import VALUE_MODELS_V31
from .precise_answer_v3 import (
    PRECISE_ANSWER_VERSION,
    PRECISE_VALUE_LIMIT,
    PreciseAnswerEnvelopeV3,
    parse_precise_answer,
)
from .business_v3_security import ProtectedOutputV3, event
from .business_v3_support import SourceCatalogV3
from .chat_transport_v2 import (
    AttemptResult,
    ChatTransportV2,
    decode_response,
    request_payload,
)
from .eval_f_contracts import FEndpointConfig
from .eval_f_transport import source_spans
from .model_wire_v2 import WireIssue, strict_json

PRECISE_WIRE_SCHEMA = PRECISE_ANSWER_VERSION

#: The fixed part of the model-visible prompt.  It is a module constant so the batch and
#: Run fingerprints cover it: changing one word of the instructions is a new candidate.
PRECISE_SYSTEM_PROMPT = (
    "你是只读采购审查的隔离文档提取器，规则版本rules 3.1.0，答案合同finding-evidence.v3。"
    "只输出一个严格JSON对象；禁止围栏、注释和解释正文。每个checks恰好一个finding。"
    "每个finding只写：obligation_id、status、statement、recommended_action，以及下面按状态要求的字段；"
    "assertion由服务端按status生成，不要输出。"
    "status=supported：normalized_value为本义务Schema对象的JSON值（对象即可，不必再序列化成字符串），"
    "evidence_refs为授权引用，risk_level不要输出，missing_information不要输出。"
    "status=violated：同上，且必须给出risk_level（low|medium|high|critical）。"
    "status=unknown：不要输出normalized_value与evidence_refs，也不要为该义务输出任何witness；missing_information必须逐项写明真实缺失。"
    "witness只允许出现在本分支声明了取值的义务上，且field必须是该义务Schema中列出的键名；否则服务端不会读取它。"
    "每个witness的obligation_id与field必须同属一个义务：不得把某义务的字段挂到另一个义务下"
    "（例如O-02的special或O-04的exceptions不能写在O-01的witness里）；一条取值只用它自己义务的字段与引用。"
    "status=not_applicable：仅O-01的applies=false可用，其它义务不得用；需要normalized_value与evidence_refs。"
    "statement只写“原文提取”或“材料不足”；recommended_action只写“人工核对来源”或“补充所缺证据”。"
    "normalized_value的对象键必须来自该义务的Schema键名，值保持原文；缺实际材料即unknown，不用其它分支材料补齐，"
    "也不得判整个Case。每个字段都须有witness：witness.value就是该字段的值，"
    "列表或对象直接写数组或对象（服务端会规范化），字符串保持原样，不要用空字符串表示省略。"
    "witness的obligation_id/field/source_ref只允许使用本任务已授权词表；start/end必须是完整子句边界，"
    "直接复制sources给出的span_options，可引用包含全部支持内容的完整原文区间。"
    "条件字段只依据本分支授权原文，逐字复制、不得改写或缩写：政策分支的O-04先给出阈值与适用类别；"
    "排除只写原文确实写出的排除（例如“租赁类交易不适用竞争性报价要求”），exclusion_basis逐字复制该排除的完整子句；"
    "原文没有写排除时，excluded_categories写空数组、给出片段不含排除措辞的witness，并且不要输出exclusion_basis。"
    "例外只写原文登记的例外，给出required与逐字复制的exception_basis；"
    "原文没有登记例外时，exceptions与exception_requires写空数组，并且不要输出exception_basis。"
    "不得用适用类别、常识或其它分支材料推断排除与例外。"
    "合同分支的O-04给出本次交易类别及其basis，以及原文明确陈述的单一来源、书面说明、审批引用、价格合理与承诺现行状态；"
    "原文未载明交易类别时该义务写unknown并逐项写明缺失，不要给出空值或空字符串witness。"
    "commitment_current只在原文自身写出承诺是否现行有效时才填写：原文出现替代或变更原承诺的声明（如补充协议改写质保期）时填false并在superseding_amendment逐字复制该声明；原文没有写出该状态时整个省略该键，不要默认填true。"
    "承诺是否现行由合同分支O-04的commitment_current表达：原文明确写出原承诺已被补充协议或"
    "变更协议替代时填false，并在superseding_amendment逐字复制该声明；原文没有写出该状态时整个"
    "省略该键，不要默认填true。服务端据此判定以被替代承诺支撑的O-08不成立，模型不必也不得为此改动O-08。"
    "未在原文出现的维度不得猜测：政策不得猜交易类别，合同不得复述政策文字；"
    "没有把握的维度就整个省略，不要用空值或空字符串占位。空列表也必须给出witness"
    "（其片段不得含对应排除或例外的措辞）。"
    "键名已表达字段含义：buyer/seller/order/contract只填对应标识；"
    "subject/standard/responsibility取标题冒号后的正文；cap只取数值及币种。"
    "O-06的amount只填两位小数金额字符串，currency另填币种代码；"
    "other必须保留原文的正面适用事项和任何排除其它事项的限定。"
)


#: The refusals that must reach the Run as themselves.  An unauthorized citation, an
#: unplanned check or a security refusal is an authorization/security fact, never a
#: correctable spelling problem: wrapping them as ``invalid_schema`` would invite the one
#: correction to re-ask a model that asked for material it was never authorized to use.
FINAL_WIRE_CODES = frozenset(
    {
        "security_boundary",
        "check_outside_branch",
        "citation_outside_branch",
    }
)


def precise_answer_schema(request) -> dict:
    """The model-visible schema of the v3 answer, narrowed to this branch's vocabulary.

    The duty ids, the witness field names and the authorized source references are derived
    from the request itself, so the schema can only offer what this branch is allowed to
    answer with.
    """

    schema = PreciseAnswerEnvelopeV3.model_json_schema()
    schema["properties"]["findings"]["minItems"] = len(request.checks)
    schema["properties"]["findings"]["maxItems"] = len(request.checks)
    schema["$defs"]["PreciseAnswerWitnessV3"]["properties"]["source_ref"]["enum"] = [
        source.ref for source in request.sources
    ]
    schema["$defs"]["PreciseAnswerWitnessV3"]["properties"]["field"]["enum"] = sorted(
        {
            name
            for model in VALUE_MODELS_V31[request.worker].values()
            for name in model.model_fields
        }
    )
    for name in ("PreciseAnswerFindingV3", "PreciseAnswerWitnessV3"):
        schema["$defs"][name]["properties"]["obligation_id"]["enum"] = list(
            VALUE_MODELS_V31[request.worker]
        )
    return schema


def precise_wire_fingerprint() -> str:
    """The digest of what the model can see: the schema name, the prompt and the registry.

    A Run binds this digest in its frozen snapshot, so a prompt or schema change is a new
    candidate and an execution under the old snapshot stops instead of re-running the Case
    against instructions it was not created with.
    """

    return hashlib.sha256(
        json.dumps(
            {
                "wire": PRECISE_WIRE_SCHEMA,
                "answer": PRECISE_ANSWER_VERSION,
                "value_limit": PRECISE_VALUE_LIMIT,
                "prompt": PRECISE_SYSTEM_PROMPT,
                "registry": {
                    worker: {
                        duty: sorted(model.model_fields)
                        for duty, model in models.items()
                    }
                    for worker, models in VALUE_MODELS_V31.items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def precise_payload(
    config: FEndpointConfig,
    request,
    correction: str | None = None,
    correction_fields: tuple[str, ...] = (),
    correction_details: tuple[str, ...] = (),
    correction_kind: str = "contract",
) -> dict:
    """One versioned request body: the 3.1.0 schema, prompt and source spans."""

    payload = request_payload(config, request)
    schema = precise_answer_schema(request)
    if config.prompt_style == "compact":

        def compact(value):
            if isinstance(value, dict):
                return {
                    key: compact(child)
                    for key, child in value.items()
                    if key not in {"title", "description"}
                }
            return (
                [compact(child) for child in value]
                if isinstance(value, list)
                else value
            )

        schema = compact(schema)
    system = PRECISE_SYSTEM_PROMPT
    if config.prompt_style == "baseline" or config.mode != "json_schema":
        system += "\n输出Schema：" + json.dumps(
            schema, ensure_ascii=False, separators=(",", ":")
        )
    user = {
        "worker": request.worker,
        "question": request.question,
        "checks": [
            {
                "obligation_id": check.proposition.check_id,
                "instruction": check.instruction,
            }
            for check in request.checks
        ],
        "sources": [
            {
                "ref": source.ref,
                "content": source.content,
                "unicode_length": len(source.content),
                "span_options": source_spans(source.content),
            }
            for source in request.sources
        ],
    }
    payload["messages"] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(user, ensure_ascii=False, separators=(",", ":")),
        },
    ]
    if correction:
        # The correction names fixed codes, declared fields and bounded location tokens
        # only: the refused answer is never fed back as instructions.
        located = (
            "；定位：" + ",".join(correction_details) if correction_details else ""
        )
        if correction_kind == "citation":
            instruction = (
                "上次输出中这些字段的引用不被其来源支持；请只重做这些字段："
                "值必须逐字复制原文的完整子句，并引用包含该子句的原文区间；"
                "原文没有的维度整个省略该键；其余字段与结论保持不变。错误码："
            )
        else:
            instruction = "上次输出合同无效；请重做，错误码："
        payload["messages"].append(
            {
                "role": "system",
                "content": instruction
                + correction
                + "；固定字段："
                + ",".join(correction_fields)
                + located,
            }
        )
    if config.mode == "json_schema":
        payload["response_format"]["json_schema"].update(
            name="finding_evidence_v2", schema=schema
        )
    return payload


@dataclass(frozen=True)
class PreciseRunAttemptResult(AttemptResult):
    """One physical attempt plus the versioned evidence the boundary accepted.

    ``evidence`` is the decoded document and witnesses the admission window produced:
    decoding happens once, inside the boundary that admitted the response, so the runner
    grounds values instead of parsing the accepted body a second time.
    """

    evidence_content: str | None = None
    evidence: tuple | None = None
    #: The bounded security events this attempt's own boundary produced.  A refused output
    #: is a security fact with a fixed category and a location identity, so the decision can
    #: escalate to a human path instead of reading the refusal as a mere pipeline failure.
    security_events: tuple = ()


class PreciseRunChatTransport(ChatTransportV2):
    """The versioned product transport; one instance serves one execution."""

    def __init__(
        self,
        config: FEndpointConfig,
        credential: SecretStr,
        *,
        catalog: SourceCatalogV3,
        protection: ProtectedOutputV3 | None = None,
        transport=None,
    ) -> None:
        super().__init__(config, credential, transport=transport)
        self.catalog = catalog
        self.protection = protection or ProtectedOutputV3()

    async def send_admitted(
        self,
        payload: dict,
        request,
        seconds: float,
        *,
        admit=None,
    ) -> AttemptResult:
        """One physical send whose response is admitted before anything decodes it.

        The response body is protected material: it carries the authorized sources quoted
        back and the branch's own extracted values.  The caller passes the consumption
        admission of the live generation, and it runs between the receipt and the first
        decode, so a generation that lost its authority can no longer have the body
        parsed, grounded or persisted.  A transport-level failure is already settled and
        has no body to admit.
        """

        receipt = await self.post_once(payload, seconds)
        if isinstance(receipt, AttemptResult):
            return receipt
        if admit is not None:
            admit()
        return self.decode(receipt, request, self.credential.get_secret_value())

    def decode(self, response, request, credential) -> PreciseRunAttemptResult:
        accepted: list[str] = []
        evidence: list[tuple] = []
        blocked: set[str] = set()

        def guard(value, credential=""):
            try:
                self.protection.assert_safe(value, credential)
            except WireIssue as issue:
                blocked.update(issue.details)
                raise

        def parse(content, branch):
            try:
                strict_json(content)
            except ValueError:
                raise WireIssue("invalid_json", correctable=True) from None
            try:
                document, witnesses = parse_precise_answer(
                    content, branch, protection=self.protection
                )
            except WireIssue as issue:
                if issue.code in FINAL_WIRE_CODES:
                    raise
                # A refused envelope is a correctable format problem; the fields and the
                # bounded location tokens the refusal named travel with it so the one
                # correction can target them.
                raise WireIssue(
                    "invalid_schema",
                    correctable=True,
                    fields=issue.fields,
                    details=issue.details,
                ) from issue
            except (ValueError, ValidationError):
                raise WireIssue(
                    "invalid_schema", correctable=True, fields=("witnesses",)
                ) from None
            accepted.append(content)
            evidence.append((document, witnesses))
            return document.findings

        result = decode_response(
            response,
            request,
            self.config,
            credential,
            output_guard=guard,
            parse_output=parse,
        )
        return PreciseRunAttemptResult(
            **vars(result),
            evidence_content=accepted[0] if accepted else None,
            evidence=evidence[0] if evidence else None,
            security_events=tuple(
                event(code, "output", request.worker) for code in sorted(blocked)
            ),
        )


__all__ = [
    "FINAL_WIRE_CODES",
    "PRECISE_SYSTEM_PROMPT",
    "PRECISE_WIRE_SCHEMA",
    "PreciseRunAttemptResult",
    "PreciseRunChatTransport",
    "precise_answer_schema",
    "precise_payload",
    "precise_wire_fingerprint",
]
