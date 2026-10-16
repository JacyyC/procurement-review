"""Rules 3.1.0 model-visible schema, prompt and source validation.

What the two document Workers are asked for under the conditional rule set: the
Policy branch declares the sourced quote rule (threshold, categories in scope,
the exclusion with its basis, and the registered exception with its required
evidence), and the Contract branch verifies the transaction category with its
basis and the commitment facts.  Neither branch is asked to restate the other's
material: the rule comes from the policy text, the transaction from the contract
text, and the deterministic facts from the read-only tools.

The schema is deliberately flat and witness-able: every value a Worker reports is
a scalar or a compact JSON list of registry words, and every reported value must
be bound back to a clause of its own authorized source before it is used.
"""

from __future__ import annotations

import json
import re

from pydantic import Field, model_validator

from .business_v3_conditional import (
    CategoryExclusionV31,
    ContractTransactionV31,
    ExceptionKind,
    ExceptionRequirement,
    QuoteRuleV31,
    RuleExceptionV31,
    TransactionCategory,
)
from .business_v3_documents import (
    DocumentValuesV3,
    _compact_schema,
    decode_document_values,
)
from .business_v3_types import (
    VALUE_MODELS,
    Count,
    ScopeTerms,
    Short,
)
from .business_v3_types import BusinessPropositionIdentity as PropositionIdentity
from .eval_v2_contracts import EvalV2Model
from .model_wire_v2 import BranchRequestV2, CheckSlot

_EXCLUSION_MARKER = re.compile(r"不适用|排除|excluded|not\s+applicable", re.I)
_EXCEPTION_MARKER = re.compile(r"例外|豁免|经.{0,6}可|exception|exempt", re.I)

#: Fields the closed registries require *together*: declaring the first without the rest
#: makes the whole duty's value unreadable, so the requirement is stated where the value
#: is asked for instead of only in the prose rules.  A test keeps this table in step with
#: the cross-field validators.
CO_REQUIRED_FIELDS = {
    ("policy", "O-04"): (
        ("exceptions", ("exception_requires", "exception_basis")),
        ("excluded_categories", ("exclusion_basis",)),
    )
}

#: One shape-only example per duty that needs one.  The values are deliberately unrelated
#: to any Case; only the shape is shown, and the task text says so.
SHAPE_EXAMPLES = {
    ("policy", "O-04"): (
        "示例（仅示形，数值/类别/条款必须逐字取自本次原文；此例的5家与服务类不得照抄）："
        '{"minimum_quotes":5,"applies_to":["service"],"excluded_categories":["lease"],'
        '"exclusion_basis":"<逐字复制写出该排除的完整子句>","exceptions":["emergency"],'
        '"exception_requires":["written_justification","approval_reference"],'
        '"exception_basis":"<逐字复制登记该例外的完整子句>"}；'
        "原文没有排除时省略excluded_categories与exclusion_basis，没有登记例外时省略后三项，"
        "绝不能只写其中一半。"
    ),
}

#: Pairs of fields whose declarations must travel together: the closed registries refuse
#: the whole duty's value when only one half is declared, so the requirement is stated in
#: the schema the model reads, not only in prose.
CO_REQUIRED_FIELDS = {
    ("policy", "O-04"): (
        ("exceptions", ("exception_requires", "exception_basis")),
        ("excluded_categories", ("exclusion_basis",)),
    )
}


def _with_co_requirements(schema: dict, requirements) -> dict:
    """Add the closed registries' together-declared requirements to one duty's schema."""

    if not requirements:
        return schema
    return {
        **schema,
        "allOf": [
            {
                "if": {
                    "properties": {field: {"minItems": 1}},
                    "required": [field],
                },
                "then": {"required": list(requires)},
            }
            for field, requires in requirements
        ],
    }


CATEGORY_PHRASES = {
    "purchase": r"采购|purchase",
    "lease": r"租赁|lease",
    "service": r"服务|service",
    "framework_call_off": r"框架|framework",
}
EXCEPTION_PHRASES = {
    "single_source": r"单一来源|唯一来源|single[\s-]?source",
    "emergency": r"紧急|应急|emergency",
    "existing_contract_extension": r"续签|合同延期|extension",
}
REQUIREMENT_PHRASES = {
    "written_justification": r"书面说明|书面理由|written\s+justification",
    "approval_reference": r"审批|批准|授权|approval",
    "price_reasonableness": r"价格合理|报价合理|price\s+reasonableness",
}


class QuoteRuleValuesV31(EvalV2Model):
    """The sourced competitive-quote rule as the Policy branch reports it."""

    minimum_quotes: Count
    applies_to: tuple[TransactionCategory, ...] = Field(min_length=1, max_length=4)
    excluded_categories: tuple[TransactionCategory, ...] = Field(
        default=(), max_length=4
    )
    exclusion_basis: Short | None = None
    exceptions: tuple[ExceptionKind, ...] = Field(default=(), max_length=3)
    exception_requires: tuple[ExceptionRequirement, ...] = Field(
        default=(), max_length=3
    )
    exception_basis: Short | None = None

    @model_validator(mode="after")
    def each_claim_carries_its_basis(self):
        if len(set(self.excluded_categories)) != len(self.excluded_categories):
            raise ValueError("one category cannot be excluded twice")
        if set(self.excluded_categories) & set(self.applies_to):
            raise ValueError("a category cannot be both in scope and excluded")
        if self.excluded_categories and self.exclusion_basis is None:
            raise ValueError("an exclusion requires its policy basis")
        if not self.excluded_categories and self.exclusion_basis is not None:
            raise ValueError("an exclusion basis without an exclusion")
        if len(set(self.exceptions)) != len(self.exceptions):
            raise ValueError("one exception kind cannot be registered twice")
        if self.exceptions and (
            not self.exception_requires or not self.exception_basis
        ):
            raise ValueError("a registered exception requires its evidence and basis")
        if not self.exceptions and (self.exception_requires or self.exception_basis):
            raise ValueError("exception requirements without a registered exception")
        return self

    def to_quote_rule(self) -> QuoteRuleV31:
        return QuoteRuleV31(
            minimum_quotes=self.minimum_quotes,
            applies_to=self.applies_to,
            excluded=tuple(
                CategoryExclusionV31(category=category, basis=self.exclusion_basis)
                for category in self.excluded_categories
            ),
            exceptions=tuple(
                RuleExceptionV31(
                    kind=kind,
                    requires=self.exception_requires,
                    basis=self.exception_basis,
                )
                for kind in self.exceptions
            ),
        )


class ContractTransactionValuesV31(EvalV2Model):
    """The transaction facts the Contract branch verified from its own material."""

    category: TransactionCategory
    category_basis: Short
    single_source: bool | None = None
    written_justification: bool | None = None
    approval_reference: Short | None = None
    price_reasonableness: bool | None = None
    commitment_current: bool | None = None
    superseding_amendment: Short | None = None

    def to_transaction(self) -> ContractTransactionV31:
        return ContractTransactionV31(
            category=self.category,
            category_basis=self.category_basis,
            single_source=self.single_source,
            written_justification=self.written_justification,
            approval_reference=self.approval_reference,
            price_reasonableness=self.price_reasonableness,
            commitment_current=self.commitment_current,
            superseding_amendment=self.superseding_amendment,
        )


#: The rules 3.1.0 value registry.  O-01/O-02 keep the frozen shapes; the rule and
#: the transaction are new duty slots of their own, so neither frozen slot changes.
VALUE_MODELS_V31 = {
    "policy": {
        "O-01": ScopeTerms,
        "O-02": VALUE_MODELS["policy"]["O-02"],
        "O-04": QuoteRuleValuesV31,
    },
    "contract": {
        "O-04": ContractTransactionValuesV31,
        "O-05": VALUE_MODELS["contract"]["O-05"],
        "O-06": VALUE_MODELS["contract"]["O-06"],
        "O-08": VALUE_MODELS["contract"]["O-08"],
        "O-09": VALUE_MODELS["contract"]["O-09"],
    },
}


def document_request_v31(context, *, case_id, worker, question, sources):
    """One isolated extraction task whose schema names the conditional duties."""

    if worker not in VALUE_MODELS_V31:
        raise ValueError("procurement does not have a model task")
    checks = []
    for obligation in VALUE_MODELS_V31[worker]:
        checks.append(
            CheckSlot(
                proposition=PropositionIdentity(
                    scope_id=context.scope_id,
                    subject_type="purchase_order",
                    subject_id=context.purchase_order_id,
                    check_id=obligation,
                    field_path=f"{worker}.{obligation.lower().replace('-', '_')}",
                    effective_at=context.as_of,
                    source_version="typed-business-values.v2",
                ),
                instruction=(
                    "提取本项具体条款，normalized_value须为下述Schema对象的JSON字符串；"
                    "缺实际材料即unknown，不用其它分支材料补齐；不得判整个Case。"
                ),
            )
        )
    schemas = {
        duty: _with_co_requirements(
            _compact_schema(model), CO_REQUIRED_FIELDS.get((worker, duty), ())
        )
        for duty, model in VALUE_MODELS_V31[worker].items()
    }
    task = (
        question
        + "\n本分支仅作文本提取；其它分支各自核对采购事实或条款，聚合层才跨域比较。"
        "版本为rules 3.1.0：阈值来自政策原文，适用类别、排除与例外都必须有原文依据。"
        "O-01的applies按本Case范围/日期判定；O-02的special须回答问题点名的专项具体值/清单/期限或明确豁免依据，泛称有条款无效。"
        "O-04对政策分支是比价规则：minimum_quotes取原文阈值；applies_to列出规则明文适用的交易类别；"
        "excluded_categories只列被原文明确排除的类别，exclusion_basis引用该排除句；"
        "exceptions只列政策自身登记过的例外种类，exception_requires逐项列出该例外要求的证据，exception_basis引用登记句。"
        "类别词表为purchase/lease/service/framework_call_off，例外种类表为single_source/emergency/existing_contract_extension，"
        "要求词表为written_justification/approval_reference/price_reasonableness；未登记的词不得使用，缺失维度保持unknown。"
        "O-04对合同分支是本次交易事实：category只在合同原文明确类别时填写并给出category_basis引用该句；"
        "single_source/written_justification/price_reasonableness只在原文明确陈述时填写，否则留empty；"
        "approval_reference与superseding_amendment只在原文出现对应引用时填写；commitment_current按原文是否仍有效填写。"
        "合同不重新证明政策是否存在，也不复述政策文字；政策不猜交易类别。"
        "O-05提取双方标的引用；O-06提取价款币种；O-08提取时限、标准责任；O-09提取书面变更、赔付和上限，other覆盖问题/政策声明适用的分包、知识产权、争议等内容或明确不适用依据。"
        "数字天数不代表标准/责任完整。只有实际原文支持提取时supported；明确违反本分支条款义务时violated；缺实际材料unknown。"
        "只有O-01可not_applicable且applies=false，exempt逐项列明原文豁免的义务，不适用不得自动豁免其它事项；其它豁免留明确文字依据。"
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


def decode_document_values_v31(request, findings) -> DocumentValuesV3:
    """Parse one 3.1.0 answer into the typed rule and transaction values."""

    return decode_document_values(request, findings, registry=VALUE_MODELS_V31)


def _json_words(value, vocabulary) -> list[str] | None:
    """The registry words one compact JSON list claims, or ``None`` if malformed."""

    if not isinstance(value, str):
        return None
    try:
        words = json.loads(value)
    except ValueError:
        return None
    if not isinstance(words, list) or not words:
        return None
    if any(not isinstance(word, str) or word not in vocabulary for word in words):
        return None
    return sorted(set(words))


def _scope_words_visible(span, words, marker) -> bool:
    """Each claimed category/kind must be named next to the rule's own marker."""

    return bool(re.search(marker, span, re.I)) and all(
        re.search(
            CATEGORY_PHRASES.get(word, EXCEPTION_PHRASES.get(word, "")), span, re.I
        )
        for word in words
    )


def _bool_claim(span, subject, claimed, positive, negative) -> bool:
    """A boolean is only supported by the explicit wording of its own direction.

    A negative claim needs the material to say so in its own words; silence is not
    support for either direction.
    """

    if not re.search(subject, span, re.I):
        return False
    return bool(re.search(positive if claimed else negative, span, re.I))


def supports_field_v31(worker, obligation, field, value, span) -> bool:
    """Conservative clause normalizers for the rules 3.1.0 fields.

    The frozen 3.0.0 recognizers stay untouched and are delegated to for the
    duties both versions share; the conditional fields are judged here, and an
    unrecognized wording returns False instead of being read as support.
    """

    if obligation != "O-04":
        from .business_v3_support import supports_field

        return supports_field(worker, obligation, field, value, span)
    if worker == "policy":
        if field == "minimum_quotes":
            numbers = {
                int(m[1])
                for m in re.finditer(
                    r"(?:报价|quotes?)\s*(?:至少|不少于|>=|at\s+least)\s*(\d+)(?!\d|[./]\d|[eE][+-]?\d|\s*(?:to\b|[-–]|or\b)\s*\d)",
                    span,
                    re.I,
                )
            }
            return type(value) is int and numbers == {value}
        if field == "applies_to":
            words = _json_words(value, CATEGORY_PHRASES)
            return words is not None and _scope_words_visible(
                span, words, r"适用于|适用范围|applies\s+to"
            )
        if field == "excluded_categories":
            if value == "[]":
                # An empty exclusion list is a claim about this clause too: it is
                # supported only by the absence of exclusion wording, exactly as the
                # frozen scope terms treat their empty exemption list.
                return not _EXCLUSION_MARKER.search(span)
            words = _json_words(value, CATEGORY_PHRASES)
            return words is not None and _scope_words_visible(
                span, words, _EXCLUSION_MARKER.pattern
            )
        if field == "exclusion_basis":
            return (
                isinstance(value, str)
                and value in span
                and bool(_EXCLUSION_MARKER.search(span))
            )
        if field == "exceptions":
            if value == "[]":
                return not _EXCEPTION_MARKER.search(span)
            words = _json_words(value, EXCEPTION_PHRASES)
            return words is not None and _scope_words_visible(
                span, words, _EXCEPTION_MARKER.pattern
            )
        if field == "exception_requires":
            if value == "[]":
                return not _EXCEPTION_MARKER.search(span)
            words = _json_words(value, REQUIREMENT_PHRASES)
            return (
                words is not None
                and bool(_EXCEPTION_MARKER.search(span))
                and all(
                    re.search(REQUIREMENT_PHRASES[word], span, re.I) for word in words
                )
            )
        if field == "exception_basis":
            return (
                isinstance(value, str)
                and value in span
                and bool(_EXCEPTION_MARKER.search(span))
            )
        return False
    if worker == "contract":
        if field == "category":
            return (
                isinstance(value, str)
                and value in CATEGORY_PHRASES
                and bool(re.search(CATEGORY_PHRASES[value], span, re.I))
            )
        if field == "category_basis":
            return (
                isinstance(value, str)
                and value in span
                and bool(re.search(r"本次|本合同|类别|category", span, re.I))
            )
        if field == "single_source":
            return type(value) is bool and _bool_claim(
                span,
                r"来源|供应商|supplier|source",
                value,
                r"单一来源|唯一来源|single[\s-]?source",
                r"非单一来源|非唯一来源|not\s+single[\s-]?source",
            )
        if field == "written_justification":
            return type(value) is bool and _bool_claim(
                span,
                r"说明|理由|justification",
                value,
                r"书面说明|书面理由|written\s+justification",
                r"无书面|没有书面|no\s+written",
            )
        if field == "approval_reference":
            return (
                isinstance(value, str)
                and value in span
                and bool(re.search(r"审批|批准|授权|approval", span, re.I))
            )
        if field == "price_reasonableness":
            return type(value) is bool and _bool_claim(
                span,
                r"价格|报价|price",
                value,
                r"价格合理|报价合理|price\s+reasonableness|reasonable",
                r"价格不合理|报价不合理|not\s+reasonable|unreasonable",
            )
        if field == "commitment_current":
            return type(value) is bool and _bool_claim(
                span,
                r"质保|交付|承诺|warranty|delivery|commitment",
                value,
                r"现行|仍然有效|继续有效|in\s+force|current",
                r"已被.{0,8}替代|不再有效|终止|superseded|no\s+longer",
            )
        if field == "superseding_amendment":
            return (
                isinstance(value, str)
                and value in span
                and bool(re.search(r"替代|补充协议|变更协议|amendment", span, re.I))
            )
        return False
    return False


__all__ = [
    "CATEGORY_PHRASES",
    "EXCEPTION_PHRASES",
    "REQUIREMENT_PHRASES",
    "ContractTransactionValuesV31",
    "QuoteRuleValuesV31",
    "VALUE_MODELS_V31",
    "decode_document_values_v31",
    "document_request_v31",
    "supports_field_v31",
]
