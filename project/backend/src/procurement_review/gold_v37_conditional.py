"""Independent rules 3.1.0 DEV material, its own oracle and a bounded truth file.

The eight cases exercise the conditional handover on purpose: a sourced lease
exclusion, a missing category, a category disagreement, an unverified and a
verified single-source exception, both threshold outcomes, and a superseded
commitment.  Every case is built from one existing DEV fact set (so the other
duties stay decidable) while its policy and contract wording is authored here and
differs case by case, so no two documents are the same text.

The oracle answers only from the emitted text and the facts — it re-decides the
two conditional duties in its own words and never imports the production rules —
and the generator's declared expectation is compared against it, so the two must
agree before either can be used as a reference.
"""

from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import Field

from .eval_f_context import load_cases
from .eval_v2_contracts import EvalV2Model

CONDITIONAL_SET_VERSION = "gold-set.v3.7"
CONDITIONAL_TRUTH_VERSION = "v1-07-truth.v1.7"
CONDITIONAL_ORACLE_VERSION = "conditional-oracle.v1"
SUPPORTED_DEV_VERSION = "gold-set.v3.1"
SET_SIZE = 8

_DOMAIN_NOUNS = (
    "中部验证区办公设备",
    "东部试点区检测装置",
    "南部联调区计量仪表",
    "北部实训区监控终端",
    "西部巡检区测绘器材",
    "沿海实验区通信模块",
    "内陆仓储区包装机械",
    "高原作业区供电机组",
)


class ConditionalCaseV1(EvalV2Model):
    """One authored conditional case: its material, its facts and its expectation."""

    case_id: str
    split: Literal["dev-conditional"] = "dev-conditional"
    language: Literal["zh-CN"] = "zh-CN"
    generated_at: str = "2026-10-06"
    question: str
    policy_text: str
    contract_text: str
    facts: dict[str, Any]
    #: The read-only dimension rules 3.1.0 adds to the order fact; absent means the
    #: MCP side states no category, which the aggregator must keep UNKNOWN.
    transaction_category: str | None = None
    expected: dict[str, str]
    expected_reasons: dict[str, str]
    expected_missing: dict[str, tuple[str, ...]] = Field(default_factory=dict)


class ConditionalTruthV1(EvalV2Model):
    """The oracle's own answer for one case, plus the identity of the material."""

    case_id: str
    truth_contract: Literal["v1-07-truth.v1.7"] = CONDITIONAL_TRUTH_VERSION
    oracle_version: Literal["conditional-oracle.v1"] = CONDITIONAL_ORACLE_VERSION
    policy_sha256: str
    contract_sha256: str
    obligations: dict[str, str]
    reasons: dict[str, str]


class FieldSupportV1(EvalV2Model):
    """How one claimed field is verified against the material, independently.

    ``literal``: the claimed value must equal the material's own value *and* that value
    has to appear in the cited excerpt (the claim is a quotation of the source).
    ``token``: the claim is a closed-form token the material determines (a rule word, a
    boolean, a list); the excerpt has to carry the ``needles`` that state it.
    ``needles``: free text judged by required substrings on both the claim and the cited
    excerpt, so a fabricated sentence is not supported by a clause that never says it.
    """

    kind: Literal["literal", "token", "needles"]
    value: Any = None
    needles: tuple[str, ...] = ()


def _fact_case():
    return load_cases("dev", SUPPORTED_DEV_VERSION)[0]


def _base_policy(facts, noun: str) -> list[str]:
    return [
        f"{noun}采购管理办法（2026年度）",
        "本办法适用于本项目在中部验证区发生的采购活动，有效期为2026-01-01至2026-12-31。",
        "供应商要求：登记状态为registry: active；采购状态为procurement: approved；税号已登记；尽调due_diligence: verified；筛查screening: clear。",
        "订单与发票等额；交易累计不得超过订单；订单、发票、交易币种必须相同。",
        "记账日期不得早于下单日期。",
        "交付不超过15日；验收不超过7日；质保至少90日。",
    ]


def _quote_rules(
    *,
    excludes_lease: bool,
    registers_single_source: bool,
    noun: str,
) -> list[str]:
    rules = [f"{noun}类交易报价规则：竞争性报价至少3家，适用于采购与服务类交易。"]
    if excludes_lease:
        rules.append(
            f"租赁类交易不适用竞争性报价要求（{noun}租赁按单一承租人方式执行）。"
        )
    if registers_single_source:
        rules.append(
            "唯一来源采购属于本办法登记的例外：须提供书面说明与审批授权，"
            "经批准后方可不适用竞争性报价要求。"
        )
    return rules


def _contract(
    facts,
    *,
    noun: str,
    category_line: str,
    single_source_line: str | None,
    justification_line: str | None,
    approval_line: str | None,
    commitment_line: str | None,
    supersede_line: str | None,
) -> str:
    amount = f"{facts.order_amount_minor // 100}.{facts.order_amount_minor % 100:02}"
    lines = [
        f"{noun}合同（双方签署）",
        f"买方BUYER: {facts.counterparty_reference}；卖方SELLER: {facts.supplier_id}；标的SUBJECT: {noun}一批。",
        f"订单ORDER: {facts.purchase_order_id}；合同CONTRACT: {facts.contract_reference}。",
        f"价款记录: 订单 {amount} {facts.order_currency}。",
        f"交付DELIVERY {facts.delivery_days} 日；验收ACCEPTANCE {facts.acceptance_days} 日；质保WARRANTY {facts.warranty_days} 日。",
        "验收标准STANDARD: 全部设备通过约定检验。",
        "质保责任WARRANTY_RESPONSIBILITY: 卖方自费维修故障设备。",
        "变更须双方书面签署后生效。",
        "赔付DAMAGES: 逾期交付按订单金额每日0.6%赔付。",
        "责任上限CAP: 84000.00 CNY。",
        category_line,
    ]
    for line in (
        single_source_line,
        justification_line,
        approval_line,
        commitment_line,
        supersede_line,
    ):
        if line:
            lines.append(line)
    lines.append(
        "本次只审变更、赔付与责任上限事项，不增加分包、知识产权或争议程序检查。"
    )
    return "\n".join(lines)


def conditional_cases() -> tuple[ConditionalCaseV1, ...]:
    """The eight authored cases, each with its own wording and its own expectation."""

    source = _fact_case()
    facts = source.model_input.business_facts
    facts = facts.model_copy(
        update={
            "contract_reference": f"{facts.contract_reference}-C",
            "counterparty_reference": f"{facts.counterparty_reference}-C",
        }
    )
    base = facts.model_dump(mode="json")
    cases: list[ConditionalCaseV1] = []
    plans = (
        (
            "V107-D-9001",
            {"excludes_lease": True, "registers": True},
            {
                "category": "租赁类交易（本次合同以设备租赁方式执行）。",
                "single_source": None,
                "justification": None,
                "approval": None,
                "commitment": None,
                "supersede": None,
            },
            "lease",
            {"competitive_quote_count": 1},
            {"O-04": "not_applicable"},
            {"O-04": "category_excluded_by_policy"},
            {},
        ),
        (
            "V107-D-9002",
            {"excludes_lease": True, "registers": True},
            {
                "category": None,
                "single_source": None,
                "justification": None,
                "approval": None,
                "commitment": None,
                "supersede": None,
            },
            None,
            {"competitive_quote_count": 1},
            {"O-04": "unknown"},
            {"O-04": "transaction_category_missing"},
            {"O-04": ("sourced_transaction_category",)},
        ),
        (
            "V107-D-9003",
            {"excludes_lease": True, "registers": True},
            {
                "category": "租赁类交易（本次合同以设备租赁方式执行）。",
                "single_source": None,
                "justification": None,
                "approval": None,
                "commitment": None,
                "supersede": None,
            },
            "purchase",
            {"competitive_quote_count": 1},
            {"O-04": "unknown"},
            {"O-04": "transaction_category_conflict"},
            {"O-04": ("agreed_transaction_category",)},
        ),
        (
            "V107-D-9004",
            {"excludes_lease": False, "registers": True},
            {
                "category": "采购类交易（本次合同以设备采购方式执行）。",
                "single_source": "本合同为单一来源采购。",
                "justification": "书面说明: 该型号为唯一可兼容的供应商产品。",
                "approval": None,
                "commitment": None,
                "supersede": None,
            },
            "purchase",
            {"competitive_quote_count": 1},
            {"O-04": "unknown"},
            {"O-04": "exception_requirements_unverified"},
            {"O-04": ("approval_reference",)},
        ),
        (
            "V107-D-9005",
            {"excludes_lease": False, "registers": True},
            {
                "category": "采购类交易（本次合同以设备采购方式执行）。",
                "single_source": "本合同为单一来源采购。",
                "justification": "书面说明: 该型号为唯一可兼容的供应商产品。",
                "approval": "审批授权: APPROVAL-2026-014。",
                "commitment": None,
                "supersede": None,
            },
            "purchase",
            {"competitive_quote_count": 1},
            {"O-04": "not_applicable"},
            {"O-04": "registered_exception_met"},
            {},
        ),
        (
            "V107-D-9006",
            {"excludes_lease": False, "registers": True},
            {
                "category": "采购类交易（本次合同以设备采购方式执行）。",
                "single_source": None,
                "justification": None,
                "approval": None,
                "commitment": None,
                "supersede": None,
            },
            "purchase",
            {"competitive_quote_count": 2},
            {"O-04": "violated"},
            {"O-04": "quote_threshold_violated"},
            {},
        ),
        (
            "V107-D-9007",
            {"excludes_lease": False, "registers": True},
            {
                "category": "采购类交易（本次合同以设备采购方式执行）。",
                "single_source": None,
                "justification": None,
                "approval": None,
                "commitment": None,
                "supersede": None,
            },
            "purchase",
            {"competitive_quote_count": 3},
            {"O-04": "supported"},
            {"O-04": "quote_threshold_met"},
            {},
        ),
        (
            "V107-D-9008",
            {"excludes_lease": False, "registers": True},
            {
                "category": "采购类交易（本次合同以设备采购方式执行）。",
                "single_source": None,
                "justification": None,
                "approval": None,
                "commitment": None,
                "supersede": "补充协议二将质保期改为30日，原质保承诺已被补充协议替代。",
            },
            "purchase",
            {"competitive_quote_count": 3},
            {"O-04": "supported", "O-08": "unknown"},
            {"O-04": "quote_threshold_met", "O-08": "commitment_not_current"},
            {"O-08": ("current_binding_commitment",)},
        ),
    )
    for index, (
        case_id,
        rules,
        contract,
        category,
        order,
        expected,
        reasons,
        missing,
    ) in enumerate(plans):
        noun = _DOMAIN_NOUNS[index]
        policy_text = "\n".join(
            (
                *_base_policy(facts, noun),
                *_quote_rules(
                    excludes_lease=rules["excludes_lease"],
                    registers_single_source=rules["registers"],
                    noun=noun,
                ),
            )
        )
        contract_text = _contract(
            facts,
            noun=noun,
            category_line=contract["category"] or "本次交易类别未在合同中载明。",
            single_source_line=contract["single_source"],
            justification_line=contract["justification"],
            approval_line=contract["approval"],
            commitment_line=contract["commitment"],
            supersede_line=contract["supersede"],
        )
        case_facts = dict(base)
        case_facts.update(order)
        cases.append(
            ConditionalCaseV1(
                case_id=case_id,
                question=f"请审查{noun}采购记录，逐项核对资格、引用、金额币种、日期、履约和安全边界；缺失事实只能标为未知。",
                policy_text=policy_text,
                contract_text=contract_text,
                facts=case_facts,
                transaction_category=category,
                expected=expected,
                expected_reasons=reasons,
                expected_missing=missing,
            )
        )
    return tuple(cases)


# ------------------------------------------------------------- boundary adapters


def as_gold_case(case: ConditionalCaseV1):
    """The authored case as the case shape the shared context helpers read.

    Only the two documents, the question and the typed facts are replaced; the
    identity fields the fact set already carries stay as they are, so the Case
    identity, its scope and its order/supplier/transaction ids remain consistent.
    """

    base = _fact_case()
    documents = {}
    for key, body in (
        ("policy_document", case.policy_text),
        ("contract_document", case.contract_text),
    ):
        documents[key] = getattr(base.model_input, key).model_copy(
            update={
                "content": body,
                "content_sha256": hashlib.sha256(body.encode()).hexdigest(),
            }
        )
    facts_type = type(base.model_input.business_facts)
    facts = facts_type.model_validate_json(json.dumps(case.facts))
    model_input = base.model_input.model_copy(
        update={
            "question": case.question,
            "business_facts": facts,
            **documents,
        }
    )
    return base.model_copy(update={"model_input": model_input})


def conditional_procurement(case: ConditionalCaseV1):
    """The read-only fact answer, with the conditional dimension carried or absent.

    A case whose material states no category leaves the key out entirely: the
    aggregator must read that as a missing sourced category, never as a default.
    """

    from .eval_f_context import oracle_procurement

    base = oracle_procurement(as_gold_case(case))
    facts = []
    for fact in base.facts:
        body = dict(fact.result)
        if fact.tool_name == "get_purchase_order":
            body.pop("transaction_category", None)
            if case.transaction_category is not None:
                body["transaction_category"] = case.transaction_category
        facts.append(SimpleNamespace(**{**vars(fact), "result": body}))
    return SimpleNamespace(
        status=base.status,
        facts=tuple(facts),
        requested_dataset_version=base.requested_dataset_version,
    )


# ------------------------------------------------------------------------ oracle

_CATEGORY_WORDS = {
    "租赁": "lease",
    "采购": "purchase",
    "服务": "service",
    "框架": "framework_call_off",
}

#: The wording each transaction category is carried by, in the contract's own words.
CATEGORY_MARKERS = {"租赁类交易": "lease", "采购类交易": "purchase"}
#: The English tokens the policy's applicability clause is expressed with.
APPLIES_TOKENS = (("采购", "purchase"), ("服务", "service"))


def subject_noun(contract: str) -> str | None:
    match = re.search(r"标的SUBJECT:\s*(.+?)一批", contract)
    return match.group(1) if match else None


def category_clause(contract: str) -> tuple[str, str] | None:
    for marker, token in CATEGORY_MARKERS.items():
        if f"{marker}（本次合同" in contract:
            return token, marker
    return None


def approval_reference(contract: str) -> str | None:
    match = re.search(r"审批授权:\s*([A-Z0-9-]+)", contract)
    return match.group(1) if match else None


def cap_clause(contract: str) -> str | None:
    match = re.search(r"责任上限CAP:\s*(.+?)。", contract)
    return match.group(1) if match else None


def damages_marker(contract: str) -> str | None:
    match = re.search(r"每日([\d.]+%)赔付", contract)
    return match.group(1) if match else None


def policy_rule_fields(policy: str) -> dict[str, FieldSupportV1]:
    """The O-04 policy claims the quote rule states, read off the text alone."""

    rule_line = next((line for line in policy.split("\n") if "报价规则" in line), "")
    applies = [
        token
        for marker, token in APPLIES_TOKENS
        if marker in rule_line.split("适用于")[-1]
    ]
    threshold = re.search(r"至少\s*(\d+)\s*家", policy)
    lease_excluded = bool(re.search(r"租赁类交易不适用竞争性报价要求", policy))
    exception_registered = bool(re.search(r"唯一来源采购属于本办法登记的例外", policy))
    exception_needles = tuple(
        marker for marker in ("登记的例外", "书面说明", "审批授权") if marker in policy
    )
    requires = [
        name
        for marker, name in (
            ("须提供书面说明", "written_justification"),
            ("与审批授权", "approval_reference"),
        )
        if marker in policy
    ]
    fields: dict[str, FieldSupportV1] = {}
    if applies:
        fields["policy.applies_to"] = FieldSupportV1(kind="token", value=applies)
    if threshold is not None:
        fields["policy.minimum_quotes"] = FieldSupportV1(
            kind="literal", value=int(threshold.group(1))
        )
    if exception_registered:
        fields["policy.exceptions"] = FieldSupportV1(
            kind="token", value=["single_source"], needles=exception_needles
        )
        if requires:
            fields["policy.exception_requires"] = FieldSupportV1(
                kind="token", value=requires, needles=exception_needles
            )
        fields["policy.exception_basis"] = FieldSupportV1(
            kind="needles", needles=exception_needles
        )
    else:
        fields["policy.exceptions"] = FieldSupportV1(kind="token", value=[])
    fields["policy.excluded_categories"] = FieldSupportV1(
        kind="token",
        value=["lease"] if lease_excluded else [],
        needles=("不适用竞争性报价要求",) if lease_excluded else (),
    )
    if lease_excluded:
        fields["policy.exclusion_basis"] = FieldSupportV1(
            kind="needles", needles=("不适用竞争性报价要求",)
        )
    return fields


def contract_o04_fields(contract: str) -> dict[str, FieldSupportV1]:
    """The O-04 contract claims the transaction's own wording carries."""

    fields: dict[str, FieldSupportV1] = {}
    category = category_clause(contract)
    if category is not None:
        token, marker = category
        fields["contract.category"] = FieldSupportV1(
            kind="token", value=token, needles=(marker,)
        )
        fields["contract.category_basis"] = FieldSupportV1(
            kind="needles", needles=(marker,)
        )
    if "本合同为单一来源采购" in contract:
        fields["contract.single_source"] = FieldSupportV1(
            kind="token", value=True, needles=("单一来源",)
        )
    if "书面说明:" in contract:
        fields["contract.written_justification"] = FieldSupportV1(
            kind="token", value=True, needles=("书面说明",)
        )
    reference = approval_reference(contract)
    if reference is not None:
        fields["contract.approval_reference"] = FieldSupportV1(
            kind="literal", value=reference
        )
    if re.search(r"已被补充协议替代", contract):
        fields["contract.commitment_current"] = FieldSupportV1(
            kind="token", value=False, needles=("已被补充协议替代",)
        )
        fields["contract.superseding_amendment"] = FieldSupportV1(
            kind="needles", needles=("补充协议", "替代")
        )
    return fields


def derive_conditional_truth(case: ConditionalCaseV1) -> ConditionalTruthV1:
    """Re-decide O-04 and O-08 from the emitted text and the facts, independently."""

    policy = case.policy_text
    contract = case.contract_text
    facts = case.facts
    threshold_match = re.search(r"至少\s*(\d+)\s*家", policy)
    threshold = int(threshold_match.group(1)) if threshold_match else None
    lease_excluded = bool(re.search(r"租赁类交易不适用竞争性报价要求", policy))
    exception_registered = bool(re.search(r"唯一来源采购属于本办法登记的例外", policy))
    requires_justification = bool(re.search(r"须提供书面说明", policy))
    requires_approval = bool(re.search(r"与审批授权", policy))

    contract_category = next(
        (
            word
            for word, phrase in (
                ("lease", "租赁类交易（本次合同"),
                ("purchase", "采购类交易（本次合同"),
            )
            if phrase in contract
        ),
        None,
    )
    fact_category = case.transaction_category
    quote_count = facts.get("competitive_quote_count")
    single_source = "本合同为单一来源采购" in contract
    justification = "书面说明:" in contract
    approval = re.search(r"审批授权:\s*([A-Z0-9-]+)", contract)

    obligations: dict[str, str] = {}
    reasons: dict[str, str] = {}
    if contract_category is not None and fact_category is not None:
        category, conflict = (
            (None, True)
            if contract_category != fact_category
            else (contract_category, False)
        )
    else:
        category, conflict = (contract_category or fact_category), False
    if conflict:
        obligations["O-04"] = "unknown"
        reasons["O-04"] = "transaction_category_conflict"
    elif category is None:
        obligations["O-04"] = "unknown"
        reasons["O-04"] = "transaction_category_missing"
    elif category == "lease" and lease_excluded:
        obligations["O-04"] = "not_applicable"
        reasons["O-04"] = "category_excluded_by_policy"
    elif category not in {"purchase", "service"}:
        obligations["O-04"] = "unknown"
        reasons["O-04"] = "category_outside_registered_rule_scope"
    elif single_source:
        if not exception_registered:
            obligations["O-04"] = "unknown"
            reasons["O-04"] = "registered_exception_rule"
        elif (requires_justification and not justification) or (
            requires_approval and not approval
        ):
            obligations["O-04"] = "unknown"
            reasons["O-04"] = "exception_requirements_unverified"
        else:
            obligations["O-04"] = "not_applicable"
            reasons["O-04"] = "registered_exception_met"
    elif threshold is None or quote_count is None:
        obligations["O-04"] = "unknown"
        reasons["O-04"] = "quote_or_reference_missing"
    elif quote_count < threshold:
        obligations["O-04"] = "violated"
        reasons["O-04"] = "quote_threshold_violated"
    else:
        obligations["O-04"] = "supported"
        reasons["O-04"] = "quote_threshold_met"

    if re.search(r"已被补充协议替代", contract):
        obligations["O-08"] = "unknown"
        reasons["O-08"] = "commitment_not_current"
    elif "质保承诺仍现行有效" in contract:
        obligations["O-08"] = "supported"
        reasons["O-08"] = "delivery_requirements_met"
    return ConditionalTruthV1(
        case_id=case.case_id,
        policy_sha256=hashlib.sha256(policy.encode()).hexdigest(),
        contract_sha256=hashlib.sha256(contract.encode()).hexdigest(),
        obligations=obligations,
        reasons=reasons,
    )


def conditional_truth_file(cases: tuple[ConditionalCaseV1, ...] | None = None) -> dict:
    """The bounded truth document G's DEV diagnosis reads, with its own hashes."""

    chosen = cases if cases is not None else conditional_cases()
    truths = tuple(derive_conditional_truth(case) for case in chosen)
    return {
        "schema_version": CONDITIONAL_TRUTH_VERSION,
        "oracle_version": CONDITIONAL_ORACLE_VERSION,
        "gold_set_version": CONDITIONAL_SET_VERSION,
        "truths": [truth.model_dump(mode="json") for truth in truths],
    }


__all__ = [
    "APPLIES_TOKENS",
    "CATEGORY_MARKERS",
    "CONDITIONAL_ORACLE_VERSION",
    "CONDITIONAL_SET_VERSION",
    "CONDITIONAL_TRUTH_VERSION",
    "SET_SIZE",
    "ConditionalCaseV1",
    "ConditionalTruthV1",
    "FieldSupportV1",
    "approval_reference",
    "cap_clause",
    "category_clause",
    "conditional_cases",
    "conditional_truth_file",
    "contract_o04_fields",
    "damages_marker",
    "derive_conditional_truth",
    "policy_rule_fields",
    "subject_noun",
]
