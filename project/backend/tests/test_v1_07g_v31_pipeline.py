"""Rules 3.1.0 vertical slice: schema, prompt, parse, grounding, truth and report.

Every case is authored independently of the production rules (its expected state
and reason are hand-written and then compared with the oracle in
``gold_v37_conditional``), and the model answers below are composed by hand from
the emitted clauses, so a passing case means the versioned pipeline really carried
the conditional facts from the material to the recorded decision.
"""

from __future__ import annotations

import json

import pytest

from procurement_review.business_v3_procurement import review_procurement_v3
from procurement_review.business_v3_security import ProtectedOutputV3
from procurement_review.business_v3_support import SourceCatalogV3
from procurement_review.business_v31_documents import (
    ContractTransactionValuesV31,
    QuoteRuleValuesV31,
    document_request_v31,
    supports_field_v31,
)
from procurement_review.business_v31_review import (
    finalize_review_v31,
    ground_document_v31,
    parse_evidence_v31,
)
from procurement_review.eval_f_context import context_for, located_sources
from procurement_review.gold_v37_conditional import (
    as_gold_case,
    conditional_cases,
    conditional_procurement,
    conditional_truth_file,
    derive_conditional_truth,
)


def _span(text: str, needle: str) -> tuple[int, int]:
    start = text.index(needle)
    return start, start + len(needle)


def _witness(obligation, field, value, source_ref, start, end):
    return dict(
        obligation_id=obligation,
        field=field,
        value=value,
        source_ref=source_ref,
        start=start,
        end=end,
    )


def _finding(obligation, value, *, status="supported", refs):
    unknown = status == "unknown"
    return dict(
        obligation_id=obligation,
        status=status,
        assertion="unknown" if unknown else "compliant",
        risk_level="unknown" if unknown else "none",
        normalized_value="UNKNOWN"
        if unknown
        else json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        evidence_refs=list(refs),
        missing_information=["material_missing"] if unknown else [],
        statement="原文提取",
        recommended_action="人工核对来源",
    )


def _policy_answer(case, request) -> dict:
    source = request.sources[0]
    text, ref = source.content, source.ref
    lines = text.split("\n")
    scope_line = next(line for line in lines if "适用于" in line)
    supplier_line = next(line for line in lines if "供应商要求" in line)
    money_line = next(line for line in lines if "订单与发票等额" in line)
    dates_line = next(line for line in lines if "记账日期" in line)
    terms_line = next(line for line in lines if "交付不超过" in line)
    rule_line = next(line for line in lines if "报价规则" in line)
    exclusion_line = next(
        (line for line in lines if line.startswith("租赁类交易不适用竞争性报价要求")),
        None,
    )
    exception_line = next((line for line in lines if "登记的例外" in line), None)

    quotes = _finding(
        "O-02",
        {
            "special": terms_line.rstrip("。"),
            "quotes": 3,
            "delivery": 15,
            "acceptance": 7,
            "warranty": 90,
            "supplier": "all_qualified",
            "money": "equal_invoice_capped_tx",
            "dates": "booked_ge_ordered",
        },
        refs=[ref],
    )
    findings = [
        _finding(
            "O-01",
            {"applies": True, "basis": "2026", "exempt": []},
            refs=[ref],
        ),
        quotes,
    ]
    witnesses = [
        _witness("O-01", "applies", True, ref, *_span(text, scope_line)),
        _witness("O-01", "basis", "2026", ref, *_span(text, scope_line)),
        _witness("O-01", "exempt", "[]", ref, *_span(text, scope_line)),
        _witness(
            "O-02", "special", terms_line.rstrip("。"), ref, *_span(text, terms_line)
        ),
        _witness("O-02", "quotes", 3, ref, *_span(text, rule_line)),
        _witness("O-02", "delivery", 15, ref, *_span(text, terms_line)),
        _witness("O-02", "acceptance", 7, ref, *_span(text, terms_line)),
        _witness("O-02", "warranty", 90, ref, *_span(text, terms_line)),
        _witness("O-02", "supplier", "all_qualified", ref, *_span(text, supplier_line)),
        _witness(
            "O-02", "money", "equal_invoice_capped_tx", ref, *_span(text, money_line)
        ),
        _witness("O-02", "dates", "booked_ge_ordered", ref, *_span(text, dates_line)),
    ]

    rule = {
        "minimum_quotes": 3,
        "applies_to": ["purchase", "service"],
        "excluded_categories": [],
        "exceptions": [],
    }
    if exclusion_line is not None:
        rule["excluded_categories"] = ["lease"]
        rule["exclusion_basis"] = "不适用竞争性报价要求"
    if exception_line is not None:
        rule["exceptions"] = ["single_source"]
        rule["exception_requires"] = ["written_justification", "approval_reference"]
        rule["exception_basis"] = "登记的例外"
    findings.append(_finding("O-04", rule, refs=[ref]))
    witnesses.extend(
        (
            _witness("O-04", "minimum_quotes", 3, ref, *_span(text, rule_line)),
            _witness(
                "O-04",
                "applies_to",
                '["purchase","service"]',
                ref,
                *_span(text, rule_line),
            ),
        )
    )
    if exclusion_line is not None:
        witnesses.extend(
            (
                _witness(
                    "O-04",
                    "excluded_categories",
                    '["lease"]',
                    ref,
                    *_span(text, exclusion_line),
                ),
                _witness(
                    "O-04",
                    "exclusion_basis",
                    "不适用竞争性报价要求",
                    ref,
                    *_span(text, exclusion_line),
                ),
            )
        )
    else:
        witnesses.append(
            _witness("O-04", "excluded_categories", "[]", ref, *_span(text, rule_line))
        )
    if exception_line is not None:
        witnesses.extend(
            (
                _witness(
                    "O-04",
                    "exceptions",
                    '["single_source"]',
                    ref,
                    *_span(text, exception_line),
                ),
                _witness(
                    "O-04",
                    "exception_requires",
                    '["written_justification","approval_reference"]',
                    ref,
                    *_span(text, exception_line),
                ),
                _witness(
                    "O-04",
                    "exception_basis",
                    "登记的例外",
                    ref,
                    *_span(text, exception_line),
                ),
            )
        )
    else:
        witnesses.append(
            _witness("O-04", "exceptions", "[]", ref, *_span(text, rule_line))
        )
    return {
        "schema_version": "finding-evidence.v2",
        "findings": findings,
        "witnesses": witnesses,
    }


def _contract_answer(case, request) -> dict:
    source = request.sources[0]
    text, ref = source.content, source.ref
    facts = case.facts
    lines = text.split("\n")
    party_line = next(line for line in lines if "买方BUYER" in line)
    refs_line = next(line for line in lines if "订单ORDER" in line)
    price_line = next(line for line in lines if "价款记录" in line)
    terms_line = next(line for line in lines if "交付DELIVERY" in line)
    standard_line = next(line for line in lines if "验收标准" in line)
    responsibility_line = next(line for line in lines if "质保责任" in line)
    change_line = next(line for line in lines if "书面签署" in line)
    damages_line = next(line for line in lines if "赔付DAMAGES" in line)
    cap_line = next(line for line in lines if "责任上限" in line)
    other_line = next(line for line in lines if "本次只审变更" in line)
    category_line = next(
        line
        for line in lines
        if "本次交易类别未在合同中载明" in line or "类交易（本次合同" in line
    )
    source_line = next((line for line in lines if "单一来源采购" in line), None)
    justification_line = next((line for line in lines if "书面说明:" in line), None)
    approval_line = next((line for line in lines if "审批授权:" in line), None)
    commitment_line = next(
        (line for line in lines if "质保承诺仍现行有效" in line), None
    )
    supersede_line = next((line for line in lines if "已被补充协议替代" in line), None)

    amount = (
        f"{facts['order_amount_minor'] // 100}.{facts['order_amount_minor'] % 100:02}"
    )
    subject = f"{case.question.split('请审查')[1].split('采购记录')[0]}一批"
    findings = [
        _finding(
            "O-05",
            {
                "buyer": facts["counterparty_reference"],
                "seller": facts["supplier_id"],
                "subject": subject,
                "order": facts["purchase_order_id"],
                "contract": facts["contract_reference"],
            },
            refs=[ref],
        ),
        _finding(
            "O-06", {"amount": amount, "currency": facts["order_currency"]}, refs=[ref]
        ),
        _finding(
            "O-08",
            {
                "delivery": facts["delivery_days"],
                "acceptance": facts["acceptance_days"],
                "warranty": facts["warranty_days"],
                "standard": "全部设备通过约定检验",
                "responsibility": "卖方自费维修故障设备",
            },
            refs=[ref],
        ),
        _finding(
            "O-09",
            {
                "written": True,
                "damages": "逾期交付按订单金额每日0.6%赔付",
                "cap": "84000.00 CNY",
                "other": "本次只审变更、赔付与责任上限事项，不增加分包、知识产权或争议程序检查",
            },
            refs=[ref],
        ),
    ]
    witnesses = [
        _witness(
            "O-05",
            "buyer",
            facts["counterparty_reference"],
            ref,
            *_span(text, party_line),
        ),
        _witness("O-05", "seller", facts["supplier_id"], ref, *_span(text, party_line)),
        _witness("O-05", "subject", subject, ref, *_span(text, party_line)),
        _witness(
            "O-05", "order", facts["purchase_order_id"], ref, *_span(text, refs_line)
        ),
        _witness(
            "O-05",
            "contract",
            facts["contract_reference"],
            ref,
            *_span(text, refs_line),
        ),
        _witness("O-06", "amount", amount, ref, *_span(text, price_line)),
        _witness(
            "O-06", "currency", facts["order_currency"], ref, *_span(text, price_line)
        ),
        _witness(
            "O-08", "delivery", facts["delivery_days"], ref, *_span(text, terms_line)
        ),
        _witness(
            "O-08",
            "acceptance",
            facts["acceptance_days"],
            ref,
            *_span(text, terms_line),
        ),
        _witness(
            "O-08", "warranty", facts["warranty_days"], ref, *_span(text, terms_line)
        ),
        _witness(
            "O-08", "standard", "全部设备通过约定检验", ref, *_span(text, standard_line)
        ),
        _witness(
            "O-08",
            "responsibility",
            "卖方自费维修故障设备",
            ref,
            *_span(text, responsibility_line),
        ),
        _witness("O-09", "written", True, ref, *_span(text, change_line)),
        _witness(
            "O-09",
            "damages",
            "逾期交付按订单金额每日0.6%赔付",
            ref,
            *_span(text, damages_line),
        ),
        _witness("O-09", "cap", "84000.00 CNY", ref, *_span(text, cap_line)),
        _witness(
            "O-09",
            "other",
            "本次只审变更、赔付与责任上限事项，不增加分包、知识产权或争议程序检查",
            ref,
            *_span(text, other_line),
        ),
    ]

    # The Contract branch reports the category its own material states; on the
    # disagreement case that is deliberately not the read-only fact's category.
    contract_category = next(
        (
            word
            for word, phrase in (
                ("lease", "租赁类交易（本次合同"),
                ("purchase", "采购类交易（本次合同"),
            )
            if phrase in text
        ),
        None,
    )
    if contract_category is None:
        findings.append(_finding("O-04", None, status="unknown", refs=[ref]))
    else:
        category_basis = (
            "本次合同以设备租赁方式执行"
            if contract_category == "lease"
            else "本次合同以设备采购方式执行"
        )
        transaction = {
            "category": contract_category,
            "category_basis": category_basis,
        }
        if source_line is not None:
            transaction["single_source"] = True
        if justification_line is not None:
            transaction["written_justification"] = True
        if approval_line is not None:
            transaction["approval_reference"] = "APPROVAL-2026-014"
        if supersede_line is not None:
            # The amendment supersedes the original commitment; the branch reports
            # the current state, not the wording it replaced.
            transaction["commitment_current"] = False
            transaction["superseding_amendment"] = "补充协议二将质保期改为30日"
        elif commitment_line is not None:
            transaction["commitment_current"] = True
        findings.append(_finding("O-04", transaction, refs=[ref]))
        witnesses.extend(
            (
                _witness(
                    "O-04",
                    "category",
                    contract_category,
                    ref,
                    *_span(text, category_line),
                ),
                _witness(
                    "O-04",
                    "category_basis",
                    category_basis,
                    ref,
                    *_span(text, category_line),
                ),
            )
        )
        claims = []
        if source_line is not None:
            claims.append(("single_source", source_line, True))
        if justification_line is not None:
            claims.append(("written_justification", justification_line, True))
        if approval_line is not None:
            claims.append(("approval_reference", approval_line, "APPROVAL-2026-014"))
        if supersede_line is not None:
            claims.append(("commitment_current", supersede_line, False))
            claims.append(
                ("superseding_amendment", supersede_line, "补充协议二将质保期改为30日")
            )
        elif commitment_line is not None:
            claims.append(("commitment_current", commitment_line, True))
        for field, line, value in claims:
            witnesses.append(_witness("O-04", field, value, ref, *_span(text, line)))
    return {
        "schema_version": "finding-evidence.v2",
        "findings": findings,
        "witnesses": witnesses,
    }


def _run_case(case, *, policy_mutator=None):
    gold = as_gold_case(case)
    located = located_sources(gold)
    context = context_for(gold)
    sources = tuple(item.source for item in located)
    catalog = SourceCatalogV3(located)
    protection = ProtectedOutputV3()
    procurement = review_procurement_v3(
        context, conditional_procurement(case), authorized_scope_id=context.scope_id
    )
    grounded = []
    for worker, build in (("policy", _policy_answer), ("contract", _contract_answer)):
        request = document_request_v31(
            context,
            case_id=case.case_id,
            worker=worker,
            question=case.question,
            sources=tuple(item for item in sources if item.worker == worker),
        )
        answer = build(case, request)
        if worker == "policy" and policy_mutator is not None:
            answer = policy_mutator(answer, request.sources[0])
        body = json.dumps(answer, ensure_ascii=False)
        document, witnesses = parse_evidence_v31(body, request, protection=protection)
        grounded.append(ground_document_v31(request, document, witnesses, catalog))
    decision = finalize_review_v31(
        context,
        *grounded,
        procurement,
        sources=sources,
        protection=protection,
        pipeline_failures=(),
        control_events=(),
    )
    return decision, grounded


def _duty(decision, obligation):
    return next(
        row for row in decision.business.findings if row.obligation_id == obligation
    )


def test_the_authored_material_and_its_oracle_agree_on_every_case():
    cases = conditional_cases()
    assert len(cases) == 8
    for case in cases:
        truth = derive_conditional_truth(case)
        assert truth.obligations == case.expected, case.case_id
        assert truth.reasons == case.expected_reasons, case.case_id
        assert truth.policy_sha256 != truth.contract_sha256
    # Every policy and contract text is distinct: the lesson the retrieval audit
    # recorded is applied here instead of inventing two cases that cannot be told apart.
    policy_texts = {case.policy_text for case in cases}
    contract_texts = {case.contract_text for case in cases}
    assert len(policy_texts) == len(cases)
    assert len(contract_texts) == len(cases)
    truth_file = conditional_truth_file(cases)
    assert len(truth_file["truths"]) == len(cases)


def test_the_model_visible_schema_names_the_conditional_duties():
    case = conditional_cases()[0]
    gold = as_gold_case(case)
    located = located_sources(gold)
    context = context_for(gold)
    policy_request = document_request_v31(
        context,
        case_id=case.case_id,
        worker="policy",
        question=case.question,
        sources=tuple(
            item.source for item in located if item.source.worker == "policy"
        ),
    )
    contract_request = document_request_v31(
        context,
        case_id=case.case_id,
        worker="contract",
        question=case.question,
        sources=tuple(
            item.source for item in located if item.source.worker == "contract"
        ),
    )
    assert {check.proposition.check_id for check in policy_request.checks} == {
        "O-01",
        "O-02",
        "O-04",
    }
    assert {check.proposition.check_id for check in contract_request.checks} == {
        "O-04",
        "O-05",
        "O-06",
        "O-08",
        "O-09",
    }
    for text, fields in (
        (
            policy_request.question,
            (
                "minimum_quotes",
                "applies_to",
                "excluded_categories",
                "exclusion_basis",
                "exceptions",
                "exception_requires",
                "exception_basis",
            ),
        ),
        (
            contract_request.question,
            (
                "category_basis",
                "single_source",
                "written_justification",
                "approval_reference",
                "commitment_current",
                "superseding_amendment",
            ),
        ),
    ):
        for field in fields:
            assert field in text
    assert "3.1.0" in policy_request.question
    assert "政策不猜交易类别" in policy_request.question
    # The typed registry is the versioned one, and its slots convert into the rules.
    rule = QuoteRuleValuesV31(
        minimum_quotes=3,
        applies_to=("purchase",),
        excluded_categories=("lease",),
        exclusion_basis="租赁类交易不适用竞争性报价要求",
        exceptions=("single_source",),
        exception_requires=("written_justification",),
        exception_basis="唯一来源采购属于本办法登记的例外",
    )
    assert rule.to_quote_rule().exclusion_for("lease").basis
    transaction = ContractTransactionValuesV31(
        category="lease", category_basis="本次合同以设备租赁方式执行"
    )
    assert transaction.to_transaction().category == "lease"


def test_the_conditional_field_recognizers_are_conservative():
    lease_span = (
        "租赁类交易不适用竞争性报价要求（中部验证区设备租赁按单一承租人方式执行）。"
    )
    assert supports_field_v31(
        "policy", "O-04", "excluded_categories", '["lease"]', lease_span
    )
    assert not supports_field_v31(
        "policy", "O-04", "excluded_categories", '["lease"]', "租赁类交易按本规则执行。"
    )
    assert not supports_field_v31(
        "policy", "O-04", "excluded_categories", '["service"]', lease_span
    )
    assert not supports_field_v31(
        "policy", "O-04", "excluded_categories", "not-a-list", lease_span
    )
    assert supports_field_v31(
        "policy",
        "O-04",
        "excluded_categories",
        "[]",
        "竞争性报价至少3家，适用于采购与服务类交易。",
    )
    assert not supports_field_v31(
        "policy", "O-04", "excluded_categories", "[]", lease_span
    )
    assert supports_field_v31(
        "policy",
        "O-04",
        "exceptions",
        '["single_source"]',
        "唯一来源采购属于本办法登记的例外：须提供书面说明与审批授权。",
    )
    assert not supports_field_v31(
        "policy",
        "O-04",
        "exceptions",
        '["emergency"]',
        "唯一来源采购属于本办法登记的例外。",
    )
    assert supports_field_v31("contract", "O-04", "category", "lease", "租赁类交易。")
    assert not supports_field_v31(
        "contract", "O-04", "category", "lease", "采购类交易。"
    )
    assert supports_field_v31(
        "contract", "O-04", "single_source", True, "本合同为单一来源采购。"
    )
    assert not supports_field_v31(
        "contract", "O-04", "single_source", True, "本合同经比价采购。"
    )
    assert supports_field_v31(
        "contract", "O-04", "commitment_current", False, "原质保承诺已被补充协议替代。"
    )
    assert not supports_field_v31(
        "contract", "O-04", "commitment_current", False, "质保承诺仍现行有效。"
    )
    # The duties both versions share still use the frozen recognizers unchanged.
    assert supports_field_v31("policy", "O-02", "quotes", 3, "竞争性报价至少3家。")
    assert not supports_field_v31(
        "contract", "O-05", "buyer", "BUYER-X", "买方BUYER-Y。"
    )


@pytest.mark.parametrize("case", conditional_cases(), ids=lambda case: case.case_id)
def test_each_conditional_case_reaches_its_independently_derived_state(case):
    decision, grounded = _run_case(case)
    truth = derive_conditional_truth(case)
    for worker_index, duty_id in enumerate(("O-04", "O-08")):
        if duty_id not in truth.obligations:
            continue
        row = _duty(decision, duty_id)
        assert row.state == truth.obligations[duty_id], (
            case.case_id,
            duty_id,
            row.state,
        )
        assert row.reason == truth.reasons[duty_id], (case.case_id, duty_id, row.reason)
    for duty_id, missing in case.expected_missing.items():
        row = _duty(decision, duty_id)
        assert set(missing) <= set(row.missing), (case.case_id, duty_id, row.missing)
    # The recorded versions are the conditional ones, not the frozen strings.
    assert decision.business.rules_version == "3.1.0"
    assert decision.schema_version == "business-review.v3.1"
    assert all(row.rules_version == "3.1.0" for row in decision.business.findings)
    # The versioned grounding accepted the hand-composed witnesses: no duty was
    # dropped and no citation failed.
    assert decision.support_failures == (), (case.case_id, decision.support_failures)
    assert {row.obligation_id for row in decision.business.findings} >= {
        "O-01",
        "O-02",
        "O-03",
        "O-04",
        "O-05",
        "O-06",
        "O-07",
        "O-08",
        "O-09",
        "O-10",
        "O-11",
        "O-12",
    }
    assert grounded[0].support_failures == ()
    assert grounded[1].support_failures == ()
    # The other duties stay decidable: the conditional handover is local.
    assert _duty(decision, "O-03").state == "supported"
    assert _duty(decision, "O-06").state == "supported"
    assert _duty(decision, "O-07").state == "supported"
    assert _duty(decision, "O-05").state == "supported"


def test_an_unsupported_rule_value_cannot_reach_the_decision():
    """A rule the source does not carry is dropped instead of being applied."""

    case = conditional_cases()[0]

    def move_exclusion_basis(answer, source):
        for witness in answer["witnesses"]:
            if witness["field"] == "exclusion_basis":
                start, _end = _span(source.content, "交付不超过")
                witness["start"] = start
                witness["end"] = start + len(
                    "交付不超过15日；验收不超过7日；质保至少90日。"
                )
        return answer

    decision, grounded = _run_case(case, policy_mutator=move_exclusion_basis)
    assert "citation_proposition_unsupported" in grounded[0].support_failures
    # The rule value is dropped as a whole: the aggregator sees no quote rule and
    # keeps the duty UNKNOWN instead of applying a rule it cannot attribute.
    assert "O-04" not in grounded[0].document.values
    row = _duty(decision, "O-04")
    assert row.state == "unknown"
    assert row.reason == "quote_rule_or_source_missing"
    # The other duties are untouched by the failed rule citation.
    assert _duty(decision, "O-02").state == "supported"
