"""Rules 3.1.0 handover counterexamples: sourced conditions, exceptions, synonyms.

Expectations are hand-written from the business contract, never produced by the
rules under test.  Rules 3.0.0 stays frozen and is asserted unchanged here.
"""

from __future__ import annotations

import pytest

from procurement_review.business_v3_aggregate import aggregate_business_v3
from procurement_review.business_v3_conditional import (
    CategoryExclusionV31,
    ContractTransactionV31,
    QuoteRuleV31,
    RuleExceptionV31,
    aggregate_business_v31,
    quote_check_v31,
)
from procurement_review.business_v3_procurement import review_procurement_v3
from procurement_review.eval_f_text import whole_field
from v1_07d_support import context, document, mcp_result, states

RULE = QuoteRuleV31(
    minimum_quotes=3,
    applies_to=("purchase", "service"),
    excluded=(
        CategoryExclusionV31(category="lease", basis="租赁类采购不适用竞争性报价要求"),
    ),
    exceptions=(
        RuleExceptionV31(
            kind="single_source",
            requires=("written_justification", "approval_reference"),
            basis="唯一来源采购经书面说明与授权审批后可免于比价",
        ),
    ),
)


def reviewed(*, order=None, transaction=None, quote_rule=RULE):
    """One deterministic handover: Policy rule, Contract verification, MCP facts."""

    ctx = context()
    policy = document("policy")
    contract = document("contract")
    procurement = review_procurement_v3(
        ctx,
        mcp_result(order=order),
        authorized_scope_id=ctx.scope_id,
    )
    decision = aggregate_business_v31(
        ctx,
        policy,
        contract,
        procurement,
        transaction=transaction,
        quote_rule=quote_rule,
    )
    return decision, procurement, policy, contract


def duty(decision, obligation):
    return next(row for row in decision.findings if row.obligation_id == obligation)


def test_a_sourced_lease_exclusion_makes_the_duty_not_applicable():
    decision, *_ = reviewed(
        order={"transaction_category": "lease", "competitive_quote_count": 1},
    )
    row = duty(decision, "O-04")
    # One quote is far below the threshold, yet the sourced exclusion applies.
    assert row.state == "not_applicable"
    assert row.reason == "category_excluded_by_policy"
    assert row.evidence
    assert decision.recommendation == "recommend_approve"


def test_an_exclusion_is_never_earned_without_a_sourced_category():
    decision, *_ = reviewed(order={"competitive_quote_count": 1})
    row = duty(decision, "O-04")
    assert row.state == "unknown"
    assert row.reason == "transaction_category_missing"
    assert "sourced_transaction_category" in row.missing
    # The gap stays local: the other verified duties are untouched.
    assert states(decision)["O-03"] == "supported"
    assert states(decision)["O-06"] == "supported"
    assert states(decision)["O-07"] == "supported"


def test_a_category_disagreement_is_a_conflict_and_not_a_guess():
    decision, *_ = reviewed(
        order={"transaction_category": "purchase", "competitive_quote_count": 1},
        transaction=ContractTransactionV31(
            category="lease", category_basis="合同第一条：本合同为设备租赁"
        ),
    )
    row = duty(decision, "O-04")
    assert row.state == "unknown"
    assert row.reason == "transaction_category_conflict"
    assert "agreed_transaction_category" in row.missing


def test_a_single_source_exception_requires_its_registered_evidence():
    claimed = ContractTransactionV31(
        category="purchase",
        category_basis="合同第一条：设备采购",
        single_source=True,
    )
    decision, *_ = reviewed(
        order={"transaction_category": "purchase", "competitive_quote_count": 1},
        transaction=claimed,
    )
    row = duty(decision, "O-04")
    # Claiming the exception is not enough; each registered requirement is checked.
    assert row.state == "unknown"
    assert row.reason == "exception_requirements_unverified"
    assert set(row.missing) == {"written_justification", "approval_reference"}

    verified = ContractTransactionV31(
        category="purchase",
        category_basis="合同第一条：设备采购",
        single_source=True,
        written_justification=True,
        approval_reference="APPROVAL-2026-014",
    )
    decision, *_ = reviewed(
        order={"transaction_category": "purchase", "competitive_quote_count": 1},
        transaction=verified,
    )
    row = duty(decision, "O-04")
    assert row.state == "not_applicable"
    assert row.reason == "registered_exception_met"


def test_an_exception_the_policy_never_registered_cannot_be_used():
    decision, *_ = reviewed(
        order={
            "transaction_category": "purchase",
            "competitive_quote_count": 1,
            "procurement_exception": "emergency",
        },
    )
    row = duty(decision, "O-04")
    assert row.state == "unknown"
    assert "registered_exception_rule" in row.missing


def test_the_threshold_still_applies_to_an_in_scope_category():
    decision, *_ = reviewed(
        order={"transaction_category": "purchase", "competitive_quote_count": 2},
        transaction=ContractTransactionV31(
            category="purchase", category_basis="合同第一条：设备采购"
        ),
    )
    row = duty(decision, "O-04")
    assert row.state == "violated" and row.reason == "quote_threshold_violated"
    assert row.remediable and not row.blocking
    decision, *_ = reviewed(
        order={"transaction_category": "purchase", "competitive_quote_count": 3},
        transaction=ContractTransactionV31(
            category="purchase", category_basis="合同第一条：设备采购"
        ),
    )
    row = duty(decision, "O-04")
    assert row.state == "supported" and row.reason == "quote_threshold_met"


def test_a_category_outside_the_registered_scope_stays_unknown():
    decision, *_ = reviewed(
        order={
            "transaction_category": "framework_call_off",
            "competitive_quote_count": 1,
        },
    )
    row = duty(decision, "O-04")
    assert row.state == "unknown"
    assert row.reason == "category_outside_registered_rule_scope"


def test_a_superseded_commitment_cannot_support_the_duty_it_proves():
    baseline, *_ = reviewed(order={"transaction_category": "purchase"})
    assert duty(baseline, "O-08").state == "supported"
    superseded, *_ = reviewed(
        order={"transaction_category": "purchase"},
        transaction=ContractTransactionV31(
            category="purchase",
            category_basis="合同第一条：设备采购",
            commitment_current=False,
            superseding_amendment="补充协议二将质保期改为30日",
        ),
    )
    row = duty(superseded, "O-08")
    assert row.state == "unknown" and row.reason == "commitment_not_current"
    assert "current_binding_commitment" in row.missing


def test_a_missing_rule_or_source_never_forces_an_upstream_guess():
    ctx = context()
    policy = document("policy")
    procurement = review_procurement_v3(
        ctx, mcp_result(), authorized_scope_id=ctx.scope_id
    )
    row = quote_check_v31(procurement, None, policy.evidence("O-02"), None, ())
    assert row.state == "unknown"
    assert row.reason == "quote_rule_or_source_missing"
    # The Policy side is not asked to invent a category it has no source for.
    assert "quote_policy_and_facts" in row.missing


def test_the_rule_rejects_a_category_that_is_both_in_scope_and_excluded():
    with pytest.raises(ValueError, match="excluded by it"):
        QuoteRuleV31(
            minimum_quotes=3,
            applies_to=("purchase", "lease"),
            excluded=(CategoryExclusionV31(category="lease", basis="租赁排除"),),
        )
    with pytest.raises(ValueError, match="requires its basis"):
        ContractTransactionV31(category="lease")


def test_rules_3_0_0_stays_frozen_and_unconditional():
    """The frozen path keeps its bare threshold comparison and its own version."""

    ctx = context()
    policy = document("policy")
    contract = document("contract")
    procurement = review_procurement_v3(
        ctx,
        mcp_result(
            order={"transaction_category": "lease", "competitive_quote_count": 1}
        ),
        authorized_scope_id=ctx.scope_id,
    )
    baseline = aggregate_business_v3(ctx, policy, contract, procurement)
    row = next(row for row in baseline.findings if row.obligation_id == "O-04")
    assert row.state == "violated" and row.reason == "quote_threshold_violated"
    assert baseline.rules_version == "3.0.0"
    conditional, *_ = reviewed(
        order={"transaction_category": "lease", "competitive_quote_count": 1}
    )
    assert duty(conditional, "O-04").state == "not_applicable"


def test_synonymous_headings_pass_only_when_the_recognizer_registers_them():
    """A novel synonym fails closed; it is never read as support by guessing."""

    for source in (
        "责任上限：100000.00 CNY。",
        "累计赔偿责任上限：100000.00 CNY。",
        "liability cap: 100000.00 CNY.",
        "liability limit: 100000.00 CNY.",
    ):
        assert whole_field("cap", "100000.00 CNY", source) is True
    for source in (
        "最高赔付额度：100000.00 CNY。",
        "赔偿天花板：100000.00 CNY。",
        "the most we will pay is 100000.00 CNY.",
    ):
        assert whole_field("cap", "100000.00 CNY", source) is False
    # Fields the recognizer does not adjudicate stay untouched rather than guessed.
    assert whole_field("amount", "100000.00 CNY", "任意正文 100000.00 CNY") is True
