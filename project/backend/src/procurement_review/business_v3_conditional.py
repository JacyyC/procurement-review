"""Opt-in rules 3.1.0: sourced applicability, conditions and exceptions.

Rules 3.0.0 stays frozen and is never reinterpreted.  This module adds what
§5.3 of the V1-07 Spec requires and 3.0.0 could not express: a policy rule that
carries its own applicability condition and its own exceptions with a basis, a
transaction category verified by the Contract worker, and an Aggregator that
applies the rule instead of comparing a bare threshold.

The handover discipline is the point of the module.  Policy never guesses a fact
that only exists downstream, Contract never has to restate the policy text, and a
locally missing dimension leaves the other verified duties alone instead of
blocking a legitimate handover.  When the merged, sourced result still lacks a
key dimension, the duty stays UNKNOWN.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .business_v3_procurement import count, present_reference
from .eval_v2_contracts import EvalV2Model
from .business_v3_types import (
    BusinessDecisionV3,
    BusinessFindingV3,
    Count,
    Short,
    finding,
)

RULES_VERSION_31 = "3.1.0"
TransactionCategory = Literal["purchase", "lease", "service", "framework_call_off"]
ExceptionKind = Literal["single_source", "emergency", "existing_contract_extension"]
ExceptionRequirement = Literal[
    "written_justification", "approval_reference", "price_reasonableness"
]


class BusinessFindingV31(BusinessFindingV3):
    """One rules 3.1.0 duty row: the same closed shape, its own version."""

    rules_version: Literal["3.1.0"] = RULES_VERSION_31


class BusinessDecisionV31(BusinessDecisionV3):
    """A decision computed under the conditional rule set, reporting 3.1.0.

    The precedence and the decision rule stay the frozen 3.0.0 implementation;
    only the recorded version and the row types are the ones this rule set owns.
    """

    rules_version: Literal["3.1.0"] = RULES_VERSION_31
    findings: tuple[BusinessFindingV31, ...]


def finding_v31(row: BusinessFindingV3) -> BusinessFindingV31:
    """Re-issue one duty row under the 3.1.0 version it was decided with."""

    return BusinessFindingV31(**row.model_dump(exclude={"rules_version"}))


def decide_business_v31(
    findings,
    *,
    pipeline_failures=(),
    security_events=(),
    conflict_pairs=(),
    observations=(),
) -> BusinessDecisionV31:
    """Run the frozen precedence and record the outcome as a 3.1.0 decision."""

    from .business_v3_aggregate import decide_business_v3

    decision = decide_business_v3(
        findings,
        pipeline_failures=pipeline_failures,
        security_events=security_events,
        conflict_pairs=conflict_pairs,
        observations=observations,
    )
    return BusinessDecisionV31(
        recommendation=decision.recommendation,
        findings=tuple(finding_v31(row) for row in decision.findings),
        violations=decision.violations,
        unknown_obligations=decision.unknown_obligations,
        pipeline_failures=decision.pipeline_failures,
        security_events=decision.security_events,
        conflict_pairs=decision.conflict_pairs,
        observations=decision.observations,
    )


class CategoryExclusionV31(EvalV2Model):
    """One sourced exclusion; the basis is policy text, not a Case-specific patch."""

    category: TransactionCategory
    basis: Short


class RuleExceptionV31(EvalV2Model):
    """One sourced exception and the requirements that must be verified to use it."""

    kind: ExceptionKind
    requires: tuple[ExceptionRequirement, ...] = Field(min_length=1, max_length=3)
    basis: Short

    @model_validator(mode="after")
    def requirements_are_unique(self) -> "RuleExceptionV31":
        if len(set(self.requires)) != len(self.requires):
            raise ValueError("an exception cannot require the same evidence twice")
        return self


class QuoteRuleV31(EvalV2Model):
    """The competitive-quote rule with its own applicability and exceptions."""

    minimum_quotes: Count
    applies_to: tuple[TransactionCategory, ...] = Field(min_length=1, max_length=4)
    excluded: tuple[CategoryExclusionV31, ...] = Field(default=(), max_length=4)
    exceptions: tuple[RuleExceptionV31, ...] = Field(default=(), max_length=3)

    @model_validator(mode="after")
    def exclusions_are_disjoint_from_scope(self) -> "QuoteRuleV31":
        excluded = {item.category for item in self.excluded}
        if excluded & set(self.applies_to):
            raise ValueError(
                "a category cannot be both in scope of the rule and excluded by it"
            )
        if len(excluded) != len(self.excluded):
            raise ValueError("one category cannot be excluded twice")
        if len({item.kind for item in self.exceptions}) != len(self.exceptions):
            raise ValueError("one exception kind cannot be registered twice")
        return self

    def exclusion_for(self, category: str) -> CategoryExclusionV31 | None:
        return next((item for item in self.excluded if item.category == category), None)

    def exception_for(self, kind: str) -> RuleExceptionV31 | None:
        return next((item for item in self.exceptions if item.kind == kind), None)


class ContractTransactionV31(EvalV2Model):
    """What the Contract worker verified about this transaction, with sources."""

    category: TransactionCategory | None = None
    category_basis: Short | None = None
    single_source: bool | None = None
    written_justification: bool | None = None
    approval_reference: Short | None = None
    price_reasonableness: bool | None = None
    commitment_current: bool | None = None
    superseding_amendment: Short | None = None

    @model_validator(mode="after")
    def claims_carry_their_basis(self) -> "ContractTransactionV31":
        if self.category is not None and self.category_basis is None:
            raise ValueError("a verified transaction category requires its basis")
        if self.approval_reference is not None and not self.approval_reference.strip():
            raise ValueError("an approval reference cannot be blank")
        return self


def _category_from_procurement(order) -> str | None:
    """A category the read-only facts state on their own; never inferred."""

    if order is None:
        return None
    value = order.result.get("transaction_category")
    return value if value in TransactionCategory.__args__ else None


def category_agreement(
    contract: ContractTransactionV31 | None, procurement_order
) -> tuple[str | None, tuple[str, ...]]:
    """Merge the two sourced categories; a disagreement is a conflict, not a guess."""

    from_contract = contract.category if contract is not None else None
    from_facts = _category_from_procurement(procurement_order)
    if from_contract is not None and from_facts is not None:
        if from_contract != from_facts:
            return None, ("transaction_category_conflict",)
        return from_contract, ()
    return (from_contract or from_facts), ()


def quote_check_v31(
    procurement,
    policy_terms: QuoteRuleV31 | None,
    policy_evidence: tuple,
    contract: ContractTransactionV31 | None,
    contract_evidence: tuple,
) -> BusinessFindingV3:
    """Apply the sourced rule to the admitted facts; never guess a dimension."""

    evidence = (
        procurement.evidence("order", "competitive_quote_count", "contract_reference")
        + procurement.evidence("transactions", "counterparty_reference")
        + policy_evidence
        + contract_evidence
    )
    if procurement.order is None or policy_terms is None:
        # The upstream side is not forced to invent what only downstream holds, and
        # a legitimate handover is not blocked: this duty alone stays UNKNOWN.
        return finding(
            "O-04",
            "unknown",
            "quote_rule_or_source_missing",
            evidence,
            missing=("quote_policy_and_facts",),
        )
    category, conflicts = category_agreement(contract, procurement.order)
    if conflicts:
        return finding(
            "O-04",
            "unknown",
            "transaction_category_conflict",
            evidence,
            missing=("agreed_transaction_category",),
        )
    if category is None:
        return finding(
            "O-04",
            "unknown",
            "transaction_category_missing",
            evidence,
            missing=("sourced_transaction_category",),
        )
    exclusion = policy_terms.exclusion_for(category)
    if exclusion is not None:
        return finding(
            "O-04",
            "not_applicable",
            "category_excluded_by_policy",
            evidence,
        )
    if category not in policy_terms.applies_to:
        return finding(
            "O-04",
            "unknown",
            "category_outside_registered_rule_scope",
            evidence,
            missing=("applicable_quote_rule",),
        )
    order = procurement.order.result
    quotes = count(order.get("competitive_quote_count"))
    missing = []
    if quotes is None:
        missing.append("competitive_quote_count")
    if not present_reference(order.get("contract_reference")):
        missing.append("contract_reference")
    if any(
        not present_reference(item.result.get("counterparty_reference"))
        for item in procurement.transactions
    ):
        missing.append("counterparty_reference")
    exception = _claimed_exception(contract, order)
    if exception is not None:
        unmet = _unmet_requirements(exception, contract, policy_terms)
        if unmet:
            # An unverified exception is neither a pass nor a blanket block.
            return finding(
                "O-04",
                "unknown",
                "exception_requirements_unverified",
                evidence,
                missing=tuple(unmet),
            )
        return finding("O-04", "not_applicable", "registered_exception_met", evidence)
    if missing:
        return finding(
            "O-04", "unknown", "quote_or_reference_missing", evidence, missing=missing
        )
    if quotes < policy_terms.minimum_quotes:
        return finding(
            "O-04",
            "violated",
            "quote_threshold_violated",
            evidence,
            risk="medium",
            remediable=True,
        )
    return finding("O-04", "supported", "quote_threshold_met", evidence)


def _claimed_exception(contract: ContractTransactionV31 | None, order) -> str | None:
    """The exception actually claimed by a source; never one inferred from a count."""

    if contract is not None and contract.single_source is True:
        return "single_source"
    declared = order.get("procurement_exception")
    return declared if declared in ExceptionKind.__args__ else None


def _unmet_requirements(
    kind: str,
    contract: ContractTransactionV31 | None,
    rule: QuoteRuleV31,
) -> list[str]:
    """Only an exception the policy itself registered, with a basis, can be used."""

    registered = rule.exception_for(kind)
    if registered is None:
        return ["registered_exception_rule"]
    requires = registered.requires
    verified = {
        "written_justification": bool(contract and contract.written_justification),
        "approval_reference": bool(
            contract and present_reference(contract.approval_reference)
        ),
        "price_reasonableness": bool(contract and contract.price_reasonableness),
    }
    return [name for name in requires if not verified[name]]


def commitment_check_v31(
    contract: ContractTransactionV31 | None,
    evidence: tuple,
    base: BusinessFindingV3,
) -> BusinessFindingV3:
    """A superseded or non-current commitment cannot support the duty it proves."""

    if contract is None or contract.commitment_current is None:
        return base
    if contract.commitment_current:
        return base
    if base.state == "supported":
        return finding(
            base.obligation_id,
            "unknown",
            "commitment_not_current",
            evidence,
            missing=("current_binding_commitment",),
        )
    return base


def aggregate_business_v31(
    context,
    policy,
    contract,
    procurement,
    *,
    transaction: ContractTransactionV31 | None,
    quote_rule: QuoteRuleV31 | None,
    contract_evidence: tuple = (),
    policy_evidence: tuple | None = None,
    pipeline_failures=(),
    security_events=(),
    conflict_pairs=(),
):
    """Rules 3.1.0: the frozen 3.0.0 aggregation with two duties re-applied.

    Only O-04 and O-08 change meaning, so the baseline decision is reused and the
    deterministic decider recomputes the recommendation from the substituted rows.
    Precedence, conflict and security handling therefore stay identical to 3.0.0.
    """

    from .business_v3_aggregate import aggregate_business_v3

    baseline = aggregate_business_v3(
        context,
        policy,
        contract,
        procurement,
        pipeline_failures=pipeline_failures,
        security_events=security_events,
        conflict_pairs=conflict_pairs,
    )
    substituted = quote_check_v31(
        procurement,
        quote_rule,
        policy.evidence("O-02") if policy_evidence is None else policy_evidence,
        transaction,
        contract_evidence,
    )
    rows = []
    for row in baseline.findings:
        if row.obligation_id == "O-04":
            rows.append(substituted)
        elif row.obligation_id == "O-08":
            rows.append(
                commitment_check_v31(transaction, contract.evidence("O-08"), row)
            )
        else:
            rows.append(row)
    return decide_business_v31(
        rows,
        pipeline_failures=baseline.pipeline_failures,
        security_events=baseline.security_events,
        conflict_pairs=baseline.conflict_pairs,
        observations=baseline.observations,
    )


def aggregate_business_v31_documents(
    context,
    policy,
    contract,
    procurement,
    *,
    pipeline_failures=(),
    security_events=(),
    conflict_pairs=(),
):
    """The document-level entry: the rule and the transaction come from their branches.

    Policy supplies the sourced quote rule under its own duty slot and Contract the
    verified transaction facts under its own; neither side is asked to restate the
    other's material, and a locally missing dimension leaves the other duties alone.
    A branch that reports the model-visible values is converted here, so the
    aggregation below works on the rule types it was verified against.
    """

    rule_value = policy.values.get("O-04")
    transaction_value = contract.values.get("O-04")
    return aggregate_business_v31(
        context,
        policy,
        contract,
        procurement,
        transaction=(
            transaction_value.to_transaction()
            if hasattr(transaction_value, "to_transaction")
            else transaction_value
        ),
        quote_rule=(
            rule_value.to_quote_rule()
            if hasattr(rule_value, "to_quote_rule")
            else rule_value
        ),
        contract_evidence=contract.evidence("O-04"),
        policy_evidence=policy.evidence("O-02") + policy.evidence("O-04"),
        pipeline_failures=pipeline_failures,
        security_events=security_events,
        conflict_pairs=conflict_pairs,
    )


__all__ = [
    "RULES_VERSION_31",
    "BusinessDecisionV31",
    "BusinessFindingV31",
    "CategoryExclusionV31",
    "ContractTransactionV31",
    "QuoteRuleV31",
    "RuleExceptionV31",
    "aggregate_business_v31",
    "aggregate_business_v31_documents",
    "category_agreement",
    "commitment_check_v31",
    "decide_business_v31",
    "finding_v31",
    "quote_check_v31",
]
