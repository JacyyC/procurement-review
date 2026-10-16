"""Pure comparisons for frozen business obligations; no model, corpus or labels."""

from .business_v3_procurement import calendar_day, count, money_minor, present_reference
from .business_v3_types import finding


def supplier_check(procurement, policy):
    evidence = procurement.evidence(
        "supplier",
        "registry_status",
        "procurement_status",
        "tax_identifier",
        "due_diligence_status",
        "sanctions_screening",
    )
    evidence += policy.evidence("O-02")
    if procurement.supplier is None or policy.values.get("O-02") is None:
        return finding(
            "O-03",
            "unknown",
            "supplier_rule_or_source_missing",
            evidence,
            missing=("supplier_policy_and_facts",),
        )
    data = procurement.supplier.result
    allowed = {
        "registry_status": "active",
        "procurement_status": "approved",
        "due_diligence_status": "verified",
        "sanctions_screening": "clear",
    }
    forbidden = {
        "registry_status": "inactive",
        "procurement_status": "blocked",
        "due_diligence_status": "failed",
        "sanctions_screening": "potential_match",
    }
    missing = [
        key
        for key in allowed
        if not isinstance(data.get(key), str)
        or data.get(key) not in {allowed[key], forbidden[key]}
    ]
    if (
        not isinstance(data.get("tax_identifier"), str)
        or not data["tax_identifier"].strip()
    ):
        missing.append("tax_identifier")
    # An explicit blocking fact is not erased by an unrelated incomplete field.
    if any(data.get(key) == value for key, value in forbidden.items()):
        return finding(
            "O-03",
            "violated",
            "supplier_blocking_fact",
            evidence,
            risk="high",
            blocking=True,
            missing=missing,
        )
    if missing:
        return finding(
            "O-03", "unknown", "supplier_fact_missing", evidence, missing=missing
        )
    return finding("O-03", "supported", "supplier_requirements_met", evidence)


def quote_check(procurement, policy):
    evidence = procurement.evidence(
        "order", "competitive_quote_count", "contract_reference"
    )
    evidence += procurement.evidence(
        "transactions", "counterparty_reference"
    ) + policy.evidence("O-02")
    terms = policy.values.get("O-02")
    if procurement.order is None or terms is None:
        return finding(
            "O-04",
            "unknown",
            "quote_rule_or_source_missing",
            evidence,
            missing=("quote_policy_and_facts",),
        )
    order = procurement.order.result
    quotes = count(order.get("competitive_quote_count"))
    missing = []
    if quotes is None:
        missing.append("competitive_quote_count")
    if not present_reference(order.get("contract_reference")):
        missing.append("contract_reference")
    if any(
        not present_reference(t.result.get("counterparty_reference"))
        for t in procurement.transactions
    ):
        missing.append("counterparty_reference")
    if missing:
        return finding(
            "O-04", "unknown", "quote_or_reference_missing", evidence, missing=missing
        )
    if quotes < terms.quotes:
        return finding(
            "O-04",
            "violated",
            "quote_threshold_violated",
            evidence,
            risk="medium",
            remediable=True,
        )
    return finding("O-04", "supported", "quote_threshold_met", evidence)


def amount_check(procurement, contract, policy):
    evidence = procurement.evidence(
        "order", "amount_minor", "currency", "invoice_amount_minor", "invoice_currency"
    )
    evidence += (
        procurement.evidence("transactions", "amount_minor", "currency")
        + contract.evidence("O-06")
        + policy.evidence("O-02")
    )
    price = contract.values.get("O-06")
    if procurement.order is None or price is None or policy.values.get("O-02") is None:
        return finding(
            "O-06",
            "unknown",
            "amount_rule_or_source_missing",
            evidence,
            missing=("contract_price_and_procurement",),
        )
    order = procurement.order.result
    values = [
        count(order.get("amount_minor")),
        count(order.get("invoice_amount_minor")),
    ]
    values += [count(t.result.get("amount_minor")) for t in procurement.transactions]
    currencies = [order.get("currency"), order.get("invoice_currency")]
    currencies += [t.result.get("currency") for t in procurement.transactions]
    contract_amount = money_minor(price.amount, price.currency)
    if (
        contract_amount is None
        or any(v is None for v in values)
        or any(not isinstance(c, str) or c not in {"CNY", "USD"} for c in currencies)
    ):
        return finding(
            "O-06",
            "unknown",
            "amount_or_currency_missing",
            evidence,
            missing=("valid_amount_and_currency",),
        )
    if currencies[0] != currencies[1]:
        # The mismatch is proven; no cross-currency sum is performed.
        return finding(
            "O-06", "violated", "currency_equality_violated", evidence, risk="medium"
        )
    if any(c != currencies[0] for c in (*currencies[2:], price.currency)):
        return finding(
            "O-06",
            "unknown",
            "currency_conversion_unavailable",
            evidence,
            missing=("explicit_exchange_rate",),
        )
    if (
        values[0] != values[1]
        or contract_amount != values[0]
        or sum(values[2:]) > values[0]
    ):
        return finding(
            "O-06",
            "violated",
            "amount_rule_violated",
            evidence,
            risk="high",
            blocking=True,
        )
    return finding("O-06", "supported", "amount_currency_requirements_met", evidence)


def date_check(context, procurement, policy):
    evidence = (
        procurement.evidence("order", "ordered_on")
        + procurement.evidence("transactions", "booked_on")
        + policy.evidence("O-02")
    )
    if procurement.order is None or policy.values.get("O-02") is None:
        return finding(
            "O-07",
            "unknown",
            "date_rule_or_source_missing",
            evidence,
            missing=("date_policy_and_facts",),
        )
    ordered = calendar_day(procurement.order.result.get("ordered_on"), context.timezone)
    booked = [
        calendar_day(t.result.get("booked_on"), context.timezone)
        for t in procurement.transactions
    ]
    if ordered is None or any(d is None for d in booked):
        return finding(
            "O-07",
            "unknown",
            "date_missing_or_invalid",
            evidence,
            missing=("calendar_dates_with_timezone",),
        )
    if any(d < ordered for d in booked):
        return finding(
            "O-07",
            "violated",
            "date_sequence_violated",
            evidence,
            risk="high",
            blocking=True,
        )
    return finding("O-07", "supported", "date_sequence_met", evidence)


def contract_checks(context, procurement, contract, policy):
    results = []
    core = contract.values.get("O-05")
    if core is None:
        results.append(
            finding(
                "O-05",
                "unknown",
                "contract_core_terms_missing",
                contract.evidence("O-05"),
                missing=("parties_subject_references",),
            )
        )
    else:
        refs = (
            contract.evidence("O-05")
            + procurement.evidence("order", "contract_reference")
            + procurement.evidence("transactions", "counterparty_reference")
        )
        if (
            procurement.order is None
            or not present_reference(procurement.order.result.get("contract_reference"))
            or any(
                not present_reference(t.result.get("counterparty_reference"))
                for t in procurement.transactions
            )
        ):
            results.append(
                finding(
                    "O-05",
                    "unknown",
                    "contract_identity_comparison_missing",
                    refs,
                    missing=("procurement_references",),
                )
            )
        else:
            matches = (
                core.seller == context.supplier_id
                and core.order == context.purchase_order_id
                and core.contract == context.contract_id
                and procurement.order.result.get("contract_reference") == core.contract
                and all(
                    t.result.get("counterparty_reference") == core.buyer
                    for t in procurement.transactions
                )
            )
            results.append(
                finding(
                    "O-05",
                    "supported" if matches else "violated",
                    "contract_identity_complete"
                    if matches
                    else "contract_identity_mismatch",
                    refs,
                    risk="none" if matches else "high",
                    blocking=not matches,
                )
            )
    terms, limits = contract.values.get("O-08"), policy.values.get("O-02")
    refs = contract.evidence("O-08") + policy.evidence("O-02")
    if terms is None or limits is None:
        results.append(
            finding(
                "O-08",
                "unknown",
                "delivery_terms_missing",
                refs,
                missing=("delivery_acceptance_warranty",),
            )
        )
    else:
        breach = (
            terms.delivery > limits.delivery
            or terms.acceptance > limits.acceptance
            or terms.warranty < limits.warranty
        )
        results.append(
            finding(
                "O-08",
                "violated" if breach else "supported",
                "delivery_rule_violated" if breach else "delivery_requirements_met",
                refs,
                risk="medium" if breach else "none",
                remediable=breach,
            )
        )
    change = contract.values.get("O-09")
    refs = contract.evidence("O-09")
    if change is None:
        results.append(
            finding(
                "O-09",
                "unknown",
                "change_terms_missing",
                refs,
                missing=("written_change_damages_cap",),
            )
        )
    else:
        results.append(
            finding(
                "O-09",
                "supported" if change.written else "violated",
                "change_requirements_met"
                if change.written
                else "written_change_violated",
                refs,
                risk="none" if change.written else "medium",
                remediable=not change.written,
            )
        )
    return tuple(results)
