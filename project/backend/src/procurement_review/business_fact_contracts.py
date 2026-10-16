"""The registered result shapes of the read-only procurement projection (FIX-03A).

A ``BusinessFact.result`` is the projection of one registered synthetic procurement record,
so its key space and the type of each key are the dataset's contract rather than a free
mapping.  Two shipped datasets produce these records — the synthetic procurement fixture
and the smaller security fixture — and both spellings of the same tool are registered, so a
tool name outside this table is a contract violation and never a new fact.
"""

from __future__ import annotations

from .value_rules import (
    ValueRule,
    calendar_day,
    count,
    free_text,
    free_text_list,
    number,
    text,
    word,
)

BUSINESS_FACT_RESULT_RULES: dict[str, dict[str, ValueRule]] = {
    "purchase_order": {
        "amount": number(),
        "amount_minor": count(),
        "marker": free_text(max_length=64),
        "competitive_quote_count": count(nullable=True),
        "contract_reference": text(max_length=64, nullable=True),
        "currency": word(frozenset({"CNY", "USD"})),
        "invoice_amount_minor": count(),
        "invoice_currency": word(frozenset({"CNY", "USD"})),
        "ordered_on": calendar_day(),
        "purchase_order_id": text(max_length=160),
        "requesting_unit": free_text(max_length=64),
        "status": word(frozenset({"pending_review", "approved", "blocked"})),
        "supplier_id": text(max_length=160),
        # Rules 3.1.0 adds the verified transaction category to the order fact; a
        # record written before the extension simply does not carry the key.
        "transaction_category": word(
            frozenset({"purchase", "lease", "service", "framework_call_off"})
        ),
    },
    "supplier": {
        "display_name": free_text(max_length=64),
        "marker": free_text(max_length=64),
        "name": free_text(max_length=64),
        "due_diligence_status": text(max_length=32),
        "jurisdiction": text(max_length=16),
        "onboarded_on": calendar_day(),
        "procurement_status": text(max_length=32),
        "registry_status": text(max_length=32),
        "risk_signals": free_text_list(),
        "sanctions_screening": text(max_length=32),
        "sanctions_status": text(max_length=32),
        "supplier_id": text(max_length=160),
        "tax_identifier": text(max_length=64, nullable=True),
    },
    "historical_transaction": {
        "amount": number(),
        "amount_minor": count(),
        "marker": free_text(max_length=64),
        "booked_on": calendar_day(),
        "counterparty_reference": text(max_length=64, nullable=True),
        "currency": word(frozenset({"CNY", "USD"})),
        "purchase_order_id": text(max_length=160),
        "status": word(frozenset({"settled", "pending", "reversed"})),
        "supplier_id": text(max_length=160),
        "transaction_id": text(max_length=160),
    },
}

#: Registered tool names, mapped to the record kind whose result shape they return.  Both
#: the runtime MCP spelling and the evaluation spelling of the same tool are registered;
#: an unregistered tool name is a contract violation, never a new fact.
BUSINESS_FACT_TOOL_KINDS: dict[str, str] = {
    "get_purchase_order": "purchase_order",
    "procurement.get_purchase_order": "purchase_order",
    "get_supplier": "supplier",
    "procurement.get_supplier": "supplier",
    "list_historical_transactions": "historical_transaction",
    "procurement.list_historical_transactions": "historical_transaction",
}

#: The business-fact tool names the write path may persist.
BUSINESS_FACT_TOOLS = frozenset(BUSINESS_FACT_TOOL_KINDS)

#: Every key any registered result shape may carry.
BUSINESS_FACT_RESULT_KEYS = frozenset(
    key for rules in BUSINESS_FACT_RESULT_RULES.values() for key in rules
)


#: Registered tool names, mapped to the record kind whose result shape they return.  Both
#: the runtime MCP spelling and the evaluation spelling of the same tool are registered.
BUSINESS_FACT_TOOL_KINDS: dict[str, str] = {
    "get_purchase_order": "purchase_order",
    "procurement.get_purchase_order": "purchase_order",
    "get_supplier": "supplier",
    "procurement.get_supplier": "supplier",
    "list_historical_transactions": "historical_transaction",
    "procurement.list_historical_transactions": "historical_transaction",
}

#: The business-fact tool names the write path may persist.
BUSINESS_FACT_TOOLS = frozenset(BUSINESS_FACT_TOOL_KINDS)

#: Every key any registered result shape may carry.
BUSINESS_FACT_RESULT_KEYS = frozenset(
    key for rules in BUSINESS_FACT_RESULT_RULES.values() for key in rules
)


def business_fact_namespace(tool_name: str) -> str:
    """The namespace whose result rules govern one business fact."""

    kind = BUSINESS_FACT_TOOL_KINDS.get(tool_name)
    return f"business_fact:{kind}" if kind is not None else "business_fact:unregistered"


__all__ = [
    "BUSINESS_FACT_RESULT_KEYS",
    "BUSINESS_FACT_RESULT_RULES",
    "BUSINESS_FACT_TOOL_KINDS",
    "BUSINESS_FACT_TOOLS",
    "business_fact_namespace",
]
