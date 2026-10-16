"""Independent validation documents for the complete-text candidate.

Six new families preserve English language variation, all frozen obligations and
scenario quotas. No model answers, production rules or extractors are read here.
"""

FAMILY_NAMES = (
    "maintenance-warrant",
    "instrument-lot",
    "compliance-addendum",
    "quality-workbook",
    "assembly-supply",
    "delivery-compact",
)


def validation_documents(style, facts):
    names = (
        "Maintenance procurement control warrant",
        "Purchasing conditions for the instrument lot",
        "Compliance addendum for equipment acquisition",
        "Quality office workbook for sourcing controls",
        "Acquisition controls for assembly supplies",
        "Delivery compact and purchasing governance",
    )
    scopes = (
        "This warrant applies to the current purchase in 2026.",
        "These lot purchasing conditions apply to this purchase throughout 2026.",
        "The addendum governs the current purchase during the year 2026.",
        "Scope: this purchase for the entire 2026 financial year.",
        "Applicability: this purchase in calendar 2026 under these controls.",
        "For this purchase in 2026 the buyer shall apply these purchasing controls.",
    )
    suppliers = (
        "The supplier must satisfy all five conditions: registry active; procurement approved; tax identifier registered; due diligence verified; screening clear.",
        "Require active registry and approved procurement, a registered tax identifier, verified due diligence and clear screening for the supplier.",
        "Registry status must be active. Procurement status must be approved. The tax identifier must be registered, due diligence must be verified, and screening must be clear.",
    )
    money = (
        "Invoice amount must equal the order amount; cumulative transactions must not exceed the order amount; the currencies must match.",
        "Keep order and invoice amounts equal, transaction total at most the order amount, and currencies identical across the three records.",
        "Order equals invoice, and transaction total <= order. Use the same currency for the order, invoice and historical transactions.",
    )
    sourcing = (
        "Competitive quotes shall be at least 3. Both the contract reference and the counterparty reference must be retained.",
        "Quote count >= 3; record a contract reference together with a counterparty reference.",
        "At least 3 competitive quotes are required before purchasing. Retain both contract and counterparty references for reconciliation.",
    )
    chronology = (
        "Booking shall occur no earlier than the ordered date.",
        "The booked date must be on or after the date of the order.",
        "Do not book an acquisition before the order date.",
    )
    durations = (
        "Delivery must take no more than 15 days; acceptance must take no more than 7 days; warranty must last no fewer than 90 days.",
        "Delivery period must be at most 15 days. Acceptance period must be at most 7 days. Warranty period must be at least 90 days.",
        "Maximum delivery is 15 days; maximum acceptance is 7 days; minimum warranty is 90 days.",
    )
    blocks = (
        suppliers[style % 3],
        money[(style + 2) % 3],
        sourcing[(style + 1) % 3],
        chronology[(style + 2) % 3],
        durations[style % 3],
    )
    permutations = (
        (1, 3, 0, 4, 2),
        (4, 0, 3, 1, 2),
        (2, 1, 4, 3, 0),
        (0, 3, 2, 4, 1),
        (3, 4, 1, 0, 2),
        (4, 2, 0, 3, 1),
    )
    policy = "\n".join(
        [names[style], scopes[style]]
        + [
            f"Control item {i + 1} — {blocks[j]}"
            for i, j in enumerate(permutations[style])
        ]
    )
    standards = (
        "each set meets the signed performance criteria",
        "all instruments pass the specified bench test",
        "every module is complete and free from damage",
    )
    duties = (
        "seller replaces damaged components without charge",
        "seller restores faulty instruments at its own cost",
        "seller supplies and fits replacement parts free",
    )
    damages = (
        "late installation costs 0.2% of the order amount per day",
        "late completion costs 0.3% of the invoice amount per week",
        "late arrival costs 0.1% of the contract price per day",
    )
    scope = (
        "changes, damages and liability cap; no other duties apply",
        "damages, written changes and cap; no other terms apply",
        "written changes and liability; no other obligations apply",
    )
    terms = (
        f"Buyer: {facts.counterparty_reference}; seller: {facts.supplier_id}; subject: portable diagnostic modules.",
        f"Order: {facts.purchase_order_id}; contract: {facts.contract_reference}.",
        f"Price: {facts.order_amount_minor // 100}.{facts.order_amount_minor % 100:02} {facts.order_currency}.",
        f"Delivery: {facts.delivery_days} days; acceptance: {facts.acceptance_days} days; warranty: {facts.warranty_days} days.",
        f"Acceptance standard: {standards[(style + 1) % 3]}.",
        f"Warranty responsibility: {duties[style % 3]}.",
        "Changes become effective only after both parties sign in writing.",
        f"Damages: {damages[(style + 2) % 3]}.",
        "Liability cap: 175000.00 CNY.",
        f"Other applicable terms: {scope[(style + 1) % 3]}.",
    )
    sequence = (
        (1, 0, 2, 4, 3, 5, 6, 8, 7, 9) if style % 2 else (0, 1, 3, 5, 4, 2, 7, 6, 8, 9)
    )
    contract = "\n".join(
        [
            "Signed " + FAMILY_NAMES[style].replace("-", " "),
            "These executed provisions govern the supply of diagnostic modules and the responsibilities of both signatories.",
            *(f"Article {i + 1} — {terms[j]}" for i, j in enumerate(sequence)),
        ]
    )
    question = "Evaluate the acquisition for compliance with the purchasing controls and the signed supply agreement. Use authorized records to explain the review result, evidence and unresolved matters."
    return question, policy, contract
