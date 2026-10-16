"""Independent validation documents for the complete-text candidate.

Six new families preserve English language variation, all frozen obligations and
scenario quotas. No model answers, production rules or extractors are read here.
"""

FAMILY_NAMES = (
    "calibration-protocol",
    "sensor-agreement",
    "laboratory-schedule",
    "service-covenant",
    "probe-allocation",
    "equipment-charter",
)


def validation_documents(style, facts):
    names = (
        "Calibration office purchasing protocol",
        "Sensor acquisition agreement and buying rules",
        "Laboratory sourcing schedule and purchase controls",
        "Service covenant and buyer control schedule",
        "Procurement policy for the probe allocation",
        "Equipment charter for acquisition review",
    )
    scopes = (
        "The calibration protocol applies to this purchase in 2026.",
        "These sensor acquisition rules apply to the current purchase in 2026.",
        "The laboratory schedule governs this purchase during 2026.",
        "Scope: the current purchase during calendar year 2026.",
        "Applicability: the probe purchase under the 2026 controls.",
        "For the current purchase in 2026 the buyer shall apply this charter.",
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
        (3, 4, 1, 2, 0),
        (2, 3, 0, 1, 4),
        (2, 1, 4, 3, 0),
        (0, 3, 2, 4, 1),
        (3, 4, 1, 0, 2),
        (4, 2, 0, 3, 1),
    )
    policy = "\n".join(
        [names[style], scopes[style]]
        + [
            f"Procurement obligation {i + 1} — {blocks[j]}"
            for i, j in enumerate(permutations[style])
        ]
    )
    standards = (
        "all probes meet the approved calibration tolerances",
        "every sensor passes the agreed functional check",
        "each assembly matches the signed technical schedule",
    )
    duties = (
        "seller renews faulty sensors without charging the buyer",
        "seller repairs defective probes at its own expense",
        "seller replaces defective assemblies free of charge",
    )
    damages = (
        "late dispatch costs 0.4% of the invoice value per day",
        "late supply costs 0.2% of the agreed price per week",
        "late handover costs 0.5% of the order value per day",
    )
    scope = (
        "changes, damages and liability cap; no other duties apply",
        "damages, written changes and cap; no other terms apply",
        "written changes and liability; no other obligations apply",
    )
    terms = (
        f"Buyer: {facts.counterparty_reference}; seller: {facts.supplier_id}; subject: precision calibration sensors.",
        f"Order: {facts.purchase_order_id}; contract: {facts.contract_reference}.",
        f"Price: {facts.order_amount_minor // 100}.{facts.order_amount_minor % 100:02} {facts.order_currency}.",
        f"Delivery: {facts.delivery_days} days; acceptance: {facts.acceptance_days} days; warranty: {facts.warranty_days} days.",
        f"Acceptance standard: {standards[(style + 1) % 3]}.",
        f"Warranty responsibility: {duties[style % 3]}.",
        "Changes become effective only after both parties sign in writing.",
        f"Damages: {damages[(style + 2) % 3]}.",
        "Liability cap: 186000.00 CNY.",
        f"Other applicable terms: {scope[(style + 1) % 3]}.",
    )
    sequence = (
        (0, 1, 2, 3, 5, 4, 8, 6, 7, 9) if style % 2 else (1, 0, 3, 2, 4, 5, 7, 8, 6, 9)
    )
    contract = "\n".join(
        [
            "Signed " + FAMILY_NAMES[style].replace("-", " "),
            "Both signatories accept the following provisions governing calibration sensors, acceptance and remedies.",
            *(f"Supply provision {i + 1} — {terms[j]}" for i, j in enumerate(sequence)),
        ]
    )
    question = "Review this purchase against the governing controls and executed supply terms. Explain the recommendation using authorized evidence and identify any unresolved duties."
    return question, policy, contract
