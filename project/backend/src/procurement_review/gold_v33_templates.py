"""Fresh English validation families frozen before any candidate observations.

Documents state the existing business obligations; they do not consume model
outputs or import extraction/scoring code. DEV and older corpora stay immutable.
"""

FAMILY_NAMES = (
    "instruction",
    "rider",
    "service-book",
    "undertaking",
    "annex",
    "supply-note",
)


def validation_documents(style, facts):
    policy_titles = (
        "Purchasing instruction to the sourcing team",
        "Governance rider for the purchasing department",
        "Service procurement rulebook",
        "Purchaser undertaking on internal controls",
        "Procurement controls annex",
        "Internal supply approval notice",
    )
    scope = (
        "These purchasing requirements apply to this purchase during 2026.",
        "Applicability: this purchase in the financial year 2026.",
        "This rulebook governs this purchase throughout 2026.",
        "Scope: this purchase, effective for calendar year 2026.",
        "The annex applies to this purchase in 2026; its controls are binding.",
        "For this purchase in 2026 the purchasing team shall apply these rules.",
    )
    eligibility = (
        "Registry must be active, procurement must be approved, the tax identifier must be registered, due diligence must be verified and screening must be clear.",
        "An eligible supplier has active registry and approved procurement status, a registered tax identifier, verified due diligence and clear screening.",
        "All of these supplier checks are mandatory: registry active; procurement approved; tax identifier registered; due diligence verified; screening clear.",
    )
    sourcing = (
        "The minimum competitive quote count is 3. Keep both the contract reference and the counterparty reference.",
        "At least 3 competitive quotes are required. The contract and counterparty references must both be recorded.",
        "Competitive quotes >= 3; a contract reference and a counterparty reference are mandatory.",
    )
    financial = (
        "The invoice amount must equal the order amount. Cumulative transactions must not exceed the order amount. All currencies must match.",
        "Order equals invoice; transaction total <= order. Order, invoice and transactions must use the same currency.",
        "Match the invoice amount to the order amount; cap cumulative transactions at the order amount and use one common currency throughout.",
    )
    dates = (
        "Booking shall occur no earlier than the order date.",
        "The booked date must be on or after the ordered date.",
        "Do not book a transaction before its order date.",
    )
    periods = (
        "Delivery at most 15 days, acceptance at most 7 days and warranty at least 90 days.",
        "Maximum delivery 15 days; maximum acceptance 7 days; minimum warranty 90 days.",
        "Delivery must take no more than 15 days; acceptance must take no more than 7 days; warranty must last at least 90 days.",
    )
    blocks = (
        eligibility[style % 3],
        sourcing[(style + 1) % 3],
        financial[(style + 2) % 3],
        dates[style % 3],
        periods[(style + 2) % 3],
    )
    ordering = (
        (0, 2, 3, 4, 1),
        (1, 4, 0, 2, 3),
        (2, 3, 1, 0, 4),
        (3, 0, 4, 1, 2),
        (4, 1, 2, 3, 0),
        (2, 4, 3, 0, 1),
    )
    policy = "\n".join(
        [
            policy_titles[style],
            scope[style],
            *(f"Control {i + 1}. {blocks[j]}" for i, j in enumerate(ordering[style])),
        ]
    )
    titles = (
        "Supply order instruction",
        "Executed contract rider",
        "Signed service delivery book",
        "Supplier performance undertaking",
        "Operative supply annex",
        "Agreed supply notification",
    )
    opening = (
        "The buyer and seller accept the following signed supply terms.",
        "This rider records the mutual obligations agreed by the two signatories.",
        "The signed service book identifies the order and the binding performance duties.",
        "The parties undertake performance on the following agreed conditions.",
        "This annex is binding on the named parties to the referenced order.",
        "The purchaser and supplier confirm the contractual obligations set out below.",
    )
    standard = (
        "all units conform to the signed specification",
        "each unit passes the agreed functional test",
        "every delivered unit is intact and complete",
    )[style % 3]
    responsibility = (
        "seller replaces defective units without charge",
        "seller repairs faults at its own expense",
        "seller supplies replacement parts free of charge",
    )[(style + 1) % 3]
    damages = (
        "late dispatch costs 0.3% of the order amount per day",
        "late supply costs 0.2% of the invoice amount per week",
        "late shipment costs 0.1% of the contract price per day",
    )[(style + 2) % 3]
    amount = f"{facts.order_amount_minor // 100}.{facts.order_amount_minor % 100:02}"
    sections = (
        f"Buyer: {facts.counterparty_reference}; seller: {facts.supplier_id}; subject: modular calibration devices.",
        f"Order: {facts.purchase_order_id}; contract: {facts.contract_reference}.",
        f"Price: {amount} {facts.order_currency}.",
        f"Delivery: {facts.delivery_days} days; acceptance: {facts.acceptance_days} days; warranty: {facts.warranty_days} days.",
        f"Acceptance standard: {standard}.",
        f"Warranty responsibility: {responsibility}.",
        "Changes become valid only after both parties sign in writing.",
        f"Damages: {damages}.",
        "Liability cap: 125000.00 CNY.",
        "Other applicable terms: changes, damages and the cap; no other duties apply.",
    )
    contract = "\n".join(
        [
            titles[style],
            opening[style],
            *(f"Clause {i + 1}: {value}" for i, value in enumerate(sections)),
        ]
    )
    question = "Assess the purchasing transaction against its policy and signed terms, using only authorized procurement records. Report the review findings with supporting evidence and any missing facts."
    return question, policy, contract
