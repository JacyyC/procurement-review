"""Six replacement validation families, frozen independently of model results.

The business obligations, English language coverage and scenario quotas are the
same as v3.3. Document structure, substantive clauses and entities are new; no
production recognizer or model output is imported to produce these materials.
"""

FAMILY_NAMES = (
    "commissioning-record",
    "equipment-covenant",
    "framework-release",
    "laboratory-schedule",
    "dispatch-agreement",
    "purchase-charter",
)


def validation_documents(style, facts):
    heading = (
        "Commissioning office purchasing controls",
        "Equipment acquisition governance covenant",
        "Release of goods under the procurement framework",
        "Laboratory purchasing schedule and financial controls",
        "Dispatch office acquisition agreement",
        "Charter for purchasing inspection instruments",
    )[style]
    scope = (
        "These acquisition requirements apply to this purchase during calendar year 2026.",
        "The covenant governs this purchase for the year 2026.",
        "Scope: the current purchase in the 2026 accounting year.",
        "For this purchase in 2026 the laboratory shall apply the following controls.",
        "Applicability: the current purchase, with effect throughout 2026.",
        "These binding controls apply to this purchase in 2026.",
    )[style]
    supplier = (
        "Supplier registry shall be active and procurement status shall be approved. The tax identifier must be registered; due diligence shall be verified and screening shall be clear.",
        "Supplier checks require active registry, approved procurement, registered tax identifier, verified due diligence and clear screening.",
        "Before ordering, check registry: active; procurement: approved; tax identifier: registered; due diligence: verified; screening: clear.",
    )[style % 3]
    finance = (
        "The order and invoice amounts must be equal. Cumulative transactions must be at most the order amount; currencies must be identical.",
        "Invoice must match the order amount. Transaction total must not exceed the order amount and use the same currency for all three records.",
        "Order equals invoice in amount. Transactions must remain bounded by the order amount. The order, invoice and transactions require matching currencies.",
    )[(style + 1) % 3]
    quotes = (
        "No fewer than 3 competitive quotes are required; retain a contract reference together with the counterparty reference.",
        "Competitive quote count must be at least 3. Keep the contract and counterparty references in the transaction file.",
        "Minimum competitive quote count is 3; the sourcing file must include contract and counterparty references.",
    )[(style + 2) % 3]
    dates = (
        "Booking shall occur on or after the ordered date.",
        "The booked date shall be no earlier than the order date.",
        "Do not book any acquisition before its order date.",
    )[style % 3]
    periods = (
        "Delivery within 15 days; acceptance within 7 days; warranty no fewer than 90 days.",
        "Maximum delivery period is 15 days. Maximum acceptance period is 7 days. Minimum warranty period is 90 days.",
        "Delivery shall take at most 15 days, acceptance shall take at most 7 days, and warranty shall last at least 90 days.",
    )[(style + 1) % 3]
    blocks = (supplier, finance, quotes, dates, periods)
    orders = (
        (4, 2, 1, 0, 3),
        (3, 1, 4, 2, 0),
        (1, 0, 2, 4, 3),
        (2, 3, 0, 1, 4),
        (0, 4, 3, 2, 1),
        (3, 2, 4, 0, 1),
    )
    policy = "\n".join(
        [heading, scope]
        + [f"Requirement {i + 1}: {blocks[j]}" for i, j in enumerate(orders[style])]
    )
    standards = (
        "all devices pass the agreed calibration test",
        "every kit contains all parts on the signed list",
        "each instrument meets the stated accuracy range",
    )
    duties = (
        "seller repairs defective devices without charge",
        "seller replaces failed components at its expense",
        "seller bears all costs of repairing faulty kits",
    )
    damages = (
        "late delivery costs 0.4% of the order amount per week",
        "late handover costs 0.2% of the invoice amount per day",
        "late dispatch costs 0.5% of the contract price per week",
    )
    other = (
        "written changes, damages and cap; no other terms apply",
        "damages, cap and changes; no other obligations apply",
        "changes, compensation and cap; no other duties apply",
    )
    terms = (
        f"Order: {facts.purchase_order_id}; contract: {facts.contract_reference}.",
        f"Buyer: {facts.counterparty_reference}; seller: {facts.supplier_id}; subject: field calibration kits.",
        f"Price: {facts.order_amount_minor // 100}.{facts.order_amount_minor % 100:02} {facts.order_currency}.",
        f"Delivery: {facts.delivery_days} days; acceptance: {facts.acceptance_days} days; warranty: {facts.warranty_days} days.",
        f"Acceptance standard: {standards[style % 3]}.",
        f"Warranty responsibility: {duties[(style + 2) % 3]}.",
        "Changes take effect after both parties sign in writing.",
        f"Damages: {damages[(style + 1) % 3]}.",
        "Liability cap: 160000.00 CNY.",
        f"Other applicable terms: {other[style % 3]}.",
    )
    # Alternate contract section order without changing the substantive duties.
    order = (
        (0, 1, 2, 3, 4, 5, 8, 6, 7, 9) if style % 2 else (1, 0, 3, 2, 5, 4, 7, 8, 6, 9)
    )
    contract = "\n".join(
        [
            "Executed " + FAMILY_NAMES[style].replace("-", " "),
            "The parties sign these provisions for the instrument supply and commissioning transaction.",
            *(f"Provision {i + 1}. {terms[j]}" for i, j in enumerate(order)),
        ]
    )
    question = "Review this acquisition using its operative purchasing controls, executed supply provisions and authorized procurement records. Identify supported findings, evidence gaps and matters needing a person's decision."
    return question, policy, contract
