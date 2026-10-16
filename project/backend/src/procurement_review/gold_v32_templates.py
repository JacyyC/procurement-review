"""Replacement validation document families, authored before candidate repair.

These are source documents, never prompts or expected answers. The retired
control-sheet/register families are not recycled by changing IDs or amounts.
Do not import these renderers into production extraction or independent scoring.
"""

FAMILY_NAMES = ("memo", "agreement", "schedule", "letter", "handbook", "minutes")


def validation_documents(style, facts):
    """Six independently authored presentations of the same frozen obligations."""
    headings = (
        ("Procurement committee memorandum", "Executed supply memorandum"),
        ("Purchasing governance agreement", "Mutual delivery agreement"),
        ("Operational purchasing schedule", "Signed performance schedule"),
        ("Notice to purchasing officers", "Letter of supply commitments"),
        ("Purchasing control handbook extract", "Supply conditions extract"),
        ("Purchasing committee resolutions", "Recorded contractual commitments"),
    )
    policy_openings = (
        "This memorandum applies to this purchase during 2026. The following controls govern the review.",
        "During 2026 this agreement governs this purchase. Each control below is binding on the purchaser.",
        "Scope of the schedule: this purchase in 2026. The controls listed here are mandatory.",
        "For this purchase in 2026, purchasing officers shall apply the controls in this notice.",
        "Applicability: this purchase, calendar year 2026. The handbook establishes these mandatory controls.",
        "The committee adopts the following binding controls for this purchase in 2026.",
    )
    supplier = (
        "Supplier eligibility requires registry active, procurement approved, a registered tax identifier, due diligence verified and screening clear.",
        "All supplier conditions must hold: registry=active; procurement=approved; tax identifier registered; due diligence=verified; screening=clear.",
        "Supplier admission: active registry; approved procurement status; registered tax identifier; verified due diligence; clear screening.",
        "Use a supplier only when its registry is active, procurement status approved, tax identifier registered, due diligence verified, and screening clear.",
        "Supplier controls specify registry active and procurement approved, together with registered tax identification, verified due diligence and clear screening.",
        "The supplier must have active registry status, approved procurement status, a registered tax identifier, verified due diligence and clear screening.",
    )
    quotes = (
        "At least 3 competitive quotes are required, with contract and counterparty references present.",
        "The minimum competitive quote count is 3. Both contract reference and counterparty reference are required.",
        "Quote count >= 3; contract reference and counterparty reference must be present.",
        "Obtain no fewer than 3 competitive quotes and retain contract and counterparty references.",
        "Three controls apply to sourcing: competitive quotes >= 3, contract reference present, counterparty reference present.",
        "The committee requires a minimum of 3 competitive quotes and both contract and counterparty references.",
    )
    money = (
        "Order and invoice amounts must be equal; cumulative transactions must not exceed the order amount; all three currencies must match.",
        "Amounts: order equals invoice; transaction total <= order. Currencies: order, invoice and transactions use the same currency.",
        "Monetary control requires invoice amount equal to order amount, total transactions at most the order amount, and matching currencies.",
        "Match the invoice amount to the order amount. Cap cumulative transactions at the order amount. Use one common currency throughout.",
        "The order amount equals the invoice amount, cumulative transactions do not exceed that order amount, and all currencies are identical.",
        "Resolve monetary review by requiring order/invoice equality, transaction total bounded by the order amount, and currency equality across all records.",
    )
    dates = (
        "The booked date must be on or after the ordered date.",
        "Date control: booked_on >= ordered_on.",
        "Booking shall occur no earlier than the order date.",
        "Do not book a transaction before its order date.",
        "The booking date is required to equal or follow the ordering date.",
        "The ordered date is the earliest permitted booked date.",
    )
    service = (
        "Delivery within 15 days; acceptance within 7 days; warranty at least 90 days.",
        "Delivery <= 15 days; acceptance <= 7 days; warranty >= 90 days.",
        "Maximum delivery 15 days, maximum acceptance 7 days, minimum warranty 90 days.",
        "Delivery must take no more than 15 days. Acceptance must take no more than 7 days. Warranty must last at least 90 days.",
        "Service controls: delivery capped at 15 days; acceptance capped at 7 days; warranty floor 90 days.",
        "The delivery limit is 15 days and acceptance limit is 7 days; the minimum warranty period is 90 days.",
    )
    # Titles, ordering, connectors and clause structures form distinct source
    # families. All remain complete business materials under the frozen Spec.
    sections = [
        supplier[style],
        quotes[style],
        money[style],
        dates[style],
        service[style],
    ]
    orders = (
        (0, 1, 2, 3, 4),
        (4, 2, 0, 3, 1),
        (1, 3, 4, 0, 2),
        (3, 0, 2, 1, 4),
        (2, 4, 1, 3, 0),
        (4, 3, 2, 1, 0),
    )
    policy = "\n".join(
        [
            headings[style][0],
            policy_openings[style],
            *(sections[i] for i in orders[style]),
        ]
    )
    amount = f"{facts.order_amount_minor // 100}.{facts.order_amount_minor % 100:02}"
    identity = (
        f"Buyer: {facts.counterparty_reference}; seller: {facts.supplier_id}; subject: replacement laboratory instruments.\n"
        f"Order: {facts.purchase_order_id}; contract: {facts.contract_reference}."
    )
    price = f"Price: {amount} {facts.order_currency}."
    periods = f"Delivery: {facts.delivery_days} days; acceptance: {facts.acceptance_days} days; warranty: {facts.warranty_days} days."
    terms = (
        "Acceptance standard: every ordered unit delivered without damage.\n"
        "Warranty responsibility: seller repairs or replaces defects free of charge.\n"
        "Changes become effective only after both parties sign in writing.\n"
        "Damages: late delivery costs 0.1% of the order price per day.\n"
        "Liability cap: 100000.00 CNY.\n"
        "Applicable rights: written changes, damages and liability cap only."
    )
    framing = (
        "Both parties record their commitments in the following memorandum.",
        "The parties execute this agreement and accept its operative terms.",
        "This schedule sets the contractual identity, consideration and performance terms.",
        "We confirm the supply commitments stated in this signed letter.",
        "These extracted conditions form part of the parties' executed contract.",
        "The signed minutes record the parties' final supply commitments.",
    )
    blocks = (identity, price, periods, terms)
    contract = "\n".join([headings[style][1], framing[style], *blocks])
    question = "Review this purchase using the supplied policy, contractual commitments and authorized procurement facts. Identify supported findings and genuine missing information."
    return question, policy, contract
