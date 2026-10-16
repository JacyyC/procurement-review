"""Finite English clause support; values remain bound to literal source spans.

This recognizes explicit labelled terms and relational phrases, not business
outcomes. Unrecognized or incomplete statements return False.
"""

import re


def explicit_other_scope(value, text):
    """Keep both a positive term list and its explicit exhaustive qualifier."""
    if not isinstance(value, str):
        return False
    for match in re.finditer(
        r"(?:other applicable terms|applicable rights)\s*[:=]\s*"
        r"([^.;\n]+;\s*no other (?:duties|terms|obligations) apply)(?=[.\n]|$)",
        text,
        re.I,
    ):
        positive = match[1].split(";", 1)[0].strip()
        if len(positive) < 4 or re.search(
            r"\b(?:none|unknown|unavailable)\b", positive, re.I
        ):
            continue
        if value in (match[1], match[1] + ".") and value in text:
            return True
    return False


def english_field(field, value, text, worker):
    clause = re.sub(r"(?<=[a-z])[_-](?=[a-z])", " ", text.casefold())
    clause = clause.replace("≥", ">=").replace("≤", "<=")
    if field == "special" and worker == "policy" and isinstance(value, str):
        limits = re.fullmatch(
            r"delivery<=(\d+);acceptance<=(\d+);warranty>=(\d+)", value
        )
        if limits:
            return all(
                _number(name, int(number), clause, worker)
                for name, number in zip(
                    ("delivery", "acceptance", "warranty"), limits.groups()
                )
            )
    if field in {"delivery", "acceptance", "warranty", "quotes"}:
        return _number(field, value, clause, worker)
    if field == "applies":
        scope = bool(
            re.search(
                r"(?:appl(?:y|ies|icability)|governs?|scope|binding controls).{0,100}(?:purchase|2026)|(?:purchase|2026).{0,100}(?:shall apply|binding controls)",
                clause,
            )
        )
        negative = bool(re.search(r"not applicable|exempt", clause))
        return type(value) is bool and (
            scope and not negative if value else negative and not scope
        )
    if field == "supplier":
        pairs = (
            ("registry", "active"),
            ("procurement", "approved"),
            ("due diligence", "verified"),
            ("screening", "clear"),
        )
        return (
            value == "all_qualified"
            and all(
                re.search(
                    rf"\b{noun}\s*(?:status\s*)?(?:[:=]|must be|shall be|is required to be|is|remains)?\s*{status}\b|\b{status}\s+{noun}\b",
                    clause,
                )
                for noun, status in pairs
            )
            and bool(
                re.search(r"tax.{0,20}(?:registered|present)|registered tax", clause)
            )
        )
    if field == "money":
        equality = re.search(
            r"order.{0,25}(?:equals?|equal|equality).{0,25}invoice|invoice.{0,25}(?:equals?|equal|match).{0,25}order|order and invoice.{0,25}equal|match the invoice amount to the order|order/invoice equality",
            clause,
        )
        ceiling = re.search(
            r"transactions?.{0,40}(?:<=|not exceed|not above|at most|do not exceed|bounded by).{0,25}order|cap cumulative transactions at the order",
            clause,
        )
        currencies = re.search(
            r"currenc(?:y|ies).{0,35}(?:same|equal|match|identical)|(?:same|common|matching) currenc|currenc(?:y|ies) equality",
            clause,
        )
        return value == "equal_invoice_capped_tx" and bool(
            equality and ceiling and currencies
        )
    if field == "dates":
        relation = re.search(
            r"book(?:ed|ing)(?: date| on| shall occur)?\s*(?:must be|is required to|shall be)?\s*(?:>=|on or after|on-or-after|no earlier than|equal or follow).{0,20}order|do not book.{0,35}before.{0,15}order|ordered date.{0,30}earliest.{0,20}booked",
            clause,
        )
        return value == "booked_ge_ordered" and bool(relation)
    if field == "written":
        return (
            type(value) is bool
            and value
            and bool(
                re.search(
                    r"changes?\s*(?:[:=]\s*)?(?:become[s]? valid|takes? effect|becomes? effective|effective|valid|binding)?\s*(?:only\s+)?after both parties sign in writing",
                    clause,
                )
            )
        )
    if field in {"standard", "responsibility", "other", "cap"}:
        markers = {
            "standard": r"acceptance(?: standard)?|standard",
            "responsibility": r"warranty(?: responsibility)?",
            "other": r"applicable rights|other applicable terms",
            "cap": r"liability (?:cap|limit)|cap",
        }
        return (
            isinstance(value, str)
            and len(value) >= 4
            and value in text
            and (field != "cap" or bool(re.search(r"\d", value)))
            and not re.search(r"as specified|terms exist", value, re.I)
            and bool(
                re.search(
                    r"(?:"
                    + markers[field]
                    + r")\s*[:=]\s*"
                    + re.escape(value.casefold()),
                    clause,
                )
            )
        )
    return False


def _number(field, value, clause, worker):
    if type(value) is not int:
        return False
    noun = (
        r"(?:competitive )?quotes?(?: count)?"
        if field == "quotes"
        else field + r"(?: period| days)?"
    )
    # Duration values have a day dimension. A bare number or another unit
    # cannot be adopted as days; do not erase a numeric minus sign either.
    unit = (
        r"\s*days?\b"
        if field != "quotes"
        else r"(?!\d|[./]\d|[eE][+-]?\d|\s*(?:to\b|[-–]|or\b)\s*\d)"
    )
    if worker == "contract":
        patterns = [rf"\b{noun}\s*[:=]\s*(\d+){unit}"]
    else:
        lower = field in {"quotes", "warranty"}
        operation = (
            r">=|at least|no fewer than|not less than|minimum(?: of)?|floor"
            if lower
            else r"<=|at most|no more than|not above|maximum|within|capped at|limit(?: is)?"
        )
        patterns = [
            rf"\b{noun}\s*(?:must (?:be|last|take)|shall (?:be|last|take)|is required(?: to last)?|count is|period is)?\s*(?:{operation})\s*(\d+){unit}",
            rf"\b(?:{operation})\s*(?:{noun}\s*(?:count|period)?\s*(?:is)?\s*)?(\d+){unit}\s*(?:of\s*)?{noun}\b",
            rf"\b{'minimum' if lower else 'maximum'}\s+{noun}\s*(?:period|count)?\s*(?:is)?\s*(\d+){unit}",
        ]
    numbers = {int(m[1]) for pattern in patterns for m in re.finditer(pattern, clause)}
    return numbers == {value}
