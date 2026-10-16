"""Independent English source adjudication. No production rule imports."""

import re


def literal_english(field, value, source, worker):
    text = re.sub(r"(?<=[a-z])[_-](?=[a-z])", " ", source.lower())
    text = text.replace("≥", ">=").replace("≤", "<=")
    if field == "special" and worker == "policy":
        if not isinstance(value, str):
            return False
        items = value.split(";")
        if len(items) != 3:
            return False
        values = []
        for item, name, relation in zip(
            items, ("delivery", "acceptance", "warranty"), ("<=", "<=", ">=")
        ):
            prefix = name + relation
            if not item.startswith(prefix) or not item[len(prefix) :].isdigit():
                return False
            values.append((name, int(item[len(prefix) :])))
        return all(
            literal_english(name, number, source, worker) for name, number in values
        )
    if field in {"quotes", "delivery", "acceptance", "warranty"}:
        if type(value) is not int:
            return False
        label = (
            "(?:competitive )?quotes?(?: count)?"
            if field == "quotes"
            else field + "(?: days| period)?"
        )
        measure = (
            r"\s*days?\b"
            if field != "quotes"
            else r"(?!\d|[./]\d|[eE][+-]?\d|\s*(?:to\b|[-–]|or\b)\s*\d)"
        )
        if worker == "contract":
            expressions = [rf"\b{label}\s*[:=]\s*(\d+){measure}"]
        else:
            ge = field in {"quotes", "warranty"}
            relation = (
                ">=|at least|minimum(?: of)?|no fewer than|not less than|floor"
                if ge
                else "<=|at most|maximum|within|no more than|not above|capped at|limit(?: is)?"
            )
            expressions = [
                rf"\b{label}\s*(?:(?:must|shall) (?:be|last|take)|is required(?: to last)?|count is|period is)?\s*(?:{relation})\s*(\d+){measure}",
                rf"(?:{relation})\s*(\d+){measure}\s*(?:of\s*)?{label}\b",
                rf"{'minimum' if ge else 'maximum'}\s+(?:of\s+)?{label}\s*(?:count|period)?\s*(?:is)?\s*(\d+){measure}",
            ]
        found = {
            int(n) for expression in expressions for n in re.findall(expression, text)
        }
        return found == {value}
    if field == "applies":
        scope = any(
            re.search(p, text)
            for p in (
                r"(?:apply|applies) to.{0,60}purchase",
                r"applicability.{0,60}(purchase|2026)",
                r"scope.{0,60}purchase",
                r"governs.{0,60}purchase",
                r"purchase.{0,90}shall apply",
                r"binding controls.{0,60}purchase",
            )
        )
        excluded = bool(re.search(r"not applicable|exempt", text))
        return type(value) is bool and (
            scope and not excluded if value else excluded and not scope
        )
    if field == "supplier":
        checks = []
        for noun, status in (
            ("registry", "active"),
            ("procurement", "approved"),
            ("due diligence", "verified"),
            ("screening", "clear"),
        ):
            checks.append(
                bool(
                    re.search(
                        rf"\b{noun}\s*(status\s*)?([:=]|must be|shall be|is required to be|is|remains)?\s*{status}\b|\b{status}\s+{noun}\b",
                        text,
                    )
                )
            )
        return (
            value == "all_qualified"
            and all(checks)
            and bool(
                re.search(r"tax[^.;\n]{0,20}(registered|present)|registered tax", text)
            )
        )
    if field == "money":
        checks = (
            r"order.{0,25}\b(equal|equals|equality)\b.{0,25}invoice|invoice.{0,25}\b(equal|equals|match)\b.{0,25}order|order and invoice.{0,25}\bequal\b|match the invoice amount to the order|order/invoice equality",
            r"transaction[s]?.{0,40}(<=|not exceed|not above|at most|do not exceed|bounded by).{0,25}order|cap cumulative transactions at the order",
            r"currenc(y|ies).{0,35}(same|equal|match|identical)|(same|common|matching) currenc|currenc(y|ies) equality",
        )
        return value == "equal_invoice_capped_tx" and all(
            re.search(p, text) for p in checks
        )
    if field == "dates":
        checks = (
            r"book(ed|ing)( date| on| shall occur)?\s*(must be|is required to|shall be)?\s*(>=|on or after|no earlier than|equal or follow).{0,20}order",
            r"do not book.{0,35}before.{0,15}order",
            r"ordered date.{0,30}earliest.{0,20}booked",
        )
        return value == "booked_ge_ordered" and any(re.search(p, text) for p in checks)
    if field == "written":
        affirmative = bool(
            re.search(
                r"changes?\s*[:=]?\s*((becomes? valid|takes? effect|becomes? effective|effective|valid|binding)\s*)?(only\s+)?after both parties sign in writing",
                text,
            )
        )
        return type(value) is bool and value and affirmative
    labels = {
        "standard": r"acceptance(?: standard)?|standard",
        "responsibility": r"warranty(?: responsibility)?",
        "other": r"applicable rights|other applicable terms",
        "cap": r"liability (?:cap|limit)|cap",
    }
    if field in labels:
        if not isinstance(value, str) or value not in source or len(value) < 4:
            return False
        if field == "cap" and not re.search(r"\d", value):
            return False
        if re.search(r"as specified|terms exist", value, re.I):
            return False
        return bool(
            re.search(
                r"(?:" + labels[field] + r")\s*[:=]\s*" + re.escape(value.lower()), text
            )
        )
    return False
