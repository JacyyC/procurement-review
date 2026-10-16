"""Reject a text extraction that stops inside its operative field clause."""

import re


NEXT_FIELD = (
    r"(?:(?:acceptance\s+)?standard|warranty(?:\s+responsibility)?|"
    r"liability\s+(?:cap|limit)|damages|other\s+applicable\s+terms|"
    r"applicable\s+rights|验收标准|质保责任|违约责任|赔偿|赔付|变更程序)\s*(?:为|[:：=])"
    r"|(?:累计赔偿)?(?:责任)?上限\s*(?:为|[:：=])?"
)


def complete_text_value(field, value, source):
    if field not in {"standard", "responsibility", "damages", "cap", "other"}:
        return True
    if not isinstance(value, str) or not value:
        return False
    labels = {
        "standard": r"(?:验收)?标准|(?:acceptance[ _])?standard",
        "responsibility": r"质保责任|warranty(?:[ _]responsibility)?",
        "damages": r"违约责任|赔偿|赔付|damages",
        "cap": r"(?:累计赔偿)?(?:责任)?上限|liability[ _](?:limit|cap)|cap",
        "other": r"other[ _]applicable[ _]terms|applicable[ _]rights",
    }
    for match in re.finditer(re.escape(value), source):
        prefix = source[: match.start()]
        labelled = re.search(
            r"(?:" + labels[field] + r")\s*(?:为|[:：=])?\s*$", prefix, re.I
        )
        # Historical Chinese source clauses can name their own field without a
        # separate heading. They must start at an actual clause boundary.
        self_labelled = (
            field in {"damages", "other"}
            and re.match(
                r"逾期|本次只审|本例仅适用|分包|知识产权|争议|ONLY_REVIEW|SUBCONTRACT|DISPUTE",
                value,
                re.I,
            )
            and (not prefix or prefix[-1] in "\n。；;，,|")
        )
        if not (labelled or self_labelled):
            continue
        # Stop at the end of this paragraph or at an explicitly labelled next
        # field. A comma/semicolon alone cannot erase a calculation or qualifier.
        tail = source[match.end() :].split("\n", 1)[0].strip()
        if not tail or tail in {".", "。", ";", "；"}:
            return True
        if tail.startswith("。") or re.match(r"^\.(?!\d)", tail):
            return True
        if value.endswith("。") or (value.endswith(".") and not tail[0].isdigit()):
            return True
        if re.match(r"^[,，;；]\s*(?:" + NEXT_FIELD + ")", tail, re.I):
            return True
    return False
