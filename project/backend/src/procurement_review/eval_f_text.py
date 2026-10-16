"""Independent whole-field boundary adjudication for literal text values."""

import re


def whole_field(field, value, source):
    if field not in ("standard", "responsibility", "damages", "cap", "other"):
        return True
    if not isinstance(value, str) or value not in source:
        return False
    labels = (
        r"acceptance\s+standard",
        r"standard",
        r"warranty\s+responsibility",
        r"warranty",
        r"liability\s+cap",
        r"liability\s+limit",
        r"damages",
        r"other\s+applicable\s+terms",
        r"applicable\s+rights",
        "验收标准",
        "质保责任",
        "累计赔偿责任上限",
        "责任上限",
        "上限",
        "违约责任",
        "赔偿",
        "赔付",
        "变更程序",
    )
    headings = {
        "standard": (r"(?:acceptance[ _])?standard", r"(?:验收)?标准"),
        "responsibility": (r"warranty(?:[ _]responsibility)?", "质保责任"),
        "damages": ("damages", "违约责任", "赔付", "赔偿"),
        "cap": (r"liability[ _](?:limit|cap)", "cap", r"(?:累计赔偿)?(?:责任)?上限"),
        "other": (r"other[ _]applicable[ _]terms", r"applicable[ _]rights"),
    }
    for line in source.splitlines():
        start = 0
        while (position := line.find(value, start)) >= 0:
            prefix = line[:position]
            marked = any(
                re.search(label + r"\s*(为|[:：=])?\s*$", prefix, re.I)
                for label in headings[field]
            )
            named_in_value = (
                field in ("damages", "other")
                and re.match(
                    r"(逾期|本次只审|本例仅适用|分包|知识产权|争议|ONLY_REVIEW|SUBCONTRACT|DISPUTE)",
                    value,
                    re.I,
                )
                and (position == 0 or prefix[-1] in "。；;，,|")
            )
            if not (marked or named_in_value):
                start = position + 1
                continue
            remainder = line[position + len(value) :].strip()
            if remainder in ("", ".", "。", ";", "；"):
                return True
            if remainder.startswith("。") or re.match(r"\.[^0-9]", remainder):
                return True
            if value.endswith("。") or (
                value.endswith(".") and not remainder[0].isdigit()
            ):
                return True
            if remainder[0] in ",，;；":
                following = remainder[1:].lstrip()
                if re.match(r"(累计赔偿)?(责任)?上限\s*(为|[:：=])?", following):
                    return True
                if any(
                    re.match(label + r"\s*(?:[:：=]|为)", following, re.I)
                    for label in labels
                ):
                    return True
            start = position + 1
    return False


# G v3 uses whole statements. These helpers are deliberately separate from the F
# adjudicator above: accepting an alias never deletes an actor, condition or negation.
def statement_v3(value):
    import unicodedata

    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFKC", value).strip().strip('"“”')
    return re.sub(r"\s+", "", text).rstrip("。.!！")


def semantic_value_v3(field, value):
    """A bounded, whole-field grammar; unmatched or extra words remain significant."""
    import json
    from decimal import Decimal

    if field == "amount":
        text = str(value).strip()
        if type(value) is bool or not re.fullmatch(
            r"(?:\d+|[1-9]\d{0,2}(?:,\d{3})+)(?:\.\d+)?", text
        ):
            return None
        return ("money", Decimal(text.replace(",", "")))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            pass
    if isinstance(value, list):
        aliases = {
            "采购": "purchase",
            "服务": "service",
            "租赁": "lease",
            "唯一来源": "single_source",
            "书面说明": "written_justification",
            "审批授权": "approval_reference",
        }
        items = [aliases.get(item, item) for item in value if isinstance(item, str)]
        if len(items) != len(value) or len(set(items)) != len(items):
            return None
        return ("set", tuple(sorted(items)))
    if type(value) is bool:
        return ("bool", value)
    if type(value) in (int, float):
        return ("number", Decimal(str(value)))
    text = statement_v3(value)
    if text is None:
        return None
    if field == "cap":
        match = re.fullmatch(
            r"((?:\d+|[1-9]\d{0,2}(?:,\d{3})+)(?:\.\d+)?)\s*(CNY|USD|人民币|元)", text
        )
        if match:
            return (
                "cap",
                Decimal(match[1].replace(",", "")),
                "CNY" if match[2] in {"人民币", "元"} else match[2],
            )
    if field == "special":
        aliases = {"交付": "delivery", "验收": "acceptance", "质保": "warranty"}
        terms = []
        for part in re.split("[;；]", text):
            match = re.fullmatch(
                r"(delivery|acceptance|warranty|交付|验收|质保)(<=|>=|≤|≥|不超过|至少)(\d+)(?:日|天)?",
                part,
            )
            if not match:
                return None
            name = aliases.get(match[1], match[1])
            relation = "<=" if match[2] in {"<=", "≤", "不超过"} else ">="
            terms.append((name, relation, int(match[3])))
        if len(terms) != 3 or len({term[0] for term in terms}) != 3:
            return None
        return ("limits", tuple(sorted(terms)))
    if field == "responsibility":
        text = text.replace("供应方", "卖方").replace("供货方", "卖方")
        text = text.replace("修理", "维修").replace("损坏设备", "故障设备")
        text = text.replace("自担费用", "自费").replace("自行承担费用", "自费")
        if text in {
            "卖方自费维修故障设备",
            "卖方负责维修故障设备并承担费用",
            "故障设备由卖方维修且费用由卖方承担",
        }:
            return (
                "responsibility",
                "seller",
                "repair",
                "faulty_equipment",
                "seller_cost",
                True,
            )
    aliases = {
        "standard": (
            ("所有设备", "全部设备"),
            ("约定检验", "约定检验"),
            ("约定检测", "约定检验"),
        ),
        "damages": (("延期交付", "逾期交付"), ("每天", "每日"), ("赔偿", "赔付")),
        "basis": (("本办法适用于", "适用于"), ("本制度适用于", "适用于")),
        "category": (("采购", "purchase"), ("租赁", "lease"), ("服务", "service")),
    }
    for old, new in aliases.get(field, ()):
        text = text.replace(old, new)
    return ("text", text)


def claim_supported_v3(descriptor, value, excerpt):
    """Equality of full semantics plus the complete authored supporting clause(s)."""
    field = descriptor["field"]
    wanted = semantic_value_v3(field, descriptor["value"])
    actual = semantic_value_v3(field, value)
    failures = []
    if wanted is None or actual != wanted:
        failures.append("semantic_value_mismatch")
    normalized = statement_v3(excerpt) or ""
    clauses = descriptor.get("clauses", [])
    if not clauses or any(
        (statement_v3(clause) or "\x00") not in normalized for clause in clauses
    ):
        failures.append("excerpt_does_not_prove_field")
    return {
        "kind": "whole_semantics_v3",
        "failures": failures,
        "value_ok": "semantic_value_mismatch" not in failures,
        "support_ok": "excerpt_does_not_prove_field" not in failures,
        "ok": not failures,
    }
