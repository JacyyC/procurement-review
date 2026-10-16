"""Independent source adjudication for F. Never imports production aggregators,
field support recognizers, or document value extraction. Frozen truth is separate.
"""

import hashlib
import json
import re

from .gold_v3_contracts import payload_sha256
from .eval_f_english import literal_english


def digest_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def source_identity(doc):
    return dict(
        source_id=doc.source_id,
        version=doc.version,
        locator=doc.section,
        content_sha256=doc.content_sha256,
    )


def matches_value(field, value, text, worker):
    from .eval_f_text import whole_field

    if not whole_field(field, value, text):
        return False
    """Conservative independent comparison to the cited literal content.

    It accepts labelled source expressions, not a model's own reason or status.
    Unrecognized/ambiguous prose is a failed adjudication, never automatic credit.
    """
    if not current_commitment(text):
        return False
    if re.search(
        r"假设|示例|举例|并非|并不|不是|是否|尚未|未约定|未提供|可能|或许|已废止|已失效|[？?]|\b(?:hypothetical|example|illustration|illustrative|uncertain|suppose|may|might|could|expired|repealed)\b|\bnot\s+(?!(?:exceed|applicable|above|below|less|more|book)\b)",
        text,
        re.I,
    ):
        return False
    if field not in {"applies", "basis", "exempt", "special", "other"} and re.search(
        r"不适用|豁免|not\s+applicable|exempt", text, re.I
    ):
        return False
    negative_scope = text
    if worker == "contract" and field == "other" and isinstance(value, str):
        for clause in re.finditer(
            r"(?:Other applicable terms|Applicable rights)\s*[:=]\s*"
            r"(?P<body>[^.;\n]+;\s*no other (duties|terms|obligations) apply)(?=\.|\n|$)",
            text,
            re.I,
        ):
            terms = clause["body"].split(";", 1)[0].strip()
            if len(terms) < 4 or re.search(
                r"\b(none|unknown|unavailable)\b", terms, re.I
            ):
                continue
            if value in (clause["body"], clause["body"] + ".") and value in text:
                # Only this field's complete exhaustive declaration is exempt
                # from the generic negative-word check. Other negations remain.
                negative_scope = negative_scope.replace(
                    clause["body"], clause["body"].split(";")[0]
                )
    if re.search(
        r"\bno\s+(?!(more than|fewer than|earlier than|charge|additional)\b)",
        negative_scope,
        re.I,
    ):
        return False
    if field in {"delivery", "acceptance", "warranty", "quotes", "special"}:
        names = {
            "delivery": "delivery|交付",
            "acceptance": "acceptance|验收",
            "warranty": "warranty|质保",
            "quotes": "quotes?|报价",
            "special": "delivery|acceptance|warranty|交付|验收|质保",
        }
        if re.search(
            "(?:"
            + names[field]
            + r")\s*((days|period|must last|must take|shall last|shall take|is required|to last)\s*)*((不少于|至少|不超过|至多|>=|<=|at least|at most|no more than|no fewer than|within|[:：=])\s*)?([0-9]+\.[0-9]+|[0-9]+\s*(to|[-–])\s*[0-9]+)",
            text,
            re.I,
        ):
            return False
    if field == "special" and isinstance(value, str) and worker == "policy":
        limits = re.fullmatch(
            r"delivery<=([0-9]+);acceptance<=([0-9]+);warranty>=([0-9]+)", value
        )
        if limits:
            return all(
                matches_value(name, int(number), text, worker)
                for name, number in zip(
                    ("delivery", "acceptance", "warranty"), limits.groups()
                )
            )
    if literal_english(field, value, text, worker):
        return True
    labels = {
        "buyer": "买方|BUYER",
        "seller": "卖方|SELLER",
        "subject": "标的|SUBJECT",
        "order": "订单|ORDER",
        "contract": "合同|CONTRACT",
        "delivery": "交付|DELIVERY",
        "acceptance": "验收|ACCEPTANCE",
        "warranty": "质保|WARRANTY",
    }
    if field in ("delivery", "acceptance", "warranty", "quotes"):
        label = "报价|QUOTES" if field == "quotes" else labels[field]
        op = (
            (
                r"(?:不少于|至少|>=|at least)"
                if field in ("quotes", "warranty")
                else r"(?:不超过|至多|<=|at most)"
            )
            if worker == "policy"
            else r"[:：=]?"
        )
        unit = (
            r"(?!\d|[./]\d|[eE][+-]?\d|\s*(?:to\b|[-–]|or\b)\s*\d)"
            if field == "quotes"
            else r"\s*(?:日|天|days?\b)"
        )
        found = [
            int(m)
            for m in re.findall(
                r"(?<![a-z_])(?:" + label + r")\s*" + op + r"\s*(\d+)" + unit,
                text,
                re.I,
            )
        ]
        return type(value) is int and bool(found) and set(found) == {value}
    if field in ("buyer", "seller", "subject", "order", "contract"):
        return isinstance(value, str) and bool(
            re.search(
                r"(?:"
                + labels[field]
                + r")\s*[:：=]?\s*"
                + re.escape(value)
                + r"(?=$|[\s,，;；。.|])",
                text,
                re.I,
            )
        )
    if field == "applies":
        negative = bool(re.search(r"不适用|豁免|not applicable|exempt", text, re.I))
        positive = bool(re.search(r"适用于|applies to", text, re.I))
        return type(value) is bool and (
            positive and not negative if value else negative and not positive
        )
    if field == "exempt":
        try:
            values = json.loads(value)
        except (ValueError, TypeError):
            return False
        return (values == [] and not re.search(r"豁免|exempt|不适用", text, re.I)) or (
            bool(values)
            and all(v in text for v in values)
            and bool(re.search(r"豁免|exempt", text, re.I))
        )
    if field == "basis":
        return (
            isinstance(value, str)
            and value in text
            and bool(re.search(r"2026|applicab|适用", value, re.I))
        )
    if field == "supplier":
        return value == "all_qualified" and all(
            re.search(p, text, re.I)
            for p in [
                r"(登记|registry)\s*[:=]?\s*active",
                r"(采购|procurement)\s*[:=]?\s*approved",
                r"(尽调|due_diligence)\s*[:=]?\s*verified",
                r"(筛查|screening)\s*[:=]?\s*clear",
                r"税号存在|tax.{0,12}registered",
            ]
        )
    if field == "money":
        return value == "equal_invoice_capped_tx" and all(
            re.search(p, text, re.I)
            for p in [
                r"发票金额等于订单金额|订单与发票等额|order.{0,12}equals.{0,12}invoice",
                r"交易累计(?:不|不得)超过订单|transactions?.{0,12}(?:<=|not exceed).{0,12}order",
                r"币种必须相同|同币|same.currency",
            ]
        )
    if field == "dates":
        return value == "booked_ge_ordered" and bool(
            re.search(
                r"记账(?:日期)?(?:不|不得)早于下单|booked.{0,12}>=.{0,12}ordered",
                text,
                re.I,
            )
        )
    if field == "special":
        return (
            isinstance(value, str)
            and value in text
            and all(
                re.search(x, value, re.I)
                for x in ("交付|delivery", "验收|acceptance", "质保|warranty")
            )
            and bool(re.search(r"\d", value))
        )
    if field in ("amount", "currency"):
        found = re.findall(
            r"(?:价款记录：订单|价款|PRICE)\s*[:=]?\s*(\d+\.\d{2})\s+(CNY|USD)",
            text,
            re.I,
        )
        return bool(found) and {p[0 if field == "amount" else 1] for p in found} == {
            value
        }
    if field == "written":
        return type(value) is bool and (
            bool(
                re.search(
                    r"变更须双方书面签署|双方书面签署后变更生效|CHANGE=effective after both parties sign in writing",
                    text,
                    re.I,
                )
            )
            if value
            else bool(re.search(r"无需书面|without.{0,8}writing", text, re.I))
        )
    markers = {
        "standard": "验收标准|STANDARD",
        "responsibility": "质保责任|WARRANTY_RESPONSIBILITY",
        "damages": "违约责任|逾期|DAMAGES",
        "cap": "责任上限|上限|LIABILITY_LIMIT|CAP",
        "other": "分包|知识产权|争议|本次只审|ONLY_REVIEW|SUBCONTRACT|DISPUTE",
    }
    if field in markers:
        if not isinstance(value, str) or value not in text or len(value) < 4:
            return False
        if field in ("damages", "cap") and not re.search(r"\d", value):
            return False
        return bool(
            re.search(
                r"(?:" + markers[field] + r")\s*(?:为|[:：=])?\s*" + re.escape(value),
                text,
                re.I,
            )
            or (
                field in ("damages", "other") and re.search(markers[field], value, re.I)
            )
        )
    return False


def current_commitment(text):
    """Independent adjudication of explicit withdrawal and modality qualifiers.

    No production normalizer is used to establish scoring truth. These are
    conservative negative observations, not a general natural-language oracle.
    """
    normalized = text.casefold()
    words = set(re.findall(r"[a-z]+(?:[-][a-z]+)?", normalized))
    if words & {
        "optional",
        "nonbinding",
        "non-binding",
        "revoked",
        "withdrawn",
        "cancelled",
        "canceled",
        "draft",
        "unless",
        "except",
        "alternatively",
        "approximately",
    }:
        return False
    if any(
        term in normalized
        for term in (
            "可选",
            "仅供参考",
            "尚未生效",
            "已废止",
            "已失效",
            "仅作举例",
            "no longer",
            "除非",
        )
    ):
        return False
    patterns = (
        r"不构成.{0,8}(强制|约束)",
        r"(条款|要求|规定).{0,8}(撤销|取消|撤回)",
        r"(明年|下年|下一年度).{0,12}生效|生效.{0,12}(明年|下年|下一年度)",
        r"non\s+binding|for\s+(guidance|reference)\s+only",
        r"not\s+(yet\s+effective|required|binding)\b",
        r"(takes?\s+effect|effective\s+from)\s+next\b",
        r"\bsubject\s+to\b",
        r"(日|天|\bdays?\b)\s*(或|or\b)\s*[0-9]",
    )
    return not any(re.search(pattern, normalized) for pattern in patterns)


def check_span(case, span):
    from .eval_f_text import whole_field

    docs = {
        d.source_id: d
        for d in (case.model_input.policy_document, case.model_input.contract_document)
    }
    doc = docs.get(span.evidence.source_id)
    if doc is None or span.evidence.model_dump() != source_identity(doc):
        return False
    p = span.proposition
    if (p.scope_id, p.subject_type, p.subject_id, p.effective_at, p.source_version) != (
        case.model_input.business_facts.scope_id,
        "purchase_order",
        case.model_input.business_facts.purchase_order_id,
        case.as_of,
        doc.version,
    ):
        return False
    worker = "policy" if doc == case.model_input.policy_document else "contract"
    parts = p.field_path.split(".")
    if len(parts) != 3 or parts[:2] != [worker, p.check_id.lower().replace("-", "_")]:
        return False
    owners = {
        "O-01": ("policy", {"applies", "basis", "exempt"}),
        "O-02": (
            "policy",
            {
                "special",
                "quotes",
                "delivery",
                "acceptance",
                "warranty",
                "supplier",
                "money",
                "dates",
            },
        ),
        "O-05": ("contract", {"buyer", "seller", "subject", "order", "contract"}),
        "O-06": ("contract", {"amount", "currency"}),
        "O-08": (
            "contract",
            {"delivery", "acceptance", "warranty", "standard", "responsibility"},
        ),
        "O-09": ("contract", {"written", "damages", "cap", "other"}),
    }
    if (
        p.check_id not in owners
        or worker != owners[p.check_id][0]
        or parts[-1] not in owners[p.check_id][1]
    ):
        return False
    if not 0 <= span.start < span.end <= len(doc.content):
        return False
    clause = doc.content[span.start : span.end]
    # Include the surrounding paragraph so a clipped witness cannot hide a
    # following nonbinding, revoked, or future-effective qualification.
    start = doc.content.rfind("\n", 0, span.start) + 1
    end = doc.content.find("\n", span.end)
    if not whole_field(
        parts[-1], span.value, doc.content[start : end if end >= 0 else None]
    ):
        return False
    if not current_commitment(doc.content[start : end if end >= 0 else None]):
        return False
    if parts[-1] not in {
        "applies",
        "basis",
        "exempt",
        "special",
        "other",
    } and re.search(
        r"不适用|豁免|not\s+applicable|exempt",
        doc.content[start : end if end >= 0 else None],
        re.I,
    ):
        return False
    boundaries = "\n。；;，,|"
    if span.start and doc.content[span.start - 1] not in boundaries:
        return False
    if (
        span.end < len(doc.content)
        and doc.content[span.end] not in boundaries
        and doc.content[span.end - 1] not in boundaries
    ):
        return False
    return digest_text(clause) == span.span_sha256 and matches_value(
        parts[-1], span.value, clause, worker
    )


def fact_catalog(case):
    """Reconstruct exact tool rows directly from the source record for comparison."""
    f = case.model_input.business_facts.model_dump(mode="json")
    rows = {
        "purchase_order": {
            k: f[k]
            for k in (
                "purchase_order_id",
                "supplier_id",
                "competitive_quote_count",
                "contract_reference",
                "ordered_on",
                "invoice_amount_minor",
                "invoice_currency",
            )
        },
        "supplier": {
            k: f[k]
            for k in (
                "supplier_id",
                "registry_status",
                "procurement_status",
                "tax_identifier",
                "due_diligence_status",
                "sanctions_screening",
            )
        },
        "historical_transaction": {
            k: f[k]
            for k in (
                "transaction_id",
                "purchase_order_id",
                "supplier_id",
                "counterparty_reference",
                "booked_on",
            )
        },
    }
    rows["purchase_order"].update(
        amount_minor=f["order_amount_minor"], currency=f["order_currency"]
    )
    rows["historical_transaction"].update(
        amount_minor=f["transaction_amount_minor"], currency=f["transaction_currency"]
    )
    ids = {
        "purchase_order": f["purchase_order_id"],
        "supplier": f["supplier_id"],
        "historical_transaction": f["transaction_id"],
    }
    return {ids[k]: (k, v, f["source_version"]) for k, v in rows.items()}


def evidence_valid(case, evidence, decision):
    for doc in (case.model_input.policy_document, case.model_input.contract_document):
        if evidence.model_dump() == source_identity(doc):
            return True
    facts = fact_catalog(case)
    if evidence.source_id in facts:
        kind, row, version = facts[evidence.source_id]
        prefix, _, field = evidence.locator.partition(".")
        return (
            prefix == kind
            and (field in row or field == "identity")
            and evidence.version == version
            and evidence.content_sha256 == payload_sha256(row)
        )
    if evidence.source_id == "trusted-review-control":
        refs = []
        for worker, doc in [
            ("policy", case.model_input.policy_document),
            ("contract", case.model_input.contract_document),
        ]:
            # Same public identity definition, not the production support function.
            from .model_wire_v2 import SourceSlice
            from .business_v3_types import BusinessEvidenceIdentity

            refs.append(
                SourceSlice(
                    scope_id=case.model_input.business_facts.scope_id,
                    worker=worker,
                    identity=BusinessEvidenceIdentity(**source_identity(doc)),
                    content=doc.content,
                ).ref
            )
        return (
            evidence.version == "business-review.v3"
            and evidence.locator == "source-and-output-inspection"
            and evidence.content_sha256
            == payload_sha256(
                {
                    "sources": refs,
                    "events": [s.model_dump(mode="json") for s in decision.security],
                }
            )
        )
    return False


def fact_observation_valid(case, observation):
    """A conflict side must cite its own exact fact field, object, time and value."""
    proposition = observation.proposition
    record = fact_catalog(case).get(proposition.subject_id)
    if record is None or not observation.evidence:
        return False
    kind, fields, version = record
    expected_kind = "transaction" if kind == "historical_transaction" else kind
    prefix, separator, field = proposition.field_path.partition(".")
    duty = {
        "currency": "O-06",
        "invoice_currency": "O-06",
        "amount_minor": "O-06",
        "invoice_amount_minor": "O-06",
        "ordered_on": "O-07",
        "booked_on": "O-07",
        "registry_status": "O-03",
        "procurement_status": "O-03",
        "tax_identifier": "O-03",
        "due_diligence_status": "O-03",
        "sanctions_screening": "O-03",
        "competitive_quote_count": "O-04",
        "contract_reference": "O-04",
        "counterparty_reference": "O-04",
    }.get(field)
    if (
        prefix != expected_kind
        or not separator
        or field not in fields
        or proposition.scope_id != case.model_input.business_facts.scope_id
        or proposition.subject_type != expected_kind
        or proposition.effective_at != case.as_of
        or proposition.source_version != version
        or duty is None
        or proposition.check_id != duty
        or type(observation.value) is not type(fields[field])
        or observation.value != fields[field]
    ):
        return False
    return all(
        e.source_id == proposition.subject_id
        and e.version == version
        and e.locator == kind + "." + field
        and e.content_sha256 == payload_sha256(fields)
        for e in observation.evidence
    )
