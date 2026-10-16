"""Exact source witnesses. Unrecognized prose is unresolved, never entailed by ID."""

from dataclasses import dataclass
from datetime import date
import hashlib
import re
from typing import Annotated

from .business_v3_prose import english_field, explicit_other_scope
from .business_v3_text import complete_text_value

from pydantic import Field

from .business_v3_types import BusinessEvidenceIdentity, BusinessPropositionIdentity
from .business_v3_types import VALUE_MODELS
from .eval_v2_contracts import EvalV2Model
from .model_wire_v2 import SourceSlice


class SupportIssue(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


#: The fields the frozen rules 3.0.0 allow exemption wording in.  A rule set that
#: cites exclusions on purpose passes its own set instead of widening this one.
EXEMPTION_LANGUAGE_FIELDS = frozenset(
    {"applies", "basis", "exempt", "special", "other"}
)


class WitnessV3(EvalV2Model):
    obligation_id: Annotated[str, Field(pattern=r"^O-(0[1-9]|1[0-2])$")]
    field: Annotated[str, Field(pattern=r"^[a-z_]+$", max_length=48)]
    value: (
        Annotated[str, Field(min_length=1, max_length=256)]
        | Annotated[int, Field(ge=0, le=10**18)]
        | bool
    )
    source_ref: Annotated[str, Field(pattern=r"^ref-[0-9a-f]{64}$")]
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]


class VerifiedSpanV3(EvalV2Model):
    evidence: BusinessEvidenceIdentity
    proposition: BusinessPropositionIdentity
    value: str | int | bool
    start: int
    end: int
    span_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


@dataclass(frozen=True)
class LocatedSourceV3:
    """Metadata supplied by the authorized retrieval boundary, never by a model."""

    source: SourceSlice
    subject_type: str
    subject_id: str
    effective_at: date


class SourceCatalogV3:
    def __init__(self, sources):
        self._sources = {}
        locators = {}
        for item in sources:
            source = item.source
            identity = source.identity
            if (
                hashlib.sha256(source.content.encode()).hexdigest()
                != identity.content_sha256
            ):
                raise SupportIssue("source_identity_content_drift")
            key = (
                source.scope_id,
                source.worker,
                identity.source_id,
                identity.version,
                identity.locator,
            )
            if key in locators and locators[key] != source.ref:
                raise SupportIssue("source_identity_content_drift")
            if source.ref in self._sources and self._sources[source.ref] != item:
                raise SupportIssue("source_identity_metadata_drift")
            locators[key] = source.ref
            self._sources[source.ref] = item

    def resolve(self, ref):
        if ref not in self._sources:
            raise SupportIssue("citation_outside_branch")
        return self._sources[ref]

    def verify(self, request, witness, *, supports=None, exemption_fields=None):
        checker = supports or supports_field
        # A rule set that legitimately cites exclusions names its own fields here;
        # the frozen 3.0.0 set stays the default so no existing call changes.
        fields_allowing_exemption = (
            EXEMPTION_LANGUAGE_FIELDS
            if exemption_fields is None
            else frozenset(exemption_fields)
        )
        item = self.resolve(witness.source_ref)
        source = item.source
        allowed = {s.ref for s in request.sources}
        slots = {c.proposition.check_id: c.proposition for c in request.checks}
        if witness.source_ref not in allowed or witness.obligation_id not in slots:
            raise SupportIssue("citation_outside_branch")
        slot = slots[witness.obligation_id]
        if (
            source.scope_id != request.scope_id
            or source.worker != request.worker
            or (item.subject_type, item.subject_id, item.effective_at)
            != (slot.subject_type, slot.subject_id, slot.effective_at)
        ):
            raise SupportIssue("citation_object_time_mismatch")
        if not 0 <= witness.start < witness.end <= len(source.content):
            raise SupportIssue("citation_span_invalid")
        boundaries = "\n。；;，,|"
        if (
            witness.start
            and source.content[witness.start - 1] not in boundaries
            or witness.end != len(source.content)
            and source.content[witness.end] not in boundaries
            and source.content[witness.end - 1] not in boundaries
        ):
            raise SupportIssue("citation_clause_boundary_invalid")
        span = source.content[witness.start : witness.end]
        # A model cannot trim a withdrawal or optionality qualifier from the
        # same source paragraph and turn the remaining words into an obligation.
        paragraph_start = source.content.rfind("\n", 0, witness.start) + 1
        paragraph_end = source.content.find("\n", witness.end)
        paragraph = source.content[
            paragraph_start : paragraph_end if paragraph_end >= 0 else None
        ]
        if not complete_text_value(witness.field, witness.value, paragraph):
            raise SupportIssue("citation_proposition_unsupported")
        if _nonbinding_or_future(paragraph):
            raise SupportIssue("citation_proposition_unsupported")
        if witness.field not in fields_allowing_exemption and re.search(
            r"不适用|豁免|not\s+applicable|exempt", paragraph, re.I
        ):
            raise SupportIssue("citation_proposition_unsupported")
        if not checker(
            request.worker, witness.obligation_id, witness.field, witness.value, span
        ):
            raise SupportIssue("citation_proposition_unsupported")
        proposition = BusinessPropositionIdentity(
            **(
                slot.model_dump()
                | {
                    "field_path": f"{request.worker}.{witness.obligation_id.lower().replace('-', '_')}.{witness.field}",
                    "source_version": source.identity.version,
                }
            )
        )
        return VerifiedSpanV3(
            evidence=BusinessEvidenceIdentity(**source.identity.model_dump()),
            proposition=proposition,
            value=witness.value,
            start=witness.start,
            end=witness.end,
            span_sha256=hashlib.sha256(span.encode()).hexdigest(),
        )


_LABELS = {
    "buyer": ("买方", "BUYER"),
    "seller": ("卖方", "SELLER"),
    "subject": ("标的", "SUBJECT"),
    "order": ("订单", "ORDER"),
    "contract": ("合同", "CONTRACT"),
    "delivery": ("交付", "DELIVERY"),
    "acceptance": ("验收", "ACCEPTANCE"),
    "warranty": ("质保", "WARRANTY"),
}


def _nonbinding_or_future(text):
    """Finite disqualifiers; unresolved timing/modality never proves commitment."""
    return bool(
        re.search(
            r"可选|仅供参考|尚未生效|已废止|已失效|仅作举例|不构成.{0,8}(?:强制|约束)|(?:规定|要求|条款).{0,8}(?:取消|撤销|撤回)"
            r"|(?:明年|下年|下一年度).{0,12}生效|生效.{0,12}(?:明年|下年|下一年度)"
            r"|\b(?:optional|non[- ]?binding|revoked|withdrawn|cancelled|canceled|draft)\b"
            r"|\bno\s+longer\b|\bnot\s+(?:yet\s+effective|required|binding)\b|\bfor\s+(?:reference|guidance)\s+only\b"
            r"|\b(?:takes?\s+effect|effective\s+from)\s+next\b"
            r"|\b(?:unless|except|alternatively|approximately)\b|\bsubject\s+to\b|除非"
            r"|(?:日|天|\bdays?\b)\s*(?:或|or\b)\s*[0-9]",
            text,
            re.I,
        )
    )


def _labelled_number(field, span, policy):
    labels = _LABELS[field]
    prefix = r"(?<![a-z_])(?:" + "|".join(labels) + ")"
    # Directions belong to the policy field, never inferred from the number alone.
    op = (
        r"(?:不少于|至少|>=|at\s+least)"
        if field == "warranty"
        else r"(?:不超过|至多|<=|at\s+most)"
    )
    relation = op if policy else r"(?:[:：=]\s*)?"
    pattern = prefix + r"\s*" + relation + r"\s*([0-9]+)\s*(?:日|天|days?\b)"
    return {int(m[1]) for m in re.finditer(pattern, span, re.I)}


def supports_field(worker, obligation, field, value, span):
    """Finite, conservative clause normalizers for the frozen business contract.

    The model selects and interprets text; this independent check binds its
    normalized value to a labelled proposition at an exact position. Unknown
    wording returns False. No corpus IDs, templates, labels or expected answers.
    """
    if worker not in {"policy", "contract"}:
        return False
    model = VALUE_MODELS[worker].get(obligation)
    if model is None or field not in model.model_fields:
        return False
    if not complete_text_value(field, value, span):
        return False
    if _nonbinding_or_future(span):
        return False
    if re.search(
        r"示例|举例|假设|并非|并不|未约定|未提供|不是|是否|尚未|可能|或许|已废止|已失效|[？?]|\b(?:hypothetical|example|illustration|illustrative|uncertain|suppose|may|might|could|expired|repealed)\b|not\s+(?!(?:exceed|applicable|above|below|less|more|book)\b)",
        span,
        re.I,
    ):
        return False
    if field not in {"applies", "basis", "exempt", "special", "other"} and re.search(
        r"不适用|豁免|not\s+applicable|exempt", span, re.I
    ):
        return False
    negation_span = span
    if worker == "contract" and field == "other" and explicit_other_scope(value, span):
        negation_span = span.replace(value, value.rsplit(";", 1)[0])
    if re.search(
        r"\bno\s+(?!(?:more than|fewer than|earlier than|charge|additional)\b)",
        negation_span,
        re.I,
    ):
        return False
    if field in {"delivery", "acceptance", "warranty", "quotes", "special"}:
        labels = (
            "delivery|acceptance|warranty|交付|验收|质保"
            if field == "special"
            else "|".join(_LABELS.get(field, ("报价", "quotes?")))
        )
        if re.search(
            r"(?:"
            + labels
            + r")\s*(?:(?:days|period|must last|must take|shall last|shall take|is required|to last)\s*)*(?:(?:不少于|至少|不超过|至多|>=|<=|at least|at most|no more than|no fewer than|within|[:：=])\s*)?(?:\d+\.\d+|\d+\s*(?:to|[-–])\s*\d+)",
            span,
            re.I,
        ):
            return False
    if field == "special" and isinstance(value, str):
        limits = re.fullmatch(
            r"delivery<=(\d+);acceptance<=(\d+);warranty>=(\d+)", value
        )
        if limits:
            return all(
                supports_field(worker, obligation, name, int(number), span)
                for name, number in zip(
                    ("delivery", "acceptance", "warranty"), limits.groups()
                )
            )
    if english_field(field, value, span, worker):
        return True
    if worker == "policy" and obligation == "O-01":
        if field == "basis":
            return (
                isinstance(value, str)
                and value in span
                and bool(re.search(r"\d{4}|适用范围|applicab", span, re.I))
            )
        if field == "applies":
            if type(value) is not bool:
                return False
            negative_pattern = r"豁免|不适用(?:于|范围)?|exempt|not\s+applicable"
            negative = bool(re.search(negative_pattern, span, re.I))
            positive = bool(
                re.search(
                    r"(?:适用于|适用范围|applies\s+to)",
                    re.sub(negative_pattern, "", span, flags=re.I),
                    re.I,
                )
            )
            return positive and not negative if value else negative and not positive
        if field == "exempt":
            return (
                value == "[]"
                and not re.search(r"豁免|不适用|exempt", span, re.I)
                or isinstance(value, str)
                and bool(re.fullmatch(r'\["O-0[2-9]"(?:,"O-0[2-9]")*\]', value))
                and all(v in span for v in re.findall(r"O-0[2-9]", value))
                and bool(re.search(r"豁免|exempt", span, re.I))
            )
        return False
    if worker == "policy" and obligation == "O-02":
        if field == "quotes":
            numbers = {
                int(m[1])
                for m in re.finditer(
                    r"(?:报价|quotes?)\s*(?:至少|>=|at\s+least)\s*(\d+)(?!\d|[./]\d|[eE][+-]?\d|\s*(?:to\b|[-–]|or\b)\s*\d)",
                    span,
                    re.I,
                )
            }
            return type(value) is int and numbers == {value}
        if field in {"delivery", "acceptance", "warranty"}:
            return type(value) is int and _labelled_number(field, span, True) == {value}
        if field == "supplier":
            return (
                value == "all_qualified"
                and all(
                    re.search(pattern, span, re.I)
                    for pattern in (
                        r"(?:登记|registry)\s*[:=]?\s*active\b",
                        r"(?:采购|procurement)\s*[:=]?\s*approved\b",
                        r"(?:尽调|due_diligence)\s*[:=]?\s*verified\b",
                        r"(?:筛查|screening)\s*[:=]?\s*clear\b",
                    )
                )
                and bool(
                    re.search(r"税号(?:存在|已登记)|tax.{0,12}registered", span, re.I)
                )
            )
        if field == "money":
            return (
                value == "equal_invoice_capped_tx"
                and bool(
                    re.search(
                        r"订单与发票等额|发票金额等于订单金额|order.{0,12}equals.{0,12}invoice",
                        span,
                        re.I,
                    )
                )
                and bool(
                    re.search(
                        r"交易累计(?:不|不得)超过订单|transactions?.{0,12}(?:<=|not exceed).{0,12}order",
                        span,
                        re.I,
                    )
                )
                and bool(
                    re.search(
                        r"同币|订单、发票、交易币种必须相同|same.currency", span, re.I
                    )
                )
            )
        if field == "dates":
            return value == "booked_ge_ordered" and bool(
                re.search(
                    r"记账(?:日期)?(?:不|不得)早于下单(?:日期)?|booked.{0,12}>=.{0,12}ordered",
                    span,
                    re.I,
                )
            )
        if field == "special":
            return (
                isinstance(value, str)
                and value in span
                and bool(
                    re.search(
                        r"\d|清单[:：]|list[:=]|明确不适用|explicitly exempt",
                        value,
                        re.I,
                    )
                )
            )
        return False
    if (
        worker == "contract"
        and obligation == "O-05"
        and field in {"buyer", "seller", "subject", "order", "contract"}
    ):
        labels = "|".join(_LABELS[field])
        return isinstance(value, str) and bool(
            re.search(
                r"(?:"
                + labels
                + r")\s*[:：=]?\s*"
                + re.escape(value)
                + r"(?=$|[\s,，。.;；|])",
                span,
                re.I,
            )
        )
    if worker == "contract" and obligation == "O-06":
        prices = re.findall(
            r"(?:价款(?:记录[:：]\s*订单)?|PRICE)\s*[:：=]?\s*((?:0|[1-9][0-9]*)\.[0-9]{2})\s+(CNY|USD)\b",
            span,
            re.I,
        )
        index = {"amount": 0, "currency": 1}.get(field)
        return (
            index is not None
            and isinstance(value, str)
            and {p[index] for p in prices} == {value}
        )
    if worker == "contract" and obligation == "O-08":
        if field in {"delivery", "acceptance", "warranty"}:
            return type(value) is int and _labelled_number(field, span, False) == {
                value
            }
        markers = {
            "standard": r"标准|STANDARD",
            "responsibility": r"质保责任|WARRANTY_RESPONSIBILITY",
        }
        return (
            field in markers
            and isinstance(value, str)
            and value in span
            and len(value) >= 4
            and bool(
                re.search(
                    r"(?:"
                    + markers[field]
                    + r")\s*[:：=]?\s*"
                    + re.escape(value)
                    + r"(?=$|[\s,，。;；|])",
                    span,
                    re.I,
                )
            )
            and not re.search(r"已列明|按条款执行|条款存在|as\s+specified", value, re.I)
        )
    if worker == "contract" and obligation == "O-09":
        if field == "written":
            positive = bool(
                re.search(
                    r"(?:变更须双方书面签署|双方书面签署后变更生效|CHANGE=effective after both parties sign in writing)",
                    span,
                    re.I,
                )
            )
            negative = bool(
                re.search(r"变更无需书面|change.{0,8}without.{0,8}writing", span, re.I)
            )
            return type(value) is bool and (
                positive and not negative if value else negative and not positive
            )
        if field in {"damages", "cap", "other"}:
            markers = {
                "damages": r"赔偿|赔付|逾期|DAMAGES",
                "cap": r"上限|LIMIT|CAP",
                "other": r"分包|知识产权|争议|本次只审|SUBCONTRACT|IP_RIGHTS|DISPUTE|ONLY_REVIEW",
            }
            concrete = field == "other" or bool(re.search(r"\d", str(value)))
            return (
                concrete
                and isinstance(value, str)
                and value in span
                and (
                    bool(
                        re.search(
                            r"(?:"
                            + markers[field]
                            + r")\s*(?:为|[:：=])?\s*"
                            + re.escape(value)
                            + r"(?=$|[\s,，。.;；|])",
                            span,
                            re.I,
                        )
                    )
                    or field != "cap"
                    and bool(re.search(markers[field], value, re.I))
                )
            )
    return False
