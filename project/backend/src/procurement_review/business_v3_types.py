"""Opt-in rules 3.0.0 business types; no dataset, label or production wiring."""

from datetime import date
from typing import Annotated, Literal

from pydantic import Field, WithJsonSchema, model_validator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .eval_v2_contracts import EvalV2Model
from .gold_v3_contracts import EvidenceIdentity, PropositionIdentity

RULES_VERSION = "3.0.0"
Text = Annotated[str, Field(min_length=1, max_length=160)]
Short = Annotated[str, Field(min_length=1, max_length=64)]
Count = Annotated[int, Field(ge=0, le=100000)]
State = Literal["supported", "violated", "unknown", "not_applicable"]
Risk = Literal["none", "low", "medium", "high", "critical", "unknown"]


class BusinessEvidenceIdentity(EvidenceIdentity):
    source_id: Text
    locator: Annotated[
        str,
        Field(min_length=1, max_length=160),
        WithJsonSchema(
            {
                "type": "string",
                "minLength": 1,
                "maxLength": 160,
                "format": "evidence-locator",
            }
        ),
    ]


class BusinessPropositionIdentity(PropositionIdentity):
    scope_id: Text
    subject_id: Text


class BusinessContextV3(EvalV2Model):
    scope_id: Text
    purchase_order_id: Text
    supplier_id: Text
    contract_id: Text
    as_of: date
    dataset_version: Text
    timezone: str = "Asia/Shanghai"

    @model_validator(mode="after")
    def valid_zone(self):
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("invalid case timezone") from None
        return self


class ScopeTerms(EvalV2Model):
    applies: bool
    basis: Short
    exempt: tuple[
        Literal["O-02", "O-03", "O-04", "O-05", "O-06", "O-07", "O-08", "O-09"], ...
    ] = Field(max_length=8)

    @model_validator(mode="after")
    def explicit_exemptions(self):
        if len(set(self.exempt)) != len(self.exempt) or (self.applies and self.exempt):
            raise ValueError("invalid explicit exemption list")
        if not self.applies and not self.exempt:
            raise ValueError("exemption requires explicit obligations")
        return self


class PolicyTerms(EvalV2Model):
    special: Short
    quotes: Count
    delivery: Count
    acceptance: Count
    warranty: Count
    supplier: Literal["all_qualified"]
    money: Literal["equal_invoice_capped_tx"]
    dates: Literal["booked_ge_ordered"]


class CoreTerms(EvalV2Model):
    buyer: Short
    seller: Short
    subject: Short
    order: Short
    contract: Short


class DeliveryTerms(EvalV2Model):
    delivery: Count
    acceptance: Count
    warranty: Count
    standard: Short
    responsibility: Short


class ChangeTerms(EvalV2Model):
    written: bool
    damages: Short
    cap: Short
    other: Short


class PriceTerms(EvalV2Model):
    amount: Short
    currency: Literal["CNY", "USD"]


VALUE_MODELS = {
    "policy": {"O-01": ScopeTerms, "O-02": PolicyTerms},
    "contract": {
        "O-05": CoreTerms,
        "O-06": PriceTerms,
        "O-08": DeliveryTerms,
        "O-09": ChangeTerms,
    },
}


class BusinessFindingV3(EvalV2Model):
    rules_version: Literal["3.0.0"] = RULES_VERSION
    obligation_id: Literal[
        "O-01",
        "O-02",
        "O-03",
        "O-04",
        "O-05",
        "O-06",
        "O-07",
        "O-08",
        "O-09",
        "O-10",
        "O-11",
        "O-12",
    ]
    state: State
    risk: Risk
    reason: Annotated[str, Field(pattern=r"^[a-z_]+$", max_length=96)]
    evidence: tuple[BusinessEvidenceIdentity, ...]
    missing: tuple[Short, ...] = ()
    critical: bool = True
    blocking: bool = False
    remediable: bool = False

    @model_validator(mode="after")
    def consistent(self):
        if self.state == "unknown":
            if self.risk != "unknown" or not self.missing:
                raise ValueError("unknown requires missing facts and unknown risk")
        elif self.state == "violated":
            if (
                self.risk not in {"low", "medium", "high", "critical"}
                or not self.evidence
            ):
                raise ValueError("violation requires evidence and known risk")
        elif self.risk != "none" or not self.evidence or self.missing:
            raise ValueError("supported or exempt requires evidence and no missing")
        if self.blocking and self.state != "violated":
            raise ValueError("only proven violations can block")
        return self


class BusinessObservationV3(EvalV2Model):
    proposition: BusinessPropositionIdentity
    value: str | int | bool | None
    evidence: tuple[BusinessEvidenceIdentity, ...]


class BusinessDecisionV3(EvalV2Model):
    rules_version: Literal["3.0.0"] = RULES_VERSION
    recommendation: Literal[
        "recommend_approve",
        "recommend_conditional",
        "recommend_reject",
        "insufficient_information",
        "needs_human_review",
    ]
    findings: tuple[BusinessFindingV3, ...]
    violations: tuple[str, ...]
    unknown_obligations: tuple[str, ...]
    pipeline_failures: tuple[str, ...]
    security_events: tuple[str, ...]
    conflict_pairs: tuple[str, ...]
    observations: tuple[BusinessObservationV3, ...] = ()


def finding(obligation, state, reason, evidence=(), *, missing=(), risk=None, **flags):
    return BusinessFindingV3(
        obligation_id=obligation,
        state=state,
        reason=reason,
        evidence=tuple(
            BusinessEvidenceIdentity(**e.model_dump()) for e in dict.fromkeys(evidence)
        ),
        missing=tuple(missing),
        risk=risk or ("unknown" if state == "unknown" else "none"),
        **flags,
    )
