"""Strict contracts for the V1-07 synthetic corpus and independent truth.

The model-visible cases and the answer-bearing truth file deliberately use separate
schemas.  This module has no dependency on the production Aggregator or rule tables;
the independent oracle lives in :mod:`gold_v3_oracle`.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

GOLD_SET_V3 = "gold-set.v3"
GOLD_CORPUS_V3 = "synthetic-gold-corpus.v3"
GOLD_GENERATOR_V3 = "procurement-gold-generator.v3"
GOLD_CASE_SCHEMA_V2 = "gold-case.v2"
GOLD_LABEL_SCHEMA_V2 = "gold-label.v2"
GOLD_TRUTH_SCHEMA_V1 = "v1-07-truth.v1"
GOLD_MANIFEST_SCHEMA_V2 = "gold-manifest.v2"
GOLD_CASES_FILE_SCHEMA_V2 = "gold-cases.v2"
GOLD_CASE_COUNT_V3 = 180
GOLD_GENERATOR_SEED_V3 = 20260919
GOLD_AS_OF_V3 = date(2026, 6, 30)

HexSha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CaseId = Annotated[str, Field(pattern=r"^V107-[DV]-[0-9]{4}$")]
OpaqueId = Annotated[str, Field(pattern=r"^V107-[A-Z0-9-]+$")]
ObligationId = Annotated[str, Field(pattern=r"^O-(0[1-9]|1[0-2])$")]
NonEmptyText = Annotated[str, Field(min_length=1, max_length=4_000)]
BoundedValue = Annotated[str, Field(min_length=1, max_length=256)]


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical byte representation used by all v3 digests."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class V3Split(StrEnum):
    DEV = "dev"
    VALIDATION = "validation"


class V3Recommendation(StrEnum):
    APPROVE = "recommend_approve"
    CONDITIONAL = "recommend_conditional"
    REJECT = "recommend_reject"
    INSUFFICIENT = "insufficient_information"
    HUMAN = "needs_human_review"


class ObligationState(StrEnum):
    SUPPORTED = "supported"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class TruthSeverity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(StrEnum):
    PROMPT_INJECTION = "prompt_injection_privilege"
    SECRET_EXFILTRATION = "synthetic_secret_exfiltration"


class EvidenceDocument(FrozenModel):
    source_id: OpaqueId
    version: Literal["synthetic-gold-document.v3"] = "synthetic-gold-document.v3"
    section: Annotated[str, Field(min_length=1, max_length=96)]
    content: NonEmptyText
    content_sha256: HexSha256

    @model_validator(mode="after")
    def checksum_matches(self) -> Self:
        expected = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_sha256 != expected:
            raise ValueError("document content checksum mismatch")
        return self


class V3BusinessFacts(FrozenModel):
    scope_id: OpaqueId
    supplier_id: OpaqueId
    purchase_order_id: OpaqueId
    transaction_id: OpaqueId
    source_version: Literal["synthetic-gold-procurement.v3"] = (
        "synthetic-gold-procurement.v3"
    )
    source_status: Literal["known", "failed", "version_mismatch"] = "known"
    registry_status: Literal["active", "inactive"] | None
    procurement_status: Literal["approved", "blocked"] | None
    tax_identifier: Annotated[str, Field(min_length=1, max_length=64)] | None
    due_diligence_status: Literal["verified", "failed"] | None
    sanctions_screening: Literal["clear", "potential_match"] | None
    competitive_quote_count: Annotated[int, Field(ge=0, le=20)] | None
    contract_reference: OpaqueId | None
    counterparty_reference: OpaqueId | None
    order_amount_minor: Annotated[int, Field(gt=0)] | None
    invoice_amount_minor: Annotated[int, Field(gt=0)] | None
    transaction_amount_minor: Annotated[int, Field(gt=0)] | None
    order_currency: Literal["CNY", "USD"] | None
    invoice_currency: Literal["CNY", "USD"] | None
    transaction_currency: Literal["CNY", "USD"] | None
    ordered_on: date | None
    booked_on: date | None
    delivery_days: Annotated[int, Field(gt=0, le=365)] | None
    acceptance_days: Annotated[int, Field(gt=0, le=365)] | None
    warranty_days: Annotated[int, Field(gt=0, le=3_650)] | None
    written_change_required: bool | None
    liability_cap_present: bool | None


class V3ModelInput(FrozenModel):
    question: NonEmptyText
    policy_document: EvidenceDocument
    contract_document: EvidenceDocument
    business_facts: V3BusinessFacts


class GoldCaseV3(FrozenModel):
    schema_version: Literal["gold-case.v2"] = GOLD_CASE_SCHEMA_V2
    gold_set_version: Literal["gold-set.v3"] = GOLD_SET_V3
    corpus_snapshot_id: Literal["synthetic-gold-corpus.v3"] = GOLD_CORPUS_V3
    case_id: CaseId
    case_version: Literal["2.0"] = "2.0"
    split: V3Split
    generation_family_id: OpaqueId
    near_duplicate_group_id: OpaqueId
    entity_root_id: OpaqueId
    metamorphic_parent_id: CaseId | None = None
    language: Literal["zh-CN", "en-US"]
    jurisdiction: Literal["CN-MAINLAND-DEMO-1"] = "CN-MAINLAND-DEMO-1"
    as_of: date = GOLD_AS_OF_V3
    pii_classification: Literal["synthetic_only"] = "synthetic_only"
    license_state: Literal["synthetic"] = "synthetic"
    model_input: V3ModelInput
    source_checksum: HexSha256

    @model_validator(mode="after")
    def checksum_and_identity_match(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"source_checksum"})
        if self.source_checksum != payload_sha256(payload):
            raise ValueError("case source checksum mismatch")
        if self.metamorphic_parent_id == self.case_id:
            raise ValueError("a case cannot be its own metamorphic parent")
        return self


class EvidenceIdentity(FrozenModel):
    source_id: OpaqueId
    version: Annotated[str, Field(min_length=1, max_length=96)]
    locator: Annotated[str, Field(min_length=1, max_length=160)]
    content_sha256: HexSha256


class PropositionIdentity(FrozenModel):
    scope_id: OpaqueId
    subject_type: Annotated[str, Field(pattern=r"^[a-z_]+$", max_length=48)]
    subject_id: OpaqueId
    check_id: ObligationId
    field_path: Annotated[str, Field(pattern=r"^[a-z0-9_.]+$", max_length=96)]
    effective_at: date
    source_version: Annotated[str, Field(min_length=1, max_length=96)]


class ObligationTruth(FrozenModel):
    obligation_id: ObligationId
    state: ObligationState
    proposition: PropositionIdentity
    normalized_value: BoundedValue
    evidence: tuple[EvidenceIdentity, ...] = Field(max_length=4)
    severity: TruthSeverity = TruthSeverity.NONE
    reason_code: Annotated[str, Field(pattern=r"^[a-z0-9_]+$", max_length=64)]

    @model_validator(mode="after")
    def evidence_and_severity_are_consistent(self) -> Self:
        if self.state in {ObligationState.SUPPORTED, ObligationState.VIOLATED}:
            if not self.evidence:
                raise ValueError("decidable obligation truth requires evidence")
        if self.state is ObligationState.VIOLATED:
            if self.severity is TruthSeverity.NONE:
                raise ValueError("violated obligation truth requires severity")
        elif self.severity is not TruthSeverity.NONE:
            raise ValueError("only violated obligations carry severity")
        return self


class ConflictTruth(FrozenModel):
    pair_id: HexSha256
    relation: Literal["must_equal"] = "must_equal"
    left: PropositionIdentity
    right: PropositionIdentity
    left_value: BoundedValue
    right_value: BoundedValue

    @model_validator(mode="after")
    def pair_is_canonical(self) -> Self:
        if self.left == self.right or self.left_value == self.right_value:
            raise ValueError("a conflict requires distinct propositions and values")
        canonical = sorted(
            (
                self.left.model_dump(mode="json"),
                self.right.model_dump(mode="json"),
            ),
            key=lambda item: canonical_json_bytes(item),
        )
        expected = payload_sha256(
            {"left": canonical[0], "relation": self.relation, "right": canonical[1]}
        )
        if self.pair_id != expected:
            raise ValueError("conflict pair id mismatch")
        return self


class SecurityTruth(FrozenModel):
    event_id: HexSha256
    event_type: SecurityEventType
    source_id: OpaqueId
    location: Annotated[str, Field(min_length=1, max_length=160)]


class TruthRecordV1(FrozenModel):
    schema_version: Literal["gold-label.v2"] = GOLD_LABEL_SCHEMA_V2
    truth_contract: Literal["v1-07-truth.v1"] = GOLD_TRUTH_SCHEMA_V1
    case_id: CaseId
    case_source_checksum: HexSha256
    expected_recommendation: V3Recommendation
    obligations: tuple[ObligationTruth, ...] = Field(min_length=12, max_length=12)
    conflicts: tuple[ConflictTruth, ...] = Field(max_length=4)
    security_events: tuple[SecurityTruth, ...] = Field(max_length=4)
    required_evidence: tuple[EvidenceIdentity, ...] = Field(min_length=1, max_length=16)
    high_risk: bool
    adjudication_source: Literal["v1-07-business-spec.v1"] = "v1-07-business-spec.v1"
    truth_checksum: HexSha256

    @model_validator(mode="after")
    def truth_is_closed_and_consistent(self) -> Self:
        ids = tuple(item.obligation_id for item in self.obligations)
        if ids != tuple(f"O-{index:02d}" for index in range(1, 13)):
            raise ValueError("truth must contain O-01 through O-12 in order")
        if len(self.required_evidence) != len(set(self.required_evidence)):
            raise ValueError("required evidence identities must be unique")
        if self.expected_recommendation is V3Recommendation.HUMAN:
            if not self.conflicts and not self.security_events:
                raise ValueError(
                    "human review truth requires a conflict or security event"
                )
        elif self.conflicts or self.security_events:
            raise ValueError("conflict/security events must result in human review")
        if self.expected_recommendation is V3Recommendation.INSUFFICIENT:
            if not any(
                item.state is ObligationState.UNKNOWN for item in self.obligations
            ):
                raise ValueError("insufficient truth requires an unknown obligation")
        if self.expected_recommendation is V3Recommendation.APPROVE:
            if any(
                item.state
                not in {
                    ObligationState.SUPPORTED,
                    ObligationState.NOT_APPLICABLE,
                }
                for item in self.obligations
            ):
                raise ValueError("approve truth requires every applicable obligation")
        payload = self.model_dump(mode="json", exclude={"truth_checksum"})
        if self.truth_checksum != payload_sha256(payload):
            raise ValueError("truth checksum mismatch")
        return self


class GoldCasesV3File(FrozenModel):
    schema_version: Literal["gold-cases.v2"] = GOLD_CASES_FILE_SCHEMA_V2
    gold_set_version: Literal["gold-set.v3"] = GOLD_SET_V3
    generator_version: Literal["procurement-gold-generator.v3"] = GOLD_GENERATOR_V3
    seed: Literal[20260919] = GOLD_GENERATOR_SEED_V3
    cases: tuple[GoldCaseV3, ...] = Field(
        min_length=GOLD_CASE_COUNT_V3, max_length=GOLD_CASE_COUNT_V3
    )
    payload_sha256: HexSha256

    @model_validator(mode="after")
    def cases_are_unique_and_hashed(self) -> Self:
        ids = tuple(item.case_id for item in self.cases)
        if len(ids) != len(set(ids)):
            raise ValueError("v3 case ids must be unique")
        expected = payload_sha256(
            {"cases": [item.model_dump(mode="json") for item in self.cases]}
        )
        if self.payload_sha256 != expected:
            raise ValueError("v3 cases payload checksum mismatch")
        return self


class GoldTruthV1File(FrozenModel):
    schema_version: Literal["v1-07-truth.v1"] = GOLD_TRUTH_SCHEMA_V1
    gold_set_version: Literal["gold-set.v3"] = GOLD_SET_V3
    generator_version: Literal["procurement-gold-generator.v3"] = GOLD_GENERATOR_V3
    seed: Literal[20260919] = GOLD_GENERATOR_SEED_V3
    truths: tuple[TruthRecordV1, ...] = Field(
        min_length=GOLD_CASE_COUNT_V3, max_length=GOLD_CASE_COUNT_V3
    )
    payload_sha256: HexSha256

    @model_validator(mode="after")
    def truths_are_unique_and_hashed(self) -> Self:
        ids = tuple(item.case_id for item in self.truths)
        if len(ids) != len(set(ids)):
            raise ValueError("v3 truth case ids must be unique")
        expected = payload_sha256(
            {"truths": [item.model_dump(mode="json") for item in self.truths]}
        )
        if self.payload_sha256 != expected:
            raise ValueError("v3 truth payload checksum mismatch")
        return self


class GoldManifestV3(FrozenModel):
    schema_version: Literal["gold-manifest.v2"] = GOLD_MANIFEST_SCHEMA_V2
    gold_set_version: Literal["gold-set.v3"] = GOLD_SET_V3
    corpus_snapshot_id: Literal["synthetic-gold-corpus.v3"] = GOLD_CORPUS_V3
    generator_version: Literal["procurement-gold-generator.v3"] = GOLD_GENERATOR_V3
    seed: Literal[20260919] = GOLD_GENERATOR_SEED_V3
    case_count: Literal[180] = GOLD_CASE_COUNT_V3
    truth_count: Literal[180] = GOLD_CASE_COUNT_V3
    split_counts: dict[V3Split, int]
    recommendation_counts: dict[V3Split, dict[V3Recommendation, int]]
    high_risk_counts: dict[V3Split, int]
    conflict_counts: dict[V3Split, int]
    security_counts: dict[V3Split, int]
    generation_family_count: Annotated[int, Field(gt=1)]
    near_duplicate_group_count: Annotated[int, Field(gt=1)]
    cases_payload_sha256: HexSha256
    truth_payload_sha256: HexSha256
    case_ids_sha256: HexSha256
    truth_ids_sha256: HexSha256
    manifest_sha256: HexSha256

    @model_validator(mode="after")
    def manifest_is_hashed(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if self.manifest_sha256 != payload_sha256(payload):
            raise ValueError("v3 manifest checksum mismatch")
        return self


EXPECTED_RECOMMENDATION_COUNTS: dict[V3Split, Counter[V3Recommendation]] = {
    V3Split.DEV: Counter(
        {
            V3Recommendation.APPROVE: 24,
            V3Recommendation.CONDITIONAL: 18,
            V3Recommendation.REJECT: 24,
            V3Recommendation.INSUFFICIENT: 18,
            V3Recommendation.HUMAN: 36,
        }
    ),
    V3Split.VALIDATION: Counter(
        {
            V3Recommendation.APPROVE: 12,
            V3Recommendation.CONDITIONAL: 9,
            V3Recommendation.REJECT: 12,
            V3Recommendation.INSUFFICIENT: 9,
            V3Recommendation.HUMAN: 18,
        }
    ),
}


__all__ = [
    "EXPECTED_RECOMMENDATION_COUNTS",
    "EvidenceDocument",
    "EvidenceIdentity",
    "GOLD_CASE_COUNT_V3",
    "GOLD_CASE_SCHEMA_V2",
    "GOLD_CASES_FILE_SCHEMA_V2",
    "GOLD_CORPUS_V3",
    "GOLD_GENERATOR_SEED_V3",
    "GOLD_GENERATOR_V3",
    "GOLD_LABEL_SCHEMA_V2",
    "GOLD_MANIFEST_SCHEMA_V2",
    "GOLD_SET_V3",
    "GOLD_TRUTH_SCHEMA_V1",
    "GoldCaseV3",
    "GoldCasesV3File",
    "GoldManifestV3",
    "GoldTruthV1File",
    "ObligationState",
    "ObligationTruth",
    "PropositionIdentity",
    "SecurityEventType",
    "SecurityTruth",
    "TruthRecordV1",
    "TruthSeverity",
    "V3BusinessFacts",
    "V3ModelInput",
    "V3Recommendation",
    "V3Split",
    "canonical_json_bytes",
    "payload_sha256",
]
