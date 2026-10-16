"""Versioned, deterministic synthetic procurement fixtures.

The committed fixture is an envelope around a payload. Its SHA-256 is computed
from compact, key-sorted UTF-8 JSON with exactly one trailing LF, so the digest
does not depend on the pretty-printing used for the envelope itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import HumanReviewStatus, RecommendationStatus, RiskLevel

FIXTURE_SCHEMA_VERSION = "procurement-synthetic-fixture.v1"
ENVELOPE_SCHEMA_VERSION = "procurement-synthetic-envelope.v1"
DATASET_VERSION = "synthetic-procurement.v1"
GENERATOR_VERSION = "procurement-synthetic-generator.v1"
DEFAULT_SEED = 20260830
PAYLOAD_CANONICALIZATION = "utf8-json-sort-keys-compact-with-one-trailing-lf.v1"
DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "resources" / "synthetic" / "fixture.v1.json"
)

StableId = Annotated[str, Field(pattern=r"^[A-Z]+-[0-9]{3}$")]
NonEmptyString = Annotated[str, Field(min_length=1)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _FixtureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class RegistryStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


class ProcurementStatus(StrEnum):
    APPROVED = "approved"
    BLOCKED = "blocked"


class DueDiligenceStatus(StrEnum):
    VERIFIED = "verified"
    EXPIRED = "expired"
    MISSING = "missing"


class ScreeningStatus(StrEnum):
    CLEAR = "clear"
    POTENTIAL_MATCH = "potential_match"
    UNKNOWN = "unknown"


class PurchaseOrderStatus(StrEnum):
    PENDING_REVIEW = "pending_review"


class TransactionStatus(StrEnum):
    SETTLED = "settled"


class ScenarioKind(StrEnum):
    NORMAL = "normal"
    HIGH_RISK = "high_risk"
    MISSING_INFORMATION = "missing_information"
    FACT_CONFLICT = "fact_conflict"


class GroundTruthReason(StrEnum):
    BASELINE_CONTROLS_SATISFIED = "baseline_controls_satisfied"
    SANCTIONS_POTENTIAL_MATCH = "sanctions_potential_match"
    SUPPLIER_BLOCKED = "supplier_blocked"
    DUE_DILIGENCE_EXPIRED = "due_diligence_expired"
    TRANSACTION_TOTAL_EXCEEDS_ORDER = "transaction_total_exceeds_order"
    TAX_IDENTIFIER_MISSING = "tax_identifier_missing"
    DUE_DILIGENCE_MISSING = "due_diligence_missing"
    CONTRACT_REFERENCE_MISSING = "contract_reference_missing"
    COMPETITIVE_QUOTES_MISSING = "competitive_quotes_missing"
    TRANSACTION_COUNTERPARTY_MISSING = "transaction_counterparty_missing"
    SUPPLIER_STATUS_CONFLICT = "supplier_status_conflict"
    CURRENCY_CONFLICT = "currency_conflict"
    AMOUNT_CONFLICT = "amount_conflict"


class MissingField(StrEnum):
    SUPPLIER_TAX_IDENTIFIER = "supplier.tax_identifier"
    SUPPLIER_DUE_DILIGENCE_STATUS = "supplier.due_diligence_status"
    ORDER_CONTRACT_REFERENCE = "purchase_order.contract_reference"
    ORDER_COMPETITIVE_QUOTE_COUNT = "purchase_order.competitive_quote_count"
    TRANSACTION_COUNTERPARTY_REFERENCE = "historical_transaction.counterparty_reference"


class ConflictCode(StrEnum):
    SUPPLIER_STATUS = "supplier_status_conflict"
    ORDER_INVOICE_CURRENCY = "order_invoice_currency_conflict"
    ORDER_INVOICE_AMOUNT = "order_invoice_amount_conflict"


class SyntheticSupplier(_FixtureModel):
    supplier_id: StableId
    display_name: Annotated[str, Field(pattern=r"^Synthetic Supplier [0-9]{2}$")]
    jurisdiction: Literal["CN-11", "CN-31", "CN-44"]
    registry_status: RegistryStatus
    procurement_status: ProcurementStatus
    due_diligence_status: DueDiligenceStatus
    sanctions_screening: ScreeningStatus
    tax_identifier: Annotated[str, Field(pattern=r"^SYN-TAX-[0-9]{3}$")] | None
    onboarded_on: date
    risk_signals: tuple[GroundTruthReason, ...] = ()


class SyntheticPurchaseOrder(_FixtureModel):
    purchase_order_id: StableId
    supplier_id: StableId
    ordered_on: date
    amount_minor: Annotated[int, Field(strict=True, gt=0)]
    currency: Literal["CNY"]
    invoice_amount_minor: Annotated[int, Field(strict=True, gt=0)]
    invoice_currency: Literal["CNY", "USD"]
    contract_reference: Annotated[str, Field(pattern=r"^SYN-CONTRACT-[0-9]{3}$")] | None
    competitive_quote_count: Annotated[int, Field(strict=True, ge=0)] | None
    requesting_unit: Literal[
        "Synthetic Facilities",
        "Synthetic IT",
        "Synthetic Operations",
    ]
    status: Literal[PurchaseOrderStatus.PENDING_REVIEW]


class SyntheticHistoricalTransaction(_FixtureModel):
    transaction_id: StableId
    purchase_order_id: StableId
    supplier_id: StableId
    booked_on: date
    amount_minor: Annotated[int, Field(strict=True, gt=0)]
    currency: Literal["CNY", "USD"]
    status: Literal[TransactionStatus.SETTLED]
    counterparty_reference: (
        Annotated[str, Field(pattern=r"^SYN-COUNTERPARTY-[0-9]{3}$")] | None
    )


class GroundTruthConflict(_FixtureModel):
    conflict_code: ConflictCode
    left_reference: NonEmptyString
    left_value: NonEmptyString
    right_reference: NonEmptyString
    right_value: NonEmptyString


class SyntheticGroundTruthCase(_FixtureModel):
    case_id: StableId
    scenario: ScenarioKind
    supplier_id: StableId
    purchase_order_id: StableId
    transaction_ids: Annotated[tuple[StableId, ...], Field(min_length=1)]
    expected_recommendation: RecommendationStatus
    expected_risk_level: RiskLevel
    expected_review_state: HumanReviewStatus
    must_fail_closed: bool
    expected_reasons: Annotated[tuple[GroundTruthReason, ...], Field(min_length=1)]
    expected_missing_fields: tuple[MissingField, ...] = ()
    expected_conflicts: tuple[GroundTruthConflict, ...] = ()

    @model_validator(mode="after")
    def validate_expected_outcome(self) -> Self:
        if self.scenario is ScenarioKind.NORMAL:
            if (
                self.expected_recommendation
                is not RecommendationStatus.RECOMMEND_APPROVE
                or self.expected_risk_level is not RiskLevel.LOW
                or self.expected_review_state
                is not HumanReviewStatus.PENDING_CONFIRMATION
                or self.must_fail_closed
                or self.expected_missing_fields
                or self.expected_conflicts
            ):
                raise ValueError("normal cases must be complete, low-risk approvals")
        elif self.scenario is ScenarioKind.HIGH_RISK:
            if (
                self.expected_recommendation
                is not RecommendationStatus.RECOMMEND_REJECT
                or self.expected_risk_level is not RiskLevel.CRITICAL
                or self.expected_review_state
                is not HumanReviewStatus.PENDING_CONFIRMATION
                or not self.must_fail_closed
            ):
                raise ValueError("high-risk cases must fail closed as critical rejects")
        elif self.scenario is ScenarioKind.MISSING_INFORMATION:
            if (
                self.expected_recommendation
                is not RecommendationStatus.INSUFFICIENT_INFORMATION
                or self.expected_risk_level is not RiskLevel.UNKNOWN
                or self.expected_review_state is not HumanReviewStatus.REQUIRED
                or not self.must_fail_closed
                or not self.expected_missing_fields
            ):
                raise ValueError(
                    "missing-information cases must fail closed for human review"
                )
        elif (
            self.expected_recommendation is not RecommendationStatus.NEEDS_HUMAN_REVIEW
            or self.expected_risk_level is not RiskLevel.HIGH
            or self.expected_review_state is not HumanReviewStatus.REQUIRED
            or not self.must_fail_closed
            or not self.expected_conflicts
        ):
            raise ValueError("fact-conflict cases must fail closed for human review")
        return self


_SCENARIO_REASONS: dict[ScenarioKind, tuple[GroundTruthReason, ...]] = {
    ScenarioKind.NORMAL: (GroundTruthReason.BASELINE_CONTROLS_SATISFIED,),
    ScenarioKind.HIGH_RISK: (
        GroundTruthReason.SANCTIONS_POTENTIAL_MATCH,
        GroundTruthReason.SUPPLIER_BLOCKED,
        GroundTruthReason.DUE_DILIGENCE_EXPIRED,
        GroundTruthReason.TRANSACTION_TOTAL_EXCEEDS_ORDER,
    ),
    ScenarioKind.MISSING_INFORMATION: (
        GroundTruthReason.TAX_IDENTIFIER_MISSING,
        GroundTruthReason.DUE_DILIGENCE_MISSING,
        GroundTruthReason.CONTRACT_REFERENCE_MISSING,
        GroundTruthReason.COMPETITIVE_QUOTES_MISSING,
        GroundTruthReason.TRANSACTION_COUNTERPARTY_MISSING,
    ),
    ScenarioKind.FACT_CONFLICT: (
        GroundTruthReason.SUPPLIER_STATUS_CONFLICT,
        GroundTruthReason.CURRENCY_CONFLICT,
        GroundTruthReason.AMOUNT_CONFLICT,
    ),
}

_SCENARIO_MISSING_FIELDS: dict[ScenarioKind, tuple[MissingField, ...]] = {
    ScenarioKind.NORMAL: (),
    ScenarioKind.HIGH_RISK: (),
    ScenarioKind.MISSING_INFORMATION: (
        MissingField.SUPPLIER_TAX_IDENTIFIER,
        MissingField.SUPPLIER_DUE_DILIGENCE_STATUS,
        MissingField.ORDER_CONTRACT_REFERENCE,
        MissingField.ORDER_COMPETITIVE_QUOTE_COUNT,
        MissingField.TRANSACTION_COUNTERPARTY_REFERENCE,
    ),
    ScenarioKind.FACT_CONFLICT: (),
}

_SCENARIO_CONFLICT_CODES: dict[ScenarioKind, tuple[ConflictCode, ...]] = {
    ScenarioKind.NORMAL: (),
    ScenarioKind.HIGH_RISK: (),
    ScenarioKind.MISSING_INFORMATION: (),
    ScenarioKind.FACT_CONFLICT: (
        ConflictCode.SUPPLIER_STATUS,
        ConflictCode.ORDER_INVOICE_CURRENCY,
        ConflictCode.ORDER_INVOICE_AMOUNT,
    ),
}


def _has_supplier_status_conflict(supplier: SyntheticSupplier) -> bool:
    compatible_status_pairs = {
        (RegistryStatus.ACTIVE, ProcurementStatus.APPROVED),
        (RegistryStatus.SUSPENDED, ProcurementStatus.BLOCKED),
    }
    return (
        supplier.registry_status,
        supplier.procurement_status,
    ) not in compatible_status_pairs


def _derive_missing_fields(
    supplier: SyntheticSupplier,
    order: SyntheticPurchaseOrder,
    transactions: tuple[SyntheticHistoricalTransaction, ...],
) -> tuple[MissingField, ...]:
    missing_fields: list[MissingField] = []
    if supplier.tax_identifier is None:
        missing_fields.append(MissingField.SUPPLIER_TAX_IDENTIFIER)
    if supplier.due_diligence_status is DueDiligenceStatus.MISSING:
        missing_fields.append(MissingField.SUPPLIER_DUE_DILIGENCE_STATUS)
    if order.contract_reference is None:
        missing_fields.append(MissingField.ORDER_CONTRACT_REFERENCE)
    if order.competitive_quote_count is None:
        missing_fields.append(MissingField.ORDER_COMPETITIVE_QUOTE_COUNT)
    if any(item.counterparty_reference is None for item in transactions):
        missing_fields.append(MissingField.TRANSACTION_COUNTERPARTY_REFERENCE)
    return tuple(missing_fields)


def _derive_conflicts(
    supplier: SyntheticSupplier,
    order: SyntheticPurchaseOrder,
) -> tuple[GroundTruthConflict, ...]:
    conflicts: list[GroundTruthConflict] = []
    if _has_supplier_status_conflict(supplier):
        conflicts.append(
            GroundTruthConflict(
                conflict_code=ConflictCode.SUPPLIER_STATUS,
                left_reference=f"supplier:{supplier.supplier_id}:registry_status",
                left_value=supplier.registry_status.value,
                right_reference=f"supplier:{supplier.supplier_id}:procurement_status",
                right_value=supplier.procurement_status.value,
            )
        )
    if order.currency != order.invoice_currency:
        conflicts.append(
            GroundTruthConflict(
                conflict_code=ConflictCode.ORDER_INVOICE_CURRENCY,
                left_reference=f"purchase_order:{order.purchase_order_id}:currency",
                left_value=order.currency,
                right_reference=(
                    f"purchase_order:{order.purchase_order_id}:invoice_currency"
                ),
                right_value=order.invoice_currency,
            )
        )
    if order.amount_minor != order.invoice_amount_minor:
        conflicts.append(
            GroundTruthConflict(
                conflict_code=ConflictCode.ORDER_INVOICE_AMOUNT,
                left_reference=f"purchase_order:{order.purchase_order_id}:amount_minor",
                left_value=str(order.amount_minor),
                right_reference=(
                    f"purchase_order:{order.purchase_order_id}:invoice_amount_minor"
                ),
                right_value=str(order.invoice_amount_minor),
            )
        )
    return tuple(conflicts)


def _derive_supplier_risk_signals(
    supplier: SyntheticSupplier,
) -> tuple[GroundTruthReason, ...]:
    reasons: list[GroundTruthReason] = []
    status_conflict = _has_supplier_status_conflict(supplier)
    if supplier.sanctions_screening is ScreeningStatus.POTENTIAL_MATCH:
        reasons.append(GroundTruthReason.SANCTIONS_POTENTIAL_MATCH)
    if supplier.procurement_status is ProcurementStatus.BLOCKED and not status_conflict:
        reasons.append(GroundTruthReason.SUPPLIER_BLOCKED)
    if supplier.due_diligence_status is DueDiligenceStatus.EXPIRED:
        reasons.append(GroundTruthReason.DUE_DILIGENCE_EXPIRED)
    if supplier.tax_identifier is None:
        reasons.append(GroundTruthReason.TAX_IDENTIFIER_MISSING)
    if supplier.due_diligence_status is DueDiligenceStatus.MISSING:
        reasons.append(GroundTruthReason.DUE_DILIGENCE_MISSING)
    if status_conflict:
        reasons.append(GroundTruthReason.SUPPLIER_STATUS_CONFLICT)
    return tuple(reasons)


def _derive_reasons(
    supplier: SyntheticSupplier,
    order: SyntheticPurchaseOrder,
    transactions: tuple[SyntheticHistoricalTransaction, ...],
    missing_fields: tuple[MissingField, ...],
    conflicts: tuple[GroundTruthConflict, ...],
) -> tuple[GroundTruthReason, ...]:
    reasons = list(_derive_supplier_risk_signals(supplier))
    if sum(item.amount_minor for item in transactions) > order.amount_minor:
        reasons.append(GroundTruthReason.TRANSACTION_TOTAL_EXCEEDS_ORDER)

    missing_reason_by_field = {
        MissingField.ORDER_CONTRACT_REFERENCE: (
            GroundTruthReason.CONTRACT_REFERENCE_MISSING
        ),
        MissingField.ORDER_COMPETITIVE_QUOTE_COUNT: (
            GroundTruthReason.COMPETITIVE_QUOTES_MISSING
        ),
        MissingField.TRANSACTION_COUNTERPARTY_REFERENCE: (
            GroundTruthReason.TRANSACTION_COUNTERPARTY_MISSING
        ),
    }
    for missing_field in missing_fields:
        missing_reason = missing_reason_by_field.get(missing_field)
        if missing_reason is not None:
            reasons.append(missing_reason)

    conflict_reason_by_code = {
        ConflictCode.SUPPLIER_STATUS: GroundTruthReason.SUPPLIER_STATUS_CONFLICT,
        ConflictCode.ORDER_INVOICE_CURRENCY: GroundTruthReason.CURRENCY_CONFLICT,
        ConflictCode.ORDER_INVOICE_AMOUNT: GroundTruthReason.AMOUNT_CONFLICT,
    }
    for conflict in conflicts:
        conflict_reason = conflict_reason_by_code[conflict.conflict_code]
        if conflict_reason not in reasons:
            reasons.append(conflict_reason)

    baseline_controls_satisfied = (
        supplier.registry_status is RegistryStatus.ACTIVE
        and supplier.procurement_status is ProcurementStatus.APPROVED
        and supplier.due_diligence_status is DueDiligenceStatus.VERIFIED
        and supplier.sanctions_screening is ScreeningStatus.CLEAR
        and not missing_fields
        and not conflicts
        and sum(item.amount_minor for item in transactions) <= order.amount_minor
    )
    if not reasons and baseline_controls_satisfied:
        reasons.append(GroundTruthReason.BASELINE_CONTROLS_SATISFIED)
    return tuple(reasons)


class SyntheticFixturePayload(_FixtureModel):
    fixture_schema_version: Literal["procurement-synthetic-fixture.v1"]
    dataset_version: Literal["synthetic-procurement.v1"]
    generator_version: Literal["procurement-synthetic-generator.v1"]
    seed: Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
    as_of: date
    suppliers: Annotated[tuple[SyntheticSupplier, ...], Field(min_length=12)]
    purchase_orders: Annotated[tuple[SyntheticPurchaseOrder, ...], Field(min_length=24)]
    historical_transactions: Annotated[
        tuple[SyntheticHistoricalTransaction, ...], Field(min_length=72)
    ]
    ground_truth_cases: Annotated[
        tuple[SyntheticGroundTruthCase, ...], Field(min_length=8)
    ]

    @model_validator(mode="after")
    def validate_references_and_ground_truth(self) -> Self:
        supplier_by_id = {item.supplier_id: item for item in self.suppliers}
        order_by_id = {item.purchase_order_id: item for item in self.purchase_orders}
        transaction_by_id = {
            item.transaction_id: item for item in self.historical_transactions
        }
        case_by_id = {item.case_id: item for item in self.ground_truth_cases}

        if len(supplier_by_id) != len(self.suppliers):
            raise ValueError("supplier IDs must be unique")
        if len(order_by_id) != len(self.purchase_orders):
            raise ValueError("purchase order IDs must be unique")
        if len(transaction_by_id) != len(self.historical_transactions):
            raise ValueError("transaction IDs must be unique")
        if len(case_by_id) != len(self.ground_truth_cases):
            raise ValueError("case IDs must be unique")

        for order in self.purchase_orders:
            if order.supplier_id not in supplier_by_id:
                raise ValueError(
                    f"order {order.purchase_order_id} references an unknown supplier"
                )

        for transaction in self.historical_transactions:
            order = order_by_id.get(transaction.purchase_order_id)
            if order is None:
                raise ValueError(
                    f"transaction {transaction.transaction_id} references an unknown order"
                )
            if transaction.supplier_id != order.supplier_id:
                raise ValueError(
                    f"transaction {transaction.transaction_id} supplier does not match order"
                )

        present_scenarios = tuple(case.scenario for case in self.ground_truth_cases)
        required_scenarios = (
            ScenarioKind.NORMAL,
            ScenarioKind.HIGH_RISK,
            ScenarioKind.MISSING_INFORMATION,
            ScenarioKind.FACT_CONFLICT,
        )
        if not all(scenario in present_scenarios for scenario in required_scenarios):
            raise ValueError("ground truth must cover all required scenarios")

        for case in self.ground_truth_cases:
            supplier = supplier_by_id.get(case.supplier_id)
            order = order_by_id.get(case.purchase_order_id)
            if supplier is None or order is None:
                raise ValueError(f"case {case.case_id} has an unknown subject")
            if order.supplier_id != supplier.supplier_id:
                raise ValueError(f"case {case.case_id} subject references do not match")

            transactions: list[SyntheticHistoricalTransaction] = []
            for transaction_id in case.transaction_ids:
                transaction = transaction_by_id.get(transaction_id)
                if transaction is None:
                    raise ValueError(
                        f"case {case.case_id} references an unknown transaction"
                    )
                if transaction.purchase_order_id != order.purchase_order_id:
                    raise ValueError(
                        f"case {case.case_id} transaction does not belong to its order"
                    )
                transactions.append(transaction)

            transaction_tuple = tuple(transactions)
            derived_missing_fields = _derive_missing_fields(
                supplier,
                order,
                transaction_tuple,
            )
            derived_conflicts = _derive_conflicts(supplier, order)
            derived_reasons = _derive_reasons(
                supplier,
                order,
                transaction_tuple,
                derived_missing_fields,
                derived_conflicts,
            )

            expected_reasons = _SCENARIO_REASONS[case.scenario]
            expected_missing_fields = _SCENARIO_MISSING_FIELDS[case.scenario]
            expected_conflict_codes = _SCENARIO_CONFLICT_CODES[case.scenario]
            derived_conflict_codes = tuple(
                conflict.conflict_code for conflict in derived_conflicts
            )
            if (
                derived_reasons != expected_reasons
                or derived_missing_fields != expected_missing_fields
                or derived_conflict_codes != expected_conflict_codes
            ):
                raise ValueError(
                    f"case {case.case_id} source facts do not match its "
                    f"{case.scenario.value} scenario"
                )
            if supplier.risk_signals != _derive_supplier_risk_signals(supplier):
                raise ValueError(
                    f"case {case.case_id} supplier risk signals do not match source facts"
                )
            if case.expected_reasons != derived_reasons:
                raise ValueError(
                    f"case {case.case_id} ground truth reasons do not match source facts"
                )
            if case.expected_missing_fields != derived_missing_fields:
                raise ValueError(
                    f"case {case.case_id} ground truth missing fields do not match "
                    "source facts"
                )
            if case.expected_conflicts != derived_conflicts:
                raise ValueError(
                    f"case {case.case_id} ground truth conflicts do not match source facts"
                )

        return self


class SyntheticFixtureEnvelope(_FixtureModel):
    envelope_schema_version: Literal["procurement-synthetic-envelope.v1"]
    digest_algorithm: Literal["sha256"]
    payload_canonicalization: Literal[
        "utf8-json-sort-keys-compact-with-one-trailing-lf.v1"
    ]
    payload_sha256: Sha256Hex
    payload: SyntheticFixturePayload

    @model_validator(mode="after")
    def validate_payload_digest(self) -> Self:
        actual = payload_sha256(self.payload)
        if self.payload_sha256 != actual:
            raise ValueError(
                f"payload SHA-256 mismatch: declared {self.payload_sha256}, actual {actual}"
            )
        return self


class FixtureCheckError(ValueError):
    """Raised when a committed fixture is not the exact deterministic output."""


def _validate_generation_inputs(seed: int, dataset_version: str) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be between 0 and 2**63 - 1")
    if dataset_version != DATASET_VERSION:
        raise ValueError(f"unsupported dataset version: {dataset_version}")


def _scenario_for_supplier(index: int) -> ScenarioKind:
    scenarios = (
        ScenarioKind.NORMAL,
        ScenarioKind.HIGH_RISK,
        ScenarioKind.MISSING_INFORMATION,
        ScenarioKind.FACT_CONFLICT,
    )
    return scenarios[(index - 1) % len(scenarios)]


def _supplier_for_scenario(
    index: int,
    scenario: ScenarioKind,
    rng: random.Random,
) -> SyntheticSupplier:
    common: dict[str, Any] = {
        "supplier_id": f"SUP-{index:03d}",
        "display_name": f"Synthetic Supplier {index:02d}",
        "jurisdiction": rng.choice(("CN-11", "CN-31", "CN-44")),
        "onboarded_on": date(2024, 1, 1) + timedelta(days=rng.randrange(0, 500)),
    }

    if scenario is ScenarioKind.NORMAL:
        return SyntheticSupplier(
            **common,
            registry_status=RegistryStatus.ACTIVE,
            procurement_status=ProcurementStatus.APPROVED,
            due_diligence_status=DueDiligenceStatus.VERIFIED,
            sanctions_screening=ScreeningStatus.CLEAR,
            tax_identifier=f"SYN-TAX-{index:03d}",
        )
    if scenario is ScenarioKind.HIGH_RISK:
        return SyntheticSupplier(
            **common,
            registry_status=RegistryStatus.SUSPENDED,
            procurement_status=ProcurementStatus.BLOCKED,
            due_diligence_status=DueDiligenceStatus.EXPIRED,
            sanctions_screening=ScreeningStatus.POTENTIAL_MATCH,
            tax_identifier=f"SYN-TAX-{index:03d}",
            risk_signals=(
                GroundTruthReason.SANCTIONS_POTENTIAL_MATCH,
                GroundTruthReason.SUPPLIER_BLOCKED,
                GroundTruthReason.DUE_DILIGENCE_EXPIRED,
            ),
        )
    if scenario is ScenarioKind.MISSING_INFORMATION:
        return SyntheticSupplier(
            **common,
            registry_status=RegistryStatus.ACTIVE,
            procurement_status=ProcurementStatus.APPROVED,
            due_diligence_status=DueDiligenceStatus.MISSING,
            sanctions_screening=ScreeningStatus.UNKNOWN,
            tax_identifier=None,
            risk_signals=(
                GroundTruthReason.TAX_IDENTIFIER_MISSING,
                GroundTruthReason.DUE_DILIGENCE_MISSING,
            ),
        )
    return SyntheticSupplier(
        **common,
        registry_status=RegistryStatus.ACTIVE,
        procurement_status=ProcurementStatus.BLOCKED,
        due_diligence_status=DueDiligenceStatus.VERIFIED,
        sanctions_screening=ScreeningStatus.CLEAR,
        tax_identifier=f"SYN-TAX-{index:03d}",
        risk_signals=(GroundTruthReason.SUPPLIER_STATUS_CONFLICT,),
    )


def _order_for_scenario(
    order_index: int,
    supplier_id: str,
    scenario: ScenarioKind,
    rng: random.Random,
) -> SyntheticPurchaseOrder:
    if scenario is ScenarioKind.HIGH_RISK:
        amount_minor = rng.randrange(5_000_000, 15_000_001, 100)
    else:
        amount_minor = rng.randrange(200_000, 1_200_001, 100)

    contract_reference: str | None = f"SYN-CONTRACT-{order_index:03d}"
    competitive_quote_count: int | None = rng.randrange(3, 6)
    invoice_amount_minor = amount_minor
    invoice_currency = "CNY"
    if scenario is ScenarioKind.MISSING_INFORMATION:
        contract_reference = None
        competitive_quote_count = None
    elif scenario is ScenarioKind.FACT_CONFLICT:
        invoice_amount_minor = amount_minor + rng.randrange(10_000, 50_001, 100)
        invoice_currency = "USD"

    return SyntheticPurchaseOrder(
        purchase_order_id=f"PO-{order_index:03d}",
        supplier_id=supplier_id,
        ordered_on=date(2026, 1, 1) + timedelta(days=rng.randrange(0, 120)),
        amount_minor=amount_minor,
        currency="CNY",
        invoice_amount_minor=invoice_amount_minor,
        invoice_currency=invoice_currency,
        contract_reference=contract_reference,
        competitive_quote_count=competitive_quote_count,
        requesting_unit=rng.choice(
            (
                "Synthetic Facilities",
                "Synthetic IT",
                "Synthetic Operations",
            )
        ),
        status=PurchaseOrderStatus.PENDING_REVIEW,
    )


def _transactions_for_order(
    transaction_start: int,
    order: SyntheticPurchaseOrder,
    scenario: ScenarioKind,
) -> tuple[SyntheticHistoricalTransaction, ...]:
    if scenario is ScenarioKind.HIGH_RISK:
        amounts = (
            order.amount_minor // 2,
            order.amount_minor // 2,
            order.amount_minor // 2,
        )
    else:
        first = order.amount_minor // 3
        second = order.amount_minor // 3
        amounts = (first, second, order.amount_minor - first - second)

    transactions: list[SyntheticHistoricalTransaction] = []
    for offset, amount_minor in enumerate(amounts):
        transaction_index = transaction_start + offset
        counterparty_reference: str | None = f"SYN-COUNTERPARTY-{transaction_index:03d}"
        if scenario is ScenarioKind.MISSING_INFORMATION and offset == 0:
            counterparty_reference = None
        transaction_currency = (
            "USD" if scenario is ScenarioKind.FACT_CONFLICT else "CNY"
        )
        transactions.append(
            SyntheticHistoricalTransaction(
                transaction_id=f"TXN-{transaction_index:03d}",
                purchase_order_id=order.purchase_order_id,
                supplier_id=order.supplier_id,
                booked_on=order.ordered_on + timedelta(days=7 * (offset + 1)),
                amount_minor=amount_minor,
                currency=transaction_currency,
                status=TransactionStatus.SETTLED,
                counterparty_reference=counterparty_reference,
            )
        )
    return tuple(transactions)


def _ground_truth_for_case(
    supplier_index: int,
    supplier: SyntheticSupplier,
    order: SyntheticPurchaseOrder,
    transactions: tuple[SyntheticHistoricalTransaction, ...],
    scenario: ScenarioKind,
) -> SyntheticGroundTruthCase:
    common: dict[str, Any] = {
        "case_id": f"CASE-{supplier_index:03d}",
        "scenario": scenario,
        "supplier_id": supplier.supplier_id,
        "purchase_order_id": order.purchase_order_id,
        "transaction_ids": tuple(item.transaction_id for item in transactions),
    }
    if scenario is ScenarioKind.NORMAL:
        return SyntheticGroundTruthCase(
            **common,
            expected_recommendation=RecommendationStatus.RECOMMEND_APPROVE,
            expected_risk_level=RiskLevel.LOW,
            expected_review_state=HumanReviewStatus.PENDING_CONFIRMATION,
            must_fail_closed=False,
            expected_reasons=(GroundTruthReason.BASELINE_CONTROLS_SATISFIED,),
        )
    if scenario is ScenarioKind.HIGH_RISK:
        return SyntheticGroundTruthCase(
            **common,
            expected_recommendation=RecommendationStatus.RECOMMEND_REJECT,
            expected_risk_level=RiskLevel.CRITICAL,
            expected_review_state=HumanReviewStatus.PENDING_CONFIRMATION,
            must_fail_closed=True,
            expected_reasons=(
                GroundTruthReason.SANCTIONS_POTENTIAL_MATCH,
                GroundTruthReason.SUPPLIER_BLOCKED,
                GroundTruthReason.DUE_DILIGENCE_EXPIRED,
                GroundTruthReason.TRANSACTION_TOTAL_EXCEEDS_ORDER,
            ),
        )
    if scenario is ScenarioKind.MISSING_INFORMATION:
        return SyntheticGroundTruthCase(
            **common,
            expected_recommendation=RecommendationStatus.INSUFFICIENT_INFORMATION,
            expected_risk_level=RiskLevel.UNKNOWN,
            expected_review_state=HumanReviewStatus.REQUIRED,
            must_fail_closed=True,
            expected_reasons=(
                GroundTruthReason.TAX_IDENTIFIER_MISSING,
                GroundTruthReason.DUE_DILIGENCE_MISSING,
                GroundTruthReason.CONTRACT_REFERENCE_MISSING,
                GroundTruthReason.COMPETITIVE_QUOTES_MISSING,
                GroundTruthReason.TRANSACTION_COUNTERPARTY_MISSING,
            ),
            expected_missing_fields=(
                MissingField.SUPPLIER_TAX_IDENTIFIER,
                MissingField.SUPPLIER_DUE_DILIGENCE_STATUS,
                MissingField.ORDER_CONTRACT_REFERENCE,
                MissingField.ORDER_COMPETITIVE_QUOTE_COUNT,
                MissingField.TRANSACTION_COUNTERPARTY_REFERENCE,
            ),
        )

    return SyntheticGroundTruthCase(
        **common,
        expected_recommendation=RecommendationStatus.NEEDS_HUMAN_REVIEW,
        expected_risk_level=RiskLevel.HIGH,
        expected_review_state=HumanReviewStatus.REQUIRED,
        must_fail_closed=True,
        expected_reasons=(
            GroundTruthReason.SUPPLIER_STATUS_CONFLICT,
            GroundTruthReason.CURRENCY_CONFLICT,
            GroundTruthReason.AMOUNT_CONFLICT,
        ),
        expected_conflicts=(
            GroundTruthConflict(
                conflict_code=ConflictCode.SUPPLIER_STATUS,
                left_reference=f"supplier:{supplier.supplier_id}:registry_status",
                left_value=supplier.registry_status.value,
                right_reference=f"supplier:{supplier.supplier_id}:procurement_status",
                right_value=supplier.procurement_status.value,
            ),
            GroundTruthConflict(
                conflict_code=ConflictCode.ORDER_INVOICE_CURRENCY,
                left_reference=f"purchase_order:{order.purchase_order_id}:currency",
                left_value=order.currency,
                right_reference=f"purchase_order:{order.purchase_order_id}:invoice_currency",
                right_value=order.invoice_currency,
            ),
            GroundTruthConflict(
                conflict_code=ConflictCode.ORDER_INVOICE_AMOUNT,
                left_reference=f"purchase_order:{order.purchase_order_id}:amount_minor",
                left_value=str(order.amount_minor),
                right_reference=(
                    f"purchase_order:{order.purchase_order_id}:invoice_amount_minor"
                ),
                right_value=str(order.invoice_amount_minor),
            ),
        ),
    )


def generate_payload(
    *,
    seed: int = DEFAULT_SEED,
    dataset_version: str = DATASET_VERSION,
) -> SyntheticFixturePayload:
    """Generate a validated payload from only its explicit version and seed."""
    _validate_generation_inputs(seed, dataset_version)
    rng = random.Random(seed)
    suppliers: list[SyntheticSupplier] = []
    purchase_orders: list[SyntheticPurchaseOrder] = []
    historical_transactions: list[SyntheticHistoricalTransaction] = []
    ground_truth_cases: list[SyntheticGroundTruthCase] = []

    order_index = 1
    transaction_index = 1
    for supplier_index in range(1, 13):
        scenario = _scenario_for_supplier(supplier_index)
        supplier = _supplier_for_scenario(supplier_index, scenario, rng)
        suppliers.append(supplier)
        case_order: SyntheticPurchaseOrder | None = None
        case_transactions: tuple[SyntheticHistoricalTransaction, ...] | None = None

        for order_offset in range(2):
            order = _order_for_scenario(
                order_index,
                supplier.supplier_id,
                scenario,
                rng,
            )
            transactions = _transactions_for_order(
                transaction_index,
                order,
                scenario,
            )
            purchase_orders.append(order)
            historical_transactions.extend(transactions)
            if order_offset == 0:
                case_order = order
                case_transactions = transactions
            order_index += 1
            transaction_index += len(transactions)

        if case_order is None or case_transactions is None:
            raise AssertionError("every supplier must have a ground-truth order")
        ground_truth_cases.append(
            _ground_truth_for_case(
                supplier_index,
                supplier,
                case_order,
                case_transactions,
                scenario,
            )
        )

    return SyntheticFixturePayload(
        fixture_schema_version=FIXTURE_SCHEMA_VERSION,
        dataset_version=dataset_version,
        generator_version=GENERATOR_VERSION,
        seed=seed,
        as_of=date(2026, 6, 30),
        suppliers=tuple(suppliers),
        purchase_orders=tuple(purchase_orders),
        historical_transactions=tuple(historical_transactions),
        ground_truth_cases=tuple(ground_truth_cases),
    )


def canonical_payload_bytes(payload: SyntheticFixturePayload) -> bytes:
    encoded = json.dumps(
        payload.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{encoded}\n".encode()


def payload_sha256(payload: SyntheticFixturePayload) -> str:
    return hashlib.sha256(canonical_payload_bytes(payload)).hexdigest()


def generate_fixture(
    *,
    seed: int = DEFAULT_SEED,
    dataset_version: str = DATASET_VERSION,
) -> SyntheticFixtureEnvelope:
    payload = generate_payload(seed=seed, dataset_version=dataset_version)
    return SyntheticFixtureEnvelope(
        envelope_schema_version=ENVELOPE_SCHEMA_VERSION,
        digest_algorithm="sha256",
        payload_canonicalization=PAYLOAD_CANONICALIZATION,
        payload_sha256=payload_sha256(payload),
        payload=payload,
    )


def render_fixture(fixture: SyntheticFixtureEnvelope) -> bytes:
    encoded = json.dumps(
        fixture.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        separators=(",", ": "),
    )
    return f"{encoded}\n".encode()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureCheckError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_fixture_bytes(raw: bytes) -> SyntheticFixtureEnvelope:
    try:
        decoded = raw.decode("utf-8")
        data = json.loads(decoded, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureCheckError(f"fixture is not valid UTF-8 JSON: {exc}") from exc
    return SyntheticFixtureEnvelope.model_validate(data)


def load_fixture(path: Path = DEFAULT_FIXTURE_PATH) -> SyntheticFixtureEnvelope:
    return parse_fixture_bytes(path.read_bytes())


def write_fixture(
    path: Path = DEFAULT_FIXTURE_PATH,
    *,
    seed: int = DEFAULT_SEED,
    dataset_version: str = DATASET_VERSION,
) -> SyntheticFixtureEnvelope:
    fixture = generate_fixture(seed=seed, dataset_version=dataset_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_fixture(fixture))
    return fixture


def check_fixture(
    path: Path = DEFAULT_FIXTURE_PATH,
    *,
    seed: int = DEFAULT_SEED,
    dataset_version: str = DATASET_VERSION,
) -> SyntheticFixtureEnvelope:
    actual_bytes = path.read_bytes()
    actual_fixture = parse_fixture_bytes(actual_bytes)
    expected_fixture = generate_fixture(seed=seed, dataset_version=dataset_version)
    expected_bytes = render_fixture(expected_fixture)
    if actual_bytes != expected_bytes:
        raise FixtureCheckError(
            "fixture differs from the deterministic output for the requested version and seed"
        )
    return actual_fixture


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write or verify the deterministic synthetic procurement fixture."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="write the fixture")
    action.add_argument("--check", action="store_true", help="verify exact bytes")
    parser.add_argument("--path", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    return parser


def _summary(fixture: SyntheticFixtureEnvelope, path: Path) -> str:
    payload = fixture.payload
    return (
        f"path={path} sha256={fixture.payload_sha256} "
        f"suppliers={len(payload.suppliers)} "
        f"purchase_orders={len(payload.purchase_orders)} "
        f"transactions={len(payload.historical_transactions)} "
        f"ground_truth_cases={len(payload.ground_truth_cases)}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.write:
            fixture = write_fixture(
                args.path,
                seed=args.seed,
                dataset_version=args.dataset_version,
            )
            print(f"wrote {_summary(fixture, args.path)}")
        else:
            fixture = check_fixture(
                args.path,
                seed=args.seed,
                dataset_version=args.dataset_version,
            )
            print(f"verified {_summary(fixture, args.path)}")
    except (OSError, TypeError, ValueError) as exc:
        print(f"fixture error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
