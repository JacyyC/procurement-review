"""Validate the three read-only MCP results without a model or document context."""

import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .business_fact_contracts import BUSINESS_FACT_TOOL_KINDS
from .business_v3_types import finding, BusinessEvidenceIdentity as EvidenceIdentity
from .contracts import FactStatus
from .gold_v3_contracts import canonical_json_bytes
from .worker_contracts import ProcurementReadStatus


def money_minor(value, currency):
    if (
        not isinstance(currency, str)
        or currency not in {"CNY", "USD"}
        or not isinstance(value, str)
    ):
        return None
    if not re.fullmatch(r"(?:0|[1-9][0-9]{0,17})(?:\.[0-9]{1,2})?", value):
        return None
    try:
        return int(Decimal(value) * 100)
    except (InvalidOperation, ValueError):
        return None


def calendar_day(value, timezone):
    if not isinstance(value, str):
        return None
    try:
        zone = ZoneInfo(timezone)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return date.fromisoformat(value)
        timestamp = datetime.fromisoformat(value)
        if timestamp.utcoffset() is None:
            return None
        return timestamp.astimezone(zone).date()
    except (ValueError, ZoneInfoNotFoundError):
        return None


def present_reference(value):
    return isinstance(value, str) and bool(value.strip())


def count(value):
    return value if type(value) is int and 0 <= value <= 10**18 else None


def field_evidence(fact, field):
    return EvidenceIdentity(
        source_id=fact.record_id,
        version=fact.dataset_version,
        locator=f"{BUSINESS_FACT_TOOL_KINDS[fact.tool_name]}.{field}",
        content_sha256=hashlib.sha256(canonical_json_bytes(fact.result)).hexdigest(),
    )


@dataclass(frozen=True)
class ProcurementValuesV3:
    order: object | None
    supplier: object | None
    transactions: tuple
    source_finding: object
    failures: tuple[str, ...]
    model_request_count: int = 0

    def evidence(self, kind, *fields):
        facts = self.transactions if kind == "transactions" else (getattr(self, kind),)
        return tuple(
            field_evidence(fact, field)
            for fact in facts
            if fact is not None
            for field in fields
        )


def review_procurement_v3(context, result, *, authorized_scope_id):
    """The scope proof is supplied by the admitted MCP boundary, never the model."""
    failures = []
    if authorized_scope_id != context.scope_id:
        failures.append("procurement_scope_mismatch")
    if result.status is not ProcurementReadStatus.KNOWN:
        failures.append("procurement_source_unavailable")
    if result.requested_dataset_version != context.dataset_version:
        failures.append("procurement_version_mismatch")
    buckets = {"purchase_order": [], "supplier": [], "historical_transaction": []}
    for fact in result.facts:
        kind = BUSINESS_FACT_TOOL_KINDS.get(fact.tool_name)
        if (
            kind is None
            or fact.status is not FactStatus.KNOWN
            or not fact.read_only
            or fact.dataset_version != context.dataset_version
            or calendar_day(fact.as_of.isoformat(), context.timezone) != context.as_of
        ):
            failures.append("procurement_fact_invalid")
            continue
        payload = fact.result
        expected_key, expected_id = (
            ("supplier_id", context.supplier_id)
            if kind == "supplier"
            else ("purchase_order_id", context.purchase_order_id)
        )
        if (
            payload.get(expected_key) != expected_id
            or payload.get("supplier_id") != context.supplier_id
        ):
            failures.append("procurement_object_mismatch")
            continue
        own_key = {
            "supplier": "supplier_id",
            "purchase_order": "purchase_order_id",
            "historical_transaction": "transaction_id",
        }[kind]
        if payload.get(own_key) != fact.record_id:
            failures.append("procurement_record_mismatch")
            continue
        buckets[kind].append(fact)
    if (
        len(buckets["purchase_order"]) != 1
        or len(buckets["supplier"]) != 1
        or not buckets["historical_transaction"]
    ):
        failures.append("procurement_cardinality_unknown")
    identities = [
        (BUSINESS_FACT_TOOL_KINDS.get(f.tool_name), f.record_id, f.dataset_version)
        for f in result.facts
    ]
    if len(identities) != len(set(identities)):
        failures.append("procurement_duplicate_record")
    if failures:
        return ProcurementValuesV3(
            None,
            None,
            (),
            finding(
                "O-10",
                "unknown",
                "source_unavailable_or_incompatible",
                missing=("admitted_read_only_source",),
            ),
            tuple(sorted(set(failures))),
        )
    order, supplier = buckets["purchase_order"][0], buckets["supplier"][0]
    transactions = tuple(buckets["historical_transaction"])
    sources = tuple(
        field_evidence(f, "identity") for f in (order, supplier, *transactions)
    )
    return ProcurementValuesV3(
        order,
        supplier,
        transactions,
        finding(
            "O-10",
            "supported",
            "source_identity_complete",
            sources,
        ),
        (),
    )
