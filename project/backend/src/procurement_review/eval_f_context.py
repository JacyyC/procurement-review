"""Oracle-context F input assembly, deliberately independent of labels/scoring.

F reads synthetic facts directly. This is not G's PostgreSQL/MCP path.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from pathlib import Path
import hashlib
import json

from .business_v3_types import BusinessContextV3, BusinessEvidenceIdentity
from .business_v3_support import LocatedSourceV3
from .contracts import FactStatus
from .model_wire_v2 import SourceSlice
from .worker_contracts import ProcurementReadStatus
from .gold_v31_contracts import CasesFileV31, ManifestV31

DATA = Path(__file__).parent / "resources/gold_set/v3.1"


def data_directory(version="gold-set.v3.1"):
    if version not in {
        "gold-set.v3.1",
        "gold-set.v3.2",
        "gold-set.v3.3",
        "gold-set.v3.4",
        "gold-set.v3.5",
        "gold-set.v3.6",
    }:
        raise ValueError("unsupported F dataset version")
    return DATA.parent / version.removeprefix("gold-set.")


def load_cases(split="dev", version="gold-set.v3.1"):
    # This module never opens truth.json. The scorer has a separate entry point.
    cases_type, manifest_type = CasesFileV31, ManifestV31
    if version == "gold-set.v3.2":
        from .gold_v32_contracts import CasesFileV32, ManifestV32

        cases_type, manifest_type = CasesFileV32, ManifestV32
    elif version == "gold-set.v3.3":
        from .gold_v33_contracts import CasesFileV33, ManifestV33

        cases_type, manifest_type = CasesFileV33, ManifestV33
    elif version == "gold-set.v3.4":
        from .gold_v34_contracts import CasesFileV34, ManifestV34

        cases_type, manifest_type = CasesFileV34, ManifestV34
    elif version == "gold-set.v3.5":
        from .gold_v35_contracts import CasesFileV35, ManifestV35

        cases_type, manifest_type = CasesFileV35, ManifestV35
    elif version == "gold-set.v3.6":
        from .gold_v36_contracts import CasesFileV36, ManifestV36

        cases_type, manifest_type = CasesFileV36, ManifestV36
    data = data_directory(version)
    cases = cases_type.model_validate_json((data / "cases.json").read_bytes())
    manifest = manifest_type.model_validate_json((data / "manifest.json").read_bytes())
    if cases.payload_sha256 != manifest.cases_payload_sha256:
        raise ValueError("case manifest mismatch")
    return tuple(c for c in cases.cases if c.split.value == split)


def input_fingerprints(version="gold-set.v3.1"):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            data_directory(version) / "cases.json",
            data_directory(version) / "manifest.json",
        )
    }


def context_for(case):
    f = case.model_input.business_facts
    return BusinessContextV3(
        scope_id=f.scope_id,
        purchase_order_id=f.purchase_order_id,
        supplier_id=f.supplier_id,
        contract_id=f.contract_reference or f.purchase_order_id,
        as_of=case.as_of,
        dataset_version=f.source_version,
    )


def located_sources(case):
    return tuple(
        LocatedSourceV3(
            SourceSlice(
                scope_id=case.model_input.business_facts.scope_id,
                worker=role,
                identity=BusinessEvidenceIdentity(
                    source_id=doc.source_id,
                    version=doc.version,
                    locator=doc.section,
                    content_sha256=doc.content_sha256,
                ),
                content=doc.content,
            ),
            "purchase_order",
            case.model_input.business_facts.purchase_order_id,
            case.as_of,
        )
        for role, doc in (
            ("policy", case.model_input.policy_document),
            ("contract", case.model_input.contract_document),
        )
    )


def oracle_procurement(case):
    """Map declared facts without adding defaults for missing business values.

    A private in-memory port permits null fields from the newer dataset; it does
    not bypass or widen the old product BusinessFact persistence guard.
    """
    f = case.model_input.business_facts
    data = f.model_dump(mode="json")
    order = {
        k: data[k]
        for k in (
            "purchase_order_id",
            "supplier_id",
            "competitive_quote_count",
            "contract_reference",
            "ordered_on",
            "invoice_amount_minor",
            "invoice_currency",
        )
    }
    order.update(amount_minor=f.order_amount_minor, currency=f.order_currency)
    supplier = {
        k: data[k]
        for k in (
            "supplier_id",
            "registry_status",
            "procurement_status",
            "tax_identifier",
            "due_diligence_status",
            "sanctions_screening",
        )
    }
    tx = {
        k: data[k]
        for k in (
            "transaction_id",
            "purchase_order_id",
            "supplier_id",
            "counterparty_reference",
            "booked_on",
        )
    }
    tx.update(amount_minor=f.transaction_amount_minor, currency=f.transaction_currency)
    rows = [
        ("get_purchase_order", f.purchase_order_id, order),
        ("get_supplier", f.supplier_id, supplier),
        ("list_historical_transactions", f.transaction_id, tx),
    ]
    facts = tuple(
        SimpleNamespace(
            tool_name=tool,
            record_id=identity,
            read_only=True,
            dataset_version=f.source_version,
            as_of=datetime.combine(case.as_of, datetime.min.time(), UTC),
            status=FactStatus.KNOWN,
            result=body,
        )
        for tool, identity, body in rows
    )
    return SimpleNamespace(
        status=ProcurementReadStatus.KNOWN
        if f.source_status == "known"
        else ProcurementReadStatus.UPSTREAM_FAILURE,
        facts=facts,
        requested_dataset_version=f.source_version,
    )


def stable_write(path, value):
    from .eval_attempt_ledger import _atomic_write

    _atomic_write(
        path,
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode(),
    )
