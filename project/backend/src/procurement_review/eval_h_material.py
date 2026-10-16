"""Product registration and independent evidence identities for H material."""

from dataclasses import replace

from .eval_g_retrieval import document_row
from .eval_h_contracts import HCase, HPlan
from .precise_candidate import canonical_json_sha256
from .security_fixture import (
    SecuritySeedPlan,
    SeedCase,
    SeedProcurementRecord,
    SeedSubject,
    SeedTenant,
)


def documents(plan: HPlan, case: HCase):
    result = []
    for source in case.documents:
        if source.version != plan.document_version:
            raise ValueError("h_material_document_version")
        row = document_row(
            source_id=source.source_id,
            content=source.content,
            document_id=source.document_id,
            ownership="case_document",
            tenant_id=plan.tenant_id,
            case_id=case.case_id,
            category=source.role,
            document_version=source.version,
        )
        result.append(
            replace(
                row,
                index_version=plan.index_version,
                chunk_payload={
                    **row.chunk_payload,
                    "index_version": plan.index_version,
                },
            )
        )
    return tuple(result)


def source_identities(plan, case):
    return {
        row.source_id: dict(
            source_id=row.source_id,
            version=row.document_version,
            locator=row.chunk_id,
            content_sha256=row.content_checksum[7:],
        )
        for row in documents(plan, case)
    }


def seed_plan(plan: HPlan, cases: tuple[HCase, ...]) -> SecuritySeedPlan:
    docs, records, registrations = [], [], []
    for case in cases:
        own = documents(plan, case)
        docs.extend(own)
        records.extend(
            SeedProcurementRecord(
                plan.tenant_id,
                case.case_id,
                {
                    "purchase_order": "order",
                    "supplier": "supplier",
                    "historical_transaction": "transaction",
                }[record.kind],
                record.record_id,
                plan.dataset_version,
                case.as_of,
                record.payload,
            )
            for record in case.records
        )
        ids = {
            kind: tuple(
                record.record_id for record in case.records if record.kind == kind
            )
            for kind in ("purchase_order", "supplier", "historical_transaction")
        }
        registrations.append(
            SeedCase(
                tenant_id=plan.tenant_id,
                case_id=case.case_id,
                label=case.case_id,
                case_type="purchase_order",
                scenario="low",
                purchase_order_id=case.purchase_order_id,
                case_version=case.case_id.lower() + ".h1",
                document_version=plan.document_version,
                dataset_version=plan.dataset_version,
                index_version=plan.index_version,
                rules_version="3.1.0",
                case_document_ids=tuple(row.document_id for row in own),
                shared_document_ids=(),
                order_ids=ids["purchase_order"],
                supplier_ids=ids["supplier"],
                transaction_ids=ids["historical_transaction"],
                granted_subject_ids=(plan.subject_id,),
            )
        )
    return SecuritySeedPlan(
        fixture_version="v1-07h-seed.v1",
        document_version=plan.document_version,
        dataset_version=plan.dataset_version,
        index_version=plan.index_version,
        tenants=(SeedTenant(plan.tenant_id, "H isolated material"),),
        subjects=(
            SeedSubject(
                plan.tenant_id,
                plan.subject_id,
                "H operator",
                ("operator", "approver", "auditor"),
                plan.case_ids,
            ),
        ),
        cases=tuple(registrations),
        documents=tuple(docs),
        records=tuple(records),
        retrieval_scores={},
    )


def truth_registration(plan, cases, labels):
    """Adapt explicit H labels to independent v3 field adjudication, without product rules."""
    truths, extensions, material = {}, {}, {}
    for case in cases:
        label = next(label for label in labels if label.case_id == case.case_id)
        sources = source_identities(plan, case)
        branch_sources, texts = {}, {}
        # The legacy adjudicator needs one source per worker. The H span adapter
        # below selects the actual claimed document when a domain has several.
        for document in case.documents:
            branch_sources.setdefault(document.role, sources[document.source_id])
            texts.setdefault(document.role, document.content)
        support = {}
        for item in label.support:
            support.setdefault(item.obligation_id, {})[item.role + "." + item.field] = {
                "field": item.field,
                "value": item.value,
                "clauses": list(item.clauses),
                "requirement": item.requirement,
            }
        facts = {}
        for record in case.records:
            for field, value in {
                **record.payload,
                "identity": record.record_id,
            }.items():
                facts[record.kind + "." + field] = dict(
                    subject_id=record.record_id,
                    subject_type="transaction"
                    if record.kind == "historical_transaction"
                    else record.kind,
                    effective_at=case.as_of,
                    source_id=record.record_id,
                    version=plan.dataset_version,
                    content_sha256=canonical_json_sha256(record.payload),
                    value=value,
                )
        truths[case.case_id] = dict(
            recommendation=label.recommendation.value,
            high_risk=label.high_risk,
            expectations=list(label.expectations),
        )
        conflicts = []
        for pair in label.conflict_pairs:
            sides = {}
            for name in ("left", "right"):
                side = dict(pair[name])
                descriptor = facts.get(side["locator"], {})
                side.update(
                    scope_id=plan.tenant_id + ":" + case.case_id,
                    subject_id=case.purchase_order_id,
                    subject_type="purchase_order",
                    effective_at=case.as_of,
                    source_version=plan.dataset_version,
                    source_id=descriptor.get("source_id"),
                    content_sha256=descriptor.get("content_sha256"),
                )
                sides[name] = side
            conflicts.append({**pair, **sides})
        from .precise_candidate import location_sha256

        security = [
            {
                **event,
                "location_sha256": location_sha256(branch_sources[event["document"]]),
            }
            for event in label.security_events
        ]
        extensions[case.case_id] = dict(
            subject_id=case.purchase_order_id,
            as_of=case.as_of,
            scope_id=plan.tenant_id + ":" + case.case_id,
            support=support,
            branch_sources=branch_sources,
            facts=facts,
            required_facts=label.required_facts,
            conflict_pairs=conflicts,
            security_events=security,
        )
        material[case.case_id] = texts
    return dict(
        case_ids=list(plan.case_ids),
        truth=truths,
        truth_extension=extensions,
        material_text=material,
        extends={"extension_version": "v1-07-truth-extension.v3"},
    )
