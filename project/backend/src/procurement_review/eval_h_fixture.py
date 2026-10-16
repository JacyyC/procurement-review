"""Disposable authored DEV material; this builder never produces an unseen batch."""

from pathlib import Path
import secrets

from .eval_g_retrieval import G_DATASET_VERSION, G_INDEX_VERSION
from .eval_h_contracts import (
    C_PROBE_SEED,
    H_PLAN,
    H_PLAN_V2,
    H_PROBE,
    HCase,
    HDocument,
    HLabel,
    HPlan,
    HPlanV2,
    HProbePlan,
    HRecipe,
    HRecord,
    HSupport,
    PROBE_CASE_RANGE,
)
from .eval_h_identity import identity
from .eval_h_seal import HSeal, plan_digest
from .eval_h_storage import sha, write, write_bytes
from .gold_v3_contracts import payload_sha256
from .gold_v3_contracts import V3Recommendation
from .gold_v37_conditional import as_gold_case
from .gold_v38_dev_quality import dev_quality_cases, id_map, mapped_case
from .precise_candidate import truth_extension_v3
from .precise_dev_paired import paired_configuration
from .precise_dev_sweep import mapped_truth


def fixture_case(case):
    mapped = mapped_case(case)
    gold = as_gold_case(mapped)
    facts, mapping = case.facts, id_map(case)
    order_id, supplier_id, transaction_id = (
        mapping[str(facts[name])]
        for name in ("purchase_order_id", "supplier_id", "transaction_id")
    )
    records = (
        HRecord(
            kind="purchase_order",
            record_id=order_id,
            payload={
                "purchase_order_id": order_id,
                "supplier_id": supplier_id,
                "competitive_quote_count": facts["competitive_quote_count"],
                "contract_reference": mapping[str(facts["contract_reference"])],
                "ordered_on": facts["ordered_on"],
                "invoice_amount_minor": facts["invoice_amount_minor"],
                "invoice_currency": facts["invoice_currency"],
                "amount_minor": facts["order_amount_minor"],
                "currency": facts["order_currency"],
                **(
                    {"transaction_category": case.transaction_category}
                    if case.transaction_category
                    else {}
                ),
            },
        ),
        HRecord(
            kind="supplier",
            record_id=supplier_id,
            payload={
                "supplier_id": supplier_id,
                **{
                    name: facts[name]
                    for name in (
                        "registry_status",
                        "procurement_status",
                        "tax_identifier",
                        "due_diligence_status",
                        "sanctions_screening",
                    )
                },
            },
        ),
        HRecord(
            kind="historical_transaction",
            record_id=transaction_id,
            payload={
                "transaction_id": transaction_id,
                "purchase_order_id": order_id,
                "supplier_id": supplier_id,
                "counterparty_reference": mapping[str(facts["counterparty_reference"])],
                "booked_on": facts["booked_on"],
                "amount_minor": facts["transaction_amount_minor"],
                "currency": facts["transaction_currency"],
            },
        ),
    )
    documents = tuple(
        HDocument(
            role=role,
            source_id=document.source_id,
            version=document.version,
            document_id=f"H-DOC-{case.case_id}-{role.upper()}",
            content=document.content,
            content_sha256=document.content_sha256,
        )
        for role in ("policy", "contract")
        for document in (getattr(gold.model_input, role + "_document"),)
    )
    return HCase(
        case_id=case.case_id,
        generation_family_id="h-family-disposable",
        near_duplicate_group_id="h-group-disposable",
        entity_root_id="h-entity-disposable",
        metamorphic_parent_id=None,
        question=case.question,
        as_of=gold.as_of.isoformat(),
        documents=documents,
        records=records,
        purchase_order_id=order_id,
    )


def fixture_label(case):
    truth, extension = mapped_truth(case), truth_extension_v3(case)
    support = tuple(
        HSupport(
            role=key.split(".")[0],
            obligation_id=duty,
            field=descriptor["field"],
            value=descriptor["value"],
            clauses=tuple(descriptor["clauses"]),
            requirement=descriptor["requirement"],
        )
        for duty, fields in extension["support"].items()
        for key, descriptor in fields.items()
    )
    material = fixture_case(case)
    return HLabel(
        case_id=case.case_id,
        recommendation=V3Recommendation(truth["recommendation"]),
        high_risk=truth["high_risk"],
        expectations=tuple(truth["expectations"]),
        support=support,
        required_sources={
            role: tuple(
                document.source_id
                for document in material.documents
                if document.role == role
            )
            for role in ("policy", "contract")
        },
        conflict_pairs=tuple(extension["conflict_pairs"]),
        security_events=tuple(extension["security_events"]),
        required_facts={
            key: tuple(value) for key, value in extension["required_facts"].items()
        },
        adjudication="independent authored DEV truth; already seen",
    )


def make_disposable(
    root: Path,
    count: int = 2,
    *,
    batch_id="disposable-h-prep",
    plan_version=H_PLAN,
    mode=None,
    thinking=None,
    generation_deadline_seconds=None,
    rate_limit_window_seconds=None,
    rate_limit_burst_threshold=None,
):
    """Build one disposable or probe batch over authored DEV material.

    The historical V1-07 plan keeps its exact behaviour.  The V1-08 contract requires its
    registered parameters — thinking, one accepted deadline and the dense-429 window and
    threshold — and refuses a batch that leaves any of them unstated rather than inventing
    a value here.  The C probe shape requires the same parameters plus its own registered
    size (8–12 already-seen DEV Cases), the thinking candidate and the 180-second probe
    deadline, and it is the only shape here that never registers a second repeat.
    """

    root = Path(root).absolute()
    if ".local" not in root.parts or root.is_symlink():
        raise ValueError("h_private_local_root_required")
    if type(count) is not int or not 1 <= count <= 12:
        raise ValueError("h_fixture_count")
    supplied = (
        mode,
        thinking,
        generation_deadline_seconds,
        rate_limit_window_seconds,
        rate_limit_burst_threshold,
    )
    if plan_version == H_PLAN and any(value is not None for value in supplied):
        raise ValueError("h_fixture_contract_parameter_without_v2")
    if plan_version not in (H_PLAN, H_PLAN_V2, H_PROBE):
        raise ValueError("h_fixture_plan_version_unknown")
    if plan_version == H_PROBE and (
        mode not in (None, "probe")
        or not PROBE_CASE_RANGE[0] <= count <= PROBE_CASE_RANGE[1]
    ):
        raise ValueError("h_fixture_probe_shape")
    source = dev_quality_cases()[:count]
    recipe = HRecipe(
        generator_version="v1-07h-material-generator.v1",
        cases=tuple(fixture_case(case) for case in source),
        labels=tuple(fixture_label(case) for case in source),
    )
    current = identity()
    exclusion = dict(
        schema_version="v1-07h-exclusion.v1",
        source_sha256={},
        keys=[],
        materials=[],
        archived_case_count=0,
        archived_cases_loaded=0,
        source_identities=0,
        source_contents_loaded=0,
        missing_cases=[],
        missing_source_contents=[],
        incomplete_relations=[],
        history_complete=False,
    )
    exclusion["registry_sha256"] = payload_sha256(exclusion)
    seed = secrets.token_bytes(32)
    import hashlib
    from .eval_h_storage import json_bytes

    recipe_body = recipe.model_dump(mode="json")
    batch_mode = "probe" if plan_version == H_PROBE else (mode or "disposable")
    safety = dict(
        schema_version="v1-07h-safety.v1",
        product_sha256=current["product_sha256"],
        mode=batch_mode,
        categories={},
        sources={},
        reason="not_measured",
    )
    shared = dict(
        batch_id=batch_id,
        **{
            name: current[name]
            for name in ("candidate_sha256", "consumer_sha256", "product_sha256")
        },
        generator_spec_sha256=hashlib.sha256(json_bytes(recipe_body)).hexdigest(),
        seed_sha256=hashlib.sha256(seed).hexdigest(),
        exclusion_sha256=exclusion["registry_sha256"],
        case_ids=tuple(case.case_id for case in recipe.cases),
        repeat_seeds=(C_PROBE_SEED,) if plan_version == H_PROBE else (101, 202),
        document_version=recipe.cases[0].documents[0].version,
        dataset_version=G_DATASET_VERSION,
        index_version=G_INDEX_VERSION,
        tenant_id="tenant-v1-07h-disposable",
        subject_id="subject-v1-07h-disposable",
        authorization="explicit disposable mock only",
        lifecycle_evidence_sha256=hashlib.sha256(json_bytes(safety)).hexdigest(),
    )
    if plan_version == H_PROBE:
        if (
            thinking is None
            or generation_deadline_seconds is None
            or rate_limit_window_seconds is None
            or rate_limit_burst_threshold is None
        ):
            raise ValueError("h_fixture_probe_parameters_required")
        plan = HProbePlan(
            configuration=paired_configuration().model_copy(
                update={"thinking": thinking}
            ),
            generation_deadline_seconds=generation_deadline_seconds,
            rate_limit_window_seconds=rate_limit_window_seconds,
            rate_limit_burst_threshold=rate_limit_burst_threshold,
            **shared,
        )
    elif plan_version == H_PLAN_V2:
        if (
            thinking is None
            or generation_deadline_seconds is None
            or rate_limit_window_seconds is None
            or rate_limit_burst_threshold is None
        ):
            raise ValueError("h_fixture_v2_parameters_required")
        plan = HPlanV2(
            mode=batch_mode,
            configuration=paired_configuration().model_copy(
                update={"thinking": thinking}
            ),
            generation_deadline_seconds=generation_deadline_seconds,
            rate_limit_window_seconds=rate_limit_window_seconds,
            rate_limit_burst_threshold=rate_limit_burst_threshold,
            **shared,
        )
    else:
        plan = HPlan(
            mode="disposable",
            configuration=paired_configuration(),
            **shared,
        )
    seal = HSeal.create(root, plan)
    write(seal.directory / "private/recipe.json", recipe_body)
    write_bytes(seal.directory / "private/seed.bin", seed)
    write(seal.directory / "private/registry.json", exclusion)
    write(seal.directory / "private/safety.json", safety)
    return {
        "root": str(seal.directory),
        "plan_sha256": plan_digest(plan),
        "case_count": count,
        "network_calls": 0,
        "safety_sha256": sha(seal.directory / "private/safety.json"),
    }
