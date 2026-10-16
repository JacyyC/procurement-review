"""``v1-07g-dev-sweep.v1``: the pre-registered DEV product sweep and its seeding.

The sweep is the *product* path and nothing else: every observation comes from one Run the
workbench API created, waited for and read back (``ReviewService`` → precise runner →
endpoint), and the Case material is the one the independent truth of
:mod:`procurement_review.gold_v38_dev_quality` was authored for.  Nothing here scores: the
observations are written for :mod:`procurement_review.precise_dev_score`, which judges them
per repeat against that truth.

The three steps are deliberately separable so the registration is frozen before anything is
sent:

* ``prepare`` writes the pre-registration — cases, repeats, material hashes, the mapped
  independent truth, the implementation digests and the candidate fingerprint — with no
  network access at all;
* ``seed`` builds one isolated PostgreSQL database for the tenent, registers the twelve
  Cases' material and read-only records, and issues one local credential;
* ``run`` drives the API for the registered plan and persists every observation, including
  the per-attempt facts the first-answer reading needs.

The record ids the product's read-only tools answer are product-shaped (``PO-901``…), so the
documents and the records are mapped together: the text the model sees, the facts it is
compared with and the role of each identifier stay the same material.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .eval_g_retrieval import G_DATASET_VERSION, G_INDEX_VERSION, document_row
from .gold_v37_conditional import as_gold_case
from .gold_v38_dev_quality import (
    CASE_IDS,
    DEV_QUALITY_SET_VERSION,
    DEV_QUALITY_TRUTH_VERSION,
    DevQualityCaseV1,
    dev_quality_cases,
    dev_quality_truth,
    id_map,
    mapped_case,
)
from .precise_candidate import truth_extension, project_base
from .precise_dev_paired_metering import bind_runs as bind_runs
from .precise_dependencies import PRODUCT_ROOTS, DIAGNOSTIC_ROOTS, package_closure
from .security_fixture import (
    SecuritySeedPlan,
    SeedCase,
    SeedDocument,
    SeedProcurementRecord,
    SeedSubject,
    SeedTenant,
)

DEV_SWEEP_VERSION = "v1-07g-dev-sweep.v1"
SWEEP_TENANT = "tenant-v1-07g-dev-quality"
SWEEP_SUBJECT = "subject-v1-07g-dev-quality"
BROWSER_SAFE_REPEATS = 2
DEFAULT_REPEATS = 2
MAX_ATTEMPTS_PER_BRANCH = 2
DEFAULT_TIMEOUT_SECONDS = 240.0
#: The G line's registered model identities (Spec §6.4): the scorer refuses a reading
#: whose Run resolved anything else.
SWEEP_REQUESTED_MODEL = "deepseek-v4.1-flash"
SWEEP_RESOLVED_MODEL = "deepseek-v4-1-flash"
#: The versions one Run of this sweep freezes; the binding block registers them so a
#: reading can be refused when its ledger or snapshot describes another build.
SWEEP_EXECUTION_VERSION = "review-execution.v2"
SWEEP_SUMMARY_SCHEMA = "1.0"
SWEEP_RULES_VERSION = "3.1.0"
#: The verification extension's own version: the identity/position/support metadata the
#: v2 scorer reads beside the unchanged v1.8 labels.
SWEEP_TRUTH_EXTENSION_VERSION = "v1-07-truth-extension.v1"
#: The keys the scorer adds on top of the pre-registered base file.  The extension is a
#: separate block on purpose: the base file keeps its own bytes, and the projection back
#: to the base is what proves nothing pre-registered was rewritten.
EXTENSION_KEYS = ("truth_extension", "binding", "extends")


def binding_block(cases: tuple[DevQualityCaseV1, ...]) -> dict[str, Any]:
    """The build and version identities every observed Run of this plan must carry."""

    from .precise_run_scope import implementation_fingerprint

    versions = {
        as_gold_case(mapped_case(case)).model_input.policy_document.version
        for case in cases
    }
    if len(versions) != 1:
        raise ValueError("the plan's cases do not share one document version")
    return {
        "expected_rules_version": SWEEP_RULES_VERSION,
        "expected_document_version": versions.pop(),
        "expected_dataset_version": G_DATASET_VERSION,
        "expected_index_version": G_INDEX_VERSION,
        "expected_execution_version": SWEEP_EXECUTION_VERSION,
        "expected_summary_schema": SWEEP_SUMMARY_SCHEMA,
        "product_implementation_sha256": implementation_fingerprint(),
        "requested_model": SWEEP_REQUESTED_MODEL,
        "resolved_model": SWEEP_RESOLVED_MODEL,
        "branches": ["policy", "contract"],
        "max_attempts_per_branch": MAX_ATTEMPTS_PER_BRANCH,
    }


def mapped_truth(case: DevQualityCaseV1) -> dict[str, Any]:
    """The independent truth, expressed in the identifiers the product will answer with."""

    truth = dev_quality_truth(case)
    replacements = id_map(case)

    def map_text(value: str) -> str:
        for source, target in sorted(
            replacements.items(), key=lambda item: -len(item[0])
        ):
            value = value.replace(source, target)
        return value

    material = {
        "policy_sha256": hashlib.sha256(case.policy_text.encode()).hexdigest(),
        "contract_sha256": hashlib.sha256(case.contract_text.encode()).hexdigest(),
        "variant": case.variant,
    }
    expectations = []
    for row in truth.expectations:
        body = row.model_dump(mode="json")
        body.pop("case_id", None)
        # The support registry travels in the extension block, not in the base shape: the
        # pre-registered file keeps its own bytes and the scorer gets the extra semantics
        # as an explicit, separately digestible addition.
        body.pop("support", None)
        # The clauses and required substrings name the identifiers the product will answer
        # with, so the truth is expressed in the same identifiers as the material.
        body["document_clauses"] = [map_text(item) for item in body["document_clauses"]]
        body["contains"] = {
            field: [map_text(item) for item in needles]
            for field, needles in body["contains"].items()
        }
        body["values"] = {
            field: (map_text(value) if isinstance(value, str) else value)
            for field, value in body["values"].items()
        }
        expectations.append(body)
    return {
        "case_id": case.case_id,
        "variant": truth.variant,
        "material": material,
        "material_sha256": truth.material_sha256,
        "expectations": expectations,
        "recommendation": truth.recommendation,
        "conflict_relations": list(truth.conflict_relations),
        "security_codes": list(truth.security_codes),
        "high_risk": truth.high_risk,
    }


def dev_quality_seed_plan(
    cases: tuple[DevQualityCaseV1, ...] | None = None,
) -> SecuritySeedPlan:
    """One isolated registry: the twelve Cases, their documents and their fact rows."""

    chosen = cases if cases is not None else dev_quality_cases()
    documents: list[SeedDocument] = []
    records: list[SeedProcurementRecord] = []
    seeded: list[SeedCase] = []
    version = None
    for case in chosen:
        mapped = mapped_case(case)
        gold = as_gold_case(mapped)
        version = gold.model_input.policy_document.version
        document_ids = []
        for domain, document in (
            ("policy", gold.model_input.policy_document),
            ("contract", gold.model_input.contract_document),
        ):
            document_id = f"DOC-{case.case_id}-{domain.upper()}"
            document_ids.append(document_id)
            documents.append(
                document_row(
                    source_id=document.source_id,
                    content=document.content,
                    document_id=document_id,
                    ownership="case_document",
                    tenant_id=SWEEP_TENANT,
                    case_id=case.case_id,
                    category=domain,
                    document_version=version,
                )
            )
        mapping = id_map(case)
        order_id = mapping[str(case.facts["purchase_order_id"])]
        records.extend(_procurement_rows(case))
        seeded.append(
            SeedCase(
                tenant_id=SWEEP_TENANT,
                case_id=case.case_id,
                label=case.case_id,
                case_type="purchase_order",
                scenario="low",
                purchase_order_id=order_id,
                case_version=f"{case.case_id.lower()}.v1",
                document_version=version,
                dataset_version=G_DATASET_VERSION,
                index_version=G_INDEX_VERSION,
                rules_version="3.1.0",
                case_document_ids=tuple(document_ids),
                shared_document_ids=(),
                order_ids=(order_id,),
                supplier_ids=(mapping[str(case.facts["supplier_id"])],),
                transaction_ids=(mapping[str(case.facts["transaction_id"])],),
                granted_subject_ids=(SWEEP_SUBJECT,),
            )
        )

    return SecuritySeedPlan(
        fixture_version=f"{DEV_SWEEP_VERSION}.seed",
        document_version=version,
        dataset_version=G_DATASET_VERSION,
        index_version=G_INDEX_VERSION,
        tenants=(SeedTenant(SWEEP_TENANT, "V1-07G DEV quality tenant"),),
        subjects=(
            SeedSubject(
                tenant_id=SWEEP_TENANT,
                subject_id=SWEEP_SUBJECT,
                display_name="V1-07G DEV quality operator",
                roles=("operator", "approver", "auditor"),
                case_grants=CASE_IDS,
            ),
        ),
        cases=tuple(seeded),
        documents=tuple(documents),
        records=tuple(records),
        retrieval_scores={},
    )


def _procurement_rows(
    case: DevQualityCaseV1,
) -> tuple[SeedProcurementRecord, ...]:
    """The read-only fact rows one Case's MCP boundary answers, in product identifiers.

    The identifiers come from the Case's own mapping and everything else from its authored
    facts, so the records and the documents describe the same Case.
    """

    facts = case.facts
    mapping = id_map(case)
    order_id = mapping[str(case.facts["purchase_order_id"])]
    supplier_id = mapping[str(case.facts["supplier_id"])]
    transaction_id = mapping[str(case.facts["transaction_id"])]
    order = {
        "purchase_order_id": order_id,
        "supplier_id": supplier_id,
        "competitive_quote_count": facts["competitive_quote_count"],
        "contract_reference": mapping[str(case.facts["contract_reference"])],
        "ordered_on": facts["ordered_on"],
        "invoice_amount_minor": facts["invoice_amount_minor"],
        "invoice_currency": facts["invoice_currency"],
        "amount_minor": facts["order_amount_minor"],
        "currency": facts["order_currency"],
    }
    if case.transaction_category is not None:
        order["transaction_category"] = case.transaction_category
    supplier = {
        "supplier_id": supplier_id,
        "registry_status": facts["registry_status"],
        "procurement_status": facts["procurement_status"],
        "tax_identifier": facts["tax_identifier"],
        "due_diligence_status": facts["due_diligence_status"],
        "sanctions_screening": facts["sanctions_screening"],
    }
    transaction = {
        "transaction_id": transaction_id,
        "purchase_order_id": order_id,
        "supplier_id": supplier_id,
        "counterparty_reference": mapping[str(case.facts["counterparty_reference"])],
        "booked_on": facts["booked_on"],
        "amount_minor": facts["transaction_amount_minor"],
        "currency": facts["transaction_currency"],
    }
    as_of = as_gold_case(case).as_of.isoformat()
    return tuple(
        SeedProcurementRecord(
            tenant_id=SWEEP_TENANT,
            case_id=case.case_id,
            kind=kind,
            record_id=record_id,
            dataset_version=G_DATASET_VERSION,
            as_of=as_of,
            payload=payload,
        )
        for kind, record_id, payload in (
            ("order", order_id, order),
            ("supplier", supplier_id, supplier),
            ("transaction", transaction_id, transaction),
        )
    )


def implementation_digests() -> dict[str, str]:
    """The digests of the implementation this sweep's readings are bound to."""

    from .precise_candidate import candidate_fingerprint

    from .precise_dependencies import module_path

    names = package_closure(PRODUCT_ROOTS + DIAGNOSTIC_ROOTS)
    digests = {
        name: hashlib.sha256(module_path(name).read_bytes()).hexdigest()
        for name in names
    }
    fingerprint = candidate_fingerprint()
    digests["candidate_sha256"] = fingerprint["candidate_sha256"]
    return digests


def registration(
    repeats: int = DEFAULT_REPEATS, case_ids: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """The frozen manifest: what is sent, over what material, and why.

    ``case_ids`` narrows the plan to a pre-registered subset: the same material and the
    same truth, a smaller denominator, for a bounded probe before a full sweep.
    """

    chosen = tuple(case_ids) if case_ids else CASE_IDS
    cases = tuple(case for case in dev_quality_cases() if case.case_id in chosen)
    if len(cases) != len(chosen):
        raise ValueError("the pre-registered subset names an unknown Case")
    return {
        "contract": DEV_SWEEP_VERSION,
        "stage": "V1-07G",
        "revision": 1,
        "question": (
            "同候选的真实产品路径在完整 DEV 质量集上，逐重复的适用质量门是否成立？"
        ),
        "information_gain": (
            "给定候选、独立真值与预注册重复，逐重复给出义务状态/值/来源/证据位置、"
            "五类建议与宏 F1、批准精确率、高风险放行、升级与额外接管、冲突与安全的"
            "精确匹配、引用支持与首答有效性；零分母判为不可判定，不用合并读数代替。"
        ),
        "protection_set": (
            "仅使用自撰合成材料（gold-set.v3.8 的十二个 DEV 用例），"
            "不含真实企业数据、凭据或个人信息；结果只保留有界投影与计量。"
        ),
        "case_ids": [case.case_id for case in cases],
        "repeats": repeats,
        "plan": {
            "cases": len(cases),
            "repeats": repeats,
            "runs": len(cases) * repeats,
        },
        "limits": {
            "max_attempts_per_branch": MAX_ATTEMPTS_PER_BRANCH,
            "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        },
        "stop_conditions": [
            "任一 Run 在期限内未到终态即记该观测为未完成，不重发",
            "认证、模型身份漂移或安全边界事件立即停止整个批次",
            "批次期限内不发新 Run；已发请求照常结算",
        ],
        "material": {case.case_id: mapped_truth(case)["material"] for case in cases},
        # The mapped text itself, so a score can be recomputed from persisted inputs alone
        # while the hashes above still bind it to the authored material.
        "material_text": {
            case.case_id: {
                "policy": mapped_case(case).policy_text,
                "contract": mapped_case(case).contract_text,
            }
            for case in cases
        },
        "truth": {case.case_id: mapped_truth(case) for case in cases},
        "set_version": DEV_QUALITY_SET_VERSION,
        "truth_version": DEV_QUALITY_TRUTH_VERSION,
        "implementation": implementation_digests(),
        "candidate_sha256": implementation_digests()["candidate_sha256"],
    }


def seed(state_path: Path) -> dict[str, Any]:
    """One isolated database, the twelve Cases' material, and one local credential."""

    import sys

    from sqlalchemy import create_engine, text

    from .authorization import generate_credential
    from .review_store import ReviewStore

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from v1_03_support import _url_for_database, require_admin_dsn

    base = require_admin_dsn()
    database = f"v1_07g_sweep_{secrets.token_hex(4)}"
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(
                text(f'CREATE DATABASE "{database}" WITH TEMPLATE = template0')
            )
    finally:
        admin.dispose()
    url = _url_for_database(base, database).replace(
        "application_name=v1_03_security_recovery",
        "application_name=v1_07g_dev_sweep",
    )
    store = ReviewStore.from_url(url)
    store.migrate()
    store.seed(dev_quality_seed_plan())
    token, credential_id, digest = generate_credential()
    now = datetime.now(UTC)
    store.issue_credential(
        credential_id=credential_id,
        tenant_id=SWEEP_TENANT,
        subject_id=SWEEP_SUBJECT,
        token_digest=digest,
        issued_at=now,
        expires_at=now + timedelta(days=1),
    )
    store.close()
    state = {
        "database": database,
        "database_url": url,
        "admin_dsn": base,
        "tenant_id": SWEEP_TENANT,
        "subject_id": SWEEP_SUBJECT,
        "credential_id": credential_id,
        "token": token,
        "case_ids": list(CASE_IDS),
        "created_at": now.isoformat(),
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    return {key: value for key, value in state.items() if key != "token"}


def _summarise(run: dict) -> dict[str, Any]:
    execution = run.get("execution") or {}
    decision = execution.get("decision") or {}
    spans = []
    for worker in ("policy", "contract"):
        branch = execution.get(worker) or {}
        for span in branch.get("spans") or []:
            proposition = span.get("proposition") or {}
            spans.append(
                {
                    "worker": worker,
                    "check_id": proposition.get("check_id"),
                    "field": (proposition.get("field_path") or "").rsplit(".", 1)[-1],
                    "field_path": proposition.get("field_path"),
                    "subject_id": proposition.get("subject_id"),
                    "source_version": proposition.get("source_version"),
                    "value": span.get("value"),
                    "start": span.get("start"),
                    "end": span.get("end"),
                    "source_id": (span.get("evidence") or {}).get("source_id"),
                    "version": (span.get("evidence") or {}).get("version"),
                    "locator": (span.get("evidence") or {}).get("locator"),
                    "text": None,
                }
            )
    return {
        "status": run.get("status"),
        "error_code": run.get("error_code"),
        "execution_version": execution.get("execution_version"),
        "execution_mode": run.get("execution_mode"),
        "recommendation": decision.get("recommendation"),
        "findings": [
            {
                "obligation_id": row.get("obligation_id"),
                "state": row.get("state"),
                "reason": row.get("reason"),
                "risk": row.get("risk"),
                "blocking": row.get("blocking"),
                "missing": row.get("missing"),
                "evidence": [
                    {
                        "source_id": item.get("source_id"),
                        "version": item.get("version"),
                        "locator": item.get("locator"),
                    }
                    for item in row.get("evidence") or []
                ],
            }
            for row in decision.get("findings") or []
        ],
        "conflicts": [
            {
                "pair_id": pair.get("pair_id"),
                "kind": pair.get("kind"),
                "relation": pair.get("relation"),
                "left": pair.get("left"),
                "right": pair.get("right"),
            }
            for pair in execution.get("conflicts") or []
        ],
        "security": [
            {
                "code": event.get("code"),
                "origin": event.get("origin"),
                "location_sha256": event.get("location_sha256"),
            }
            for event in execution.get("security") or []
        ],
        "metering": execution.get("metering"),
        "stop_reason": execution.get("stop_reason"),
        "spans": spans,
        "branches": {
            worker: {
                "outcome": (execution.get(worker) or {}).get("outcome"),
                "support_failures": (execution.get(worker) or {}).get(
                    "support_failures"
                ),
                "failure_code": (execution.get(worker) or {}).get("failure_code"),
                "attempts": [
                    {
                        "attempt_no": attempt.get("attempt_no"),
                        "error_code": attempt.get("error_code"),
                        "correctable": attempt.get("correctable"),
                        "latency_ms": attempt.get("latency_ms"),
                        "input_tokens": attempt.get("input_tokens"),
                        "output_tokens": attempt.get("output_tokens"),
                        "usage_unknown": attempt.get("usage_unknown"),
                    }
                    for attempt in (execution.get(worker) or {}).get("attempts") or []
                ],
            }
            for worker in ("policy", "contract")
            if execution.get(worker) is not None
        },
        "summary_schema": (run.get("summary") or {}).get("schema_version"),
        "rules_version": (run.get("summary") or {}).get("rules_version"),
    }


def run_sweep(
    registration_path: Path,
    state_path: Path,
    output_path: Path,
    *,
    client: Any | None = None,
    repeats: int | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Drive the product API for the registered plan and persist every observation."""

    import httpx

    registration_body = json.loads(registration_path.read_text())
    state = json.loads(state_path.read_text())
    base = state.get("api_base", "http://127.0.0.1:8011")
    token = state["token"]
    chosen = list(registration_body["case_ids"])
    plan_repeats = repeats or registration_body["repeats"]
    owns = client is None
    session = client or httpx.Client(base_url=base, timeout=30.0)
    headers = {"Authorization": f"Bearer {token}"}
    started = datetime.now(UTC).isoformat()
    observations: list[dict[str, Any]] = []
    try:
        listed = session.get("/api/v2/cases", headers=headers)
        if listed.status_code != 200:
            raise SystemExit(f"cases listing failed: {listed.status_code}")
        versions = {
            item["case_id"]: item["case_version"] for item in listed.json()["cases"]
        }
        sweep_id = secrets.token_hex(8)
        for case_id in chosen:
            if case_id not in versions:
                observations.append({"case_id": case_id, "missing": True})
                continue
            for repeat in range(1, plan_repeats + 1):
                key = f"gdev-{sweep_id}-{case_id}-r{repeat}"
                created = session.post(
                    "/api/v2/reviews",
                    headers={**headers, "Idempotency-Key": key},
                    json={
                        "api_version": "review-api.v2",
                        "case_id": case_id,
                        "case_version": versions[case_id],
                    },
                )
                if created.status_code not in {200, 201, 202}:
                    observations.append(
                        {
                            "case_id": case_id,
                            "repeat": repeat,
                            "create_status": created.status_code,
                            "create_body": created.text[:400],
                        }
                    )
                    continue
                run_id = created.json()["run_id"]
                begun_at = datetime.now(UTC)
                begin = time.monotonic()
                detail = None
                deadline = begin + timeout_seconds
                while time.monotonic() < deadline:
                    fetched = session.get(f"/api/v2/reviews/{run_id}", headers=headers)
                    if fetched.status_code != 200:
                        break
                    detail = fetched.json()
                    if detail.get("status") in {
                        "pending_confirmation",
                        "pending_human_review",
                        "failed",
                        "cancelled",
                    }:
                        break
                    time.sleep(0.75)
                elapsed = int((time.monotonic() - begin) * 1000)
                observed_at = datetime.now(UTC)
                created_at = detail.get("created_at") if detail else None
                record = {
                    "case_id": case_id,
                    "repeat": repeat,
                    "run_id": run_id,
                    "create_status": created.status_code,
                    "terminal_status": 200 if detail else 0,
                    "elapsed_ms": elapsed,
                    "client_begun_at": begun_at.isoformat(),
                    "service_created_at": created_at,
                    "observed_at": observed_at.isoformat(),
                    **(detail and _summarise(detail) or {"body": None}),
                }
                observations.append(record)
                output_path.write_text(
                    json.dumps(
                        {
                            "sweep_version": DEV_SWEEP_VERSION,
                            "registration_sha256": hashlib.sha256(
                                registration_path.read_bytes()
                            ).hexdigest(),
                            "started_at": started,
                            "observations": observations,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                )
        payload = {
            "sweep_version": DEV_SWEEP_VERSION,
            "registration_sha256": hashlib.sha256(
                registration_path.read_bytes()
            ).hexdigest(),
            "started_at": started,
            "finished_at": datetime.now(UTC).isoformat(),
            "observations": observations,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        )
        return {"runs": len(observations), "written": str(output_path)}
    finally:
        if owns:
            session.close()


def extend_registration(base_path: Path, output_path: Path) -> dict[str, Any]:
    """Add the verification extension to a pre-registered plan without rewriting it.

    The base file's bytes stay the record of what was registered before anything was
    sent.  The extension names, per case, the identities and support the scorer needs;
    it is emitted only after the current oracle reproduces every pre-registered field of
    the base (the implementation digest block excepted, which records the build of that
    moment), so a label that drifted is an error instead of a silent edit.
    """

    base_bytes = base_path.read_bytes()
    base = json.loads(base_bytes)
    if any(key in base for key in EXTENSION_KEYS):
        raise SystemExit("the base registration already carries an extension")
    current = registration(repeats=base["repeats"], case_ids=tuple(base["case_ids"]))
    for key, value in current.items():
        if key in {"implementation", "candidate_sha256"}:
            continue
        if base.get(key) != value:
            raise SystemExit(
                f"the base registration disagrees with the oracle at {key}"
            )
    chosen = tuple(
        case for case in dev_quality_cases() if case.case_id in base["case_ids"]
    )
    body = dict(base)
    body["truth_extension"] = {case.case_id: truth_extension(case) for case in chosen}
    body["binding"] = binding_block(chosen)
    body["extends"] = {
        "base_registration_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "extension_version": SWEEP_TRUTH_EXTENSION_VERSION,
    }
    projection = json.dumps(
        project_base(body), ensure_ascii=False, sort_keys=True, indent=2
    ).encode()
    if (
        hashlib.sha256(projection).hexdigest()
        != body["extends"]["base_registration_sha256"]
    ):
        raise SystemExit("the projected base does not reproduce the base file bytes")
    output_path.write_text(
        json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return {
        "written": str(output_path),
        "base_registration_sha256": body["extends"]["base_registration_sha256"],
        "extension_cases": len(chosen),
    }


def main() -> None:
    from .eval_f_validation import main_sweep_cli

    main_sweep_cli()


__all__ = [
    "DEV_SWEEP_VERSION",
    "DEFAULT_REPEATS",
    "EXTENSION_KEYS",
    "SWEEP_SUBJECT",
    "SWEEP_TENANT",
    "SWEEP_TRUTH_EXTENSION_VERSION",
    "binding_block",
    "dev_quality_seed_plan",
    "extend_registration",
    "implementation_digests",
    "mapped_truth",
    "project_base",
    "registration",
    "run_sweep",
    "seed",
]


if __name__ == "__main__":
    main()
