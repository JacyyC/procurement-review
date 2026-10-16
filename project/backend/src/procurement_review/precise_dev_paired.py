"""The DEV Oracle/actual-retrieval paired diagnostic (V-G-03).

V-G-01 measured what the product's own retrieval returns; this batch measures what that
retrieval costs the *decision*.  One pre-registered DEV subset is executed twice through
the same versioned 3.1.0 wire and the same checked attempt machinery:

* the **Oracle** arm receives the Case's own authored documents, as the frozen line's
  Oracle context did;
* the **actual** arm receives only what the product's scope-bound retrieval port returned
  for the branch's own question, over an isolated PostgreSQL registry that also holds a
  distractor library from unrelated generation families.

Both arms read their facts through the product's scope-bound read-only MCP path, and both
answers are grounded and finalized by the product's own 3.1.0 pipeline.  The grounding
runs inside the attempt, so a citation the source boundary refuses can be corrected by
the branch's own remaining attempt under the product's shared two-attempt budget.  Every conditional
duty is then judged against the case's independent truth, and a difference is attributed:
a required document the actual arm never retrieved is a *retrieval* loss, an answer that
could not be grounded over the retrieved material is an *assembly* loss, and a grounded
answer that still reports the wrong state is a *judgment* loss.  Nothing is backfilled
from the Oracle into the actual arm, and the actual arm's sources are never widened by one
line of the Oracle text.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .eval_f_contracts import FEndpointConfig
from .eval_attempt_ledger import LedgerError
from .eval_f_registration import configuration
from .eval_g_retrieval import (
    G_DATASET_VERSION,
    G_INDEX_VERSION,
    document_row,
)
from .precise_dependencies import DIAGNOSTIC_ROOTS, PRODUCT_ROOTS, package_closure
from .precise_run_scope import PRECISE_QUESTIONS, RETRIEVAL_LIMIT
from .gold_v37_conditional import (
    ConditionalCaseV1,
    as_gold_case,
    conditional_cases,
    derive_conditional_truth,
)
from .security_fixture import (
    SecuritySeedPlan,
    SeedCase,
    SeedDocument,
    SeedProcurementRecord,
    SeedSubject,
    SeedTenant,
)
from sqlalchemy import create_engine

#: The paired diagnostic contract.  v2 is the first version whose document branches
#: ground their answer inside the attempt and may spend their second attempt on a
#: citation refusal, so its readings come from the same grounding and the same shared
#: budget the product line uses; v1 ran the wire contract alone.
PAIRED_CONTRACT = "v1-07g-dev-paired.v2"
CORPUS_DSN_ENV = "V1_07G_PAIRED_CORPUS_DSN"
CORPUS_APPLICATION_NAME = "v1_07g_paired_corpus"

#: The pre-registered subset: the authored family's eight conditional shapes, one case
#: each — a sourced lease exclusion, a missing category, a category disagreement, an
#: unverified and a verified single-source exception, both threshold outcomes and a
#: superseded commitment.  The subset is the whole family because each case answers a
#: shape the others do not.
CASE_IDS = (
    "V107-D-9001",
    "V107-D-9002",
    "V107-D-9003",
    "V107-D-9004",
    "V107-D-9005",
    "V107-D-9006",
    "V107-D-9007",
    "V107-D-9008",
)
ARMS = ("oracle", "actual")
WORKERS = ("policy", "contract")

#: Two pre-registered corpus configurations, both reported with their own denominators.
#: ``own`` is the canonical DEV configuration: the Case's authorized material is its own
#: documents, so the pair measures the model, grounding and judgment link with retrieval
#: held constant.  ``library`` adds the distractor library the V-G-01 stress measurement
#: used, so the difference between the arms is the loss the product's own retrieval and
#: query carry — attributed per duty instead of being folded into a quality number.
CONFIGURATIONS = ("own", "library")
MAX_REQUESTS_PER_BRANCH = 2
MAX_REQUESTS = (
    len(CONFIGURATIONS)
    * len(CASE_IDS)
    * len(ARMS)
    * len(WORKERS)
    * MAX_REQUESTS_PER_BRANCH
)
DISTRACTOR_COUNT = 40

#: The G model identity this stage registers; a resolved identity other than this one
#: stops the batch instead of metering under the wrong model.
G_REQUESTED_MODEL = "deepseek-v4.1-flash"
G_RESOLVED_MODEL = "deepseek-v4-1-flash"

PAIR_TENANT = "tenant-v1-07g-paired"
PAIR_SUBJECT = "subject-v1-07g-paired"
DATASET_VERSION = G_DATASET_VERSION
INDEX_VERSION = G_INDEX_VERSION
RULES_VERSION = "3.1.0"

#: The implementation this batch depends on is not a list of its own: it is the product
#: execution closure plus the diagnostic closure of the authoritative inventory
#: (``precise_dependencies``), so a module split out of another one, or a newly imported
#: dependency, is inside the registration without anyone editing a list here.
PAIRED_IMPLEMENTATION_FILES = package_closure(PRODUCT_ROOTS + DIAGNOSTIC_ROOTS)


def paired_configuration() -> FEndpointConfig:
    """The stage's endpoint configuration: the G model under the fixed wire settings."""

    base = configuration().model_dump(mode="json")
    return FEndpointConfig(
        **{
            **base,
            "requested_model": G_REQUESTED_MODEL,
            "resolved_model": G_RESOLVED_MODEL,
            "single_attempt_seconds": 90.0,
        }
    )


def paired_implementation() -> dict[str, str]:
    """The digest of every implementation file and generated schema this batch depends on.

    The per-file digests come from the files themselves and the schema digests from the
    schema the live request builds, so a change in the prompt, the value registry or the
    visible schema moves the registration and the batch cannot be re-used under the old one.
    """

    from .precise_candidate import generated_digests

    from .precise_dependencies import module_path

    digests = {
        name: hashlib.sha256(module_path(name).read_bytes()).hexdigest()
        for name in PAIRED_IMPLEMENTATION_FILES
    }
    generated = generated_digests()
    digests["generated:wireschema"] = generated["wire_fingerprint"]
    digests["generated:system_prompt"] = generated["system_prompt_sha256"]
    digests["generated:answer_version"] = generated["answer_version"]
    for worker, entry in generated["workers"].items():
        digests[f"generated:schema:{worker}"] = entry["schema_sha256"]
        digests[f"generated:task:{worker}"] = entry["task_text_sha256"]
    return digests


def _id_map(case: ConditionalCaseV1) -> dict[str, str]:
    """The product-shaped identifiers one authored Case's records and text carry.

    The product's read-only tools accept stable synthetic record ids, so the documents
    that name the Case's order, supplier and transaction and the records the MCP boundary
    answers are mapped together: the text the model sees, the facts it is compared with
    and the role of each identifier stay the same material.
    """

    index = CASE_IDS.index(case.case_id) + 1
    facts = case.facts
    return {
        str(facts["purchase_order_id"]): f"PO-{900 + index}",
        str(facts["supplier_id"]): f"SUP-{900 + index}",
        str(facts["transaction_id"]): f"TXN-{900 + index}",
        str(facts["contract_reference"]): f"CONTRACT-{900 + index}",
        str(facts["counterparty_reference"]): f"TXN-{900 + index}",
    }


def _prepared_case(case: ConditionalCaseV1) -> ConditionalCaseV1:
    """The Case whose documents name its records the way the MCP boundary reads them."""

    replacements = _id_map(case)
    texts = {}
    for name, text_value in (
        ("policy_text", case.policy_text),
        ("contract_text", case.contract_text),
    ):
        for source, target in sorted(
            replacements.items(), key=lambda item: -len(item[0])
        ):
            text_value = text_value.replace(source, target)
        texts[name] = text_value
    return case.model_copy(update=texts)


def paired_cases() -> tuple[ConditionalCaseV1, ...]:
    chosen = {case.case_id: case for case in conditional_cases()}
    return tuple(_prepared_case(chosen[case_id]) for case_id in CASE_IDS)


def _order_id(case: ConditionalCaseV1) -> str:
    """The product-shaped order identifier this Case's MCP read asks for."""

    return _id_map(case)[str(case.facts["purchase_order_id"])]


def _document_version(case: ConditionalCaseV1) -> str:
    gold = as_gold_case(case)
    versions = {
        gold.model_input.policy_document.version,
        gold.model_input.contract_document.version,
    }
    if len(versions) != 1:
        raise LedgerError("the paired corpus registers exactly one material version")
    return next(iter(versions))


def registration() -> dict:
    """The frozen manifest: what is sent, over what material, and why."""

    cases = paired_cases()
    truths = {case.case_id: derive_conditional_truth(case) for case in cases}
    return {
        "contract": PAIRED_CONTRACT,
        "stage": "V1-07G",
        "revision": 2,
        "question": (
            "DEV 条件子集上，实际检索链与 Oracle 上下文各自的 3.1.0 判定质量如何，"
            "差异分别由检索、装配还是判断造成？"
        ),
        "information_gain": (
            "V-G-01 只测到检索回召；本批给出八种条件形状上的配对分母："
            "若 Oracle 正确而实际错误，按缺失必需来源（检索）、无法接地（装配）、"
            "接地但状态错误（判断）归因；两臂都对则该形状的真实链路不构成损失。"
        ),
        "protection_set": (
            "仅使用自撰合成材料（gold-set.v3.7 条件用例与其 DEV 干扰库），"
            "不含真实企业数据、凭据或个人信息；结果只保留有界投影与计量。"
        ),
        "case_ids": list(CASE_IDS),
        "arms": list(ARMS),
        "workers": list(WORKERS),
        "configurations": {
            "own": {
                "shared_documents": 0,
                "reading": "canonical DEV configuration: the Case's own authorized material",
            },
            "library": {
                "shared_documents": 2 * DISTRACTOR_COUNT,
                "reading": (
                    "V-G-01 stress configuration: the Case's own documents plus the "
                    "distractor library of unrelated generation families"
                ),
            },
        },
        "max_attempts_per_branch": MAX_REQUESTS_PER_BRANCH,
        "max_requests": MAX_REQUESTS,
        "distractor_count": DISTRACTOR_COUNT,
        "retrieval": {
            "query": PRECISE_QUESTIONS,
            "limit": RETRIEVAL_LIMIT,
            "vector_adapter": "no-vector-v1",
            "lexical_adapter": "deterministic-lexical-idf-v1",
            "corpus": "isolated PostgreSQL, the Case's own documents plus one distractor library",
            "fact_source": "the product's scope-bound read-only MCP path",
        },
        "stop_conditions": [
            "任一分支两次尝试仍无有效 envelope 即停止该分支，不重发",
            "认证、模型身份漂移或安全事件立即停止整个批次",
            "批次期限或请求上限到达即停止，已发请求照常结算",
        ],
        "configuration": paired_configuration().model_dump(mode="json"),
        "base_dataset": "gold-set.v3.7",
        "implementation": paired_implementation(),
        "case_inputs": [
            {
                "case_id": case.case_id,
                "policy_sha256": hashlib.sha256(case.policy_text.encode()).hexdigest(),
                "contract_sha256": hashlib.sha256(
                    case.contract_text.encode()
                ).hexdigest(),
                "truth": truths[case.case_id].model_dump(mode="json"),
            }
            for case in cases
        ],
    }


# ------------------------------------------------------------------- the corpus


def _database_url(base_dsn: str, database: str) -> str:
    head, _, _tail = base_dsn.partition("?")
    scheme_and_host, _, _old = head.rpartition("/")
    return f"{scheme_and_host}/{database}?application_name={CORPUS_APPLICATION_NAME}"


def _admin_engine(base_dsn: str):
    return create_engine(base_dsn, isolation_level="AUTOCOMMIT")


def corpus_procurement_rows(
    case: ConditionalCaseV1,
) -> tuple[SeedProcurementRecord, ...]:
    """The read-only fact rows one Case's MCP boundary answers, from its own material."""

    facts = case.facts
    mapping = _id_map(case)
    order_id = mapping[str(facts["purchase_order_id"])]
    supplier_id = mapping[str(facts["supplier_id"])]
    transaction_id = mapping[str(facts["transaction_id"])]
    contract_reference = mapping[str(facts["contract_reference"])]
    counterparty_reference = mapping[str(facts["counterparty_reference"])]
    order = {
        "purchase_order_id": order_id,
        "supplier_id": supplier_id,
        "competitive_quote_count": facts["competitive_quote_count"],
        "contract_reference": contract_reference,
        "ordered_on": facts["ordered_on"],
        "invoice_amount_minor": facts["invoice_amount_minor"],
        "invoice_currency": facts["invoice_currency"],
    }
    order.update(
        amount_minor=facts["order_amount_minor"], currency=facts["order_currency"]
    )
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
        "counterparty_reference": counterparty_reference,
        "booked_on": facts["booked_on"],
    }
    transaction.update(
        amount_minor=facts["transaction_amount_minor"],
        currency=facts["transaction_currency"],
    )
    return tuple(
        SeedProcurementRecord(
            tenant_id=PAIR_TENANT,
            case_id=case.case_id,
            kind=kind,
            record_id=record_id,
            dataset_version=DATASET_VERSION,
            as_of=as_gold_case(case).as_of.isoformat(),
            payload=payload,
        )
        for kind, record_id, payload in (
            ("order", order_id, order),
            ("supplier", supplier_id, supplier),
            ("transaction", transaction_id, transaction),
        )
    )


def corpus_plan(
    cases: tuple[ConditionalCaseV1, ...],
    distractors: tuple[Any, ...],
    *,
    with_library: bool = True,
):
    """One registry plan: each Case's documents, and (optionally) a distractor library."""

    cases_payload = []
    documents: list[SeedDocument] = []
    records: list[SeedProcurementRecord] = []
    shared_ids: list[str] = []
    for case in cases:
        version = _document_version(case)
        gold = as_gold_case(case)
        case_document_ids = []
        for domain, document in (
            ("policy", gold.model_input.policy_document),
            ("contract", gold.model_input.contract_document),
        ):
            document_id = f"DOC-{case.case_id}-{domain.upper()}"
            case_document_ids.append(document_id)
            documents.append(
                document_row(
                    source_id=document.source_id,
                    content=document.content,
                    document_id=document_id,
                    ownership="case_document",
                    tenant_id=PAIR_TENANT,
                    case_id=case.case_id,
                    category=domain,
                    document_version=version,
                )
            )
        facts = case.facts
        mapping = _id_map(case)
        order_id = mapping[str(facts["purchase_order_id"])]
        cases_payload.append(
            SeedCase(
                tenant_id=PAIR_TENANT,
                case_id=case.case_id,
                label=case.case_id,
                case_type="purchase_order",
                scenario="low",
                purchase_order_id=order_id,
                case_version=f"{case.case_id.lower()}.v1",
                document_version=version,
                dataset_version=DATASET_VERSION,
                index_version=INDEX_VERSION,
                rules_version=RULES_VERSION,
                case_document_ids=tuple(case_document_ids),
                shared_document_ids=(),
                order_ids=(order_id,),
                supplier_ids=(mapping[str(facts["supplier_id"])],),
                transaction_ids=(mapping[str(facts["transaction_id"])],),
                granted_subject_ids=(PAIR_SUBJECT,),
            )
        )
        records.extend(corpus_procurement_rows(case))
    for case in distractors if with_library else ():
        for domain, document in (
            ("policy", case.model_input.policy_document),
            ("contract", case.model_input.contract_document),
        ):
            document_id = f"DOC-LIB-{document.source_id}"
            shared_ids.append(document_id)
            documents.append(
                document_row(
                    source_id=document.source_id,
                    content=document.content,
                    document_id=document_id,
                    ownership="shared_fixture",
                    tenant_id=None,
                    case_id=None,
                    category=domain,
                    document_version=_document_version(cases[0]),
                )
            )
    shared = tuple(sorted(set(shared_ids))) if with_library else ()
    seeded = tuple(
        SeedCase(
            tenant_id=item.tenant_id,
            case_id=item.case_id,
            label=item.label,
            case_type=item.case_type,
            scenario=item.scenario,
            purchase_order_id=item.purchase_order_id,
            case_version=item.case_version,
            document_version=item.document_version,
            dataset_version=item.dataset_version,
            index_version=item.index_version,
            rules_version=item.rules_version,
            case_document_ids=item.case_document_ids,
            shared_document_ids=shared,
            order_ids=item.order_ids,
            supplier_ids=item.supplier_ids,
            transaction_ids=item.transaction_ids,
            granted_subject_ids=item.granted_subject_ids,
        )
        for item in cases_payload
    )
    return SecuritySeedPlan(
        fixture_version="v1-07g-dev-paired.v1",
        document_version=_document_version(cases[0]),
        dataset_version=DATASET_VERSION,
        index_version=INDEX_VERSION,
        tenants=(SeedTenant(PAIR_TENANT, "V1-07G paired diagnostic tenant"),),
        subjects=(
            SeedSubject(
                tenant_id=PAIR_TENANT,
                subject_id=PAIR_SUBJECT,
                display_name="V1-07G paired diagnostic operator",
                roles=("operator",),
                case_grants=tuple(case.case_id for case in cases),
            ),
        ),
        cases=seeded,
        documents=tuple(documents),
        records=tuple(records),
        retrieval_scores={},
    )


def distractor_documents() -> tuple[Any, ...]:
    """Unrelated generation families from the shipped DEV split, by their own registration."""

    from .eval_g_retrieval import corpus_cases

    _measured, distractors, _siblings, _truths = corpus_cases(
        "gold-set.v3.1", measured_count=1, distractor_count=DISTRACTOR_COUNT
    )
    return tuple(distractors)


__all__ = [
    "ARMS",
    "CASE_IDS",
    "CORPUS_DSN_ENV",
    "PAIRED_CONTRACT",
    "corpus_plan",
    "corpus_procurement_rows",
    "distractor_documents",
    "paired_cases",
    "paired_configuration",
    "paired_implementation",
    "registration",
]
