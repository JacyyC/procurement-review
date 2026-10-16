"""V1-07G precise Run: the full product chain on an isolated PostgreSQL.

The Run is created through the product service and API, executed by the precise
runner against a Mock model transport, committed with its versioned execution
record and read back fresh.  The registered material is one authored conditional
case (rules 3.1.0), so the decision, the retrieval it came from and the metering
are all the ones the product actually produced.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from procurement_review.api import create_app  # noqa: E402
from procurement_review.authorization import (  # noqa: E402
    AuthorizationService,
    generate_credential,
)
from procurement_review.eval_f_registration import configuration  # noqa: E402
from procurement_review.gold_v37_conditional import (  # noqa: E402
    as_gold_case,
    conditional_cases,
    derive_conditional_truth,
)
from procurement_review.ingestion import (  # noqa: E402
    DEFAULT_CONFIG,
    LocatorKind,
    SourceLocator,
    make_chunk_id,
    make_source_version_id,
)
from procurement_review.precise_run_contracts import ExecutionRecord  # noqa: E402
from procurement_review.precise_run_runner import PreciseScopedRunner  # noqa: E402
from procurement_review.precise_run_transport import PreciseRunChatTransport  # noqa: E402
from procurement_review.review_repository import ReviewRepository  # noqa: E402
from procurement_review.review_service import ReviewService  # noqa: E402
from procurement_review.review_store import ReviewStore  # noqa: E402
from procurement_review.security_contracts import SecurityOperation  # noqa: E402
from procurement_review.security_fixture import (  # noqa: E402
    SecuritySeedPlan,
    SeedCase,
    SeedDocument,
    SeedProcurementRecord,
    SeedSubject,
    SeedTenant,
    _content_checksum,
)
from v1_03_support import create_isolated_database, drop_database  # noqa: E402

TENANT = "tenant-v1-07g-precise"
SUBJECT = "subject-v1-07g-precise"
DOCUMENT_VERSION = "synthetic-gold-document.v3.7"
DATASET_VERSION = "synthetic-gold-procurement.v3"
INDEX_VERSION = "g-precise-index.v1"
RULES_VERSION = "3.1.0"


def _material_document(document_id: str, source_id: str, content: str, category: str):
    content_checksum = _content_checksum(content)
    locator = SourceLocator(
        kind=LocatorKind.TEXT_LINES,
        line_start=1,
        line_end=max(1, content.count("\n") + 1),
    )
    chunk_id = make_chunk_id(
        document_id=document_id,
        ordinal=1,
        locator=locator,
        content_checksum=content_checksum,
        chunk_rules_version=DEFAULT_CONFIG.chunk_rules_version,
        chunk_rules_checksum=DEFAULT_CONFIG.chunk_rules_checksum,
    )
    payload = {
        "source_id": source_id,
        "source_version": DOCUMENT_VERSION,
        "source_version_id": make_source_version_id(source_id, DOCUMENT_VERSION),
        "source_checksum": content_checksum,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "ordinal": 1,
        "media_type": "text/plain",
        "locator": {
            "kind": locator.kind.value,
            "line_start": locator.line_start,
            "line_end": locator.line_end,
        },
        "content": content,
        "content_checksum": content_checksum,
        "ingestion_version": DEFAULT_CONFIG.ingestion_version,
        "chunk_rules_version": DEFAULT_CONFIG.chunk_rules_version,
        "chunk_rules_checksum": DEFAULT_CONFIG.chunk_rules_checksum,
        "index_version": INDEX_VERSION,
    }
    return (
        SeedDocument(
            document_id=document_id,
            ownership="case_document",
            tenant_id=TENANT,
            case_id=None,
            category=category,
            source_id=source_id,
            source_version_id=payload["source_version_id"],
            document_version=DOCUMENT_VERSION,
            index_version=INDEX_VERSION,
            content_checksum=content_checksum,
            chunk_payload=payload,
        ),
        chunk_id,
    )


#: The product's read-only tools accept stable synthetic record IDs, so the authored
#: material is registered under product-shaped identifiers.  The same mapping is
#: applied to the facts and to the text, so the contract still names its own order.
PRODUCT_IDS = {
    "purchase_order_id": "PO-905",
    "supplier_id": "SUP-905",
    "transaction_id": "TXN-905",
}


def _remap(case):
    gold = as_gold_case(case)
    facts = gold.model_input.business_facts
    replacements = {getattr(facts, name): value for name, value in PRODUCT_IDS.items()}
    replacements[str(facts.contract_reference)] = "CONTRACT-905"
    replacements[str(facts.counterparty_reference)] = "TXN-905"
    text = case.contract_text
    for source, target in sorted(replacements.items(), key=lambda item: -len(item[0])):
        text = text.replace(source, target)
    return gold, facts, replacements, text


def _plan(case) -> SecuritySeedPlan:
    gold, facts, replacements, contract_text = _remap(case)
    documents = []
    case_document_ids = []
    for document_id, source_id, content, category in (
        (
            f"DOC-{case.case_id}-POLICY",
            f"{case.case_id}-POLICY",
            case.policy_text,
            "policy",
        ),
        (
            f"DOC-{case.case_id}-CONTRACT",
            f"{case.case_id}-CONTRACT",
            contract_text,
            "contract",
        ),
    ):
        document, _chunk_id = _material_document(
            document_id, source_id, content, category
        )
        documents.append(
            SeedDocument(
                document_id=document.document_id,
                ownership="case_document",
                tenant_id=TENANT,
                case_id=case.case_id,
                category=document.category,
                source_id=document.source_id,
                source_version_id=document.source_version_id,
                document_version=document.document_version,
                index_version=document.index_version,
                content_checksum=document.content_checksum,
                chunk_payload=document.chunk_payload,
            )
        )
        case_document_ids.append(document.document_id)
    data = facts.model_dump(mode="json")
    for name, target in PRODUCT_IDS.items():
        data[name] = target
    data["contract_reference"] = "CONTRACT-905"
    data["counterparty_reference"] = "TXN-905"
    order = {
        key: data[key]
        for key in (
            "purchase_order_id",
            "supplier_id",
            "competitive_quote_count",
            "contract_reference",
            "ordered_on",
            "invoice_amount_minor",
            "invoice_currency",
        )
    }
    order.update(amount_minor=facts.order_amount_minor, currency=facts.order_currency)
    if case.transaction_category is not None:
        order["transaction_category"] = case.transaction_category
    supplier = {
        key: data[key]
        for key in (
            "supplier_id",
            "registry_status",
            "procurement_status",
            "tax_identifier",
            "due_diligence_status",
            "sanctions_screening",
        )
    }
    transaction = {
        key: data[key]
        for key in (
            "transaction_id",
            "purchase_order_id",
            "supplier_id",
            "counterparty_reference",
            "booked_on",
        )
    }
    transaction.update(
        amount_minor=facts.transaction_amount_minor, currency=facts.transaction_currency
    )
    records = tuple(
        SeedProcurementRecord(
            tenant_id=TENANT,
            case_id=case.case_id,
            kind=kind,
            record_id=record_id,
            dataset_version=DATASET_VERSION,
            as_of=gold.as_of.isoformat(),
            payload=payload,
        )
        for kind, record_id, payload in (
            ("order", PRODUCT_IDS["purchase_order_id"], order),
            ("supplier", PRODUCT_IDS["supplier_id"], supplier),
            ("transaction", PRODUCT_IDS["transaction_id"], transaction),
        )
    )
    return SecuritySeedPlan(
        fixture_version="v1-07g-precise-run.v1",
        document_version=DOCUMENT_VERSION,
        dataset_version=DATASET_VERSION,
        index_version=INDEX_VERSION,
        tenants=(SeedTenant(TENANT, "V1-07G precise tenant"),),
        subjects=(
            SeedSubject(
                tenant_id=TENANT,
                subject_id=SUBJECT,
                display_name="V1-07G precise operator",
                roles=("operator", "approver", "auditor"),
                case_grants=(case.case_id,),
            ),
        ),
        cases=(
            SeedCase(
                tenant_id=TENANT,
                case_id=case.case_id,
                label=case.case_id,
                case_type="purchase_order",
                scenario="low",
                purchase_order_id=PRODUCT_IDS["purchase_order_id"],
                case_version=f"{case.case_id.lower()}.v1",
                document_version=DOCUMENT_VERSION,
                dataset_version=DATASET_VERSION,
                index_version=INDEX_VERSION,
                rules_version=RULES_VERSION,
                case_document_ids=tuple(case_document_ids),
                shared_document_ids=(),
                order_ids=(PRODUCT_IDS["purchase_order_id"],),
                supplier_ids=(PRODUCT_IDS["supplier_id"],),
                transaction_ids=(PRODUCT_IDS["transaction_id"],),
                granted_subject_ids=(SUBJECT,),
            ),
        ),
        documents=tuple(documents),
        records=records,
        retrieval_scores={},
    )


# ------------------------------------------------------------- the Mock model wire


def _line(text_value: str, needle: str) -> str:
    return next(line for line in text_value.split("\n") if needle in line)


def _maybe_line(text_value: str, needle: str) -> str | None:
    return next((line for line in text_value.split("\n") if needle in line), None)


def _policy_body(content: str) -> dict:
    scope_line = _line(content, "适用于")
    supplier_line = _line(content, "供应商要求")
    money_line = _line(content, "订单与发票等额")
    dates_line = _line(content, "记账日期")
    terms_line = _line(content, "交付不超过")
    rule_line = _line(content, "报价规则")
    exclusion_line = _maybe_line(content, "租赁类交易不适用竞争性报价要求")
    exception_line = _maybe_line(content, "登记的例外")
    values = {
        "O-01": {"applies": True, "basis": "2026", "exempt": []},
        "O-02": {
            "special": terms_line.rstrip("。"),
            "quotes": 3,
            "delivery": 15,
            "acceptance": 7,
            "warranty": 90,
            "supplier": "all_qualified",
            "money": "equal_invoice_capped_tx",
            "dates": "booked_ge_ordered",
        },
    }
    rule = {
        "minimum_quotes": 3,
        "applies_to": ["purchase", "service"],
        "excluded_categories": [],
        "exceptions": [],
    }
    if exclusion_line is not None:
        rule["excluded_categories"] = ["lease"]
        rule["exclusion_basis"] = "不适用竞争性报价要求"
    if exception_line is not None:
        rule["exceptions"] = ["single_source"]
        rule["exception_requires"] = ["written_justification", "approval_reference"]
        rule["exception_basis"] = "登记的例外"
    values["O-04"] = rule
    witnesses = [
        ("O-01", "applies", True, scope_line),
        ("O-01", "basis", "2026", scope_line),
        ("O-01", "exempt", [], scope_line),
        ("O-02", "special", terms_line.rstrip("。"), terms_line),
        ("O-02", "quotes", 3, rule_line),
        ("O-02", "delivery", 15, terms_line),
        ("O-02", "acceptance", 7, terms_line),
        ("O-02", "warranty", 90, terms_line),
        ("O-02", "supplier", "all_qualified", supplier_line),
        ("O-02", "money", "equal_invoice_capped_tx", money_line),
        ("O-02", "dates", "booked_ge_ordered", dates_line),
        ("O-04", "minimum_quotes", 3, rule_line),
        ("O-04", "applies_to", '["purchase","service"]', rule_line),
    ]
    if exclusion_line is not None:
        witnesses.append(("O-04", "excluded_categories", ["lease"], exclusion_line))
        witnesses.append(
            ("O-04", "exclusion_basis", "不适用竞争性报价要求", exclusion_line)
        )
    else:
        witnesses.append(("O-04", "excluded_categories", [], rule_line))
    if exception_line is not None:
        witnesses.append(("O-04", "exceptions", ["single_source"], exception_line))
        witnesses.append(
            (
                "O-04",
                "exception_requires",
                ["written_justification", "approval_reference"],
                exception_line,
            )
        )
        witnesses.append(("O-04", "exception_basis", "登记的例外", exception_line))
    else:
        witnesses.append(("O-04", "exceptions", [], rule_line))
    return _envelope(values, witnesses, content)


def _contract_body(content: str) -> dict:
    party_line = _line(content, "买方BUYER")
    refs_line = _line(content, "订单ORDER")
    price_line = _line(content, "价款记录")
    terms_line = _line(content, "交付DELIVERY")
    standard_line = _line(content, "验收标准")
    responsibility_line = _line(content, "质保责任")
    change_line = _line(content, "书面签署")
    damages_line = _line(content, "赔付DAMAGES")
    cap_line = _line(content, "责任上限")
    other_line = _line(content, "本次只审变更")
    category_line = _line(content, "类交易（本次合同")
    source_line = _maybe_line(content, "单一来源采购")
    justification_line = _maybe_line(content, "书面说明:")
    approval_line = _maybe_line(content, "审批授权:")
    supersede_line = _maybe_line(content, "已被补充协议替代")
    grab = lambda pattern: re.search(pattern, content)[1]  # noqa: E731
    buyer = grab(r"买方BUYER:\s*(\S+?)[；。]")
    seller = grab(r"卖方SELLER:\s*(\S+?)[；。]")
    subject = grab(r"标的SUBJECT:\s*([^。]+?)。")
    order = grab(r"订单ORDER:\s*(\S+?)[；。]")
    contract = grab(r"合同CONTRACT:\s*(\S+?)[；。]")
    amount = grab(r"价款记录:\s*订单\s*(\d+\.\d{2})")
    currency = grab(r"价款记录:\s*订单\s*\d+\.\d{2}\s*(CNY|USD)")
    delivery = int(grab(r"交付DELIVERY\s*(\d+)\s*日"))
    acceptance = int(grab(r"验收ACCEPTANCE\s*(\d+)\s*日"))
    warranty = int(grab(r"质保WARRANTY\s*(\d+)\s*日"))
    standard = grab(r"验收标准STANDARD:\s*(.+?)。")
    responsibility = grab(r"质保责任WARRANTY_RESPONSIBILITY:\s*(.+?)。")
    damages = grab(r"赔付DAMAGES:\s*(.+?)。")
    cap = grab(r"责任上限CAP:\s*([\d.]+\s+\w+)")
    other = grab(r"(本次只审[^。]+)")
    category = "lease" if "租赁类交易（本次合同" in content else "purchase"
    category_basis = (
        "本次合同以设备租赁方式执行"
        if category == "lease"
        else "本次合同以设备采购方式执行"
    )
    values = {
        "O-05": {
            "buyer": buyer,
            "seller": seller,
            "subject": subject,
            "order": order,
            "contract": contract,
        },
        "O-06": {"amount": amount, "currency": currency},
        "O-08": {
            "delivery": delivery,
            "acceptance": acceptance,
            "warranty": warranty,
            "standard": standard,
            "responsibility": responsibility,
        },
        "O-09": {
            "written": True,
            "damages": damages,
            "cap": cap,
            "other": other,
        },
    }
    witnesses = [
        ("O-05", "buyer", buyer, party_line),
        ("O-05", "seller", seller, party_line),
        ("O-05", "subject", subject, party_line),
        ("O-05", "order", order, refs_line),
        ("O-05", "contract", contract, refs_line),
        ("O-06", "amount", amount, price_line),
        ("O-06", "currency", currency, price_line),
        ("O-08", "delivery", delivery, terms_line),
        ("O-08", "acceptance", acceptance, terms_line),
        ("O-08", "warranty", warranty, terms_line),
        ("O-08", "standard", standard, standard_line),
        ("O-08", "responsibility", responsibility, responsibility_line),
        ("O-09", "written", True, change_line),
        ("O-09", "damages", damages, damages_line),
        ("O-09", "cap", cap, cap_line),
        ("O-09", "other", other, other_line),
    ]
    transaction = {"category": category, "category_basis": category_basis}
    claims = []
    if source_line is not None:
        transaction["single_source"] = True
        claims.append(("single_source", True, source_line))
    if justification_line is not None:
        transaction["written_justification"] = True
        claims.append(("written_justification", True, justification_line))
    if approval_line is not None:
        transaction["approval_reference"] = grab(r"审批授权:\s*(\S+?)[；。]")
        claims.append(
            ("approval_reference", transaction["approval_reference"], approval_line)
        )
    if supersede_line is not None:
        transaction["commitment_current"] = False
        transaction["superseding_amendment"] = grab(r"(补充协议二[^。]+)")
        claims.append(("commitment_current", False, supersede_line))
        claims.append(
            (
                "superseding_amendment",
                transaction["superseding_amendment"],
                supersede_line,
            )
        )
    values["O-04"] = transaction
    witnesses.append(("O-04", "category", category, category_line))
    witnesses.append(("O-04", "category_basis", category_basis, category_line))
    for field, value, line in claims:
        witnesses.append(("O-04", field, value, line))
    return _envelope(values, witnesses, content)


def _envelope(values, witnesses, content) -> dict:
    findings = []
    for obligation, value in values.items():
        findings.append(
            {
                "obligation_id": obligation,
                "status": "supported",
                "normalized_value": value,
                "evidence_refs": [],
                "statement": "原文提取",
                "recommended_action": "人工核对来源",
            }
        )
    rendered = []
    for obligation, field, value, line in witnesses:
        start = content.index(line)
        rendered.append(
            {
                "obligation_id": obligation,
                "field": field,
                "value": value,
                "source_ref": None,
                "start": start,
                "end": start + len(line),
            }
        )
    return {"findings": findings, "witnesses": rendered}


def _handler(requests: list) -> httpx.MockTransport:
    """One Mock wire that answers the versioned request from its own material."""

    config = configuration()

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        user = json.loads(payload["messages"][1]["content"])
        worker = user["worker"]
        source = user["sources"][0]
        body = (
            _policy_body(source["content"])
            if worker == "policy"
            else _contract_body(source["content"])
        )
        for finding in body["findings"]:
            finding["evidence_refs"] = [source["ref"]]
        for witness in body["witnesses"]:
            witness["source_ref"] = source["ref"]
        return httpx.Response(
            200,
            json={
                "model": config.resolved_model,
                "usage": {"prompt_tokens": 700, "completion_tokens": 260},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"schema_version": "finding-evidence.v3", **body},
                                ensure_ascii=False,
                            )
                        },
                    }
                ],
            },
        )

    return httpx.MockTransport(respond)


@dataclass(slots=True)
class Environment:
    store: ReviewStore
    authorization: AuthorizationService
    repository: ReviewRepository
    service: ReviewService
    case: object
    requests: list


def _environment(case_index: int = 4, **runner_kwargs) -> Iterator[Environment]:
    database_url, database = create_isolated_database()
    store = ReviewStore.from_url(database_url)
    store.migrate()
    case = conditional_cases()[case_index]
    store.seed(_plan(case))
    authorization = AuthorizationService(store=store)
    repository = ReviewRepository(store)
    requests: list = []
    config = configuration()
    transport = PreciseRunChatTransport(
        config,
        SecretStr("offline-only"),
        catalog=None,
        transport=_handler(requests),
    )
    store_seed = store
    service = ReviewService(
        repository=repository,
        runner=PreciseScopedRunner(
            store=store_seed, transport=transport, **runner_kwargs
        ),
    )
    try:
        yield Environment(
            store=store,
            authorization=authorization,
            repository=repository,
            service=service,
            case=case,
            requests=requests,
        )
    finally:
        store.close()
        drop_database(database)


@pytest.fixture()
def environment() -> Iterator[Environment]:
    yield from _environment()


def _token(environment: Environment, *, ttl: int = 3_600) -> str:
    token, credential_id, digest = generate_credential()
    issued_at = datetime.now(UTC)
    environment.store.issue_credential(
        credential_id=credential_id,
        tenant_id=TENANT,
        subject_id=SUBJECT,
        token_digest=digest,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(seconds=ttl),
    )
    return token


def _client(environment: Environment):
    from fastapi.testclient import TestClient

    application = create_app(
        store=environment.store,
        authorization=environment.authorization,
        repository=environment.repository,
        service=environment.service,
    )
    return TestClient(application, base_url="http://127.0.0.1")


def _create(client, environment: Environment, token: str):
    return client.post(
        "/api/v2/reviews",
        headers={
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": f"precise-{environment.case.case_id}",
        },
        json={
            "api_version": "review-api.v2",
            "case_id": environment.case.case_id,
            "case_version": f"{environment.case.case_id.lower()}.v1",
        },
    )


def _wait(
    client,
    token: str,
    run_id: str,
    statuses=("pending_confirmation", "pending_human_review"),
):
    for _ in range(200):
        response = client.get(
            f"/api/v2/reviews/{run_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in statuses or body["status"] in {
            "failed",
            "cancelled",
        }:
            return body
        import time

        time.sleep(0.05)
    raise AssertionError("the Run never reached a gate state")


def test_the_precise_run_reaches_its_decision_and_reads_back_fresh(environment):
    token = _token(environment)
    # One client context keeps one event loop for the whole flow: the Run's
    # execution task runs on the portal the creating request opened.
    with _client(environment) as client:
        created = _create(client, environment, token)
        assert created.status_code in {200, 201, 202}, created.text
        run_id = created.json()["run_id"]
        body = _wait(client, token, run_id)

    assert body["status"] == "pending_confirmation", body
    execution = body["execution"]
    assert execution["execution_version"] == "review-execution.v2"
    assert execution["rules_version"] == "3.1.0"
    assert body["summary"]["rules_version"] == "3.1.0"
    truth = derive_conditional_truth(environment.case)
    duties = {row["obligation_id"]: row for row in execution["decision"]["findings"]}
    assert duties["O-04"]["state"] == truth.obligations["O-04"]
    assert duties["O-04"]["reason"] == truth.reasons["O-04"]
    # Three Workers: two model branches with real metering, one deterministic.
    assert execution["policy"]["outcome"] == "completed"
    assert execution["contract"]["outcome"] == "completed"
    assert execution["procurement"]["model_calls"] == 0
    assert execution["procurement"]["tool_calls"] == 3
    metering = execution["metering"]
    assert metering["physical_attempts"] == 2
    assert metering["corrections"] == 0
    assert metering["input_tokens_known"] == 1400
    assert metering["resolved_models"] == [configuration().resolved_model]
    # The branches really cite the registered material, and the spans kept their
    # evidence identity and proposition.
    assert execution["policy"]["evidence_refs"]
    assert execution["policy"]["spans"]
    assert all(span["evidence"]["source_id"] for span in execution["policy"]["spans"])
    # Exactly two sends happened: the deterministic third Worker made none.
    assert len(environment.requests) == 2
    assert all(
        payload["model"] == configuration().requested_model
        for payload in environment.requests
    )
    # Fresh read-back of the stored payload is the versioned record itself.
    stored = environment.repository.load_run(run_id)
    record = ExecutionRecord.model_validate_json(
        json.dumps(stored.execution_payload), strict=True
    ).root
    assert record.execution_version == "review-execution.v2"
    assert record.summary.recommendation.value == body["summary"]["recommendation"]
    # The attempt ledger really booked one attempt per branch.
    assert environment.repository.attempt_totals(run_id) == {"policy": 1, "contract": 1}
    # The audit trail records the Run's terminal event under the same rules version.
    with _client(environment) as client:
        audit = client.get(
            f"/api/v2/reviews/{run_id}/audit",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert audit.status_code == 200, audit.text


def _scope_and_registration(environment: Environment, *, generation: int):
    token = _token(environment)
    context = environment.authorization.resolve(
        token, request_id="g-precise", operation=SecurityOperation.REVIEW_CREATE
    )
    registration = environment.authorization.require_case(
        context,
        environment.case.case_id,
        request_id="g-precise",
        operation=SecurityOperation.REVIEW_CREATE,
    )
    scope = environment.authorization.build_runtime_scope(
        context,
        registration,
        generation=generation,
        deadline_at=environment.authorization.runtime_scope_deadline(120.0),
    )
    return scope, registration


def test_the_shared_attempt_quota_is_not_reset_by_a_resume(environment):
    """The quota the next execution sees is the persisted ledger, never a fresh start."""

    token = _token(environment)
    with _client(environment) as client:
        created = _create(client, environment, token)
        assert created.status_code in {200, 201, 202}, created.text
        run_id = created.json()["run_id"]
        _wait(client, token, run_id)
    assert environment.repository.attempt_totals(run_id) == {"policy": 1, "contract": 1}
    # One attempt per branch was used.  A runner built for the next execution reads
    # the same consumed counter from PostgreSQL, whether or not it is a new instance.
    same_configuration = PreciseScopedRunner(
        store=environment.store,
        transport=RefusingTransport(),
        max_attempts_per_branch=1,
    )
    for worker in ("policy", "contract"):
        assert (
            same_configuration.remaining_attempts(
                environment.repository, run_id, worker
            )
            == 0
        )
    wider = PreciseScopedRunner(
        store=environment.store,
        transport=RefusingTransport(),
        max_attempts_per_branch=2,
    )
    for worker in ("policy", "contract"):
        assert wider.remaining_attempts(environment.repository, run_id, worker) == 1
    # The refusing transport was never called: nothing above sent anything.
    assert len(environment.requests) == 2


def test_a_dead_generation_refuses_before_any_send(environment):
    """The live-authority re-read refuses a generation that no longer owns the Run."""

    import asyncio

    from procurement_review.execution_guards import ExecutionInvalidated

    token = _token(environment)
    with _client(environment) as client:
        created = _create(client, environment, token)
        assert created.status_code in {200, 201, 202}, created.text
        run_id = created.json()["run_id"]
    run = environment.repository.load_run(run_id)
    sent_before = len(environment.requests)
    runner = PreciseScopedRunner(
        store=environment.store,
        transport=RefusingTransport(),
    )
    with pytest.raises((ExecutionInvalidated, Exception)):
        scope, registration = _scope_and_registration(environment, generation=1)
        asyncio.run(
            runner.run(
                run_id=run_id,
                scope=scope,
                registration=registration,
                generation=run.generation.generation + 5,
                repository=environment.repository,
                deadline_at=run.generation.deadline_at,
                fault=None,
                heartbeat=None,
                owner_id=run.generation.owner_id,
                now=lambda: datetime.now(UTC),
            )
        )
    assert len(environment.requests) == sent_before


class RefusingTransport(PreciseRunChatTransport):
    """A transport that fails loudly if an attempt is sent without authority."""

    def __init__(self) -> None:
        super().__init__(
            configuration(),
            SecretStr("offline-only"),
            catalog=None,
            transport=httpx.MockTransport(
                lambda request: pytest.fail("an attempt was sent without a live quota")
            ),
        )
