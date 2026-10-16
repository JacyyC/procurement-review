"""V1-03 SEC matrix: tenant/Case isolation, retrieval pre-filter and MCP scope.

These tests exercise the trusted runtime directly as well as through the API, because
SEC-09/10/11 must prove that bypassing the Review API still fails closed.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from mcp import Client
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from procurement_review.authorization import (
    generate_credential,
    verify_scope_authorization,
)
from procurement_review.contracts import (
    BusinessFact,
    EvidenceRef,
    FactStatus,
    SourceKind,
)
from procurement_review.mcp_server import (
    GET_PURCHASE_ORDER_TOOL,
    GET_SUPPLIER_TOOL,
    LIST_HISTORICAL_TRANSACTIONS_TOOL,
    READ_ONLY_TOOL_NAMES,
    ProcurementToolOutcome,
    create_procurement_mcp_server,
)
from procurement_review.mcp_startup import (
    MCP_STARTUP_VERSION,
    STARTUP_FILE_ENV,
    McpStartupError,
    load_startup_context,
    load_stdio_startup_context,
)
from procurement_review.procurement_mcp_client import OfficialProcurementMCPClient
from procurement_review.retrieval import RetrievalRequest, RetrievalStatus
from procurement_review.scoped_runtime import (
    ScopeIntegrityError,
    ingested_chunk_from_registry,
    ScopeTelemetry,
    ScopedEvidenceIndex,
    ScopedProcurementReader,
    ScopedRetrievalPort,
    ScopedReviewRuntime,
)
from procurement_review.procurement_store import ProcurementScopeError
from procurement_review.security_contracts import SecurityError, SecurityOperation
from procurement_review.worker_contracts import ProcurementWorkerInput

from v1_03_support import (
    ALPHA_OPERATOR,
    ALPHA_TENANT,
    BETA_OPERATOR,
    BETA_TENANT,
    DEMO_SUBJECT,
    SecurityHarness,
    wait_for_status,
)

ALPHA_CASE_ONE = "CASE-ALPHA-001"
ALPHA_CASE_TWO = "CASE-ALPHA-002"
BETA_CASE_TWO = "CASE-BETA-002"
DEMO_CASE = "CASE-LOW-001"


def _case_chunk_id(
    harness: SecurityHarness, *, tenant_id: str, case_id: str, document_id: str
) -> str:
    """Return the derived chunk ID of one registered Case document."""

    with harness.store.transaction() as connection:
        return str(
            connection.execute(
                text(
                    "SELECT chunk_payload ->> 'chunk_id' FROM review_document_registry "
                    "WHERE ownership = 'case_document' AND tenant_id = :t "
                    "AND case_id = :c AND document_id = :d"
                ),
                {"t": tenant_id, "c": case_id, "d": document_id},
            ).scalar_one()
        )


def _scope_for(
    harness: SecurityHarness,
    *,
    subject_id: str,
    tenant_id: str,
    case_id: str,
):
    token = harness.issue(subject_id, tenant_id=tenant_id)
    context = harness.authorization.resolve(
        token,
        request_id="test-scope",
        operation=SecurityOperation.REVIEW_CREATE,
    )
    registration = harness.authorization.require_case(
        context,
        case_id,
        request_id="test-scope",
        operation=SecurityOperation.REVIEW_CREATE,
    )
    scope = harness.authorization.build_runtime_scope(
        context,
        registration,
        generation=1,
        deadline_at=harness.authorization.runtime_scope_deadline(30.0),
    )
    return context, registration, scope


# --------------------------------------------------------------------- SEC-04/05


def test_two_tenants_with_colliding_ids_never_see_each_other(fresh_harness) -> None:
    """SEC-04: same Case/document/chunk/order/supplier IDs, different payloads."""

    _, alpha_registration, alpha_scope = _scope_for(
        fresh_harness,
        subject_id=ALPHA_OPERATOR,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_ONE,
    )
    _, beta_registration, beta_scope = _scope_for(
        fresh_harness,
        subject_id=BETA_OPERATOR,
        tenant_id=BETA_TENANT,
        case_id=ALPHA_CASE_ONE,
    )
    assert alpha_registration.case_id == beta_registration.case_id
    assert alpha_registration.checksum != beta_registration.checksum

    alpha_runtime = ScopedReviewRuntime.build(
        store=fresh_harness.store, scope=alpha_scope
    )
    beta_runtime = ScopedReviewRuntime.build(
        store=fresh_harness.store, scope=beta_scope
    )

    alpha_order = alpha_runtime.procurement.get_purchase_order(
        "PO-901", alpha_scope.dataset_version
    )
    beta_order = beta_runtime.procurement.get_purchase_order(
        "PO-901", beta_scope.dataset_version
    )
    assert alpha_order is not None and beta_order is not None
    assert alpha_order.record_id == beta_order.record_id == "PO-901"
    assert alpha_order.payload["marker"] == "ALPHA-ORDER-MARKER"
    assert beta_order.payload["marker"] == "BETA-ORDER-MARKER"

    alpha_supplier = alpha_runtime.procurement.get_supplier(
        "SUP-901", alpha_scope.dataset_version
    )
    beta_supplier = beta_runtime.procurement.get_supplier(
        "SUP-901", beta_scope.dataset_version
    )
    assert alpha_supplier is not None and beta_supplier is not None
    assert alpha_supplier.payload["marker"] == "ALPHA-SUPPLIER-MARKER"
    assert beta_supplier.payload["marker"] == "BETA-SUPPLIER-MARKER"

    for runtime, scope, marker in (
        (alpha_runtime, alpha_scope, "ALPHA-CASE-ONE"),
        (beta_runtime, beta_scope, "BETA-CASE-ONE"),
    ):
        response = runtime.retrieval.retrieve(
            RetrievalRequest(
                query="competitive bidding threshold",
                index_version=scope.index_version,
                limit=10,
                minimum_results=1,
            )
        )
        assert response.status is RetrievalStatus.KNOWN
        contents = " ".join(hit.content for hit in response.results)
        assert marker in contents
        other = "BETA-CASE-ONE" if marker.startswith("ALPHA") else "ALPHA-CASE-ONE"
        assert other not in contents


def test_candidates_are_narrowed_before_ranking(fresh_harness) -> None:
    """SEC-05: an out-of-scope document that would rank higher must not participate."""

    _, _, beta_scope = _scope_for(
        fresh_harness,
        subject_id=BETA_OPERATOR,
        tenant_id=BETA_TENANT,
        case_id=BETA_CASE_TWO,
    )
    alpha_two_chunk = _case_chunk_id(
        fresh_harness,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_TWO,
        document_id="DOC-ALPHA-002",
    )
    beta_chunk = _case_chunk_id(
        fresh_harness,
        tenant_id=BETA_TENANT,
        case_id=BETA_CASE_TWO,
        document_id="DOC-BETA-002",
    )
    assert alpha_two_chunk != beta_chunk
    port = ScopedRetrievalPort.from_store(store=fresh_harness.store, scope=beta_scope)
    response = port.retrieve(
        RetrievalRequest(
            query="payment terms termination rights",
            index_version=beta_scope.index_version,
            limit=10,
            minimum_results=1,
        )
    )
    assert response.status is RetrievalStatus.KNOWN
    returned = [hit.chunk_id for hit in response.results]
    assert beta_chunk in returned
    # The other tenant's contract document is declared at 0.72, above the beta Case
    # document at 0.66; a post-filter would have consumed a rank before dropping it.
    # FIX-07 removed the cross-scope scan this test used to assert on: the port no
    # longer reads, counts or exposes any out-of-scope chunk id, so the "would have
    # outranked" proof is the test-management read below, performed *before* the
    # restricted port is built.  See `test_fix07_retrieval_isolation.py`.
    assert alpha_two_chunk not in returned
    scores = {hit.chunk_id: hit.vector_score for hit in response.results}
    assert scores[beta_chunk] < 0.72
    assert all(chunk_id in port.allowed_chunk_ids for chunk_id in returned)
    assert not any(chunk_id == alpha_two_chunk for chunk_id in port.allowed_chunk_ids)
    expected_order = sorted(returned, key=lambda item: (-scores[item], item))
    assert returned == expected_order
    unscoped = fresh_harness.store.retrieval_scores(
        document_version=beta_scope.document_version,
        index_version=beta_scope.index_version,
        chunk_ids=[alpha_two_chunk, beta_chunk],
    )["payment terms termination rights"]
    assert unscoped[alpha_two_chunk] > unscoped[beta_chunk]


def test_shared_reference_documents_are_explicitly_registered(fresh_harness) -> None:
    """SEC-05: shared documents are read-only fixtures, never Case documents."""

    _, registration, scope = _scope_for(
        fresh_harness,
        subject_id=ALPHA_OPERATOR,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_ONE,
    )
    assert registration.case_document_ids
    assert registration.shared_document_ids
    assert not set(registration.case_document_ids) & set(
        registration.shared_document_ids
    )
    documents = fresh_harness.store.load_documents(
        case_document_ids=scope.case_document_ids,
        shared_document_ids=scope.shared_document_ids,
        tenant_id=scope.tenant_id,
        case_id=scope.case_id,
        document_version=scope.document_version,
        index_version=scope.index_version,
    )
    for document in documents:
        if document.ownership == "shared_fixture":
            assert document.tenant_id is None and document.case_id is None
        else:
            assert document.tenant_id == ALPHA_TENANT
            assert document.case_id == ALPHA_CASE_ONE


def test_registry_checksum_tampering_fails_closed(fresh_harness) -> None:
    """SEC-06: a stored checksum change stops the scoped runtime from loading."""

    token = fresh_harness.issue(ALPHA_OPERATOR, tenant_id=ALPHA_TENANT)
    context = fresh_harness.authorization.resolve(
        token,
        request_id="tamper",
        operation=SecurityOperation.REVIEW_CREATE,
    )
    registration = fresh_harness.store.load_case(
        tenant_id=ALPHA_TENANT, case_id=ALPHA_CASE_ONE, subject_id=context.subject_id
    )
    assert registration is not None
    scope = fresh_harness.authorization.build_runtime_scope(
        context,
        registration,
        generation=1,
        deadline_at=fresh_harness.authorization.runtime_scope_deadline(30.0),
    )
    original = fresh_harness.store.load_documents(
        case_document_ids=scope.case_document_ids,
        shared_document_ids=scope.shared_document_ids,
        tenant_id=scope.tenant_id,
        case_id=scope.case_id,
        document_version=scope.document_version,
        index_version=scope.index_version,
    )
    try:
        with fresh_harness.store.transaction() as connection:
            connection.execute(
                text(
                    "UPDATE review_document_registry SET content_checksum = :bad "
                    "WHERE document_id = :d AND ownership = 'case_document'"
                ),
                {"bad": "sha256:" + "0" * 64, "d": "DOC-ALPHA-001"},
            )
        with pytest.raises(ScopeIntegrityError):
            ScopedRetrievalPort.from_store(store=fresh_harness.store, scope=scope)
    finally:
        with fresh_harness.store.transaction() as connection:
            connection.execute(
                text(
                    "UPDATE review_document_registry SET content_checksum = :good "
                    "WHERE document_id = :d AND ownership = 'case_document'"
                ),
                {
                    "good": next(
                        item.content_checksum
                        for item in original
                        if item.document_id == "DOC-ALPHA-001"
                    ),
                    "d": "DOC-ALPHA-001",
                },
            )


def test_forged_and_foreign_evidence_references_are_rejected(fresh_harness) -> None:
    """SEC-06: an EvidenceRef is re-verified against the registration, not trusted."""

    _, _, alpha_scope = _scope_for(
        fresh_harness,
        subject_id=ALPHA_OPERATOR,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_ONE,
    )
    documents = fresh_harness.store.load_documents(
        case_document_ids=alpha_scope.case_document_ids,
        shared_document_ids=alpha_scope.shared_document_ids,
        tenant_id=alpha_scope.tenant_id,
        case_id=alpha_scope.case_id,
        document_version=alpha_scope.document_version,
        index_version=alpha_scope.index_version,
    )
    index = ScopedEvidenceIndex.from_scope(scope=alpha_scope, documents=documents)
    legitimate = next(item for item in documents if item.document_id == "DOC-ALPHA-001")
    legitimate_chunk = ingested_chunk_from_registry(legitimate)
    good = EvidenceRef(
        source_kind=SourceKind.CASE_DOCUMENT,
        source_id=legitimate.source_id,
        version=legitimate.source_version_id,
        chunk_id=legitimate_chunk.chunk_id,
        content_checksum=legitimate.content_checksum,
    )
    index.verify_evidence_ref(good)

    wrong_checksum = good.model_copy(update={"content_checksum": "sha256:" + "1" * 64})
    with pytest.raises(ScopeIntegrityError):
        index.verify_evidence_ref(wrong_checksum)

    foreign = good.model_copy(
        update={
            "chunk_id": _case_chunk_id(
                fresh_harness,
                tenant_id=BETA_TENANT,
                case_id=BETA_CASE_TWO,
                document_id="DOC-BETA-002",
            )
        }
    )
    with pytest.raises(ScopeIntegrityError):
        index.verify_evidence_ref(foreign)

    foreign_fact = BusinessFact(
        tool_name=GET_SUPPLIER_TOOL,
        record_id="SUP-903",
        dataset_version=alpha_scope.dataset_version,
        as_of=datetime.now(UTC),
        status=FactStatus.KNOWN,
        result={"supplier_id": "SUP-903"},
    )
    with pytest.raises(ScopeIntegrityError):
        index.verify_business_fact(foreign_fact)


# ------------------------------------------------------------------------ SEC-07


def test_shared_supplier_cannot_widen_history_across_cases(fresh_harness) -> None:
    """SEC-07: an allowed supplier still only returns this Case's transactions."""

    _, _, case_one_scope = _scope_for(
        fresh_harness,
        subject_id=ALPHA_OPERATOR,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_ONE,
    )
    _, _, case_two_scope = _scope_for(
        fresh_harness,
        subject_id=ALPHA_OPERATOR,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_TWO,
    )
    case_one = ScopedProcurementReader.from_store(
        store=fresh_harness.store, scope=case_one_scope
    )
    case_two = ScopedProcurementReader.from_store(
        store=fresh_harness.store, scope=case_two_scope
    )
    assert case_one.is_supplier_allowed("SUP-901")
    assert case_two.is_supplier_allowed("SUP-901")

    one_history = case_one.list_historical_transactions(
        dataset_version=case_one_scope.dataset_version, supplier_id="SUP-901"
    )
    two_history = case_two.list_historical_transactions(
        dataset_version=case_two_scope.dataset_version, supplier_id="SUP-901"
    )
    assert {item.record_id for item in one_history} == {"TXN-901-A", "TXN-901-B"}
    assert {item.record_id for item in two_history} == {"TXN-902-A"}
    assert not {item.record_id for item in one_history} & {
        item.record_id for item in two_history
    }

    with pytest.raises(ProcurementScopeError):
        case_one.get_purchase_order("PO-902", case_one_scope.dataset_version)
    with pytest.raises(ProcurementScopeError):
        case_one.list_historical_transactions(
            dataset_version=case_one_scope.dataset_version,
            purchase_order_id="PO-902",
        )
    assert case_one.telemetry.denied_tool_calls == 2


# ------------------------------------------------------------------- SEC-09/12


def test_mcp_factory_refuses_to_build_without_a_scope(fresh_harness) -> None:
    """SEC-09/12: there is no implicit unrestricted MCP default."""

    _, _, scope = _scope_for(
        fresh_harness,
        subject_id=ALPHA_OPERATOR,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_ONE,
    )
    reader = ScopedProcurementReader.from_store(store=fresh_harness.store, scope=scope)
    with pytest.raises(TypeError):
        create_procurement_mcp_server(reader)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        create_procurement_mcp_server(reader, scope="not-a-scope")  # type: ignore[arg-type]


def test_direct_mcp_calls_cannot_leave_the_runtime_scope(fresh_harness) -> None:
    """SEC-09/10: scope is checked before data access, not after."""

    _, _, scope = _scope_for(
        fresh_harness,
        subject_id=ALPHA_OPERATOR,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_ONE,
    )
    telemetry = ScopeTelemetry()
    reader = ScopedProcurementReader.from_store(
        store=fresh_harness.store, scope=scope, telemetry=telemetry
    )
    server = create_procurement_mcp_server(reader, scope=scope)

    async def scenario() -> dict[str, object]:
        async with Client(server, raise_exceptions=True) as client:
            tools = await client.list_tools()
            assert tuple(tool.name for tool in tools.tools) == READ_ONLY_TOOL_NAMES
            order = await client.call_tool(
                GET_PURCHASE_ORDER_TOOL,
                {
                    "purchase_order_id": "PO-902",
                    "dataset_version": scope.dataset_version,
                },
            )
            supplier = await client.call_tool(
                GET_SUPPLIER_TOOL,
                {"supplier_id": "SUP-903", "dataset_version": scope.dataset_version},
            )
            history = await client.call_tool(
                LIST_HISTORICAL_TRANSACTIONS_TOOL,
                {
                    "purchase_order_id": "PO-902",
                    "dataset_version": scope.dataset_version,
                },
            )
            return {
                "order": order.structured_content,
                "supplier": supplier.structured_content,
                "history": history.structured_content,
            }

    results = asyncio.run(scenario())
    for tool_name, payload in results.items():
        assert payload is not None, tool_name
        assert payload["outcome"] == ProcurementToolOutcome.SCOPE_DENIED.value, (
            tool_name
        )
        assert payload["error_code"] == "OUT_OF_SCOPE", tool_name
        assert payload["facts"] == [], tool_name
    assert telemetry.denied_tool_calls == 3
    assert telemetry.tool_calls == 0


def test_out_of_scope_denial_never_becomes_a_known_result(fresh_harness) -> None:
    """SEC-09: the official client keeps a scope denial closed."""

    _, _, scope = _scope_for(
        fresh_harness,
        subject_id=ALPHA_OPERATOR,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_ONE,
    )
    reader = ScopedProcurementReader.from_store(store=fresh_harness.store, scope=scope)
    server = create_procurement_mcp_server(reader, scope=scope)

    async def scenario() -> object:
        async with Client(server, raise_exceptions=True) as client:
            port = OfficialProcurementMCPClient(client)
            return await port.collect_review_facts(
                ProcurementWorkerInput(
                    case_id=ALPHA_CASE_ONE,
                    purchase_order_id="PO-902",
                    dataset_version=scope.dataset_version,
                    model_fixture_id="procurement-low",
                )
            )

    result = asyncio.run(scenario())
    assert result.status.value != "known"  # type: ignore[attr-defined]
    assert result.facts == ()  # type: ignore[attr-defined]


def test_stdio_startup_context_is_required_and_protected(
    fresh_harness, tmp_path
) -> None:
    """SEC-12: the stdio entry point refuses to start without a trusted context."""

    import json
    import os

    _, _, scope = _scope_for(
        fresh_harness,
        subject_id=ALPHA_OPERATOR,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_ONE,
    )
    directory = tmp_path / "startup"
    directory.mkdir()
    os.chmod(directory, 0o700)
    payload = {
        "startup_context_version": MCP_STARTUP_VERSION,
        "database_url": _test_database_url(fresh_harness),
        "scope": scope.model_dump(mode="json"),
    }
    protected = directory / "context.json"
    protected.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(protected, 0o644)
    with pytest.raises(McpStartupError):
        load_startup_context(protected)

    os.chmod(protected, 0o600)
    context = load_startup_context(protected)
    try:
        assert context.scope.case_id == ALPHA_CASE_ONE
        assert context.reader.is_order_allowed("PO-901")
        assert not context.reader.is_order_allowed("PO-902")
    finally:
        context.close()

    loose_directory = tmp_path / "loose"
    loose_directory.mkdir()
    os.chmod(loose_directory, 0o755)
    loose = loose_directory / "context.json"
    loose.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(loose, 0o600)
    with pytest.raises(McpStartupError):
        load_startup_context(loose)

    tampered = json.loads(json.dumps(payload))
    tampered["startup_context_version"] = "mcp-startup.v99"
    broken = directory / "tampered.json"
    broken.write_text(json.dumps(tampered), encoding="utf-8")
    os.chmod(broken, 0o600)
    with pytest.raises(McpStartupError):
        load_startup_context(broken)

    missing = directory / "absent.json"
    with pytest.raises(McpStartupError):
        load_startup_context(missing)

    os.environ.pop(STARTUP_FILE_ENV, None)
    with pytest.raises(McpStartupError):
        load_stdio_startup_context()
    os.environ[STARTUP_FILE_ENV] = str(protected)
    try:
        context = load_stdio_startup_context()
        context.close()
    finally:
        os.environ.pop(STARTUP_FILE_ENV, None)


def test_startup_context_is_revoked_with_the_case_grant(fresh_harness) -> None:
    """SEC-12: the stdio context is re-checked against the live authorization store."""

    context_holder, _, scope = _scope_for(
        fresh_harness,
        subject_id=ALPHA_OPERATOR,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_ONE,
    )
    verify_scope_authorization(fresh_harness.store, scope)
    fresh_harness.store.set_case_grant(
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_ONE,
        subject_id=context_holder.subject_id,
        granted=False,
        occurred_at=datetime.now(UTC),
    )
    try:
        with pytest.raises(SecurityError):
            verify_scope_authorization(fresh_harness.store, scope)
    finally:
        fresh_harness.store.set_case_grant(
            tenant_id=ALPHA_TENANT,
            case_id=ALPHA_CASE_ONE,
            subject_id=context_holder.subject_id,
            granted=True,
            occurred_at=datetime.now(UTC),
        )
    verify_scope_authorization(fresh_harness.store, scope)


def test_stale_scope_after_authorization_revision_change_is_rejected(
    fresh_harness,
) -> None:
    """SEC-12: a scope captured before a version change cannot be reused."""

    _, _, scope = _scope_for(
        fresh_harness,
        subject_id=ALPHA_OPERATOR,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_ONE,
    )
    assert scope.authorization_revision >= 1
    verify_scope_authorization(fresh_harness.store, scope)
    mismatched = scope.model_copy(update={"allowed_order_ids": ("PO-902",)})
    with pytest.raises(SecurityError):
        verify_scope_authorization(fresh_harness.store, mismatched)


# ----------------------------------------------------------------------- SEC-11


def test_read_only_runtime_role_cannot_change_procurement_facts(
    fresh_harness,
) -> None:
    """SEC-11: the runtime role is read-only and the MCP start loads no fixture."""

    admin_url = _test_database_url(fresh_harness)
    role = f"v1_03_reader_{fresh_harness.database[-6:]}"
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text(f'DROP ROLE IF EXISTS "{role}"'))
            connection.execute(text(f'CREATE ROLE "{role}" LOGIN'))
            connection.execute(
                text(
                    f'GRANT CONNECT ON DATABASE "{fresh_harness.database}" TO "{role}"'
                )
            )
            connection.execute(text(f'GRANT USAGE ON SCHEMA public TO "{role}"'))
            connection.execute(
                text(f'GRANT SELECT ON ALL TABLES IN SCHEMA public TO "{role}"')
            )
        read_only_url = _replace_user(admin_url, role)
        engine = create_engine(read_only_url)
        try:
            with engine.connect() as connection:
                checksum_before = connection.execute(
                    text(
                        "SELECT md5(string_agg(payload::text, '|' ORDER BY record_id)) "
                        "FROM review_procurement_record"
                    )
                ).scalar_one()
                assert (
                    connection.execute(
                        text("SELECT count(*) FROM review_case_registry")
                    ).scalar_one()
                    > 0
                )
            for statement in (
                "INSERT INTO review_procurement_record (tenant_id, case_id, "
                "record_kind, record_id, dataset_version, as_of, payload) VALUES "
                "('tenant-alpha', 'CASE-ALPHA-001', 'order', 'PO-9001', "
                "'security-procurement.v1', DATE '2026-06-30', '{}'::jsonb)",
                "UPDATE review_procurement_record SET payload = '{}'::jsonb",
                "DELETE FROM review_procurement_record",
                "CREATE TABLE v1_03_forbidden (id integer)",
                "ALTER TABLE review_procurement_record ADD COLUMN hacked text",
                "DROP TABLE review_procurement_record",
            ):
                with pytest.raises(ProgrammingError):
                    with engine.begin() as connection:
                        connection.execute(text(statement))
            with engine.connect() as connection:
                checksum_after = connection.execute(
                    text(
                        "SELECT md5(string_agg(payload::text, '|' ORDER BY record_id)) "
                        "FROM review_procurement_record"
                    )
                ).scalar_one()
            assert checksum_before == checksum_after
        finally:
            engine.dispose()
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP OWNED BY "{role}"'))
            connection.execute(text(f'DROP ROLE IF EXISTS "{role}"'))
        admin.dispose()


def test_credentials_are_never_stored_in_plaintext(fresh_harness) -> None:
    """AC-01/AC-11: the store keeps a digest, never the presented token."""

    token = fresh_harness.issue(DEMO_SUBJECT)
    credential_id = fresh_harness.credential_id_of(token)
    with fresh_harness.store.transaction() as connection:
        row = connection.execute(
            text(
                "SELECT token_digest, digest_algorithm FROM review_credential "
                "WHERE credential_id = :c"
            ),
            {"c": credential_id},
        ).one()
        values = connection.execute(
            text("SELECT token_digest FROM review_credential")
        ).scalars()
        all_values = tuple(values)
    assert row.token_digest != token
    assert len(row.token_digest) == 64
    assert row.digest_algorithm == "sha256"
    assert token not in all_values
    assert all(token not in value for value in all_values)


def test_generated_credentials_are_unique_and_opaque() -> None:
    tokens = {generate_credential()[0] for _ in range(16)}
    assert len(tokens) == 16
    for token in tokens:
        prefix, credential_id, secret = token.split(".")
        assert prefix == "rvk1"
        assert credential_id.startswith("cred-")
        assert len(secret) >= 32


def test_deadline_and_generation_bound_the_runtime(fresh_harness) -> None:
    """SEC-03: the scope carries a fixed generation and an execution deadline."""

    _, _, scope = _scope_for(
        fresh_harness,
        subject_id=ALPHA_OPERATOR,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_ONE,
    )
    assert scope.generation == 1
    delta = scope.deadline_at - scope.issued_at
    assert timedelta(seconds=29) < delta <= timedelta(seconds=30)
    assert scope.deadline_at > datetime.now(UTC)


def _test_database_url(harness: SecurityHarness) -> str:
    from v1_03_support import require_admin_dsn

    base = require_admin_dsn()
    head, _, tail = base.partition("?")
    scheme_and_host, _, _database = head.rpartition("/")
    url = f"{scheme_and_host}/{harness.database}"
    return f"{url}?{tail}" if tail else url


def _replace_user(url: str, user: str) -> str:
    scheme, _, rest = url.partition("://")
    _old_user, _, host = rest.rpartition("@")
    return f"{scheme}://{user}@{host}"


def test_waiting_helper_reaches_a_terminal_state(fresh_harness) -> None:
    """Smoke check that the shared wait helper observes the real v2 API."""

    token = fresh_harness.issue(DEMO_SUBJECT)
    headers = fresh_harness.authorization_header(token)
    with fresh_harness.client() as client:
        created = client.post(
            "/api/v2/reviews",
            headers={**headers, "Idempotency-Key": "sec-matrix-smoke"},
            json={
                "api_version": "review-api.v2",
                "case_id": DEMO_CASE,
                "case_version": "case-low-001.v1",
            },
        )
        assert created.status_code == 202, created.text
        run_id = created.json()["run_id"]
        body = wait_for_status(client, run_id, headers, "pending_confirmation")
        assert body["summary"]["rules_version"] == "2.0.0"


def test_mcp_starts_against_a_read_only_role_without_loading_any_fixture(
    fresh_harness: SecurityHarness, tmp_path
) -> None:
    """SEC-11: the stdio entry point needs read access only, never a migration role."""

    import json
    import os

    _, _, scope = _scope_for(
        fresh_harness,
        subject_id=ALPHA_OPERATOR,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_ONE,
    )
    admin_url = _test_database_url(fresh_harness)
    role = f"v1_03_mcp_{fresh_harness.database[-6:]}"
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text(f'DROP ROLE IF EXISTS "{role}"'))
            connection.execute(text(f'CREATE ROLE "{role}" LOGIN'))
            connection.execute(
                text(
                    f'GRANT CONNECT ON DATABASE "{fresh_harness.database}" TO "{role}"'
                )
            )
            connection.execute(text(f'GRANT USAGE ON SCHEMA public TO "{role}"'))
            connection.execute(
                text(f'GRANT SELECT ON ALL TABLES IN SCHEMA public TO "{role}"')
            )
            table_count = connection.execute(
                text(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                )
            ).scalar_one()
        directory = tmp_path / "startup"
        directory.mkdir()
        os.chmod(directory, 0o700)
        context_file = directory / "context.json"
        context_file.write_text(
            json.dumps(
                {
                    "startup_context_version": MCP_STARTUP_VERSION,
                    "database_url": _replace_user(admin_url, role),
                    "scope": scope.model_dump(mode="json"),
                }
            ),
            encoding="utf-8",
        )
        os.chmod(context_file, 0o600)

        context = load_startup_context(context_file)
        try:
            server = create_procurement_mcp_server(context.reader, scope=context.scope)

            async def scenario() -> object:
                async with Client(server, raise_exceptions=True) as client:
                    return await client.call_tool(
                        GET_PURCHASE_ORDER_TOOL,
                        {
                            "purchase_order_id": "PO-901",
                            "dataset_version": context.scope.dataset_version,
                        },
                    )

            result = asyncio.run(scenario())
            assert result.structured_content is not None
            assert result.structured_content["outcome"] == "KNOWN"
        finally:
            context.close()
        with admin.connect() as connection:
            after = connection.execute(
                text(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                )
            ).scalar_one()
        assert after == table_count
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP OWNED BY "{role}"'))
            connection.execute(text(f'DROP ROLE IF EXISTS "{role}"'))
        admin.dispose()
