"""FIX-07 S-01: the scoped retrieval assembly must narrow *before* it reads.

The 2026-09-17 review reproduced a cross-scope read on the normal assembly path:
``ScopedRetrievalPort.from_store`` loaded the in-scope documents and their scores, and
then ran a second statement that enumerated *every* registered chunk of the same
document/index version and read their scores as well.  That statement carried no
tenant, Case or ownership predicate, and the resulting identifiers were left on the
port object as ``out_of_scope_chunk_ids``, so any in-process holder of the port could
read another tenant's chunk ids out of it.

These tests observe the real assembly at the connection level, so they fail on the
implementation that performs the extra read and pass once the port only ever asks for
the chunk ids the ``RuntimeScope`` allows.  The "the out-of-scope document would have
ranked higher" proof lives in
:func:`test_out_of_scope_candidate_would_outrank_in_scope` and is performed through the
test-management read *before* the restricted port exists; the port itself never performs
or exposes that read.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from sqlalchemy import event, text

from procurement_review.retrieval import RetrievalRequest, RetrievalStatus
from procurement_review.scoped_runtime import (
    ScopeTelemetry,
    ScopedRetrievalPort,
    ScopedReviewRuntime,
    ingested_chunk_from_registry,
)
from procurement_review.security_contracts import SecurityOperation

from v1_03_support import (
    ALPHA_TENANT,
    BETA_OPERATOR,
    BETA_TENANT,
    SecurityHarness,
)

# Frozen synthetic-fixture constants.  They are part of the retrieval fixture contract
# (`security-fixture.v1`), not of the product logic under test: the other tenant's Case
# document is declared at 0.72 and this tenant's own Case document at 0.66 for the
# ranking query below, so a post-filter would consume a rank before dropping it.
RANKING_QUERY = "payment terms termination rights"
OUT_OF_SCOPE_SCORE = 0.72
IN_SCOPE_SCORE = 0.66

ALPHA_CASE_TWO = "CASE-ALPHA-002"
BETA_CASE_TWO = "CASE-BETA-002"
ALPHA_DOCUMENT = "DOC-ALPHA-002"
BETA_DOCUMENT = "DOC-BETA-002"

# How many registry rows and distinct chunk ids the frozen fixture registers.
REGISTERED_DOCUMENT_ROWS = 10
REGISTERED_DISTINCT_CHUNKS = 10


# --------------------------------------------------------------------- helpers


def _scope_for(
    harness: SecurityHarness,
    *,
    subject_id: str,
    tenant_id: str,
    case_id: str,
):
    """Resolve one credential into the real ``RuntimeScope`` the runtime would get."""

    token = harness.issue(subject_id, tenant_id=tenant_id)
    context = harness.authorization.resolve(
        token,
        request_id="fix07-scope",
        operation=SecurityOperation.REVIEW_CREATE,
    )
    registration = harness.authorization.require_case(
        context,
        case_id,
        request_id="fix07-scope",
        operation=SecurityOperation.REVIEW_CREATE,
    )
    scope = harness.authorization.build_runtime_scope(
        context,
        registration,
        generation=1,
        deadline_at=harness.authorization.runtime_scope_deadline(30.0),
    )
    return context, registration, scope


def _beta_scope(harness: SecurityHarness):
    return _scope_for(
        harness,
        subject_id=BETA_OPERATOR,
        tenant_id=BETA_TENANT,
        case_id=BETA_CASE_TWO,
    )[2]


def _case_chunk_id(
    harness: SecurityHarness, *, tenant_id: str, case_id: str, document_id: str
) -> str:
    """The derived chunk ID of one registered Case document."""

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


def _chunk_rows(harness: SecurityHarness, tenant_id: str) -> list[tuple[str, str]]:
    """Every registry chunk id of one tenant together with its content checksum."""

    with harness.store.transaction() as connection:
        rows = connection.execute(
            text(
                "SELECT chunk_payload ->> 'chunk_id', content_checksum "
                "FROM review_document_registry WHERE tenant_id = :t"
            ),
            {"t": tenant_id},
        ).all()
    return [(str(row[0]), str(row[1])) for row in rows]


@contextmanager
def observed_statements(harness: SecurityHarness) -> Iterator[list[tuple[str, Any]]]:
    """Record every SQL statement the engine executes inside the block.

    This is a connection-level observation, not a wrapper around the code under test:
    whatever the assembly really sends to PostgreSQL is what gets asserted on.
    """

    captured: list[tuple[str, Any]] = []

    def record(
        _connection: Any,
        _cursor: Any,
        statement: str,
        parameters: Any,
        _context: Any,
        _executemany: bool,
    ) -> None:
        captured.append((statement, parameters))

    engine = harness.store.engine
    event.listen(engine, "before_cursor_execute", record)
    try:
        yield captured
    finally:
        event.remove(engine, "before_cursor_execute", record)


def _requested_chunk_ids(parameters: Any) -> frozenset[str]:
    """The chunk-id list one observed score statement asked for."""

    if not isinstance(parameters, Mapping):
        return frozenset()
    raw = parameters.get("ids")
    if raw is None or isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return frozenset()
    return frozenset(str(item) for item in raw)


def _exposed_chunk_identifiers(target: object) -> set[str]:
    """Every chunk identifier reachable from one object's public surface.

    The point is not to name the offending attribute: an in-process caller holding the
    port can read whatever the port exposes, so the invariant is about reachability.
    """

    reachable: set[str] = set()
    for name in dir(target):
        if name.startswith("__"):
            continue
        try:
            value = getattr(target, name)
        except Exception:  # pragma: no cover - defensive: a property may raise
            continue
        if isinstance(value, (set, frozenset)) and all(
            isinstance(item, str) for item in value
        ):
            reachable |= set(value)
    return reachable


def _delete_tenant_rows(harness: SecurityHarness, tenant_id: str) -> list[str]:
    """Remove one tenant's registry and score rows from the disposable test database."""

    chunk_ids = [chunk_id for chunk_id, _checksum in _chunk_rows(harness, tenant_id)]
    with harness.store.transaction() as connection:
        connection.execute(
            text("DELETE FROM review_document_registry WHERE tenant_id = :t"),
            {"t": tenant_id},
        )
        if chunk_ids:
            connection.execute(
                text("DELETE FROM review_retrieval_score WHERE chunk_id = ANY(:ids)"),
                {"ids": chunk_ids},
            )
    return chunk_ids


# ------------------------------------------------------------------- S-01 tests


def test_scoped_retrieval_reads_only_allowed_score_rows(fresh_harness) -> None:
    """The assembly may only ask the score table for chunk ids inside its scope."""

    alpha_chunk = _case_chunk_id(
        fresh_harness,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_TWO,
        document_id=ALPHA_DOCUMENT,
    )
    beta_chunk = _case_chunk_id(
        fresh_harness,
        tenant_id=BETA_TENANT,
        case_id=BETA_CASE_TWO,
        document_id=BETA_DOCUMENT,
    )
    assert alpha_chunk != beta_chunk

    beta_scope = _beta_scope(fresh_harness)
    with observed_statements(fresh_harness) as statements:
        port = ScopedRetrievalPort.from_store(
            store=fresh_harness.store, scope=beta_scope
        )

    score_reads = [
        (statement, parameters)
        for statement, parameters in statements
        if "review_retrieval_score" in statement
    ]
    assert score_reads, "the assembly must read the scores of its own documents"

    requested: set[str] = set()
    for _statement, parameters in score_reads:
        requested |= _requested_chunk_ids(parameters)

    assert requested, "the observed score reads must carry an explicit chunk-id list"
    outside = sorted(requested - set(port.allowed_chunk_ids))
    assert not outside, (
        "the scoped retrieval assembly asked the score table for chunk ids outside "
        f"its scope: {outside}"
    )
    assert alpha_chunk not in requested, (
        "another tenant's chunk id was requested by the scoped retrieval assembly"
    )


def test_scoped_runtime_build_keeps_every_registry_read_scope_predicated(
    fresh_harness,
) -> None:
    """Every registry read on the full assembly path carries a scope predicate."""

    beta_scope = _beta_scope(fresh_harness)
    allowed = ScopedRetrievalPort.from_store(
        store=fresh_harness.store, scope=beta_scope, telemetry=ScopeTelemetry()
    ).allowed_chunk_ids

    with observed_statements(fresh_harness) as statements:
        ScopedReviewRuntime.build(store=fresh_harness.store, scope=beta_scope)

    registry_reads = [
        statement
        for statement, _parameters in statements
        if "review_document_registry" in statement
    ]
    assert registry_reads
    for statement in registry_reads:
        normalised = " ".join(statement.split()).lower()
        assert "ownership" in normalised, (
            f"a registry read has no ownership predicate: {normalised}"
        )
        assert "tenant_id" in normalised, (
            f"a registry read has no tenant predicate: {normalised}"
        )

    score_ids: set[str] = set()
    for statement, parameters in statements:
        if "review_retrieval_score" in statement:
            score_ids |= _requested_chunk_ids(parameters)
    assert score_ids, "the assembly must read the scores of its own documents"
    outside = sorted(score_ids - set(allowed))
    assert not outside, (
        "the full runtime assembly asked the score table for chunk ids outside its "
        f"scope: {outside}"
    )


def test_scoped_retrieval_port_exposes_no_out_of_scope_chunk_identifiers(
    fresh_harness,
) -> None:
    """Nothing reachable from the restricted port may name a chunk out of scope."""

    alpha_chunk = _case_chunk_id(
        fresh_harness,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_TWO,
        document_id=ALPHA_DOCUMENT,
    )
    beta_chunk = _case_chunk_id(
        fresh_harness,
        tenant_id=BETA_TENANT,
        case_id=BETA_CASE_TWO,
        document_id=BETA_DOCUMENT,
    )
    with fresh_harness.store.transaction() as connection:
        ownership = connection.execute(
            text(
                "SELECT ownership, tenant_id, case_id FROM review_document_registry "
                "WHERE chunk_payload ->> 'chunk_id' = :chunk"
            ),
            {"chunk": alpha_chunk},
        ).one()
    assert tuple(str(item) for item in ownership) == (
        "case_document",
        ALPHA_TENANT,
        ALPHA_CASE_TWO,
    )

    port = ScopedRetrievalPort.from_store(
        store=fresh_harness.store, scope=_beta_scope(fresh_harness)
    )

    assert beta_chunk in port.allowed_chunk_ids
    assert alpha_chunk not in port.allowed_chunk_ids
    reachable = _exposed_chunk_identifiers(port)
    assert alpha_chunk not in reachable, (
        "the restricted port exposes another tenant's chunk identifier"
    )
    assert reachable <= set(port.allowed_chunk_ids), (
        "the restricted port exposes chunk identifiers outside its scope: "
        f"{sorted(reachable - set(port.allowed_chunk_ids))}"
    )


def test_scoped_retrieval_observation_does_not_depend_on_out_of_scope_rows(
    fresh_harness,
) -> None:
    """What the port reports about itself must not change when foreign rows vanish.

    This is the behavioural form of "no cross-scope scanning": the port is assembled
    twice for the same scope, once while another tenant's registry and score rows exist
    and once after they are deleted from the disposable test database.  A port whose
    work, telemetry or results depended on those rows would differ between the two.
    """

    beta_scope = _beta_scope(fresh_harness)
    first = ScopeTelemetry()
    with_foreign_rows = ScopedRetrievalPort.from_store(
        store=fresh_harness.store, scope=beta_scope, telemetry=first
    )
    request = RetrievalRequest(
        query=RANKING_QUERY,
        index_version=beta_scope.index_version,
        limit=10,
        minimum_results=1,
    )
    first_response = with_foreign_rows.retrieve(request)
    assert first_response.status is RetrievalStatus.KNOWN

    removed = _delete_tenant_rows(fresh_harness, ALPHA_TENANT)
    assert removed, "the fixture must register another tenant's rows to delete"

    second = ScopeTelemetry()
    without_foreign_rows = ScopedRetrievalPort.from_store(
        store=fresh_harness.store, scope=beta_scope, telemetry=second
    )
    second_response = without_foreign_rows.retrieve(request)

    assert first.as_summary() == second.as_summary(), (
        "the restricted port's own observation depends on rows outside its scope: "
        f"{first.as_summary()} != {second.as_summary()}"
    )
    assert without_foreign_rows.allowed_chunk_ids == (
        with_foreign_rows.allowed_chunk_ids
    )
    assert [hit.chunk_id for hit in second_response.results] == [
        hit.chunk_id for hit in first_response.results
    ]


def test_out_of_scope_candidate_would_outrank_in_scope(fresh_harness) -> None:
    """The relocated proof: out-of-scope content would win, and never even competes.

    The score comparison is an explicit test-management read performed *before* the
    restricted port is built; the port then answers from its own narrowed index, and no
    statement it issues may name the out-of-scope chunk.
    """

    alpha_chunk = _case_chunk_id(
        fresh_harness,
        tenant_id=ALPHA_TENANT,
        case_id=ALPHA_CASE_TWO,
        document_id=ALPHA_DOCUMENT,
    )
    beta_chunk = _case_chunk_id(
        fresh_harness,
        tenant_id=BETA_TENANT,
        case_id=BETA_CASE_TWO,
        document_id=BETA_DOCUMENT,
    )
    beta_scope = _beta_scope(fresh_harness)

    # Test-management read: it runs on the fixture's own administrative engine, outside
    # any runtime scope, and it is the only place in this module that may look at the
    # other tenant's declared score.
    declared = fresh_harness.store.retrieval_scores(
        document_version=beta_scope.document_version,
        index_version=beta_scope.index_version,
        chunk_ids=[alpha_chunk, beta_chunk],
    )[RANKING_QUERY]
    assert declared[alpha_chunk] == OUT_OF_SCOPE_SCORE
    assert declared[beta_chunk] == IN_SCOPE_SCORE
    assert declared[alpha_chunk] > declared[beta_chunk]

    with observed_statements(fresh_harness) as statements:
        port = ScopedRetrievalPort.from_store(
            store=fresh_harness.store, scope=beta_scope
        )
        response = port.retrieve(
            RetrievalRequest(
                query=RANKING_QUERY,
                index_version=beta_scope.index_version,
                limit=10,
                minimum_results=1,
            )
        )

    assert response.status is RetrievalStatus.KNOWN
    returned = [hit.chunk_id for hit in response.results]
    assert beta_chunk in returned
    assert alpha_chunk not in returned
    assert all(chunk_id in port.allowed_chunk_ids for chunk_id in returned)
    scores = {hit.chunk_id: hit.vector_score for hit in response.results}
    assert returned == sorted(returned, key=lambda item: (-scores[item], item))

    requested: set[str] = set()
    for statement, parameters in statements:
        if "review_retrieval_score" in statement:
            requested |= _requested_chunk_ids(parameters)
    assert alpha_chunk not in requested, (
        "the restricted port read the out-of-scope chunk's score row"
    )


def test_registered_chunk_identity_is_proven_by_the_checksum_contract(
    fresh_harness,
) -> None:
    """Whether a chunk id alone identifies a chunk, checked against the real registry.

    ``ScopedRetrievalPort`` reads scores by ``chunk_id`` inside one
    ``(document_version, index_version)`` pair, so the port relies on a chunk id being
    a *content identity* rather than a tenant-scoped naming convention.  The registry
    has no unique constraint on ``chunk_id`` and the score table has no tenant column,
    so this test proves the property the port relies on directly, against every row the
    frozen fixture actually registers:

    1. each payload satisfies ``IngestedChunk``'s derived-identity validator, i.e.
       ``chunk_id == make_chunk_id(document_id, ordinal, locator, content_checksum,
       chunk rules)`` and ``content_checksum == sha256(content)`` — a checksum contract,
       not a naming convention;
    2. therefore two rows that share a ``chunk_id`` must also agree on
       ``content_checksum`` and ``document_id``; a row pair that disagreed would mean the
       score lookup could mix two different contents, and the port could no longer read
       by ``chunk_id`` alone.

    The measured counts below are the frozen fixture contract (`security-fixture.v1`).
    A fixture change that alters them fails this test on purpose: it is the signal to
    reconsider the score-table identity (see the FIX-07 Evidence note on the minimal
    migration) rather than to relax the assertion.
    """

    from procurement_review.review_store import RegisteredDocument

    with fresh_harness.store.transaction() as connection:
        rows = (
            connection.execute(text("SELECT * FROM review_document_registry"))
            .mappings()
            .all()
        )

    assert len(rows) == REGISTERED_DOCUMENT_ROWS
    seen: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        # Built through the *production* conversion, so the identity contract checked
        # here is the same one `ScopedRetrievalPort` depends on.
        chunk = ingested_chunk_from_registry(
            RegisteredDocument(
                document_id=row["document_id"],
                ownership=row["ownership"],
                tenant_id=row["tenant_id"],
                case_id=row["case_id"],
                category=row["category"],
                source_id=row["source_id"],
                source_version_id=row["source_version_id"],
                document_version=row["document_version"],
                index_version=row["index_version"],
                content_checksum=row["content_checksum"],
                chunk_payload=row["chunk_payload"],
            )
        )
        assert chunk.content_checksum == row["content_checksum"]
        seen.setdefault(chunk.chunk_id, set()).add(
            (chunk.document_id, chunk.content_checksum)
        )

    shared_ids = {
        chunk_id: identities
        for chunk_id, identities in seen.items()
        if len(identities) > 1
    }
    assert not shared_ids, (
        "one chunk id names more than one content identity in the registry: "
        f"{sorted(shared_ids)}"
    )
    assert len(seen) == REGISTERED_DISTINCT_CHUNKS
