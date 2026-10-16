"""V-G-01: real computed retrieval through the product's scoped port on PostgreSQL.

The corpus, the scope and the port are the product's own: material is registered in
an isolated PostgreSQL registry, the RuntimeScope narrows candidates before any
query, and the ranking is computed from chunk text.  No predeclared vector score
exists in this corpus, so nothing but the real lexical computation can produce a
hit, and the Oracle context is never consulted to backfill a miss.

Two corpus configurations are measured separately and neither is folded into the
other: the product scope, whose shared library holds material from unrelated
generation families, and an adversarial scope that also holds same-family siblings.
The DEV generator reuses one policy text byte for byte across a family and repeats
contract templates once identifiers are masked, so the adversarial configuration has
required identities no content-based retrieval can separate; that is reported as a
material property with its count.  A shared masked form is not accepted as an
explanation on its own: every required identity is judged by the ranking the actual
query produced, and only a byte-identical peer or a query that failed to separate the
target counts as an unavoidable miss.  The remaining measured residue is registered
exactly, never dropped from a denominator.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from procurement_review.authorization import (
    AuthorizationService,
    generate_credential,
)
from procurement_review.contracts import EvidenceRef, SourceKind
from procurement_review.eval_g_distinguishability import (
    distinguishability_rows,
    indistinguishable_identities,
    unexplained_misses,
)
from procurement_review.eval_g_retrieval import (
    G_INDEX_VERSION,
    G_OTHER_TENANT,
    G_SUBJECT,
    G_SUPERSEDED_VERSION,
    G_TENANT,
    TOP_K,
    build_corpus,
    corpus_cases,
    domain_port,
    measure_slice,
    negative_probe,
    observe_assembly,
    observe_retrieval,
    q08,
    query_for,
    required_for,
)
from procurement_review.ingestion import LocatorKind, SourceLocator
from procurement_review.retrieval import (
    RetrievalHit,
    RetrievalRequest,
)
from procurement_review.review_store import ReviewStore
from procurement_review.scoped_runtime import ScopedRetrievalPort
from procurement_review.security_contracts import SecurityOperation
from v1_03_support import create_isolated_database, drop_database

DATA_VERSION = "gold-set.v3.1"
MEASURED = 6
DISTRACTORS = 40
VARIANTS = ("original", "without_case_id")
DOMAINS = ("policy", "contract")


@dataclass(slots=True)
class Environment:
    store: ReviewStore
    authorization: AuthorizationService
    measured: tuple
    distractors: tuple
    siblings: tuple
    truths: dict


def _build(*, with_siblings: bool) -> Iterator[Environment]:
    database_url, database = create_isolated_database()
    store = ReviewStore.from_url(database_url)
    try:
        store.migrate()
        measured, distractors, siblings, truths = corpus_cases(
            DATA_VERSION, measured_count=MEASURED, distractor_count=DISTRACTORS
        )
        store.seed(
            build_corpus(
                measured=measured,
                distractors=distractors,
                siblings=siblings if with_siblings else (),
                data_version=DATA_VERSION,
                fixture_version=(
                    "v1-07g-retrieval-corpus-adversarial.v1"
                    if with_siblings
                    else "v1-07g-retrieval-corpus.v1"
                ),
            )
        )
        yield Environment(
            store=store,
            authorization=AuthorizationService(store=store),
            measured=measured,
            distractors=distractors,
            siblings=siblings,
            truths=truths,
        )
    finally:
        store.close()
        drop_database(database)


@pytest.fixture
def product_scope() -> Iterator[Environment]:
    yield from _build(with_siblings=False)


@pytest.fixture
def adversarial_scope() -> Iterator[Environment]:
    yield from _build(with_siblings=True)


def _issue(environment: Environment, subject_id: str) -> str:
    token, credential_id, digest = generate_credential()
    issued_at = datetime.now(UTC)
    environment.store.issue_credential(
        credential_id=credential_id,
        tenant_id=G_TENANT,
        subject_id=subject_id,
        token_digest=digest,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(seconds=3_600),
    )
    return token


def _scope(environment: Environment, case_id: str):
    token = _issue(environment, G_SUBJECT)
    context = environment.authorization.resolve(
        token, request_id="g-retrieval", operation=SecurityOperation.REVIEW_CREATE
    )
    registration = environment.authorization.require_case(
        context,
        case_id,
        request_id="g-retrieval",
        operation=SecurityOperation.REVIEW_CREATE,
    )
    return environment.authorization.build_runtime_scope(
        context,
        registration,
        generation=1,
        deadline_at=environment.authorization.runtime_scope_deadline(120.0),
    )


def _measure(environment: Environment):
    slices = []
    scopes = {}
    for case in environment.measured:
        scope = _scope(environment, case.case_id)
        scopes[case.case_id] = scope
        for variant in VARIANTS:
            for domain in DOMAINS:
                slices.append(
                    measure_slice(
                        domain_port(environment.store, scope, domain),
                        case=case,
                        truth=environment.truths[case.case_id],
                        domain=domain,
                        variant=variant,
                        scope=scope,
                    )
                )
    return slices, scopes


def _registered_hit(
    environment: Environment, document_id: str, *, rank: int
) -> RetrievalHit:
    """One real registered chunk, built from its own stored payload."""

    with environment.store.transaction() as connection:
        payload = connection.execute(
            text(
                "SELECT chunk_payload FROM review_document_registry WHERE document_id"
                " = :d"
            ),
            {"d": document_id},
        ).scalar_one()
    locator = payload["locator"]
    return RetrievalHit(
        rank=rank,
        source_id=payload["source_id"],
        source_version=payload["source_version"],
        source_version_id=payload["source_version_id"],
        source_checksum=payload["source_checksum"],
        document_id=payload["document_id"],
        chunk_id=payload["chunk_id"],
        ordinal=payload["ordinal"],
        locator=SourceLocator(
            kind=LocatorKind(locator["kind"]),
            line_start=locator["line_start"],
            line_end=locator["line_end"],
        ),
        content=payload["content"],
        content_checksum=payload["content_checksum"],
        ingestion_version=payload["ingestion_version"],
        chunk_rules_version=payload["chunk_rules_version"],
        chunk_rules_checksum=payload["chunk_rules_checksum"],
        index_version=payload["index_version"],
        lexical_score=0.5,
        vector_score=0.0,
        fusion_score=0.5,
        evidence_ref=EvidenceRef(
            source_kind=SourceKind.CASE_DOCUMENT,
            source_id=payload["source_id"],
            version=payload["source_version_id"],
            chunk_id=payload["chunk_id"],
            content_checksum=payload["content_checksum"],
        ),
    )


class _InjectedPort:
    """The real scoped port with registered-but-unauthorized chunks appended.

    This is the failure mode the counts exist for: an upstream hands the assembly a
    chunk the scope never admitted.  ``model_copy`` deliberately bypasses validation so
    the injected return travels exactly as a defective adapter would produce it.
    """

    def __init__(self, port, injected: tuple[RetrievalHit, ...]) -> None:
        self._port = port
        self._injected = injected

    @property
    def allowed_chunk_ids(self) -> frozenset[str]:
        return self._port.allowed_chunk_ids

    def retrieve(self, request: RetrievalRequest):
        response = self._port.retrieve(request)
        combined = (*response.results, *self._injected)
        hits = tuple(
            hit.model_copy(update={"rank": rank})
            for rank, hit in enumerate(combined, start=1)
        )
        return response.model_copy(update={"results": hits})


def test_an_injected_out_of_bounds_return_is_counted_and_fails_the_gate(product_scope):
    """An unauthorized chunk that reaches the assembly must fail, not be assumed away."""

    case = product_scope.measured[0]
    scope = _scope(product_scope, case.case_id)
    source_id = case.model_input.policy_document.source_id
    injected = (
        _registered_hit(product_scope, f"DOC-{source_id}-SUPERSEDED", rank=1),
        _registered_hit(product_scope, f"DOC-{source_id}-OTHER-TENANT", rank=2),
    )
    port = _InjectedPort(domain_port(product_scope.store, scope, "policy"), injected)

    observation = observe_retrieval(
        port.retrieve(
            RetrievalRequest(
                query=query_for(case, "policy", "original"),
                index_version=scope.index_version,
                limit=TOP_K,
            )
        ),
        scope=scope,
        allowed_chunk_ids=port.allowed_chunk_ids,
    )
    assert set(observation.out_of_scope_chunk_ids) == {
        injected[0].chunk_id,
        injected[1].chunk_id,
    }
    assert observation.version_drift_ids == (
        (
            injected[0].source_id,
            injected[0].source_version,
            injected[0].content_checksum.removeprefix("sha256:"),
        ),
    )
    assert observation.request_matches_scope is True
    assert observe_assembly(
        observation.returned_chunk_ids, allowed_chunk_ids=port.allowed_chunk_ids
    ) == (injected[0].chunk_id, injected[1].chunk_id)

    row = measure_slice(
        port,
        case=case,
        truth=product_scope.truths[case.case_id],
        domain="policy",
        variant="original",
        scope=scope,
    )
    assert row.wrong_scope_count == 2
    assert row.wrong_version_count == 1
    assert row.assembly_out_of_scope == 2
    assert row.assembly_checked is True
    # The injected chunk also sits in the assembled identity list a model context
    # would receive, so the observation is on the real data, not on a stub.
    assert (
        injected[0].source_id,
        injected[0].source_version,
        injected[0].content_checksum.removeprefix("sha256:"),
    ) in row.assembled
    report = q08([row])
    assert report["meets_threshold"] is False
    assert report["recall_at_five"]["wrong_scope_count"] == 2
    assert report["recall_at_five"]["assembly_out_of_scope"] == 2


def test_truth_identities_resolve_to_the_authorized_document_only(product_scope):
    """The shared library repeats sources, so a truth identity is not its authorization.

    For every required identity the audit reads the registry itself: exactly one row
    carrying the truth's checksum is inside the Case's authorized document set, its
    category is the domain the truth locator names, and the same source/version
    registered to another tenant is a real row that the scope never admits.
    """

    statements = []
    for case in product_scope.measured:
        scope = _scope(product_scope, case.case_id)
        authorized = set(scope.case_document_ids) | set(scope.shared_document_ids)
        truth = product_scope.truths[case.case_id]
        for domain in DOMAINS:
            required = required_for(truth, domain)
            assert required, (case.case_id, domain)
            for identity in required:
                source_id, version, checksum = identity
                with product_scope.store.transaction() as connection:
                    rows = connection.execute(
                        text(
                            "SELECT document_id, category, tenant_id, ownership FROM"
                            " review_document_registry WHERE source_id = :s AND"
                            " document_version = :v AND content_checksum = :c"
                        ),
                        {"s": source_id, "v": version, "c": f"sha256:{checksum}"},
                    ).all()
                inside = [row for row in rows if row[0] in authorized]
                assert len(inside) == 1, (case.case_id, identity, rows)
                assert inside[0][1] == domain, (case.case_id, identity, inside)
                assert rows, (case.case_id, identity)
                statements.append(
                    {
                        "case": case.case_id,
                        "domain": domain,
                        "source": source_id,
                        "matching_rows": len(rows),
                        "authorized_rows": len(inside),
                    }
                )
            assert scope.document_version == version
    # The namesake is not hypothetical: the policy source is registered once for the
    # Case and once for another tenant under the same identity.
    case = product_scope.measured[0]
    scope = _scope(product_scope, case.case_id)
    source_id = case.model_input.policy_document.source_id
    with product_scope.store.transaction() as connection:
        namesake = connection.execute(
            text(
                "SELECT tenant_id, document_id FROM review_document_registry WHERE"
                " document_id = :d"
            ),
            {"d": f"DOC-{source_id}-OTHER-TENANT"},
        ).one()
    assert namesake[0] == G_OTHER_TENANT
    assert namesake[1] not in (
        set(scope.case_document_ids) | set(scope.shared_document_ids)
    )
    assert all(item["authorized_rows"] == 1 for item in statements)


def test_the_scope_and_version_counts_are_zero_only_when_observed(product_scope):
    """The honest zero: the product port returns nothing outside its own scope."""

    slices, _scopes = _measure(product_scope)
    for row in slices:
        assert row.assembly_checked is True
        assert row.request_matches_scope is True
        assert row.wrong_scope_count == 0
        assert row.wrong_version_count == 0
        assert row.assembly_out_of_scope == 0
        assert len(row.assembled) == len(row.top_five) > 0


def test_the_dev_retrieval_gate_passes_under_the_product_scope(product_scope):
    slices, _scopes = _measure(product_scope)
    report = q08(slices)
    assert report["meets_threshold"] is True, report
    by_slice = report["by_query_variant_and_domain"]
    for name, item in by_slice.items():
        assert item["meets_threshold"] is True, (name, item)
        assert item["wrong_version_count"] == 0
        assert item["wrong_scope_count"] == 0
        # Both domains and both variants carry a real denominator: a measured gate,
        # not an empty one reported as a pass.
        assert item["denominator"] == MEASURED, (name, item)
    assert report["recall_at_five"]["denominator"] == (
        len(VARIANTS) * len(DOMAINS) * MEASURED
    )
    assert not indistinguishable_identities(
        product_scope.measured, product_scope.distractors
    )


def test_same_family_siblings_make_de_identified_policy_retrieval_ambiguous(
    adversarial_scope,
):
    """The adversarial scope is reported with its cause, not passed and not dropped."""

    slices, scopes = _measure(adversarial_scope)
    report = q08(slices)
    ambiguous = indistinguishable_identities(
        adversarial_scope.measured,
        (*adversarial_scope.distractors, *adversarial_scope.siblings),
    )
    # The corpus-level criteria still fire: the generator reuses one policy text byte
    # for byte across a family, and contract templates repeat once identifiers are
    # masked.  This is the material property; it is not the retrieval judgement.
    assert ambiguous, "expected the DEV family to reuse document text"
    criteria = {item["criterion"] for item in ambiguous.values()}
    assert criteria == {"identical_bytes", "identical_form"}

    # Judge every required identity of every measured slice with the actual query: a
    # miss is explained only when another authorized chunk is byte identical, or when
    # the real query failed to rank the target above its masked-form peers.  A shared
    # masked form is not accepted as an explanation on its own.
    corpus = (
        *adversarial_scope.measured,
        *adversarial_scope.distractors,
        *adversarial_scope.siblings,
    )
    rows = []
    for case in adversarial_scope.measured:
        scope = scopes[case.case_id]
        for variant in VARIANTS:
            for domain in DOMAINS:
                rows.extend(
                    distinguishability_rows(
                        domain_port(adversarial_scope.store, scope, domain),
                        case=case,
                        truth=adversarial_scope.truths[case.case_id],
                        domain=domain,
                        variant=variant,
                        scope=scope,
                        corpus=corpus,
                    )
                )
    measured = {row.criterion for row in rows}
    assert "identical_bytes" in measured, measured
    assert measured <= {
        "unique",
        "identical_bytes",
        "query_separates",
        "query_does_not_separate",
    }
    # A same-form peer whose query did separate the target is retrievable material:
    # the masked form alone must not be counted as an unavoidable miss.
    separated = [row for row in rows if row.criterion == "query_separates"]
    assert separated, "expected at least one masked-form identity the query separates"
    for row in separated:
        assert row.target_rank is not None
        assert all(
            peer_rank is None or peer_rank > row.target_rank
            for _name, peer_rank in row.peer_ranks
        )
    # The measured residue is registered as a known reading, not assumed away.  The
    # struck slice is V107-D-0006/contract/without_case_id: its target ranked 14 with
    # its four same-form peers at 16-20, i.e. the query did separate it from every
    # masked-form peer, but thirteen other contracts of other families scored above it
    # and the de-identified question carries no lexical token that could prefer the
    # case's own document.  The list is asserted exactly so a new unexplained miss
    # fails this test and a reading that silently disappears fails it too.
    known_residue = {("V107-D-0006", "contract", "without_case_id")}
    observed_residue = {
        (row.case_id, row.evidence_domain, row.query_variant)
        for row in unexplained_misses(rows)
    }
    assert observed_residue == known_residue, sorted(observed_residue)
    # The gate is honestly reported as not met in that configuration.
    assert report["meets_threshold"] is False
    assert (
        report["by_query_variant_and_domain"]["original:policy"]["meets_threshold"]
        is False
    )


def test_the_ranking_is_computed_and_not_carried_by_declared_scores(product_scope):
    case = product_scope.measured[0]
    scope = _scope(product_scope, case.case_id)
    port = domain_port(product_scope.store, scope, "policy")
    assert port._retriever._vector.adapter_version == "no-vector-v1"
    assert port._retriever._vector.test_only is False
    assert port._retriever._lexical.adapter_version == "deterministic-lexical-idf-v1"
    # The frozen fixture assembly reads predeclared scores instead; this corpus
    # declares none, so the two assemblies cannot produce the same evidence.
    fixture_port = ScopedRetrievalPort.from_store(
        store=product_scope.store, scope=scope
    )
    assert fixture_port._retriever._vector.adapter_version == (
        "deterministic-fake-vector-v1"
    )
    assert fixture_port._retriever._vector.recall("any query", limit=5) == ()
    request = RetrievalRequest(
        query=query_for(case, "policy", "without_case_id"),
        index_version=scope.index_version,
        limit=5,
    )
    hits = port.retrieve(request).results
    assert hits and hits[0].source_id == case.model_input.policy_document.source_id
    assert hits[0].lexical_score > 0.0 and hits[0].vector_score == 0.0


def test_scope_and_version_predicates_exclude_the_near_copies(product_scope):
    case = product_scope.measured[0]
    scope = _scope(product_scope, case.case_id)
    port = domain_port(product_scope.store, scope, "policy")
    policy_source = case.model_input.policy_document.source_id
    with product_scope.store.transaction() as connection:
        rows = list(
            connection.execute(
                text(
                    "SELECT document_id, document_version, tenant_id, ownership FROM"
                    " review_document_registry WHERE source_id = :s ORDER BY document_id"
                ),
                {"s": policy_source},
            )
        )
    registered = {row[0]: (row[1], row[2], row[3]) for row in rows}
    superseded = f"DOC-{policy_source}-SUPERSEDED"
    namesake = f"DOC-{policy_source}-OTHER-TENANT"
    # The current Case copy, its superseded version and the cross-tenant namesake
    # all exist in the registry.
    assert registered[f"DOC-{policy_source}"] == (
        case.model_input.policy_document.version,
        G_TENANT,
        "case_document",
    )
    assert registered[superseded][0] == G_SUPERSEDED_VERSION
    assert registered[namesake][1] == G_OTHER_TENANT
    with product_scope.store.transaction() as connection:
        excluded = {
            row[0]
            for row in connection.execute(
                text(
                    "SELECT chunk_payload->>'chunk_id' FROM review_document_registry"
                    " WHERE document_id IN (:a, :b)"
                ),
                {"a": superseded, "b": namesake},
            )
        }
    # Neither near copy is a candidate at all: the filter runs before the query, and
    # the domain narrowing keeps only the authorized policy material.
    assert not (excluded & set(port.allowed_chunk_ids))
    assert port.telemetry.scope_documents == len(scope.case_document_ids) + len(
        scope.shared_document_ids
    )
    assert port.telemetry.retrieval_documents < port.telemetry.scope_documents
    assert scope.index_version == G_INDEX_VERSION
    assert scope.document_version == case.model_input.policy_document.version


def test_an_absent_subject_is_not_answered_with_an_arbitrary_document(product_scope):
    case = product_scope.measured[0]
    scope = _scope(product_scope, case.case_id)
    probe = negative_probe(domain_port(product_scope.store, scope, "policy"), scope)
    assert probe["status"] == "unknown"
    assert probe["issue"] == "no_match"
    assert probe["results"] == 0
