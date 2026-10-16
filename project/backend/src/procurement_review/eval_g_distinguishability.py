"""Material distinguishability for the de-identified retrieval variant.

Whether a de-identified query can single out its own evidence is a property of the
authorized candidate set, not of the retrieval scores: two criteria are mechanical and
label-free — another candidate with the same bytes, or another with the same text once
identifiers, dates and numbers are masked — and a miss that neither explains is reported as
unexplained instead of being attributed to ranking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .eval_g_retrieval import TOP_K, deidentified_signature, query_for, required_for
from .retrieval import RetrievalRequest
from .security_contracts import RuntimeScope


def indistinguishable_identities(
    measured: Sequence[Any], others: Sequence[Any]
) -> dict[str, dict[str, Any]]:
    """Required evidence no de-identified retrieval can single out.

    Two criteria, both mechanical and both independent of any label: another
    authorized candidate with the same bytes, or another with the same text once
    every identifier, date and number is masked.  Such an identity is reported
    beside the recall instead of being folded into a ranking defect or quietly
    dropped from a denominator.
    """

    by_bytes: dict[str, list[str]] = {}
    by_signature: dict[str, list[str]] = {}
    for case in (*measured, *others):
        for document in (
            case.model_input.policy_document,
            case.model_input.contract_document,
        ):
            by_bytes.setdefault(document.content_sha256, []).append(document.source_id)
            by_signature.setdefault(
                deidentified_signature(document.content), []
            ).append(document.source_id)
    result: dict[str, dict[str, Any]] = {}
    for case in measured:
        for document in (
            case.model_input.policy_document,
            case.model_input.contract_document,
        ):
            identical = sorted(by_bytes[document.content_sha256])
            same_form = sorted(by_signature[deidentified_signature(document.content)])
            if len(identical) > 1 or len(same_form) > 1:
                result[document.source_id] = {
                    "byte_identical": tuple(identical),
                    "same_deidentified_form": tuple(same_form),
                    "criterion": (
                        "identical_bytes" if len(identical) > 1 else "identical_form"
                    ),
                }
    return result


@dataclass(frozen=True, slots=True)
class Distinguishability:
    """Whether the actual query can single one required identity out of its peers.

    ``byte_identical`` peers cannot be separated by any content-based ranking; that
    is a property of the material.  A peer that merely shares the masked form is not
    proof of the same thing, so the criterion for it is measured: the real query is
    run over every authorized candidate and the target has to rank above each peer.
    """

    case_id: str
    evidence_domain: str
    query_variant: str
    source_id: str
    required: tuple[str, str, str]
    target_rank: int | None
    gate_hit: bool
    byte_identical: tuple[str, ...]
    same_deidentified_form: tuple[str, ...]
    peer_ranks: tuple[tuple[str, int | None], ...]
    criterion: str
    distinguishable: bool | None

    @property
    def unexplained(self) -> bool:
        """A miss no material or query property accounts for."""

        return not self.gate_hit and self.distinguishable is True


def distinguishability_rows(
    port: Any,
    *,
    case: Any,
    truth: Any,
    domain: str,
    variant: str,
    scope: RuntimeScope,
    corpus: Sequence[Any],
) -> tuple[Distinguishability, ...]:
    """Run the actual query once over the whole authorized corpus and judge each duty."""

    documents = [
        document
        for item in corpus
        for document in (
            item.model_input.policy_document,
            item.model_input.contract_document,
        )
    ]
    by_signature: dict[str, list[Any]] = {}
    for document in documents:
        by_signature.setdefault(deidentified_signature(document.content), []).append(
            document
        )
    target_documents = {
        "policy": case.model_input.policy_document,
        "contract": case.model_input.contract_document,
    }
    target = target_documents[domain]
    peers = [
        document
        for document in by_signature[deidentified_signature(target.content)]
        if document.source_id != target.source_id
    ]
    request = RetrievalRequest(
        query=query_for(case, domain, variant),
        index_version=scope.index_version,
        # The contract caps one request at 50 results, which is the widest single
        # observation available; a peer ranked beyond it is below every returned hit.
        limit=min(max(1, len(port.allowed_chunk_ids)), 50),
        minimum_results=1,
    )
    response = port.retrieve(request)
    rank_by_source: dict[str, int] = {}
    for hit in response.results:
        rank_by_source.setdefault(hit.source_id, hit.rank)
    identical = tuple(
        sorted(
            document.source_id
            for document in peers
            if document.content_sha256 == target.content_sha256
        )
    )
    peer_ranks = tuple(
        (document.source_id, rank_by_source.get(document.source_id))
        for document in sorted(peers, key=lambda item: item.source_id)
    )
    rows = []
    for identity in required_for(truth, domain):
        rank = rank_by_source.get(identity[0])
        gate_hit = rank is not None and rank <= TOP_K
        if not peers:
            criterion, distinguishable = "unique", None
        elif identical:
            criterion, distinguishable = "identical_bytes", False
        else:
            separated = rank is not None and all(
                peer_rank is None or peer_rank > rank for _name, peer_rank in peer_ranks
            )
            criterion = "query_separates" if separated else "query_does_not_separate"
            distinguishable = separated
        rows.append(
            Distinguishability(
                case_id=case.case_id,
                evidence_domain=domain,
                query_variant=variant,
                source_id=identity[0],
                required=identity,
                target_rank=rank,
                gate_hit=gate_hit,
                byte_identical=identical,
                same_deidentified_form=tuple(
                    sorted(document.source_id for document in peers)
                ),
                peer_ranks=peer_ranks,
                criterion=criterion,
                distinguishable=distinguishable,
            )
        )
    return tuple(rows)


def unexplained_misses(
    rows: Sequence[Distinguishability],
) -> tuple[Distinguishability, ...]:
    """Misses with no byte-identical peer and a query that did separate the target."""

    return tuple(row for row in rows if row.unexplained)
