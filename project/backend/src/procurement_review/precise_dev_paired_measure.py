"""The paired diagnostic's attribution half: what each arm's own material decided.

Every helper here reads only the two arms' own bounded projections and the material each
arm was actually given; nothing is backfilled from the Oracle arm into the actual one.
"""

from __future__ import annotations

from typing import Any

from .business_v3_types import BusinessEvidenceIdentity
from .business_v3_support import SourceCatalogV3
from .eval_f_context import located_sources
from .gold_v37_conditional import ConditionalCaseV1, as_gold_case
from .model_wire_v2 import SourceSlice
from .precise_dev_paired_wire import PairedProjectionV2
from .precise_run_scope import PRECISE_QUESTIONS, RETRIEVAL_LIMIT
from .retrieval import RetrievalRequest
from .review_store import ReviewStore

#: The bounded attribution of one conditional duty in the actual arm.
ATTRIBUTIONS = (
    "both_correct",
    "retrieval_loss",
    "assembly_loss",
    "judgment_loss",
    "oracle_only_error",
    "both_wrong",
    "unmeasured",
)


def _actual_sources(store: ReviewStore, scope: Any, worker: str) -> tuple[tuple, tuple]:
    """What the product's own retrieval returns for this branch's own question.

    The port is the same scope-bound port V-G-01 measured: the Case's own authorized
    material first, then the distractor library, ranked by the real lexical computation
    and narrowed to this branch's category before any query.
    """

    from .scoped_runtime import ScopedRetrievalPort

    port = ScopedRetrievalPort.from_store(
        store=store,
        scope=scope,
        vector_mode="none",
        lexical_adapter="idf",
        category=worker,
    )
    response = port.retrieve(
        RetrievalRequest(
            query=PRECISE_QUESTIONS[worker],
            index_version=scope.index_version,
            limit=RETRIEVAL_LIMIT,
            minimum_results=1,
        )
    )
    return tuple(response.results), port.allowed_chunk_ids


def _own_rank(hits: tuple, checksum: str) -> int:
    for hit in hits:
        if hit.content_checksum.removeprefix("sha256:") == checksum:
            return hit.rank
    return -1


def _slices(scope_id: str, worker: str, hits: tuple) -> tuple[SourceSlice, ...]:
    return tuple(
        SourceSlice(
            scope_id=scope_id,
            worker=worker,
            identity=BusinessEvidenceIdentity(
                source_id=hit.source_id,
                version=hit.source_version,
                locator=hit.chunk_id,
                content_sha256=hit.content_checksum.removeprefix("sha256:"),
            ),
            content=hit.content,
        )
        for hit in hits
    )


def _oracle_sources(
    case: ConditionalCaseV1, worker: str, scope_id: str
) -> tuple[SourceSlice, ...]:
    """The Case's own authored documents, under this execution's own scope identity."""

    gold = as_gold_case(case)
    return tuple(
        SourceSlice(
            scope_id=scope_id,
            worker=item.source.worker,
            identity=item.source.identity,
            content=item.source.content,
        )
        for item in located_sources(gold)
        if item.source.worker == worker
    )


def _catalog(context: Any, worker: str, sources: tuple[SourceSlice, ...]) -> Any:
    from .business_v3_support import LocatedSourceV3

    return SourceCatalogV3(
        tuple(
            LocatedSourceV3(
                source=item,
                subject_type="purchase_order",
                subject_id=context.purchase_order_id,
                effective_at=context.as_of,
            )
            for item in sources
        )
    )


#: Which branch's own document each conditional duty's fields are written in.  A duty
#: whose document was not retrieved cannot be answered from the actual arm, whatever the
#: model does with what it did receive.
DUTY_WORKERS = {
    "O-04": ("policy", "contract"),
    "O-08": ("contract",),
    "O-01": ("policy",),
    "O-02": ("policy",),
    "O-05": ("contract",),
    "O-06": ("contract",),
    "O-07": ("policy",),
    "O-09": ("contract",),
}


def attribution_for_case(
    *,
    truth: dict[str, str],
    oracle_states: dict[str, str],
    actual_states: dict[str, str],
    own_ranks: dict[str, int | None],
    actual_grounded: bool,
) -> dict[str, str]:
    """Attribute one conditional duty between the two arms."""

    result: dict[str, str] = {}
    for duty, expected in truth.items():
        oracle_ok = oracle_states.get(duty) == expected
        actual_ok = actual_states.get(duty) == expected
        missing = any(
            (own_ranks.get(worker) or -1) == -1
            for worker in DUTY_WORKERS.get(duty, ("policy", "contract"))
        )
        if oracle_ok and actual_ok:
            result[duty] = "both_correct"
        elif oracle_ok and not actual_ok:
            if missing:
                result[duty] = "retrieval_loss"
            elif not actual_grounded:
                result[duty] = "assembly_loss"
            else:
                result[duty] = "judgment_loss"
        elif not oracle_ok and actual_ok:
            result[duty] = "oracle_only_error"
        else:
            result[duty] = "both_wrong"
    return result


def _branch_result(projection: PairedProjectionV2) -> dict:
    return {
        "state": projection.duties,
        "json_valid": projection.json_valid,
        "fields_valid": projection.fields_valid,
        "obligations_complete": projection.obligations_complete,
        "semantic_valid": projection.semantic_valid,
        "citations_valid": projection.citations_valid,
        "first_pass_valid": projection.first_pass_valid,
        "corrected_valid": projection.corrected_valid,
        "qualified": projection.qualified,
        "committed": projection.committed,
        "pipeline_failure": projection.pipeline_failure,
        "model_request_count": projection.model_request_count,
        "latency_ms": projection.latency_ms,
        "witnesses": projection.witnesses,
        "envelope_version": projection.envelope_version,
        "format_diagnostics": list(projection.format_diagnostics),
        "grounding_failures": list(projection.grounding_failures),
        "grounding_locations": list(projection.grounding_locations),
        "correction_kinds": list(projection.correction_kinds),
        "own_document_rank": projection.own_document_rank,
    }
