"""What one precise execution is assembled from: questions, scope-bound retrieval, wire
receipts and the frozen execution snapshot.

The runner's loop is about *when* something is allowed to happen; this module owns *what*
it is allowed to happen to — the two branch questions, the retrieval that never leaves the
Worker's own authorized category, the product-typed receipt an attempt carries, and the
snapshot that freezes mode, model, parameters, digests, versions, adapters, budget and
deadline for every generation of the Run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from .business_v3_support import LocatedSourceV3, SourceCatalogV3
from .business_v3_types import BusinessEvidenceIdentity
from .business_v31_documents import document_request_v31
from .checkpointing import IncompatibleCheckpointError, ReadAdmission
from .contracts import WorkerKind
from .eval_f_contracts import FEndpointConfig
from .model_wire_v2 import PROMPT_VERSION, WIRE_VERSION, SourceSlice
from .precise_run_contracts import (
    PRECISE_RUN_CONTRACT,
    PreciseRunSnapshotV1,
    snapshot_mismatch,
)
from .precise_dependencies import (
    PRODUCT_ROOTS,
    MissingDependency,
    module_path,
    package_closure,
)
from .precise_run_state import snapshot_from_isolation
from .precise_run_transport import precise_wire_fingerprint
from .retrieval import RetrievalRequest
from .worker_contracts import (
    EvidenceContext,
    ProcurementWorkerInput,
    WorkerModelRequest,
)

RETRIEVAL_LIMIT = 5
RETRIEVAL_VECTOR_ADAPTER = "no-vector-v1"
RETRIEVAL_LEXICAL_ADAPTER = "deterministic-lexical-idf-v1"
PRECISE_QUESTIONS = {
    "policy": "请按政策分支职责提取本 Case 适用的规则、适用范围、条件与例外，并逐项给出原文依据。",
    "contract": "请按合同分支职责提取本 Case 的合同条款、交易类别与承诺状态，并逐项给出原文依据。",
}

#: The G-side implementation one execution's snapshot binds.  The list is not written
#: here: it is the import closure of the product execution entry points, taken from the
#: one authoritative inventory the candidate fingerprint also uses, so the aggregator, the
#: summary projection, the grounding, the guard or any newly imported dependency is inside
#: the boundary instead of being missed by a hand-kept list.
SNAPSHOT_IMPLEMENTATION_FILES = package_closure(PRODUCT_ROOTS)


@dataclass(slots=True)
class BranchInputs:
    """Everything one branch is executed with, all of it inside its own authorization."""

    worker: str
    hits: tuple
    slices: tuple
    request: Any
    catalog: SourceCatalogV3


def implementation_fingerprint() -> str:
    """The digest of the G-side implementation the snapshot binds.

    A required module that cannot be read is an error: a build whose implementation is
    partly missing is not the same build with fewer files.
    """

    digests = {}
    for name in SNAPSHOT_IMPLEMENTATION_FILES:
        path = module_path(name)
        if path is None or not path.is_file():
            raise MissingDependency(f"execution dependency {name} is absent")
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()


def execution_snapshot(
    *,
    mode: Literal["test_only", "real"],
    config: FEndpointConfig,
    registration: Any,
    branch_timeout_seconds: float,
    max_attempts_per_branch: int,
    generation_deadline_seconds: float,
) -> PreciseRunSnapshotV1:
    """The execution configuration a Run created now would be frozen with."""

    return PreciseRunSnapshotV1(
        mode=mode,
        requested_model=config.requested_model,
        resolved_model=config.resolved_model,
        response_mode=config.mode,
        prompt_style=config.prompt_style,
        thinking=config.thinking,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        max_tokens=config.max_tokens,
        single_attempt_seconds=config.single_attempt_seconds,
        branch_timeout_seconds=branch_timeout_seconds,
        max_attempts_per_branch=max_attempts_per_branch,
        generation_deadline_seconds=generation_deadline_seconds,
        rules_version=registration.rules_version,
        document_version=registration.document_version,
        dataset_version=registration.dataset_version,
        index_version=registration.index_version,
        retrieval_vector_adapter=RETRIEVAL_VECTOR_ADAPTER,
        retrieval_lexical_adapter=RETRIEVAL_LEXICAL_ADAPTER,
        wire_version=WIRE_VERSION,
        prompt_version=PROMPT_VERSION,
        schema_sha256=precise_wire_fingerprint(),
        implementation_sha256=implementation_fingerprint(),
    )


def require_frozen_snapshot(
    run: Any, current: PreciseRunSnapshotV1
) -> PreciseRunSnapshotV1:
    """Stop the execution unless the runtime still matches the Run's frozen snapshot.

    The Run was created under one mode, model, parameter set, wire/prompt/schema, rule and
    material versions, retrieval adapters, budget and deadline.  A resume reuses exactly
    that; a runtime that no longer matches it stops here with the dimension that moved
    instead of silently re-running the Case under new settings.
    """

    frozen = snapshot_from_isolation(getattr(run.header, "isolation", None))
    reason = snapshot_mismatch(frozen, current)
    if reason is not None:
        raise IncompatibleCheckpointError(f"execution snapshot mismatch: {reason}")
    assert frozen is not None
    generation = getattr(run, "generation", None)
    if generation is not None and generation.created_at is not None:
        window = (generation.deadline_at - generation.created_at).total_seconds()
        if abs(window - frozen.generation_deadline_seconds) > 0.05:
            raise IncompatibleCheckpointError(
                "execution snapshot mismatch: execution_deadline"
            )
    return frozen


def wire_request(*, worker: str, request: Any, hits: tuple) -> WorkerModelRequest:
    """The product-typed receipt for one branch: the authorized hits, nothing else."""

    return WorkerModelRequest(
        worker=WorkerKind(worker),
        case_id=request.case_id,
        model_fixture_id=f"precise-{PRECISE_RUN_CONTRACT}",
        question=PRECISE_QUESTIONS[worker],
        evidence_contexts=tuple(
            EvidenceContext(evidence_ref=hit.evidence_ref, content=hit.content)
            for hit in hits
        ),
    )


def retrieve_for(
    store: Any, scope: Any, admission: ReadAdmission, worker: str
) -> tuple:
    """One worker reaches only its own authorized material category.

    The narrowing happens inside the port, before any query, exactly as the retrieval gate
    was measured: a document Worker never sees the other domain's candidates, and the scope
    and version predicates run before ranking.
    """

    from .scoped_runtime import ScopedRetrievalPort

    port = ScopedRetrievalPort.from_store(
        store=store,
        scope=scope,
        admission=admission,
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
    return tuple(response.results)


def branch_inputs(
    *,
    context: Any,
    case_id: str,
    store: Any,
    scope: Any,
    admission: ReadAdmission,
    runtime: Any,
) -> dict[str, BranchInputs]:
    """The two document branches' authorized inputs, assembled per Worker.

    Retrieval, the branch request and the source catalog are built for one Worker at a
    time: the other domain's material never enters the request, and every hit is verified
    against the runtime's own evidence boundary before it is used.
    """

    inputs: dict[str, BranchInputs] = {}
    for worker in ("policy", "contract"):
        hits = retrieve_for(store, scope, admission, worker)
        for hit in hits:
            runtime.evidence.verify_evidence_ref(hit.evidence_ref)
        slices = tuple(
            SourceSlice(
                scope_id=context.scope_id,
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
        inputs[worker] = BranchInputs(
            worker=worker,
            hits=hits,
            slices=slices,
            request=document_request_v31(
                context,
                case_id=case_id,
                worker=worker,
                question=PRECISE_QUESTIONS[worker],
                sources=slices,
            ),
            catalog=SourceCatalogV3(
                tuple(
                    LocatedSourceV3(
                        source=item,
                        subject_type="purchase_order",
                        subject_id=context.purchase_order_id,
                        effective_at=context.as_of,
                    )
                    for item in slices
                )
            ),
        )
    return inputs


def procurement_worker_input(registration: Any, case_id: str) -> ProcurementWorkerInput:
    """The deterministic branch's read-only input; it has no model task."""

    return ProcurementWorkerInput(
        case_id=case_id,
        purchase_order_id=registration.purchase_order_id,
        dataset_version=registration.dataset_version,
        model_fixture_id="precise-deterministic-branch",
    )


__all__ = [
    "PRECISE_QUESTIONS",
    "RETRIEVAL_LIMIT",
    "RETRIEVAL_LEXICAL_ADAPTER",
    "RETRIEVAL_VECTOR_ADAPTER",
    "SNAPSHOT_IMPLEMENTATION_FILES",
    "BranchInputs",
    "branch_inputs",
    "execution_snapshot",
    "implementation_fingerprint",
    "procurement_worker_input",
    "require_frozen_snapshot",
    "retrieve_for",
    "wire_request",
]
