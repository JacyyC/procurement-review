"""Offline G comparison fixtures.

The multi-agent projections here come from the real frozen F pipeline driven by a
Mock transport, so the arm-agnostic scorer is exercised on a genuine precise
decision.  The joint-arm projection is a merged *shape* fixture: it proves the
scorer no longer hardcodes two branches, and it is never quality evidence for the
single-agent arm, whose real measurements belong to H.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from procurement_review.business_v3_documents import document_request_v3
from procurement_review.business_v3_evidence import (
    EvidenceEnvelopeV3,
    ground_document_v3,
    parse_evidence_v3,
)
from procurement_review.business_v3_procurement import review_procurement_v3
from procurement_review.business_v3_review import finalize_review_v3
from procurement_review.business_v3_security import ProtectedOutputV3
from procurement_review.business_v3_support import SourceCatalogV3
from procurement_review.eval_attempt_ledger import AttemptLedger
from procurement_review.eval_f_context import (
    context_for,
    load_cases,
    located_sources,
    oracle_procurement,
)
from procurement_review.eval_f_contracts import FBudgetProof
from procurement_review.eval_f_registration import configuration
from procurement_review.eval_f_run import FAuthority
from procurement_review.eval_f_transport import (
    PreciseAttemptDriver,
    PreciseChatTransport,
)
from procurement_review.eval_g_contracts import (
    ArmResources,
    ClaimCollectionStatus,
    ComparisonArm,
    ComparisonObservationV1,
    TerminalState,
)
from procurement_review.eval_g_plan import arm_shapes, fairness, plan
from procurement_review.eval_g_projection import (
    GBranchProjection,
    GCaseProjection,
    GCaseTiming,
)
from procurement_review.eval_f_context import stable_write
from procurement_review.gold_v3_contracts import V3Recommendation, payload_sha256
from procurement_review.model_attempt_v2 import configuration_hash
from v1_07f_support import response_for

FAIRNESS_FINGERPRINTS = {
    "materials_sha256": payload_sha256({"materials": "g-fixture"}),
    "fact_tools_sha256": payload_sha256({"fact_tools": "g-fixture"}),
    "retrieval_index_sha256": payload_sha256({"index": "g-fixture"}),
}
SEEDS = (601, 602)


def frozen_plan(case_ids: Sequence[str] | None = None, *, split: str = "dev"):
    """One frozen two-arm plan over real registered DEV cases."""

    registered = case_ids
    if registered is None:
        registered = [case.case_id for case in load_cases(split, "gold-set.v3.1")][:2]
    return plan(
        plan_id="v1-07g-fixture-plan",
        stage_id="V1-07G",
        split=split,
        gold_set_version="gold-set.v3.1",
        fairness_contract=fairness(
            truth_contract="v1-07-truth.v1.1", **FAIRNESS_FINGERPRINTS
        ),
        case_ids=registered,
        repeat_seeds=SEEDS,
    )


def _driver(tmp_path: Path, case: Any, seed: int) -> PreciseAttemptDriver:
    root = tmp_path / ".local" / "g-fixture" / f"{case.case_id}.{seed}"
    root.mkdir(parents=True, exist_ok=True)
    execution = root / "execution.json"
    if not execution.exists():
        stable_write(execution, dict(generation=1, cancelled=False))
    located = located_sources(case)
    proof = FBudgetProof(
        evidence_sha256="a" * 64,
        cumulative_request_upper=7871,
        cumulative_input_upper=None,
        cumulative_output_upper=None,
        historical_unknown_input_reserve=None,
        historical_unknown_output_reserve=None,
        allocated_requests=32,
        allocated_input=None,
        allocated_output=None,
        limit_policy="user_no_balance_limit",
        authorization_basis="user-explicit-no-balance-limit",
    )
    authority = FAuthority(
        authorization_id="test-g",
        scope_ids=frozenset([context_for(case).scope_id]),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        proof=proof,
        execution_path=execution,
        generation=1,
    )
    transport = PreciseChatTransport(
        configuration(),
        SecretStr("offline-test-key"),
        protection=ProtectedOutputV3(),
        catalog=SourceCatalogV3(located),
        transport=httpx.MockTransport(lambda req: response_for(req)),
    )
    ledger_path = root / "ledger.sqlite3"
    ledger = (
        AttemptLedger(ledger_path)
        if ledger_path.exists()
        else AttemptLedger.create(
            ledger_path,
            run_id="v1-07-g-test",
            gold_manifest_sha256="b" * 64,
            configuration_sha256=configuration_hash(transport, authority),
            max_requests=32,
            max_input_tokens=None,
            max_output_tokens=None,
            stage_id="V1-07F",
            gold_set_version="gold-set.v3.1",
        )
    )
    return PreciseAttemptDriver(ledger, transport, authority)


def real_multi_agent_case(tmp_path: Path, case: Any, *, seed: int = 601):
    """Run the frozen F pipeline offline and return one real precise decision."""

    driver = _driver(tmp_path, case, seed)
    context = context_for(case)
    located = located_sources(case)
    sources = tuple(item.source for item in located)
    catalog = SourceCatalogV3(located)
    protection = ProtectedOutputV3()
    procurement = review_procurement_v3(
        context, oracle_procurement(case), authorized_scope_id=context.scope_id
    )
    grounded = []
    branches = []
    for worker in ("policy", "contract"):
        request = document_request_v3(
            context,
            case_id=case.case_id,
            question=case.model_input.question,
            worker=worker,
            sources=tuple(item for item in sources if item.worker == worker),
        )
        branch = asyncio.run(
            driver.run(
                f"{case.case_id}.{seed}.{worker}",
                request,
                deadline=datetime.now(UTC) + timedelta(seconds=120),
                reserve_input=100,
                reserve_output=200,
            )
        )
        branches.append(branch)
        document, witnesses = parse_evidence_v3(
            branch.evidence_output.model_dump_json(), request, protection=protection
        )
        grounded.append(ground_document_v3(request, document, witnesses, catalog))
    decision = finalize_review_v3(
        context,
        *grounded,
        procurement,
        sources=sources,
        protection=protection,
        pipeline_failures=(),
        control_events=tuple(
            event for branch in branches for event in branch.security_events
        ),
    )
    attempts = _attempts(driver.ledger, case.case_id, seed)
    return decision, branches, attempts


def _attempts(ledger: AttemptLedger, case_id: str, seed: int) -> list[dict[str, Any]]:
    import sqlite3

    with sqlite3.connect(
        ledger.path.resolve().as_uri() + "?mode=ro", uri=True
    ) as connection:
        connection.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in connection.execute(
                "SELECT task_key, attempt_no, receipt_json FROM attempts WHERE task_key"
                " LIKE ? ORDER BY attempt_id",
                (f"{case_id}.{seed}.%",),
            )
        ]


def g_branch(
    precise: Any, *, arm: ComparisonArm, worker: str | None = None
) -> GBranchProjection:
    body = precise.model_dump(mode="json")
    body.update(
        schema_version="eval-projection.v3",
        projection_kind="g_branch",
        arm=arm.value,
        worker=worker or body["worker"],
    )
    return GBranchProjection.model_validate_json(json.dumps(body, ensure_ascii=False))


def multi_agent_projection(
    case: Any,
    decision: Any,
    branches: Sequence[Any],
    *,
    plan_sha256: str,
    seed: int = 601,
    total_ms: int = 1_500,
) -> GCaseProjection:
    return GCaseProjection(
        arm=ComparisonArm.MULTI_AGENT,
        case_id=case.case_id,
        repeat_seed=seed,
        plan_sha256=plan_sha256,
        configuration_sha256=payload_sha256({"candidate": "g-fixture"}),
        decision=decision,
        branches=tuple(
            g_branch(branch, arm=ComparisonArm.MULTI_AGENT) for branch in branches
        ),
        timing=GCaseTiming(
            queue_ms=10,
            context_ms=5,
            retrieval_ms=20,
            mcp_ms=8,
            model_ms=1_200,
            extra_attempt_ms=0,
            aggregate_ms=15,
            total_ms=total_ms,
            scope="product_workbench",
        ),
    )


def joint_projection(
    projection: GCaseProjection, *, plan_sha256: str
) -> GCaseProjection:
    """Merge the two isolated branches into one joint branch of the same decision."""

    left, right = projection.branches
    envelope = EvidenceEnvelopeV3(
        schema_version="finding-evidence.v1",
        findings=(*left.evidence_output.findings, *right.evidence_output.findings),
        witnesses=(*left.evidence_output.witnesses, *right.evidence_output.witnesses),
    )
    joint = GBranchProjection(
        arm=ComparisonArm.SINGLE_AGENT,
        case_id=projection.case_id,
        worker="joint",
        claims=(*left.claims, *right.claims),
        details=(*left.details, *right.details),
        first_pass_valid=left.first_pass_valid and right.first_pass_valid,
        received_nonempty_body=left.received_nonempty_body
        and right.received_nonempty_body,
        corrected_valid=left.corrected_valid or right.corrected_valid,
        pipeline_failure=left.pipeline_failure or right.pipeline_failure,
        model_request_count=1,
        latency_ms=left.latency_ms + right.latency_ms,
        evidence_output=envelope,
        security_events=(*left.security_events, *right.security_events),
        rejected_witnesses=(*left.rejected_witnesses, *right.rejected_witnesses),
        first_received_nonempty_body=left.first_received_nonempty_body,
    )
    return GCaseProjection(
        arm=ComparisonArm.SINGLE_AGENT,
        case_id=projection.case_id,
        repeat_seed=projection.repeat_seed,
        plan_sha256=plan_sha256,
        configuration_sha256=projection.configuration_sha256,
        decision=projection.decision,
        branches=(joint,),
        timing=projection.timing.model_copy(
            update={"scope": "single_agent_runner", "model_ms": joint.latency_ms}
        ),
    )


def observation(
    *,
    arm: ComparisonArm,
    case_id: str,
    repeat_seed: int,
    plan_sha256: str,
    recommendation: str | None = "recommend_approve",
    terminal_state: TerminalState = TerminalState.COMMITTED,
    first_send_count: int = 1,
    first_pass_valid_count: int = 1,
    corrected_valid_count: int = 0,
    physical_request_count: int = 1,
    known_input_tokens: int | None = 100,
    known_output_tokens: int | None = 200,
    usage_unknown_attempts: int = 0,
    total_latency_ms: int | None = 1_500,
    claim_count: int = 12,
    pipeline_failure: bool = False,
    stop_reason: str | None = None,
) -> ComparisonObservationV1:
    committed = terminal_state is TerminalState.COMMITTED
    digest = payload_sha256({"case": case_id, "seed": repeat_seed, "arm": arm.value})
    return ComparisonObservationV1(
        arm=arm,
        case_id=case_id,
        repeat_seed=repeat_seed,
        plan_sha256=plan_sha256,
        terminal_state=terminal_state,
        recommendation=(
            V3Recommendation(recommendation) if committed and recommendation else None
        ),
        decision_sha256=digest if committed else None,
        claim_collection_status=(
            ClaimCollectionStatus.COMPLETE
            if committed
            else ClaimCollectionStatus.ABSENT
        ),
        claim_count=claim_count if committed else 0,
        claim_digest_sha256=digest if committed and claim_count else None,
        first_send_count=first_send_count,
        first_pass_valid_count=first_pass_valid_count if committed else 0,
        corrected_valid_count=corrected_valid_count if committed else 0,
        physical_request_count=physical_request_count,
        known_input_tokens=known_input_tokens,
        known_output_tokens=known_output_tokens,
        usage_unknown_attempts=usage_unknown_attempts,
        total_latency_ms=total_latency_ms if committed else None,
        pipeline_failure=pipeline_failure,
        stop_reason=stop_reason or (None if committed else "document_model_failed"),
    )


def resources(
    *,
    arm: ComparisonArm,
    namespace: str,
    physical_requests: int,
    known_input_tokens: int,
    known_output_tokens: int,
    usage_unknown_attempts: int = 0,
    unreturned_attempts: int = 0,
    cost_status: str = "unknown",
    cost_value: float | None = None,
    p95_total_latency_ms: int | None = 1_500,
) -> ArmResources:
    return ArmResources(
        arm_id=arm,
        namespace=namespace,
        physical_requests=physical_requests,
        receipts=physical_requests - unreturned_attempts,
        additional_attempts=0,
        known_input_tokens=known_input_tokens,
        known_output_tokens=known_output_tokens,
        usage_unknown_attempts=usage_unknown_attempts,
        unreturned_attempts=unreturned_attempts,
        cost_status=cost_status,
        cost_value=cost_value,
        p95_total_latency_ms=p95_total_latency_ms,
    )


def shapes() -> dict[ComparisonArm, Any]:
    multi, single = arm_shapes()
    return {ComparisonArm.MULTI_AGENT: multi, ComparisonArm.SINGLE_AGENT: single}


__all__ = [
    "FAIRNESS_FINGERPRINTS",
    "SEEDS",
    "frozen_plan",
    "g_branch",
    "joint_projection",
    "multi_agent_projection",
    "observation",
    "real_multi_agent_case",
    "resources",
    "shapes",
    "persist_batch",
    "write_batch_inputs",
]


def write_batch_inputs(
    folder: Path,
    plan: Any,
    *,
    projections: dict[Any, Sequence[GCaseProjection]],
    attempts: dict[Any, Sequence[dict[str, Any]]],
) -> Path:
    """Lay out the durable inputs an independent auditor must be able to read back."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stable_write(folder / "plan.json", plan.model_dump(mode="json"))
    for arm in (ComparisonArm.MULTI_AGENT, ComparisonArm.SINGLE_AGENT):
        arm_dir = folder / arm.value
        arm_dir.mkdir(parents=True, exist_ok=True)
        for projection in projections.get(arm, []):
            stable_write(
                arm_dir / f"{projection.case_id}.{projection.repeat_seed}.json",
                projection.model_dump(mode="json"),
            )
        stable_write(arm_dir / "attempts.json", list(attempts.get(arm, [])))
    return folder


def persist_batch(
    folder: Path,
    plan: Any,
    observations: Sequence[ComparisonObservationV1],
    resources_map: dict[Any, ArmResources],
    report: dict[str, Any],
    *,
    observed_fairness: dict[str, Any] | None = None,
) -> None:
    """Persist the settled observations, metering and report of one batch."""

    from procurement_review.eval_g_report import write_observation

    folder = Path(folder)
    for item in observations:
        write_observation(folder / "observations", item)
    if observed_fairness is not None:
        stable_write(folder / "fairness-observed.json", observed_fairness)
    stable_write(
        folder / "resources.json",
        {
            arm.value: {"cost_status": item.cost_status, "cost_value": item.cost_value}
            for arm, item in resources_map.items()
        },
    )
    stable_write(folder / "result.json", report)
