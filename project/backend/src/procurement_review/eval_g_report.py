"""Report, durable observation store and metering readers for the G/H comparison.

``eval-report.v4`` stays frozen for the A～F history.  This module writes
``eval-report.v6`` with strict per-observation settlement; v5 remains readable.
The per-arm absolute product gates and the neutral Q-12
completeness judgement are separate top-level results, and every denominator is
read back from the frozen plan.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .attempt_metering import classify, facts_from_receipts
from .eval_g_comparison import score_comparison
from .eval_g_contracts import (
    COMPARISON_REPORT_V5,
    ArmPlan,
    ArmResources,
    ArmShape,
    ClaimCollectionStatus,
    ComparisonArm,
    ComparisonObservationV1,
    ComparisonPlanV1,
    TerminalState,
)
from .eval_g_projection import GCaseProjection, obligations_of
from .gold_v3_contracts import V3Recommendation, payload_sha256

OBSERVATION_GLOB = "observation-*.json"


def observation_path(directory: Path, observation: ComparisonObservationV1) -> Path:
    return Path(directory) / (
        f"observation-{observation.arm.value}-{observation.case_id}-"
        f"{observation.repeat_seed}.json"
    )


def write_observation(directory: Path, observation: ComparisonObservationV1) -> Path:
    """Persist one terminal observation atomically so recovery can read it back."""

    from .eval_attempt_ledger import _atomic_write

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = observation_path(directory, observation)
    _atomic_write(
        path,
        json.dumps(
            observation.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    return path


def load_observations(directory: Path) -> tuple[ComparisonObservationV1, ...]:
    """Read back every terminal observation this batch already settled."""

    return tuple(
        ComparisonObservationV1.model_validate_json(path.read_bytes())
        for path in sorted(Path(directory).glob(OBSERVATION_GLOB))
    )


def claim_collection(
    projection: GCaseProjection, shape: ArmShape
) -> tuple[ClaimCollectionStatus, int, str | None]:
    """Bounded claim digest an auditor can read back; never a prompt or a response."""

    findings = projection.decision.business.findings
    expected_claims = sum(
        len(obligations_of(shape, branch.branch_id)) for branch in shape.model_branches
    )
    retained = sum(len(branch.claims) for branch in projection.branches)
    digest = payload_sha256(
        {
            "findings": [
                {
                    "obligation_id": item.obligation_id,
                    "state": item.state,
                    "reason": item.reason,
                    "risk": item.risk,
                    "evidence": sorted(
                        payload_sha256(evidence.model_dump(mode="json"))
                        for evidence in item.evidence
                    ),
                }
                for item in findings
            ],
            "claims": [
                claim.model_dump(mode="json")
                for branch in projection.branches
                for claim in branch.claims
            ],
            "conflict_pairs": [pair.pair_id for pair in projection.decision.conflicts],
            "security_codes": sorted(
                {event.code for event in projection.decision.security}
            ),
        }
    )
    if retained < expected_claims:
        return ClaimCollectionStatus.BLOCKED_BY_PROTECTION, len(findings), digest
    if not findings:
        return ClaimCollectionStatus.ABSENT, 0, None
    return ClaimCollectionStatus.COMPLETE, len(findings), digest


def terminal_state_for(
    projection: GCaseProjection | None, stop_reason: str | None
) -> tuple[TerminalState, str | None]:
    if projection is None:
        return TerminalState.BLOCKED_UNKNOWN, stop_reason or "projection_missing"
    failures = set(projection.decision.business.pipeline_failures)
    if stop_reason:
        mapped = {
            "cancelled": TerminalState.CANCELLED,
            "authorization_revoked": TerminalState.CANCELLED,
            "deadline": TerminalState.FAILED,
        }.get(stop_reason, TerminalState.FAILED)
        return mapped, stop_reason
    if failures & {"transport", "document_model_failed", "evidence_envelope_invalid"}:
        return TerminalState.FAILED, sorted(failures)[0]
    return TerminalState.COMMITTED, None


def observation_from(
    projection: GCaseProjection | None,
    *,
    arm: ComparisonArm,
    shape: ArmShape,
    plan_sha256: str,
    attempts: Sequence[Mapping[str, Any]],
    scored_row: Mapping[str, Any],
    stop_reason: str | None = None,
) -> ComparisonObservationV1:
    """Collect one terminal observation from a projection and its ledger attempts."""

    if projection is None:
        state, reason = terminal_state_for(None, stop_reason)
        return ComparisonObservationV1(
            arm=arm,
            case_id=scored_row["case_id"],
            repeat_seed=scored_row["repeat_seed"],
            plan_sha256=plan_sha256,
            terminal_state=state,
            stop_reason=reason,
            first_send_count=len({item["task_key"] for item in attempts}),
            physical_request_count=len(attempts),
            usage_unknown_attempts=sum(
                int(bool(_receipt(item).get("usage_unknown"))) for item in attempts
            ),
            pipeline_failure=True,
        )
    state, reason = terminal_state_for(projection, stop_reason)
    status, claim_count, digest = claim_collection(projection, shape)
    receipts = [_receipt(item) for item in attempts]
    known = [item for item in receipts if not item.get("usage_unknown")]
    counts = scored_row["counts"]
    first_sends = len({item["task_key"] for item in attempts})
    first_valid = int(counts.get("first_valid", 0))
    final_valid = int(counts.get("final_valid", 0))
    recommendation: V3Recommendation | None = None
    decision_sha256: str | None = None
    latency: int | None = None
    if state is TerminalState.COMMITTED:
        recommendation = V3Recommendation(projection.decision.business.recommendation)
        decision_sha256 = payload_sha256(projection.decision.model_dump(mode="json"))
        latency = projection.timing.total_ms
    return ComparisonObservationV1(
        arm=arm,
        case_id=projection.case_id,
        repeat_seed=projection.repeat_seed,
        plan_sha256=plan_sha256,
        terminal_state=state,
        stop_reason=reason,
        recommendation=recommendation,
        decision_sha256=decision_sha256,
        claim_collection_status=status,
        claim_count=claim_count,
        claim_digest_sha256=digest,
        first_send_count=first_sends,
        first_pass_valid_count=first_valid,
        corrected_valid_count=max(0, final_valid - first_valid),
        physical_request_count=len(attempts),
        known_input_tokens=sum(item.get("input_tokens") or 0 for item in known),
        known_output_tokens=sum(item.get("output_tokens") or 0 for item in known),
        usage_unknown_attempts=len(receipts)
        - len(known)
        + max(0, len(attempts) - len(receipts)),
        total_latency_ms=latency,
        pipeline_failure=bool(counts.get("pipeline_failed", 0)),
    )


def _receipt(attempt: Mapping[str, Any]) -> Mapping[str, Any]:
    body = attempt.get("receipt_json") or attempt.get("receipt")
    if isinstance(body, str):
        return json.loads(body)
    return body or {}


def resources_from_ledger(
    arm: ComparisonArm,
    namespace: str,
    attempts: Sequence[Mapping[str, Any]],
    *,
    cost_status: str = "unknown",
    cost_value: float | None = None,
    p95_total_latency_ms: int | None = None,
) -> ArmResources:
    """Derive one arm's metering from its durable attempts, unknowns preserved.

    A row without a persisted receipt is not a returned attempt and not a free one: the
    send state of an attempt whose receipt never arrived is undecidable from persisted
    facts, so it is counted as unreturned and its usage as unknown.  The same classifier the
    product Run uses keeps the two lines reporting the same distinctions.
    """

    metering = classify(
        facts_from_receipts(
            attempts,
            receipt_of=lambda row: _receipt_or_none(row),
        )
    )
    return ArmResources(
        arm_id=arm,
        namespace=namespace,
        physical_requests=metering.attempts_opened,
        receipts=metering.proven_sends,
        additional_attempts=metering.extra_attempts,
        known_input_tokens=metering.input_tokens_known or 0,
        known_output_tokens=metering.output_tokens_known or 0,
        usage_unknown_attempts=metering.unknown_usage_attempts,
        unreturned_attempts=metering.unresolved_attempts + metering.open_attempts,
        cost_status=cost_status,
        cost_value=cost_value,
        p95_total_latency_ms=p95_total_latency_ms,
    )


def _receipt_or_none(attempt: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The persisted receipt of one attempt, or ``None`` when there is none."""

    body = attempt.get("receipt_json") or attempt.get("receipt")
    if body is None:
        return None
    if isinstance(body, str):
        return json.loads(body)
    return body


def build_report_v5(
    plan: ComparisonPlanV1,
    observations: Sequence[ComparisonObservationV1],
    *,
    expected: Mapping[str, V3Recommendation | str],
    resources: Mapping[ComparisonArm, ArmResources],
    arm_quality: Mapping[ComparisonArm, Mapping[str, Any]],
    observed_fairness: Mapping[str, Any] | None = None,
    near_duplicate_groups: Mapping[str, str] | None = None,
    implementation: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the v5 report; the historical v4 lineage is never rewritten."""

    comparison = score_comparison(
        plan,
        observations,
        expected=expected,
        resources=resources,
        arm_quality=arm_quality,
        observed_fairness=observed_fairness,
        near_duplicate_groups=near_duplicate_groups,
    )
    return {
        "schema_version": COMPARISON_REPORT_V5,
        "metrics_version": comparison["metrics_version"],
        "comparison_contract": comparison["comparison_contract"],
        "plan_id": plan.plan_id,
        "plan_sha256": plan.plan_sha256,
        "stage_id": plan.stage_id,
        "split": plan.split,
        "gold_set_version": plan.gold_set_version,
        "truth_contract": plan.fairness.truth_contract,
        "implementation": dict(implementation or {}),
        "product_quality": comparison["product_quality"],
        "arms": {
            arm.value: {
                k: v for k, v in (arm_quality.get(arm) or {}).items() if k != "rows"
            }
            for arm in ComparisonArm
        },
        "Q-12": comparison["Q-12"],
        "history_metering": {
            "namespace": plan.history.namespace,
            "prior_scopes": [
                item.model_dump(mode="json") for item in plan.history.prior_ledgers
            ],
        },
        "network_calls": 0,
        "stage_acceptance": False,
        "stage_acceptance_reason": "requires_the_formal_preregistered_batch_and_h_gates",
    }


def resources_from_ledger_v3(arm, namespace, attempts, **kwargs):
    """A known usage marker requires both counts; receipts retain full attempt identity."""
    identities = []
    for row in attempts:
        if not isinstance(row, Mapping):
            raise ValueError("attempt_shape")
        identity = (row.get("task_key"), row.get("attempt_no"))
        if (
            not isinstance(identity[0], str)
            or not identity[0]
            or type(identity[1]) is not int
            or identity[1] < 1
            or identity in identities
        ):
            raise ValueError("attempt_identity_missing_or_duplicate")
        identities.append(identity)
        import re

        if not isinstance(row.get("request_sha256"), str) or not re.fullmatch(
            "[0-9a-f]{64}", row["request_sha256"]
        ):
            raise ValueError("request_identity_missing")
        receipt = _receipt_or_none(row)
        if receipt is None:
            if row.get("state") not in {"remote_unknown", "cancelled", "failed"}:
                raise ValueError("unsettled_attempt_without_receipt")
            continue
        if row.get("state") not in {"committed", "failed", "cancelled"}:
            raise ValueError("receipt_attempt_not_settled")
        if not isinstance(receipt, Mapping):
            raise ValueError("receipt_shape")
        if any(
            receipt.get(key) != row.get(key)
            for key in ("task_key", "attempt_no", "request_sha256")
        ):
            raise ValueError("receipt_attempt_identity_mismatch")
        if row.get("receipt_sha256") != payload_sha256(receipt):
            raise ValueError("receipt_digest_mismatch")
        if receipt.get("usage_unknown") is False:
            if any(
                type(receipt.get(key)) is not int or receipt[key] < 0
                for key in ("input_tokens", "output_tokens")
            ):
                raise ValueError("known_token_usage_missing")
        elif receipt.get("usage_unknown") is True:
            if any(
                receipt.get(key) is not None
                for key in ("input_tokens", "output_tokens")
            ):
                raise ValueError("unknown_usage_with_known_tokens")
        else:
            raise ValueError("usage_marker_missing")
    return resources_from_ledger(arm, namespace, attempts, **kwargs)


def build_report_v6(plan, observations, *, attempts_by_arm, **kwargs):
    """Versioned Q-12 with per-task ledger checks; v5 remains a historical reader."""
    if not isinstance(attempts_by_arm, Mapping) or set(attempts_by_arm) - set(plan.arm):
        raise ValueError("q12_ledger_shape:arms")
    for arm, rows in attempts_by_arm.items():
        if not isinstance(rows, (list, tuple)) or any(
            not isinstance(row, Mapping) or not isinstance(row.get("task_key"), str)
            for row in rows
        ):
            raise ValueError(f"q12_ledger_shape:{arm}")
    report = build_report_v5(plan, observations, **kwargs)
    report["schema_version"] = "eval-report.v6"
    report["metrics_version"] = "v1-07-eval-metrics.v3"
    report["comparison_contract"] = "v1-07g-comparison.v2"
    reasons = list(report["Q-12"]["incomplete_reasons"])
    identities = {}
    from .eval_g_plan import planned_task_index

    task_index = planned_task_index(plan)
    for arm, arm_plan in plan.arm.items():
        attempts = attempts_by_arm.get(arm, [])
        planned = set(arm_plan.task_keys())
        actual = {row.get("task_key") for row in attempts}
        errors = []
        if actual != planned:
            errors.append("planned_task_identity")
        for task in planned:
            numbers = [
                row.get("attempt_no") for row in attempts if row.get("task_key") == task
            ]
            if any(type(number) is not int for number in numbers):
                errors.append("attempt_identity_missing_or_duplicate")
                continue
            numbers.sort()
            if numbers != list(range(1, len(numbers) + 1)) or len(numbers) > 2:
                errors.append("cross_generation_attempt_budget")
        try:
            measured = resources_from_ledger_v3(arm, plan.history.namespace, attempts)
            resources = kwargs["resources"][arm]
            for key in (
                "physical_requests",
                "receipts",
                "additional_attempts",
                "known_input_tokens",
                "known_output_tokens",
                "usage_unknown_attempts",
                "unreturned_attempts",
            ):
                if getattr(measured, key) != getattr(resources, key):
                    errors.append(f"ledger_resource:{key}")
            for case, seed in arm_plan.observation_keys():
                own = [
                    row
                    for row in attempts
                    if row.get("task_key") in task_index
                    and task_index[row["task_key"]][1:3] == (case, seed)
                ]
                matched = [
                    row
                    for row in observations
                    if row.arm == arm
                    and row.case_id == case
                    and row.repeat_seed == seed
                ]
                if len(matched) != 1:
                    errors.append(f"observation_identity:{case}:{seed}")
                    continue
                observation = matched[0]
                individual = resources_from_ledger_v3(arm, plan.history.namespace, own)
                for resource_key, observation_key in (
                    ("physical_requests", "physical_request_count"),
                    ("known_input_tokens", "known_input_tokens"),
                    ("known_output_tokens", "known_output_tokens"),
                    ("usage_unknown_attempts", "usage_unknown_attempts"),
                ):
                    if getattr(individual, resource_key) != getattr(
                        observation, observation_key
                    ):
                        errors.append(
                            f"ledger_observation:{case}:{seed}:{observation_key}"
                        )
                if (
                    len({row["task_key"] for row in own})
                    != observation.first_send_count
                ):
                    errors.append(f"ledger_observation:{case}:{seed}:first_send_count")
        except ValueError as error:
            errors.append(str(error))
        identities[arm.value] = {
            "planned": len(planned),
            "actual": len(actual),
            "failures": sorted(set(errors)),
            "ledger_payload_sha256": payload_sha256(attempts),
        }
        reasons.extend(f"{arm.value}:{error}" for error in errors)
    report["Q-12"]["attempt_identity"] = identities
    report["Q-12"].update(
        incomplete_reasons=sorted(set(reasons)),
        complete=not reasons,
        meets_threshold=not reasons,
    )
    return report


def verify_report_v6(stored, plan, observations, *, attempts_by_arm, **kwargs):
    recomputed = build_report_v6(
        plan, observations, attempts_by_arm=attempts_by_arm, **kwargs
    )
    if payload_sha256(stored) != payload_sha256(recomputed):
        changed = (
            sorted(
                key
                for key in set(stored) | set(recomputed)
                if key not in stored
                or key not in recomputed
                or payload_sha256(stored[key]) != payload_sha256(recomputed[key])
            )
            if type(stored) is dict
            else ["report_contract"]
        )
        raise ValueError("report_replay_mismatch:" + ",".join(changed))
    return {
        "verified": True,
        "schema_version": recomputed["schema_version"],
        "Q-12": recomputed["Q-12"]["complete"],
        "stage_acceptance": recomputed["stage_acceptance"],
    }


def plan_by_arm(plan: ComparisonPlanV1) -> dict[ComparisonArm, ArmPlan]:
    return dict(plan.arm)


__all__ = [
    "OBSERVATION_GLOB",
    "build_report_v5",
    "build_report_v6",
    "verify_report_v6",
    "claim_collection",
    "load_observations",
    "observation_from",
    "observation_path",
    "plan_by_arm",
    "resources_from_ledger",
    "resources_from_ledger_v3",
    "terminal_state_for",
    "write_observation",
]
