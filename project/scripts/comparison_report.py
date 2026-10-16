#!/usr/bin/env python3
"""Zero-network independent audit of one pre-registered two-arm comparison batch.

Run with the backend Python environment.  Source artifacts are never rewritten.
The report separates three judgements that must not be conflated: the absolute
product gates of each arm, the neutral Q-12 completeness of the comparison, and
this audit's own recomputation of both from the durable artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from procurement_review.eval_f_context import load_cases
from procurement_review.eval_g_contracts import (
    COMPARISON_AUDIT_V1,
    ArmResources,
    ComparisonArm,
    ComparisonObservationV1,
    ComparisonPlanV1,
)
from procurement_review.eval_g_plan import comparison_implementation_fingerprints
from procurement_review.eval_g_projection import GCaseProjection
from procurement_review.eval_g_report import (
    build_report_v5,
    load_observations,
    observation_from,
    resources_from_ledger,
)
from procurement_review.eval_g_score import load_truth, score_arm
from procurement_review.gold_v3_contracts import payload_sha256

ARMS = tuple(ComparisonArm)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_files(folder: Path) -> list[Path]:
    files = [folder / "plan.json"]
    for name in ("result.json", "fairness-observed.json", "resources.json"):
        if (folder / name).exists():
            files.append(folder / name)
    for arm in ARMS:
        arm_dir = folder / arm.value
        files.extend(sorted(arm_dir.glob("V107-*.json")))
        for name in ("ledger-export.json", "attempts.json"):
            if (arm_dir / name).exists():
                files.append(arm_dir / name)
        files.extend(sorted((arm_dir / "observations").glob("observation-*.json")))
    files.extend(sorted((folder / "observations").glob("observation-*.json")))
    return [item for item in files if item.exists()]


def _attempts(arm_dir: Path) -> list[dict[str, Any]]:
    path = arm_dir / "attempts.json"
    if path.exists():
        return json.loads(path.read_text())
    export = arm_dir / "ledger-export.json"
    if not export.exists():
        return []
    return json.loads(export.read_text())["attempts"]


def _projections(arm_dir: Path, arm: ComparisonArm) -> list[GCaseProjection]:
    loaded = [
        GCaseProjection.model_validate_json(path.read_bytes())
        for path in sorted(arm_dir.glob("V107-*.json"))
    ]
    for item in loaded:
        if item.arm is not arm:
            raise ValueError(f"projection for {arm.value} declares another arm")
    return loaded


def rebuild(folder: Path) -> tuple[dict[str, Any], Any, Any, Any]:
    """Recompute the whole batch from its durable artifacts; zero network."""

    declared = json.loads((folder / "plan.json").read_bytes())
    if declared.get("schema_version") == "v1-07h-plan.v1":
        from procurement_review.eval_h_report import rebuild as rebuild_h
        from procurement_review.eval_h_journal import Journal
        from procurement_review.eval_h_seal import HSeal

        h_plan = HSeal(folder).plan()
        report = rebuild_h(folder)
        observations = [
            body
            for key in h_plan.observation_keys()
            if (body := Journal(folder).committed(*key))
        ]
        return declared, report, report["Q-12"]["resources"], observations
    plan = ComparisonPlanV1.model_validate_json((folder / "plan.json").read_bytes())
    expected_digest = payload_sha256(
        plan.model_dump(mode="json", exclude={"plan_sha256"})
    )
    if plan.plan_sha256 != expected_digest:
        raise ValueError("frozen plan digest does not match its content")
    if plan.split == "blind":
        raise ValueError("legacy_blind_loader_unsupported_use_explicit_h_plan")
    split = "validation" if plan.split == "validation" else "dev"
    cases = {case.case_id: case for case in load_cases(split, plan.gold_set_version)}
    truths = load_truth(plan.gold_set_version)
    quality: dict[ComparisonArm, Any] = {}
    observations: list[ComparisonObservationV1] = []
    resources: dict[ComparisonArm, ArmResources] = {}
    for arm in ARMS:
        arm_dir = folder / arm.value
        arm_plan = plan.arm[arm]
        projections = _projections(arm_dir, arm)
        attempts = _attempts(arm_dir)
        scored = score_arm(cases, truths, projections, plan=arm_plan)
        quality[arm] = scored
        rows = {(row["case_id"], row["repeat_seed"]): row for row in scored["rows"]}
        by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for attempt in attempts:
            case_id, seed, _branch = attempt["task_key"].rsplit(".", 2)
            by_key.setdefault((case_id, int(seed)), []).append(attempt)
        by_projection = {(item.case_id, item.repeat_seed): item for item in projections}
        for key, row in rows.items():
            if key not in by_projection and key not in by_key:
                # Never started: no durable record, so the plan keeps the gap visible.
                continue
            observations.append(
                observation_from(
                    by_projection.get(key),
                    arm=arm,
                    shape=arm_plan.shape,
                    plan_sha256=plan.plan_sha256,
                    attempts=by_key.get(key, ()),
                    scored_row=row,
                    stop_reason=_stop_reason(arm_dir, key),
                )
            )
        latencies = sorted(
            row["latency_ms"] for row in scored["rows"] if row["latency_ms"] is not None
        )
        p95 = (
            latencies[max(0, -(-len(latencies) * 95 // 100) - 1)] if latencies else None
        )
        resources[arm] = resources_from_ledger(
            arm,
            plan.history.namespace,
            attempts,
            cost_status=_cost_status(folder, arm),
            cost_value=_cost_value(folder, arm),
            p95_total_latency_ms=p95,
        )
    fairness_path = folder / "fairness-observed.json"
    observed_fairness = (
        json.loads(fairness_path.read_text()) if fairness_path.exists() else None
    )
    groups = {case_id: case.near_duplicate_group_id for case_id, case in cases.items()}
    report = build_report_v5(
        plan,
        observations,
        expected={
            case_id: truth.expected_recommendation for case_id, truth in truths.items()
        },
        resources=resources,
        arm_quality=quality,
        observed_fairness=observed_fairness,
        near_duplicate_groups=groups,
        implementation=comparison_implementation_fingerprints(),
    )
    return plan.model_dump(mode="json"), report, resources, observations


def _stop_reason(arm_dir: Path, key: tuple[str, int]) -> str | None:
    path = arm_dir / "stops.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(f"{key[0]}.{key[1]}")


def _cost_status(folder: Path, arm: ComparisonArm) -> str:
    declared = _declared_costs(folder).get(arm.value) or {}
    return declared.get("cost_status", "unknown")


def _cost_value(folder: Path, arm: ComparisonArm) -> float | None:
    declared = _declared_costs(folder).get(arm.value) or {}
    return declared.get("cost_value")


def _declared_costs(folder: Path) -> Mapping[str, Any]:
    path = folder / "resources.json"
    return json.loads(path.read_text()) if path.exists() else {}


def build_audit(folder: Path) -> dict[str, Any]:
    folder = Path(folder)
    if (
        json.loads((folder / "plan.json").read_bytes()).get("schema_version")
        == "v1-07h-plan.v1"
    ):
        from procurement_review.eval_h_audit import audit as audit_h

        return audit_h(folder)
    sources = _source_files(folder)
    before = {str(item.relative_to(folder)): sha(item) for item in sources}
    result_path = folder / "result.json"
    original = json.loads(result_path.read_text()) if result_path.exists() else None
    _plan, report, resources, recomputed = rebuild(folder)
    observations_dir = folder / "observations"
    stored = sorted(
        (item.arm.value, item.case_id, item.repeat_seed)
        for item in load_observations(observations_dir)
    )
    settled = sorted(
        (item.arm.value, item.case_id, item.repeat_seed) for item in recomputed
    )
    checks: dict[str, bool] = {
        "plan_digest_matches_content": True,
        "both_arms_have_a_frozen_plan": len(report["Q-12"]["coverage"]["by_arm"]) == 2,
        "recomputation_matches_stored_result": (
            original is not None and _stable(original) == _stable(report)
        ),
        "stored_observations_match_recomputation": bool(stored) and stored == settled,
        "ledger_matches_projections": all(
            not item["differences"]
            for item in report["Q-12"]["ledger_reconciliation"]["by_arm"].values()
        ),
        "history_namespaces_kept_separate": report["Q-12"]["history_namespaces"][
            "current"
        ]
        not in report["Q-12"]["history_namespaces"]["prior"],
        "no_physical_capacity_exceeded": all(
            item.physical_requests
            <= report["Q-12"]["coverage"]["by_arm"][arm.value]["physical_capacity"]
            for arm, item in resources.items()
        ),
        "originals_unchanged": before
        == {str(item.relative_to(folder)): sha(item) for item in sources},
    }
    comparison = report["Q-12"]
    return {
        "contract": COMPARISON_AUDIT_V1,
        "batch": folder.name,
        "plan_id": report["plan_id"],
        "plan_sha256": report["plan_sha256"],
        "source_sha256": before,
        "audit_checks": checks,
        "audit_passed": all(checks.values()),
        "comparison_complete": comparison["complete"],
        "incomplete_reasons": comparison["incomplete_reasons"],
        "product_gates": {
            arm: item["meets_product_gates"]
            for arm, item in report["product_quality"].items()
        },
        "outcome": comparison["paired_delta"]["outcome"],
        "paired_delta": comparison["paired_delta"]["multi_minus_single_accuracy"],
        "resource_comparison": comparison["resource_comparison"],
        "network_calls": 0,
        "stage_acceptance": False,
        "scope": (
            "independent recomputation of one pre-registered batch; winning, tying "
            "and losing are all settled outcomes and none of them gates this audit"
        ),
    }


def _stable(value: Any) -> Any:
    """Compare two reports without the run-local implementation fingerprint."""

    if isinstance(value, dict):
        return {
            key: _stable(item)
            for key, item in sorted(value.items())
            if key != "implementation"
        }
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(
        args.directory.resolve()
    ):
        parser.error("output must be new and outside the audited batch")
    result = build_audit(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "batch",
                    "audit_passed",
                    "comparison_complete",
                    "incomplete_reasons",
                    "product_gates",
                    "outcome",
                    "stage_acceptance",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
