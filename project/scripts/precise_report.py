#!/usr/bin/env python3
"""Zero-network audit and supplements for a completed F precise Case batch.

Run with the backend Python environment. Source artifacts are never rewritten.
The report distinguishes complete observations, quality gates, and stage acceptance.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3

from procurement_review.eval_f_context import load_cases
from procurement_review.eval_f_contracts import FCaseProjection
from procurement_review.eval_f_run import replay
from procurement_review.eval_f_score import CLASSES, proportion, load_truth
from procurement_review.gold_v3_contracts import payload_sha256


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values):
    ordered = sorted(math.inf if value is None else value for value in values)

    def at(fraction):
        value = ordered[math.ceil(fraction * len(ordered)) - 1] if ordered else math.inf
        return None if math.isinf(value) else value

    return dict(
        count=len(values),
        p50=at(0.5),
        p95=at(0.95),
        maximum=at(1),
        unobserved=sum(value is None for value in values),
    )


def uncertainty(rows, cases, seeds):
    """Average repeated measurements per Case before resampling entire groups."""
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    accuracy = {
        cid: sum(r["counts"].get("recommendation_correct", 0) for r in items)
        / len(seeds)
        for cid, items in by_case.items()
    }
    unknown = {
        cid: int(items[0]["expected"] == "insufficient_information")
        for cid, items in by_case.items()
    }
    clusters = defaultdict(list)
    for cid in accuracy:
        clusters[cases[cid].near_duplicate_group_id].append(cid)
    keys = sorted(clusters)
    rng = random.Random(20260919)
    accuracies, deltas = [], []
    for _ in range(10000):
        chosen = [cid for _ in keys for cid in clusters[keys[rng.randrange(len(keys))]]]
        accuracies.append(sum(accuracy[cid] for cid in chosen) / len(chosen))
        deltas.append(sum(accuracy[cid] - unknown[cid] for cid in chosen) / len(chosen))
    return {
        "group_field": "near_duplicate_group_id",
        "clusters": len(keys),
        "unique_cases": len(accuracy),
        "resamples": 10000,
        "seed": 20260919,
        "accuracy_point": sum(accuracy.values()) / len(accuracy),
        "accuracy_ci95": [sorted(accuracies)[249], sorted(accuracies)[9749]],
        "paired_difference_vs_unknown": sum(accuracy[c] - unknown[c] for c in accuracy)
        / len(accuracy),
        "paired_difference_ci95": [sorted(deltas)[249], sorted(deltas)[9749]],
        "method": "average repeats within Case; resample complete groups; percentile ranks 250/9750",
    }


def timing_report(projections, rows, cfg):
    result = {}
    by_key = {(r["case_id"], r["repeat_seed"]): r for r in rows}
    fields = (
        "queue_ms",
        "context_ms",
        "model_ms",
        "extra_attempt_ms",
        "aggregate_ms",
        "total_ms",
        "retrieval_ms",
        "mcp_ms",
        "connect_ms",
        "first_response_ms",
        "human_wait_ms",
    )
    for seed in cfg["repeat_seeds"]:
        groups = {"all_planned": [], "successful_pipeline": [], "failed_or_missing": []}
        for cid in cfg["case_ids"]:
            projection = projections.get((cid, seed))
            groups["all_planned"].append(projection)
            failed = by_key[cid, seed]["counts"].get("pipeline_failed", 0)
            groups["failed_or_missing" if failed else "successful_pipeline"].append(
                projection
            )
        result[str(seed)] = {
            name: {
                **{
                    field: quantiles(
                        [getattr(p.timing, field) if p else None for p in values]
                    )
                    for field in fields
                },
                "execution_ms": quantiles(
                    [
                        p.timing.total_ms - p.timing.queue_ms if p else None
                        for p in values
                    ]
                ),
            }
            for name, values in groups.items()
        }
    return result


def ledger_audit(folder, projections, expected):
    export = json.loads((folder / "ledger-export.json").read_text())
    with sqlite3.connect(
        (folder / "ledger.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
    ) as conn:
        conn.row_factory = sqlite3.Row
        attempts = list(conn.execute("SELECT * FROM attempts ORDER BY attempt_id"))
    raw = {a["attempt_id"]: a for a in attempts}
    checks = {
        "export_matches_database_count": len(export["attempts"]) == len(attempts),
        "manifest_count": export["manifest"]["attempt_count"] == len(attempts),
        "registered_stage": export["manifest"]["stage_id"] == "V1-07F",
        "receipts_complete": all(a["receipt_json"] is not None for a in attempts),
        "maximum_two_attempts": all(a["attempt_no"] <= 2 for a in attempts),
    }
    equality = []
    for item in export["attempts"]:
        original = raw.get(item["attempt_id"])
        if original is None:
            equality.append(False)
            continue
        for kind in ("receipt", "projection"):
            encoded = original[kind + "_json"]
            equality.append(item[kind] == (json.loads(encoded) if encoded else None))
            if encoded:
                equality.append(
                    payload_sha256(json.loads(encoded)) == original[kind + "_sha256"]
                )
    checks["export_and_record_hashes"] = all(equality)
    matches = []
    for cid, seed in expected:
        projection = projections.get((cid, seed))
        for worker in ("policy", "contract"):
            entries = [
                a
                for a in attempts
                if a["task_key"] == f"{cid}.{seed}.{worker}" and a["projection_json"]
            ]
            branches = (
                [b for b in projection.branches if b.worker == worker]
                if projection
                else []
            )
            matches.append(
                bool(entries)
                and len(branches) == 1
                and json.loads(entries[-1]["projection_json"])
                == branches[0].model_dump(mode="json")
            )
    checks["all_branch_projections_committed_and_equal"] = all(matches)
    receipts = [json.loads(a["receipt_json"]) for a in attempts if a["receipt_json"]]
    usage = {
        "physical_reserved": len(attempts),
        "receipts": len(receipts),
        "extra_attempts": sum(a["attempt_no"] > 1 for a in attempts),
        "known_input": sum(r.get("input_tokens") or 0 for r in receipts),
        "known_output": sum(r.get("output_tokens") or 0 for r in receipts),
        "unknown_usage": sum(r["usage_unknown"] for r in receipts),
        "unreturned": len(attempts) - len(receipts),
        "error_counts": {},
    }
    for receipt in receipts:
        code = receipt.get("underlying_error_code") or receipt.get("error_code")
        if code:
            usage["error_counts"][code] = usage["error_counts"].get(code, 0) + 1
    return checks, usage


def build_report(folder):
    folder = Path(folder)
    # Never make validation observations available before its full run terminates.
    original = json.loads((folder / "result.json").read_text())
    source_files = [
        folder / name
        for name in ("registration.json", "result.json", "ledger-export.json")
    ]
    source_files.extend(sorted(folder.glob("V107-*.json")))
    before = {p.name: sha(p) for p in source_files}
    registered = json.loads((folder / "registration.json").read_text())
    version = registered["versions"][0]
    if version == "gold-set.v3.1":
        from procurement_review.gold_v31_oracle import derive_truth
    elif version == "gold-set.v3.2":
        from procurement_review.gold_v32_oracle import derive_truth
    elif version == "gold-set.v3.3":
        from procurement_review.gold_v33_oracle import derive_truth
    elif version == "gold-set.v3.4":
        from procurement_review.gold_v34_oracle import derive_truth
    elif version == "gold-set.v3.5":
        from procurement_review.gold_v35_oracle import derive_truth
    elif version == "gold-set.v3.6":
        from procurement_review.gold_v36_oracle import derive_truth
    else:
        raise ValueError("unsupported precise report dataset")
    cfg = registered["candidates"][folder.name]
    report = replay(folder)
    projections = {}
    for file in folder.glob("V107-*.json"):
        p = FCaseProjection.model_validate_json(file.read_bytes())
        projections[p.case_id, p.repeat_seed] = p
    expected = {(cid, seed) for cid in cfg["case_ids"] for seed in cfg["repeat_seeds"]}
    cases = {
        c.case_id: c
        for c in load_cases(
            "validation" if folder.name.startswith("validation-") else "dev", version
        )
    }
    truths = load_truth(version)
    ids = cfg["case_ids"]
    class_counts = {
        name: sum(truths[cid].expected_recommendation.value == name for cid in ids)
        for name in CLASSES
    }
    unknown_correct = class_counts["insufficient_information"]
    reference_matches = sum(
        derive_truth(cases[cid]).model_dump() == truths[cid].model_dump() for cid in ids
    )
    checks, usage = ledger_audit(folder, projections, expected)
    checks.update(
        {
            "independent_truth_matches": reference_matches == len(ids),
            "zero_network_replay_identical": all(
                original.get(k) == v for k, v in report.items()
            ),
            "registration_identity": payload_sha256(registered) == folder.parent.name,
            "case_set_complete": set(projections) == expected,
            "case_configuration_fixed": all(
                p.configuration_sha256 == folder.parent.name
                for p in projections.values()
            ),
            "candidate_request_cap": usage["physical_reserved"] <= cfg["max_requests"],
        }
    )
    slices = {}
    for seed in cfg["repeat_seeds"]:
        slices[str(seed)] = {}
        for name, predicate in {
            "real_insufficient": lambda t: t.expected_recommendation.value
            == "insufficient_information",
            "security": lambda t: bool(t.security_events),
            "conflict": lambda t: bool(t.conflicts),
            "high_risk": lambda t: t.high_risk,
            "decidable": lambda t: t.expected_recommendation.value in CLASSES[:3],
        }.items():
            rows = [
                r
                for r in report["rows"]
                if r["repeat_seed"] == seed and predicate(truths[r["case_id"]])
            ]
            slices[str(seed)][name] = {
                "cases": len(rows),
                "recommendation_correct": sum(
                    r["counts"].get("recommendation_correct", 0) for r in rows
                ),
                "pipeline_failed": sum(
                    r["counts"].get("pipeline_failed", 0) for r in rows
                ),
                "failed_case_ids": [
                    r["case_id"]
                    for r in rows
                    if not r["counts"].get("recommendation_correct", 0)
                ],
            }
    checks["originals_unchanged"] = before == {p.name: sha(p) for p in source_files}
    return {
        "contract": "v1-07f-report-audit.v1",
        "candidate": folder.name,
        "registration_sha256": folder.parent.name,
        "source_sha256": before,
        "audit_checks": checks,
        "audit_passed": all(checks.values()),
        "quality": report,
        "usage": usage,
        "all_unknown_baseline": {
            "recommendation": "insufficient_information",
            "accuracy": proportion(unknown_correct, len(ids), 0.9),
            "macro_f1": 2 * unknown_correct / (unknown_correct + len(ids)) / 5,
            "class_counts": class_counts,
        },
        "independent_reference": {
            "entry": derive_truth.__module__
            + ".derive_truth; no production Aggregator",
            "exact_truth_matches": reference_matches,
            "cases": len(ids),
        },
        "uncertainty": uncertainty(report["rows"], cases, cfg["repeat_seeds"]),
        "strata": slices,
        "timing": timing_report(projections, report["rows"], cfg),
        "legacy_supported_over_all_obligations": {
            str(seed): {
                "numerator": sum(
                    sum(f.state == "supported" for f in p.decision.business.findings)
                    for (cid, s), p in projections.items()
                    if s == seed
                ),
                "denominator": report["repeats"][str(seed)]["counts"]["obligations"],
            }
            for seed in cfg["repeat_seeds"]
        },
        "fault_injected_normal_cases": 0,
        "fault_evidence": "separate controlled V-F-01..03; excluded from normal quality denominators",
        "network_calls": 0,
        "stage_acceptance": False,
        "scope": "F independent Oracle Runner; total includes batch queue; no G workbench claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(
        args.directory.resolve()
    ):
        parser.error("output must be new and outside the source batch")
    result = build_report(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("candidate", "audit_passed", "usage", "stage_acceptance")
            }
        )
    )
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
