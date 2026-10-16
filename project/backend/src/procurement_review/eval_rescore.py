"""Rescore a frozen eval batch offline under a selected aggregation contract.

V1-05A delivers this tool so a revised Finding/aggregation contract can be judged on the
twelve segments already on disk without spending a single model request. It loads the
persisted manifests and projections itself, re-runs the deterministic aggregation and the
frozen metric formulas under either rules 1.0 or rules 2.0, and writes a versioned
``eval-rescore.v1`` report.

The v1 mode is the credibility anchor: rescoring under the frozen contract must reproduce
the already-anchored report field for field, otherwise the tool is wrong and no v2 reading
it produces may be trusted. Source material is verified by digest, and a mismatch fails
closed the same way ``eval_score`` rejects a corpus version a segment did not run on.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .contracts import RecommendationStatus
from .hashing import sha256_file
from .eval_contract import (
    DEFAULT_RUNS_DIR,
    EVAL_BATCHES,
    MANIFEST_FILE_NAME,
    PROJECTION_FILE_NAME,
    EvalBatchSpec,
    EvalContractError,
    EvalProjection,
    EvalRunManifest,
    gold_corpus_contract,
    load_eval_manifest,
    load_projection,
    private_runs_root,
)
from .eval_metrics import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONFIDENCE,
    METRICS_VERSION,
    RECALL_K,
    CaseObservation,
    error_samples,
    observations_from_projection,
    paired_topology_deltas,
    quality_metrics,
    retrieval_report,
    structured_output_stats,
)
from .eval_score import limits_for
from .gold_corpus import build_gold_lexical_corpus
from .gold_set import (
    GOLD_CORPUS_CONTRACTS,
    GoldCase,
    GoldCorpusVersion,
    GoldLabel,
    GoldSplit,
    load_gold_cases,
    load_gold_labels_for_split,
    load_gold_manifest,
    manifest_checksum,
)
from .rules import RULE_SET, RULE_SET_V2, RuleSet

EVAL_RESCORE_VERSION = "v1-05a-eval-rescore.v1"
RESCORE_SCHEMA_VERSION = "eval-rescore.v1"
FROZEN_REPORT_SCHEMA_VERSION = "eval-report.v3"

# Only the aggregation contract is selectable here; the corpus version a batch ran on comes
# from its frozen spec, and the outbound model vocabulary is identical for both selectors.
CONTRACT_RULE_SETS: dict[str, RuleSet] = {"v1": RULE_SET, "v2": RULE_SET_V2}
CONTRACT_RESOURCES: dict[str, str] = {"v1": "v1.json", "v2": "v2.json"}

# Keys the anchored report lineage already carries. Anything else a rescore report adds is
# declared below and is expected to be absent from the frozen document.
FROZEN_REPORT_KEYS = frozenset(
    {
        "batch",
        "cost_and_reliability",
        "error_samples",
        "execution_modes",
        "gold_corpus",
        "metrics_version",
        "paired_deltas",
        "quality",
        "resolved_model_ids",
        "retrieval",
        "segment_run_ids",
        "statistical_limits",
        "statistics_contract",
        "voided_segments",
    }
)
RESCORE_ONLY_KEYS = frozenset(
    {"rescore", "decidability", "family_mapping", "contract_delta", "schema_version"}
)
_ESCALATED_RECOMMENDATIONS = frozenset(
    {
        RecommendationStatus.INSUFFICIENT_INFORMATION.value,
        RecommendationStatus.NEEDS_HUMAN_REVIEW.value,
    }
)
CAUSE_NAMESPACES = ("pipeline", "evidence", "coverage")
IN_SAMPLE_NOTE = "dev/validation 重算用于合同设计，属样本内证据；最终判定须由未参与设计的样本外批次确认"
OUT_OF_SAMPLE_NOTE = (
    "holdout 未参与 rules v2 的合同设计，本批次是 V1-05 Spec §6.2 预注册的样本外确认；"
    "该分区的一次性解封不可逆，此后不再用于合同设计、调参或追加批次"
)


class RescoreError(RuntimeError):
    """A bounded reason code for a rescore that must not proceed."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _rules_digest(contract: str) -> str:
    path = Path(__file__).parent / "resources" / "rules" / CONTRACT_RESOURCES[contract]
    return sha256_file(path)


def load_segments(
    batch_id: str, *, runs_dir: Path = DEFAULT_RUNS_DIR
) -> list[tuple[EvalRunManifest, EvalProjection, str, str]]:
    """Load one batch's segments together with the digests of the files they came from."""

    root = private_runs_root(Path(runs_dir))
    if not root.is_dir():
        raise RescoreError("runs_dir_missing")
    found: list[tuple[EvalRunManifest, EvalProjection, str, str]] = []
    for directory in sorted(root.iterdir()):
        manifest_path = directory / MANIFEST_FILE_NAME
        projection_path = directory / PROJECTION_FILE_NAME
        if not manifest_path.is_file() or not projection_path.is_file():
            continue
        manifest = load_eval_manifest(manifest_path)
        if manifest.batch_id != batch_id:
            continue
        found.append(
            (
                manifest,
                load_projection(projection_path),
                sha256_file(manifest_path),
                sha256_file(projection_path),
            )
        )
    if not found:
        raise RescoreError("no_segments_for_batch")
    return sorted(found, key=lambda item: (item[0].topology, item[0].repeat_seed))


def _assert_source_anchors(
    segments: Sequence[tuple[EvalRunManifest, EvalProjection, str, str]],
    spec: EvalBatchSpec,
    *,
    expected_checksum: str,
) -> None:
    """Refuse to rescore material that is not this batch's own frozen corpus version."""

    contract = GOLD_CORPUS_CONTRACTS[GoldCorpusVersion(spec.gold_corpus_version)]
    for manifest, _, _, _ in segments:
        if manifest.gold_manifest_checksum != expected_checksum:
            raise RescoreError("gold_version_mismatch")
        versions = manifest.source_versions
        if (
            versions.gold_set != contract.gold_set_version
            or versions.corpus_snapshot != contract.corpus_snapshot_id
        ):
            raise RescoreError("gold_version_mismatch")


def _label_index(
    corpus_version: GoldCorpusVersion, spec: EvalBatchSpec
) -> dict[str, GoldLabel]:
    """Load exactly the labels of the splits the batch ran on; holdout stays gated."""

    labels: dict[str, GoldLabel] = {}
    for split in spec.splits:
        labels.update(
            {
                label.case_id: label
                for label in load_gold_labels_for_split(
                    split,
                    corpus_version=corpus_version,
                    allow_holdout=split is GoldSplit.HOLDOUT,
                )
            }
        )
    return labels


def _observations(
    segments: Sequence[tuple[EvalRunManifest, EvalProjection, str, str]],
    cases: Mapping[str, GoldCase],
    topology: str,
    rule_set: RuleSet,
) -> tuple[CaseObservation, ...]:
    items: list[CaseObservation] = []
    for _, projection, _, _ in segments:
        if projection.topology != topology:
            continue
        items.extend(observations_from_projection(projection, cases, rule_set=rule_set))
    return tuple(items)


def _is_healthy(item: CaseObservation) -> bool:
    """R-2's precondition for judging decidability: every branch completed."""

    outcomes = tuple(item.branch_outcomes.values())
    return (
        len(outcomes) == 3
        and set(outcomes) == {"succeeded"}
        and not (item.failed_workers)
    )


def _conflict_anchor_of(subject: str) -> str:
    if subject.startswith("conflict:evidence_ref:"):
        return "evidence_ref"
    if subject.startswith("conflict:business_fact:"):
        return "business_fact"
    return "finding_key"


def _cause_namespace(code: str) -> str:
    for namespace in CAUSE_NAMESPACES:
        if code.startswith(f"{namespace}:"):
            return namespace
    return "unnamespaced_or_model_note"


def _has_both_assertions(item: CaseObservation) -> bool:
    assertions = {finding.assertion.value for finding in item.findings}
    return {"risk_present", "risk_absent"} <= assertions


def _mode(counts: Mapping[str, int]) -> str | None:
    if not counts:
        return None
    return sorted(counts, key=lambda key: (-counts[key], key))[0]


def _decidability(
    observations: Sequence[CaseObservation],
    labels: Mapping[str, GoldLabel],
) -> dict[str, Any]:
    """Raw counts behind the D-1..D-4 gates, reported without any pass/fail wording."""

    scored = [item for item in observations if item.case_id in labels]
    healthy = [item for item in scored if _is_healthy(item)]
    healthy_recommendations = Counter(item.recommendation.value for item in healthy)
    classes = Counter(
        (
            "not_directional"
            if item.escalation_class is None
            else item.escalation_class.value
        )
        for item in scored
    )
    cause_counts = Counter(
        _cause_namespace(code)
        for item in scored
        for code in (*item.missing_information, *item.failure_reasons)
    )
    conflict_total = conflict_hits = 0
    undetected_with_both_sides = 0
    detections_outside_conflict_labels = 0
    by_anchor: Counter[str] = Counter()
    by_topology: Counter[str] = Counter()
    for item in scored:
        label = labels[item.case_id]
        detected = bool(item.conflict_keys)
        if detected:
            by_topology[item.topology] += 1
            by_anchor.update(_conflict_anchor_of(key) for key in item.conflict_keys)
        if label.expected_conflicts:
            conflict_total += 1
            conflict_hits += int(detected)
            if not detected and _has_both_assertions(item):
                undetected_with_both_sides += 1
        elif detected:
            detections_outside_conflict_labels += 1
    escalated = [
        item
        for item in scored
        if item.recommendation.value in _ESCALATED_RECOMMENDATIONS
    ]
    return {
        "d1_healthy_subset": {
            "observations": len(scored),
            "healthy_observations": len(healthy),
            "healthy_share": (round(len(healthy) / len(scored), 6) if scored else None),
            "recommendation_counts": dict(sorted(healthy_recommendations.items())),
            "distinct_recommendations": len(healthy_recommendations),
        },
        "d2_conflicts": {
            "observations_with_expected_conflicts": conflict_total,
            "observations_with_detected_conflict": conflict_hits,
            "detected_conflicts_by_anchor": dict(sorted(by_anchor.items())),
            "detected_observations_by_topology": dict(sorted(by_topology.items())),
            "detections_without_expected_conflicts": detections_outside_conflict_labels,
            "undetected_with_both_assertions_present": undetected_with_both_sides,
        },
        "d3_escalation": {
            "escalated_observations": len(escalated),
            "by_class": dict(sorted(classes.items())),
            "by_cause_namespace": dict(sorted(cause_counts.items())),
        },
    }


def _family_mapping(
    observations: Sequence[CaseObservation],
    labels: Mapping[str, GoldLabel],
) -> dict[str, Any]:
    """Per-family recommendation distribution and mode, for R-7 and D-4."""

    grouped: dict[str, list[CaseObservation]] = defaultdict(list)
    for item in observations:
        if item.case_id in labels:
            grouped[item.case_family_id].append(item)
    report: dict[str, Any] = {}
    for family in sorted(grouped):
        items = grouped[family]
        healthy = [item for item in items if _is_healthy(item)]
        counts = Counter(item.recommendation.value for item in items)
        healthy_counts = Counter(item.recommendation.value for item in healthy)
        report[family] = {
            "observations": len(items),
            "healthy_observations": len(healthy),
            "expected_recommendation_counts": dict(
                sorted(
                    Counter(
                        labels[item.case_id].expected_recommendation.value
                        for item in items
                    ).items()
                )
            ),
            "recommendation_counts": dict(sorted(counts.items())),
            "mode": _mode(counts),
            "healthy_counts": dict(sorted(healthy_counts.items())),
            "healthy_mode": _mode(healthy_counts),
            "conflict_detections": sum(1 for item in items if item.conflict_keys),
            "distinct_recommendations": len(counts),
        }
    return report


def _as_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"unserialisable report value: {type(value).__name__}")


def _metric_view(quality: Mapping[str, Any]) -> dict[str, Any]:
    """The scalar readings a contract delta compares, without the stratified bulk."""

    return {
        name: payload
        for name, payload in quality.items()
        if isinstance(payload, Mapping) and "numerator" in payload
    }


def _delta_for_topology(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    old = _metric_view(before)
    new = _metric_view(after)
    changed = {
        name: {"rules_1": old.get(name), "rules_2": new.get(name)}
        for name in sorted(set(old) | set(new))
        if old.get(name) != new.get(name)
    }
    old_matrix = before.get("confusion_matrix") or {}
    new_matrix = after.get("confusion_matrix") or {}
    return {
        "changed_metric_count": len(changed),
        "metrics": changed,
        "confusion_matrix_only_in_rules_2": {
            key: new_matrix[key] for key in sorted(set(new_matrix) - set(old_matrix))
        },
        "confusion_matrix_only_in_rules_1": {
            key: old_matrix[key] for key in sorted(set(old_matrix) - set(new_matrix))
        },
    }


def _build_frozen_portion(
    *,
    spec: EvalBatchSpec,
    segments: Sequence[tuple[EvalRunManifest, EvalProjection, str, str]],
    cases: Sequence[GoldCase],
    case_index: Mapping[str, GoldCase],
    labels: Mapping[str, GoldLabel],
    observations: Mapping[str, tuple[CaseObservation, ...]],
    rule_set: RuleSet,
) -> dict[str, Any]:
    """Rebuild the anchored report fields for one contract, independently of eval_score."""

    manifests = {
        topology: [
            manifest for manifest, _, _, _ in segments if manifest.topology == topology
        ]
        for topology in ("single_agent", "multi_agent")
    }
    corpora = tuple(
        (
            split.value,
            build_gold_lexical_corpus(
                [case for case in cases if case.split is split],
                split=split,
            ),
        )
        for split in spec.splits
    )
    corpus = gold_corpus_contract(spec)
    corpus_manifest = load_gold_manifest(corpus_version=corpus.version)
    is_v2_corpus = corpus.version is GoldCorpusVersion.V2
    scored_cases = [case for case in cases if case.split in set(spec.splits)]
    report: dict[str, Any] = {
        "metrics_version": METRICS_VERSION,
        "batch": spec.model_dump(mode="json"),
        "segment_run_ids": {
            topology: [manifest.run_id for manifest in manifests[topology]]
            for topology in manifests
        },
        "execution_modes": {
            topology: sorted(
                {manifest.execution_mode for manifest in manifests[topology]}
            )
            for topology in manifests
        },
        "resolved_model_ids": {
            topology: sorted(
                {
                    model_id
                    for manifest in manifests[topology]
                    for model_id in manifest.resolved_model_ids
                }
            )
            for topology in manifests
        },
        "voided_segments": [
            manifest.run_id for manifest, _, _, _ in segments if manifest.voided
        ],
        "cost_and_reliability": {
            topology: structured_output_stats(manifests[topology])
            for topology in manifests
        },
        "retrieval": retrieval_report(corpora, scored_cases, labels, k=RECALL_K),
        "quality": {
            topology: quality_metrics(observations[topology], case_index, labels)
            for topology in manifests
        },
        "paired_deltas": [
            delta.model_dump(mode="json")
            for delta in paired_topology_deltas(
                observations["single_agent"], observations["multi_agent"], labels
            )
        ],
        "error_samples": {
            topology: list(error_samples(observations[topology], labels))
            for topology in manifests
        },
        "statistics_contract": {
            "confidence": CONFIDENCE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "interval_method": "Wilson for ratios, percentile bootstrap over cases for paired deltas",
        },
        "statistical_limits": limits_for(
            is_v2_corpus, holdout_read=GoldSplit.HOLDOUT in spec.splits
        ),
    }
    if is_v2_corpus:
        report["gold_corpus"] = {
            "version": corpus.version.value,
            "gold_set_version": corpus.gold_set_version,
            "generator_version": corpus.generator_version,
            "corpus_snapshot_id": corpus.corpus_snapshot_id,
            "document_version": corpus.document_version,
            "manifest_checksum": manifest_checksum(corpus_manifest),
            "labels_payload_sha256": corpus_manifest.labels_payload_sha256,
        }
    return report


def build_report(
    batch_id: str,
    contract: str,
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    with_delta: bool = True,
) -> dict[str, Any]:
    """Build one versioned rescore report for a frozen batch under one contract."""

    spec = EVAL_BATCHES.get(batch_id)
    if spec is None:
        raise RescoreError("batch_not_registered")
    if contract not in CONTRACT_RULE_SETS:
        raise RescoreError("contract_not_registered")
    rule_set = CONTRACT_RULE_SETS[contract]
    segments = load_segments(batch_id, runs_dir=runs_dir)
    corpus = gold_corpus_contract(spec)
    checksum = manifest_checksum(load_gold_manifest(corpus_version=corpus.version))
    _assert_source_anchors(segments, spec, expected_checksum=checksum)
    cases = load_gold_cases(corpus_version=corpus.version).cases
    holdout_batch = GoldSplit.HOLDOUT in spec.splits
    case_index = {
        case.case_id: case for case in cases if case.split in set(spec.splits)
    }
    labels = _label_index(corpus.version, spec)
    observations = {
        topology: _observations(segments, case_index, topology, rule_set)
        for topology in ("single_agent", "multi_agent")
    }
    report: dict[str, Any] = {
        "schema_version": RESCORE_SCHEMA_VERSION,
        "rescore": {
            "tool_version": EVAL_RESCORE_VERSION,
            "contract": contract,
            "rules_version": rule_set.rules_version,
            "gate_mode": rule_set.gate_mode,
            "rules_resource_sha256": _rules_digest(contract),
            "model_input_changed": False,
            "in_sample": not holdout_batch,
            "in_sample_note": (OUT_OF_SAMPLE_NOTE if holdout_batch else IN_SAMPLE_NOTE),
            "holdout_read": holdout_batch,
            "frozen_report_schema": FROZEN_REPORT_SCHEMA_VERSION,
            "frozen_report_sha256": frozen_report_digest(batch_id, runs_dir=runs_dir),
            "segments": [
                {
                    "run_id": manifest.run_id,
                    "topology": manifest.topology,
                    "repeat_seed": manifest.repeat_seed,
                    "manifest_sha256": manifest_digest,
                    "projection_sha256": projection_digest,
                    "code_baseline": manifest.source_versions.code,
                    "rules_at_run": manifest.source_versions.rules,
                }
                for manifest, _, manifest_digest, projection_digest in segments
            ],
        },
        **_build_frozen_portion(
            spec=spec,
            segments=segments,
            cases=cases,
            case_index=case_index,
            labels=labels,
            observations=observations,
            rule_set=rule_set,
        ),
    }
    if rule_set.gate_mode == "directional":
        baseline_observations = {
            topology: _observations(segments, case_index, topology, RULE_SET)
            for topology in ("single_agent", "multi_agent")
        }
        report["decidability"] = {
            topology: {
                "rules_2": _decidability(observations[topology], labels),
                "rules_1": _decidability(baseline_observations[topology], labels),
            }
            for topology in observations
        }
        report["family_mapping"] = {
            topology: {
                "rules_2": _family_mapping(observations[topology], labels),
                "rules_1": _family_mapping(baseline_observations[topology], labels),
            }
            for topology in observations
        }
        if with_delta:
            normalized = _as_json(report["quality"])
            report["contract_delta"] = {
                topology: _delta_for_topology(
                    _as_json(
                        quality_metrics(
                            baseline_observations[topology], case_index, labels
                        )
                    ),
                    normalized[topology],
                )
                for topology in ("single_agent", "multi_agent")
            }
    return report


def frozen_report_path(batch_id: str, *, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    return Path(runs_dir) / batch_id / f"{FROZEN_REPORT_SCHEMA_VERSION}.json"


def frozen_report_digest(batch_id: str, *, runs_dir: Path = DEFAULT_RUNS_DIR) -> str:
    path = frozen_report_path(batch_id, runs_dir=runs_dir)
    if not path.is_file():
        raise RescoreError("frozen_report_missing")
    return sha256_file(path)


def _diff_paths(expected: Any, actual: Any, prefix: str) -> Iterable[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted({*expected, *actual}, key=str):
            if key not in expected or key not in actual:
                yield f"{prefix}/{key}:presence"
            else:
                yield from _diff_paths(expected[key], actual[key], f"{prefix}/{key}")
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            yield f"{prefix}:length({len(expected)}!={len(actual)})"
            return
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            yield from _diff_paths(left, right, f"{prefix}/{index}")
        return
    if expected != actual:
        yield f"{prefix}:value({expected!r}!={actual!r})"


def diff_against_frozen(
    report: Mapping[str, Any], *, batch_id: str, runs_dir: Path = DEFAULT_RUNS_DIR
) -> dict[str, Any]:
    """Compare the shared fields of a rescore report with the anchored report."""

    path = frozen_report_path(batch_id, runs_dir=runs_dir)
    frozen = json.loads(path.read_text(encoding="utf-8"))
    shared = _as_json(report)
    differing: list[str] = []
    for key in sorted(FROZEN_REPORT_KEYS):
        in_report, in_frozen = key in shared, key in frozen
        if not (in_report or in_frozen):
            continue
        if in_report != in_frozen:
            differing.append(f"/{key}:presence")
            continue
        differing.extend(_diff_paths(frozen[key], shared[key], f"/{key}"))
    extra = sorted(set(report) - FROZEN_REPORT_KEYS - RESCORE_ONLY_KEYS)
    unexplained = sorted(set(frozen) - FROZEN_REPORT_KEYS - {"schema_version"})
    return {
        "frozen_report_sha256": sha256_file(path),
        "compared_keys": sorted(FROZEN_REPORT_KEYS & set(shared) & set(frozen)),
        "differing_paths": differing,
        "undeclared_extra_keys": extra,
        "unexplained_frozen_keys": unexplained,
        "consistent": not differing and not extra and not unexplained,
    }


def write_report(
    report: Mapping[str, Any], *, batch_id: str, runs_dir: Path = DEFAULT_RUNS_DIR
) -> Path:
    """Persist one rescore report beside the batch it rescoring, owner-only."""

    root = private_runs_root(Path(runs_dir), create=True)
    directory = root / batch_id
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    path = directory / f"eval-rescore.rules-{report['rescore']['contract']}.json"
    path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _print_summary(report: Mapping[str, Any]) -> None:
    rescore = report["rescore"]
    print(
        f"contract={rescore['contract']} rules={rescore['rules_version']} "
        f"segments={len(rescore['segments'])} "
        f"frozen_sha256={rescore['frozen_report_sha256'][:12]}"
    )
    for topology in ("single_agent", "multi_agent"):
        quality = report["quality"][topology]
        print(
            f"{topology}: accuracy={quality['recommendation_accuracy'].value} "
            f"conflict={quality['conflict_recall'].numerator}/"
            f"{quality['conflict_recall'].denominator} "
            f"unknown_recall={quality['unknown_recall'].value} "
            f"high_risk_miss={quality['high_risk_miss_rate'].numerator}/"
            f"{quality['high_risk_miss_rate'].denominator} "
            f"takeover={quality['human_takeover_rate'].value}"
        )
        gates = report.get("decidability", {}).get(topology)
        if not gates:
            continue
        for contract_name in ("rules_1", "rules_2"):
            block = gates[contract_name]
            d1 = block["d1_healthy_subset"]
            d2 = block["d2_conflicts"]
            d3 = block["d3_escalation"]
            print(
                f"  {contract_name} D1 healthy={d1['healthy_observations']}/"
                f"{d1['observations']} distinct={d1['distinct_recommendations']} "
                f"{json.dumps(d1['recommendation_counts'], ensure_ascii=False)}"
            )
            print(
                f"  {contract_name} D2 detected="
                f"{d2['observations_with_detected_conflict']}/"
                f"{d2['observations_with_expected_conflicts']} "
                f"anchors={json.dumps(d2['detected_conflicts_by_anchor'])} "
                f"without_expected={d2['detections_without_expected_conflicts']}"
            )
            print(
                f"  {contract_name} D3 classes={json.dumps(d3['by_class'])} "
                f"causes={json.dumps(d3['by_cause_namespace'])}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rescore a frozen eval batch offline under rules 1.0 or 2.0."
    )
    parser.add_argument("--batch", required=True, choices=sorted(EVAL_BATCHES))
    parser.add_argument("--contract", required=True, choices=sorted(CONTRACT_RULE_SETS))
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--check-frozen",
        action="store_true",
        help="diff the recomputed fields against the anchored eval-report.v3 and "
        "exit non-zero on any difference",
    )
    parser.add_argument(
        "--print-summary", action="store_true", help="print headline counts only"
    )
    parser.add_argument(
        "--write", action="store_true", help="write the versioned rescore report"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = build_report(args.batch, args.contract, runs_dir=args.runs_dir)
        check = (
            diff_against_frozen(report, batch_id=args.batch, runs_dir=args.runs_dir)
            if args.check_frozen
            else None
        )
        path = (
            write_report(report, batch_id=args.batch, runs_dir=args.runs_dir)
            if args.write
            else None
        )
    except (RescoreError, EvalContractError, OSError, ValueError) as exc:
        code = getattr(exc, "code", None) or type(exc).__name__
        print(f"eval-rescore-error: {code}", file=sys.stderr)
        return 1
    if args.print_summary:
        _print_summary(report)
    if path is not None:
        print(f"wrote {path}")
    if check is not None:
        print(
            f"check-frozen batch={args.batch} contract={args.contract} "
            f"compared_keys={len(check['compared_keys'])} "
            f"consistent={check['consistent']} "
            f"differing={len(check['differing_paths'])} "
            f"undeclared_extra={check['undeclared_extra_keys']} "
            f"unexplained={check['unexplained_frozen_keys']}"
        )
        for line in check["differing_paths"][:20]:
            print(f"  diff {line}")
        if not check["consistent"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
