"""Report orchestration for the V1-07 metric contract."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .eval_v2_contracts import EvalPredictionV2, EvalTopology
from .eval_v2_metrics import _split_for_case, score_predictions
from .gold_v3_contracts import TruthRecordV1, V3Split


def score_by_split(
    truths: Sequence[TruthRecordV1],
    predictions: Sequence[EvalPredictionV2],
    *,
    legacy: Mapping[str, Any] | None = None,
    near_duplicate_groups: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Score each v3 split and preserve topology/seed slices."""

    result: dict[str, Any] = {}
    for split in (V3Split.DEV, V3Split.VALIDATION):
        selected = [
            item
            for item in predictions
            if _split_for_case(item.case_id, truths) is split
        ]
        combined = score_predictions(
            truths,
            selected,
            split=split,
            legacy=legacy,
            near_duplicate_groups=near_duplicate_groups,
        )
        grouped: dict[tuple[EvalTopology, int], list[EvalPredictionV2]] = defaultdict(
            list
        )
        for item in selected:
            grouped[(item.topology, item.repeat_seed)].append(item)
        combined["by_topology_and_seed"] = {
            f"{topology.value}:{seed}": score_predictions(
                truths,
                items,
                split=split,
                legacy=legacy,
                near_duplicate_groups=near_duplicate_groups,
            )
            for (topology, seed), items in sorted(
                grouped.items(), key=lambda item: (item[0][0].value, item[0][1])
            )
        }
        result[split.value] = combined
    return result


def build_report_v4(
    truths: Sequence[TruthRecordV1],
    predictions: Sequence[EvalPredictionV2],
    *,
    legacy: Mapping[str, Any] | None = None,
    near_duplicate_groups: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the new report lineage without rewriting historical reports."""
    contracts = {truth.truth_contract for truth in truths}
    if len(contracts) > 1:
        raise ValueError("mixed truth revisions cannot be scored together")
    contract = next(iter(contracts), "v1-07-truth.v1")
    version = {"v1-07-truth.v1": "gold-set.v3", "v1-07-truth.v1.1": "gold-set.v3.1"}[
        contract
    ]
    return {
        "schema_version": "eval-report.v4",
        "metrics_version": "v1-07-eval-metrics.v1",
        "gold_set_version": version,
        "truth_contract": contract,
        "scores": score_by_split(
            truths,
            predictions,
            legacy=legacy,
            near_duplicate_groups=near_duplicate_groups,
        ),
    }


__all__ = ["build_report_v4", "score_by_split"]
