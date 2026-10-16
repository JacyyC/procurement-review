"""Score a V1-02C paired batch from its persisted segments and frozen Gold Labels.

Labels are read only here, after every model call has finished, and are never sent
to a model. The output is a report document under the ignored run directory: every
ratio carries its raw numerator and denominator, the topology difference carries a
paired bootstrap interval, and the statistical limits are written next to the numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .eval_contract import (
    DEFAULT_RUNS_DIR,
    EVAL_BATCHES,
    EvalBatchSpec,
    EvalContractError,
    EvalProjection,
    EvalRunManifest,
    gold_corpus_contract,
    load_batch_segments,
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


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"unserialisable report value: {type(value).__name__}")


REPORT_SCHEMA_VERSION = "eval-report.v1"
REPORT_SCHEMA_VERSION_V2 = "eval-report.v2"
# v3 carries the identical metrics but keeps no model prose in error_samples. The v1 and
# v2 documents already on disk stay the AC-06~AC-08 anchors under their recorded digests;
# this generator only ever writes v3.
REPORT_SCHEMA_VERSION_V3 = "eval-report.v3"
# Replaces the V1-02C line that deferred the final judgement to this stage.
V1_02D_LIMIT = (
    "V1-02D 的审计与汇总报告只重放已落盘的清单、投影与报告，未发起任何新增模型请求；"
    "holdout 仍封存"
)
# Replaces the same line for the one batch that reads the hidden split (V1-05 Spec §6.2).
HOLDOUT_READ_LIMIT = (
    "本批次按 V1-05 Spec §6.2 读取 holdout 120 Case 作样本外最终判定；该分区的一次性解封"
    "不可逆，此后不再用于任何合同设计、调参或追加批次"
)
STATISTICAL_LIMITS: tuple[str, ...] = (
    "480 个合成 Case 只覆盖冻结的演示规则族，不构成对真实企业分布的统计功效证明",
    "重复种子在每个 Case 内部先取均值，配对 bootstrap 的有效样本是 Case 数而不是调用数",
    "检索指标来自确定性词法索引在同一批合成文档上的实测；Gold 文档彼此主要由 4 位 "
    "Case 号区分，因此该数值不可外推为真实语料的检索质量",
    "本 Stage 不读取 holdout；任何最终判定运行都属于 V1-02D",
    "阈值取自 Evidence §23.4 的冻结合同；实测不达标时只报告，不修改 Gold Label、"
    "阈值或样本划分",
)
# v1 keeps its wording verbatim so the anchored C-1/C-2 reports stay replayable.
STATISTICAL_LIMITS_V2: tuple[str, ...] = (
    "480 个合成 Case 只覆盖冻结的演示规则族，不构成对真实企业分布的统计功效证明",
    "重复种子在每个 Case 内部先取均值，配对 bootstrap 的有效样本是 Case 数而不是调用数",
    "检索指标来自确定性词法索引在 Gold 语料 v2 同一批条款级文档上的实测；该语料的 query "
    "与文档共享同一批业务属性词，因此数值证明语料已具备分辨率，仍不可外推为真实检索质量",
    "本 Stage 不读取 holdout；任何最终判定运行都属于 V1-02D",
    "阈值取自 Evidence §23.4 的冻结合同；实测不达标时只报告，不修改 Gold Label、"
    "阈值或样本划分",
    "Gold 语料 v2 与 v1 的 C-1/C-2 不处于同一 corpus snapshot，只作版本化对照，"
    "不回退或改写 v1 证据",
)


def _label_index(
    corpus_version: GoldCorpusVersion, spec: EvalBatchSpec
) -> dict[str, GoldLabel]:
    """Load the labels of exactly the splits this batch ran on.

    The hidden split is only ever read here, and only for the batch whose frozen spec
    declares it (V1-05 Spec §6.2), so no historical report can pick it up by accident.
    """

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


def _assert_corpus_anchor(
    segments: Sequence[tuple[EvalRunManifest, EvalProjection]],
    corpus_version: GoldCorpusVersion,
    expected_checksum: str,
) -> None:
    """Refuse to rescore a segment against a corpus version it did not run on."""

    contract = GOLD_CORPUS_CONTRACTS[corpus_version]
    for manifest, _ in segments:
        if manifest.gold_manifest_checksum != expected_checksum:
            raise EvalContractError("gold_version_mismatch")
        versions = manifest.source_versions
        if (
            versions.gold_set != contract.gold_set_version
            or versions.corpus_snapshot != contract.corpus_snapshot_id
        ):
            raise EvalContractError("gold_version_mismatch")


def _case_index(cases: Sequence[GoldCase], spec: EvalBatchSpec) -> dict[str, GoldCase]:
    scope = set(spec.splits)
    return {case.case_id: case for case in cases if case.split in scope}


def _observations(
    segments: Sequence[tuple[EvalRunManifest, EvalProjection]],
    cases: Mapping[str, GoldCase],
    topology: str,
) -> tuple[CaseObservation, ...]:
    items: list[CaseObservation] = []
    for _, projection in segments:
        if projection.topology != topology:
            continue
        items.extend(observations_from_projection(projection, cases))
    return tuple(items)


def build_report(batch_id: str, *, runs_dir: Path = DEFAULT_RUNS_DIR) -> dict[str, Any]:
    """Rebuild the full AC-06..AC-08 report for one frozen batch from disk."""

    spec = EVAL_BATCHES.get(batch_id)
    if spec is None:
        raise EvalContractError("batch_not_registered")
    segments = load_batch_segments(batch_id, runs_dir=runs_dir)
    if not segments:
        raise EvalContractError("no_segments_for_batch")
    corpus = gold_corpus_contract(spec)
    corpus_manifest = load_gold_manifest(corpus_version=corpus.version)
    checksum = manifest_checksum(corpus_manifest)
    _assert_corpus_anchor(segments, corpus.version, checksum)
    cases = load_gold_cases(corpus_version=corpus.version).cases
    case_index = _case_index(cases, spec)
    labels = _label_index(corpus.version, spec)

    manifests = {
        topology: [
            manifest for manifest, _ in segments if manifest.topology == topology
        ]
        for topology in ("single_agent", "multi_agent")
    }
    observations = {
        topology: _observations(segments, case_index, topology)
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
    scored_cases = [case for case in cases if case.split in set(spec.splits)]
    is_v2 = corpus.version is GoldCorpusVersion.V2
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION_V3,
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
            manifest.run_id for manifest, _ in segments if manifest.voided
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
            is_v2, holdout_read=GoldSplit.HOLDOUT in spec.splits
        ),
    }
    if is_v2:
        report["gold_corpus"] = {
            "version": corpus.version.value,
            "gold_set_version": corpus.gold_set_version,
            "generator_version": corpus.generator_version,
            "corpus_snapshot_id": corpus.corpus_snapshot_id,
            "document_version": corpus.document_version,
            "manifest_checksum": checksum,
            "labels_payload_sha256": corpus_manifest.labels_payload_sha256,
        }
    return report


def limits_for(corpus_is_v2: bool, *, holdout_read: bool = False) -> list[str]:
    """The frozen limit wording for one corpus, with the V1-02D line updated in place."""

    base = STATISTICAL_LIMITS_V2 if corpus_is_v2 else STATISTICAL_LIMITS
    replacement = HOLDOUT_READ_LIMIT if holdout_read else V1_02D_LIMIT
    return [
        replacement if line.startswith("本 Stage 不读取 holdout") else line
        for line in base
    ]


def write_report(
    report: Mapping[str, Any], *, batch_id: str, runs_dir: Path = DEFAULT_RUNS_DIR
) -> Path:
    root = private_runs_root(runs_dir, create=True)
    directory = root / batch_id
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    path = directory / f"{report['schema_version']}.json"
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a frozen V1-02C eval batch from its run segments."
    )
    parser.add_argument("--batch", required=True, choices=sorted(EVAL_BATCHES))
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="print the headline numbers without model or response content",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = build_report(args.batch, runs_dir=args.runs_dir)
        path = write_report(report, batch_id=args.batch, runs_dir=args.runs_dir)
    except (EvalContractError, OSError, ValueError) as exc:
        code = getattr(exc, "code", None) or type(exc).__name__
        print(f"eval-score-error: {code}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    if args.print_summary:
        for topology in ("single_agent", "multi_agent"):
            quality = report["quality"][topology]
            cost = report["cost_and_reliability"][topology]
            print(
                f"{topology}: calls={cost['model_call_count']} "
                f"valid={cost['structured_output_valid_rate'].value} "
                f"accuracy={quality['recommendation_accuracy'].value} "
                f"unknown_recall={quality['unknown_recall'].value} "
                f"citation_support={quality['citation_support_precision'].value} "
                f"required_findings={quality['required_finding_coverage'].value} "
                f"high_risk_miss={quality['high_risk_miss_rate'].value}"
            )
        for delta in report["paired_deltas"]:
            print(
                f"delta {delta['metric']}: {delta['delta']} "
                f"ci=({delta['ci_low']}, {delta['ci_high']}) favors={delta['favors']} "
                f"cases={delta['case_count']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
