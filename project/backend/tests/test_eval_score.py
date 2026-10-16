"""Score-time corpus anchoring: a segment may only be rescored on its own corpus.

Both Gold versions now live beside each other, so ``eval_score`` must refuse to
recompute a batch against a corpus it never ran on. These tests dry-run real
segments offline (mock transport, zero network) and rebuild the reports from disk.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from procurement_review.eval_contract import (
    RESOLUTION_BATCH,
    EvalContractError,
)
from procurement_review.eval_run import run_batch
from procurement_review.eval_audit import classify_leaves
from procurement_review.eval_score import (
    REPORT_SCHEMA_VERSION_V3,
    V1_02D_LIMIT,
    build_report,
)
from procurement_review.gold_set import (
    GoldCorpusVersion,
    load_gold_manifest,
    manifest_checksum,
)

CALIBRATION = "v1-02c-calibration-48"


def _dry_run_runs_dir(tmp_path: Path, batch_id: str) -> Path:
    runs_dir = tmp_path / ".local" / "runs"
    written = asyncio.run(
        run_batch(batch_id=batch_id, execution_mode="dry_run", runs_dir=runs_dir)
    )
    assert len(written) == 2 if batch_id == CALIBRATION else 4
    return runs_dir


def _rewrite_manifest_digest(runs_dir: Path, batch_id: str, replacement: str) -> None:
    segments = sorted(runs_dir.glob(f"eval-*{batch_id}*/eval-run-manifest.v1.json"))
    assert segments
    path = segments[0]
    text = path.read_text(encoding="utf-8")
    current = manifest_checksum(load_gold_manifest(corpus_version="v2"))
    edited = text.replace(current, replacement, 1)
    assert edited != text, "the segment did not carry the expected corpus digest"
    path.write_text(edited, encoding="utf-8")


def test_v1_batch_report_keeps_the_anchored_schema_and_corpus(tmp_path: Path) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path, CALIBRATION)
    report = build_report(CALIBRATION, runs_dir=runs_dir)
    assert report["schema_version"] == REPORT_SCHEMA_VERSION_V3
    assert "gold_corpus" not in report
    assert report["statistical_limits"][2].startswith(
        "检索指标来自确定性词法索引在同一批合成文档上的实测"
    )
    assert V1_02D_LIMIT in report["statistical_limits"]
    assert not any(
        line.startswith("本 Stage 不读取 holdout")
        for line in report["statistical_limits"]
    )
    assert report["retrieval"]["dev"]["indexed_document_count"] == 440
    assert list(report["retrieval"]) == ["dev"]
    assert report["execution_modes"] == {
        "single_agent": ["dry_run"],
        "multi_agent": ["dry_run"],
    }
    assert report["batch"]["gold_corpus_version"] == "v1"
    assert (
        classify_leaves({"samples": report["error_samples"]})["free_form_fields"] == {}
    )


def test_resolution_batch_report_is_scored_on_the_clause_level_corpus(
    tmp_path: Path,
) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path, RESOLUTION_BATCH.batch_id)
    report = build_report(RESOLUTION_BATCH.batch_id, runs_dir=runs_dir)
    assert report["schema_version"] == REPORT_SCHEMA_VERSION_V3
    assert report["batch"]["gold_corpus_version"] == "v2"
    gold = report["gold_corpus"]
    assert gold["corpus_snapshot_id"] == "synthetic-gold-corpus.v2"
    assert gold["manifest_checksum"] == manifest_checksum(
        load_gold_manifest(corpus_version=GoldCorpusVersion.V2)
    )
    assert report["statistical_limits"][-1].startswith("Gold 语料 v2 与 v1 的 C-1/C-2")
    assert V1_02D_LIMIT in report["statistical_limits"]
    assert (
        classify_leaves({"samples": report["error_samples"]})["free_form_fields"] == {}
    )
    retrieval = report["retrieval"]
    assert list(retrieval) == ["dev"]
    variants = retrieval["dev"]["variants"]
    assert variants["raw_query"]["recall_at_5"].value > 0.9
    assert variants["masked_case_number_query"]["recall_at_5"].value > 0.9
    for topology in ("single_agent", "multi_agent"):
        quality = report["quality"][topology]
        assert quality["observation_count"] == 240
        assert quality["required_finding_coverage"].denominator == 720


def test_scoring_refuses_a_segment_from_another_gold_corpus(tmp_path: Path) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path, RESOLUTION_BATCH.batch_id)
    _rewrite_manifest_digest(
        runs_dir,
        RESOLUTION_BATCH.batch_id,
        manifest_checksum(load_gold_manifest(corpus_version="v1")),
    )
    with pytest.raises(EvalContractError) as raised:
        build_report(RESOLUTION_BATCH.batch_id, runs_dir=runs_dir)
    assert raised.value.code == "gold_version_mismatch"


def test_scoring_a_v1_batch_with_a_v2_anchored_segment_fails(tmp_path: Path) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path, CALIBRATION)
    path = next(runs_dir.glob(f"eval-*{CALIBRATION}*/eval-run-manifest.v1.json"))
    text = path.read_text(encoding="utf-8")
    v1 = manifest_checksum(load_gold_manifest(corpus_version="v1"))
    edited = text.replace(
        v1, manifest_checksum(load_gold_manifest(corpus_version="v2")), 1
    )
    assert edited != text
    path.write_text(edited, encoding="utf-8")
    with pytest.raises(EvalContractError) as raised:
        build_report(CALIBRATION, runs_dir=runs_dir)
    assert raised.value.code == "gold_version_mismatch"
