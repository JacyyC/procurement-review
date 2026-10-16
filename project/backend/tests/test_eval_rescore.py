"""V1-05A offline rescore: the v1 anchor, the v2 gate block, and fail-closed inputs.

The dry-run segments written here are produced by the mock transport (zero network, zero
model requests), so this file proves the tool's plumbing rather than any model quality.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from procurement_review.eval_contract import CALIBRATION_BATCH
from procurement_review.eval_rescore import (
    FROZEN_REPORT_KEYS,
    RESCORE_ONLY_KEYS,
    RESCORE_SCHEMA_VERSION,
    RescoreError,
    build_report,
    diff_against_frozen,
    load_segments,
    main,
    write_report,
)
from procurement_review.eval_run import run_batch
from procurement_review.eval_score import build_report as build_frozen_report
from procurement_review.eval_score import write_report as write_frozen_report

CALIBRATION = CALIBRATION_BATCH.batch_id


@pytest.fixture(name="runs_dir")
def _runs_dir(tmp_path: Path) -> Path:
    root = tmp_path / ".local" / "runs"
    asyncio.run(
        run_batch(batch_id=CALIBRATION, execution_mode="dry_run", runs_dir=root)
    )
    write_frozen_report(
        build_frozen_report(CALIBRATION, runs_dir=root),
        batch_id=CALIBRATION,
        runs_dir=root,
    )
    return root


def test_the_v1_anchor_reproduces_the_anchored_report_field_for_field(
    runs_dir: Path,
) -> None:
    report = build_report(CALIBRATION, "v1", runs_dir=runs_dir)
    check = diff_against_frozen(report, batch_id=CALIBRATION, runs_dir=runs_dir)

    assert report["schema_version"] == RESCORE_SCHEMA_VERSION
    assert check["consistent"], check["differing_paths"][:5]
    assert check["differing_paths"] == []
    assert set(check["compared_keys"]) == set(FROZEN_REPORT_KEYS) - {"gold_corpus"}
    assert report["rescore"]["model_input_changed"] is False
    assert report["rescore"]["holdout_read"] is False
    assert report["rescore"]["in_sample"] is True


def test_the_v1_anchor_survives_a_second_process(runs_dir: Path) -> None:
    first = write_report(
        build_report(CALIBRATION, "v1", runs_dir=runs_dir),
        batch_id=CALIBRATION,
        runs_dir=runs_dir,
    )
    body = first.read_bytes()
    second = write_report(
        build_report(CALIBRATION, "v1", runs_dir=runs_dir),
        batch_id=CALIBRATION,
        runs_dir=runs_dir,
    )
    assert second.read_bytes() == body
    assert second.stat().st_mode & 0o777 == 0o600
    assert "frozen_report_sha256" in json.loads(body)["rescore"]


def test_rules_2_mode_adds_the_declared_blocks_and_keeps_the_shared_fields(
    runs_dir: Path,
) -> None:
    v1 = build_report(CALIBRATION, "v1", runs_dir=runs_dir)
    v2 = build_report(CALIBRATION, "v2", runs_dir=runs_dir)

    assert "decidability" not in v1 and "contract_delta" not in v1
    assert set(v2) - set(v1) == set(RESCORE_ONLY_KEYS) - {"schema_version", "rescore"}
    for topology in ("single_agent", "multi_agent"):
        gates = v2["decidability"][topology]
        assert set(gates) == {"rules_1", "rules_2"}
        total = v2["quality"][topology]["observation_count"]
        for contract_name in ("rules_1", "rules_2"):
            block = gates[contract_name]
            assert set(block) == {"d1_healthy_subset", "d2_conflicts", "d3_escalation"}
            classes = block["d3_escalation"]["by_class"]
            # The classes must partition the batch: mutually exclusive and complete.
            assert sum(classes.values()) == total
            assert block["d1_healthy_subset"]["observations"] == total
        # Rules 1.0 cannot separate the outcomes at all; rules 2.0 can.
        assert set(gates["rules_1"]["d3_escalation"]["by_class"]) == {"not_directional"}
        assert len(gates["rules_2"]["d3_escalation"]["by_class"]) > 1
        assert (
            gates["rules_1"]["d1_healthy_subset"]["distinct_recommendations"]
            <= gates["rules_2"]["d1_healthy_subset"]["distinct_recommendations"]
        )
        mapping = v2["family_mapping"][topology]
        assert set(mapping) == {"rules_1", "rules_2"}
        assert set(mapping["rules_1"]) == set(mapping["rules_2"])
        for family, payload in mapping["rules_2"].items():
            assert family.startswith("GF-")
            assert payload["observations"] == sum(
                payload["recommendation_counts"].values()
            )
    assert v2["rescore"]["gate_mode"] == "directional"
    assert v2["contract_delta"]["multi_agent"]["changed_metric_count"] >= 0


def test_the_rules_resource_digests_differ_between_contracts(runs_dir: Path) -> None:
    v1 = build_report(CALIBRATION, "v1", runs_dir=runs_dir)["rescore"]
    v2 = build_report(CALIBRATION, "v2", runs_dir=runs_dir)["rescore"]
    assert v1["rules_version"] == "1.0.0"
    assert v2["rules_version"] == "2.0.0"
    assert v1["rules_resource_sha256"] != v2["rules_resource_sha256"]
    assert len(v1["segments"]) == 2
    assert all(entry["manifest_sha256"] for entry in v1["segments"])


def test_inputs_that_are_not_the_batchs_own_material_fail_closed(
    runs_dir: Path,
) -> None:
    assert load_segments(CALIBRATION, runs_dir=runs_dir)

    segments = sorted(runs_dir.glob(f"eval-*{CALIBRATION}*/eval-run-manifest.v1.json"))
    path = segments[0]
    report = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(
            {**report, "gold_manifest_checksum": "0" * 64},
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(RescoreError, match="gold_version_mismatch"):
        build_report(CALIBRATION, "v1", runs_dir=runs_dir)


def test_unknown_batches_and_contracts_are_refused(runs_dir: Path) -> None:
    with pytest.raises(RescoreError, match="batch_not_registered"):
        build_report("no-such-batch", "v1", runs_dir=runs_dir)
    with pytest.raises(RescoreError, match="contract_not_registered"):
        build_report(CALIBRATION, "v3", runs_dir=runs_dir)


def test_the_cli_reports_the_anchor_result_and_exits_non_zero_on_a_break(
    runs_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "--batch",
                CALIBRATION,
                "--contract",
                "v1",
                "--runs-dir",
                str(runs_dir),
                "--check-frozen",
                "--write",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "consistent=True differing=0" in printed
    assert "wrote" in printed

    broken = runs_dir / CALIBRATION / "eval-report.v3.json"
    payload = json.loads(broken.read_text(encoding="utf-8"))
    broken.write_text(
        json.dumps(
            {**payload, "metrics_version": "tampered"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "--batch",
                CALIBRATION,
                "--contract",
                "v1",
                "--runs-dir",
                str(runs_dir),
                "--check-frozen",
            ]
        )
        == 1
    )
    assert "metrics_version" in capsys.readouterr().out
