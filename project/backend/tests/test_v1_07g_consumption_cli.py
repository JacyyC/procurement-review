"""Official in-process consumers retain failed metrics and refuse qualification."""

import json
import sys

import pytest

from procurement_review.eval_g_cli import main_score_cli, main_sweep_cli
from test_v1_07g_binding_v4 import write_inputs
from v1_07g_consumption_support import fixture_v5


@pytest.fixture(scope="module")
def baseline():
    return fixture_v5()


def test_score_recheck_and_freeze_keep_quality_failure(
    tmp_path, monkeypatch, baseline, capsys
):
    paths = write_inputs(tmp_path, baseline)
    score = tmp_path / "score.json"
    args = [
        str(paths["registration"]),
        str(paths["observations"]),
        "--bindings",
        str(paths["bindings"]),
        "--source-bindings",
        str(paths["source_bindings"]),
        "--output",
        str(score),
    ]
    monkeypatch.setattr(sys, "argv", ["dev-score", "score", *args])
    main_score_cli()
    result = json.loads(score.read_bytes())
    assert result["score_version"] == "v1-07g-dev-score.v5"
    assert result["completeness"]["complete"]
    assert result["stage_acceptance"] is False
    assert "repeat-1/Q-02" in result["failed_gates"]
    monkeypatch.setattr(sys, "argv", ["dev-score", "recheck", *args])
    main_score_cli()
    assert "identical" in capsys.readouterr().out
    frozen = tmp_path / "freeze.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dev-sweep",
            "freeze-v5",
            str(paths["registration"]),
            "--observations",
            str(paths["observations"]),
            "--bindings",
            str(paths["bindings"]),
            "--source-bindings",
            str(paths["source_bindings"]),
            "--score",
            str(score),
            "--output",
            str(frozen),
        ],
    )
    with pytest.raises(ValueError, match="v5_freeze_gates_failed"):
        main_sweep_cli()
    assert not frozen.exists()


def test_cli_source_identity_fails_before_output(tmp_path, monkeypatch, baseline):
    paths = write_inputs(tmp_path, baseline)
    source = json.loads(paths["source_bindings"].read_bytes())
    run = next(iter(source["runs"].values()))
    run["execution"]["summary"]["highest_risk"] = "critical"
    paths["source_bindings"].write_text(json.dumps(source))
    output = tmp_path / "refused.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dev-score",
            "score",
            str(paths["registration"]),
            str(paths["observations"]),
            "--bindings",
            str(paths["bindings"]),
            "--source-bindings",
            str(paths["source_bindings"]),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="v4_source_evidence_file_binding"):
        main_score_cli()
    assert not output.exists()
