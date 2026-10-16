"""Complete replay identity preserves JSON types, fields and key-order freedom."""

import copy
import json
import sys

import pytest

from procurement_review.eval_g_cli import main_score_cli, main_sweep_cli
from procurement_review.eval_g_report import build_report_v6, verify_report_v6
from test_v1_07g_binding_v4 import write_inputs
from test_v1_07g_score_v3 import comparison_v3
from v1_07g_consumption_support import fixture_v5


@pytest.fixture(scope="module")
def baseline():
    return fixture_v5()


@pytest.mark.parametrize("mutation", ["boolean", "integer", "extra_null", "key_order"])
@pytest.mark.parametrize("entry", ["recheck", "freeze"])
def test_score_artifact_json_identity(tmp_path, monkeypatch, baseline, mutation, entry):
    paths = write_inputs(tmp_path, baseline)
    score = tmp_path / "score.json"
    common = [
        str(paths["registration"]),
        str(paths["observations"]),
        "--bindings",
        str(paths["bindings"]),
        "--source-bindings",
        str(paths["source_bindings"]),
        "--output",
        str(score),
    ]
    monkeypatch.setattr(sys, "argv", ["dev-score", "score", *common])
    main_score_cli()
    stored = json.loads(score.read_bytes())
    assert stored["completeness"]["complete"]
    assert stored["stage_acceptance"] is False
    assert stored["completeness"]["planned_model_tasks"] == 48
    if mutation == "boolean":
        stored["stage_acceptance"] = 0
    elif mutation == "integer":
        stored["completeness"]["planned_model_tasks"] = 48.0
    elif mutation == "extra_null":
        stored["unexpected_null"] = None
    else:
        stored = dict(reversed(list(stored.items())))
    score.write_text(json.dumps(stored))
    if entry == "recheck":
        monkeypatch.setattr(sys, "argv", ["dev-score", "recheck", *common])
        if mutation == "key_order":
            main_score_cli()
        else:
            with pytest.raises(SystemExit, match="recheck mismatch at"):
                main_score_cli()
    else:
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
        reason = (
            "v5_freeze_gates_failed"
            if mutation == "key_order"
            else "v5_freeze_full_score_mismatch"
        )
        with pytest.raises(ValueError, match=reason):
            main_sweep_cli()
        assert not frozen.exists()


@pytest.mark.parametrize("mutation", ["boolean", "integer", "extra_null", "key_order"])
def test_comparison_report_json_identity(mutation):
    plan, observed, kwargs, ledgers = comparison_v3()
    report = build_report_v6(plan, observed, attempts_by_arm=ledgers, **kwargs)
    stored = copy.deepcopy(report)
    assert stored["stage_acceptance"] is False
    assert stored["Q-12"]["complete"] is True
    if mutation == "boolean":
        stored["stage_acceptance"] = 0
    elif mutation == "integer":
        stored["Q-12"]["complete"] = 1
    elif mutation == "extra_null":
        stored["unexpected_null"] = None
    else:
        stored = dict(reversed(list(stored.items())))
    if mutation == "key_order":
        assert verify_report_v6(
            stored, plan, observed, attempts_by_arm=ledgers, **kwargs
        )["verified"]
    else:
        with pytest.raises(ValueError, match="report_replay_mismatch"):
            verify_report_v6(stored, plan, observed, attempts_by_arm=ledgers, **kwargs)
