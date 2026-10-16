"""Runtime repair cannot silently adopt changed model or semantic evidence."""

from copy import deepcopy

import pytest

from procurement_review.eval_attempt_ledger import LedgerError
from procurement_review.eval_f_context import stable_write
from procurement_review.eval_f_lineage import diagnostic_directory
from procurement_review.eval_f_registration import registration
from procurement_review.gold_v3_contracts import payload_sha256


def prior(tmp_path, monkeypatch):
    old = registration()
    digest = payload_sha256(old)
    folder = tmp_path / digest / "baseline"
    stable_write(folder / "registration.json", old)
    stable_write(folder / "result.json", {"diagnostic_passed": True})
    monkeypatch.setattr(
        "procurement_review.eval_f_run.replay", lambda _: {"diagnostic_passed": True}
    )
    new = deepcopy(old)
    new["diagnostic_predecessor"] = digest
    new["implementation"]["eval_f_budget.py"] = "a" * 64
    return new, folder


def test_runtime_only_repair_adopts_original_directory_and_original_identity(
    tmp_path, monkeypatch
):
    new, folder = prior(tmp_path, monkeypatch)
    assert diagnostic_directory(tmp_path, new, "baseline") == folder


@pytest.mark.parametrize(
    "change", ["prompt", "truth", "guard", "scorer", "production", "result"]
)
def test_quality_changes_invalidate_prior_diagnostics(tmp_path, monkeypatch, change):
    new, _ = prior(tmp_path, monkeypatch)
    if change == "prompt":
        new["candidates"]["baseline"]["configuration"]["prompt_style"] = "compact"
    elif change == "truth":
        new["gold_manifest_sha256"] = "b" * 64
    elif change == "result":
        monkeypatch.setattr(
            "procurement_review.eval_f_run.replay",
            lambda _: {"diagnostic_passed": False},
        )
    else:
        name = {
            "guard": "business_v3_security.py",
            "scorer": "eval_f_score.py",
            "production": "business_v3_support.py",
        }[change]
        new["implementation"][name] = "b" * 64
    with pytest.raises(LedgerError):
        diagnostic_directory(tmp_path, new, "baseline")
