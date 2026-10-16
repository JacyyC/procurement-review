from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from procurement_review.eval_compat import (
    FROZEN_RESOURCE_SHA256,
    CompatibilityError,
    load_gold_compatibility_bundle,
    load_rules_compatibility_version,
    read_versioned_json,
    sensitive_value_findings,
    verify_frozen_resource_hashes,
)
from procurement_review.eval_v2_contracts import (
    EvalPredictionV2,
    SemanticClaimProjection,
)
from v1_07b_support import prediction_for, v3_bundle

RESOURCES = Path(__file__).parents[1] / "src" / "procurement_review" / "resources"


def test_v1_v2_v3_gold_loaders_coexist_without_rewriting_old_resources() -> None:
    before = {
        relative: hashlib.sha256((RESOURCES / relative).read_bytes()).hexdigest()
        for relative in FROZEN_RESOURCE_SHA256
    }
    bundles = {
        version: load_gold_compatibility_bundle(version)
        for version in ("v1", "v2", "v3")
    }
    assert len(bundles["v1"].cases.cases) == 480
    assert len(bundles["v2"].cases.cases) == 480
    assert len(bundles["v3"].cases.cases) == 180
    assert bundles["v1"].manifest.gold_set_version == "gold-set.v1"
    assert bundles["v2"].manifest.gold_set_version == "gold-set.v2"
    assert bundles["v3"].manifest.gold_set_version == "gold-set.v3"
    after = {
        relative: hashlib.sha256((RESOURCES / relative).read_bytes()).hexdigest()
        for relative in FROZEN_RESOURCE_SHA256
    }
    assert before == after


def test_old_gold_and_rule_resources_keep_their_frozen_byte_hashes() -> None:
    identities = verify_frozen_resource_hashes(RESOURCES)
    assert len(identities) == 8
    assert {item.path.relative_to(RESOURCES).as_posix() for item in identities} == set(
        FROZEN_RESOURCE_SHA256
    )
    assert load_rules_compatibility_version("v1").rules_version == "1.0.0"
    assert load_rules_compatibility_version("2.0.0").rules_version == "2.0.0"
    with pytest.raises(CompatibilityError, match="unknown rules version"):
        load_rules_compatibility_version("v3")


@pytest.mark.parametrize(
    ("kind", "schema"),
    [
        ("gold_manifest", "gold-manifest.v1"),
        ("gold_manifest", "gold-manifest.v2"),
        ("run_manifest", "run-manifest.v1"),
        ("run_manifest", "eval-run-manifest.v1"),
        ("run_manifest", "eval-run-manifest.v2"),
        ("projection", "eval-projection.v1"),
        ("projection", "eval-projection.v2"),
        ("report", "eval-report.v1"),
        ("report", "eval-report.v2"),
        ("report", "eval-report.v3"),
        ("report", "eval-report.v4"),
        ("report", "eval-rescore.v1"),
    ],
)
def test_historical_and_new_artifact_lineages_are_explicitly_recognized(
    tmp_path: Path, kind: str, schema: str
) -> None:
    path = tmp_path / f"{schema}.json"
    body = json.dumps({"schema_version": schema, "sentinel": 1}).encode()
    path.write_bytes(body)
    document, identity = read_versioned_json(path, kind=kind)
    assert document["sentinel"] == 1
    assert identity.schema_version == schema
    assert identity.sha256 == hashlib.sha256(body).hexdigest()


def test_unknown_schema_duplicate_keys_and_missing_files_fail_closed(
    tmp_path: Path,
) -> None:
    unknown = tmp_path / "unknown.json"
    unknown.write_text('{"schema_version":"eval-report.v5"}', encoding="utf-8")
    with pytest.raises(CompatibilityError, match="unsupported report schema"):
        read_versioned_json(unknown, kind="report")

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"eval-report.v4","schema_version":"eval-report.v3"}',
        encoding="utf-8",
    )
    with pytest.raises(CompatibilityError, match="duplicate JSON key"):
        read_versioned_json(duplicate, kind="report")

    with pytest.raises(CompatibilityError, match="missing frozen resource"):
        verify_frozen_resource_hashes(tmp_path / "missing")


def test_sensitive_scan_ignores_field_names_but_rejects_sensitive_values() -> None:
    assert (
        sensitive_value_findings(
            {"authorization": "retention flag only", "secret_present": False}
        )
        == ()
    )
    assert sensitive_value_findings(
        {"safe": "Bearer abcdefghijklmnop", "nested": ["ok", "password=hunter2"]}
    ) == ("$.safe", "$.nested[1]")

    claim = prediction_for(v3_bundle()[1].truths[0]).claims[0]
    values = claim.model_dump(mode="python")
    values["normalized_value"] = "sk-abcdefghijklmnop"
    with pytest.raises(ValidationError, match="secret-shaped"):
        SemanticClaimProjection(**values)


def test_projection_contract_forbids_raw_prompt_response_and_extra_answer_metadata() -> (
    None
):
    projection = prediction_for(v3_bundle()[1].truths[0])
    values = projection.model_dump(mode="python")
    values["raw_response"] = "full model response"
    with pytest.raises(ValidationError, match="Extra inputs"):
        EvalPredictionV2(**values)

    evidence = (
        projection.claims[0]
        .evidence[0]
        .model_copy(update={"locator": "Bearer abcdefghijklmnop"})
    )
    claim = projection.claims[0].model_copy(update={"evidence": (evidence,)})
    values = projection.model_dump(mode="python")
    values["claims"] = (claim, *projection.claims[1:])
    with pytest.raises(ValidationError, match="secret-shaped"):
        EvalPredictionV2(**values)
