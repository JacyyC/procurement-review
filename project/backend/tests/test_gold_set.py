from __future__ import annotations

import json
from collections import Counter

import pytest
from pydantic import ValidationError

from procurement_review.gold_set import (
    GOLD_CASE_COUNT,
    GOLD_CASES_PER_FAMILY,
    GOLD_GENERATOR_SEED,
    GOLD_SPLIT_ALLOCATION,
    GoldCaseFamily,
    GoldSetCheckError,
    GoldSplit,
    check_gold_set,
    case_input_checksum,
    generate_gold_set,
    label_checksum,
    load_gold_cases,
    load_gold_labels,
    load_gold_labels_for_split,
    model_context,
    validate_gold_set,
    write_gold_set,
    _build_files,
)
from procurement_review.contracts import WorkerKind


def test_generated_gold_set_matches_frozen_contract() -> None:
    cases, labels, manifest = generate_gold_set()
    assert len(cases) == GOLD_CASE_COUNT
    assert len(labels) == GOLD_CASE_COUNT
    assert Counter(case.split for case in cases) == {
        GoldSplit.DEV: 240,
        GoldSplit.VALIDATION: 120,
        GoldSplit.HOLDOUT: 120,
    }
    assert Counter(case.case_family_id for case in cases) == {
        f"GF-{family.value}": GOLD_CASES_PER_FAMILY for family in GoldCaseFamily
    }
    assert all(
        Counter(
            case.split for case in cases if case.case_family_id == f"GF-{family.value}"
        )
        == {
            GoldSplit.DEV: GOLD_SPLIT_ALLOCATION["dev"],
            GoldSplit.VALIDATION: GOLD_SPLIT_ALLOCATION["validation"],
            GoldSplit.HOLDOUT: GOLD_SPLIT_ALLOCATION["holdout"],
        }
        for family in GoldCaseFamily
    )
    assert len({case.group_id for case in cases}) == GOLD_CASE_COUNT
    assert all(
        "holdout" in case.scenario_tags
        for case in cases
        if case.split is GoldSplit.HOLDOUT
    )
    cases_file, labels_file, manifest_file = _build_files(cases, labels, manifest)
    validate_gold_set(cases_file, labels_file, manifest_file)


def test_committed_gold_set_is_reproducible_and_readable() -> None:
    manifest = check_gold_set()
    assert manifest.case_count == GOLD_CASE_COUNT
    assert manifest.label_count == GOLD_CASE_COUNT
    assert manifest.manifest_sha256
    assert manifest.corpus_snapshot_id == "synthetic-gold-corpus.v1"
    assert manifest.gold_set_version == "gold-set.v1"
    # The C-1/C-2 run manifests and the V1-02B Smoke are anchored to this digest.
    assert (
        manifest.manifest_sha256
        == "871be1687f3ddcb25080bdba9947ae935062a2f4a28f8258293542c9d54b916f"
    )


def test_generation_is_deterministic_and_seeded() -> None:
    first = generate_gold_set(seed=GOLD_GENERATOR_SEED)
    second = generate_gold_set(seed=GOLD_GENERATOR_SEED)
    changed = generate_gold_set(seed=GOLD_GENERATOR_SEED + 1)

    assert first == second
    assert first[2].cases_payload_sha256 != changed[2].cases_payload_sha256
    assert first[0][0].source_checksum != changed[0][0].source_checksum


def test_case_context_excludes_labels_and_hidden_split_metadata() -> None:
    cases, _, _ = generate_gold_set()
    forbidden = {
        "label_id",
        "source_checksum",
        "case_family_id",
        "split",
        "scenario_tags",
        "failure_mode",
    }
    for case in (cases[0], cases[119], cases[-1]):
        for worker in WorkerKind:
            context = model_context(case, worker)
            assert not forbidden.intersection(context)
            assert not any(key.startswith("expected_") for key in context)
            serialized = json.dumps(context, ensure_ascii=False)
            assert "gold-label" not in serialized


def test_case_and_label_checksums_are_bound_to_their_inputs() -> None:
    cases, labels, _ = generate_gold_set()
    assert all(
        case.source_checksum == case_input_checksum(case.model_input) for case in cases
    )
    assert all(
        label.label_checksum == f"sha256:{label_checksum(label)}" for label in labels
    )


def test_written_gold_set_rejects_tampering_and_duplicate_keys(tmp_path) -> None:
    write_gold_set(tmp_path)
    assert check_gold_set(tmp_path).case_count == GOLD_CASE_COUNT

    cases_path = tmp_path / "cases.v1.json"
    cases_data = json.loads(cases_path.read_text(encoding="utf-8"))
    cases_data["cases"][0]["question"] = "被篡改的合成问题"
    cases_path.write_text(json.dumps(cases_data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises((ValidationError, GoldSetCheckError)):
        load_gold_cases(cases_path)

    duplicate_dir = tmp_path / "duplicate"
    write_gold_set(duplicate_dir)
    manifest_path = duplicate_dir / "manifest.v1.json"
    manifest_path.write_text(
        '{"schema_version":"gold-manifest.v1","schema_version":"duplicate"}',
        encoding="utf-8",
    )
    with pytest.raises(GoldSetCheckError, match="duplicate JSON key"):
        check_gold_set(duplicate_dir)


def test_holdout_labels_are_explicitly_protected(tmp_path) -> None:
    write_gold_set(tmp_path)
    labels_path = tmp_path / "labels.v1.json"
    cases_path = tmp_path / "cases.v1.json"

    with pytest.raises(GoldSetCheckError, match="hidden holdout"):
        load_gold_labels(labels_path)
    assert (
        len(
            load_gold_labels_for_split(
                GoldSplit.DEV,
                cases_path=cases_path,
                labels_path=labels_path,
            )
        )
        == 240
    )
    with pytest.raises(GoldSetCheckError, match="holdout"):
        load_gold_labels_for_split(
            GoldSplit.HOLDOUT,
            cases_path=cases_path,
            labels_path=labels_path,
        )
    assert (
        len(
            load_gold_labels_for_split(
                GoldSplit.HOLDOUT,
                cases_path=cases_path,
                labels_path=labels_path,
                allow_holdout=True,
            )
        )
        == 120
    )
