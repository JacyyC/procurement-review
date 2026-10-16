"""Independent H exclusion expectations, including the archived original format."""

import hashlib
import json

import pytest

from procurement_review.eval_g_exclusion import (
    ExclusionRegistryV1,
    ExclusionScope,
    UnseenClaimUnsupported,
    assert_unseen,
    build_registry,
    verify_sealed_detail,
)
from procurement_review.gold_v3_contracts import payload_sha256


def registered_detail(path, count):
    return ExclusionRegistryV1(
        scopes=(
            ExclusionScope(
                scope_id="retired-g-h-seen-materials",
                source_entry="archive/seen-materials.json",
                evidence_status="registration_only",
                case_count=count,
                source_fingerprint_count=0,
                key_kinds_available=(),
                keys_sha256=payload_sha256({"keys": []}),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
        )
    )


def test_archived_original_uses_case_ids_not_cases(tmp_path):
    path = tmp_path / "seen.json"
    path.write_text(
        json.dumps(
            {
                "case_ids": [f"V107-B-{index:04d}" for index in range(475)],
                "source_fingerprints": ["[]"],
            }
        )
    )
    assert verify_sealed_detail(path, registered_detail(path, 475)) is True


def test_current_g_authored_material_is_also_consumed():
    scopes = {scope.scope_id: scope for scope in build_registry().scopes}
    assert {"gold-set.v3.7", "gold-set.v3.8"} <= set(scopes)
    assert scopes["gold-set.v3.7"].case_count == 8
    assert scopes["gold-set.v3.8"].case_count == 12


def test_boolean_cannot_substitute_for_loaded_exclusion_keys(tmp_path):
    path = tmp_path / "seen.json"
    path.write_text(
        json.dumps({"case_ids": ["V107-B-0401"], "source_fingerprints": []})
    )
    with pytest.raises(UnseenClaimUnsupported):
        assert_unseen(registered_detail(path, 1), (), sealed_detail_verified=True)


def test_wrong_original_digest_cannot_be_read(tmp_path):
    path = tmp_path / "seen.json"
    path.write_text(json.dumps({"case_ids": ["V107-B-0401"]}))
    registry = registered_detail(path, 1)
    path.write_text(json.dumps({"case_ids": ["V107-B-0402"]}))
    assert verify_sealed_detail(path, registry) is False


def test_inventory_loads_only_original_consumed_keys_and_keeps_archive_read_only(
    tmp_path,
):
    import sqlite3
    from procurement_review.eval_h_history import load_source

    path = tmp_path / "original.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE attempt(run_id,worker,ordinal,case_id,source_fingerprints);"
            "CREATE TABLE input_snapshot(run_id,worker,ordinal,branch);"
        )
        for index, case_id, fingerprint in (
            (1, "V107-D-8001", "consumed"),
            (2, "V107-D-8001", "unconsumed-variant"),
            (3, "V107-B-0001", "consumed"),
        ):
            connection.execute(
                "INSERT INTO attempt VALUES(?,?,?,?,?)",
                (index, "policy", 1, case_id, fingerprint),
            )
            connection.execute(
                "INSERT INTO input_snapshot VALUES(?,?,?,?)",
                (index, "policy", 1, json.dumps(dict(case_id=case_id, sources=[]))),
            )
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_source(
        path, dict(case_ids=["V107-D-8001"], source_fingerprints=["consumed"])
    )
    assert loaded == dict(
        consumed_input_snapshots=[dict(case_id="V107-D-8001", sources=[])]
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not path.with_name(path.name + "-wal").exists()


@pytest.mark.parametrize(
    "parent,complete",
    [("absent", False), (None, True), (False, False), ("V107-D-6000", True)],
)
def test_history_parent_relation_requires_explicit_null_or_identity(
    tmp_path, monkeypatch, parent, complete
):
    """An absent parent key leaves the registration incomplete here because this
    record also lacks the historical scope the absence exemption requires."""
    from procurement_review import eval_h_exclusion
    from procurement_review.eval_h_storage import sha, write

    seen = tmp_path / "seen.json"
    write(seen, dict(case_ids=["V107-D-6001"], source_fingerprints=["[]"]))
    monkeypatch.setattr(
        eval_h_exclusion, "build_registry", lambda: registered_detail(seen, 1)
    )
    case = dict(
        case_id="V107-D-6001",
        generation_family_id="historical-family",
        entity_root_id="historical-entity",
        near_duplicate_group_id="historical-group",
        policy_text="Historical substantive policy material has an explicit relation record.",
    )
    if parent != "absent":
        case["metamorphic_parent_id"] = parent
    source = tmp_path / "consumed.json"
    write(source, dict(cases=[case]))
    registry = eval_h_exclusion.register(seen, {source: sha(source)})
    assert registry["history_complete"] is complete
    assert (case["case_id"] in registry["incomplete_relations"]) is (not complete)


def authored(seen, tmp_path, monkeypatch, cases):
    from procurement_review import eval_h_exclusion
    from procurement_review.eval_h_storage import sha, write

    monkeypatch.setattr(
        eval_h_exclusion, "build_registry", lambda: registered_detail(seen, len(cases))
    )
    source = tmp_path / "consumed.json"
    write(source, dict(cases=list(cases)), replace=source.exists())
    return eval_h_exclusion, eval_h_exclusion.register(seen, {source: sha(source)})


def authored_case(case_id, family, scope):
    return dict(
        case_id=case_id,
        family=family,
        scope_id=scope,
        policy_text=f"Authored material for {case_id} with its own substantive body.",
    )


def test_unrecorded_relations_are_accepted_only_without_a_sibling(
    tmp_path, monkeypatch
):
    from procurement_review.eval_h_storage import write

    seen = tmp_path / "seen.json"
    write(
        seen,
        dict(case_ids=["V107-D-6001", "V107-D-6002"], source_fingerprints=["[]"]),
    )
    cases = [
        authored_case("V107-D-6001", "family-a", "scope-a"),
        authored_case("V107-D-6002", "family-b", "scope-b"),
    ]
    exclusion, registry = authored(seen, tmp_path, monkeypatch, cases)
    assert registry["schema_version"] == "v1-07h-exclusion.v2"
    assert registry["history_complete"] is True
    assert registry["incomplete_relations"] == []
    assert registry["accepted_not_recorded"] == {
        "V107-D-6001": ["entity_root", "metamorphic_parent", "near_duplicate_group"],
        "V107-D-6002": ["entity_root", "metamorphic_parent", "near_duplicate_group"],
    }
    assert exclusion.check(registry, (), formal=True)["history_complete"] is True

    # A shared generation family is a sibling relation and forbids the exemption.
    _, registry = authored(
        seen,
        tmp_path,
        monkeypatch,
        [cases[0], authored_case("V107-D-6002", "family-a", "scope-b")],
    )
    assert registry["history_complete"] is False
    assert registry["incomplete_relations"] == ["V107-D-6001", "V107-D-6002"]
    with pytest.raises(ValueError, match="h_history_material_incomplete"):
        exclusion.check(registry, (), formal=True)

    # A shared historical scope is a sibling relation too.
    _, registry = authored(
        seen,
        tmp_path,
        monkeypatch,
        [cases[0], authored_case("V107-D-6002", "family-b", "scope-a")],
    )
    assert registry["history_complete"] is False
    assert registry["incomplete_relations"] == ["V107-D-6001", "V107-D-6002"]

    # A kind the same source assigns to any case stays required for its siblings.
    with_root = authored_case("V107-D-6001", "family-a", "scope-a")
    with_root["entity_root_id"] = "root-a"
    _, registry = authored(
        seen,
        tmp_path,
        monkeypatch,
        [with_root, authored_case("V107-D-6002", "family-b", "scope-b")],
    )
    assert registry["history_complete"] is False
    assert registry["incomplete_relations"] == ["V107-D-6002"]

    # A case without a recorded family cannot claim siblingless absence.
    _, registry = authored(
        seen,
        tmp_path,
        monkeypatch,
        [
            dict(
                case_id="V107-D-6001",
                scope_id="scope-a",
                policy_text="Body without a family record at all.",
            ),
            authored_case("V107-D-6002", "family-b", "scope-b"),
        ],
    )
    assert registry["history_complete"] is False
    assert registry["incomplete_relations"] == ["V107-D-6001"]


def test_verify_reads_both_versions_and_rejects_tampering(tmp_path):
    from procurement_review.eval_h_exclusion import verify
    from procurement_review.eval_h_storage import sha, write

    seen = tmp_path / "seen.json"
    write(seen, dict(case_ids=[], source_fingerprints=[]))
    for version, complete in (
        ("v1-07h-exclusion.v1", False),
        ("v1-07h-exclusion.v2", True),
    ):
        body = dict(
            schema_version=version,
            source_sha256={str(seen): sha(seen)},
            history_complete=complete,
        )
        body["registry_sha256"] = payload_sha256(body)
        verify(body)
        with pytest.raises(ValueError, match="registry_identity"):
            verify(dict(body, history_complete=not complete))
        with pytest.raises(ValueError, match="registry_identity"):
            verify(dict(body, schema_version="v1-07h-exclusion.v3"))
