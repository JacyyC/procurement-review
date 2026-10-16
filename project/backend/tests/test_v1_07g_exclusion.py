"""Seen-material exclusion and the pre-registered first-send task index.

Cleaning a file does not restore unseen status, and a scope whose detail is only
sealed outside the repository must block an unseen claim instead of being read as
independence.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from procurement_review.eval_f_context import load_cases
from procurement_review.eval_g_exclusion import (
    ExclusionRegistryV1,
    ExclusionScope,
    UnseenClaimUnsupported,
    assert_unseen,
    build_registry,
    candidate_keys,
    intersections,
    verify_sealed_detail,
)
from procurement_review.eval_g_plan import (
    UnregisteredTask,
    assert_within_plan,
    planned_task_index,
)
from procurement_review.gold_v3_contracts import payload_sha256
from v1_07g_support import frozen_plan


@pytest.fixture(scope="module")
def registry() -> ExclusionRegistryV1:
    return build_registry()


def test_every_consumed_scope_is_registered_at_its_own_granularity(registry):
    scopes = {item.scope_id: item for item in registry.scopes}
    for name in ("gold-set.v1", "gold-set.v2", "gold-set.v3"):
        assert name in scopes and scopes[name].case_count == (
            480 if "v3" not in name else 180
        )
    for suffix in ("v3.1", "v3.2", "v3.3", "v3.4", "v3.5", "v3.6"):
        assert f"gold-set.{suffix}" in scopes
    # The legacy sets never recorded an entity root, and the registry says so
    # rather than pretending the relation was checked.
    assert "entity_root" in scopes["gold-set.v1"].key_kinds_not_recorded
    assert "metamorphic_parent" in scopes["gold-set.v2"].key_kinds_not_recorded
    assert scopes["gold-set.v3.6"].key_kinds_not_recorded == ()
    sealed = scopes["retired-g-h-seen-materials"]
    assert sealed.evidence_status == "registration_only"
    assert sealed.case_count == 475 and sealed.keys == ()
    assert registry.registry_sha256


def test_exclusion_covers_relations_not_only_identifiers(registry):
    consumed = load_cases("dev", "gold-set.v3.6")[0]
    keys = candidate_keys(consumed)
    kinds = {key.split(":", 1)[0] for key in intersections(registry, keys)}
    # A fresh Case ID and fresh content hashes would still be caught by the
    # family, entity root, template and near-duplicate relations.
    assert {
        "case_id",
        "content_sha256",
        "generation_family",
        "entity_root",
        "template",
        "near_duplicate_group",
    } <= kinds
    renamed = [
        key for key in keys if not key.startswith(("case_id:", "content_sha256:"))
    ]
    assert intersections(registry, renamed)
    fresh = (
        "case_id:V107-B-9001",
        "content_sha256:" + payload_sha256({"fresh": "material"}),
        "generation_family:V107-GF-NEW-99",
        "entity_root:V107-ENTITY-NEW-9001",
        "near_duplicate_group:V107-NDG-NEW-9001",
        "template:" + payload_sha256({"fresh": "template"}),
    )
    assert intersections(registry, fresh) == []


def test_an_unverifiable_scope_blocks_the_unseen_claim(registry):
    cases = load_cases("dev", "gold-set.v3.6")[:3]
    with pytest.raises(UnseenClaimUnsupported, match="retired-g-h-seen-materials"):
        assert_unseen(registry, cases)
    verdict = assert_unseen(registry, cases, sealed_detail_verified=True)
    # Consumed material is still reported as consumed, at every granularity.
    assert verdict["unseen"] is False
    assert set(verdict["conflicts"]) == {case.case_id for case in cases}
    assert all(verdict["conflicts"].values())


def test_a_registry_without_readable_gaps_supports_an_independence_claim():
    scope = ExclusionScope(
        scope_id="gold-set.v3.6",
        source_entry="resources/gold_set/v3.6",
        evidence_status="read",
        case_count=1,
        source_fingerprint_count=1,
        key_kinds_available=("case_id",),
        keys_sha256=payload_sha256({"keys": ["case_id:V107-D-0001"]}),
        keys=("case_id:V107-D-0001",),
    )
    registry = ExclusionRegistryV1(scopes=(scope,))
    fresh = load_cases("validation", "gold-set.v3.6")[:2]
    verdict = assert_unseen(registry, fresh)
    assert verdict["unseen"] is True and verdict["conflicts"] == {}
    consumed = load_cases("dev", "gold-set.v3.6")[0]
    assert assert_unseen(registry, [consumed])["unseen"] is False


def test_the_sealed_detail_is_accepted_only_at_its_registered_digest(tmp_path):
    body = json.dumps({"cases": [{}] * 475}).encode("utf-8")
    matched = tmp_path / "sealed.json"
    matched.write_bytes(body)
    scope = ExclusionScope(
        scope_id="retired-g-h-seen-materials",
        source_entry="workspace/docs/work-items/records/v1-07/restart-history.json",
        evidence_status="registration_only",
        case_count=475,
        source_fingerprint_count=0,
        key_kinds_available=(),
        key_kinds_not_recorded=("case_id",),
        keys_sha256=payload_sha256({"keys": []}),
        source_sha256=hashlib.sha256(body).hexdigest(),
    )
    registry = ExclusionRegistryV1(scopes=(scope,))
    assert verify_sealed_detail(matched, registry) is True
    assert verify_sealed_detail(tmp_path / "missing.json", registry) is False
    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(body + b" ")
    assert verify_sealed_detail(tampered, registry) is False
    short = tmp_path / "short.json"
    short.write_text(json.dumps({"cases": [{}] * 474}))
    object.__setattr__(
        scope,
        "source_sha256",
        hashlib.sha256(json.dumps({"cases": [{}] * 474}).encode("utf-8")).hexdigest(),
    )
    # The digest matches but the declared Case count does not: still not evidence.
    assert verify_sealed_detail(short, registry) is False


def test_first_sends_are_bounded_by_the_frozen_plan():
    plan = frozen_plan(["V107-D-0001"])
    index = planned_task_index(plan)
    # Two isolated document branches plus one joint branch, per Case and repeat.
    assert sorted(index) == [
        "V107-D-0001.601.contract",
        "V107-D-0001.601.joint",
        "V107-D-0001.601.policy",
        "V107-D-0001.602.contract",
        "V107-D-0001.602.joint",
        "V107-D-0001.602.policy",
    ]
    assert index["V107-D-0001.601.joint"] == (
        "single_agent",
        "V107-D-0001",
        601,
        "joint",
    )
    assert assert_within_plan(plan, list(index)) == {"planned": 6, "checked": 6}
    # The deterministic procurement branch is not a first send and cannot be
    # smuggled in as one; nor can an unplanned repeat.
    for outside in (
        "V107-D-0001.601.procurement",
        "V107-D-0001.603.policy",
        "V107-D-0002.601.policy",
    ):
        with pytest.raises(UnregisteredTask, match="unregistered first-send tasks"):
            assert_within_plan(plan, [outside])
