"""Seed protocol uses disposable authored templates; no formal package is generated."""

import copy
import hashlib
import json

import pytest

from procurement_review.eval_h_contracts import HRecipe
from procurement_review.eval_h_exclusion import check, relations, texts
from procurement_review.eval_h_fixture import fixture_case, fixture_label
from procurement_review.eval_h_generation import HGenerationSpec, generate
from procurement_review.gold_v38_dev_quality import dev_quality_cases
from procurement_review.gold_v3_contracts import payload_sha256


def disposable_spec():
    authored = dev_quality_cases()[0]
    case, label = fixture_case(authored), fixture_label(authored)
    first = dict(
        variant_id="disposable-text-a",
        case=case.model_dump(mode="json"),
        label=label.model_dump(mode="json"),
    )
    second = copy.deepcopy(first)
    second["variant_id"] = "disposable-text-b"
    for document in second["case"]["documents"]:
        document["content"] = "\n".join(reversed(document["content"].splitlines()))
        document["content_sha256"] = hashlib.sha256(
            document["content"].encode()
        ).hexdigest()
    identities = {record.record_id: "H-" + record.kind for record in case.records}
    identities[case.entity_root_id] = "H-ENTITY"
    for record in case.records:
        for field in ("contract_reference", "counterparty_reference"):
            if record.payload.get(field):
                identities[record.payload[field]] = "H-CONTRACT"
    body = dict(
        generator_version="v1-07h-material-generator.v2",
        slots=[
            dict(case_id="V107-D-9901", identities=identities, variants=[first, second])
        ],
    )
    return HGenerationSpec.model_validate_json(json.dumps(body), strict=True)


def test_disposable_seed_selects_text_and_instantiates_bound_entities():
    spec = disposable_spec()
    recipe, selection = generate(spec, b"a" * 32)
    repeated, proof = generate(spec, b"a" * 32)
    assert recipe == repeated and selection == proof
    case, label = recipe.cases[0], recipe.labels[0]
    assert case.case_id == label.case_id == "V107-D-9901"
    assert case.purchase_order_id.startswith("H-purchase_order-")
    assert case.purchase_order_id != spec.slots[0].variants[0].case.purchase_order_id
    assert label.recommendation.value == "recommend_approve"
    variants, orders = set(), set()
    for number in range(16):
        item, chosen = generate(spec, bytes([number]) * 32)
        variants.add(chosen["selected_variants"]["V107-D-9901"])
        orders.add(item.cases[0].purchase_order_id)
        HRecipe.model_validate_json(item.model_dump_json(), strict=True)
    assert variants == {"disposable-text-a", "disposable-text-b"}
    assert len(orders) == 16
    with pytest.raises(ValueError, match="200|class_quotas"):
        spec.require_formal_quotas()
    with pytest.raises(ValueError, match="seed_too_short"):
        generate(spec, b"short")


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("same_text", "substantive_text"),
        ("same_name", "duplicate_variant"),
        ("stratum", "stratum_changed"),
        ("identity", "entity_bindings"),
    ],
)
def test_generator_rejects_superficial_or_unbound_variants(mutation, reason):
    body = disposable_spec().model_dump(mode="json")
    slot = body["slots"][0]
    if mutation == "same_text":
        slot["variants"][1]["case"] = copy.deepcopy(slot["variants"][0]["case"])
    elif mutation == "same_name":
        slot["variants"][1]["variant_id"] = slot["variants"][0]["variant_id"]
    elif mutation == "stratum":
        slot["variants"][1]["label"]["recommendation"] = "recommend_conditional"
    else:
        slot["identities"].clear()
    with pytest.raises(
        ValueError,
        match=reason if mutation != "identity" else "at least|entity_bindings",
    ):
        HGenerationSpec.model_validate_json(json.dumps(body), strict=True)


def test_seed_instantiation_keeps_cross_slot_metamorphic_lineage():
    body = disposable_spec().model_dump(mode="json")
    second = copy.deepcopy(body["slots"][0])
    second["case_id"] = "V107-D-9902"
    for variant in second["variants"]:
        variant["case"]["metamorphic_parent_id"] = variant["case"]["case_id"]
        variant["case"]["case_id"] = "V107-D-9003"
        variant["label"]["case_id"] = "V107-D-9003"
    body["slots"].append(second)
    spec = HGenerationSpec.model_validate_json(json.dumps(body), strict=True)
    recipe, _ = generate(spec, b"c" * 32)
    assert (
        recipe.cases[1].metamorphic_parent_id
        == recipe.cases[0].case_id
        == "V107-D-9901"
    )


def registry(material=(), keys=(), complete=True):
    body = dict(
        schema_version="v1-07h-exclusion.v1",
        source_sha256={},
        keys=list(keys),
        materials=list(material),
        history_complete=complete,
    )
    body["registry_sha256"] = payload_sha256(body)
    return body


def test_historical_embedded_sources_are_read_without_inventing_entity_roots():
    historical = dict(
        case_id="V107-D-8001",
        family="recorded-family",
        scope_id="recorded-scope",
        sources=[
            dict(
                content="A saved procurement clause with its original substantive body."
            )
        ],
    )
    original = dict(branch=json.dumps(historical))
    assert list(texts(original)) == [historical["sources"][0]["content"]]
    keys = relations(original)
    assert "case_id:V107-D-8001" in keys and "generation_family:recorded-family" in keys
    assert "historical_scope:recorded-scope" in keys
    assert not any(key.startswith("entity_root:") for key in keys)


def test_exclusion_rejects_reuse_padding_and_missing_history():
    case = fixture_case(dev_quality_cases()[0])
    known = registry(material=[case.documents[0].content])
    checked = check(known, [case], formal=False)
    assert checked["conflicts"][case.case_id]
    body = case.model_dump(mode="json")
    document = body["documents"][0]
    document["content"] += (
        "\nAn unrelated extra paragraph cannot make consumed rules unseen."
    )
    document["content_sha256"] = hashlib.sha256(
        document["content"].encode()
    ).hexdigest()
    from procurement_review.eval_h_contracts import HCase

    padded = HCase.model_validate_json(json.dumps(body), strict=True)
    assert check(known, [padded], formal=False)["conflicts"][case.case_id]
    with pytest.raises(ValueError, match="history_material_incomplete"):
        check(registry(complete=False), [case], formal=True)
