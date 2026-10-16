"""Seeded H text selection and entity instantiation from frozen private templates."""

import hashlib
import hmac
import json
import re
from typing import Literal, Self

from pydantic import Field, model_validator

from .eval_h_contracts import HCase, HCaseId, HLabel, HRecipe, H_SEEDED_GENERATOR, Name
from .gold_v3 import normalized_template
from .gold_v3_contracts import FrozenModel, payload_sha256


class HVariant(FrozenModel):
    variant_id: Name
    case: HCase
    label: HLabel

    @model_validator(mode="after")
    def independent_pair(self) -> Self:
        HRecipe(
            generator_version=H_SEEDED_GENERATOR,
            cases=(self.case,),
            labels=(self.label,),
        )
        return self


class HSlot(FrozenModel):
    case_id: HCaseId
    identities: dict[Name, Name] = Field(min_length=1, max_length=64)
    variants: tuple[HVariant, ...] = Field(min_length=2, max_length=32)

    @model_validator(mode="after")
    def variants_keep_registered_strata(self) -> Self:
        first = self.variants[0]
        names, texts = set(), set()
        for variant in self.variants:
            if variant.variant_id in names:
                raise ValueError("h_generator_duplicate_variant")
            names.add(variant.variant_id)
            label = variant.label
            if (
                label.recommendation,
                label.high_risk,
                bool(label.conflict_pairs),
                bool(label.security_events),
            ) != (
                first.label.recommendation,
                first.label.high_risk,
                bool(first.label.conflict_pairs),
                bool(first.label.security_events),
            ):
                raise ValueError("h_generator_variant_stratum_changed")
            signature = payload_sha256(
                [
                    normalized_template(document.content)
                    for document in variant.case.documents
                ]
            )
            if signature in texts:
                raise ValueError("h_generator_variants_require_substantive_text")
            texts.add(signature)
            required = {row.record_id for row in variant.case.records} | {
                variant.case.entity_root_id
            }
            if required - set(self.identities):
                raise ValueError("h_generator_entity_bindings")
        return self


class HGenerationSpec(FrozenModel):
    generator_version: Literal["v1-07h-material-generator.v2"] = H_SEEDED_GENERATOR
    slots: tuple[HSlot, ...] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def slots_are_unique(self) -> Self:
        if len({slot.case_id for slot in self.slots}) != len(self.slots):
            raise ValueError("h_generator_duplicate_slot")
        self.case_bindings()
        return self

    def case_bindings(self):
        bindings = {}
        for slot in self.slots:
            for variant in slot.variants:
                old = variant.case.case_id
                if old in bindings and bindings[old] != slot.case_id:
                    raise ValueError("h_generator_ambiguous_case_binding")
                bindings[old] = slot.case_id
        return bindings

    def require_formal_quotas(self):
        # Quotas depend on frozen descriptors, before any seed is opened.
        representatives = [slot.variants[0] for slot in self.slots]
        HRecipe(
            generator_version=H_SEEDED_GENERATOR,
            cases=tuple(
                item.case.model_copy(update={"case_id": slot.case_id})
                for slot, item in zip(self.slots, representatives)
            ),
            labels=tuple(
                item.label.model_copy(update={"case_id": slot.case_id})
                for slot, item in zip(self.slots, representatives)
            ),
        ).require_formal_quotas()


def generate(spec: HGenerationSpec, seed: bytes) -> tuple[HRecipe, dict]:
    if len(seed) < 16:
        raise ValueError("h_generation_seed_too_short")

    def digest(name):
        return hmac.new(seed, name.encode(), hashlib.sha256).hexdigest()

    def substitute(value, replacements):
        if type(value) is dict:
            return {key: substitute(item, replacements) for key, item in value.items()}
        if type(value) is list:
            return [substitute(item, replacements) for item in value]
        if type(value) is str:
            if value in replacements:
                return replacements[value]
            pattern = (
                r"(?<![A-Za-z0-9_.:-])(?:"
                + "|".join(
                    re.escape(original)
                    for original in sorted(replacements, key=len, reverse=True)
                )
                + r")(?![A-Za-z0-9_.:-])"
            )
            value = re.sub(pattern, lambda match: replacements[match[0]], value)
        return value

    cases, labels, chosen = [], [], {}
    for slot in spec.slots:
        index = int(digest("variant:" + slot.case_id), 16) % len(slot.variants)
        variant = slot.variants[index]
        # A letter leads every instantiated identifier: an all-hex suffix could carry
        # eleven consecutive digits and trip the product's PII-shaped value guard on a
        # record that is entirely synthetic.
        replacements = {
            name: prefix + "-i" + digest("entity:" + name)[:15]
            for name, prefix in slot.identities.items()
        }
        replacements.update(spec.case_bindings())
        for document in variant.case.documents:
            replacements[document.source_id] = (
                slot.case_id
                + "-"
                + document.role
                + "-"
                + "d"
                + digest(document.source_id)[:7]
            )
            replacements[document.document_id] = (
                "H-DOC-" + slot.case_id + "-d" + digest(document.document_id)[:7]
            )
        body = substitute(variant.case.model_dump(mode="json"), replacements)
        for document in body["documents"]:
            document["content_sha256"] = hashlib.sha256(
                document["content"].encode()
            ).hexdigest()
        cases.append(HCase.model_validate_json(json.dumps(body), strict=True))
        labels.append(
            HLabel.model_validate_json(
                json.dumps(
                    substitute(variant.label.model_dump(mode="json"), replacements)
                ),
                strict=True,
            )
        )
        chosen[slot.case_id] = variant.variant_id
    recipe = HRecipe(
        generator_version=H_SEEDED_GENERATOR, cases=tuple(cases), labels=tuple(labels)
    )
    return recipe, dict(
        generator_version=H_SEEDED_GENERATOR,
        selected_variants=chosen,
        seed_sha256=hashlib.sha256(seed).hexdigest(),
        recipe_sha256=payload_sha256(recipe.model_dump(mode="json")),
    )
