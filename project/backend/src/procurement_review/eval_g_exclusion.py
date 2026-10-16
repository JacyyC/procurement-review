"""Seen-material exclusion for the H blind.

Deleting a file does not restore unseen status, and neither a Case ID nor a
content hash is enough on its own: two Cases can share a generation family, an
entity root, a template or a near-duplicate edge while differing in both
identifiers.  Where the only evidence for a consumed scope is its sealed
registration, this module says so and refuses to support an unseen claim instead
of quietly treating the gap as independence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .eval_g_plan import RESTART_HISTORY
from .eval_v2_contracts import EvalV2Model
from .gold_v3 import normalized_template
from .gold_v3_contracts import HexSha256, payload_sha256

EXCLUSION_REGISTRY_V1 = "v1-07g-exclusion-registry.v1"
PACKAGE = Path(__file__).parent
GOLD_SET_RESOURCES = PACKAGE / "resources" / "gold_set"
V3_VERSIONS = ("v3", "v3.1", "v3.2", "v3.3", "v3.4", "v3.5", "v3.6")

KeyKind = Literal[
    "case_id",
    "content_sha256",
    "generation_family",
    "entity_root",
    "template",
    "near_duplicate_group",
    "metamorphic_parent",
]


class ExclusionScope(EvalV2Model):
    """One consumed material scope with the keys it can actually prove."""

    scope_id: str = Field(min_length=1, max_length=120)
    source_entry: str = Field(min_length=1, max_length=240)
    evidence_status: Literal["read", "registration_only"]
    case_count: int = Field(ge=0)
    source_fingerprint_count: int = Field(ge=0)
    key_kinds_available: tuple[KeyKind, ...]
    key_kinds_not_recorded: tuple[KeyKind, ...] = ()
    keys_sha256: HexSha256
    keys: tuple[str, ...] = ()
    source_sha256: HexSha256 | None = None

    @model_validator(mode="after")
    def keys_are_present_only_when_they_were_read(self) -> "ExclusionScope":
        if self.evidence_status == "read":
            if not self.keys:
                raise ValueError("a read scope must carry its exclusion keys")
            if self.key_kinds_not_recorded and set(self.key_kinds_not_recorded) & set(
                self.key_kinds_available
            ):
                raise ValueError("a key kind cannot be both available and missing")
        elif self.keys:
            raise ValueError(
                "a registration-only scope has no readable keys to exclude on"
            )
        if self.keys_sha256 != payload_sha256({"keys": sorted(set(self.keys))}):
            raise ValueError("exclusion key digest does not match the keys")
        return self


class ExclusionRegistryV1(EvalV2Model):
    schema_version: Literal["v1-07g-exclusion-registry.v1"] = EXCLUSION_REGISTRY_V1
    scopes: tuple[ExclusionScope, ...] = Field(min_length=1)
    registry_sha256: HexSha256 | None = None

    @model_validator(mode="after")
    def scopes_are_unique(self) -> "ExclusionRegistryV1":
        ids = [item.scope_id for item in self.scopes]
        if len(set(ids)) != len(ids):
            raise ValueError("one consumed scope cannot be registered twice")
        return self

    def keys(self) -> frozenset[str]:
        return frozenset(key for scope in self.scopes for key in scope.keys)

    def unverifiable_scopes(self) -> tuple[str, ...]:
        return tuple(
            scope.scope_id for scope in self.scopes if scope.evidence_status != "read"
        )


def _key(kind: KeyKind, value: str) -> str:
    return f"{kind}:{value}"


def _digest(keys: Iterable[str]) -> str:
    return payload_sha256({"keys": sorted(set(keys))})


def _template_of(text: str) -> str:
    return _key(
        "template",
        hashlib.sha256(normalized_template(text).encode("utf-8")).hexdigest(),
    )


def v3_scope(version: str) -> ExclusionScope:
    """Read one in-repo v3 dataset at every granularity it records."""

    from .eval_f_context import data_directory, load_cases

    gold_version = version if version.startswith("gold-set.") else f"gold-set.{version}"
    short = gold_version.removeprefix("gold-set.")
    cases = tuple(load_cases(split, gold_version) for split in ("dev", "validation"))
    flat = [case for group in cases for case in group]
    keys: list[str] = []
    for case in flat:
        keys.append(_key("case_id", case.case_id))
        keys.append(_key("content_sha256", case.source_checksum))
        keys.append(_key("generation_family", case.generation_family_id))
        keys.append(_key("entity_root", case.entity_root_id))
        keys.append(_key("near_duplicate_group", case.near_duplicate_group_id))
        if case.metamorphic_parent_id:
            keys.append(_key("metamorphic_parent", case.metamorphic_parent_id))
        keys.append(_template_of(case.model_input.question))
        for document in (
            case.model_input.policy_document,
            case.model_input.contract_document,
        ):
            keys.append(_template_of(document.content))
            keys.append(_key("content_sha256", document.content_sha256))
    return ExclusionScope(
        scope_id=f"gold-set.{short}",
        source_entry=str(data_directory(gold_version).relative_to(PACKAGE.parents[2])),
        evidence_status="read",
        case_count=len(flat),
        source_fingerprint_count=len(
            {key for key in keys if key.startswith("content_sha256:")}
        ),
        key_kinds_available=(
            "case_id",
            "content_sha256",
            "generation_family",
            "entity_root",
            "template",
            "near_duplicate_group",
            "metamorphic_parent",
        ),
        keys_sha256=_digest(keys),
        keys=tuple(sorted(set(keys))),
        source_sha256=hashlib.sha256(
            (data_directory(gold_version) / "cases.json").read_bytes()
        ).hexdigest(),
    )


def legacy_scope(name: str) -> ExclusionScope:
    """Read one legacy gold set at the granularity it actually recorded."""

    path = GOLD_SET_RESOURCES / f"cases.{name}.json"
    body = json.loads(path.read_bytes())
    keys: list[str] = []
    for case in body["cases"]:
        keys.append(_key("case_id", case["case_id"]))
        keys.append(_key("content_sha256", case["source_checksum"]))
        keys.append(_key("generation_family", case["case_family_id"]))
        keys.append(_key("near_duplicate_group", case["group_id"]))
        keys.append(_template_of(str(case["question"])))
        model_input = case.get("model_input") or {}
        for value in _document_texts(model_input):
            keys.append(_template_of(value))
    return ExclusionScope(
        scope_id=f"gold-set.{name}",
        source_entry=str(path.relative_to(PACKAGE.parents[2])),
        evidence_status="read",
        case_count=len(body["cases"]),
        source_fingerprint_count=len(
            {key for key in keys if key.startswith("content_sha256:")}
        ),
        key_kinds_available=(
            "case_id",
            "content_sha256",
            "generation_family",
            "template",
            "near_duplicate_group",
        ),
        key_kinds_not_recorded=("entity_root", "metamorphic_parent"),
        keys_sha256=_digest(keys),
        keys=tuple(sorted(set(keys))),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _document_texts(model_input: Mapping[str, Any]) -> Iterable[str]:
    for value in model_input.values():
        if isinstance(value, Mapping) and isinstance(value.get("content"), str):
            yield value["content"]
        elif isinstance(value, Sequence) and not isinstance(value, str):
            for item in value:
                if isinstance(item, Mapping) and isinstance(item.get("content"), str):
                    yield item["content"]


def sealed_scope() -> ExclusionScope:
    """The retired G/H consumption, known only by its sealed registration."""

    registered = json.loads(RESTART_HISTORY.read_text(encoding="utf-8"))
    return ExclusionScope(
        scope_id="retired-g-h-seen-materials",
        source_entry=str(RESTART_HISTORY.relative_to(PACKAGE.parents[3])),
        evidence_status="registration_only",
        case_count=registered["seen_case_count"],
        source_fingerprint_count=0,
        key_kinds_available=(),
        key_kinds_not_recorded=(
            "case_id",
            "content_sha256",
            "generation_family",
            "entity_root",
            "template",
            "near_duplicate_group",
            "metamorphic_parent",
        ),
        keys_sha256=_digest(()),
        source_sha256=registered["seen_materials_sha256"],
    )


def build_registry(*, include_legacy: bool = True) -> ExclusionRegistryV1:
    """Freeze every consumed scope the repository can actually evidence."""

    scopes: list[ExclusionScope] = []
    if include_legacy:
        scopes.extend(legacy_scope(name) for name in ("v1", "v2"))
    for version in V3_VERSIONS:
        directory = GOLD_SET_RESOURCES / version if version != "v3" else None
        if version == "v3":
            if not (GOLD_SET_RESOURCES / "cases.v3.json").exists():
                continue
            scopes.append(_v3_root_scope())
            continue
        if directory is not None and directory.exists():
            scopes.append(v3_scope(version))
    scopes.extend(authored_scope(version) for version in ("v3.7", "v3.8"))
    scopes.append(sealed_scope())
    registry = ExclusionRegistryV1(scopes=tuple(scopes))
    return registry.model_copy(
        update={
            "registry_sha256": payload_sha256(
                registry.model_dump(mode="json", exclude={"registry_sha256"})
            )
        }
    )


def _v3_root_scope() -> ExclusionScope:
    from .gold_v3 import load_bundle

    cases, _truths, _manifest = load_bundle()
    keys: list[str] = []
    for case in cases.cases:
        keys.append(_key("case_id", case.case_id))
        keys.append(_key("content_sha256", case.source_checksum))
        keys.append(_key("generation_family", case.generation_family_id))
        keys.append(_key("entity_root", case.entity_root_id))
        keys.append(_key("near_duplicate_group", case.near_duplicate_group_id))
        if case.metamorphic_parent_id:
            keys.append(_key("metamorphic_parent", case.metamorphic_parent_id))
        keys.append(_template_of(case.model_input.question))
        for document in (
            case.model_input.policy_document,
            case.model_input.contract_document,
        ):
            keys.append(_template_of(document.content))
            keys.append(_key("content_sha256", document.content_sha256))
    return ExclusionScope(
        scope_id="gold-set.v3",
        source_entry="resources/gold_set/cases.v3.json",
        evidence_status="read",
        case_count=len(cases.cases),
        source_fingerprint_count=len(
            {key for key in keys if key.startswith("content_sha256:")}
        ),
        key_kinds_available=(
            "case_id",
            "content_sha256",
            "generation_family",
            "entity_root",
            "template",
            "near_duplicate_group",
            "metamorphic_parent",
        ),
        keys_sha256=_digest(keys),
        keys=tuple(sorted(set(keys))),
        source_sha256=hashlib.sha256(
            (GOLD_SET_RESOURCES / "cases.v3.json").read_bytes()
        ).hexdigest(),
    )


def candidate_keys(case: Any) -> tuple[str, ...]:
    """Every exclusion key a candidate Case carries, at every recorded granularity."""

    keys = [
        _key("case_id", case.case_id),
        _key("content_sha256", case.source_checksum),
        _key("generation_family", case.generation_family_id),
        _key("entity_root", case.entity_root_id),
        _key("near_duplicate_group", case.near_duplicate_group_id),
        _template_of(case.model_input.question),
    ]
    if case.metamorphic_parent_id:
        keys.append(_key("metamorphic_parent", case.metamorphic_parent_id))
    for document in (
        case.model_input.policy_document,
        case.model_input.contract_document,
    ):
        keys.append(_template_of(document.content))
        keys.append(_key("content_sha256", document.content_sha256))
    return tuple(keys)


def intersections(registry: ExclusionRegistryV1, keys: Iterable[str]) -> list[str]:
    known = registry.keys()
    return sorted(set(keys) & known)


class UnseenClaimUnsupported(RuntimeError):
    """Raised when the evidence cannot support an unseen claim."""


def assert_unseen(
    registry: ExclusionRegistryV1,
    cases: Iterable[Any],
    *,
    sealed_detail_verified: bool = False,
) -> dict[str, Any]:
    """Require independence from every consumed scope, or refuse the claim."""

    blocked = sorted(registry.unverifiable_scopes())
    conflicts: dict[str, list[str]] = {}
    for case in cases:
        hits = intersections(registry, candidate_keys(case))
        if hits:
            conflicts[case.case_id] = hits
    # Known overlap can always disprove independence. A Boolean cannot prove
    # independence when the actual exclusion keys were never loaded.
    if blocked and not conflicts:
        raise UnseenClaimUnsupported(
            "consumed scopes without readable keys: " + ", ".join(blocked)
        )
    if blocked and not sealed_detail_verified:
        raise UnseenClaimUnsupported(
            "consumed scopes without readable keys: " + ", ".join(blocked)
        )
    return {
        "registry_sha256": registry.registry_sha256,
        "scopes": [item.scope_id for item in registry.scopes],
        "unseen": not conflicts,
        "conflicts": conflicts,
        "sealed_detail_verified": sealed_detail_verified,
    }


def verify_sealed_detail(path: Path, registry: ExclusionRegistryV1) -> bool:
    """Accept the sealed seen-materials detail only at its registered digest."""

    scope = next(
        item
        for item in registry.scopes
        if item.scope_id == "retired-g-h-seen-materials"
    )
    if not Path(path).exists():
        return False
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != scope.source_sha256:
        return False
    detail = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = detail.get("case_ids")
    if ids is not None:
        return (
            type(ids) is list
            and all(type(item) is str and item for item in ids)
            and len(ids) == len(set(ids)) == scope.case_count
        )
    return len(detail.get("cases", ())) == scope.case_count


def authored_cases(version: str) -> tuple:
    from .gold_v37_conditional import as_gold_case, conditional_cases
    from .gold_v38_dev_quality import dev_quality_cases, mapped_case

    if version == "v3.7":
        return tuple(
            as_gold_case(case).model_copy(update={"case_id": case.case_id})
            for case in conditional_cases()
        )
    if version == "v3.8":
        return tuple(
            as_gold_case(mapped_case(case)).model_copy(update={"case_id": case.case_id})
            for case in dev_quality_cases()
        )
    raise ValueError("unsupported authored exclusion version")


def authored_scope(version: str) -> ExclusionScope:
    cases = authored_cases(version)
    keys = sorted({key for case in cases for key in candidate_keys(case)})
    return ExclusionScope(
        scope_id="gold-set." + version,
        source_entry="src/procurement_review/"
        + {
            "v3.7": "gold_v37_conditional.py",
            "v3.8": "gold_v38_dev_quality.py",
        }[version],
        evidence_status="read",
        case_count=len(cases),
        source_fingerprint_count=len(
            {key for key in keys if key.startswith("content_sha256:")}
        ),
        key_kinds_available=(
            "case_id",
            "content_sha256",
            "generation_family",
            "entity_root",
            "template",
            "near_duplicate_group",
            "metamorphic_parent",
        ),
        keys_sha256=_digest(keys),
        keys=tuple(keys),
        source_sha256=payload_sha256([case.model_dump(mode="json") for case in cases]),
    )


def load_sealed_detail(path: Path, registry: ExclusionRegistryV1) -> ExclusionScope:
    """Read the registered original, including its actual source identity arrays."""
    if not verify_sealed_detail(path, registry):
        raise UnseenClaimUnsupported("seen_materials_original_identity")
    detail = json.loads(Path(path).read_bytes())
    if type(detail.get("case_ids")) is not list:
        raise UnseenClaimUnsupported("seen_materials_case_ids_required")
    keys = {_key("case_id", item) for item in detail["case_ids"]}
    fingerprints = detail.get("source_fingerprints")
    if type(fingerprints) is not list:
        raise UnseenClaimUnsupported("seen_materials_source_fingerprints_required")
    for fingerprint in fingerprints:
        try:
            sources = json.loads(fingerprint)
            if type(sources) is not list:
                raise ValueError
            for source in sources:
                if set(source) != {"source_id", "version", "locator", "content_sha256"}:
                    raise ValueError
                if not all(type(value) is str and value for value in source.values()):
                    raise ValueError
                if len(source["content_sha256"]) != 64:
                    raise ValueError
                keys.add(_key("content_sha256", source["content_sha256"]))
        except (ValueError, TypeError, KeyError) as error:
            raise UnseenClaimUnsupported("seen_materials_source_identity") from error
    return ExclusionScope(
        scope_id="retired-g-h-seen-materials",
        source_entry="archive/seen-materials.json",
        evidence_status="read",
        case_count=len(detail["case_ids"]),
        source_fingerprint_count=len(fingerprints),
        key_kinds_available=("case_id", "content_sha256"),
        key_kinds_not_recorded=(
            "generation_family",
            "entity_root",
            "template",
            "near_duplicate_group",
            "metamorphic_parent",
        ),
        keys_sha256=_digest(keys),
        keys=tuple(sorted(keys)),
        source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )


__all__ = [
    "EXCLUSION_REGISTRY_V1",
    "ExclusionRegistryV1",
    "ExclusionScope",
    "UnseenClaimUnsupported",
    "assert_unseen",
    "build_registry",
    "candidate_keys",
    "intersections",
    "legacy_scope",
    "sealed_scope",
    "v3_scope",
    "verify_sealed_detail",
]
