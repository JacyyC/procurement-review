"""H exclusion binds the original keys to readable material and checks graph edges.

Version 2 judges a consumed case's relation completeness against the kinds its own
original records could actually carry: a kind that no record of the case's sources
ever assigns is explicitly registered as not recorded instead of failing the case,
and that exemption is only granted when the case's sibling records show a unique
generation family and a unique historical scope, so no sibling relation exists
that the missing kind could have described.  Version 1 registries stay readable.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from .eval_g_exclusion import (
    authored_cases,
    build_registry,
    load_sealed_detail,
)
from .eval_h_contracts import HCase
from .eval_h_storage import read, sha, write
from .gold_v3 import normalized_template
from .gold_v3_contracts import payload_sha256

VERSION = "v1-07h-exclusion.v2"
READABLE_VERSIONS = ("v1-07h-exclusion.v1", VERSION)
RELATIONS = {
    "case_id": "case_id",
    "generation_family_id": "generation_family",
    "case_family_id": "generation_family",
    "entity_root_id": "entity_root",
    "near_duplicate_group_id": "near_duplicate_group",
    "group_id": "near_duplicate_group",
    "metamorphic_parent_id": "metamorphic_parent",
    "template_id": "template",
    "family": "generation_family",
    "scope_id": "historical_scope",
}


def kind_index(value):
    """Every recorded value per relation kind, alias- and embedding-aware."""

    index = {}
    for key in relations(value):
        kind, _, item = key.partition(":")
        index.setdefault(kind, set()).add(item)
    return index


def carries_material(value) -> bool:
    """True when a case record carries bodies of its own, not just references."""

    for _body in texts(value):
        return True
    return False


def embedded(name, item, owner):
    if type(item) is str and (
        name == "branch" or (name == "content" and owner.get("role") == "user")
    ):
        try:
            parsed = json.loads(item)
            return parsed if isinstance(parsed, (dict, list)) else None
        except ValueError:
            return None
    return None


def texts(value) -> Iterable[str]:
    if isinstance(value, Mapping):
        for name, item in value.items():
            parsed = embedded(name, item, value)
            if parsed is not None:
                yield from texts(parsed)
                continue
            if (
                name in {"content", "policy_text", "contract_text"}
                and type(item) is str
            ):
                yield item
            elif name not in {"labels", "truth", "truths", "truth_extension"}:
                yield from texts(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from texts(item)


def relations(value) -> set[str]:
    keys = set()
    if isinstance(value, Mapping):
        for name, item in value.items():
            parsed = embedded(name, item, value)
            if parsed is not None:
                keys.update(relations(parsed))
                continue
            if name in RELATIONS and type(item) is str and item:
                kind = RELATIONS[name]
                keys.add(kind + ":" + item)
                if name == "metamorphic_parent_id":
                    keys.add("case_id:" + item)
            elif (
                name in {"supplier_id", "purchase_order_id", "transaction_id"}
                and type(item) is str
            ):
                keys.add("entity_identity:" + item)
            elif name not in {"labels", "truth", "truths", "truth_extension"}:
                keys.update(relations(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            keys.update(relations(item))
    for body in texts(value):
        keys.add("content_sha256:" + hashlib.sha256(body.encode()).hexdigest())
        keys.add(
            "template:" + hashlib.sha256(normalized_template(body).encode()).hexdigest()
        )
    return keys


def _cases(value):
    if isinstance(value, Mapping):
        if "case_id" in value and any(
            name in value
            for name in ("model_input", "documents", "policy_text", "sources")
        ):
            yield value
        else:
            for name, item in value.items():
                if name not in {"labels", "truth", "truths"}:
                    parsed = embedded(name, item, value)
                    yield from _cases(parsed if parsed is not None else item)
    elif isinstance(value, list):
        for item in value:
            yield from _cases(item)


def register(
    seen_path: Path, archive_files: Mapping[Path, str], *, output: Path | None = None
) -> dict:
    """Source files must be explicitly selected and pinned, never guessed from a summary."""
    from .eval_f_context import load_cases
    from .gold_v3 import load_bundle
    from .eval_g_exclusion import GOLD_SET_RESOURCES, V3_VERSIONS

    registry = build_registry()
    detail = load_sealed_detail(seen_path, registry)
    original = read(seen_path)
    seen_ids = set(original["case_ids"])
    expected_hashes = {
        source["content_sha256"]
        for fingerprint in original["source_fingerprints"]
        for source in json.loads(fingerprint)
    }
    keys = set(registry.keys()) | set(detail.keys)
    bodies = set()
    sources = {str(seen_path): sha(seen_path)}
    authored = [
        case.model_dump(mode="json")
        for version in ("v3.7", "v3.8")
        for case in authored_cases(version)
    ]
    current = [case.model_dump(mode="json") for case in load_bundle()[0].cases]
    for version in V3_VERSIONS[1:]:
        current.extend(
            case.model_dump(mode="json")
            for split in ("dev", "validation")
            for case in load_cases(split, "gold-set." + version)
        )
    for version in ("v1", "v2"):
        path = GOLD_SET_RESOURCES / ("cases." + version + ".json")
        sources[str(path)] = sha(path)
        current.extend(read(path)["cases"])
    for case in (*current, *authored):
        keys.update(relations(case))
        bodies.update(texts(case))
    archive_cases, originals, probe_ids = {}, {}, set()
    source_cases = {}
    for path, expected in archive_files.items():
        if sha(path) != expected:
            raise ValueError("h_archive_source_checksum")
        from .eval_h_history import load_source

        source = load_source(path, original)
        originals[path] = source
        sources[str(path)] = expected
        rows = []
        for case in _cases(source):
            if case["case_id"] not in seen_ids:
                continue
            rows.append(case)
            archive_cases.setdefault(case["case_id"], []).append(case)
            keys.update(relations(case))
            bodies.update(texts(case))
        source_cases[str(path)] = rows
        # Some originals retain chunks separately from their Case metadata.
        for body in texts(source):
            digest = hashlib.sha256(body.encode()).hexdigest()
            if digest in expected_hashes:
                bodies.add(body)
                keys.add("content_sha256:" + digest)
    for path, source in originals.items():
        registration = (
            source.get("registration", {}) if isinstance(source, Mapping) else {}
        )
        if registration.get("kind") != "user-authorized-model-connectivity.v1":
            continue
        request = originals.get(path.with_name("request-v1.json"), {})
        from .trace import canonical_checksum

        if request.get("request_checksum") != canonical_checksum(
            request.get("payload")
        ):
            raise ValueError("h_connectivity_request_binding")
        for case_id in registration["cases"]:
            if case_id in seen_ids:
                probe_ids.add(case_id)
                archive_cases.setdefault(case_id, []).append(dict(case_id=case_id))
                keys.add("case_id:" + case_id)
        bodies.update(texts(request["payload"]))
    covered_hashes = {hashlib.sha256(body.encode()).hexdigest() for body in bodies}
    missing_hashes = sorted(expected_hashes - covered_hashes)
    missing_cases = sorted(seen_ids - set(archive_cases))
    case_kinds, family_values, scope_values = {}, {}, {}
    for rows in source_cases.values():
        for case in rows:
            case_id = case["case_id"]
            index = kind_index(case)
            case_kinds.setdefault(case_id, set()).update(index)
            family_values.setdefault(case_id, set()).update(
                index.get("generation_family", ())
            )
            scope_values.setdefault(case_id, set()).update(
                index.get("historical_scope", ())
            )
    source_kinds = {
        name: {kind for case in rows for kind in kind_index(case)}
        for name, rows in source_cases.items()
    }
    case_sources = {}
    for name, rows in source_cases.items():
        for case in rows:
            case_sources.setdefault(case["case_id"], set()).add(name)
    families = {key: next(iter(v)) for key, v in family_values.items() if len(v) == 1}
    scopes = {key: next(iter(v)) for key, v in scope_values.items() if len(v) == 1}
    # Count every value ever recorded, so a case with inconsistent rows still
    # blocks a sibling claim on the family or scope it names.
    family_counts = Counter(
        value for values in family_values.values() for value in values
    )
    scope_counts = Counter(
        value for values in scope_values.values() for value in values
    )
    relation_scopes = {
        name: dict(
            source=name,
            case_count=len({case["case_id"] for case in rows}),
            material_case_count=len(
                {case["case_id"] for case in rows if carries_material(case)}
            ),
            kinds_recorded=sorted(source_kinds[name]),
        )
        for name, rows in sorted(source_cases.items())
    }
    incomplete_relations, not_recorded, accepted = [], {}, {}
    for case_id, variants in sorted(archive_cases.items()):
        if case_id in probe_ids:
            not_recorded[case_id] = [
                "business_material_not_applicable:connectivity_probe"
            ]
            continue
        index = case_kinds.get(case_id, set())
        missing = set()
        if "template" not in index:
            missing.add("template")
        own_sources = case_sources.get(case_id, set())
        assigned_in_sources = set().union(*(source_kinds[name] for name in own_sources))
        # Without a sibling relation — no other consumed case shares this case's
        # generation family or historical scope — the relational kinds below had
        # nothing to describe, and a kind none of the case's own sources ever
        # assigned was never part of their registration, so both together make
        # the absence explicit rather than a gap.
        family, scope = families.get(case_id), scopes.get(case_id)
        no_sibling = (
            family is not None
            and scope is not None
            and family_counts[family] == 1
            and scope_counts[scope] == 1
        )
        for kind in ("generation_family", "entity_root", "near_duplicate_group"):
            if kind in index:
                continue
            if (
                kind != "generation_family"
                and kind not in assigned_in_sources
                and no_sibling
            ):
                accepted.setdefault(case_id, []).append(kind)
            else:
                missing.add(kind)
        parents = [
            case["metamorphic_parent_id"]
            for case in variants
            if "metamorphic_parent_id" in case
        ]
        valid_parent = bool(parents) and all(
            value is None or (type(value) is str and bool(value)) for value in parents
        )
        if not valid_parent:
            if parents or "metamorphic_parent" in assigned_in_sources or not no_sibling:
                missing.add("metamorphic_parent")
            else:
                accepted.setdefault(case_id, []).append("metamorphic_parent")
        if missing:
            incomplete_relations.append(case_id)
        absent = [
            name
            for name in (
                "entity_root_id",
                "near_duplicate_group_id",
                "metamorphic_parent_id",
            )
            if not any(name in case for case in variants)
        ]
        if absent:
            not_recorded[case_id] = absent
    body = dict(
        schema_version=VERSION,
        scopes=[scope.scope_id for scope in registry.scopes],
        source_sha256=sources,
        keys=sorted(keys),
        materials=sorted(bodies),
        archived_case_count=len(seen_ids),
        archived_cases_loaded=len(archive_cases),
        source_identities=len(expected_hashes),
        source_contents_loaded=len(expected_hashes & covered_hashes),
        missing_cases=missing_cases,
        missing_source_contents=missing_hashes,
        incomplete_relations=sorted(incomplete_relations),
        historical_not_recorded=not_recorded,
        accepted_not_recorded={
            case_id: sorted(kinds) for case_id, kinds in sorted(accepted.items())
        },
        relation_scopes=relation_scopes,
        connectivity_probe_ids=sorted(probe_ids),
        history_complete=not (missing_cases or missing_hashes or incomplete_relations),
    )
    body["registry_sha256"] = payload_sha256(body)
    if output is not None:
        write(output, body)
    return body


def verify(registry: dict, *, check_sources: bool = True) -> None:
    if registry.get("schema_version") not in READABLE_VERSIONS or registry.get(
        "registry_sha256"
    ) != payload_sha256(
        {key: value for key, value in registry.items() if key != "registry_sha256"}
    ):
        raise ValueError("h_exclusion_registry_identity")
    if check_sources:
        for entry, expected in registry["source_sha256"].items():
            if sha(Path(entry)) != expected:
                raise ValueError("h_exclusion_source_changed")


def _grams(body: str) -> frozenset[str]:
    normalized = normalized_template(body)
    return frozenset(
        normalized[index : index + 5] for index in range(max(0, len(normalized) - 4))
    )


def _similar(left: frozenset[str], right: frozenset[str]) -> bool:
    if not left or not right:
        return False
    if min(len(left), len(right)) / max(len(left), len(right)) < 0.82:
        return False
    shared = len(left & right)
    return shared / (len(left) + len(right) - shared) >= 0.82


def check(registry: dict, cases: Iterable[HCase], *, formal: bool) -> dict:
    verify(registry)
    cases = tuple(cases)
    if formal and registry["history_complete"] is not True:
        raise ValueError("h_history_material_incomplete")
    known_keys = set(registry["keys"])
    # Paragraph edges also detect a reused substantive body padded with unrelated text.
    known_text = {
        unit
        for body in registry["materials"]
        for unit in (body, *body.splitlines())
        if len(normalized_template(unit)) >= 32
    }
    known_grams = tuple(_grams(body) for body in known_text)
    conflicts = {}
    nodes = []
    for case in cases:
        hits = sorted(relations(case.model_dump(mode="json")) & known_keys)
        for document in case.documents:
            units = [document.content, *document.content.splitlines()]
            for unit in units:
                grams = _grams(unit)
                if len(normalized_template(unit)) >= 32 and any(
                    _similar(grams, other) for other in known_grams
                ):
                    hits.append("near_duplicate_5gram:" + document.source_id)
            nodes.append((case, document, _grams(document.content)))
        if hits:
            conflicts[case.case_id] = sorted(set(hits))
    parents = list(range(len(nodes)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    edges = []
    for index, (case, document, grams) in enumerate(nodes):
        for previous in range(index):
            other, peer, peer_grams = nodes[previous]
            if document.role == peer.role and _similar(grams, peer_grams):
                parents[root(index)] = root(previous)
                edges.append((case.case_id, other.case_id))
    groups = {}
    for index, (case, _document, _grams_value) in enumerate(nodes):
        groups.setdefault(root(index), set()).add(case.near_duplicate_group_id)
    if any(len(identities) != 1 for identities in groups.values()):
        raise ValueError("h_near_duplicate_component_group_mismatch")
    by_case = {case.case_id: case for case in cases}
    for case in cases:
        for other in cases:
            if (
                case.generation_family_id == other.generation_family_id
                or case.entity_root_id == other.entity_root_id
                or case.metamorphic_parent_id == other.case_id
            ):
                if case.near_duplicate_group_id != other.near_duplicate_group_id:
                    raise ValueError("h_family_entity_metamorphic_group_mismatch")
        if case.metamorphic_parent_id and case.metamorphic_parent_id not in by_case:
            raise ValueError("h_metamorphic_parent_outside_batch")
    result = dict(
        registry_sha256=registry["registry_sha256"],
        unseen=not conflicts,
        history_complete=registry["history_complete"],
        conflicts=conflicts,
        near_duplicate_edges=edges,
        disposable=not formal,
    )
    if formal and conflicts:
        raise ValueError("h_consumed_material_overlap")
    return result
