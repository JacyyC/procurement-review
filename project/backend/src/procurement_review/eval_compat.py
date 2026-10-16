"""Read-only compatibility and integrity checks for V1-07 evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gold_set import (
    DEFAULT_GOLD_SET_DIR,
    GOLD_CORPUS_CONTRACTS,
    GoldCorpusVersion,
    load_gold_cases,
    load_gold_labels,
    load_gold_manifest,
    validate_gold_set,
)
from .gold_v3 import load_bundle
from .rules import RuleSet, rule_set_for_version


class CompatibilityError(ValueError):
    """An artifact cannot be read under a registered historical contract."""


SUPPORTED_SCHEMAS: dict[str, frozenset[str]] = {
    "gold_manifest": frozenset({"gold-manifest.v1", "gold-manifest.v2"}),
    "run_manifest": frozenset(
        {"run-manifest.v1", "eval-run-manifest.v1", "eval-run-manifest.v2"}
    ),
    "projection": frozenset({"eval-projection.v1", "eval-projection.v2"}),
    "report": frozenset(
        {
            "eval-report.v1",
            "eval-report.v2",
            "eval-report.v3",
            "eval-report.v4",
            "eval-rescore.v1",
        }
    ),
}

FROZEN_RESOURCE_SHA256 = {
    "gold_set/cases.v1.json": "b5ab8cc4b96179e6ab782fc403e0f41eca16f4055bdc5b26b4a714961e1dd08e",
    "gold_set/labels.v1.json": "9804b3dba5d1e11ba723f9924114616851438dc9ba86bb9b75530429e42ea322",
    "gold_set/manifest.v1.json": "4f895aff77595e590baa70b8a3c52c9311135428b695404c437fc57895344d68",
    "gold_set/cases.v2.json": "1667f95c1e1df722cb35ad6608ec53c5f21f4dfc1977e37d951cb841ee270b18",
    "gold_set/labels.v2.json": "e2b6fe8dba19a23aaefc81724148cd7edd194de9acb30a5b62fc47c0bb27fb96",
    "gold_set/manifest.v2.json": "b5dcda29f760f6d8462a1eb28727b0e6f2dad4366965dac6a37fabd9113ab294",
    "rules/v1.json": "b3aa7e31503d96bb900714c2e0c3bebb6484b613bbee4150f59b577d7905b205",
    "rules/v2.json": "39c321cfefad6d70e64f910c7bca16b90f6a9ace5be0e816fcd2e091fdb373f9",
}

_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[a-z0-9._-]{12,}"),
    re.compile(r"(?i)\bsk-[a-z0-9_-]{12,}"),
    re.compile(r"(?i)(api[_-]?key|authorization|password|secret)\s*[:=]"),
    re.compile(r"SYNTHETIC_SECRET_TO_EXTERNAL_SINK"),
)


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    path: Path
    schema_version: str
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class GoldCompatibilityBundle:
    version: str
    cases: Any
    truth: Any
    manifest: Any


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CompatibilityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_versioned_json(
    path: Path, *, kind: str
) -> tuple[dict[str, Any], ArtifactIdentity]:
    """Read and identify an artifact without rewriting or normalizing its bytes."""

    if kind not in SUPPORTED_SCHEMAS:
        raise CompatibilityError(f"unknown compatibility artifact kind: {kind}")
    try:
        body = path.read_bytes()
        document = json.loads(body, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompatibilityError(
            f"unreadable compatibility artifact: {path}"
        ) from error
    if not isinstance(document, dict):
        raise CompatibilityError("versioned artifact root must be an object")
    schema = document.get("schema_version")
    if schema not in SUPPORTED_SCHEMAS[kind]:
        raise CompatibilityError(
            f"unsupported {kind} schema {schema!r}; explicit migration is required"
        )
    return document, ArtifactIdentity(
        path=path,
        schema_version=schema,
        sha256=hashlib.sha256(body).hexdigest(),
        byte_count=len(body),
    )


def load_gold_compatibility_bundle(
    version: str,
    *,
    directory: Path = DEFAULT_GOLD_SET_DIR,
) -> GoldCompatibilityBundle:
    """Load v1/v2 labels or v3 truth through their own immutable contracts."""
    if version == "v3.1":
        from .gold_v31 import load_bundle as load_revision

        cases, truth, manifest = load_revision(directory / "v3.1")
        return GoldCompatibilityBundle(version, cases, truth, manifest)
    if version == "v3":
        cases, truth, manifest = load_bundle(directory)
        return GoldCompatibilityBundle(version, cases, truth, manifest)
    try:
        corpus_version = GoldCorpusVersion(version)
        contract = GOLD_CORPUS_CONTRACTS[corpus_version]
    except (KeyError, ValueError) as error:
        raise CompatibilityError(
            f"unknown Gold compatibility version: {version}"
        ) from error
    cases = load_gold_cases(
        directory / contract.cases_file_name,
        corpus_version=corpus_version,
    )
    labels = load_gold_labels(
        directory / contract.labels_file_name,
        allow_holdout=True,
        corpus_version=corpus_version,
    )
    manifest = load_gold_manifest(
        directory / contract.manifest_file_name,
        corpus_version=corpus_version,
    )
    validate_gold_set(cases, labels, manifest)
    return GoldCompatibilityBundle(version, cases, labels, manifest)


def load_rules_compatibility_version(version: str) -> RuleSet:
    aliases = {"v1": "1.0.0", "v2": "2.0.0", "1": "1.0.0", "2": "2.0.0"}
    resolved = aliases.get(version, version)
    try:
        return rule_set_for_version(resolved)
    except ValueError as error:
        raise CompatibilityError(str(error)) from error


def verify_frozen_resource_hashes(resources_root: Path) -> tuple[ArtifactIdentity, ...]:
    """Verify the eight pre-V1-07 resources byte-for-byte and perform no writes."""

    root = resources_root.resolve()
    identities: list[ArtifactIdentity] = []
    for relative, expected in FROZEN_RESOURCE_SHA256.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
            body = path.read_bytes()
        except (OSError, ValueError) as error:
            raise CompatibilityError(f"missing frozen resource: {relative}") from error
        observed = hashlib.sha256(body).hexdigest()
        if observed != expected:
            raise CompatibilityError(f"frozen resource hash drift: {relative}")
        document = json.loads(body)
        identities.append(
            ArtifactIdentity(
                path=path,
                schema_version=str(document.get("schema_version", "unversioned")),
                sha256=observed,
                byte_count=len(body),
            )
        )
    return tuple(identities)


def sensitive_value_findings(value: Any, path: str = "$") -> tuple[str, ...]:
    """Return bounded JSON paths whose values look sensitive; keys alone do not count."""

    findings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            findings.extend(sensitive_value_findings(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            findings.extend(sensitive_value_findings(item, f"{path}[{index}]"))
    elif isinstance(value, str) and any(
        pattern.search(value) for pattern in _SENSITIVE_VALUE_PATTERNS
    ):
        findings.append(path)
    return tuple(findings)


__all__ = [
    "ArtifactIdentity",
    "CompatibilityError",
    "FROZEN_RESOURCE_SHA256",
    "GoldCompatibilityBundle",
    "SUPPORTED_SCHEMAS",
    "load_gold_compatibility_bundle",
    "load_rules_compatibility_version",
    "read_versioned_json",
    "sensitive_value_findings",
    "verify_frozen_resource_hashes",
]
