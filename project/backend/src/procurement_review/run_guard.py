"""Shared privacy guard for versioned run manifests written outside the repository.

Manifests must declare what they did *not* retain, which means their own field
names mention secrets. Screening declared field names by pattern therefore always
misfires, so each manifest schema is closed (``extra="forbid"``) and declares
retention as ``Literal[False]``. This module only screens free-form mapping keys,
server-supplied labels, and every string value.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from .contracts import contains_secret_value

REDACTED_MARKER = "redacted_value"

_FREE_FORM_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "chain_of_thought",
    "cookie",
    "credential",
    "gold_label",
    "password",
    "prompt",
    "reasoning",
    "response_body",
    "secret",
    "set_cookie",
    "thought",
    "token",
)
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")


class ManifestGuardError(RuntimeError):
    """A bounded reason code for a manifest that cannot be made safe."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@runtime_checkable
class GuardableManifest(Protocol):
    """What the guard needs from any manifest model."""

    @property
    def calls(self) -> Sequence[Any]: ...

    def model_dump(self, **kwargs: Any) -> dict[str, Any]: ...


def is_safe_model_id(value: str) -> bool:
    return bool(_MODEL_ID.fullmatch(value))


def is_sensitive_free_form_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in _FREE_FORM_KEY_PARTS)


def secret_value_paths(
    value: Any,
    *,
    credential: str,
    path: tuple[str | int, ...] = (),
) -> list[tuple[str | int, ...]]:
    offenders: list[tuple[str | int, ...]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            offenders += secret_value_paths(
                child, credential=credential, path=(*path, str(key))
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            offenders += secret_value_paths(
                child, credential=credential, path=(*path, index)
            )
    elif isinstance(value, str):
        carries_credential = len(credential) >= 8 and credential in value
        if carries_credential or contains_secret_value(value):
            offenders.append(path)
    return offenders


def sensitive_name_paths(
    payload: Mapping[str, Any],
    *,
    calls_field: str,
    screened_fields: Sequence[str],
) -> list[tuple[str | int, ...]]:
    """Flag server-supplied labels that would re-enter the manifest as mapping keys."""

    paths: list[tuple[str | int, ...]] = []
    for index, call in enumerate(payload.get(calls_field) or ()):
        if not isinstance(call, Mapping):
            continue
        for field in screened_fields:
            value = call.get(field)
            if isinstance(value, str) and is_sensitive_free_form_key(value):
                paths.append((calls_field, index, field))
    return paths


def guard_payload(
    payload: Mapping[str, Any],
    *,
    credential: str,
    open_mapping_fields: Sequence[str],
    calls_field: str,
    screened_fields: Sequence[str],
) -> tuple[str, ...]:
    """Return bounded guard reasons; empty means the payload is safe to write."""

    reasons: list[str] = []
    for field in open_mapping_fields:
        reasons += [
            f"free_form_key:{field}:{key}"
            for key in sorted(payload[field])
            if is_sensitive_free_form_key(str(key))
        ]
    reasons += [
        "secret_value:" + ".".join(str(part) for part in path)
        for path in secret_value_paths(payload, credential=credential)
    ]
    reasons += [
        "sensitive_name:" + ".".join(str(part) for part in path)
        for path in sensitive_name_paths(
            payload, calls_field=calls_field, screened_fields=screened_fields
        )
    ]
    return tuple(reasons)


def _scrub_payload(
    payload: dict[str, Any],
    paths: frozenset[tuple[str | int, ...]],
    dropped_keys: Mapping[str, set[str]],
) -> dict[str, Any]:
    def walk(value: Any, path: tuple[str | int, ...]) -> Any:
        if isinstance(value, Mapping):
            removed = dropped_keys.get(".".join(str(part) for part in path), set())
            return {
                key: walk(child, (*path, str(key)))
                for key, child in value.items()
                if str(key) not in removed
            }
        if isinstance(value, list):
            return [walk(child, (*path, index)) for index, child in enumerate(value)]
        if isinstance(value, str) and path in paths:
            return REDACTED_MARKER
        return value

    scrubbed = walk(payload, ())
    return scrubbed if isinstance(scrubbed, dict) else {}


def finalize_payload(
    manifest: BaseModel,
    *,
    credential: str,
    open_mapping_fields: Sequence[str],
    calls_field: str,
    screened_fields: Sequence[str],
    entry_model: type[BaseModel],
    derive: Callable[[Sequence[Any]], dict[str, Any]],
    failure_reason: str,
    close_code: str,
    outcome_field: str = "outcome",
) -> dict[str, Any]:
    """Screen a manifest, scrub it, and return the fields of the closed variant.

    A guard hit never discards the run: aggregates are recomputed from the scrubbed
    branch entries and the outcome records why the run closed.
    """

    payload = manifest.model_dump(mode="json")
    dropped = {
        field: {
            str(key) for key in payload[field] if is_sensitive_free_form_key(str(key))
        }
        for field in open_mapping_fields
    }
    paths = frozenset(
        [
            *secret_value_paths(payload, credential=credential),
            *sensitive_name_paths(
                payload, calls_field=calls_field, screened_fields=screened_fields
            ),
        ]
    )
    cleaned = _scrub_payload(payload, paths, dropped)
    entries = tuple(entry_model.model_validate(item) for item in cleaned[calls_field])
    cleaned.update(
        {
            **derive(entries),
            calls_field: tuple(entry.model_dump(mode="json") for entry in entries),
            outcome_field: failure_reason,
            "failure_code": close_code,
        }
    )
    return cleaned
