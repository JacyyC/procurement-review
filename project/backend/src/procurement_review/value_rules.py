"""The value-rule algebra of a registered open mapping (FIX-03A).

An open mapping is open in its *key space*, never in its value types.  Every registered key
declares what may live there: a bounded count, a bounded number, a registered vocabulary
word, a specific identifier or digest slot, a bounded text, a calendar day, or a bounded
array of one of those.  This module owns that vocabulary and the recursion that judges one
value against it; the registries that *use* it live beside the records they belong to.

A value is refused for the shape it actually has — an object or an array where a scalar is
declared, a string where a count is declared, a boolean where a number is declared, a word
outside its vocabulary, an empty or oversized text — so no annotation and no leaf name can
grant an exemption by accident.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from .field_guard import ScanFinding, ScanReport, scan_typed_value

#: Upper bound of every recorded count.
COUNT_LIMIT = 1_000_000
#: Upper bound of a recorded monetary amount.
NUMBER_LIMIT = 10**12
#: Upper bound of a bounded text/identifier leaf and of a registered id array.
TEXT_LIMIT = 200
LIST_LIMIT = 64

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
#: One bounded diagnostic token: a fixed name, an ``=`` and a value in the same charset as
#: an identifier.  A refusal's locations are name=value pairs from a closed vocabulary, so
#: they are neither free text nor an identifier and must not be able to carry prose.
_TOKEN_PATTERN = re.compile(r"^[a-z_]+=[A-Za-z0-9][A-Za-z0-9._:/-]*$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True, slots=True)
class ValueRule:
    """What one registered mapping key is allowed to carry."""

    kind: Literal[
        "count",
        "number",
        "bool",
        "enum",
        "text",
        "free_text",
        "identifier",
        "checksum",
        "date",
        "count_list",
        "text_list",
        "identifier_list",
        "token_list",
    ]
    vocabulary: frozenset[str] | None = None
    maximum: int | None = None
    max_length: int | None = None
    max_items: int | None = None
    nullable: bool = False

    @property
    def scans_as(self) -> str:
        """Which leaf rule the sensitive-value scanner applies to this key."""

        if self.kind == "identifier":
            return "identifier"
        if self.kind == "checksum":
            return "checksum"
        return "text"

    @property
    def item_kind(self) -> str:
        return {
            "count_list": "count",
            "text_list": "text",
            "free_text_list": "free_text",
            "identifier_list": "identifier",
            "token_list": "token",
        }[self.kind]


def count(*, nullable: bool = False) -> ValueRule:
    return ValueRule("count", maximum=COUNT_LIMIT, nullable=nullable)


def number(*, maximum: int = NUMBER_LIMIT, nullable: bool = False) -> ValueRule:
    """A bounded finite number: a monetary amount, never a count and never a string."""

    return ValueRule("number", maximum=maximum, nullable=nullable)


def flag(*, nullable: bool = False) -> ValueRule:
    return ValueRule("bool", nullable=nullable)


def word(vocabulary: frozenset[str], *, nullable: bool = False) -> ValueRule:
    return ValueRule("enum", vocabulary=vocabulary, nullable=nullable)


def text(*, max_length: int = TEXT_LIMIT, nullable: bool = False) -> ValueRule:
    """Bounded text in the registered identifier charset (codes, ids, references)."""

    return ValueRule("text", max_length=max_length, nullable=nullable)


def free_text(*, max_length: int = TEXT_LIMIT, nullable: bool = False) -> ValueRule:
    """Bounded human-readable text: no charset rule, still length-bound and screened."""

    return ValueRule("free_text", max_length=max_length, nullable=nullable)


def identity(*, nullable: bool = False) -> ValueRule:
    return ValueRule("identifier", max_length=160, nullable=nullable)


def digest(*, nullable: bool = False) -> ValueRule:
    return ValueRule("checksum", max_length=71, nullable=nullable)


def calendar_day(*, nullable: bool = False) -> ValueRule:
    return ValueRule("date", max_length=10, nullable=nullable)


def text_list(*, max_items: int = LIST_LIMIT) -> ValueRule:
    return ValueRule("text_list", max_items=max_items, max_length=TEXT_LIMIT)


def free_text_list(*, max_items: int = LIST_LIMIT) -> ValueRule:
    return ValueRule("free_text_list", max_items=max_items, max_length=TEXT_LIMIT)


def identity_list(*, max_items: int = LIST_LIMIT) -> ValueRule:
    return ValueRule("identifier_list", max_items=max_items, max_length=160)


def token_list(*, max_items: int = LIST_LIMIT) -> ValueRule:
    """A bounded list of ``name=value`` diagnostic tokens, never prose."""

    return ValueRule("token_list", max_items=max_items, max_length=TEXT_LIMIT)


#: Every key a Trace step summary may carry, with the value type that key owns.  Both the
#: input and the output summary use this one table: the same name means the same thing
#: wherever it appears, and a name that is not here is not a step fact.


def _finding(
    report: ScanReport, category: str, location: str, path: str, detail: str
) -> None:
    report.findings.append(
        ScanFinding(category=category, location=f"{location}:{path}", detail=detail)
    )


@dataclass(frozen=True, slots=True)
class MappingContract:
    """One registered open mapping: its name and the rule each key owns."""

    name: str
    fields: Mapping[str, ValueRule]

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self.fields)


def check_rule(
    value: Any,
    rule: ValueRule,
    *,
    path: str,
    location: str,
    report: ScanReport,
) -> None:
    """Judge one value against the type its registered key declares."""

    if value is None:
        if not rule.nullable:
            _finding(
                report, "unregistered_value", location, path, "value cannot be null"
            )
        return
    if rule.kind in {
        "count_list",
        "text_list",
        "free_text_list",
        "identifier_list",
        "token_list",
    }:
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(
            value, (list, tuple)
        ):
            _finding(
                report,
                "unregistered_shape",
                location,
                path,
                "registered key carries a scalar where an array is declared",
            )
            return
        if rule.max_items is not None and len(value) > rule.max_items:
            _finding(
                report, "unregistered_value", location, path, "array exceeds its bound"
            )
            return
        item_rule = ValueRule(
            rule.item_kind,
            vocabulary=rule.vocabulary,
            maximum=rule.maximum,
            max_length=rule.max_length,
        )
        for index, item in enumerate(value):
            check_rule(
                item,
                item_rule,
                path=f"{path}[{index}]",
                location=location,
                report=report,
            )
        return
    if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
        _finding(
            report,
            "unregistered_shape",
            location,
            path,
            "registered key carries a container where a scalar is declared",
        )
        return
    if rule.kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _finding(
                report, "unregistered_value", location, path, "value is not a number"
            )
            return
        if (
            not math.isfinite(float(value))
            or value < 0
            or (rule.maximum is not None and value > rule.maximum)
        ):
            _finding(
                report, "unregistered_value", location, path, "number is out of range"
            )
        return
    if rule.kind == "count":
        if isinstance(value, bool) or not isinstance(value, int):
            _finding(
                report, "unregistered_value", location, path, "count is not an integer"
            )
            return
        if value < 0 or (rule.maximum is not None and value > rule.maximum):
            _finding(
                report, "unregistered_value", location, path, "count is out of range"
            )
        return
    if rule.kind == "bool":
        if not isinstance(value, bool):
            _finding(
                report, "unregistered_value", location, path, "flag is not a boolean"
            )
        return
    if not isinstance(value, str):
        _finding(report, "unregistered_value", location, path, "value is not a string")
        return
    limit = rule.max_length if rule.max_length is not None else TEXT_LIMIT
    if not value or len(value) > limit:
        _finding(
            report, "unregistered_value", location, path, "text is empty or too long"
        )
        return
    if rule.kind == "enum" and (
        rule.vocabulary is None or value not in rule.vocabulary
    ):
        _finding(report, "unregistered_value", location, path, "word is not registered")
        return
    if rule.kind == "token" and not _TOKEN_PATTERN.match(value):
        _finding(
            report, "unregistered_value", location, path, "token is outside its form"
        )
        return
    if rule.kind in {"identifier", "text"} and not _IDENTIFIER_PATTERN.match(value):
        _finding(
            report,
            "unregistered_value",
            location,
            path,
            "text is outside the key's form",
        )
        return
    if rule.kind == "checksum" and not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        _finding(report, "unregistered_value", location, path, "value is not a digest")
        return
    if rule.kind == "date":
        if not _DATE_PATTERN.match(value):
            _finding(
                report, "unregistered_value", location, path, "value is not a date"
            )
            return
        try:
            date.fromisoformat(value)
        except ValueError:
            _finding(
                report, "unregistered_value", location, path, "value is not a date"
            )
            return
    report.scanned_values += 1
    outcome = scan_typed_value(value, kind=rule.scans_as, location=location, path=path)
    report.scanned_values += outcome.scanned_values
    report.findings.extend(outcome.findings)


def check_mapping(
    mapping: Mapping[Any, Any],
    contract: MappingContract,
    *,
    path: str,
    location: str,
    report: ScanReport,
    forbidden_key: Any,
) -> None:
    """Judge one registered open mapping key by key, then value by value.

    ``forbidden_key`` is the shared key-name predicate of the field guard, passed in so
    this module and the record guard cannot drift apart.
    """

    for key, value in mapping.items():
        name = str(key)
        report.scanned_values += 1
        if name not in contract.fields:
            _finding(
                report,
                "unregistered_key",
                location,
                f"{path}.{name}",
                f"key is outside the {contract.name} key space",
            )
            continue
        if forbidden_key(name):
            _finding(
                report,
                "forbidden_key",
                location,
                f"{path}.{name}",
                "key carries a forbidden part",
            )
            continue
        check_rule(
            value,
            contract.fields[name],
            path=f"{path}.{name}",
            location=location,
            report=report,
        )


__all__ = [
    "COUNT_LIMIT",
    "LIST_LIMIT",
    "MappingContract",
    "NUMBER_LIMIT",
    "TEXT_LIMIT",
    "ValueRule",
    "calendar_day",
    "check_rule",
    "count",
    "digest",
    "flag",
    "free_text",
    "free_text_list",
    "identity",
    "identity_list",
    "number",
    "text",
    "text_list",
    "word",
]
