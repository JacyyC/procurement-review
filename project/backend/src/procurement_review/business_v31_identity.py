"""Canonical identity references of the 3.1.0 contract answer.

An identity field is a reference token inside a sentence: the authored material states
``订单ORDER: PO-907；合同CONTRACT: CONTRACT-907。`` and the closing punctuation belongs to
the sentence, not to the reference.  A model that copies the sentence terminator (or a
quotation mark, or trailing whitespace) into the field has copied the *same* reference in
a different format, and the byte comparison the deterministic rule performs would read it
as a different contract, a different party and a different order at once — a high-risk
violation that the material does not support.

:func:`canonical_reference` removes only what cannot be part of an identifier: surrounding
whitespace, one matched enclosing pair, and terminal sentence punctuation.  It never
removes an alphanumeric character or an interior separator, and it never returns an empty
string, so ``CONTRACT-907`` and ``CONTRACT-907-C`` remain different references and a
genuinely mismatched reference is still a violation instead of being normalized into a
match.  The canonical value is what the source boundary validates and what the aggregation
compares, so the validated value and the used value stay the same value.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The duty whose fields are compared as identities rather than read as prose.
IDENTITY_WORKER = "contract"
IDENTITY_DUTY = "O-05"
IDENTITY_FIELDS = frozenset({"buyer", "seller", "order", "contract"})

#: Enclosing pairs a copy may carry around a reference.
_ENCLOSING = frozenset(
    {
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
        ("「", "」"),
        ("『", "』"),
        ("《", "》"),
        ("〈", "〉"),
        ("（", "）"),
        ("(", ")"),
        ("【", "】"),
        ("[", "]"),
    }
)

#: Sentence punctuation that terminates the clause a reference sits in.
_TERMINAL = frozenset("。．.,;，；、:：!！?？")

_WHITESPACE = " \t\r\n\u3000\x0b\x0c"


def canonical_reference(value: str) -> str:
    """The reference one copied field really names, with sentence formatting removed."""

    text = value
    while text:
        stripped = text.strip(_WHITESPACE)
        if stripped != text:
            text = stripped
            continue
        if len(text) >= 2 and (text[0], text[-1]) in _ENCLOSING:
            text = text[1:-1]
            continue
        if text[-1] in _TERMINAL:
            text = text[:-1]
            continue
        break
    # A field made only of punctuation is not a reference: the declared value stands as
    # it was rather than being emptied into something that matches nothing.
    return text or value


def canonical_identity_field(value: Any, *, worker: str, duty: str, field: str) -> Any:
    """One declared field of one duty, canonicalized only when it is an identity."""

    if (
        worker != IDENTITY_WORKER
        or duty != IDENTITY_DUTY
        or field not in IDENTITY_FIELDS
    ):
        return value
    if not isinstance(value, str):
        return value
    return canonical_reference(value)


def canonicalize_declared_value(
    value: Mapping[str, Any], *, worker: str, duty: str
) -> dict[str, Any]:
    """The declared value object of one duty with its identity fields canonicalized."""

    return {
        key: canonical_identity_field(item, worker=worker, duty=duty, field=key)
        for key, item in value.items()
    }


__all__ = [
    "IDENTITY_DUTY",
    "IDENTITY_FIELDS",
    "IDENTITY_WORKER",
    "canonical_identity_field",
    "canonical_reference",
    "canonicalize_declared_value",
]
