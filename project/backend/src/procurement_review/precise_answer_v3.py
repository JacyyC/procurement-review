"""The ``finding-evidence.v3`` answer contract of the rules 3.1.0 document branches.

The frozen ``finding-evidence.v2`` wire asks the model to (a) restate four mechanically
derived fields so that every state combination is self-consistent and (b) double-encode
the value it extracted into a JSON string capped at 256 characters.  A real endpoint was
measured against it: both authored policy cases were refused twice, and the bounded
location diagnostics of the refusals name one cause each time — the O-04 duty rule of the
policy branch cannot fit its own required content (threshold, categories in scope, the
exclusion with its basis and the registered exception with its requirements) into 256
characters, and a list-valued witness arrives in the shape the model naturally writes.

This version keeps the *evidence* semantics of v2 — the same duties, the same closed value
registry, the same grounding, the same persistence — and moves only the mechanics to the
server:

* the value of a duty is declared as the JSON object it is (a JSON string of that object
  is accepted too); the server canonicalizes it into the compact text the value registry
  and the witness comparison already use, under a bound that the richest duty can satisfy;
* ``assertion`` is derived from ``status``; ``risk_level`` is required only where it carries
  information (a violated duty's severity); the unknown state's literal value and the empty
  missing list of a decided state are written by the server.

Nothing is inferred from prose and nothing invalid is repaired: every derived field is a
function of the model's own declaration, the value still has to satisfy the closed per-duty
model, a citation still has to be one of the branch's authorized references, and an answer
that violates the shape is refused with the same bounded, value-free location tokens.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError, model_validator

from .business_v31_identity import canonical_identity_field, canonicalize_declared_value
from .business_v3_support import WitnessV3
from .eval_v2_contracts import EvalV2Model, SemanticClaimProjection
from .gold_v3_contracts import ObligationId
from .model_wire_v2 import (
    Ref,
    Text,
    ValidatedFindingV2,
    WireIssue,
    strict_json,
)
from .business_v31_review import SCHEMA_FIELDS

PRECISE_ANSWER_VERSION = "finding-evidence.v3"

#: The canonical value text one duty may carry.  It is a backstop, not the contract: every
#: field of every duty model is already bounded, and the richest registered duty (the
#: policy quote rule with both of its 64-character bases) stays well inside this.
PRECISE_VALUE_LIMIT = 512
#: The canonical text of one witness value (a serialized list or object).
PRECISE_WITNESS_LIMIT = 256

#: The four states written by the model and the three fields the server derives from them.
_ASSERTIONS = {
    "supported": "compliant",
    "violated": "non_compliant",
    "unknown": "unknown",
    "not_applicable": "not_applicable",
}
_RISKS = {
    "supported": "none",
    "not_applicable": "none",
    "unknown": "unknown",
}
_SEVERITIES = frozenset({"low", "medium", "high", "critical"})


class PreciseAnswerFindingV3(EvalV2Model):
    """One duty as the model declares it, before the mechanical fields are derived.

    Only the state, the value, the citations and the missing list carry information; the
    severity is asked for exactly where it does (a violated duty), and everything else the
    stored finding needs is a function of the state.
    """

    obligation_id: ObligationId
    status: Literal["supported", "violated", "unknown", "not_applicable"]
    statement: Text
    recommended_action: Text
    normalized_value: dict[str, Any] | str | None = None
    risk_level: Literal["low", "medium", "high", "critical"] | None = None
    evidence_refs: tuple[Ref, ...] = Field(default=(), max_length=4)
    missing_information: tuple[Text, ...] = Field(default=(), max_length=4)


class PreciseAnswerWitnessV3(EvalV2Model):
    """One field witness; a list or an object is the value, not an encoded string."""

    obligation_id: ObligationId
    field: Annotated[str, Field(pattern=r"^[a-z_]+$", max_length=48)]
    value: str | int | bool | list[Any] | dict[str, Any]
    source_ref: Ref
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]


class PreciseAnswerEnvelopeV3(EvalV2Model):
    """The whole answer of one document branch under rules 3.1.0."""

    schema_version: Literal["finding-evidence.v3"]
    findings: tuple[PreciseAnswerFindingV3, ...] = Field(min_length=1, max_length=12)
    witnesses: tuple[PreciseAnswerWitnessV3, ...] = Field(default=(), max_length=192)


class PreciseFindingV3(EvalV2Model):
    """The canonical finding after the mechanical fields have been derived.

    Its meaning is exactly the v2 finding's: the same states, the same closed value text,
    the same citation and missing rules.  The difference is where the fields came from.
    """

    obligation_id: ObligationId
    status: Literal["supported", "violated", "unknown", "not_applicable"]
    assertion: Literal["compliant", "non_compliant", "unknown", "not_applicable"]
    risk_level: Literal["none", "low", "medium", "high", "critical", "unknown"]
    normalized_value: Annotated[
        str, Field(min_length=1, max_length=PRECISE_VALUE_LIMIT)
    ]
    statement: Text
    recommended_action: Text
    evidence_refs: tuple[Ref, ...] = Field(default=(), max_length=4)
    missing_information: tuple[Text, ...] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def state_is_self_consistent(self) -> PreciseFindingV3:
        if self.assertion != _ASSERTIONS[self.status]:
            raise ValueError("derived assertion disagrees with its state")
        if self.status == "unknown":
            if self.normalized_value != "UNKNOWN" or self.evidence_refs:
                raise ValueError(
                    "an unknown finding carries the literal and no citation"
                )
            if not self.missing_information:
                raise ValueError("an unknown finding states what is missing")
        else:
            if self.normalized_value == "UNKNOWN" or not self.evidence_refs:
                raise ValueError("a decided finding carries a value and its citation")
            if self.missing_information:
                raise ValueError("a decided finding carries no missing list")
        if (self.risk_level in _SEVERITIES) != (self.status == "violated"):
            raise ValueError("only a violated finding carries a severity")
        return self


class PreciseEnvelopeV3(EvalV2Model):
    schema_version: Literal["finding-evidence.v3"]
    findings: tuple[PreciseFindingV3, ...] = Field(min_length=1, max_length=12)


class PreciseClaimProjection(SemanticClaimProjection):
    """The precise line's claim projection: the same shape, the value bound of its wire."""

    normalized_value: Annotated[
        str, Field(min_length=1, max_length=PRECISE_VALUE_LIMIT)
    ]


class PreciseValidatedFindingV3(ValidatedFindingV2):
    """One validated duty of the precise line: the same shape, its own finding type."""

    finding: PreciseFindingV3

    def projection(self) -> PreciseClaimProjection:
        import hashlib

        from .gold_v3_contracts import ObligationState

        return PreciseClaimProjection(
            obligation_id=self.finding.obligation_id,
            state=ObligationState(self.finding.status),
            proposition=self.proposition,
            normalized_value=self.finding.normalized_value,
            evidence=self.evidence,
            reason_codes=(),
            statement_sha256=hashlib.sha256(
                self.finding.statement.encode()
            ).hexdigest(),
            recommended_action_sha256=hashlib.sha256(
                self.finding.recommended_action.encode()
            ).hexdigest(),
        )


# ------------------------------------------------------------------ refusals


def _refusal(
    body: Any, request: Any, error: Exception
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The declared fields and bounded location tokens one v3 refusal may name."""

    from .business_v31_review import _SchemaDiagnostics

    diagnostics = _SchemaDiagnostics(error, body, request)
    return diagnostics.summary()


def _fail(body: Any, request: Any, error: Exception) -> WireIssue:
    fields, tokens = _refusal(body, request, error)
    return WireIssue("invalid_schema", correctable=True, fields=fields, details=tokens)


# ------------------------------------------------------------- canonicalization


def _canonical_value(finding: PreciseAnswerFindingV3, request: Any) -> str:
    """The compact value text one declared duty stands for, or a refusal."""

    if finding.status == "unknown":
        if finding.normalized_value not in (None, "UNKNOWN"):
            raise WireIssue(
                "invalid_schema",
                correctable=True,
                fields=("normalized_value",),
                details=(
                    "slot=findings",
                    f"duty={finding.obligation_id}",
                    "path=normalized_value",
                    "category=value_rule",
                    "expected=absent",
                    "actual=" + _basic_type(finding.normalized_value),
                ),
            )
        return "UNKNOWN"
    declared = finding.normalized_value
    if isinstance(declared, dict):
        declared = canonicalize_declared_value(
            declared, worker=request.worker, duty=finding.obligation_id
        )
        text = json.dumps(declared, ensure_ascii=False, separators=(",", ":"))
    elif isinstance(declared, str):
        text = declared
    else:
        raise WireIssue(
            "invalid_schema",
            correctable=True,
            fields=("normalized_value",),
            details=(
                "slot=findings",
                f"duty={finding.obligation_id}",
                "path=normalized_value",
                "category=type_mismatch",
                "expected=object",
                "actual=" + _basic_type(declared),
            ),
        )
    try:
        parsed = strict_json(text)
    except ValueError:
        raise WireIssue(
            "invalid_schema",
            correctable=True,
            fields=("normalized_value",),
            details=(
                "slot=findings",
                f"duty={finding.obligation_id}",
                "path=normalized_value",
                "category=value_rule",
                "expected=json_object",
                "actual=string",
                f"size={min(len(text), 9999)}",
            ),
        ) from None
    if not isinstance(parsed, dict):
        raise WireIssue(
            "invalid_schema",
            correctable=True,
            fields=("normalized_value",),
            details=(
                "slot=findings",
                f"duty={finding.obligation_id}",
                "path=normalized_value",
                "category=value_rule",
                "expected=json_object",
                "actual=" + _basic_type(parsed),
            ),
        )
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _basic_type(value: Any) -> str:
    if value is None:
        return "null"
    return {
        str: "string",
        int: "number",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }.get(type(value), "unknown")


def _canonical_witness_value(witness: Any, request: Any) -> Any:
    """A witness value in the form the grounding comparison uses.

    A list or an object is serialized by the server instead of being asked for as an
    encoded string; scalars travel as they are, except that an identity reference is
    canonicalized exactly as the duty's declared value is: the grounding validates the
    canonical value and the aggregation compares that same value, so a copied sentence
    terminator cannot become a different contract.  An empty string stays invalid — the
    model omits a dimension it cannot state instead of claiming an empty one.
    """

    value = witness.value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return canonical_identity_field(
        value,
        worker=getattr(request, "worker", ""),
        duty=getattr(witness, "obligation_id", ""),
        field=getattr(witness, "field", ""),
    )


def canonical_answer(body: str, request: Any):
    """Validate one v3 answer and derive the mechanical fields it no longer restates.

    Returns the canonical finding documents and witnesses in the stored form.  Nothing is
    repaired: only fields that are a function of the model's own declarations are filled.
    """

    try:
        raw = strict_json(body)
    except (ValueError, UnicodeError):
        raise WireIssue("invalid_json", correctable=True) from None
    if not isinstance(raw, dict):
        raise WireIssue(
            "invalid_schema",
            correctable=True,
            fields=("findings",),
            details=(
                "slot=answer",
                "path=answer",
                "category=type_mismatch",
                "expected=object",
                "actual=" + _basic_type(raw),
            ),
        )
    try:
        answer = PreciseAnswerEnvelopeV3.model_validate_json(
            json.dumps(raw), strict=True
        )
    except ValidationError as error:
        raise _fail(raw, request, error) from None
    allowed = {check.proposition.check_id for check in request.checks}
    if {f.obligation_id for f in answer.findings} - allowed:
        raise WireIssue(
            "check_outside_branch",
            fields=("obligation_id",),
            details=(
                "slot=findings",
                "path=obligation_id",
                "category=unauthorized_check",
            ),
        )
    canonical = []
    for finding in answer.findings:
        canonical.append(
            {
                "obligation_id": finding.obligation_id,
                "status": finding.status,
                "assertion": _ASSERTIONS[finding.status],
                "risk_level": (
                    finding.risk_level
                    if finding.status == "violated"
                    else _RISKS[finding.status]
                ),
                "normalized_value": _canonical_value(finding, request),
                "statement": finding.statement,
                "recommended_action": finding.recommended_action,
                "evidence_refs": [ref for ref in finding.evidence_refs],
                "missing_information": (
                    list(finding.missing_information)
                    if finding.status == "unknown"
                    else []
                ),
            }
        )
    witnesses = [
        {
            "obligation_id": witness.obligation_id,
            "field": witness.field,
            "value": _canonical_witness_value(witness, request),
            "source_ref": witness.source_ref,
            "start": witness.start,
            "end": witness.end,
        }
        for witness in answer.witnesses
    ]
    return {
        "schema_version": "finding-evidence.v3",
        "findings": canonical,
    }, witnesses


def parse_precise_answer(body: str, request: Any, *, protection):
    """Decode one 3.1.0 v3 answer into the grounded document values of the line.

    The canonical answer is validated by the line's own closed envelope, so a derived
    field is checked exactly like a declared one.  An answer that declares a state
    combination the model was told not to declare is refused — by the server's contract,
    not by a model-side self-consistency requirement.
    """

    raw = strict_json(body)
    protection.assert_safe(raw)
    document, witnesses = canonical_answer(body, request)
    return decode_canonical_answer(document, request, witnesses)


def decode_canonical_answer(document: Any, request: Any, witnesses: Any):
    """Decode one already-canonical v3 answer; used by the live path and by a reuse."""

    from .business_v31_documents import VALUE_MODELS_V31
    from .business_v3_documents import decode_document_values

    try:
        envelope = PreciseEnvelopeV3.model_validate_json(
            json.dumps(document), strict=True
        )
    except ValidationError as error:
        raise _fail(document, request, error) from None
    findings = bind_findings(request, envelope)
    return (
        decode_document_values(request, findings, registry=VALUE_MODELS_V31),
        _canonical_witnesses(witnesses),
    )


def _canonical_witnesses(witnesses: Any):
    """The stored witnesses of one answer, refused by name when one cannot be stored."""

    canonical = []
    for index, item in enumerate(witnesses):
        try:
            canonical.append(WitnessV3.model_validate(item, strict=True))
        except ValidationError as error:
            fields = tuple(
                sorted(
                    {
                        str(part)
                        for entry in error.errors(
                            include_input=False,
                            include_context=False,
                            include_url=False,
                        )
                        for part in entry.get("loc", ())
                        if str(part) in SCHEMA_FIELDS
                    }
                )
            ) or ("value",)
            declared = item.get("value") if isinstance(item, dict) else None
            tokens = ["slot=witnesses", f"index={index}"]
            if isinstance(item, dict) and isinstance(item.get("field"), str):
                tokens.append(f"path={item['field']}")
            tokens.append(
                "category=size_out_of_range"
                if declared == ""
                else "category=type_mismatch"
            )
            tokens.append("expected=scalar")
            tokens.append(f"actual={_basic_type(declared)}")
            if isinstance(declared, str):
                tokens.append(f"size={min(len(declared), 9999)}")
            raise WireIssue(
                "invalid_schema", correctable=True, fields=fields, details=tuple(tokens)
            ) from None
    return tuple(canonical)


def bind_findings(request: Any, envelope: PreciseEnvelopeV3):
    """Bind every canonical finding to its slot and to the sources it cites."""

    slots = {c.proposition.check_id: c.proposition for c in request.checks}
    refs = {s.ref: s.identity for s in request.sources}
    ids = [f.obligation_id for f in envelope.findings]
    if set(ids) - set(slots):
        raise WireIssue("check_outside_branch")
    if set(ids) != set(slots) or len(ids) != len(set(ids)):
        raise WireIssue("required_check_missing_or_duplicate", correctable=True)
    validated = []
    for item in envelope.findings:
        if set(item.evidence_refs) - set(refs):
            raise WireIssue("citation_outside_branch")
        if len(item.evidence_refs) != len(set(item.evidence_refs)):
            raise WireIssue("duplicate_citation", correctable=True)
        validated.append(
            PreciseValidatedFindingV3(
                worker=request.worker,
                finding=item,
                proposition=slots[item.obligation_id],
                evidence=tuple(refs[ref] for ref in item.evidence_refs),
            )
        )
    return tuple(validated)


__all__ = [
    "PRECISE_ANSWER_VERSION",
    "PRECISE_VALUE_LIMIT",
    "PRECISE_WITNESS_LIMIT",
    "PreciseAnswerEnvelopeV3",
    "PreciseAnswerFindingV3",
    "PreciseAnswerWitnessV3",
    "PreciseClaimProjection",
    "PreciseEnvelopeV3",
    "PreciseFindingV3",
    "PreciseValidatedFindingV3",
    "bind_findings",
    "canonical_answer",
    "decode_canonical_answer",
    "parse_precise_answer",
]
