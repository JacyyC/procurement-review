"""Rules 3.1.0 evidence grounding, review finalization and its report shape.

The same disciplines as the frozen 3.0.0 line apply unchanged — inspection of raw
and decoded output, one witness per reported field, full-span citation, bounded
projection — while the versioned value registry and the conditional field
recognizers decide what counts as support.  The result records
``business-review.v3.1`` and a rules 3.1.0 decision, so a stored review says
which rule set produced it instead of inheriting the frozen version string.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field, ValidationError

from .business_v3_conflicts import ConflictPairV3, conflict_pairs_v3
from .business_v3_documents import DocumentValuesV3
from .business_v3_evidence import GroundedDocumentV3
from .business_v3_security import (
    ProtectedOutputV3,
    SecurityEventV3,
    inspect_untrusted_input,
)
from .business_v3_support import (
    EXEMPTION_LANGUAGE_FIELDS,
    SourceCatalogV3,
    SupportIssue,
    WitnessV3,
)
from .business_v3_conditional import (
    BusinessDecisionV31,
    aggregate_business_v31_documents,
    decide_business_v31,
)
from .business_v3_types import (
    BusinessEvidenceIdentity,
    BusinessObservationV3,
    finding,
)
from .business_v3_aggregate import business_observations
from .business_v31_documents import VALUE_MODELS_V31, supports_field_v31
from .eval_v2_contracts import EvalV2Model
from .gold_v3_contracts import payload_sha256
from .model_wire_v2 import WireFinding, WireIssue, parse_findings, strict_json

REVIEW_V31_VERSION = "business-review.v3.1"

#: Fields whose own wording is allowed to carry exclusion language.  The policy
#: rule and the contract transaction are exactly the places where an exclusion or
#: an exception is the proposition being reported, so they join the frozen set.
EXEMPTION_FIELDS_V31 = EXEMPTION_LANGUAGE_FIELDS | {
    "minimum_quotes",
    "applies_to",
    "excluded_categories",
    "exclusion_basis",
    "exceptions",
    "exception_requires",
    "exception_basis",
    "category",
    "category_basis",
    "single_source",
    "written_justification",
    "approval_reference",
    "price_reasonableness",
    "commitment_current",
    "superseding_amendment",
}


class EvidenceEnvelopeV31(EvalV2Model):
    schema_version: Literal["finding-evidence.v2"]
    findings: tuple[WireFinding, ...] = Field(min_length=1, max_length=12)
    witnesses: tuple[WitnessV3, ...] = Field(max_length=192)


class PreciseDecisionV31(EvalV2Model):
    """The stored review of the conditional line, with its own version string."""

    schema_version: Literal["business-review.v3.1"] = REVIEW_V31_VERSION
    rules_version: Literal["3.1.0"] = "3.1.0"
    business: BusinessDecisionV31
    conflicts: tuple[ConflictPairV3, ...]
    security: tuple[SecurityEventV3, ...]
    spans: tuple
    support_failures: tuple[str, ...]


def evidence_schema_v31():
    return EvidenceEnvelopeV31.model_json_schema()


#: The declared keys a schema refusal may name, so the field set stays a fixed
#: vocabulary instead of echoing model text back into the receipt.
SCHEMA_FIELDS = frozenset(
    {
        "schema_version",
        "findings",
        "witnesses",
        "obligation_id",
        "field",
        "value",
        "source_ref",
        "start",
        "end",
        "status",
        "assertion",
        "risk_level",
        "normalized_value",
        "statement",
        "recommended_action",
        "evidence_refs",
        "missing_information",
    }
)


#: How many location tokens one refusal may carry.  The set is bounded here as well as in
#: the receipt contract, so a badly damaged answer cannot grow the diagnostic without end.
DIAGNOSTIC_TOKEN_LIMIT = 12

#: How many refused locations one refusal reports.  The first location is the primary
#: root cause; further ones are reported only while the token budget allows.
DIAGNOSTIC_GROUP_LIMIT = 4


class _SchemaDiagnostics:
    """What one refused 3.1.0 envelope is allowed to say about itself.

    The refusal names the declared field(s) it touched, in the fixed vocabulary, plus a
    bounded set of location tokens: the slot and index it failed at, the duty the finding
    declared (only when it is one of the authorized checks), the validation category, the
    expected type word beside the *basic* JSON type actually found, and a bounded size.
    Values, answer text, prompts and arbitrary model prose never leave this boundary.

    One refused field is one root cause.  A finding is judged by the union alternative its
    own declared state selects, so the other alternatives' mismatches are dropped instead
    of being reported as five separate faults; a state word that selects nothing is itself
    the root cause; and a scalar field refused by every alternative of a scalar union is
    reported once, as "expected=scalar".
    """

    __slots__ = ("_authorized", "_groups", "_raw")

    _FINDING_BRANCHES = {
        "supported": "SupportedFinding",
        "violated": "ViolatedFinding",
        "unknown": "UnknownFinding",
        "not_applicable": "NotApplicableFinding",
    }
    _WORD_BY_ERROR = {
        "missing": ("required_missing", "present"),
        "literal_error": ("word_not_registered", "registered_word"),
        "enum": ("word_not_registered", "registered_word"),
        "string_type": ("type_mismatch", "string"),
        "int_type": ("type_mismatch", "integer"),
        "bool_type": ("type_mismatch", "boolean"),
        "float_type": ("type_mismatch", "number"),
        "too_short": ("size_out_of_range", "minimum"),
        "too_long": ("size_out_of_range", "maximum"),
        "string_too_short": ("size_out_of_range", "minimum"),
        "string_too_long": ("size_out_of_range", "maximum"),
        "list_type": ("type_mismatch", "array"),
        "dict_type": ("type_mismatch", "object"),
        "value_error": ("value_rule", "rule"),
    }

    #: The closed rule vocabulary of this line's own state validator, keyed by the fixed
    #: sentence it raises.  A refusal that names the rule pinpoints the one field to redo
    #: instead of leaving the correction to guess; the sentence itself never leaves this
    #: boundary, only its registered word and the field it is about.
    _RULE_BY_MESSAGE = {
        "derived assertion disagrees with its state": (
            "derived_assertion",
            "assertion",
        ),
        "an unknown finding carries the literal and no citation": (
            "unknown_value_or_citation",
            "evidence_refs",
        ),
        "an unknown finding states what is missing": (
            "unknown_missing_required",
            "missing_information",
        ),
        "a decided finding carries a value and its citation": (
            "decided_value_or_citation",
            "evidence_refs",
        ),
        "a decided finding carries no missing list": (
            "decided_missing_forbidden",
            "missing_information",
        ),
        "only a violated finding carries a severity": (
            "severity_scope",
            "risk_level",
        ),
    }

    def __init__(self, error: Exception, raw: Any, request: Any) -> None:
        self._raw = raw if isinstance(raw, dict) else {}
        self._authorized = {
            check.proposition.check_id for check in getattr(request, "checks", ())
        }
        errors = getattr(error, "errors", None)
        item_groups: dict[tuple[str, int | None, str], list[str]] = {}
        container_groups: dict[tuple[str, int | None, str], list[str]] = {}
        if callable(errors):
            for item in errors(
                include_input=False, include_context=False, include_url=False
            ):
                location = item.get("loc") if isinstance(item, dict) else None
                if not isinstance(location, tuple) or not location:
                    continue
                slot = str(location[0])
                index = (
                    location[1]
                    if len(location) > 1 and isinstance(location[1], int)
                    else None
                )
                if slot == "findings" and self._is_foreign_branch(location, index):
                    continue
                field = next(
                    (part for part in reversed(location) if part in SCHEMA_FIELDS), None
                )
                if field is None:
                    if slot == "findings" and index is not None:
                        # The state word selects no alternative: that is the fault.
                        field = "status"
                    else:
                        field = "witnesses" if slot == "witnesses" else "findings"
                if str(item.get("type")) == "value_error":
                    # A rule of this line's own validator: report which rule and which
                    # field it is about instead of the container the finding lives in.
                    word, target_field = self._rule_of(item.get("msg"))
                    field = target_field or field
                    target = item_groups if index is not None else container_groups
                    target.setdefault((slot, index, field), []).append(
                        f"value_error:{word}"
                    )
                    continue
                target = item_groups if index is not None else container_groups
                target.setdefault((slot, index, field), []).append(
                    str(item.get("type"))
                )
        # A container that reports "not one item survived" only says that its items
        # failed; the items themselves are the root cause and are reported instead.
        for key, types in container_groups.items():
            if key[0] in ("findings", "witnesses") and any(
                other[0] == key[0] and other[1] is not None for other in item_groups
            ):
                continue
            item_groups[key] = types
        self._groups = item_groups

    def _rule_of(self, message: Any) -> tuple[str, str | None]:
        """The registered word and field of one of this line's own rule refusals."""

        text = str(message or "").removeprefix("Value error, ")
        return self._RULE_BY_MESSAGE.get(text, ("unregistered_rule", None))

    def _is_foreign_branch(self, location: tuple, index: int | None) -> bool:
        """Whether one finding error belongs to a union alternative the answer rejected."""

        if index is None:
            return False
        item = self._finding(index)
        state = item.get("status") if isinstance(item, dict) else None
        selected = self._FINDING_BRANCHES.get(state) if isinstance(state, str) else None
        if selected is None:
            # An unregistered or missing state selects nothing: only the state field's
            # own refusal is the root cause, and every alternative's mismatch is noise.
            return location[-1] != "status"
        if len(location) < 3:
            return False
        tag = str(location[2])
        if not (tag.endswith("Finding]") or tag in self._FINDING_BRANCHES.values()):
            # Not a union alternative tag at all: a single-model envelope locates its
            # errors by field name, and this one is about this answer.
            return False
        return tag != selected and not tag.endswith(f", {selected}]")

    def _finding(self, index: int) -> Any:
        items = self._raw.get("findings")
        if not isinstance(items, list) or index >= len(items):
            return None
        return items[index]

    def summary(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The declared field names and the bounded tokens this refusal may name."""

        if not self._groups:
            return ("witnesses",), ()
        fields: set[str] = set()
        ordered: list[str] = []
        groups = sorted(
            self._groups.items(),
            key=lambda item: (
                item[0][0],
                item[0][1] if item[0][1] is not None else -1,
                item[0][2],
            ),
        )
        for (slot, index, field), error_types in groups[:DIAGNOSTIC_GROUP_LIMIT]:
            fields.add(field)
            rule = next(
                (
                    name.split(":", 1)[1]
                    for name in error_types
                    if name.startswith("value_error:")
                ),
                None,
            )
            words = [
                self._WORD_BY_ERROR.get(name)
                for name in error_types
                if not name.startswith("value_error:")
            ]
            words = [word for word in words if word is not None]
            if rule is not None:
                words.append(("value_rule", "rule"))
            category, expected = (
                words[0] if words else ("unclassified", "declared_type")
            )
            value = self._value_at(slot, index, field)
            actual = basic_json_type(value)
            if actual == "null" and category in {
                "word_not_registered",
                "type_mismatch",
            }:
                # A declared slot that arrived as null is a missing answer, not a
                # misspelled one: the expected word says which of the two it is.
                category, expected = "required_missing", "present"
            group_tokens = [f"slot={slot}"]
            duty = self._declared_duty(slot, index)
            if duty is not None:
                group_tokens.append(f"duty={duty}")
            if index is not None:
                group_tokens.append(f"index={index}")
            group_tokens.append(f"path={field}")
            group_tokens.append(f"category={category}")
            group_tokens.append(
                "expected="
                + (expected if len({word[1] for word in words}) <= 1 else "scalar")
            )
            if rule is not None:
                # The registered rule name is what the one correction is re-asked to fix.
                group_tokens.append(f"rule={rule}")
            group_tokens.append(f"actual={actual}")
            size = _bounded_size(value)
            if size is not None:
                group_tokens.append(f"size={size}")
            for token in group_tokens:
                if token not in ordered:
                    ordered.append(token)
        return tuple(sorted(fields)) or ("witnesses",), tuple(
            ordered[:DIAGNOSTIC_TOKEN_LIMIT]
        )

    def _declared_duty(self, slot: str, index: int | None) -> str | None:
        if slot != "findings" or index is None:
            return None
        item = self._finding(index)
        duty = item.get("obligation_id") if isinstance(item, dict) else None
        # Only an authorized check id may be echoed; anything else stays unquoted.
        return duty if isinstance(duty, str) and duty in self._authorized else None

    def _value_at(self, slot: str, index: int | None, field: str) -> Any:
        container: Any = self._raw.get(slot)
        if index is not None:
            if not isinstance(container, list) or index >= len(container):
                return _ABSENT
            container = container[index]
        if not isinstance(container, dict) or field not in container:
            return _ABSENT
        return container[field]


_ABSENT: Any = object()

_BASIC_TYPES = {
    str: "string",
    int: "number",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def basic_json_type(value: Any) -> str:
    """The basic JSON type of one raw answer value, or ``absent``."""

    if value is _ABSENT:
        return "absent"
    return _BASIC_TYPES.get(type(value), "unknown")


def _bounded_size(value: Any) -> int | None:
    """A bounded length/count of one raw value, never its content."""

    if isinstance(value, str):
        return min(len(value), 9999)
    if isinstance(value, list):
        return min(len(value), 999)
    return None


def _schema_fields(error: Any, raw: Any, request: Any):
    """The declared field names and bounded location tokens one refusal may name."""

    diagnostics = _SchemaDiagnostics(error, raw, request)
    return diagnostics.summary()


def parse_evidence_v31(body, request, *, protection: ProtectedOutputV3):
    """Decode one 3.1.0 sidecar; security and citation failures stay final."""

    try:
        raw = strict_json(body)
        protection.assert_safe(raw)
        envelope = EvidenceEnvelopeV31.model_validate_json(json.dumps(raw), strict=True)
    except WireIssue:
        raise
    except (ValueError, ValidationError) as error:
        # The refusal names the declared field(s) and the bounded location tokens so the
        # one correction can target them; the offending value itself never leaves here.
        fields, tokens = _schema_fields(error, raw, request)
        raise WireIssue(
            "evidence_envelope_invalid", fields=fields, details=tokens
        ) from None
    findings = parse_findings(
        json.dumps(
            {
                "schema_version": "finding-envelope.v2",
                "findings": [f.model_dump(mode="json") for f in envelope.findings],
            }
        ),
        request,
    )
    from .business_v31_documents import decode_document_values_v31

    return decode_document_values_v31(request, findings), envelope.witnesses


def ground_document_v31(request, document, witnesses, catalog: SourceCatalogV3):
    """Bind every reported 3.1.0 field to its own attributed clause."""

    grounded, _locations = ground_document_v31_located(
        request, document, witnesses, catalog
    )
    return grounded


def ground_document_v31_located(request, document, witnesses, catalog: SourceCatalogV3):
    """Bind every field, and name the bounded location of each refusal.

    The locations are the registered obligation id, the registered field name and the
    fixed support category — never a value, a span text or a raw answer.  A caller that
    may ask one bounded correction uses them; a caller that only needs the document
    ignores them.

    A refusal costs the duty it belongs to and nothing else.  A witness whose field is not
    one of the duty's own declared dimensions makes that duty unresolved, exactly as an
    unsupported value does, instead of discarding every other duty of the same document;
    the document-level ``document_citation_invalid`` is written only when no duty survived,
    because that is the only case in which the document itself is unusable.  Other duties
    of the same branch keep their own resolved values, in line with the rule that a locally
    missing dimension does not erase the facts a branch did establish.
    """

    if document.worker != request.worker:
        raise SupportIssue("document_role_mismatch")
    values = dict(document.values)
    spans, failures, invalid = [], [], set()
    locations: list[str] = []
    allowed_fields = {
        key: set(model.model_fields)
        for key, model in VALUE_MODELS_V31[request.worker].items()
    }
    cited = {
        f.finding.obligation_id: set(f.finding.evidence_refs) for f in document.findings
    }
    for witness in witnesses:
        key = witness.obligation_id
        if key not in allowed_fields or witness.field not in allowed_fields[key]:
            # The witness claims a dimension this branch was not asked about. Its own duty
            # is unresolved; it cannot cost the duties that were answered correctly.
            invalid.add(key)
            failures.append("citation_field_outside_task")
            locations.append("category=citation_field_outside_task")
            locations.append(f"obligation={key}")
            locations.append(f"field={witness.field}")
            continue
        if key not in values:
            # The field name is fine: the *value* of that duty did not survive the closed
            # registry (a half-declared exception, an out-of-vocabulary word). Reporting it
            # as a field outside the task would name the wrong root cause and ask for the
            # wrong correction, so the duty with no accepted value is named as such.
            invalid.add(key)
            failures.append("citation_duty_value_invalid")
            locations.append("category=citation_duty_value_invalid")
            locations.append(f"obligation={key}")
            locations.append(f"field={witness.field}")
            continue
        try:
            if witness.source_ref not in cited.get(key, set()):
                raise SupportIssue("citation_witness_not_cited")
            spans.append(
                catalog.verify(
                    request,
                    witness,
                    supports=supports_field_v31,
                    exemption_fields=EXEMPTION_FIELDS_V31,
                )
            )
        except SupportIssue as exc:
            invalid.add(key)
            failures.append(exc.code)
            locations.append(f"category={exc.code}")
            locations.append(f"obligation={key}")
            locations.append(f"field={witness.field}")
    for key, value in values.items():
        for name, expected in value.model_dump().items():
            if expected is None:
                # A dimension the source did not state is not a claim: the Worker
                # leaves it empty and the aggregator must read it as missing.
                continue
            if isinstance(expected, tuple):
                expected = json.dumps(expected, separators=(",", ":"))
            matches = [
                s
                for s in spans
                if s.proposition.check_id == key
                and s.proposition.field_path.endswith("." + name)
                and type(s.value) is type(expected)
                and s.value == expected
            ]
            if not matches:
                invalid.add(key)
                failures.append("citation_field_value_unsupported")
                locations.append("category=citation_field_value_unsupported")
                locations.append(f"obligation={key}")
                locations.append(f"field={name}")
    valid = {key: value for key, value in values.items() if key not in invalid}
    spans = tuple(s for s in spans if s.proposition.check_id not in invalid)
    errors = (
        *document.failures,
        *(("document_citation_invalid",) if failures and not valid else ()),
    )
    return (
        GroundedDocumentV3(
            DocumentValuesV3(request.worker, valid, document.findings, errors),
            spans,
            tuple(sorted(set(failures))),
        ),
        tuple(dict.fromkeys(locations)),
    )


def finalize_review_v31(
    context,
    policy,
    contract,
    procurement,
    *,
    sources,
    protection: ProtectedOutputV3,
    pipeline_failures=(),
    control_events=(),
    branch_inspections=None,
):
    """Apply rules 3.1.0 to the grounded branches and record a 3.1.0 review.

    ``branch_inspections`` carries the bounded security projection of a branch whose
    output was inspected when it was first executed.  A reused branch restates its
    findings from its commit, so its text is inspected once, by the execution that
    actually received it; every other branch is inspected here as before.
    """
    permitted = {
        "timeout",
        "transport",
        "cancelled",
        "authorization_revoked",
        "deadline",
        "safety_refusal",
        "security_boundary",
        "tool_call_forbidden",
        "document_model_failed",
        "evidence_envelope_invalid",
    }
    if set(pipeline_failures) - permitted:
        raise ValueError("unregistered control failure")
    if any(type(e) is not SecurityEventV3 for e in control_events):
        raise ValueError("invalid control event")
    security = list(control_events)
    for source in sources:
        security.extend(inspect_untrusted_input(source.content, location=source.ref))
    documents = []
    spans = []
    support_failures = []
    for grounded in (policy, contract):
        if not isinstance(grounded, GroundedDocumentV3):
            raise TypeError("grounded document required")
        document = grounded.document
        if branch_inspections and document.worker in branch_inspections:
            checked = list(branch_inspections[document.worker])
        else:
            checked = protection.inspect(
                [f.model_dump(mode="json") for f in document.findings],
                location=document.worker,
            )
        security.extend(checked)
        support_failures.extend(grounded.support_failures)
        if any(e.escalates for e in checked):
            documents.append(
                DocumentValuesV3(document.worker, {}, (), ("document_output_blocked",))
            )
        else:
            documents.append(document)
            spans.extend(grounded.spans)
    baseline = aggregate_business_v31_documents(context, *documents, procurement)
    observations = list(business_observations(context, procurement, ()))
    observations.extend(
        BusinessObservationV3(
            proposition=s.proposition, value=s.value, evidence=(s.evidence,)
        )
        for s in spans
    )
    pairs = conflict_pairs_v3(observations)
    security = tuple(
        sorted(set(security), key=lambda e: (e.code, e.origin, e.location_sha256))
    )
    control_evidence = BusinessEvidenceIdentity(
        source_id="trusted-review-control",
        version=REVIEW_V31_VERSION,
        locator="source-and-output-inspection",
        content_sha256=payload_sha256(
            {
                "sources": [s.ref for s in sources],
                "events": [e.model_dump(mode="json") for e in security],
            }
        ),
    )
    reasons = tuple(sorted({e.code for e in security if e.escalates}))
    pair_evidence = tuple(
        e for pair in pairs for side in (pair.left, pair.right) for e in side.evidence
    )
    rows = list(baseline.findings)
    rows.append(
        finding(
            "O-11",
            "violated" if pairs else "unknown" if support_failures else "supported",
            "source_conflict_proven"
            if pairs
            else "source_support_unresolved"
            if support_failures
            else "sources_consistent",
            pair_evidence or (control_evidence,),
            risk="medium" if pairs else "unknown" if support_failures else "none",
            missing=("supported_source_propositions",)
            if support_failures and not pairs
            else (),
        )
    )
    rows.append(
        finding(
            "O-12",
            "supported",
            "output_boundary_checked",
            (control_evidence,),
        )
    )
    decision = decide_business_v31(
        rows,
        pipeline_failures=(*baseline.pipeline_failures, *pipeline_failures),
        security_events=reasons,
        conflict_pairs=tuple(p.pair_id for p in pairs),
        observations=observations,
    )
    result = PreciseDecisionV31(
        business=decision,
        conflicts=pairs,
        security=security,
        spans=tuple(spans),
        support_failures=tuple(sorted(set(support_failures))),
    )
    protection.assert_safe(result.model_dump(mode="json"))
    return result


__all__ = [
    "ground_document_v31_located",
    "EXEMPTION_FIELDS_V31",
    "REVIEW_V31_VERSION",
    "EvidenceEnvelopeV31",
    "PreciseDecisionV31",
    "evidence_schema_v31",
    "finalize_review_v31",
    "ground_document_v31",
    "parse_evidence_v31",
]
