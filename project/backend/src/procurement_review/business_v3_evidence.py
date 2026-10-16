"""finding-evidence.v1 sidecar and field-by-field source support validation."""

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, ValidationError

from .business_v3_documents import DocumentValuesV3, decode_document_values
from .business_v3_security import ProtectedOutputV3
from .business_v3_support import SupportIssue, WitnessV3
from .business_v3_types import VALUE_MODELS
from .eval_v2_contracts import EvalV2Model
from .model_wire_v2 import WireFinding, WireIssue, parse_findings, strict_json


class EvidenceEnvelopeV3(EvalV2Model):
    schema_version: Literal["finding-evidence.v1"]
    findings: tuple[WireFinding, ...] = Field(min_length=1, max_length=12)
    witnesses: tuple[WitnessV3, ...] = Field(max_length=128)


@dataclass(frozen=True)
class GroundedDocumentV3:
    document: DocumentValuesV3
    spans: tuple
    support_failures: tuple[str, ...]


def evidence_schema_v3():
    return EvidenceEnvelopeV3.model_json_schema()


def parse_evidence_v3(body, request, *, protection: ProtectedOutputV3):
    """No correction decision is made here; security/citation failures are final."""
    try:
        raw = strict_json(body)
        protection.assert_safe(raw)
        envelope = EvidenceEnvelopeV3.model_validate_json(json.dumps(raw), strict=True)
    except WireIssue:
        raise
    except (ValueError, ValidationError):
        raise WireIssue("evidence_envelope_invalid") from None
    findings = parse_findings(
        json.dumps(
            {
                "schema_version": "finding-envelope.v2",
                "findings": [f.model_dump(mode="json") for f in envelope.findings],
            }
        ),
        request,
    )
    return decode_document_values(request, findings), envelope.witnesses


def ground_document_v3(request, document, witnesses, catalog):
    if document.worker != request.worker:
        raise SupportIssue("document_role_mismatch")
    values = dict(document.values)
    spans, failures, invalid = [], [], set()
    allowed_fields = {
        key: set(model.model_fields)
        for key, model in VALUE_MODELS[request.worker].items()
    }
    cited = {
        f.finding.obligation_id: set(f.finding.evidence_refs) for f in document.findings
    }
    for witness in witnesses:
        key = witness.obligation_id
        if (
            key not in allowed_fields
            or witness.field not in allowed_fields[key]
            or key not in values
        ):
            # An extra role/check/field cannot silently disappear from an answer.
            return GroundedDocumentV3(
                DocumentValuesV3(
                    request.worker, {}, (), ("document_citation_invalid",)
                ),
                (),
                ("citation_field_outside_task",),
            )
        try:
            if witness.source_ref not in cited.get(key, set()):
                raise SupportIssue("citation_witness_not_cited")
            spans.append(catalog.verify(request, witness))
        except SupportIssue as exc:
            invalid.add(key)
            failures.append(exc.code)
    for key, value in values.items():
        for name, expected in value.model_dump().items():
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
    valid = {key: value for key, value in values.items() if key not in invalid}
    # All sides of a proven contradiction survive, including two sides in one finding.
    spans = tuple(s for s in spans if s.proposition.check_id not in invalid)
    errors = (*document.failures, *(("document_citation_invalid",) if failures else ()))
    return GroundedDocumentV3(
        DocumentValuesV3(request.worker, valid, document.findings, errors),
        spans,
        tuple(sorted(set(failures))),
    )
