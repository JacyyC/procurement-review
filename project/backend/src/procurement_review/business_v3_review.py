"""Opt-in precise review: support first, then conflicts and orthogonal events."""

from typing import Literal

from .business_v3_aggregate import (
    aggregate_business_v3,
    business_observations,
    decide_business_v3,
)
from .business_v3_conflicts import ConflictPairV3, conflict_pairs_v3
from .business_v3_documents import DocumentValuesV3
from .business_v3_evidence import GroundedDocumentV3
from .business_v3_security import (
    SecurityEventV3,
    ProtectedOutputV3,
    inspect_untrusted_input,
)
from .business_v3_support import VerifiedSpanV3
from .business_v3_types import (
    BusinessDecisionV3,
    BusinessEvidenceIdentity,
    BusinessObservationV3,
    finding,
)
from .eval_v2_contracts import EvalV2Model
from .gold_v3_contracts import payload_sha256


class PreciseDecisionV3(EvalV2Model):
    schema_version: Literal["business-review.v3"] = "business-review.v3"
    business: BusinessDecisionV3
    conflicts: tuple[ConflictPairV3, ...]
    security: tuple[SecurityEventV3, ...]
    spans: tuple[VerifiedSpanV3, ...]
    support_failures: tuple[str, ...]


def finalize_review_v3(
    context,
    policy,
    contract,
    procurement,
    *,
    sources,
    protection: ProtectedOutputV3,
    pipeline_failures=(),
    control_events=(),
):
    # Only control code can supply failures. The model sidecar has no such field.
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
    baseline = aggregate_business_v3(context, *documents, procurement)
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
        version="business-review.v3",
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
    decision = decide_business_v3(
        rows,
        pipeline_failures=(*baseline.pipeline_failures, *pipeline_failures),
        security_events=reasons,
        conflict_pairs=tuple(p.pair_id for p in pairs),
        observations=observations,
    )
    result = PreciseDecisionV3(
        business=decision,
        conflicts=pairs,
        security=security,
        spans=tuple(spans),
        support_failures=tuple(sorted(set(support_failures))),
    )
    protection.assert_safe(result.model_dump(mode="json"))
    return result
