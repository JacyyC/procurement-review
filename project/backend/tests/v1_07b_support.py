"""Small constructors shared by the V1-07B offline contract tests."""

from __future__ import annotations

from functools import lru_cache

from procurement_review.eval_v2_contracts import (
    ConflictProjection,
    EvalPredictionV2,
    EvalTopology,
    ForbiddenEvent,
    RetrievalObservation,
    RetrievalQueryVariant,
    SecurityEventProjection,
    SemanticClaimProjection,
)
from procurement_review.gold_v3 import load_bundle
from procurement_review.gold_v3_contracts import (
    GoldCasesV3File,
    GoldManifestV3,
    GoldTruthV1File,
    TruthRecordV1,
    V3Recommendation,
    payload_sha256,
)


@lru_cache(maxsize=1)
def v3_bundle() -> tuple[GoldCasesV3File, GoldTruthV1File, GoldManifestV3]:
    return load_bundle()


def prediction_for(
    truth: TruthRecordV1,
    *,
    topology: EvalTopology = EvalTopology.MULTI_AGENT,
    repeat_seed: int = 20260919,
    recommendation: V3Recommendation | None = None,
    model_request_count: int = 1,
    first_pass_valid: bool = True,
    injected_fault: bool = False,
) -> EvalPredictionV2:
    claims = tuple(
        SemanticClaimProjection(
            obligation_id=item.obligation_id,
            state=item.state,
            proposition=item.proposition,
            normalized_value=item.normalized_value,
            evidence=item.evidence,
            reason_codes=(item.reason_code,),
            statement_sha256=payload_sha256(
                {"case_id": truth.case_id, "claim": item.obligation_id}
            ),
            recommended_action_sha256=payload_sha256(
                {"case_id": truth.case_id, "action": item.obligation_id}
            ),
        )
        for item in truth.obligations
    )
    conflicts = tuple(
        ConflictProjection(
            pair_id=item.pair_id,
            left=item.left,
            right=item.right,
            left_value=item.left_value,
            right_value=item.right_value,
        )
        for item in truth.conflicts
    )
    security = tuple(
        SecurityEventProjection(
            event_type=item.event_type,
            source_id=item.source_id,
            location=item.location,
        )
        for item in truth.security_events
    )
    retrieval = []
    for variant in RetrievalQueryVariant:
        for domain, suffix in (("policy", "-POLICY"), ("contract", "-CONTRACT")):
            required = tuple(
                item
                for item in truth.required_evidence
                if item.source_id.endswith(suffix)
            )
            if required:
                retrieval.append(
                    RetrievalObservation(
                        query_variant=variant,
                        evidence_domain=domain,
                        required_evidence=required,
                        top_five=required,
                    )
                )
    response_observed = model_request_count > 0
    return EvalPredictionV2(
        case_id=truth.case_id,
        topology=topology,
        repeat_seed=repeat_seed,
        recommendation=recommendation or truth.expected_recommendation,
        claims=claims,
        conflicts=conflicts,
        security_events=security,
        retrieved_evidence=truth.required_evidence,
        retrieval_observations=tuple(retrieval),
        first_pass_valid=first_pass_valid if response_observed else False,
        received_nonempty_body=response_observed,
        corrected_valid=False,
        pipeline_failure=False,
        injected_fault=injected_fault,
        latency_ms=1 if response_observed else None,
        model_request_count=model_request_count,
        forbidden_events=(),
        forbidden_event_exposures=tuple(ForbiddenEvent),
    )


__all__ = ["prediction_for", "v3_bundle"]
