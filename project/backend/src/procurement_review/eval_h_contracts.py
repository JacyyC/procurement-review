"""Explicit H contracts. Historical G shapes and readers keep their own semantics."""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from .eval_f_contracts import FEndpointConfig
from .gold_v3_contracts import FrozenModel, HexSha256, ObligationId, V3Recommendation
from .run_stop import (
    # Re-exported: the frozen names keep their original import path, and the single
    # classification lives in ``run_stop``.
    H_STOP_MEASURED as H_STOP_MEASURED,
    H_STOP_UNTRUSTED as H_STOP_UNTRUSTED,
)

H_PLAN = "v1-07h-plan.v1"
H_PLAN_V2 = "v1-08h-plan.v1"
H_PROBE = "v1-08c-probe.v1"
H_CASES = "v1-07h-cases.v1"
H_LABELS = "v1-07h-labels.v1"
H_GENERATOR = "v1-07h-material-generator.v1"
H_SEEDED_GENERATOR = "v1-07h-material-generator.v2"
H_SEAL = "blind-seal.v2"
H_SCORE = "v1-07h-score.v2"
H_CONSUMPTION = "v1-07h-consumption.v2"
H_SAFETY_SOURCE = "v1-07h-safety-source.v1"
#: The V1-08 accepted Run-deadline candidates (Spec P-04); the C probe registers its own
#: deadline from this set, and D's final product configuration is locked to one of them.
#: 240 seconds is a boundary that needs its own decision and is not accepted here.
ACCEPTED_GENERATION_DEADLINES = (120.0, 180.0)
#: The generation deadline the frozen V1-07 batches were created with.  A plan that does
#: not state a deadline (the historical schema) is read with this value, so no old
#: observation is re-interpreted by the new field.
V1_GENERATION_DEADLINE_SECONDS = 30.0
#: The C probe's own registered Run deadline (Spec P-04): it must be the candidate that
#: covers the 90–150 second decision interval, so only the 180-second value is admissible
#: for a probe — a probe registered at 120 seconds would censor the very interval its
#: deadline is chosen from, and a deadline picked after the probe would not be a
#: pre-registration at all.
PROBE_GENERATION_DEADLINE_SECONDS = 180.0
#: The C probe's registered size: 8–12 already-seen DEV Cases, each with its two model
#: branches and exactly one initial observation (Spec P-03).
PROBE_CASE_RANGE = (8, 12)
#: The one seed a C probe registers.  A single seed is what keeps the probe at one initial
#: observation per Case: the second repeat of the historical grid is not a probe.
C_PROBE_SEED = 901
ARMS = ("multi_agent", "single_agent")
ROLES = ("policy", "contract")
DOCUMENT_DUTIES = {
    "policy": ("O-01", "O-02", "O-04"),
    "contract": ("O-04", "O-05", "O-06", "O-08", "O-09"),
}
QUOTAS = {
    "recommend_approve": 40,
    "recommend_conditional": 30,
    "recommend_reject": 40,
    "insufficient_information": 30,
    "needs_human_review": 60,
}
Sha = HexSha256
# H uses B-prefixed formal planning identifiers in addition to the D/V
# identifiers carried by disposable material.  Keep the historical G type
# unchanged so its readers retain their original contract.
HCaseId = Annotated[str, Field(pattern=r"^V107-[BDV]-[0-9]{4}$")]
Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,128}$")]


class HDocument(FrozenModel):
    role: Literal["policy", "contract"]
    source_id: Name
    version: Name
    document_id: Name
    content: Annotated[str, Field(min_length=1, max_length=16_000)]
    content_sha256: Sha

    @model_validator(mode="after")
    def content_is_authentic(self) -> Self:
        import hashlib

        if hashlib.sha256(self.content.encode()).hexdigest() != self.content_sha256:
            raise ValueError("h_document_checksum")
        return self


class HRecord(FrozenModel):
    kind: Literal["purchase_order", "supplier", "historical_transaction"]
    record_id: Name
    payload: dict[str, Any]


class HCase(FrozenModel):
    case_id: HCaseId
    generation_family_id: Name
    near_duplicate_group_id: Name
    entity_root_id: Name
    metamorphic_parent_id: HCaseId | None
    question: Annotated[str, Field(min_length=1, max_length=4_000)]
    as_of: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    documents: tuple[HDocument, ...] = Field(max_length=32)
    records: tuple[HRecord, ...] = Field(min_length=3, max_length=16)
    purchase_order_id: Name

    @model_validator(mode="after")
    def records_are_product_inputs(self) -> Self:
        from datetime import date

        date.fromisoformat(self.as_of)
        ids = [document.document_id for document in self.documents]
        refs = [document.source_id for document in self.documents]
        if len(ids) != len(set(ids)) or len(refs) != len(set(refs)):
            raise ValueError("h_document_identity_duplicate")
        kinds = set()
        record_ids = []
        for row in self.records:
            kinds.add(row.kind)
            record_ids.append(row.record_id)
            if (
                row.payload.get(
                    {
                        "purchase_order": "purchase_order_id",
                        "supplier": "supplier_id",
                        "historical_transaction": "transaction_id",
                    }[row.kind]
                )
                != row.record_id
            ):
                raise ValueError("h_record_payload_identity")
        if kinds != {"purchase_order", "supplier", "historical_transaction"}:
            raise ValueError("h_fact_tools_incomplete")
        if (
            len(record_ids) != len(set(record_ids))
            or self.purchase_order_id not in record_ids
        ):
            raise ValueError("h_record_identity")
        if self.metamorphic_parent_id == self.case_id:
            raise ValueError("h_metamorphic_self_parent")
        return self


class HSupport(FrozenModel):
    role: Literal["policy", "contract"]
    obligation_id: ObligationId
    field: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    value: Any
    clauses: tuple[str, ...] = Field(min_length=1, max_length=8)
    requirement: Literal["required", "optional"]


class HLabel(FrozenModel):
    case_id: HCaseId
    recommendation: V3Recommendation
    high_risk: bool
    expectations: tuple[dict[str, Any], ...] = Field(min_length=12, max_length=12)
    support: tuple[HSupport, ...] = Field(max_length=192)
    required_sources: dict[Literal["policy", "contract"], tuple[str, ...]]
    conflict_pairs: tuple[dict[str, Any], ...] = Field(max_length=8)
    security_events: tuple[dict[str, Any], ...] = Field(max_length=8)
    required_facts: dict[ObligationId, tuple[str, ...]]
    adjudication: Annotated[str, Field(min_length=1, max_length=1_000)]

    @model_validator(mode="after")
    def independent_catalog_is_complete(self) -> Self:
        if tuple(row.get("obligation_id") for row in self.expectations) != tuple(
            f"O-{number:02d}" for number in range(1, 13)
        ):
            raise ValueError("h_truth_obligation_catalog")
        from .gold_v38_dev_quality import ObligationExpectationV1
        import json

        for row in self.expectations:
            ObligationExpectationV1.model_validate_json(json.dumps(row), strict=True)
        keys = [(item.role, item.obligation_id, item.field) for item in self.support]
        if len(keys) != len(set(keys)):
            raise ValueError("h_truth_duplicate_field")
        if any(
            item.obligation_id not in DOCUMENT_DUTIES[item.role]
            for item in self.support
        ):
            raise ValueError("h_truth_field_role")
        if set(self.required_sources) != set(ROLES):
            raise ValueError("h_truth_required_source_domains")
        if self.conflict_pairs or self.security_events:
            if self.recommendation.value != "needs_human_review":
                raise ValueError("h_truth_event_priority")
        return self


class HRecipe(FrozenModel):
    generator_version: Literal[
        "v1-07h-material-generator.v1", "v1-07h-material-generator.v2"
    ]
    cases: tuple[HCase, ...] = Field(min_length=1, max_length=200)
    labels: tuple[HLabel, ...] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def labels_are_bound_to_material(self) -> Self:
        case_ids = [case.case_id for case in self.cases]
        labels = {label.case_id: label for label in self.labels}
        if (
            len(case_ids) != len(set(case_ids))
            or len(labels) != len(self.labels)
            or set(labels) != set(case_ids)
        ):
            raise ValueError("h_recipe_case_label_set")
        for case in self.cases:
            label = labels[case.case_id]
            for support in label.support:
                texts = [
                    document.content
                    for document in case.documents
                    if document.role == support.role
                ]
                if not all(
                    any(clause in body for body in texts) for clause in support.clauses
                ):
                    raise ValueError("h_truth_clause_absent")
            for role, required in label.required_sources.items():
                if len(set(required)) != len(required) or set(required) - {
                    document.source_id
                    for document in case.documents
                    if document.role == role
                }:
                    raise ValueError("h_truth_required_source_identity")
        return self

    def require_formal_quotas(self) -> None:
        if (
            len(self.cases) != 200
            or Counter(label.recommendation.value for label in self.labels) != QUOTAS
        ):
            raise ValueError("h_formal_class_quotas")
        if sum(label.high_risk for label in self.labels) != 60:
            raise ValueError("h_formal_high_risk_quota")
        if any(
            label.high_risk != (label.recommendation.value == "recommend_reject")
            for label in self.labels
            if label.recommendation.value != "needs_human_review"
        ):
            raise ValueError("h_formal_high_risk_class")
        if sum(bool(label.conflict_pairs) for label in self.labels) != 30:
            raise ValueError("h_formal_conflict_quota")
        if sum(bool(label.security_events) for label in self.labels) != 30:
            raise ValueError("h_formal_security_quota")
        if any(label.conflict_pairs and label.security_events for label in self.labels):
            raise ValueError("h_formal_events_overlap")


class _ObservationShape:
    """The planned observation grid every H plan version declares for itself."""

    case_ids: tuple[HCaseId, ...]
    repeat_seeds: tuple[int, ...]

    def arms(self) -> tuple[str, ...]:
        """Which arms this plan runs.  The historical grid runs both; a probe runs one."""

        return ARMS

    def observation_keys(self) -> tuple[tuple[str, int, str], ...]:
        return tuple(
            (case, seed, arm)
            for seed in self.repeat_seeds
            for case in self.case_ids
            for arm in self.arms()
        )

    def task_keys(self) -> tuple[tuple[str, int, str, str], ...]:
        return tuple(
            (case, seed, arm, branch)
            for case, seed, arm in self.observation_keys()
            for branch in (ROLES if arm == "multi_agent" else ("joint",))
        )

    def capacity(self) -> dict[str, int]:
        """The planned capacity of this exact shape, arm by arm.

        The multi Worker arm sends one request per model branch (two per observation); the
        single Agent arm sends one merged request.  Every logical task may spend at most two
        shared physical attempts, so the physical bound is twice the first-send count of
        the arms this plan declares.
        """

        observations = len(self.case_ids) * len(self.repeat_seeds)
        arms = self.arms()
        multi = observations if "multi_agent" in arms else 0
        single = observations if "single_agent" in arms else 0
        first_sends = multi * 2 + single
        return {
            "cases": len(self.case_ids),
            "repeats": len(self.repeat_seeds),
            "multi_observations": multi,
            "single_observations": single,
            "first_sends": first_sends,
            "physical_capacity": first_sends * 2,
            "max_per_task": 2,
        }


class HPlan(_ObservationShape, FrozenModel):
    schema_version: Literal["v1-07h-plan.v1"] = H_PLAN
    stage_id: Literal["V1-07H"] = "V1-07H"
    batch_id: Annotated[str, Field(pattern=r"^(disposable|formal)-h-[a-z0-9-]{3,64}$")]
    mode: Literal["disposable", "formal"]
    candidate_sha256: Sha
    consumer_sha256: Sha
    product_sha256: Sha
    generator_spec_sha256: Sha
    seed_sha256: Sha
    exclusion_sha256: Sha
    case_ids: tuple[HCaseId, ...] = Field(min_length=1, max_length=200)
    repeat_seeds: tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]]
    configuration: FEndpointConfig
    document_version: Name
    dataset_version: Name
    index_version: Name
    tenant_id: Name
    subject_id: Name
    authorization: Annotated[str, Field(min_length=1, max_length=240)]
    lifecycle_evidence_sha256: Sha

    @model_validator(mode="after")
    def plan_is_closed(self) -> Self:
        if not self.batch_id.startswith(self.mode + "-h-"):
            raise ValueError("h_mode_batch_identity")
        if (
            len(set(self.case_ids)) != len(self.case_ids)
            or len(set(self.repeat_seeds)) != 2
        ):
            raise ValueError("h_plan_duplicate_identity")
        if self.mode == "formal" and len(self.case_ids) != 200:
            raise ValueError("h_formal_200_cases_required")
        config = self.configuration
        if (
            config.temperature != 0.0
            or config.thinking != "disabled"
            or config.mode != "json_schema"
        ):
            raise ValueError("h_fair_model_parameters")
        if (
            config.requested_model != "deepseek-v4.1-flash"
            or config.resolved_model != "deepseek-v4-1-flash"
        ):
            raise ValueError("h_registered_model_identity")
        return self

    def stop_policy(self):
        """The stop policy of this plan's own version — the historical one registered none.

        The frozen V1-07 contract predates the dense-429 window and threshold, so running
        it with no window is what its version declares; the V1-08 versions register both
        parameters and answer with the policy they were admitted under.  The runner reads
        the policy through this one accessor, so a plan version can never be run under a
        different version's policy — or under none while it registered one.
        """

        return None


class HPlanV2(_ObservationShape, FrozenModel):
    """The V1-08 batch contract: the registered thinking, deadline and stop parameters.

    The frozen V1-07 plan keeps its own schema and stays readable; this version is where
    the new demonstration contract registers what the Spec freezes: a thinking value both
    arms execute with, one of the two accepted Run deadlines, and the dense-429 window and
    threshold.  All three are batch admission parameters — a plan that omits any of them,
    or states a value outside the accepted bounds, is refused instead of being run with an
    unstated policy.  Scale and quota checks for the formal demonstration batches belong to
    the stage that registers them; this contract only fixes the execution parameters.
    """

    schema_version: Literal["v1-08h-plan.v1"] = H_PLAN_V2
    stage_id: Literal["V1-08"] = "V1-08"
    batch_id: Annotated[str, Field(pattern=r"^(disposable|probe)-h-[a-z0-9-]{3,64}$")]
    mode: Literal["disposable", "probe"]
    candidate_sha256: Sha
    consumer_sha256: Sha
    product_sha256: Sha
    generator_spec_sha256: Sha
    seed_sha256: Sha
    exclusion_sha256: Sha
    case_ids: tuple[HCaseId, ...] = Field(min_length=1, max_length=200)
    repeat_seeds: tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]]
    configuration: FEndpointConfig
    generation_deadline_seconds: float
    rate_limit_window_seconds: Annotated[int, Field(ge=1, le=3_600)]
    rate_limit_burst_threshold: Annotated[int, Field(ge=2, le=64)]
    document_version: Name
    dataset_version: Name
    index_version: Name
    tenant_id: Name
    subject_id: Name
    authorization: Annotated[str, Field(min_length=1, max_length=240)]
    lifecycle_evidence_sha256: Sha

    @model_validator(mode="after")
    def plan_is_closed(self) -> Self:
        if not self.batch_id.startswith(self.mode + "-h-"):
            raise ValueError("h_mode_batch_identity")
        if (
            len(set(self.case_ids)) != len(self.case_ids)
            or len(set(self.repeat_seeds)) != 2
        ):
            raise ValueError("h_plan_duplicate_identity")
        if self.generation_deadline_seconds not in ACCEPTED_GENERATION_DEADLINES:
            raise ValueError("h_generation_deadline_not_accepted")
        config = self.configuration
        if (
            config.temperature != 0.0
            or config.thinking is None
            or config.mode != "json_schema"
        ):
            raise ValueError("h_fair_model_parameters")
        if (
            config.requested_model != "deepseek-v4.1-flash"
            or config.resolved_model != "deepseek-v4-1-flash"
        ):
            raise ValueError("h_registered_model_identity")
        return self

    def stop_policy(self):
        """The registered dense-429 parameters as the shared controller reads them.

        The pair is re-checked where it is assembled, so a plan that carries an
        out-of-range value is refused instead of being run with an unchecked window.
        """

        from .run_stop import StopPolicy

        policy = StopPolicy(
            rate_limit_window_seconds=self.rate_limit_window_seconds,
            rate_limit_burst_threshold=self.rate_limit_burst_threshold,
        )
        policy.check()
        return policy


class HProbePlan(_ObservationShape, FrozenModel):
    """The V1-08 C probe contract: the product arm alone, one observation per Case.

    The historical grid cannot stand in for the probe (Spec P-03): a two-arm, two-repeat
    batch measures a comparison and a stability question, not the 8–12 Case product send
    the capability question is answered from.  This version is therefore its own shape —
    8 to 12 already-seen DEV Cases, the single registered probe seed, the multi Worker arm
    only (its two model branches per Case), one initial observation each, the registered
    thinking candidate, the probe's own 180-second Run deadline and the dense-429
    window/threshold — so its arithmetic is exactly ``Cases × 2`` first sends and
    ``Cases × 4`` physical attempts at most (16–24 / 32–48), never the 48–72 first sends
    of a doubled grid.  It consumes no formal material, opens no label and claims no
    capability by being registered.
    """

    schema_version: Literal["v1-08c-probe.v1"] = H_PROBE
    stage_id: Literal["V1-08C"] = "V1-08C"
    batch_id: Annotated[str, Field(pattern=r"^probe-h-[a-z0-9-]{3,64}$")]
    mode: Literal["probe"] = "probe"
    candidate_sha256: Sha
    consumer_sha256: Sha
    product_sha256: Sha
    generator_spec_sha256: Sha
    seed_sha256: Sha
    exclusion_sha256: Sha
    case_ids: tuple[HCaseId, ...] = Field(min_length=8, max_length=12)
    repeat_seeds: tuple[Annotated[int, Field(gt=0)], ...] = Field(min_length=1)
    configuration: FEndpointConfig
    generation_deadline_seconds: float
    rate_limit_window_seconds: Annotated[int, Field(ge=1, le=3_600)]
    rate_limit_burst_threshold: Annotated[int, Field(ge=2, le=64)]
    document_version: Name
    dataset_version: Name
    index_version: Name
    tenant_id: Name
    subject_id: Name
    authorization: Annotated[str, Field(min_length=1, max_length=240)]
    lifecycle_evidence_sha256: Sha

    @model_validator(mode="after")
    def probe_is_closed(self) -> Self:
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("h_plan_duplicate_identity")
        if self.repeat_seeds != (C_PROBE_SEED,):
            # One initial observation per Case is the probe's own definition; a second
            # repeat is the historical grid's stability question, not a capability probe.
            raise ValueError("h_probe_single_registered_seed_required")
        if self.generation_deadline_seconds != PROBE_GENERATION_DEADLINE_SECONDS:
            raise ValueError("h_probe_deadline_required")
        config = self.configuration
        if (
            config.temperature != 0.0
            or config.thinking != "enabled"
            or config.mode != "json_schema"
        ):
            # The probe registers the thinking candidate itself; a non-thinking value is
            # the comparison P-03 refuses to add, and a non-zero temperature or a weaker
            # output mode would make the answer unusable as capability evidence.
            raise ValueError("h_probe_thinking_candidate_required")
        if (
            config.requested_model != "deepseek-v4.1-flash"
            or config.resolved_model != "deepseek-v4-1-flash"
        ):
            raise ValueError("h_registered_model_identity")
        return self

    def arms(self) -> tuple[str, ...]:
        """The probe runs the multi Worker product arm only (Spec P-03)."""

        return ("multi_agent",)

    def stop_policy(self):
        """The registered dense-429 parameters as the shared controller reads them.

        The pair is re-checked where it is assembled, so a probe that carries an
        out-of-range value is refused instead of being run with an unchecked window.
        """

        from .run_stop import StopPolicy

        policy = StopPolicy(
            rate_limit_window_seconds=self.rate_limit_window_seconds,
            rate_limit_burst_threshold=self.rate_limit_burst_threshold,
        )
        policy.check()
        return policy


HPlanLike = HPlan | HPlanV2 | HProbePlan


def load_h_plan(raw: bytes) -> HPlanLike:
    """Read a plan by the version it declares; an unknown version fails closed.

    The versions are validated independently, so a V1-07 plan is never judged by the
    V1-08 field set, a V1-08 field never appears inside an old batch's own reading, and
    the C probe shape never inherits the two-arm two-repeat grid's fields.
    """

    import json

    try:
        declared = json.loads(raw)
        schema = declared.get("schema_version") if type(declared) is dict else None
    except ValueError:
        schema = None
    if schema == H_PLAN:
        return HPlan.model_validate_json(raw, strict=True)
    if schema == H_PLAN_V2:
        return HPlanV2.model_validate_json(raw, strict=True)
    if schema == H_PROBE:
        return HProbePlan.model_validate_json(raw, strict=True)
    raise ValueError("h_plan_version_unknown")


def plan_generation_deadline(plan: HPlanLike) -> float:
    """The registered Run deadline; the historical schema keeps its frozen 30 seconds."""

    if isinstance(plan, HPlanV2 | HProbePlan):
        return plan.generation_deadline_seconds
    return V1_GENERATION_DEADLINE_SECONDS
