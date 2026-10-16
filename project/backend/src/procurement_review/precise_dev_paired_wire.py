"""The paired diagnostic's wire half: its projection contract and its driver.

The projection records the five validation layers of one attempt separately and the driver
grounds the accepted answer inside the attempt, so the batch's readings can say whether an
answer failed to parse, to satisfy the schema, to answer every authorized duty, to stay
self-consistent, or to support its citations — and a citation the source boundary refuses
can be spent on the branch's own remaining attempt under the shared two-attempt budget.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .business_v31_review import ground_document_v31, ground_document_v31_located
from .eval_f_transport import PreciseAttemptDriver
from .eval_v2_contracts import EvalV2Model
from .model_attempt_projection import ClaimDetail
from .precise_run_transport import precise_payload


class PairedProjection(EvalV2Model):
    """The ledger projection of one paired attempt: bounded, no answer text.

    This is the diagnostic's own projection contract.  It carries the branch's bounded
    conclusions and its own retrieval facts, and it deliberately does not reuse the
    frozen comparison projections: the fact source is the product's read-only MCP path,
    whose record identifiers are the product's stable synthetic ids, while the comparison
    contracts were written against the DEV material's own identifiers.

    It is kept only so a ledger written before the layered projection existed can still be
    read by its own contract; new attempts are recorded as :class:`PairedProjectionV2`.
    """

    schema_version: Literal["paired-projection.v1"] = "paired-projection.v1"
    projection_kind: Literal["paired_branch"] = "paired_branch"
    case_id: str
    worker: Literal["policy", "contract"]
    arm: Literal["oracle", "actual"]
    details: tuple[ClaimDetail, ...] = Field(default=(), max_length=12)
    first_pass_valid: bool
    received_nonempty_body: bool
    corrected_valid: bool
    pipeline_failure: bool
    model_request_count: int = Field(ge=1, le=2)
    latency_ms: int = Field(ge=0)
    envelope_version: str | None = None
    witnesses: int = Field(default=0, ge=0)
    duties: tuple[tuple[str, str], ...] = Field(default=(), max_length=12)
    format_fields: tuple[str, ...] = ()
    format_diagnostics: tuple[str, ...] = ()
    own_document_rank: int | None = None
    own_document_retrieved: bool | None = None


#: Which validation layer each terminal code of the line belongs to.  The layers are
#: reported separately so a wire-level refusal is never read as a semantic one and a
#: grounded refusal is never read as a successful answer.
_LAYERS = (
    "json_valid",
    "fields_valid",
    "obligations_complete",
    "semantic_valid",
    "citations_valid",
)
_CODE_LAYERS = {
    "invalid_json": (False, False, False, False, False),
    "invalid_schema": (True, False, False, False, False),
    "evidence_envelope_invalid": (True, False, False, False, False),
    "check_outside_branch": (True, True, False, False, False),
    "required_check_missing_or_duplicate": (True, True, False, False, False),
    "citation_outside_branch": (True, True, True, False, False),
    "duplicate_citation": (True, True, True, False, False),
}


class PairedProjectionV2(EvalV2Model):
    """One paired attempt under the same grounding and correction the product line uses.

    The five validation layers are recorded separately, so the diagnostic can say whether
    an answer failed to parse, to satisfy the schema, to answer every authorized duty, to
    stay semantically self-consistent, or to support its citations.  ``first_pass_valid``
    requires all five on the first attempt, ``corrected_valid`` requires all five after the
    one bounded correction — a later success never rewrites the first attempt's result —
    and ``committed`` marks the terminal ledger state, which is not by itself a statement
    that the answer qualified.
    """

    schema_version: Literal["paired-projection.v2"] = "paired-projection.v2"
    projection_kind: Literal["paired_branch_v2"] = "paired_branch_v2"
    case_id: str
    worker: Literal["policy", "contract"]
    arm: Literal["oracle", "actual"]
    details: tuple[ClaimDetail, ...] = Field(default=(), max_length=12)
    json_valid: bool
    fields_valid: bool
    obligations_complete: bool
    semantic_valid: bool
    citations_valid: bool
    first_pass_valid: bool
    corrected_valid: bool
    #: The branch produced a grounded, fully valid answer; a committed attempt without this
    #: is a terminal state that did not qualify.
    qualified: bool
    #: The ledger committed this attempt (terminal), whatever its quality.
    committed: bool
    received_nonempty_body: bool
    pipeline_failure: bool
    model_request_count: int = Field(ge=1, le=2)
    latency_ms: int = Field(ge=0)
    envelope_version: str | None = None
    witnesses: int = Field(default=0, ge=0)
    duties: tuple[tuple[str, str], ...] = Field(default=(), max_length=12)
    format_fields: tuple[str, ...] = ()
    format_diagnostics: tuple[str, ...] = ()
    #: The bounded locations of the final attempt's grounding refusals: registered category,
    #: obligation and declared field.  Never a value or an answer text.
    grounding_failures: tuple[str, ...] = ()
    grounding_locations: tuple[str, ...] = ()
    correction_kinds: tuple[str, ...] = ()
    own_document_rank: int | None = None
    own_document_retrieved: bool | None = None

    @property
    def layers(self) -> tuple[bool, ...]:
        return tuple(getattr(self, name) for name in _LAYERS)


class PairedDriver(PreciseAttemptDriver):
    """The same attempt machinery, sending the versioned 3.1.0 wire per arm.

    It grounds the accepted answer inside the attempt, so a citation the source boundary
    refuses can be corrected by the branch's own remaining attempt under the same shared
    budget the product uses, instead of being recorded as a successful answer or as an
    unlocated failure.
    """

    stage_id = "V1-07G"
    projection_type = PairedProjectionV2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: In-memory only: the request, its catalog and the grounded document of each
        #: successful attempt, so the pair can be finalized without persisting an answer
        #: body anywhere.
        self.requests: dict[str, Any] = {}
        self.catalogs: dict[str, Any] = {}
        self.grounded: dict[str, Any] = {}
        self._last_grounded: dict[str, Any] = {}
        #: The rank the Case's own document was retrieved at in one arm's own sources:
        #: positive when it was returned, ``-1`` when it was not, absent for the Oracle.
        self.ranks: dict[str, int] = {}
        self.kinds: dict[str, list[str]] = {}
        self.refusals: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        self._pending_kind = "contract"
        self._pending_details: tuple[str, ...] = ()

    def register(self, task_key: str, request: Any, catalog: Any) -> None:
        self.requests[task_key] = request
        self.catalogs[task_key] = catalog

    def note_rank(self, task_key: str, rank: int | None) -> None:
        """Record where the Case's own document ranked in this arm's own sources."""

        self.ranks[task_key] = rank

    def refusal_of(self, observed, request, task_key):
        """Ground the accepted answer and report the first citation refusal, if any.

        The refusal carries the fixed category, the duty and the declared field of what was
        refused; the values themselves never leave the boundary, so the one correction can
        name what to redo without ever restating the answer.
        """

        parsed, witnesses = observed.evidence
        document, locations = ground_document_v31_located(
            request, parsed, witnesses, self.catalogs[task_key]
        )
        # Held aside: only the attempt that produced the final projection publishes its
        # document, so a later attempt that failed at the wire cannot leave an earlier
        # one's document behind for the pair to use.
        self._last_grounded[task_key] = document
        self.refusals[task_key] = (document.support_failures, locations)
        if document.support_failures:
            self._pending_kind = "citation"
            self._pending_details = locations
            return document.support_failures[0], locations
        return None

    def on_correction(self, task_key, correction, correction_fields, refusal):
        self.kinds.setdefault(task_key, []).append(
            "citation" if refusal is not None else "contract"
        )

    def make_payload(self, request, correction, correction_fields):
        """The next request, with the refused fields named when one correction is left."""

        if not correction:
            self._pending_details = ()
            self._pending_kind = "contract"
            return precise_payload(self.transport.config, request)
        refused = tuple(
            token.removeprefix("field=")
            for token in self._pending_details
            if token.startswith("field=")
        )
        return precise_payload(
            self.transport.config,
            request,
            correction,
            refused or correction_fields,
            self._pending_details,
            self._pending_kind,
        )

    def _layers_for(self, observed, success: bool, task_key: str) -> dict[str, bool]:
        code = None if success else observed.error_code
        if success:
            grounded = self.grounded.get(task_key)
            citations = bool(grounded is not None and not grounded.support_failures)
            return dict.fromkeys(_LAYERS, True) | {"citations_valid": citations}
        flags = _CODE_LAYERS.get(str(code), (False,) * 5)
        return dict(zip(_LAYERS, flags))

    def make_projection(self, observed, success, *, task_key, **fields):
        case_id, _configuration, arm, _worker = task_key.split(".")
        version = None
        witnesses = 0
        duties: tuple[tuple[str, str], ...] = ()
        if success and observed.evidence is not None:
            document, witness_rows = observed.evidence
            version = "finding-evidence.v3"
            witnesses = len(witness_rows)
            duties = tuple(
                (item.finding.obligation_id, item.finding.status)
                for item in document.findings
            )
            grounded = self.grounded.get(task_key)
            if grounded is None:
                grounded = ground_document_v31(
                    self.requests[task_key],
                    document,
                    witness_rows,
                    self.catalogs[task_key],
                )
            self.grounded[task_key] = grounded
        if task_key in self._last_grounded:
            self.grounded[task_key] = self._last_grounded.pop(task_key)
        layers = self._layers_for(observed, success, task_key)
        all_valid = all(layers.values())
        correction_used = fields["model_request_count"] > 1
        failures, locations = self.refusals.get(task_key, ((), ()))
        rank = self.ranks.get(task_key)
        projection = PairedProjectionV2(
            case_id=case_id,
            worker=fields["worker"],
            arm=arm,
            details=tuple(fields.get("details") or ()),
            **layers,
            first_pass_valid=all_valid and not correction_used,
            corrected_valid=all_valid and correction_used,
            qualified=all_valid,
            committed=success,
            received_nonempty_body=fields["received_nonempty_body"],
            pipeline_failure=fields["pipeline_failure"],
            model_request_count=fields["model_request_count"],
            latency_ms=fields["latency_ms"],
            envelope_version=version,
            witnesses=witnesses,
            duties=duties,
            format_fields=tuple(getattr(observed, "format_fields", ()) or ()),
            format_diagnostics=tuple(getattr(observed, "format_diagnostics", ()) or ()),
            grounding_failures=tuple(failures),
            grounding_locations=tuple(locations),
            correction_kinds=tuple(self.kinds.get(task_key, ())),
            own_document_rank=rank,
            own_document_retrieved=(None if rank is None else rank > 0),
        )
        return PairedProjectionV2.model_validate_json(projection.model_dump_json())
