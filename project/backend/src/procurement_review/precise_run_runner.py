"""The precise product Run: scope-bound retrieval, two gated model branches, one
deterministic branch, and the versioned execution record the product stores.

The summary projection lives in :mod:`.precise_run_summary` and the assembled inputs in
:mod:`.precise_run_scope`; this module owns the execution itself — the gated attempts,
grounding, the durable branch commits, the stop conditions and the record.

Every model attempt is admitted the product way: the attempt row is opened before anything
is read, the request is built and received inside one short transaction that re-reads the
live authority, the response is admitted *before* it is decoded, and the branch result is
committed atomically with its attempt.  A branch that already has a committed projection is
reused instead of being sent again, the two branches share one persistent per-worker
attempt quota, and the whole execution runs under the Run's frozen snapshot: a runtime that
no longer matches the mode, model identity, parameters, digests, versions, adapters, budget
or deadline the Run was created with stops with the dimension that moved.  The procurement
branch performs the fixed read-only MCP sequence and makes no model call at all.

A failure that ends the generation's sending — authentication, configuration, model
identity drift, an unauthorized citation or a security boundary — travels as an explicit
stop path, never as an incidental exception, and the failed branch keeps its own code.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from .business_v3_evidence import GroundedDocumentV3
from .business_v3_procurement import review_procurement_v3
from .business_v3_security import ProtectedOutputV3
from .business_v31_review import (
    finalize_review_v31,
    ground_document_v31_located,
)
from .checkpointing import (
    BranchTelemetry,
    IncompatibleCheckpointError,
    ReadAdmission,
)
from .model_invocation import RunModelInvocation, await_call
from .procurement_mcp_client import OfficialProcurementMCPClient
from .precise_branch_records import (
    STOP_ERROR_CODES,
    BranchOutcome,
    adopted_outcome,
    attempt_record,
    branch_record,
    failed_document,
    guard_failure_code,
    pipeline_failures,
    resume_refused_commit,
    skipped_record,
)
from .precise_run_contracts import (
    PRECISE_RUN_CONTRACT,
    DeterministicProcurementBranchV1,
    PreciseBranchOutcome,
    PreciseDocumentBranchV1,
    PreciseModelAttemptV1,
    PreciseRunExecutionV1,
    PreciseRunSnapshotV1,
)
from .precise_run_scope import (
    PRECISE_QUESTIONS,
    BranchInputs,
    branch_inputs,
    execution_snapshot,
    procurement_worker_input,
    require_frozen_snapshot,
    wire_request,
)
from .precise_run_state import (
    adopted_branch_commit,
    attempt_ledger_rows,
    attempt_receipt_draft,
    attempt_receipts,
    commit_from_branch,
    commit_precise_branch,
    finish_attempt_with_receipt,
    metering_from_ledger,
    precise_contract_identity,
)
from .precise_run_summary import (
    business_context,
    evidence_map,
    sha256_of_text,
    summary_from_precise,
)
from .precise_run_transport import PreciseRunChatTransport, precise_payload
from .result_admission import RunResultAdmission
from .review_repository import ReviewRepository
from .run_errors import RecordContractError
from .run_stop import MAX_RETRY_BACKOFF_SECONDS, BatchStopController
from .scoped_runtime import ScopedReviewRuntime
from .worker_contracts import ModelCallHandle, WorkerModelRequest
from .mcp_server import create_procurement_mcp_server

MAX_MODEL_ATTEMPTS_PER_BRANCH = 2
DEFAULT_BRANCH_TIMEOUT_SECONDS = 90.0


@dataclass(slots=True)
class _BranchPlan:
    """The one send one branch attempt is allowed to perform."""

    worker: str
    case_id: str
    scope_id: str
    request: Any
    correction: str | None = None
    correction_fields: tuple[str, ...] = ()
    correction_details: tuple[str, ...] = ()
    correction_kind: str = "contract"
    seconds: float = DEFAULT_BRANCH_TIMEOUT_SECONDS
    consume: Callable[[], None] | None = None


@dataclass(slots=True)
class _AttemptHandle:
    """The handle of one already-received call; only its result is asynchronous."""

    task: asyncio.Task

    async def result(self) -> Any:
        return await self.task


class PreciseSyncReceiver:
    """The synchronous receive half of the precise wire.

    ``start`` never waits: it checks that the receipt really is a product
    ``WorkerModelRequest`` for the planned branch and hands the send to the loop.  A receipt
    that does not match the plan refuses the call instead of sending something unplanned,
    and the send itself happens outside every transaction that admitted it.
    """

    def __init__(
        self, *, transport: PreciseRunChatTransport, plans: dict[str, _BranchPlan]
    ) -> None:
        self._transport = transport
        self._plans = plans
        self.receipts = 0

    def start(self, request: Any) -> ModelCallHandle:
        if type(request) is not WorkerModelRequest:
            raise TypeError(
                "the precise receiver accepts only a product WorkerModelRequest receipt"
            )
        plan = self._plans.get(request.worker.value)
        if plan is None:
            raise TypeError("the received worker is not part of this execution plan")
        if request.case_id != plan.case_id:
            raise TypeError("the received case does not match the planned branch")
        if len(request.evidence_contexts) != len(plan.request.sources):
            raise TypeError("the received context does not match the planned branch")
        self.receipts += 1
        return _AttemptHandle(asyncio.ensure_future(self._send(plan)))

    async def _send(self, plan: _BranchPlan):
        payload = precise_payload(
            self._transport.config,
            plan.request,
            plan.correction,
            plan.correction_fields,
            plan.correction_details,
            plan.correction_kind,
        )
        return await self._transport.send_admitted(
            payload, plan.request, plan.seconds, admit=plan.consume
        )


class PreciseScopedRunner:
    """Assemble one precise, scope-bound review execution for the service."""

    def __init__(
        self,
        *,
        store: Any,
        transport: PreciseRunChatTransport,
        branch_timeout_seconds: float = DEFAULT_BRANCH_TIMEOUT_SECONDS,
        max_attempts_per_branch: int = MAX_MODEL_ATTEMPTS_PER_BRANCH,
        generation_deadline_seconds: float = 30.0,
        mode: Literal["test_only", "real"] = "test_only",
        telemetry: BranchTelemetry | None = None,
        stop: BatchStopController | None = None,
    ) -> None:
        self._store = store
        self._transport = transport
        self._branch_timeout_seconds = branch_timeout_seconds
        self._max_attempts = max_attempts_per_branch
        self._generation_deadline_seconds = generation_deadline_seconds
        self._mode = mode
        self._telemetry = telemetry or BranchTelemetry()
        #: The optional batch stop gate: present only when an evaluation batch runs this
        #: execution.  Every physical attempt is admitted through it, so a batch that has
        #: already stopped cannot start a new send from a Run that was mid-flight.
        self._stop = stop
        self.branch_telemetry = self._telemetry

    def execution_snapshot(self, registration: Any) -> dict:
        """The execution configuration a Run created now would be frozen with."""

        return self._snapshot(registration).model_dump(mode="json")

    def _snapshot(self, registration: Any) -> PreciseRunSnapshotV1:
        return execution_snapshot(
            mode=self._mode,
            config=self._transport.config,
            registration=registration,
            branch_timeout_seconds=self._branch_timeout_seconds,
            max_attempts_per_branch=self._max_attempts,
            generation_deadline_seconds=self._generation_deadline_seconds,
        )

    def require_frozen_snapshot(
        self, run: Any, registration: Any
    ) -> PreciseRunSnapshotV1:
        return require_frozen_snapshot(run, self._snapshot(registration))

    def remaining_attempts(
        self, repository: ReviewRepository, run_id: str, worker: str
    ) -> int:
        """The quota the next execution of this Run would still have for one branch.

        The counter is the persisted product ledger, so a new runner instance, a resumed
        generation and a fresh process all read the same number instead of starting over.
        """

        used = int(repository.attempt_totals(run_id).get(worker, 0))
        return max(0, self._max_attempts - used)

    async def run(
        self,
        *,
        run_id: str,
        scope: Any,
        registration: Any,
        generation: int,
        repository: ReviewRepository,
        deadline_at: datetime,
        fault: Any = None,
        heartbeat: Callable[[], bool] | None = None,
        owner_id: str | None = None,
        now: Callable[[], datetime],
    ) -> PreciseRunExecutionV1:
        del heartbeat
        run = repository.load_run(run_id)
        if run is None:
            raise IncompatibleCheckpointError("the Run to execute no longer exists")
        frozen = self.require_frozen_snapshot(run, registration)
        admission = ReadAdmission(
            repository=repository,
            run_id=run_id,
            generation=generation,
            now=now,
            owner_id=owner_id,
            telemetry=self._telemetry,
            fault=fault,
        )
        consumption = RunResultAdmission(
            repository=repository,
            run_id=run_id,
            generation=generation,
            now=now,
            owner_id=owner_id,
            telemetry=self._telemetry,
            fault=fault,
        )
        runtime = ScopedReviewRuntime.build(
            store=self._store,
            scope=scope,
            admission=admission,
            vector_mode="none",
            lexical_adapter="idf",
        )
        from mcp import Client

        server = create_procurement_mcp_server(runtime.procurement, scope=scope)
        async with Client(server, raise_exceptions=True) as client:
            procurement_read = await OfficialProcurementMCPClient(
                client
            ).collect_review_facts(
                procurement_worker_input(registration, registration.case_id)
            )
        context = business_context(scope, procurement_read)
        protection = ProtectedOutputV3()
        inputs = branch_inputs(
            context=context,
            case_id=registration.case_id,
            store=self._store,
            scope=scope,
            admission=admission,
            runtime=runtime,
        )
        plans = {
            worker: _BranchPlan(
                worker=worker,
                case_id=registration.case_id,
                scope_id=context.scope_id,
                request=inputs[worker].request,
                consume=lambda worker=worker: consumption.admit_result(
                    site=f"precise.{worker}.answer"
                ),
            )
            for worker in inputs
        }
        invocation = RunModelInvocation(
            repository=repository,
            run_id=run_id,
            generation=generation,
            now=now,
            owner_id=owner_id,
            receiver=PreciseSyncReceiver(transport=self._transport, plans=plans),
            telemetry=self._telemetry,
            fault=fault,
        )
        grounded: dict[str, GroundedDocumentV3] = {}
        records: dict[str, PreciseDocumentBranchV1] = {}
        inspections: dict[str, tuple] = {}
        stop_reason: str | None = None
        reused = 0
        for worker in ("policy", "contract"):
            branch = inputs[worker]
            if stop_reason is not None:
                records[worker] = skipped_record(worker=worker, branch=branch)
                grounded[worker] = failed_document(worker)
                inspections[worker] = ()
                continue
            outcome = await self._run_branch(
                branch=branch,
                plan=plans[worker],
                invocation=invocation,
                repository=repository,
                run_id=run_id,
                generation=generation,
                owner_id=owner_id,
                now=now,
                scope=scope,
                snapshot=frozen,
                deadline_at=deadline_at,
                protection=protection,
            )
            records[worker] = outcome.record
            grounded[worker] = outcome.document
            inspections[worker] = outcome.security
            reused += 1 if outcome.reused else 0
            stop_reason = stop_reason or outcome.stop_reason
        procurement = review_procurement_v3(
            context, procurement_read, authorized_scope_id=context.scope_id
        )
        decision = finalize_review_v31(
            context,
            grounded["policy"],
            grounded["contract"],
            procurement,
            sources=(*inputs["policy"].slices, *inputs["contract"].slices),
            protection=protection,
            pipeline_failures=pipeline_failures(records, procurement_read, stop_reason),
            control_events=(),
            branch_inspections=inspections,
        )
        return PreciseRunExecutionV1(
            run_id=run_id,
            case_id=registration.case_id,
            decision=decision.business,
            security=decision.security,
            conflicts=decision.conflicts,
            policy=records["policy"],
            contract=records["contract"],
            procurement=DeterministicProcurementBranchV1(
                status=procurement_read.status.value,
                dataset_version=procurement_read.requested_dataset_version,
                tool_calls=len(procurement_read.facts),
                record_count=len(procurement_read.facts),
                detail=procurement_read.detail,
            ),
            stop_reason=stop_reason,
            metering=metering_from_ledger(
                attempt_ledger_rows(repository, run_id),
                receipts=attempt_receipts(repository, run_id),
                requested_model=self._transport.config.requested_model,
                resolved_models=[
                    attempt.resolved_model
                    for record in records.values()
                    for attempt in record.attempts
                    if attempt.resolved_model is not None
                ],
                reused_branches=reused,
                explicit_recoveries=max(0, generation - 1),
            ),
            summary=summary_from_precise(
                run_id=run_id,
                case_id=registration.case_id,
                decision=decision.business,
                security=decision.security,
                conflicts=decision.conflicts,
                evidence_map=evidence_map(
                    {worker: inputs[worker].hits for worker in inputs},
                    procurement_read.facts,
                ),
                procurement_facts=procurement_read.facts,
            ),
        )

    async def _run_branch(
        self,
        *,
        branch: BranchInputs,
        plan: _BranchPlan,
        invocation: RunModelInvocation,
        repository: ReviewRepository,
        run_id: str,
        generation: int,
        owner_id: str | None,
        now: Callable[[], datetime],
        scope: Any,
        snapshot: PreciseRunSnapshotV1,
        deadline_at: datetime,
        protection: ProtectedOutputV3,
    ) -> BranchOutcome:
        worker = branch.worker
        request = branch.request
        request_sha = sha256_of_text(request.model_dump_json())
        committed = adopted_branch_commit(repository, run_id, worker)
        if committed is not None:
            if committed.request_sha256 != request_sha or committed.worker != worker:
                raise IncompatibleCheckpointError(
                    "the committed branch does not match this execution's inputs"
                )
            self._telemetry.note_reuse(worker)
            return adopted_outcome(committed, request, request_sha)
        attempts: list[PreciseModelAttemptV1] = []
        document: GroundedDocumentV3 | None = None
        security: tuple[Any, ...] = ()
        blocked = False
        outcome = PreciseBranchOutcome.NO_ATTEMPTS_LEFT
        failure_code: str | None = None
        answer_sha: str | None = None
        stop_reason: str | None = None
        used = int(repository.attempt_totals(run_id).get(worker, 0))
        remaining = self.remaining_attempts(repository, run_id, worker)
        for index in range(remaining):
            if self._stop is not None:
                # The batch gate runs before the attempt row is opened: a stopped batch
                # never starts a new physical send, and the rows already opened keep
                # their own settlement.
                self._stop.gate()
            attempt_no = used + index + 1
            repository.begin_attempt(
                run_id=run_id,
                generation=generation,
                worker=worker,
                attempt_no=attempt_no,
                now=now,
                owner_id=owner_id,
            )
            seconds = min(
                self._branch_timeout_seconds,
                max(0.0, (min(deadline_at, scope.deadline_at) - now()).total_seconds()),
            )
            if seconds <= 0.0:
                failure_code = "deadline_exhausted"
                repository.finish_attempt(
                    run_id=run_id,
                    generation=generation,
                    worker=worker,
                    attempt_no=attempt_no,
                    outcome="failed",
                    error_code=failure_code,
                    now=now(),
                )
                outcome = PreciseBranchOutcome.FAILED
                break
            plan.seconds = seconds
            started = datetime.now(UTC)
            call = invocation.enter(
                site=f"precise.{worker}",
                request_factory=lambda: wire_request(
                    worker=worker, request=request, hits=branch.hits
                ),
            )
            observed = await await_call(call)
            latency_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)
            # A refused output is this branch's own security fact: it carries the fixed
            # category and a location identity, so the decision can escalate it to a human
            # path instead of reading the refusal as an ordinary pipeline failure.
            security = (
                *security,
                *tuple(getattr(observed, "security_events", ()) or ()),
            )
            corrected_document: GroundedDocumentV3 | None = None
            citation_locations: tuple[str, ...] = ()
            if observed.error_code is None:
                # Grounding runs before the attempt is settled, so a citation the source
                # boundary refused can still be corrected by the second attempt instead
                # of being committed as a degraded branch.
                parsed, witnesses = observed.evidence
                corrected_document, citation_locations = ground_document_v31_located(
                    request, parsed, witnesses, branch.catalog
                )
            refusals = (
                tuple(corrected_document.support_failures[:1])
                if observed.error_code is None and corrected_document is not None
                else ()
            )
            error_code = observed.error_code or (refusals[0] if refusals else None)
            if self._stop is not None and error_code is not None:
                self._stop.note_attempt(error_code, now())
            can_retry = index + 1 < remaining
            correctable = can_retry and (bool(observed.correctable) or bool(refusals))
            citation_refused = can_retry and bool(refusals)
            # A known recoverable transport failure is retried inside the same shared
            # two-attempt budget when the endpoint asked for a bounded delay; a delay
            # beyond the accepted backoff keeps the failure as the branch's own outcome
            # instead of waiting an unbounded time.  The next loop iteration re-reads the
            # remaining generation deadline, so the backoff spends the real budget.
            retry_after = float(getattr(observed, "retry_after", 0.0) or 0.0)
            transport_retry = (
                can_retry
                and not correctable
                and bool(observed.retryable)
                and 0.0 <= retry_after <= MAX_RETRY_BACKOFF_SECONDS
            )
            # Which correction the *next* attempt of this branch will be sent for.  It is
            # decided here, before the attempt is settled, so the Run's own metering can
            # tell a citation correction from a contract correction and a transport retry
            # instead of pooling every extra attempt as a format correction.
            next_correction_kind = (
                "citation"
                if citation_refused
                else "contract"
                if can_retry and error_code is not None and bool(observed.correctable)
                else "transport"
                if transport_retry
                else None
            )
            attempts.append(
                attempt_record(
                    attempt_no,
                    observed,
                    latency_ms,
                    correctable=correctable,
                    error_code=error_code,
                )
            )
            finish_attempt_with_receipt(
                repository,
                run_id=run_id,
                generation=generation,
                worker=worker,
                attempt_no=attempt_no,
                outcome=(
                    "completed"
                    if error_code is None
                    else "format_corrected"
                    if correctable
                    else "failed"
                ),
                error_code=error_code,
                now=now(),
                receipt=attempt_receipt_draft(
                    worker=worker,
                    generation=generation,
                    attempt_no=attempt_no,
                    request_sha256=request_sha,
                    observed=observed,
                    latency_ms=latency_ms,
                    next_correction_kind=next_correction_kind,
                ),
            )
            if observed.error_code is None:
                # The answer was admitted before it was decoded; grounding and the commit
                # both run under the live generation, and the commit is atomic with the
                # attempt that produced it.
                content = observed.evidence_content or ""
                answer_sha = sha256_of_text(content)
                document = corrected_document
                if citation_refused:
                    # One bounded correction names the refused fields and the fixed
                    # category, never a value or the model's own text.
                    plan.correction = refusals[0]
                    plan.correction_kind = "citation"
                    plan.correction_fields = tuple(
                        token.removeprefix("field=")
                        for token in citation_locations
                        if token.startswith("field=")
                    )
                    plan.correction_details = citation_locations
                    self._telemetry.retries += 1
                    continue
                if refusals and not document.document.values:
                    # No attempt is left and nothing of the document survived its source
                    # check: the branch fails with its own code instead of being committed
                    # as a clean success.
                    failure_code = refusals[0]
                    outcome = PreciseBranchOutcome.FAILED
                    break
                if refusals:
                    # A refusal costs the duty it belongs to and nothing else: the duties
                    # this branch did establish are committed with the branch's own support
                    # failures recorded, and the refused ones stay unresolved instead of
                    # taking the whole branch's evidence down with them.
                    failure_code = refusals[0]
                inspected, blocked = _inspect_branch(document, protection)
                security = (*security, *inspected)
                record = branch_record(
                    worker=worker,
                    document=document,
                    request_sha=request_sha,
                    answer_sha=answer_sha,
                    attempts=tuple(attempts),
                    hits=branch.hits,
                )
                try:
                    commit = commit_precise_branch(
                        repository,
                        run_id=run_id,
                        generation=generation,
                        worker=worker,
                        attempt_no=attempt_no,
                        input_checksum="sha256:" + request_sha,
                        contract_checksum=precise_contract_identity(snapshot, worker),
                        payload=commit_from_branch(
                            worker=worker,
                            branch=record,
                            document=document,
                            security=security,
                            blocked=blocked,
                        ),
                        now=now,
                        owner_id=owner_id,
                    )
                except RecordContractError as error:
                    # The stored-record guard refused this branch's own payload: the
                    # answer carries something the product may not persist.  The branch
                    # ends in a controlled state that names the guard's own category.
                    failure_code = guard_failure_code(error.categories)
                    outcome = PreciseBranchOutcome.FAILED
                    break
                if not commit.adopted:
                    self._telemetry.late_discarded += 1
                    self._telemetry.discarded_late_results += 1
                    return resume_refused_commit(
                        commit,
                        repository=repository,
                        run_id=run_id,
                        worker=worker,
                        request=request,
                        request_sha=request_sha,
                    )
                outcome = PreciseBranchOutcome.COMPLETED
                break
            failure_code = observed.error_code
            if correctable:
                plan.correction = observed.error_code
                plan.correction_kind = "contract"
                plan.correction_fields = tuple(observed.format_fields)
                plan.correction_details = tuple(observed.format_diagnostics)
                self._telemetry.retries += 1
                continue
            if transport_retry:
                self._telemetry.retries += 1
                await _spend_backoff(retry_after, deadline_at, scope, now)
                continue
            outcome = PreciseBranchOutcome.FAILED
            if observed.error_code in STOP_ERROR_CODES:
                stop_reason = observed.error_code
            break
        return BranchOutcome(
            record=branch_record(
                worker=worker,
                document=document,
                request_sha=request_sha,
                answer_sha=answer_sha,
                attempts=tuple(attempts),
                hits=branch.hits,
                outcome=outcome,
                failure_code=failure_code,
            ),
            document=document if document is not None else failed_document(worker),
            security=tuple(
                sorted(set(security), key=lambda e: (e.code, e.location_sha256))
            ),
            blocked=blocked,
            stop_reason=stop_reason,
        )


def _inspect_branch(
    document: GroundedDocumentV3, protection: ProtectedOutputV3
) -> tuple[tuple[Any, ...], bool]:
    """The bounded security projection of one branch's own output."""

    events = tuple(
        protection.inspect(
            [f.model_dump(mode="json") for f in document.document.findings],
            location=document.document.worker,
        )
    )
    return events, any(event.escalates for event in events)


async def _spend_backoff(
    delay: float, deadline_at: datetime, scope: Any, now: Callable[[], datetime]
) -> None:
    """Wait one bounded retry delay that comes out of the remaining real deadline.

    The wait never extends the Run: the next attempt recomputes its own seconds from the
    same deadline, so a backoff that reaches the deadline turns into a controlled
    ``deadline_exhausted`` failure instead of a silent extension.
    """

    if delay <= 0.0:
        return
    remaining = max(0.0, (min(deadline_at, scope.deadline_at) - now()).total_seconds())
    await asyncio.sleep(min(delay, remaining))


__all__ = [
    "DEFAULT_BRANCH_TIMEOUT_SECONDS",
    "MAX_MODEL_ATTEMPTS_PER_BRANCH",
    "PRECISE_QUESTIONS",
    "PRECISE_RUN_CONTRACT",
    "PreciseScopedRunner",
    "PreciseSyncReceiver",
]
