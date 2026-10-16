"""Checkpoint-aware Worker execution with an attempt ledger and bounded retries.

The graph, the fan-in and the rule semantics are unchanged: this module only wraps the
three Worker ports so that (a) an already-adopted branch result is reused without any
read call, (b) every attempt start is committed before the read-only port is invoked,
and (c) a transient read failure is retried at most once inside the fixed budget.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from .fault_hooks import FaultController
from .record_guard import validate_stored_model
from .review_repository import (
    ExecutionInvalidated,
    ReviewRepository,
    RunNotFoundError,
    TraceDraft,
)
from .run_errors import RecordUnavailableError
from .security_contracts import RuntimeScope
from .trace import canonical_checksum
from .worker_contracts import (
    WorkerBranchResult,
    WorkerFailureCode,
    WorkerOutcome,
)

CHECKPOINT_CONTRACT_VERSION = "review-checkpoint.v1"
ATTEMPT_BUDGET_PER_WORKER = 3
GENERATION_RETRY_BUDGET = 1
RETRY_BACKOFF_SECONDS = 0.25
ATTEMPT_TIMEOUT_SECONDS = 5.0
GENERATION_DEADLINE_SECONDS = 30.0
BRANCH_TIMEOUT_SECONDS = 12.0

TRANSIENT_FAILURE_CODES = frozenset(
    {
        WorkerFailureCode.TIMEOUT,
    }
)
TRANSIENT_FAILURE_DETAILS = frozenset(
    {
        "worker_branch_timeout",
        "mcp_transient_unavailable",
        "retrieval_transient_unavailable",
        "port_transient_unavailable",
        "transient_read_failure",
    }
)
EXHAUSTED_FAILURE_DETAILS = frozenset(
    {
        "worker_attempt_budget_exhausted",
        "worker_generation_deadline_exceeded",
    }
)


_execution_worker: contextvars.ContextVar[str] = contextvars.ContextVar(
    "execution_worker", default="unattributed"
)


def current_execution_worker() -> str:
    """The Worker whose read is being admitted.

    Each branch runs inside its own asyncio task, so this label cannot leak from one
    Worker's read site into another's.
    """

    return _execution_worker.get()


@dataclass(frozen=True, slots=True)
class ReadAdmission:
    """Gate one actual read on the live authority of the generation performing it.

    A booked attempt or an earlier check is not evidence that a read was allowed, so the
    read implementations call this at the last point before they touch data.  An admitted
    read is an already-issued read-only request: it may still complete, because a read
    cannot be physically cancelled, and its late result is kept out of the Run by the guard
    inside the commit transaction.

    The gate serves both kinds of read.  A port call reads an index that was built earlier,
    so it is called with no connection: the admission owns its own short transaction and
    closes it before the port runs, which keeps every network, model and port call outside
    a database row lock.  A *preload* reads the database itself, so the store hands it the
    connection of the transaction the read will use — the check and the read then share one
    short transaction, and a revocation either lands first or waits for it.
    """

    repository: ReviewRepository
    run_id: str
    generation: int
    now: Callable[[], datetime]
    telemetry: BranchTelemetry
    fault: FaultController | None = None
    owner_id: str | None = None

    def __call__(
        self, connection: Any | None = None, *, site: str | None = None
    ) -> None:
        label = site or current_execution_worker()
        try:
            if connection is None:
                self.repository.assert_generation_live(
                    run_id=self.run_id,
                    generation=self.generation,
                    now=self.now,
                    owner_id=self.owner_id,
                    site=label,
                )
            else:
                self.repository.require_generation_live(
                    connection,
                    run_id=self.run_id,
                    generation=self.generation,
                    now=self.now,
                    owner_id=self.owner_id,
                )
            if connection is None:
                self.telemetry.note_read_entered(label)
                if self.fault is not None:
                    # The next statement in the caller is the data access: this is the window a
                    # revocation can overlap while the read is genuinely in flight.
                    self.fault.barrier(f"during_port_read:{label}")
                self.repository.assert_generation_live(
                    run_id=self.run_id,
                    generation=self.generation,
                    owner_id=self.owner_id,
                    now=self.now,
                    site=label,
                )
                return
            self.telemetry.note_preload_admitted(label)
            if self.fault is not None:
                # Parked inside the read's own transaction, with the authority share locks still
                # held: a revocation that overlaps a preload has to wait for it.  The next
                # statement is the protected SELECT of this same transaction.
                self.fault.barrier(f"during_guarded_read:{label}")
            self.repository._guards.require_execution_time(
                connection,
                run_id=self.run_id,
                generation=self.generation,
                now=self.now,
            )

        except ExecutionInvalidated:
            self.telemetry.note_read_denied(label)
            raise
        except SQLAlchemyError as error:
            # The authorization store is unreachable. The document Workers run their port
            # call inside asyncio.to_thread and turn any Exception into one fail-closed
            # branch result, which would let a Run summarize on a cache it can no longer
            # authorize; this signal escapes that boundary and stops the whole execution.
            raise StorageUnavailableSignal(type(error).__name__) from error


class StorageUnavailableSignal(BaseException):
    """The review store itself failed.

    This deliberately inherits from :class:`BaseException`: a storage outage must reach
    the Run-level handler and become an interruption, and must never be degraded into one
    Worker's fail-closed branch result by the graph's ``except Exception`` boundary.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class IncompatibleCheckpointError(BaseException):
    """Raised when a committed checkpoint cannot be reused under this contract.

    Like :class:`StorageUnavailableSignal` this must escape the graph's per-branch
    ``except Exception`` boundary: an untrusted checkpoint stops the whole Run with a
    recorded reason instead of degrading into one Worker's failed branch result.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(slots=True)
class BranchTelemetry:
    """Exact per-run call accounting; it is reported as Evidence."""

    calls: int = 0
    reused: int = 0
    retries: int = 0
    exhausted: int = 0
    timeouts: int = 0
    late_discarded: int = 0
    discarded_late_results: int = 0
    repeated_reads: int = 0
    denied_calls: int = 0
    reads_entered: dict[str, int] = field(default_factory=dict)
    read_denials: dict[str, int] = field(default_factory=dict)
    preload_reads: dict[str, int] = field(default_factory=dict)
    result_admissions: dict[str, int] = field(default_factory=dict)
    result_denials: dict[str, int] = field(default_factory=dict)
    model_entries: dict[str, int] = field(default_factory=dict)
    model_entry_denials: dict[str, int] = field(default_factory=dict)
    per_worker_calls: dict[str, int] = field(default_factory=dict)
    per_worker_reused: dict[str, int] = field(default_factory=dict)

    def note_call(self, worker: str) -> None:
        self.calls += 1
        self.per_worker_calls[worker] = self.per_worker_calls.get(worker, 0) + 1

    def note_reuse(self, worker: str) -> None:
        self.reused += 1
        self.per_worker_reused[worker] = self.per_worker_reused.get(worker, 0) + 1

    def note_read_entered(self, site: str) -> None:
        """One read body actually reached after live admission passed."""

        self.reads_entered[site] = self.reads_entered.get(site, 0) + 1

    def note_preload_admitted(self, site: str) -> None:
        """One protected preload read admitted in the same transaction it reads in.

        Preloads run before any Worker exists, so they are counted apart from the Worker
        read sites: mixing them would make "which Worker read" unanswerable.
        """

        self.preload_reads[site] = self.preload_reads.get(site, 0) + 1

    def note_read_denied(self, site: str) -> None:
        """One read site refused by live admission; the reason travels with the signal."""

        self.read_denials[site] = self.read_denials.get(site, 0) + 1

    def note_result_admitted(self, site: str) -> None:
        """One already-returned result admitted for consumption at a Worker boundary.

        Kept apart from the read counters on purpose: a consumption check is not a new
        read, and mixing the two would make "how much did this execution read" unanswerable.
        """

        self.result_admissions[site] = self.result_admissions.get(site, 0) + 1

    def note_result_denied(self, site: str) -> None:
        """One consumption boundary refused by live admission, by site."""

        self.result_denials[site] = self.result_denials.get(site, 0) + 1

    def note_model_call_entered(self, site: str) -> None:
        """One model adapter synchronously received a request it was authorized to get.

        Raised only after ``start()`` returned the call handle inside the admitting
        transaction.  A booked attempt, a written intent, a created coroutine or a moved
        counter is deliberately not this event.
        """

        self.model_entries[site] = self.model_entries.get(site, 0) + 1

    def note_model_entry_denied(self, site: str) -> None:
        """One model entry refused by the live-authority re-read, by Worker site."""

        self.model_entry_denials[site] = self.model_entry_denials.get(site, 0) + 1

    def as_summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "reused": self.reused,
            "retries": self.retries,
            "exhausted": self.exhausted,
            "timeouts": self.timeouts,
            "late_discarded": self.late_discarded,
            "repeated_reads": self.repeated_reads,
            "reads_entered": dict(sorted(self.reads_entered.items())),
            "read_denials": dict(sorted(self.read_denials.items())),
            "preload_reads": dict(sorted(self.preload_reads.items())),
            "result_admissions": dict(sorted(self.result_admissions.items())),
            "result_denials": dict(sorted(self.result_denials.items())),
            "per_worker_calls": dict(sorted(self.per_worker_calls.items())),
            "per_worker_reused": dict(sorted(self.per_worker_reused.items())),
        }


def is_transient_failure(result: WorkerBranchResult) -> bool:
    """Return whether one failed branch result carries an explicit transient marker."""

    if result.outcome is not WorkerOutcome.FAILED:
        return False
    if result.failure_code in TRANSIENT_FAILURE_CODES:
        return True
    detail = result.failure_detail or ""
    return (
        result.failure_code is WorkerFailureCode.PORT_FAILURE
        and detail in TRANSIENT_FAILURE_DETAILS
    )


def contract_checksum(scope: RuntimeScope, worker: str) -> str:
    """Return the contract identity a checkpoint must match to stay reusable."""

    return canonical_checksum(
        {
            "checkpoint_contract": CHECKPOINT_CONTRACT_VERSION,
            "worker": worker,
            "document_version": scope.document_version,
            "dataset_version": scope.dataset_version,
            "index_version": scope.index_version,
            "rules_version": scope.rules_version,
            "case_version": scope.case_version,
        }
    )


def input_checksum(request: Any) -> str:
    if hasattr(request, "model_dump"):
        return canonical_checksum(request.model_dump(mode="json"))
    return canonical_checksum(request)


class CheckpointedWorker:
    """Wrap one Worker port with checkpoint reuse, an attempt ledger and retry."""

    def __init__(
        self,
        *,
        worker: str,
        inner: Any,
        repository: ReviewRepository,
        run_id: str,
        scope: RuntimeScope,
        generation: int,
        owner_id: str | None = None,
        telemetry: BranchTelemetry,
        now: Callable[[], datetime],
        fault: FaultController | None = None,
        deadline_at: datetime | None = None,
        attempt_budget: int = ATTEMPT_BUDGET_PER_WORKER,
        retry_budget: int = GENERATION_RETRY_BUDGET,
        retry_backoff_seconds: float = RETRY_BACKOFF_SECONDS,
        attempt_timeout_seconds: float = ATTEMPT_TIMEOUT_SECONDS,
    ) -> None:
        self._worker = worker
        self._inner = inner
        self._repository = repository
        self._run_id = run_id
        self._scope = scope
        self._generation = generation
        self._owner_id = owner_id
        self._telemetry = telemetry
        self._now = now
        self._fault = fault
        self._deadline_at = deadline_at or scope.deadline_at
        self._attempt_budget = attempt_budget
        self._retry_budget = retry_budget
        self._retry_backoff_seconds = retry_backoff_seconds
        self._attempt_timeout_seconds = attempt_timeout_seconds
        self._retries_used = 0

    async def review(self, request: Any) -> WorkerBranchResult:
        bound = _execution_worker.set(self._worker)
        try:
            return await self._review_with_checkpoints(request)
        except SQLAlchemyError as error:
            raise StorageUnavailableSignal(type(error).__name__) from error
        finally:
            _execution_worker.reset(bound)

    async def _review_with_checkpoints(self, request: Any) -> WorkerBranchResult:
        self._repository.assert_generation_live(
            run_id=self._run_id,
            generation=self._generation,
            owner_id=self._owner_id,
            now=self._now,
            site=f"{self._worker}.checkpoint_reuse",
        )
        try:
            adopted = self._repository.adopted_checkpoint(self._run_id, self._worker)
        except RecordUnavailableError as error:
            raise IncompatibleCheckpointError(
                f"checkpoint record unavailable for {self._worker}"
            ) from error
        expected_input = input_checksum(request)
        expected_contract = contract_checksum(self._scope, self._worker)
        if adopted is not None:
            stored_input = str(adopted.get("input_checksum", ""))
            stored_contract = str(adopted.get("contract_checksum", ""))
            if stored_contract != expected_contract:
                raise IncompatibleCheckpointError(
                    f"checkpoint contract mismatch for {self._worker}"
                )
            if stored_input != expected_input:
                raise IncompatibleCheckpointError(
                    f"checkpoint input mismatch for {self._worker}"
                )
            self._telemetry.note_reuse(self._worker)
            return validate_stored_model(WorkerBranchResult, adopted["result"])

        while True:
            attempt_no = self._next_attempt_no()
            if attempt_no > self._attempt_budget:
                self._telemetry.exhausted += 1
                return self._controlled_failure(
                    "worker_attempt_budget_exhausted",
                )
            self._repository.begin_attempt(
                run_id=self._run_id,
                generation=self._generation,
                worker=self._worker,
                attempt_no=attempt_no,
                now=self._now,
                owner_id=self._owner_id,
            )
            if self._fault is not None:
                self._fault.barrier(f"attempt_started:{self._worker}")
            result = await self._call_once(request, attempt_no)
            self._repository.assert_generation_live(
                run_id=self._run_id,
                generation=self._generation,
                owner_id=self._owner_id,
                now=self._now,
                site=f"{self._worker}.attempt_result",
            )
            if result.outcome is WorkerOutcome.FAILED and is_transient_failure(result):
                if self._retries_used < self._retry_budget:
                    self._retries_used += 1
                    self._telemetry.retries += 1
                    self._repository.finish_attempt(
                        run_id=self._run_id,
                        generation=self._generation,
                        worker=self._worker,
                        attempt_no=attempt_no,
                        outcome="transient_failure",
                        error_code=result.failure_code.value
                        if result.failure_code
                        else "transient_failure",
                        now=self._now(),
                    )
                    await asyncio.sleep(self._retry_backoff_seconds)
                    continue
            if result.outcome is WorkerOutcome.FAILED:
                self._repository.finish_attempt(
                    run_id=self._run_id,
                    generation=self._generation,
                    worker=self._worker,
                    attempt_no=attempt_no,
                    outcome="failed",
                    error_code=result.failure_code.value
                    if result.failure_code
                    else "unknown_failure",
                    now=self._now(),
                )
                return result
            if self._fault is not None:
                self._fault.barrier(f"before_branch_commit:{self._worker}")
            commit = self._repository.commit_branch(
                run_id=self._run_id,
                generation=self._generation,
                worker=self._worker,
                attempt_no=attempt_no,
                input_checksum=expected_input,
                contract_checksum=expected_contract,
                result_payload={
                    "input_checksum": expected_input,
                    "contract_checksum": expected_contract,
                    "result": result.model_dump(mode="json"),
                },
                now=self._now,
                owner_id=self._owner_id,
                events=self._branch_events(result, attempt_no),
            )
            if not commit.adopted:
                self._telemetry.late_discarded += 1
                self._telemetry.discarded_late_results += 1
                return result
            if self._fault is not None:
                self._fault.barrier(f"branch_committed:{self._worker}")
            return result

    async def _call_once(self, request: Any, attempt_no: int) -> WorkerBranchResult:
        from .workers import failed_worker_result

        if self._fault is not None and self._fault.transient_count(
            self._worker, attempt_no
        ):
            self._fault.note_transient(self._worker, attempt_no)
            self._telemetry.note_call(self._worker)
            return failed_worker_result(
                self._worker_kind,
                WorkerFailureCode.PORT_FAILURE,
                "transient_read_failure",
            )
        delay = (
            self._fault.slow_seconds(self._worker, attempt_no)
            if self._fault is not None
            else 0.0
        )
        if delay and self._fault is not None:
            self._fault.note_slow(self._worker, attempt_no)

        async def timed_call() -> WorkerBranchResult:
            # The injected delay is part of the attempt, so it is bounded by — and can
            # trip — the per-attempt timeout exactly like a slow read would.
            if delay:
                await asyncio.sleep(delay)
            return await self._inner.review(request)

        self._telemetry.note_call(self._worker)
        try:
            return await asyncio.wait_for(
                timed_call(),
                timeout=self._attempt_timeout_seconds,
            )
        except TimeoutError:
            self._telemetry.timeouts += 1
            from .workers import failed_worker_result

            return failed_worker_result(
                self._worker_kind,
                WorkerFailureCode.TIMEOUT,
                "worker_branch_timeout",
            )
        except IncompatibleCheckpointError:
            raise
        except SQLAlchemyError:
            raise
        except Exception:  # noqa: BLE001 - the port boundary is fail-closed
            from .workers import failed_worker_result

            return failed_worker_result(
                self._worker_kind,
                WorkerFailureCode.PORT_FAILURE,
                "worker_branch_exception",
            )

    @property
    def _worker_kind(self):
        from .contracts import WorkerKind

        return WorkerKind(self._worker)

    def _branch_events(
        self, result: WorkerBranchResult, attempt_no: int
    ) -> tuple[TraceDraft, ...]:
        failed = result.outcome is WorkerOutcome.FAILED
        output_summary: dict[str, Any] = {
            "worker": self._worker,
            "outcome": result.outcome.value,
            "finding_count": len(result.findings),
            "evidence_ref_count": sum(
                len(finding.evidence_refs) for finding in result.findings
            ),
            "business_fact_count": sum(
                len(finding.business_fact_refs) for finding in result.findings
            ),
        }
        evidence_ids = sorted(
            {
                item.chunk_id or f"{item.source_id}:{item.version}"
                for finding in result.findings
                for item in finding.evidence_refs
            }
        )
        fact_ids = sorted(
            {
                f"{item.tool_name}:{item.record_id}"
                for finding in result.findings
                for item in finding.business_fact_refs
            }
        )
        if evidence_ids:
            output_summary["evidence_ref_ids"] = evidence_ids
        if fact_ids:
            output_summary["business_fact_ids"] = fact_ids
        if result.failure_code is not None:
            output_summary["failure_code"] = result.failure_code.value
        return (
            TraceDraft(
                event_type=("step_failed" if failed else "step_completed"),
                step=f"worker.{self._worker}",
                output_summary=output_summary,
                source_versions={
                    "worker_result": "worker-result.v1",
                    "checkpoint": CHECKPOINT_CONTRACT_VERSION,
                },
                error=result.failure_detail if failed else None,
                generation=self._generation,
                attempt=attempt_no,
                worker=self._worker,
            ),
        )

    def _next_attempt_no(self) -> int:
        totals = self._repository.attempt_totals(self._run_id)
        return int(totals.get(self._worker, 0)) + 1

    def _controlled_failure(self, detail: str) -> WorkerBranchResult:
        from .workers import failed_worker_result

        return failed_worker_result(
            self._worker_kind,
            WorkerFailureCode.PORT_FAILURE,
            detail,
        )


class CheckpointedPorts:
    """Build the three checkpoint-aware ports and remember their telemetry."""

    def __init__(
        self,
        *,
        repository: ReviewRepository,
        run_id: str,
        scope: RuntimeScope,
        generation: int,
        owner_id: str | None = None,
        now: Callable[[], datetime],
        fault: FaultController | None = None,
        telemetry: BranchTelemetry | None = None,
    ) -> None:
        self.telemetry = telemetry or BranchTelemetry()
        self._repository = repository
        self._run_id = run_id
        self._scope = scope
        self._generation = generation
        self._owner_id = owner_id
        self._now = now
        self._fault = fault

    def wrap(self, worker: str, inner: Any) -> CheckpointedWorker:
        return CheckpointedWorker(
            worker=worker,
            inner=inner,
            repository=self._repository,
            run_id=self._run_id,
            scope=self._scope,
            generation=self._generation,
            telemetry=self.telemetry,
            now=self._now,
            owner_id=self._owner_id,
            fault=self._fault,
        )


def run_not_found(run_id: str) -> RunNotFoundError:
    return RunNotFoundError(run_id)


__all__ = [
    "ATTEMPT_BUDGET_PER_WORKER",
    "ATTEMPT_TIMEOUT_SECONDS",
    "BRANCH_TIMEOUT_SECONDS",
    "BranchTelemetry",
    "CHECKPOINT_CONTRACT_VERSION",
    "CheckpointedPorts",
    "CheckpointedWorker",
    "EXHAUSTED_FAILURE_DETAILS",
    "GENERATION_DEADLINE_SECONDS",
    "GENERATION_RETRY_BUDGET",
    "IncompatibleCheckpointError",
    "RETRY_BACKOFF_SECONDS",
    "ReadAdmission",
    "StorageUnavailableSignal",
    "TRANSIENT_FAILURE_DETAILS",
    "contract_checksum",
    "current_execution_worker",
    "input_checksum",
    "is_transient_failure",
    "run_not_found",
]
