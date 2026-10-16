"""The narrow model-call boundary: authorize, build the request, receive it.

FIX-08 closed the two *result* boundaries — an already-returned read and an already-returned
model answer are both re-admitted before anything derived from them is used.  It did not
close the *entry*: the consumption admission returns when its own short transaction closes,
and the Worker only afterwards built the protected context and entered the model adapter, so
a revocation committed in that window still got one call carrying the whole context.

This module owns the missing linearization step.  One call to :meth:`RunModelInvocation.enter`
opens a short transaction, re-reads the live authority of the Run and generation, runs the
Worker's request factory, and lets the adapter *synchronously receive* the request — all
inside that same transaction.  The transaction then closes, and only afterwards is the
asynchronous result awaited.  Two orders are therefore both well defined:

* a revocation that commits first is seen by the re-read, so the factory never runs, the
  adapter never receives anything and no protected context leaves the Worker;
* a call the adapter received first was received under a live authority, so it may finish —
  but the revocation does not wait for the model, and the late answer is still dropped by
  the existing result admission.

The trust boundary this proves is the in-process adapter's synchronous receipt, not the
moment a remote vendor writes to a socket.  Proving the latter would need a revocable
dispatcher, an outbox or a vendor protocol, which is a different product decision.

Nothing here is a default.  The production runner builds this boundary for every execution
and the Workers refuse to be assembled without one; an execution with no Run to re-authorize
states its own named policy (:class:`OfflineModelInvocation`) instead of being handed a
silent pass.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from .checkpointing import StorageUnavailableSignal
from .execution_guards import ExecutionInvalidated
from .review_repository import ReviewRepository
from .worker_contracts import (
    ModelCall,
    SynchronousModelReceiver,
    WorkerModelRequest,
)

#: The barrier that parks an execution after the adapter has synchronously received the
#: request but before the admitting transaction has been released.  Only the *site* varies.
AFTER_MODEL_CALL_RECEIVED = "after_model_call_received"

#: The barrier that parks an execution after the live authority has been re-read and its
#: share locks are held, but before the request factory has run.
DURING_MODEL_ENTRY_AUTHORIZATION = "during_model_entry_authorization"


class RunModelInvocation:
    """Re-read the live authority of one Run and generation around one model receipt.

    One instance is created per execution, so the identity it carries can neither leak into
    a later generation that took the Run over nor into another Run.  ``now`` is called at
    each entry rather than captured once, because a lease that expired since the Worker was
    assembled must end the generation exactly like a revoked credential would.
    """

    offline = False

    def __init__(
        self,
        *,
        repository: ReviewRepository,
        run_id: str,
        generation: int,
        owner_id: str | None = None,
        now: Callable[[], datetime],
        receiver: Any,
        telemetry: Any | None = None,
        fault: Any | None = None,
    ) -> None:
        if not isinstance(receiver, SynchronousModelReceiver):
            # No default, no bridge and no coroutine-only fallback: an adapter that cannot
            # receive synchronously cannot be the linearization point, and silently
            # degrading here would put the whole gap back.
            raise TypeError(
                "the gated model-call boundary requires an adapter with a synchronous "
                "start() receive API"
            )
        self._repository = repository
        self._run_id = run_id
        self._generation = generation
        self._owner_id = owner_id
        self._now = now
        self._receiver = receiver
        self._telemetry = telemetry
        self._fault = fault

    def enter(
        self,
        *,
        site: str,
        request_factory: Callable[[], WorkerModelRequest],
    ) -> ModelCall:
        """Admit, build and receive one call inside a single short transaction.

        The share locks the live-authority re-read takes are held until this block ends, so
        a revocation either finishes first and is seen here, or waits for this short
        transaction — never for the model.
        """

        try:
            with self._repository.store.transaction() as connection:
                self._repository.require_generation_live(
                    connection,
                    run_id=self._run_id,
                    generation=self._generation,
                    now=self._now,
                    owner_id=self._owner_id,
                )
                if self._fault is not None:
                    # Parked here, the authorization locks are held and the request does not
                    # exist yet: the exact window the factory must not survive a revocation in.
                    self._fault.barrier(f"{DURING_MODEL_ENTRY_AUTHORIZATION}:{site}")
                self._repository._guards.require_execution_time(
                    connection,
                    run_id=self._run_id,
                    generation=self._generation,
                    now=self._now,
                )
                request = request_factory()
                self._repository._guards.require_execution_time(
                    connection,
                    run_id=self._run_id,
                    generation=self._generation,
                    now=self._now,
                )
                handle = self._receiver.start(request)
                if self._fault is not None:
                    # Parked here, the adapter has received the request and the transaction
                    # that admitted it has not been released yet.
                    self._fault.barrier(f"{AFTER_MODEL_CALL_RECEIVED}:{site}")
        except ExecutionInvalidated:
            # The whole generation lost its execution authority: the reasons travel with
            # the signal and the Run-level handler turns them into one controlled stop.
            if self._telemetry is not None:
                self._telemetry.note_model_entry_denied(site)
            raise
        except SQLAlchemyError as error:
            # The authorization store itself is unreachable.  This must not degrade into one
            # Worker's fail-closed branch result, and it must not fall back to an offline
            # policy that would let the call through without any authorization at all.
            raise StorageUnavailableSignal(type(error).__name__) from error
        if self._telemetry is not None:
            # Raised only now: the adapter has taken the request, which is what "the call
            # entered" means.  A booked attempt or a moved counter is not this.
            self._telemetry.note_model_call_entered(site)
        return ModelCall(request=request, handle=handle)


class OfflineModelInvocation:
    """The explicit policy of an execution with no persisted Run to re-authorize.

    The frozen in-memory runner, an isolated Worker assertion and an offline evaluation all
    have no Run, generation or review database, and none of them may be forced to invent
    one.  They state this policy on purpose; it is never a default and never what the
    production runner passes.

    It is not a weaker *receipt*: an adapter that offers the synchronous ``start`` API is
    used through it exactly like the production path, and the entry is counted as such.  An
    adapter that offers only the coroutine entry is reported as such rather than being
    mistaken for a synchronous receipt, so the evidence can name which one happened.
    """

    offline = True

    def __init__(self, *, receiver: Any) -> None:
        if isinstance(receiver, SynchronousModelReceiver):
            self._start = receiver.start
            self._generate: Any | None = None
            self.synchronous_receive = True
        elif hasattr(receiver, "generate"):
            self._start = None
            self._generate = receiver.generate
            self.synchronous_receive = False
        else:
            raise TypeError("the offline model policy requires a model adapter")
        self.coroutine_entries = 0

    def enter(
        self,
        *,
        site: str,
        request_factory: Callable[[], WorkerModelRequest],
    ) -> ModelCall:
        del site
        request = request_factory()
        if self._start is not None:
            return ModelCall(request=request, handle=self._start(request))
        self.coroutine_entries += 1
        return ModelCall(
            request=request, handle=_CoroutineCall(self._generate, request)
        )


class _CoroutineCall:
    """The handle of an adapter that only offers the coroutine entry, stated explicitly."""

    __slots__ = ("_generate", "_request")

    def __init__(self, generate: Any, request: WorkerModelRequest) -> None:
        self._generate = generate
        self._request = request

    async def result(self) -> object:
        return await self._generate(self._request)


async def await_call(call: ModelCall) -> object:
    """Wait for a received call outside every transaction that could admit another one."""

    return await call.result()


__all__ = [
    "AFTER_MODEL_CALL_RECEIVED",
    "DURING_MODEL_ENTRY_AUTHORIZATION",
    "OfflineModelInvocation",
    "RunModelInvocation",
    "await_call",
]
