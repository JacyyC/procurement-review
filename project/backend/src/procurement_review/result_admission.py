"""The narrow admission that lets one already-returned result be consumed.

A read that has already returned, and a model answer that has already come back, are the
two points where a generation that lost its authority could still carry data forward: the
check that admitted the *read* says nothing about whether the *result* may be used, and the
commit guard runs far too late to keep it out of a model context.

The vocabulary of invalidation lives in ``execution_guards`` and the caller-connection half
of the check lives in ``admitted_reads``; this module only applies that same single
authority at a consumption boundary.  It opens its own short transaction, closes it before
the calling Worker builds anything, and never runs while a port, network or model call is in
flight.  It deliberately does *not* reuse ``ReadAdmission``: consuming a result is not a new
read, so it must not move a read counter or trip a read barrier.  The share locks it takes —
the tenant authorization revision row, then the Case grant — are the same ones and in the
same order as every other protected step, so the lock order is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from .checkpointing import StorageUnavailableSignal
from .execution_guards import ExecutionInvalidated
from .review_repository import ReviewRepository

#: The barrier a fault plan parks at to show the window between a passed admission and the
#: consumption it admits.  Only the *site* varies; the name of the boundary is fixed.
AFTER_RESULT_ADMISSION = "after_result_admission"


class OfflineResultConsumption:
    """The explicit policy of an execution with no persisted Run to re-authorize.

    The frozen in-memory runner, an isolated Worker assertion and an offline evaluation all
    have no Run, generation or review database, and none of them may be forced to invent
    one.  They state this policy on purpose.  It is never a default and never what the
    production runner passes: a Worker refuses to be built without a policy at all.
    """

    offline = True

    def admit_result(self, *, site: str) -> None:
        return None


class RunResultAdmission:
    """Re-read the live authority of one Run and generation at a consumption boundary.

    One instance is created per execution, so the identity it carries can neither leak into
    a later generation that took the Run over nor into another Run.  ``now`` is called at
    each admission rather than captured once, because a lease that has expired since the
    Worker was assembled must end the generation just like a revoked credential would.
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
        telemetry: Any | None = None,
        fault: Any | None = None,
    ) -> None:
        self._repository = repository
        self._run_id = run_id
        self._generation = generation
        self._owner_id = owner_id
        self._now = now
        self._telemetry = telemetry
        self._fault = fault

    def admit_result(self, *, site: str) -> None:
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
                self._fault.barrier(f"{AFTER_RESULT_ADMISSION}:{site}")
            self._repository.assert_generation_live(
                run_id=self._run_id,
                generation=self._generation,
                owner_id=self._owner_id,
                now=self._now,
                site=site,
            )
            if self._telemetry is not None:
                self._telemetry.note_result_admitted(site)

        except ExecutionInvalidated:
            # The whole generation lost its execution authority: the reasons travel with
            # the signal and the Run-level handler turns them into one controlled stop.
            if self._telemetry is not None:
                self._telemetry.note_result_denied(site)
            raise
        except SQLAlchemyError as error:
            # The authorization store itself is unreachable.  This must not degrade into
            # one Worker's fail-closed branch result, which would let the Run summarize on
            # data it can no longer authorize.
            raise StorageUnavailableSignal(type(error).__name__) from error


#: The one offline policy object, shared because it holds no state at all.
OFFLINE_RESULT_CONSUMPTION = OfflineResultConsumption()


__all__ = [
    "AFTER_RESULT_ADMISSION",
    "OFFLINE_RESULT_CONSUMPTION",
    "OfflineResultConsumption",
    "RunResultAdmission",
]
