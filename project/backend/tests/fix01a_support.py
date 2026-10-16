"""Shared harness for the FIX-01A authorization and recovery verifications.

Every scenario here runs against its own disposable PostgreSQL database, and the
revoking side always uses a *second* engine, so it is a genuinely separate database
backend (checkable through ``pg_stat_activity``) rather than a reuse of the application's
own connection.

Timing windows come from the product's own file barriers in :mod:`
procurement_review.fault_hooks`: a parked barrier freezes the service's event loop, which
makes "the revocation committed while the execution stood here" a fact of the run instead
of a probability.  No scenario guesses with a bare sleep, and no scenario trusts an
in-process lock to serialize what the database must serialize.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

from procurement_review.api_contracts import ReviewCreateRequestV2
from procurement_review.authorization import AuthorizationService, generate_credential
from procurement_review.fault_hooks import BarrierSpec, FaultController
from procurement_review.local_runtime import ScopedWorkflowRunner
from procurement_review.review_repository import ReviewRepository
from procurement_review.review_service import ReviewService
from procurement_review.review_store import ReviewStore
from procurement_review.security_contracts import (
    AuthContext,
    RuntimeScope,
    SecurityOperation,
)
from procurement_review.security_fixture import build_seed_plan

from v1_03_support import drop_database, require_admin_dsn

DEMO_TENANT = "tenant-local-demo"
DEMO_SUBJECT = "subject-local-demo"
ALPHA_TENANT = "tenant-alpha"
ALPHA_OPERATOR = "subject-alpha-operator"
ALPHA_APPROVER = "subject-alpha-approver"
LOW_CASE = "CASE-LOW-001"
ALPHA_CASE_ONE = "CASE-ALPHA-001"
BARRIER_ROOT = Path("/tmp") / f"fix01a-bars-{secrets.token_hex(4)}"
APP_TAG = "fix01a_app"
DRIVER_TAG = "fix01a_driver"


class Fix01aEnvironmentError(RuntimeError):
    """Raised when a scenario asks for something the harness cannot prove."""


def _database_url(base: str, database: str, application_name: str) -> str:
    head, _, tail = base.partition("?")
    scheme_and_host, _, _old = head.rpartition("/")
    query = (
        f"application_name={application_name}&{tail}"
        if tail
        else f"application_name={application_name}"
    )
    return f"{scheme_and_host}/{database}?{query}"


class Loop:
    """One event loop in its own thread, so a parked barrier cannot block the driver."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def start(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def call(self, coro: Any, timeout: float = 120.0) -> Any:
        return self.start(coro).result(timeout=timeout)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)


@dataclass(frozen=True, slots=True)
class Identity:
    """One resolved credential plus the Case view the API would hand to it."""

    token: str
    credential_id: str
    context: AuthContext
    registration: Any
    scope: RuntimeScope


class Fix01aFixture:
    """A migrated, seeded database with two separate connection pools."""

    def __init__(self, name: str, *, application_name: str = APP_TAG) -> None:
        base = require_admin_dsn()
        self.application_name = application_name
        self.database = f"fix01a_{name}_{secrets.token_hex(4)}"
        admin = create_engine(base, isolation_level="AUTOCOMMIT")
        try:
            with admin.connect() as connection:
                connection.execute(
                    text(f'CREATE DATABASE "{self.database}" WITH TEMPLATE = template0')
                )
        finally:
            admin.dispose()
        self.app = ReviewStore.from_url(
            _database_url(base, self.database, application_name)
        )
        self.app.migrate()
        self.app.seed(build_seed_plan())
        # A second engine means a second backend: the revocation is committed by a
        # connection the application can neither see nor share.
        self.driver = ReviewStore.from_url(
            _database_url(base, self.database, DRIVER_TAG)
        )
        # A third engine for observing facts while the other two are busy or blocked.
        self.probe = ReviewStore.from_url(
            _database_url(base, self.database, "fix01a_probe")
        )
        self.repository = ReviewRepository(self.app)
        # The same repository over the observation connection: a scenario that terminates
        # every backend tagged as the application still has to scan and read afterwards.
        self.observer = ReviewRepository(self.probe)

    def close(self) -> None:
        self.app.close()
        self.driver.close()
        self.probe.close()
        drop_database(self.database)

    # ------------------------------------------------------------- identities

    def identity(
        self,
        subject_id: str = DEMO_SUBJECT,
        case_id: str = LOW_CASE,
        *,
        tenant_id: str = DEMO_TENANT,
        ttl_seconds: int = 3600,
    ) -> Identity:
        authorization = AuthorizationService(store=self.app)
        token, credential_id, digest = generate_credential()
        now = datetime.now(UTC)
        self.app.issue_credential(
            credential_id=credential_id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            token_digest=digest,
            issued_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        request_id = "fix01a"
        context = authorization.resolve(
            token, request_id=request_id, operation=SecurityOperation.REVIEW_CREATE
        )
        registration = authorization.require_case(
            context,
            case_id,
            request_id=request_id,
            operation=SecurityOperation.REVIEW_CREATE,
        )
        scope = authorization.build_runtime_scope(
            context,
            registration,
            generation=1,
            deadline_at=authorization.runtime_scope_deadline(30.0),
        )
        return Identity(
            token=token,
            credential_id=credential_id,
            context=context,
            registration=registration,
            scope=scope,
        )

    def create_request(self, identity: Identity) -> ReviewCreateRequestV2:
        return ReviewCreateRequestV2(
            case_id=identity.registration.case_id,
            case_version=identity.registration.case_version,
        )

    # ------------------------------------------------------------- revocations

    def revoke_credential(self, credential_id: str) -> bool:
        """Commit one credential revocation on the independent connection."""

        return self.driver.revoke_credential(
            credential_id=credential_id,
            revoked_at=datetime.now(UTC),
            reason="fix01a",
        )

    def expire_credential(self, credential_id: str) -> None:
        with self.driver.transaction() as connection:
            connection.execute(
                text(
                    "UPDATE review_credential SET expires_at = :at "
                    "WHERE credential_id = :c"
                ),
                {"at": datetime.now(UTC) - timedelta(seconds=1), "c": credential_id},
            )

    def disable_subject(self, subject_id: str) -> int:
        return self.driver.set_subject_disabled(subject_id=subject_id, disabled=True)

    def revoke_case_grant(self, subject_id: str, case_id: str) -> bool:
        return self.driver.set_case_grant(
            tenant_id=DEMO_TENANT,
            case_id=case_id,
            subject_id=subject_id,
            granted=False,
            occurred_at=datetime.now(UTC),
        )

    def bump_authorization_revision(self, tenant_id: str = DEMO_TENANT) -> int:
        """Move the tenant revision without touching any other fact."""

        with self.driver.transaction() as connection:
            connection.execute(
                text(
                    "UPDATE review_authorization_revision SET revision = revision + 1 "
                    "WHERE tenant_id = :t"
                ),
                {"t": tenant_id},
            )
        return self.driver.authorization_revision(tenant_id)

    # --------------------------------------------------------------- readings

    def ledger(self, run_id: str) -> dict[str, Any]:
        """The committed facts of one Run, read on the independent connection."""

        with self.driver.transaction() as connection:
            run = connection.execute(
                text(
                    "SELECT tenant_id, case_id, status, revision, resume_count, "
                    "error_code FROM review_run WHERE run_id = :r"
                ),
                {"r": run_id},
            ).one()
            attempts = connection.execute(
                text(
                    "SELECT coalesce(array_agg(generation || '/' || worker || '/' || "
                    "attempt_no ORDER BY generation, worker, attempt_no),'{}') "
                    "FROM review_worker_attempt WHERE run_id = :r"
                ),
                {"r": run_id},
            ).scalar_one()
            checkpoints = connection.execute(
                text(
                    "SELECT coalesce(array_agg(worker || '@' || generation ORDER BY "
                    "worker),'{}') FROM review_branch_checkpoint WHERE run_id = :r"
                ),
                {"r": run_id},
            ).scalar_one()
            summaries = connection.execute(
                text("SELECT count(*) FROM review_run_summary WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
            generations = connection.execute(
                text(
                    "SELECT coalesce(array_agg(generation || '=' || status ORDER BY "
                    "generation),'{}') FROM review_run_generation WHERE run_id = :r"
                ),
                {"r": run_id},
            ).scalar_one()
            audit = connection.execute(
                text(
                    "SELECT coalesce(array_agg(audit_id || '/' || event_class || ':' || "
                    "coalesce(new_status,'-') || ':' || "
                    "coalesce(event_payload #>> '{detail_code}','-') ORDER BY sequence),"
                    "'{}') FROM review_audit_event WHERE run_id = :r"
                ),
                {"r": run_id},
            ).scalar_one()
            trace = connection.execute(
                text(
                    "SELECT coalesce(array_agg(event_ordinal || '/' || step || ':' || "
                    "event_type ORDER BY event_ordinal),'{}') "
                    "FROM review_trace_event WHERE run_id = :r"
                ),
                {"r": run_id},
            ).scalar_one()
            leases = connection.execute(
                text(
                    "SELECT coalesce(array_agg(generation || '@' || "
                    "extract(epoch from lease_expires_at)::text ORDER BY generation),"
                    "'{}') FROM review_run_generation WHERE run_id = :r"
                ),
                {"r": run_id},
            ).scalar_one()
        return {
            "status": run.status,
            "revision": int(run.revision),
            "resume_count": int(run.resume_count),
            "error_code": run.error_code,
            "attempts": list(attempts),
            "checkpoints": list(checkpoints),
            "summaries": int(summaries),
            "generations": list(generations),
            "audit": list(audit),
            "trace": list(trace),
            "leases": list(leases),
        }

    def url_for(self, application_name: str) -> str:
        """This scenario's own database URL under a connection tag of the caller's choice.

        A scenario that starts a second *process*-side engine — the stdio startup context
        carries a database URL of its own — uses this instead of the application engine URL,
        so the observation backends stay distinguishable in ``pg_stat_activity``.
        """

        return _database_url(require_admin_dsn(), self.database, application_name)

    def admin_engine(self, *, autocommit: bool = True) -> Any:
        """A fresh engine for one connection, so a thread can contend on its own."""

        return create_engine(
            _database_url(require_admin_dsn(), self.database, "fix01a_contender"),
            **({"isolation_level": "AUTOCOMMIT"} if autocommit else {}),
        )

    def latest_run(self) -> str:
        """The newest Run in this scenario's own database."""

        with self.probe.transaction() as connection:
            return str(
                connection.execute(
                    text(
                        "SELECT run_id FROM review_run ORDER BY created_at DESC LIMIT 1"
                    )
                ).scalar_one()
            )

    def app_backends(self) -> list[str]:
        """The other live backends of this database, tagged with their connection."""

        with self.driver.transaction() as connection:
            rows = connection.execute(
                text(
                    "SELECT pid, application_name FROM pg_stat_activity "
                    "WHERE datname = current_database() AND pid <> pg_backend_pid() "
                    "ORDER BY pid"
                )
            ).all()
        return [f"{row.pid}:{row.application_name}" for row in rows]

    def held_review_locks(self) -> list[str]:
        """Granted locks on this project's tables held by any other backend.

        Used to show that a port read happens with no row lock held: the read admission's
        transaction is closed before the data is touched.
        """

        with self.driver.transaction() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT l.pid, c.relname, l.mode
                    FROM pg_locks l
                    JOIN pg_class c ON c.oid = l.relation
                    JOIN pg_stat_activity a ON a.pid = l.pid
                    WHERE a.datname = current_database()
                      AND l.granted AND l.pid <> pg_backend_pid()
                      AND c.relname LIKE 'review\\_%'
                    ORDER BY l.pid, c.relname
                    """
                )
            ).all()
        return [f"{row.pid}:{row.relname}:{row.mode}" for row in rows]


# ------------------------------------------------------------------- barriers


def barrier_dir(name: str) -> Path:
    directory = BARRIER_ROOT / name
    directory.mkdir(parents=True, exist_ok=True)
    for item in directory.glob("*"):
        item.unlink()
    return directory


def wait_barrier(directory: Path, name: str, timeout: float = 60.0) -> float:
    """Block until the service reports that it stands at one named boundary."""

    marker = directory / f"{name.replace(':', '_')}.reached"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reached = marker_time(marker)
        if reached is not None:
            return reached
        time.sleep(0.02)
    raise AssertionError(f"the barrier {name} was never reached")


def marker_time(marker: Path) -> float | None:
    """The timestamp one barrier marker carries, or ``None`` while it is not readable yet.

    Existence alone is not the boundary: ``write_text`` truncates before it writes, so a
    marker can be visible with an empty body, and reading it as a number then fails the
    scenario that was only waiting for the service to arrive.
    """

    try:
        body = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return float(body)
    except ValueError:
        return None


def release_barrier(directory: Path, name: str) -> None:
    (directory / f"{name.replace(':', '_')}.release").write_text("go", encoding="utf-8")


def parked_fault(directory: Path, *names: str) -> FaultController:
    """A plan that parks the service at each named boundary."""

    return FaultController(
        enabled=True,
        directory=directory,
        barriers={
            name: BarrierSpec(action="wait", timeout_seconds=90.0) for name in names
        },
    )


@dataclass(frozen=True, slots=True)
class Runtime:
    """One service over one store, plus the runner that carries its read accounting."""

    service: ReviewService
    runner: ScopedWorkflowRunner
    repository: ReviewRepository

    @property
    def reads_entered(self) -> dict[str, int]:
        """Read bodies actually entered, by Worker site, since the process started."""

        return dict(self.runner.telemetry.reads_entered)

    @property
    def read_denials(self) -> dict[str, int]:
        return dict(self.runner.telemetry.read_denials)

    @property
    def preload_reads(self) -> dict[str, int]:
        """Preload reads admitted in their own transaction, by protected read site."""

        return dict(self.runner.telemetry.preload_reads)


def runtime(
    fixture: Fix01aFixture,
    fault: FaultController | None = None,
    **kwargs: Any,
) -> Runtime:
    settings: dict[str, Any] = {
        "lease_seconds": 1000.0,
        "heartbeat_seconds": 1000.0,
        "scan_interval_seconds": 0.2,
    }
    settings.update(kwargs)
    runner = ScopedWorkflowRunner(store=fixture.app)
    repository = ReviewRepository(fixture.app, fault=fault)
    return Runtime(
        service=ReviewService(
            repository=repository,
            runner=runner,
            fault=fault,
            **settings,
        ),
        runner=runner,
        repository=repository,
    )


def settle(
    fixture: Fix01aFixture, run_id: str, *statuses: str, timeout: float = 60.0
) -> dict[str, Any]:
    """Wait for one of the given committed statuses, or report where it stayed."""

    deadline = time.monotonic() + timeout
    body = fixture.ledger(run_id)
    while time.monotonic() < deadline:
        if not statuses or body["status"] in statuses:
            return body
        time.sleep(0.05)
        body = fixture.ledger(run_id)
    raise AssertionError(
        f"run {run_id} stayed {body['status']}"
        f" (wanted {statuses}); error_code={body['error_code']}"
    )
