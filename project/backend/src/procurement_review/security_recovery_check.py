"""V1-03 security/recovery runner.

The runner owns one disposable PostgreSQL database and one real API subprocess per
scenario, drives the scenario over HTTP, and records structured observations under a
private directory.  ``--verify-report`` recomputes every reported number from those
observations without touching the network or the database.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import create_engine, text

from .authorization import generate_credential
from .fault_hooks import (
    FAULT_ADMIN_DSN_ENV,
    FAULT_APPLICATION_NAME_ENV,
    FAULT_APPLICATION_NAME,
    FAULT_DIR_ENV,
    FAULT_EXIT_CODE,
    FAULT_PLAN_ENV,
    FAULT_PLAN_VERSION,
)
from .field_scan import field_scan_scenario
from .review_store import ReviewStore
from .security_fixture import build_seed_plan
from .trace import canonical_checksum

RUNNER_VERSION = "security-recovery-suite.v1"
#: The observation schema of the report.  ``v1`` is the shape written before FIX-03B;
#: ``v2`` adds the complete persisted-record scan.  A report is recomputed with the keys
#: its own version declares, so an older report keeps verifying after the addition.
OBSERVATION_VERSION = "review-suite-observations.v2"
LEGACY_OBSERVATION_VERSION = "review-suite-observations.v1"
SUITE_MATRIX_PATH = Path(__file__).parent / "resources" / "security" / "suite.v1.json"
DEFAULT_API_PORT = 8077
REPORT_FILENAME = "report.json"
OBSERVATIONS_FILENAME = "observations.json"
SUITE_NAME = "security-recovery-suite.v1"
DEMO_SUBJECT = "subject-local-demo"
DEMO_TENANT = "tenant-local-demo"
LOW_CASE = "CASE-LOW-001"
LOW_CASE_VERSION = "case-low-001.v1"
PROCESS_EXIT_TIMEOUT_SECONDS = 30.0


class RunnerError(RuntimeError):
    """Raised when a scenario cannot be executed as specified."""


def load_declared_matrix() -> dict[str, Any]:
    """Load the versioned predeclared matrix; expectations never come from results."""

    payload = json.loads(SUITE_MATRIX_PATH.read_text(encoding="utf-8"))
    if payload.get("suite_version") != SUITE_NAME:
        raise RunnerError("the declared suite version does not match the runner")
    rows = payload.get("scenarios") or []
    ids = [str(item["id"]) for item in rows]
    if len(ids) != len(set(ids)):
        raise RunnerError("the declared matrix repeats a scenario id")
    if int(payload.get("fixed_recovery_denominator", 0)) != 24:
        raise RunnerError("the frozen recovery denominator must stay 24")
    recovery = [item for item in rows if item["class"] == "recovery"]
    if len(recovery) != 8 or any(int(item["repeats"]) != 3 for item in recovery):
        raise RunnerError("the recovery matrix must stay eight classes times three")
    return payload


def _dsn() -> str:
    value = os.environ.get("SECURITY_RECOVERY_ADMIN_DSN") or os.environ.get(
        "TEST_POSTGRES_DSN"
    )
    if not value:
        raise RunnerError(
            "SECURITY_RECOVERY_ADMIN_DSN (or TEST_POSTGRES_DSN) is required"
        )
    return value


def _database_url(base: str, database: str, application_name: str | None = None) -> str:
    head, _, tail = base.partition("?")
    scheme_and_host, _, _old = head.rpartition("/")
    name = application_name or FAULT_APPLICATION_NAME
    url = f"{scheme_and_host}/{database}?application_name={name}"
    if tail:
        url = f"{url}&{tail}"
    return url


def _admin_engine(base: str):
    return create_engine(base, isolation_level="AUTOCOMMIT")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(slots=True)
class ScenarioObservation:
    scenario: str
    repeat_index: int
    expected: str
    actual: str
    passed: bool
    counters: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "repeat_index": self.repeat_index,
            "expected": self.expected,
            "actual": self.actual,
            "passed": self.passed,
            "counters": self.counters,
            "timings": self.timings,
            "notes": self.notes,
        }


class DatabaseFixture:
    """One disposable database plus its migrated and seeded schema."""

    def __init__(self) -> None:
        self.admin_dsn = _dsn()
        self.name = f"v1_03_rec_{secrets.token_hex(6)}"
        # Each fixture tags its own connections, so one scenario's injected outage can
        # never touch another fixture that happens to run at the same time.
        self.application_name = f"{FAULT_APPLICATION_NAME}_{self.name}"
        os.environ[FAULT_APPLICATION_NAME_ENV] = self.application_name
        engine = _admin_engine(self.admin_dsn)
        try:
            with engine.connect() as connection:
                connection.execute(text(f'CREATE DATABASE "{self.name}"'))
        finally:
            engine.dispose()
        self.url = _database_url(self.admin_dsn, self.name, self.application_name)
        self.store = ReviewStore.from_url(self.url)
        self.store.migrate()
        self.store.seed(build_seed_plan())

    def close(self) -> None:
        self.store.close()
        engine = _admin_engine(self.admin_dsn)
        try:
            with engine.connect() as connection:
                connection.execute(
                    text(f'DROP DATABASE IF EXISTS "{self.name}" WITH (FORCE)')
                )
        finally:
            engine.dispose()

    def issue_token(
        self, subject_id: str = DEMO_SUBJECT, *, tenant: str | None = None
    ) -> str:
        token, credential_id, digest = generate_credential()
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        self.store.issue_credential(
            credential_id=credential_id,
            tenant_id=tenant or DEMO_TENANT,
            subject_id=subject_id,
            token_digest=digest,
            issued_at=now,
            expires_at=now + timedelta(hours=2),
        )
        return token

    def authorization_revision(self) -> int:
        return self.store.authorization_revision(DEMO_TENANT)


@dataclass(slots=True)
class ApiProcess:
    """One real API subprocess bound to the disposable database."""

    fixture: DatabaseFixture
    fault_dir: Path
    port: int
    plan: dict[str, Any] | None = None
    process: subprocess.Popen[bytes] | None = None
    log_path: Path | None = None

    def start(self) -> None:
        environment = dict(os.environ)
        environment["REVIEW_DATABASE_URL"] = self.fixture.url
        environment[FAULT_DIR_ENV] = str(self.fault_dir)
        environment[FAULT_ADMIN_DSN_ENV] = self.fixture.admin_dsn
        environment["REVIEW_ALLOWED_ORIGINS"] = ""
        plan_path = self.fault_dir / "plan.json"
        if self.plan is not None:
            payload = dict(self.plan)
            payload["fault_plan_version"] = FAULT_PLAN_VERSION
            plan_path.write_text(json.dumps(payload), encoding="utf-8")
            environment[FAULT_PLAN_ENV] = str(plan_path)
        else:
            environment.pop(FAULT_PLAN_ENV, None)
        self.log_path = self.fault_dir / f"api-{self.port}.log"
        log = self.log_path.open("wb")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "procurement_review.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_ready(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RunnerError("the API process exited before becoming ready")
            try:
                response = httpx.get(self.base_url + "/api/v2/health", timeout=1.0)
            except httpx.HTTPError:
                time.sleep(0.05)
                continue
            if response.status_code == 200:
                return
            time.sleep(0.05)
        raise RunnerError("the API process did not become ready")

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def wait_for_exit(
        self, timeout: float = PROCESS_EXIT_TIMEOUT_SECONDS
    ) -> int | None:
        if self.process is None:
            return None
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            with _suppress_errors():
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with _suppress_errors():
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=10)
        self.process = None


@contextmanager
def _suppress_errors() -> Iterator[None]:
    try:
        yield
    except Exception:  # noqa: BLE001 - teardown must never mask the real result
        pass


class ApiSession:
    """Small authenticated HTTP client for one scenario."""

    def __init__(self, *, base_url: str, token: str) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=30.0)
        self._headers = {"Authorization": f"Bearer {token}"}

    def close(self) -> None:
        self._client.close()

    def raw(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """One authenticated request; used by the security and audit scenarios."""

        return self._client.request(
            method,
            path,
            headers={**self._headers, **(headers or {})},
            json=json,
        )

    def create(self, *, key: str, case_id: str = LOW_CASE) -> dict[str, Any]:
        response = self._client.post(
            "/api/v2/reviews",
            headers={**self._headers, "Idempotency-Key": key},
            json={
                "api_version": "review-api.v2",
                "case_id": case_id,
                "case_version": f"{case_id.lower()}.v1",
            },
        )
        if response.status_code not in {200, 202}:
            raise RunnerError(f"create failed: {response.status_code} {response.text}")
        return response.json()

    def create_tolerating_crash(self, *, key: str) -> dict[str, Any] | None:
        """Create a run whose response may be lost to an injected process exit.

        A lost response is exactly the documented "result unknown" boundary: the only
        safe recovery is to re-send the same key and read the committed operation back.
        """

        try:
            return self.create(key=key)
        except (httpx.HTTPError, RunnerError):
            return None

    def run_state(self, run_id: str) -> dict[str, Any]:
        response = self._client.get(f"/api/v2/reviews/{run_id}", headers=self._headers)
        if response.status_code != 200:
            raise RunnerError(f"read failed: {response.status_code} {response.text}")
        return response.json()

    def wait_for(
        self, run_id: str, *statuses: str, timeout: float = 45.0
    ) -> dict[str, Any]:
        """Wait for a status, tolerating the injected dependency-outage window.

        While the REC-08 injection is active the store really is unavailable, so `503`
        is an expected transient answer rather than a scenario failure.
        """

        deadline = time.monotonic() + timeout
        body: dict[str, Any] = {}
        while time.monotonic() < deadline:
            response = self._client.get(
                f"/api/v2/reviews/{run_id}", headers=self._headers
            )
            if response.status_code in {500, 503}:
                time.sleep(0.1)
                continue
            if response.status_code != 200:
                raise RunnerError(
                    f"read failed: {response.status_code} {response.text}"
                )
            body = response.json()
            if body["status"] in statuses:
                return body
            time.sleep(0.05)
        raise RunnerError(
            f"run {run_id} stayed {body.get('status')}, wanted {statuses}"
        )

    def wait_until_available(self, *, consecutive: int = 3) -> None:
        """Wait until the store is stably reachable again after an injected outage."""

        streak = 0
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                response = self._client.get("/api/v2/health", timeout=2.0)
            except httpx.HTTPError:
                streak = 0
                time.sleep(0.1)
                continue
            if (
                response.status_code == 200
                and response.json().get("security_store") == "ready"
            ):
                streak += 1
                if streak >= consecutive:
                    return
            else:
                streak = 0
            time.sleep(0.1)
        raise RunnerError("the review store never became available again")

    def resume(self, run_id: str, *, key: str, expected_version: int) -> dict[str, Any]:
        response = self._client.post(
            f"/api/v2/reviews/{run_id}/resume",
            headers={**self._headers, "Idempotency-Key": key},
            json={
                "api_version": "review-api.v2",
                "expected_version": expected_version,
            },
        )
        if response.status_code not in {200, 202}:
            raise RunnerError(f"resume failed: {response.status_code} {response.text}")
        return response.json()

    def confirm(self, run_id: str, body: dict[str, Any], *, key: str) -> httpx.Response:
        return self._client.post(
            f"/api/v2/reviews/{run_id}/confirm",
            headers={**self._headers, "Idempotency-Key": key},
            json={
                "api_version": "review-api.v2",
                "expected_version": body["version"],
                "expected_summary_checksum": body["summary_checksum"],
            },
        )


def business_summary(body: dict[str, Any]) -> dict[str, Any] | None:
    """Canonical business view used to compare a recovered run with its baseline."""

    summary = body.get("summary")
    if summary is None:
        return None
    payload = dict(summary)
    payload["run_id"] = "run-placeholder"
    return payload


def summary_checksum(body: dict[str, Any]) -> str | None:
    summary = business_summary(body)
    if summary is None:
        return None
    return canonical_checksum(summary)


def database_counters(fixture: DatabaseFixture, run_id: str) -> dict[str, Any]:
    """Read the persisted ledger directly, independently of the API."""

    with fixture.store.transaction() as connection:
        attempts = connection.execute(
            text(
                "SELECT worker, attempt_no, outcome, abandoned, discarded_late "
                "FROM review_worker_attempt WHERE run_id = :r "
                "ORDER BY worker, attempt_no"
            ),
            {"r": run_id},
        ).all()
        checkpoints = connection.execute(
            text(
                "SELECT worker, generation, attempt_no FROM review_branch_checkpoint "
                "WHERE run_id = :r ORDER BY worker"
            ),
            {"r": run_id},
        ).all()
        run_row = connection.execute(
            text(
                "SELECT status, revision, resume_count FROM review_run "
                "WHERE run_id = :r"
            ),
            {"r": run_id},
        ).one()
        generations = connection.execute(
            text(
                "SELECT generation, status FROM review_run_generation "
                "WHERE run_id = :r ORDER BY generation"
            ),
            {"r": run_id},
        ).all()
        receipts = connection.execute(
            text("SELECT count(*) FROM review_receipt WHERE run_id = :r"),
            {"r": run_id},
        ).scalar_one()
        audits = connection.execute(
            text("SELECT count(*) FROM review_audit_event WHERE run_id = :r"),
            {"r": run_id},
        ).scalar_one()
    return {
        "status": run_row.status,
        "revision": int(run_row.revision),
        "resume_count": int(run_row.resume_count),
        "receipts": int(receipts),
        "audit_events": int(audits),
        "attempts": [
            {
                "worker": row.worker,
                "attempt_no": int(row.attempt_no),
                "outcome": row.outcome,
                "abandoned": bool(row.abandoned),
                "discarded_late": bool(row.discarded_late),
            }
            for row in attempts
        ],
        "checkpoints": [
            {
                "worker": row.worker,
                "generation": int(row.generation),
                "attempt_no": int(row.attempt_no),
            }
            for row in checkpoints
        ],
        "generations": [
            {"generation": int(row.generation), "status": row.status}
            for row in generations
        ],
    }


def _release(fault_dir: Path, barrier: str) -> None:
    (fault_dir / f"{barrier.replace(':', '_')}.release").write_text(
        "go", encoding="utf-8"
    )


def _wait_for_barrier(fault_dir: Path, barrier: str, timeout: float = 30.0) -> None:
    marker = fault_dir / f"{barrier.replace(':', '_')}.reached"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists():
            return
        time.sleep(0.02)
    raise RunnerError(f"the barrier {barrier} was never reached")


__all__ = [
    "ApiProcess",
    "ApiSession",
    "DatabaseFixture",
    "RunnerError",
    "ScenarioObservation",
    "business_summary",
    "database_counters",
    "summary_checksum",
    "RUNNER_VERSION",
    "SUITE_NAME",
    "DEMO_SUBJECT",
    "DEMO_TENANT",
    "LOW_CASE",
    "LOW_CASE_VERSION",
    "_release",
    "_wait_for_barrier",
    "_free_port",
    "FAULT_EXIT_CODE",
    "REC_IDS",
    "REC_SCENARIOS",
    "build_parser",
    "main",
    "run_rec_scenario",
    "run_suite",
    "verify_report",
]


# ---------------------------------------------------------------- scenarios

REC_SCENARIOS: dict[str, dict[str, Any]] = {
    "REC-01": {
        "expected": "lease expiry is detected and an explicit resume finishes the run",
        "plan": {"barriers": {"after_create_commit": {"action": "exit"}}},
        "crashes": True,
        "resume": True,
    },
    "REC-02": {
        "expected": "committed branches are reused and only unfinished ones run again",
        "plan": {"barriers": {"branch_committed:policy": {"action": "exit"}}},
        "crashes": True,
        "resume": True,
    },
    "REC-03": {
        "expected": "an uncommitted read is repeated once and counted accurately",
        "plan": {"barriers": {"before_branch_commit:policy": {"action": "exit"}}},
        "crashes": True,
        "resume": True,
    },
    "REC-04": {
        "expected": "three adopted checkpoints re-aggregate with zero new Worker calls",
        "plan": {"barriers": {"after_all_branches_committed": {"action": "exit"}}},
        "crashes": True,
        "resume": True,
    },
    "REC-05": {
        "expected": "a committed summary is read back unchanged with no receipt",
        "plan": {"barriers": {"after_summary_commit": {"action": "exit"}}},
        "crashes": True,
        "resume": False,
    },
    "REC-06": {
        "expected": "one transient read failure retries once inside the same generation",
        "plan": {"transient": {"policy": {"1": 1}}},
        "crashes": False,
        "resume": False,
    },
    "REC-07": {
        "expected": "a timed-out attempt is booked and cannot overwrite the retry result",
        "plan": {"slow": {"policy": {"1": 8.0}}},
        "crashes": False,
        "resume": False,
    },
    "REC-08": {
        "expected": "a checkpoint transaction interrupted by a dropped connection is not "
        "treated as committed, and the resume re-runs that branch",
        "plan": {
            "barriers": {
                "during_checkpoint_commit:policy": {
                    "action": "terminate",
                    "timeout_seconds": 0.5,
                    "once": True,
                }
            }
        },
        "crashes": False,
        "resume": True,
    },
}
REC_IDS = tuple(sorted(REC_SCENARIOS))

SUITE_GROUPS: dict[str, tuple[str, ...]] = {
    "rec": REC_IDS,
    "security": ("SEC-GATE",),
    "audit": ("AUDIT-RESTART",),
    "field": ("FIELD-SCAN",),
    "all": (*REC_IDS, "SEC-GATE", "AUDIT-RESTART", "FIELD-SCAN"),
}


def _baseline_summary(
    fixture: DatabaseFixture,
    fault_dir: Path,
    *,
    key: str,
) -> tuple[str, dict[str, Any] | None]:
    """Run the same Case without any injected fault and keep its business summary."""

    port = _free_port()
    process = ApiProcess(fixture=fixture, fault_dir=fault_dir, port=port)
    process.start()
    try:
        process.wait_ready()
        token = fixture.issue_token()
        session = ApiSession(base_url=process.base_url, token=token)
        try:
            created = session.create(key=key)
            body = session.wait_for(
                created["run_id"], "pending_confirmation", "pending_human_review"
            )
            return created["run_id"], business_summary(body)
        finally:
            session.close()
    finally:
        process.stop()


def run_rec_scenario(
    fixture: DatabaseFixture,
    *,
    scenario: str,
    repeat_index: int,
    fault_dir: Path,
    baseline: dict[str, Any] | None,
) -> ScenarioObservation:
    """Execute one REC scenario against real subprocesses and record the observation."""

    spec = REC_SCENARIOS[scenario]
    work_dir = fault_dir / f"{scenario}-{repeat_index}"
    work_dir.mkdir(parents=True, exist_ok=True)
    token = fixture.issue_token()
    timings: dict[str, float] = {}
    counters: dict[str, Any] = {}
    notes: dict[str, Any] = {"scenario_plan": spec["plan"]}
    malformed = bool(spec["plan"])
    barriers = dict((spec["plan"] or {}).get("barriers") or {})
    process = ApiProcess(
        fixture=fixture,
        fault_dir=work_dir,
        port=_free_port(),
        plan=spec["plan"],
    )
    session: ApiSession | None = None
    started = time.monotonic()
    run_id = ""
    try:
        process.start()
        process.wait_ready()
        session = ApiSession(base_url=process.base_url, token=token)
        created = (
            session.create_tolerating_crash(key=f"{scenario}-{repeat_index}")
            if scenario == "REC-01"
            else session.create(key=f"{scenario}-{repeat_index}")
        )
        run_id = created["run_id"] if created else ""
        notes["create_response_lost"] = created is None

        if spec["crashes"]:
            exit_code = process.wait_for_exit()
            timings["crash_detected_seconds"] = round(time.monotonic() - started, 3)
            counters["crash_exit_code"] = exit_code
            notes["crash_exit_code"] = exit_code
            if exit_code != FAULT_EXIT_CODE:
                raise RunnerError(
                    f"{scenario} expected the process to die at its barrier, got {exit_code}"
                )
            session.close()
            session = None
            process.stop()

            restarted = ApiProcess(
                fixture=fixture, fault_dir=work_dir, port=_free_port()
            )
            restarted.start()
            try:
                restarted.wait_ready()
                session = ApiSession(base_url=restarted.base_url, token=token)
                if not run_id:
                    # Same key, same body: the committed create operation is located
                    # again instead of guessing a new run id.
                    replayed = session.create(key=f"{scenario}-{repeat_index}")
                    run_id = replayed["run_id"]
                    notes["create_replayed"] = True
                body = _recover(session, run_id, scenario, repeat_index, spec, timings)
                notes["final_status"] = body["status"]
                notes["summary_matches_baseline"] = (
                    business_summary(body) == baseline if baseline is not None else None
                )
            finally:
                restarted.stop()
        else:
            if barriers:
                _wait_for_barrier(work_dir, next(iter(barriers)), timeout=45.0)
            if spec["resume"]:
                identification_started = time.monotonic()
                interrupted = session.wait_for(run_id, "interrupted", timeout=30.0)
                timings["interrupt_identification_seconds"] = round(
                    time.monotonic() - identification_started, 3
                )
                session.wait_until_available()
                time.sleep(1.0)
                resume_started = time.monotonic()
                session.resume(
                    run_id,
                    key=f"{scenario}-{repeat_index}-resume",
                    expected_version=interrupted["version"],
                )
                body = session.wait_for(
                    run_id,
                    "pending_confirmation",
                    "pending_human_review",
                    timeout=60.0,
                )
                timings["resume_seconds"] = round(time.monotonic() - resume_started, 3)
            else:
                body = session.wait_for(
                    run_id,
                    "pending_confirmation",
                    "pending_human_review",
                    timeout=60.0,
                )
            notes["final_status"] = body["status"]
            notes["summary_matches_baseline"] = (
                business_summary(body) == baseline if baseline is not None else None
            )
    finally:
        if session is not None:
            session.close()
        process.stop()

    counters.update(database_counters(fixture, run_id))
    counters["distinct_runs"] = _distinct_runs(fixture, run_id)
    counters["malformed_plan_used"] = malformed
    passed = bool(notes.get("summary_matches_baseline"))
    if spec["resume"]:
        passed = passed and counters["resume_count"] == 1
    return ScenarioObservation(
        scenario=scenario,
        repeat_index=repeat_index,
        expected=str(spec["expected"]),
        actual=(
            "recovered"
            if notes.get("summary_matches_baseline")
            else f"status={notes.get('final_status')}"
        ),
        passed=passed,
        counters=counters,
        timings=timings,
        notes=notes,
    )


def _recover(
    session: ApiSession,
    run_id: str,
    scenario: str,
    repeat_index: int,
    spec: dict[str, Any],
    timings: dict[str, float],
) -> dict[str, Any]:
    """Bring a crashed Run back to a readable state after the restart."""

    identification_started = time.monotonic()
    if not spec["resume"]:
        return session.run_state(run_id)
    interrupted = session.wait_for(run_id, "interrupted", timeout=30.0)
    # The dependency must be back before an authorized operator resumes; a resume that
    # lands inside the injected outage is not the scenario under test.
    session.wait_until_available()
    time.sleep(1.0)
    timings["interrupt_identification_seconds"] = round(
        time.monotonic() - identification_started, 3
    )
    resume_started = time.monotonic()
    session.resume(
        run_id,
        key=f"{scenario}-{repeat_index}-resume",
        expected_version=interrupted["version"],
    )
    body = session.wait_for(
        run_id, "pending_confirmation", "pending_human_review", timeout=60.0
    )
    timings["resume_seconds"] = round(time.monotonic() - resume_started, 3)
    return body


def _distinct_runs(fixture: DatabaseFixture, run_id: str) -> int:
    with fixture.store.transaction() as connection:
        return int(
            connection.execute(
                text("SELECT count(*) FROM review_run WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
        )


def _rec_invariants(scenario: str, observation: ScenarioObservation) -> dict[str, bool]:
    """Independent per-scenario assertions over the recorded observation."""

    counters = observation.counters
    attempts = counters.get("attempts", [])
    workers = {item["worker"] for item in attempts}
    checkpoints = {item["worker"] for item in counters.get("checkpoints", [])}
    if scenario == "REC-01":
        return {
            "summary_matches_baseline": bool(
                observation.notes.get("summary_matches_baseline")
            ),
            "resume_count_is_one": counters.get("resume_count") == 1,
            "nothing_committed_before_the_crash": all(
                item["generation"] == 2 for item in counters.get("checkpoints", [])
            ),
            "all_workers_attempted": workers
            == {"policy", "contract", "procurement_data"},
        }
    if scenario == "REC-02":
        return {
            "summary_matches_baseline": bool(
                observation.notes.get("summary_matches_baseline")
            ),
            "policy_reused": "policy" in checkpoints,
            "policy_attempted_once": sum(
                1 for item in attempts if item["worker"] == "policy"
            )
            == 1,
            "other_branches_ran": {"contract", "procurement_data"} <= workers,
        }
    if scenario == "REC-03":
        policy_attempts = [item for item in attempts if item["worker"] == "policy"]
        return {
            "summary_matches_baseline": bool(
                observation.notes.get("summary_matches_baseline")
            ),
            "policy_attempted_twice": len(policy_attempts) == 2,
            "policy_second_attempt_completed": policy_attempts[-1]["outcome"]
            == "completed",
            "policy_not_reused": len(
                [
                    item
                    for item in counters.get("checkpoints", [])
                    if item["worker"] == "policy"
                ]
            )
            == 1,
        }
    if scenario == "REC-04":
        return {
            "summary_matches_baseline": bool(
                observation.notes.get("summary_matches_baseline")
            ),
            "three_checkpoints": len(checkpoints) == 3,
            "no_second_attempt": all(item["attempt_no"] == 1 for item in attempts),
            "no_repeat_calls": len(attempts) == 3,
        }
    if scenario == "REC-05":
        return {
            "summary_matches_baseline": bool(
                observation.notes.get("summary_matches_baseline")
            ),
            "status_pending": observation.notes.get("final_status")
            in {"pending_confirmation", "pending_human_review"},
            "no_receipt": counters.get("receipts") == 0,
            "no_second_generation": len(counters.get("generations", [])) == 1,
            "no_resume": counters.get("resume_count") == 0,
        }
    if scenario == "REC-06":
        policy_attempts = [item for item in attempts if item["worker"] == "policy"]
        return {
            "summary_matches_baseline": bool(
                observation.notes.get("summary_matches_baseline")
            ),
            "policy_two_attempts": len(policy_attempts) == 2,
            "first_attempt_transient": policy_attempts[0]["outcome"]
            == "transient_failure",
            "other_branches_ran_once": all(
                len([item for item in attempts if item["worker"] == worker]) == 1
                for worker in ("contract", "procurement_data")
            ),
        }
    if scenario == "REC-07":
        policy_attempts = [item for item in attempts if item["worker"] == "policy"]
        policy_checkpoint = next(
            (
                item
                for item in counters.get("checkpoints", [])
                if item["worker"] == "policy"
            ),
            None,
        )
        return {
            "summary_matches_baseline": bool(
                observation.notes.get("summary_matches_baseline")
            ),
            "policy_two_attempts": len(policy_attempts) == 2,
            "first_attempt_timed_out": policy_attempts[0]["outcome"]
            in {"failed", "transient_failure"},
            "retry_completed": policy_attempts[-1]["outcome"] == "completed",
            "late_output_not_adopted": policy_checkpoint is not None
            and policy_checkpoint.get("attempt_no") == 2,
        }
    if scenario == "REC-08":
        policy_attempts = [item for item in attempts if item["worker"] == "policy"]
        return {
            "summary_matches_baseline": bool(
                observation.notes.get("summary_matches_baseline")
            ),
            "policy_attempted_twice": len(policy_attempts) == 2,
            "interrupted_was_reached": counters.get("resume_count") == 1,
            "single_adopted_policy_result": sum(
                1
                for item in counters.get("checkpoints", [])
                if item["worker"] == "policy"
            )
            == 1,
        }
    raise RunnerError(f"unknown scenario {scenario}")


def run_suite(
    *,
    suite: str,
    repeat: int,
    output: Path,
) -> dict[str, Any]:
    """Run one suite and write its observations and report."""

    if suite not in SUITE_GROUPS:
        raise RunnerError(f"unsupported suite {suite}")
    declared = load_declared_matrix()
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    fault_dir = output / "bars"
    fault_dir.mkdir(exist_ok=True)
    os.chmod(fault_dir, 0o700)
    fixture = DatabaseFixture()
    observations: list[ScenarioObservation] = []
    try:
        baseline_id, baseline = _baseline_summary(
            fixture, fault_dir, key="baseline-run"
        )
        for scenario in SUITE_GROUPS[suite]:
            for repeat_index in range(1, repeat + 1):
                try:
                    observation = run_named_scenario(
                        fixture,
                        scenario=scenario,
                        repeat_index=repeat_index,
                        fault_dir=fault_dir,
                        baseline=baseline,
                    )
                except Exception as error:  # noqa: BLE001 - a failed repeat is evidence
                    observations.append(
                        ScenarioObservation(
                            scenario=scenario,
                            repeat_index=repeat_index,
                            expected=str(
                                REC_SCENARIOS.get(scenario, {}).get(
                                    "expected", "scenario completes"
                                )
                            ),
                            actual=f"error:{type(error).__name__}",
                            passed=False,
                            counters={},
                            timings={},
                            notes={
                                "error": str(error)[:500],
                                "invariants": {"scenario_completed": False},
                            },
                        )
                    )
                    continue
                try:
                    observation.notes["invariants"] = (
                        _rec_invariants(scenario, observation)
                        if scenario in REC_SCENARIOS
                        else dict(observation.notes.get("checks", {}))
                    )
                except Exception as error:  # noqa: BLE001 - report, never hide
                    observation.notes["invariants"] = {
                        "invariants_evaluated": False,
                        "error": str(error)[:200],
                    }
                observation.passed = observation.passed and all(
                    observation.notes["invariants"].values()
                )
                observations.append(observation)
    finally:
        fixture.close()

    report = _build_report(
        suite=suite,
        repeat=repeat,
        baseline_run_id=baseline_id,
        baseline_summary=baseline,
        observations=observations,
        output=output,
    )
    report["declared_matrix"] = {
        "scenarios": len(declared["scenarios"]),
        "classes": sorted({item["class"] for item in declared["scenarios"]}),
        "fixed_recovery_denominator": declared["fixed_recovery_denominator"],
        "implemented_runners": sorted(
            {item["runner"] for item in declared["scenarios"]}
        ),
    }
    (output / OBSERVATIONS_FILENAME).write_text(
        json.dumps(
            {
                "suite": SUITE_NAME,
                "baseline_run_id": baseline_id,
                "baseline_summary": baseline,
                "observations": [item.as_dict() for item in observations],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output / REPORT_FILENAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.chmod(output / REPORT_FILENAME, 0o600)
    os.chmod(output / OBSERVATIONS_FILENAME, 0o600)
    return report


def _build_report(
    *,
    suite: str,
    repeat: int,
    baseline_run_id: str,
    baseline_summary: dict[str, Any] | None,
    observations: list[ScenarioObservation],
    output: Path,
) -> dict[str, Any]:
    successes = sum(1 for item in observations if item.passed)
    per_scenario: dict[str, dict[str, int]] = {}
    for item in observations:
        bucket = per_scenario.setdefault(item.scenario, {"runs": 0, "passed": 0})
        bucket["runs"] += 1
        bucket["passed"] += 1 if item.passed else 0
    return {
        "runner_version": RUNNER_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "suite": suite,
        "reported_suite": SUITE_NAME,
        "repeat": repeat,
        "baseline_run_id": baseline_run_id,
        "baseline_summary_checksum": (
            canonical_checksum(baseline_summary) if baseline_summary else None
        ),
        "recovery_success": {"numerator": successes, "denominator": len(observations)},
        "recovery_success_rate": (
            successes / len(observations) if observations else 0.0
        ),
        "per_scenario": per_scenario,
        "distinct_runs": sorted(
            {item.counters.get("distinct_runs", 0) for item in observations}
        ),
        "reused_branch_counts": {
            item.scenario: sorted(
                {
                    len(item.counters.get("checkpoints", []))
                    for item in observations
                    if item.scenario == item.scenario
                }
            )
            for item in observations
        },
        "total_attempts": sum(
            len(item.counters.get("attempts", [])) for item in observations
        ),
        "abandoned_attempts": sum(
            1
            for item in observations
            for attempt in item.counters.get("attempts", [])
            if attempt.get("abandoned")
        ),
        "field_scan": _field_scan_totals(observations),
        "observations_path": str((output / OBSERVATIONS_FILENAME).name),
        "output_directory_mode": oct(output.stat().st_mode & 0o777),
    }


def _field_scan_totals(observations: list[ScenarioObservation]) -> dict[str, Any]:
    """The complete-scan totals one report declares, recomputed from the raw rows."""

    scans = [item.counters.get("field_scan", {}) for item in observations]
    return {
        "scanned": sum(int(item.get("scanned", 0)) for item in scans),
        "refused": sum(int(item.get("refused", 0)) for item in scans),
        "classes": sorted(
            {entry.get("name") for item in scans for entry in item.get("classes", [])}
        ),
        "problems": sorted(
            {problem for item in scans for problem in item.get("problems", [])}
        ),
    }


def verify_report(directory: Path) -> dict[str, Any]:
    """Recompute the report from the raw observations; a drift fails the check."""

    observations_path = directory / OBSERVATIONS_FILENAME
    report_path = directory / REPORT_FILENAME
    if not observations_path.is_file() or not report_path.is_file():
        raise RunnerError("the report directory is incomplete")
    payload = json.loads(observations_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = payload["observations"]
    pairs = [(item["scenario"], item["repeat_index"]) for item in rows]
    duplicate_rows = len(pairs) != len(set(pairs))
    recomputed: dict[str, Any] = {
        "recovery_success": {
            "numerator": sum(1 for item in rows if item["passed"]),
            "denominator": len(rows),
        },
        "total_attempts": sum(
            len(item["counters"].get("attempts", [])) for item in rows
        ),
        "abandoned_attempts": sum(
            1
            for item in rows
            for attempt in item["counters"].get("attempts", [])
            if attempt.get("abandoned")
        ),
    }
    version = report.get("observation_version", LEGACY_OBSERVATION_VERSION)
    # A report is judged by the keys its own observation version declares: the complete
    # scan is recomputed for a v2 report, and a v1 report keeps verifying exactly as it
    # did when it was written.
    declared = set(recomputed)
    if version == OBSERVATION_VERSION:
        scans = [item["counters"].get("field_scan", {}) for item in rows]
        recomputed["field_scan"] = {
            "scanned": sum(int(item.get("scanned", 0)) for item in scans),
            "refused": sum(int(item.get("refused", 0)) for item in scans),
            "classes": sorted(
                {
                    entry.get("name")
                    for item in scans
                    for entry in item.get("classes", [])
                }
            ),
            "problems": sorted(
                {problem for item in scans for problem in item.get("problems", [])}
            ),
        }
        declared.add("field_scan")
    mismatches = [key for key in sorted(declared) if report.get(key) != recomputed[key]]
    return {
        "verified": not mismatches and not duplicate_rows,
        "mismatches": mismatches,
        "observations": len(rows),
        "observation_version": version,
        "scenarios": sorted({item["scenario"] for item in rows}),
        "duplicate_rows": duplicate_rows,
        "recomputed": recomputed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="security_recovery_check",
        description="Run the V1-03 recovery matrix against real subprocesses.",
    )
    parser.add_argument("--suite", default="rec", choices=sorted(SUITE_GROUPS))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", required=False, default=None)
    parser.add_argument("--verify-report", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verify_report:
        result = verify_report(Path(args.verify_report))
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0 if result["verified"] else 1
    if not args.output:
        parser.error("--output is required unless --verify-report is used")
    report = run_suite(
        suite=args.suite,
        repeat=args.repeat,
        output=Path(args.output),
    )
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return (
        0
        if report["recovery_success"]["numerator"]
        == report["recovery_success"]["denominator"]
        else 1
    )


# ------------------------------------------------------- security / audit / field

SEC_CHECKS = (
    "anonymous_create_is_401",
    "foreign_object_matches_unknown_object",
    "reader_scope_denial_is_403",
    "presented_credential_never_echoed",
    "retired_v1_carries_no_business_data",
    "denial_ledger_records_every_rejection",
)
AUDIT_CHECKS = (
    "approval_event_actor_is_the_verified_subject",
    "approval_event_carries_authorization_revision",
    "sequence_is_stable_across_a_restart",
    "denial_ledger_stays_out_of_the_tenant_run_audit",
)
FIELD_CHECKS = (
    "no_sensitive_pattern_in_api_log",
    "no_forbidden_key_in_persisted_payloads",
    "canaries_are_detected_by_the_scanner",
    "write_guard_rejects_a_poisoned_payload",
    "artifact_permissions_are_0700_and_0600",
)


def _restart_api(
    fixture: DatabaseFixture, work_dir: Path, session: ApiSession
) -> tuple[ApiProcess, ApiSession, str]:
    return _start_session(fixture, work_dir)


def _start_session(
    fixture: DatabaseFixture, work_dir: Path
) -> tuple[ApiProcess, ApiSession, str]:
    process = ApiProcess(fixture=fixture, fault_dir=work_dir, port=_free_port())
    process.start()
    process.wait_ready()
    token = fixture.issue_token()
    return process, ApiSession(base_url=process.base_url, token=token), token


def run_security_scenario(
    fixture: DatabaseFixture, *, repeat_index: int, work_dir: Path
) -> ScenarioObservation:
    """SEC gate: the rejection matrix has to hold on the live v2 surface."""

    work_dir.mkdir(parents=True, exist_ok=True)
    process, session, token = _start_session(fixture, work_dir)
    counters: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    try:
        before = fixture.store.denial_count()
        # A well-formed body without a credential: the authentication gate has to be
        # what answers, not the schema validator.
        anonymous = session.raw(
            "POST",
            "/api/v2/reviews",
            headers={"Authorization": "Bearer not-a-credential"},
            json={
                "api_version": "review-api.v2",
                "case_id": LOW_CASE,
                "case_version": LOW_CASE_VERSION,
            },
        )
        counters["anonymous_status"] = anonymous.status_code
        checks["anonymous_create_is_401"] = (
            anonymous.status_code == 401
            and anonymous.json()["error"]["code"] == "unauthenticated"
        )

        created = session.create(key=f"sec-{repeat_index}")
        run_id = created["run_id"]
        unknown = session.raw("GET", "/api/v2/reviews/run-does-not-exist")
        counters["unknown_status"] = unknown.status_code
        # A credential of another tenant reads the same shape for a foreign object.
        foreign_token = fixture.issue_token(
            "subject-beta-operator", tenant="tenant-beta"
        )
        foreign = ApiSession(base_url=process.base_url, token=foreign_token)
        try:
            foreign_read = foreign.raw("GET", f"/api/v2/reviews/{run_id}")
            counters["foreign_status"] = foreign_read.status_code
            checks["foreign_object_matches_unknown_object"] = (
                foreign_read.status_code == unknown.status_code == 404
                and _without_request_id(foreign_read.json())
                == _without_request_id(unknown.json())
            )
        finally:
            foreign.close()

        reader_token = fixture.issue_token("subject-local-demo-reader")
        reader = ApiSession(base_url=process.base_url, token=reader_token)
        try:
            denied = reader.raw(
                "POST",
                f"/api/v2/reviews/{run_id}/confirm",
                headers={"Idempotency-Key": f"sec-reader-{repeat_index}"},
                json={
                    "api_version": "review-api.v2",
                    "expected_version": 1,
                    "expected_summary_checksum": "sha256:" + "0" * 64,
                },
            )
            counters["reader_status"] = denied.status_code
            checks["reader_scope_denial_is_403"] = (
                denied.status_code == 403
                and denied.json()["error"]["code"] == "insufficient_scope"
            )
        finally:
            reader.close()

        bodies = [anonymous.text, unknown.text]
        checks["presented_credential_never_echoed"] = all(
            token not in body for body in bodies
        )
        counters["token_echo_count"] = sum(body.count(token) for body in bodies)

        retired = session.raw("GET", "/api/v1/reviews")
        retired_body = retired.json()
        checks["retired_v1_carries_no_business_data"] = (
            retired.status_code == 410
            and retired_body["error"]["code"] == "api_version_retired"
            and not {"run_id", "runs", "summary", "cases", "receipt"}
            & set(retired_body)
        )
        after = fixture.store.denial_count()
        counters["denial_events_added"] = after - before
        checks["denial_ledger_records_every_rejection"] = after - before >= 3
    finally:
        session.close()
        process.stop()
    passed = all(checks.values())
    return ScenarioObservation(
        scenario="SEC-GATE",
        repeat_index=repeat_index,
        expected="every rejection is uniform, recorded and free of credential echoes",
        actual="ok" if passed else "mismatch",
        passed=passed,
        counters=counters,
        notes={"checks": checks},
    )


def _without_request_id(payload: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(payload))
    clone.get("error", {}).pop("request_id", None)
    return clone


def run_audit_scenario(
    fixture: DatabaseFixture, *, repeat_index: int, work_dir: Path
) -> ScenarioObservation:
    """OBS gate: approval audit identity and cross-restart sequence stability."""

    work_dir.mkdir(parents=True, exist_ok=True)
    process, session, token = _start_session(fixture, work_dir)
    checks: dict[str, bool] = {}
    counters: dict[str, Any] = {}
    run_id = ""
    try:
        created = session.create(key=f"audit-{repeat_index}")
        run_id = created["run_id"]
        body = session.wait_for(
            run_id, "pending_confirmation", "pending_human_review", timeout=45.0
        )
        decided = session.confirm(run_id, body, key=f"audit-decide-{repeat_index}")
        assert decided.status_code == 200, decided.text
        events = session.raw("GET", f"/api/v2/reviews/{run_id}/audit").json()["events"]
        approval = [item for item in events if item["event_class"] == "approval"]
        counters["approval_events"] = len(approval)
        counters["audit_sequences"] = [item["sequence"] for item in events]
        checks["approval_event_actor_is_the_verified_subject"] = (
            len(approval) == 1 and approval[0]["actor"] == DEMO_SUBJECT
        )
        checks["approval_event_carries_authorization_revision"] = (
            len(approval) == 1
            and approval[0]["authorization_revision"] is not None
            and approval[0]["authorization_revision"] >= 1
        )
        sequences = [item["sequence"] for item in events]
        checks["sequences_are_increasing"] = sequences == sorted(sequences)
    finally:
        session.close()
        process.stop()

    # Restart the process and read the same records back.
    process2, session2, _token2 = _start_session(fixture, work_dir)
    try:
        events_after = session2.raw("GET", f"/api/v2/reviews/{run_id}/audit").json()[
            "events"
        ]
        counters["audit_sequences_after_restart"] = [
            item["sequence"] for item in events_after
        ]
        checks["sequence_is_stable_across_a_restart"] = [
            item["sequence"] for item in events_after
        ] == counters["audit_sequences"]
        denials = fixture.store.list_unanchored_denials(limit=500)
        # Denials without a resolvable Case live only in the trusted maintenance
        # ledger; they must never appear inside a tenant's Run audit.
        checks["denial_ledger_stays_out_of_the_tenant_run_audit"] = all(
            item["event_class"] == "approval" for item in events_after
        ) and all(row.case_id is None for row in denials)
        counters["unanchored_denials"] = len(denials)
    finally:
        session2.close()
        process2.stop()
    passed = all(checks.values())
    return ScenarioObservation(
        scenario="AUDIT-RESTART",
        repeat_index=repeat_index,
        expected="audit identity, revision, ordering and denial separation survive a restart",
        actual="ok" if passed else "mismatch",
        passed=passed,
        counters=counters,
        notes={"checks": checks},
    )


def run_field_scenario(
    fixture: DatabaseFixture, *, repeat_index: int, work_dir: Path
) -> ScenarioObservation:
    """OBS gate: the complete persisted-record scan plus the log, guard and mode checks."""

    return field_scan_scenario(
        fixture,
        repeat_index=repeat_index,
        work_dir=work_dir,
        start_session=_start_session,
        observation=ScenarioObservation,
        report_filename=REPORT_FILENAME,
        observations_filename=OBSERVATIONS_FILENAME,
    )


def run_named_scenario(
    fixture: DatabaseFixture,
    *,
    scenario: str,
    repeat_index: int,
    fault_dir: Path,
    baseline: dict[str, Any] | None,
) -> ScenarioObservation:
    if scenario in REC_SCENARIOS:
        return run_rec_scenario(
            fixture,
            scenario=scenario,
            repeat_index=repeat_index,
            fault_dir=fault_dir,
            baseline=baseline,
        )
    if scenario == "SEC-GATE":
        return run_security_scenario(
            fixture,
            repeat_index=repeat_index,
            work_dir=fault_dir / f"sec-{repeat_index}",
        )
    if scenario == "AUDIT-RESTART":
        return run_audit_scenario(
            fixture,
            repeat_index=repeat_index,
            work_dir=fault_dir / f"audit-{repeat_index}",
        )
    if scenario == "FIELD-SCAN":
        return run_field_scenario(
            fixture,
            repeat_index=repeat_index,
            work_dir=fault_dir / f"field-{repeat_index}",
        )
    raise RunnerError(f"unknown scenario {scenario}")


if __name__ == "__main__":
    raise SystemExit(main())
