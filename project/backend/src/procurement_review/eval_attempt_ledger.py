"""Durable logical-task/physical-attempt ledger for V1-07 evaluation runs."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .eval_attempt_schema import SCHEMA_SQL
from .eval_attempt_manifest import data_versions, read_manifest
from .eval_attempt_budget import token_exposure
from .eval_v2_contracts import (
    AttemptReceiptV1,
    AttemptState,
    BudgetSnapshotV1,
    EvalPredictionV2,
    EvalRunManifestV2,
    TaskState,
)
from .gold_v3_contracts import payload_sha256
from .model_attempt_projection import BranchProjectionV2, load_projection

LEDGER_SCHEMA_VERSION = "eval-attempt-ledger.v1"


class LedgerError(RuntimeError):
    """The durable attempt state cannot accept the requested transition."""


class BudgetExceeded(LedgerError):
    """A physical attempt cannot be reserved within the fixed run brakes."""


class InjectedCrash(BaseException):
    """Test-only signal used to prove process-boundary recovery behavior."""


@dataclass(frozen=True, slots=True)
class ReservedAttempt:
    attempt_id: int
    task_key: str
    attempt_no: int
    request_sha256: str
    reserved_input_tokens: int
    reserved_output_tokens: int


@dataclass(frozen=True, slots=True)
class RecoverySummary:
    sent_marked_remote_unknown: int
    reserved_ready_to_send: int
    committed_reusable: int


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class AttemptLedger:
    """One SQLite-backed evaluation ledger with monotonic reservations."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        if ".local" not in self.path.parts:
            raise LedgerError("evaluation ledgers must live under a .local directory")
        if not self.path.is_file():
            raise LedgerError("evaluation ledger does not exist")
        self._assert_identity()

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        run_id: str,
        gold_manifest_sha256: str,
        configuration_sha256: str,
        max_requests: int,
        max_input_tokens: int | None,
        max_output_tokens: int | None,
        stage_id: str = "V1-07B",
        gold_set_version: str = "gold-set.v3",
    ) -> AttemptLedger:
        versions = data_versions(gold_set_version)
        # Validate the same contract used on readback before any filesystem or DB
        # mutation. Otherwise an invalid Stage/identity/budget leaves an unreadable
        # but apparently successfully created ledger behind.
        created_at = _utc_now()
        EvalRunManifestV2(
            **versions,
            stage_id=stage_id,
            run_id=run_id,
            gold_manifest_sha256=gold_manifest_sha256,
            configuration_sha256=configuration_sha256,
            created_at=datetime.fromisoformat(created_at),
            task_count=0,
            committed_task_count=0,
            blocked_unknown_task_count=0,
            attempt_count=0,
            budget=BudgetSnapshotV1(
                max_requests=max_requests,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                reserved_requests=0,
                reserved_input_tokens=0,
                reserved_output_tokens=0,
                known_input_tokens=0,
                known_output_tokens=0,
                unknown_usage_attempts=0,
            ),
        )
        resolved = path.resolve()
        if ".local" not in resolved.parts:
            raise LedgerError("evaluation ledgers must live under a .local directory")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.parent.chmod(0o700)
        if resolved.exists():
            raise LedgerError("evaluation ledger already exists")
        connection = sqlite3.connect(resolved)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(SCHEMA_SQL)
            metadata = {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "run_id": run_id,
                "gold_manifest_sha256": gold_manifest_sha256,
                "configuration_sha256": configuration_sha256,
                "created_at": created_at,
                "stage_id": stage_id,
                **versions,
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
            )
            connection.execute(
                """
                INSERT INTO budget(
                    singleton, max_requests, max_input_tokens, max_output_tokens
                ) VALUES (1, ?, ?, ?)
                """,
                (max_requests, max_input_tokens, max_output_tokens),
            )
            connection.commit()
        except BaseException:
            connection.close()
            with contextlib.suppress(OSError):
                resolved.unlink()
            raise
        finally:
            with contextlib.suppress(Exception):
                connection.close()
        resolved.chmod(0o600)
        return cls(resolved)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _metadata(self, connection: sqlite3.Connection | None = None) -> dict[str, str]:
        owns = connection is None
        current = connection or self._connect()
        try:
            return {
                row["key"]: row["value"]
                for row in current.execute("SELECT key, value FROM metadata")
            }
        finally:
            if owns:
                current.close()

    def _assert_identity(self) -> None:
        metadata = self._metadata()
        if metadata.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise LedgerError("unknown evaluation ledger version")

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        event_type: str,
        *,
        task_key: str | None = None,
        attempt_id: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO events(event_type, task_key, attempt_id, occurred_at, detail_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_type, task_key, attempt_id, _utc_now(), _json(detail or {})),
        )

    def register_task(self, task_key: str, payload_hash: str) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT payload_sha256 FROM tasks WHERE task_key = ?", (task_key,)
            ).fetchone()
            if row is not None:
                if row["payload_sha256"] != payload_hash:
                    raise LedgerError("task key was reused with a different payload")
                return
            now = _utc_now()
            connection.execute(
                """
                INSERT INTO tasks(task_key, payload_sha256, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (task_key, payload_hash, TaskState.PENDING.value, now, now),
            )
            self._event(connection, "task_registered", task_key=task_key)

    def _budget_row(self, connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM budget WHERE singleton = 1").fetchone()
        if row is None:
            raise LedgerError("budget row is missing")
        return row

    def reserve_attempt(
        self,
        task_key: str,
        *,
        request_sha256: str,
        reserve_input_tokens: int,
        reserve_output_tokens: int,
        max_attempts: int = 2,
    ) -> ReservedAttempt:
        if type(max_attempts) is not int or max_attempts not in {1, 2}:
            raise LedgerError("physical attempt limit must be one or two")
        if reserve_input_tokens < 0 or reserve_output_tokens < 0:
            raise LedgerError("attempt reservations cannot be negative")
        with self._transaction() as connection:
            task = connection.execute(
                "SELECT * FROM tasks WHERE task_key = ?", (task_key,)
            ).fetchone()
            if task is None:
                raise LedgerError("logical task is not registered")
            if task["state"] == TaskState.COMMITTED.value:
                raise LedgerError("logical task is already committed")
            if task["state"] == TaskState.BLOCKED_UNKNOWN.value:
                raise LedgerError("logical task has an unknown remote result")
            active = connection.execute(
                """
                SELECT * FROM attempts
                WHERE task_key = ? AND state IN (?, ?)
                ORDER BY attempt_no DESC LIMIT 1
                """,
                (task_key, AttemptState.RESERVED.value, AttemptState.SENT.value),
            ).fetchone()
            if active is not None:
                if active["state"] == AttemptState.SENT.value:
                    raise LedgerError(
                        "sent attempt must be recovered before another send"
                    )
                if (
                    active["request_sha256"],
                    active["reserved_input_tokens"],
                    active["reserved_output_tokens"],
                ) != (request_sha256, reserve_input_tokens, reserve_output_tokens):
                    raise LedgerError("existing reservation does not match request")
                return self._reserved(active)
            previous = connection.execute(
                "SELECT COALESCE(MAX(attempt_no), 0) AS value FROM attempts WHERE task_key = ?",
                (task_key,),
            ).fetchone()["value"]
            attempt_no = int(previous) + 1
            if attempt_no > max_attempts:
                raise LedgerError("physical attempt budget exhausted")
            budget = self._budget_row(connection)
            input_exposure, output_exposure = token_exposure(connection)
            request_total = budget["reserved_requests"] + 1
            input_total = (
                max(budget["reserved_input_tokens"], input_exposure)
                + reserve_input_tokens
            )
            output_total = (
                max(budget["reserved_output_tokens"], output_exposure)
                + reserve_output_tokens
            )
            if request_total > budget["max_requests"]:
                raise BudgetExceeded("request budget exhausted")
            if (
                budget["max_input_tokens"] is not None
                and input_total > budget["max_input_tokens"]
            ):
                raise BudgetExceeded("input token budget exhausted")
            if (
                budget["max_output_tokens"] is not None
                and output_total > budget["max_output_tokens"]
            ):
                raise BudgetExceeded("output token budget exhausted")
            now = _utc_now()
            cursor = connection.execute(
                """
                INSERT INTO attempts(
                    task_key, attempt_no, request_sha256, state,
                    reserved_input_tokens, reserved_output_tokens, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_key,
                    attempt_no,
                    request_sha256,
                    AttemptState.RESERVED.value,
                    reserve_input_tokens,
                    reserve_output_tokens,
                    now,
                ),
            )
            attempt_id = int(cursor.lastrowid)
            connection.execute(
                """
                UPDATE budget SET reserved_requests = ?, reserved_input_tokens = ?,
                    reserved_output_tokens = ? WHERE singleton = 1
                """,
                (request_total, input_total, output_total),
            )
            connection.execute(
                "UPDATE tasks SET state = ?, updated_at = ? WHERE task_key = ?",
                (TaskState.IN_PROGRESS.value, now, task_key),
            )
            self._event(
                connection,
                "attempt_reserved",
                task_key=task_key,
                attempt_id=attempt_id,
                detail={
                    "attempt_no": attempt_no,
                    "reserve_input_tokens": reserve_input_tokens,
                    "reserve_output_tokens": reserve_output_tokens,
                },
            )
            return ReservedAttempt(
                attempt_id,
                task_key,
                attempt_no,
                request_sha256,
                reserve_input_tokens,
                reserve_output_tokens,
            )

    @staticmethod
    def _reserved(row: sqlite3.Row) -> ReservedAttempt:
        return ReservedAttempt(
            attempt_id=row["attempt_id"],
            task_key=row["task_key"],
            attempt_no=row["attempt_no"],
            request_sha256=row["request_sha256"],
            reserved_input_tokens=row["reserved_input_tokens"],
            reserved_output_tokens=row["reserved_output_tokens"],
        )

    def mark_sent(self, attempt_id: int) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT task_key, state FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            if row is None or row["state"] != AttemptState.RESERVED.value:
                raise LedgerError("only a reserved attempt can be sent")
            now = _utc_now()
            connection.execute(
                "UPDATE attempts SET state = ?, sent_at = ? WHERE attempt_id = ?",
                (AttemptState.SENT.value, now, attempt_id),
            )
            self._event(
                connection,
                "attempt_sent",
                task_key=row["task_key"],
                attempt_id=attempt_id,
            )

    @staticmethod
    def _invoke_fault(fault: Callable[[str], None] | None, boundary: str) -> None:
        if fault is not None:
            fault(boundary)

    def commit_success(
        self,
        attempt_id: int,
        *,
        receipt: AttemptReceiptV1,
        projection: EvalPredictionV2,
        fault: Callable[[str], None] | None = None,
    ) -> None:
        if receipt.outcome != "succeeded":
            raise LedgerError("success commit requires a succeeded receipt")
        self.commit_result(
            attempt_id, receipt=receipt, projection=projection, fault=fault
        )

    def commit_result(
        self,
        attempt_id: int,
        *,
        receipt: AttemptReceiptV1,
        projection: EvalPredictionV2 | BranchProjectionV2,
        fault: Callable[[str], None] | None = None,
    ) -> None:
        if receipt.outcome not in {"succeeded", "failed", "cancelled"}:
            raise LedgerError("terminal result requires a completed attempt")
        if receipt.outcome != "succeeded" and not projection.pipeline_failure:
            raise LedgerError("failed result must retain pipeline failure")
        receipt_json = _json(receipt.model_dump(mode="json"))
        projection_json = _json(projection.model_dump(mode="json"))
        receipt_hash = payload_sha256(receipt.model_dump(mode="json"))
        projection_hash = payload_sha256(projection.model_dump(mode="json"))
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if row is None or row["state"] != AttemptState.SENT.value:
                raise LedgerError("only a sent attempt can commit")
            if (
                receipt.task_key != row["task_key"]
                or receipt.attempt_no != row["attempt_no"]
            ):
                raise LedgerError("receipt identity does not match attempt")
            if receipt.request_sha256 != row["request_sha256"]:
                raise LedgerError("receipt request hash does not match attempt")
            task = connection.execute(
                "SELECT state FROM tasks WHERE task_key = ?", (row["task_key"],)
            ).fetchone()
            if task["state"] == TaskState.COMMITTED.value:
                raise LedgerError("logical task was already committed")
            self._invoke_fault(fault, "before_atomic_commit")
            now = _utc_now()
            connection.execute(
                """
                UPDATE attempts SET receipt_json = ?, receipt_sha256 = ?
                WHERE attempt_id = ?
                """,
                (receipt_json, receipt_hash, attempt_id),
            )
            self._invoke_fault(fault, "after_receipt_write")
            connection.execute(
                """
                UPDATE attempts SET projection_json = ?, projection_sha256 = ?
                WHERE attempt_id = ?
                """,
                (projection_json, projection_hash, attempt_id),
            )
            self._invoke_fault(fault, "after_projection_write")
            connection.execute(
                """
                UPDATE attempts SET state = ?, terminal_at = ? WHERE attempt_id = ?
                """,
                (
                    AttemptState.COMMITTED.value,
                    now,
                    attempt_id,
                ),
            )
            connection.execute(
                """
                UPDATE tasks SET state = ?, committed_attempt_id = ?, projection_json = ?,
                    projection_sha256 = ?, updated_at = ? WHERE task_key = ?
                """,
                (
                    TaskState.COMMITTED.value,
                    attempt_id,
                    projection_json,
                    projection_hash,
                    now,
                    row["task_key"],
                ),
            )
            self._invoke_fault(fault, "after_commit_marker_write")
            self._record_usage(connection, receipt)
            self._event(
                connection,
                "attempt_committed",
                task_key=row["task_key"],
                attempt_id=attempt_id,
                detail={
                    "receipt_sha256": receipt_hash,
                    "projection_sha256": projection_hash,
                },
            )
            self._invoke_fault(fault, "before_transaction_commit")
        self._invoke_fault(fault, "after_atomic_commit")

    def _record_usage(
        self, connection: sqlite3.Connection, receipt: AttemptReceiptV1
    ) -> None:
        if receipt.usage_unknown:
            connection.execute(
                "UPDATE budget SET unknown_usage_attempts = unknown_usage_attempts + 1 WHERE singleton = 1"
            )
            return
        connection.execute(
            """
            UPDATE budget SET known_input_tokens = known_input_tokens + ?,
                known_output_tokens = known_output_tokens + ? WHERE singleton = 1
            """,
            (receipt.input_tokens or 0, receipt.output_tokens or 0),
        )

    def commit_failure(self, attempt_id: int, receipt: AttemptReceiptV1) -> None:
        if receipt.outcome not in {"failed", "cancelled"}:
            raise LedgerError("failure commit requires a failed/cancelled receipt")
        receipt_json = _json(receipt.model_dump(mode="json"))
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if row is None or row["state"] not in {
                AttemptState.RESERVED.value,
                AttemptState.SENT.value,
            }:
                raise LedgerError("attempt cannot transition to failed")
            if (
                receipt.task_key != row["task_key"]
                or receipt.attempt_no != row["attempt_no"]
            ):
                raise LedgerError("receipt identity does not match attempt")
            if receipt.request_sha256 != row["request_sha256"]:
                raise LedgerError("receipt request hash does not match attempt")
            state = (
                AttemptState.CANCELLED.value
                if receipt.outcome == "cancelled"
                else AttemptState.FAILED.value
            )
            now = _utc_now()
            connection.execute(
                """
                UPDATE attempts SET state = ?, receipt_json = ?, receipt_sha256 = ?,
                    terminal_at = ? WHERE attempt_id = ?
                """,
                (
                    state,
                    receipt_json,
                    payload_sha256(receipt.model_dump(mode="json")),
                    now,
                    attempt_id,
                ),
            )
            connection.execute(
                "UPDATE tasks SET state = ?, updated_at = ? WHERE task_key = ?",
                (TaskState.PENDING.value, now, row["task_key"]),
            )
            self._record_usage(connection, receipt)
            self._event(
                connection,
                f"attempt_{state}",
                task_key=row["task_key"],
                attempt_id=attempt_id,
                detail={"error_code": receipt.error_code},
            )

    def _mark_remote_unknown(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> None:
        receipt = AttemptReceiptV1(
            task_key=row["task_key"],
            attempt_no=row["attempt_no"],
            outcome="remote_unknown",
            request_sha256=row["request_sha256"],
            usage_unknown=True,
            error_code="remote_result_unknown_after_restart",
        )
        now = _utc_now()
        connection.execute(
            """
            UPDATE attempts SET state = ?, receipt_json = ?, receipt_sha256 = ?,
                terminal_at = ? WHERE attempt_id = ?
            """,
            (
                AttemptState.REMOTE_UNKNOWN.value,
                _json(receipt.model_dump(mode="json")),
                payload_sha256(receipt.model_dump(mode="json")),
                now,
                row["attempt_id"],
            ),
        )
        connection.execute(
            "UPDATE tasks SET state = ?, updated_at = ? WHERE task_key = ?",
            (TaskState.BLOCKED_UNKNOWN.value, now, row["task_key"]),
        )
        self._record_usage(connection, receipt)
        self._event(
            connection,
            "attempt_remote_unknown",
            task_key=row["task_key"],
            attempt_id=row["attempt_id"],
        )

    def recover(self) -> RecoverySummary:
        with self._transaction() as connection:
            sent = list(
                connection.execute(
                    "SELECT * FROM attempts WHERE state = ? ORDER BY attempt_id",
                    (AttemptState.SENT.value,),
                )
            )
            for row in sent:
                self._mark_remote_unknown(connection, row)
            reserved = connection.execute(
                "SELECT COUNT(*) AS value FROM attempts WHERE state = ?",
                (AttemptState.RESERVED.value,),
            ).fetchone()["value"]
            committed = connection.execute(
                "SELECT COUNT(*) AS value FROM tasks WHERE state = ?",
                (TaskState.COMMITTED.value,),
            ).fetchone()["value"]
            return RecoverySummary(len(sent), int(reserved), int(committed))

    def next_action(self, task_key: str) -> str:
        connection = self._connect()
        try:
            task = connection.execute(
                "SELECT state FROM tasks WHERE task_key = ?", (task_key,)
            ).fetchone()
            if task is None:
                return "register_task"
            if task["state"] == TaskState.COMMITTED.value:
                return "reuse_committed"
            if task["state"] == TaskState.BLOCKED_UNKNOWN.value:
                return "blocked_remote_unknown"
            active = connection.execute(
                """
                SELECT state FROM attempts WHERE task_key = ? AND state IN (?, ?)
                ORDER BY attempt_no DESC LIMIT 1
                """,
                (task_key, AttemptState.RESERVED.value, AttemptState.SENT.value),
            ).fetchone()
            if active is None:
                return "reserve_attempt"
            return (
                "send_reserved"
                if active["state"] == AttemptState.RESERVED.value
                else "recover_sent_attempt"
            )
        finally:
            connection.close()

    def load_committed_projection(
        self, task_key: str
    ) -> EvalPredictionV2 | BranchProjectionV2:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT state, projection_json FROM tasks WHERE task_key = ?",
                (task_key,),
            ).fetchone()
            if row is None or row["state"] != TaskState.COMMITTED.value:
                raise LedgerError("task has no committed projection")
            if row["projection_json"] is None:
                raise LedgerError("committed task is missing its projection")
            return load_projection(row["projection_json"])
        finally:
            connection.close()

    def budget_snapshot(self) -> BudgetSnapshotV1:
        connection = self._connect()
        try:
            row = self._budget_row(connection)
            return BudgetSnapshotV1(
                max_requests=row["max_requests"],
                max_input_tokens=row["max_input_tokens"],
                max_output_tokens=row["max_output_tokens"],
                reserved_requests=row["reserved_requests"],
                reserved_input_tokens=row["reserved_input_tokens"],
                reserved_output_tokens=row["reserved_output_tokens"],
                known_input_tokens=row["known_input_tokens"],
                known_output_tokens=row["known_output_tokens"],
                unknown_usage_attempts=row["unknown_usage_attempts"],
            )
        finally:
            connection.close()

    def manifest(self) -> EvalRunManifestV2:
        connection = self._connect()
        try:
            return read_manifest(connection)
        finally:
            connection.close()

    def export(self, output: Path) -> Path:
        from .eval_attempt_export import export_ledger

        return export_ledger(self, output)

    def audit_summary(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            task_count = connection.execute(
                "SELECT COUNT(*) AS value FROM tasks"
            ).fetchone()["value"]
            terminal = connection.execute(
                "SELECT COUNT(*) AS value FROM tasks WHERE state IN (?, ?)",
                (TaskState.COMMITTED.value, TaskState.BLOCKED_UNKNOWN.value),
            ).fetchone()["value"]
            unknown = connection.execute(
                "SELECT COUNT(*) AS value FROM attempts WHERE state = ?",
                (AttemptState.REMOTE_UNKNOWN.value,),
            ).fetchone()["value"]
            return {
                "complete": int(task_count) == int(terminal),
                "task_count": int(task_count),
                "terminal_task_count": int(terminal),
                "unknown_remote_attempts": int(unknown),
            }
        finally:
            connection.close()


__all__ = [
    "AttemptLedger",
    "BudgetExceeded",
    "InjectedCrash",
    "LEDGER_SCHEMA_VERSION",
    "LedgerError",
    "RecoverySummary",
    "ReservedAttempt",
]
