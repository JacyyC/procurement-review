"""The complete persisted-record field scan (FIX-03B, AC-05).

The offline scanner judges every record the product persists by the same closed, typed
rules the write path applies, and it does so from an explicit inventory rather than from a
convenient subset:

* the inventory is declared as data — one entry per record class, naming the table, the
  JSON column that carries the payload and the kind whose contract governs it;
* every class reports its own numbers: the rows the table actually holds, the rows that
  were scanned and the rows that were refused, plus the declared reason when a class
  legitimately holds no row;
* a missing table, a declared class that scans fewer rows than the table holds, an
  unknown record kind and a raised validation error are all explicit failures, never a
  quiet zero.

Rows are judged by the rules in :mod:`.record_rows` — the same functions an online reader
calls, on the same complete row with all of its columns.  A scan therefore cannot disagree
with a read, and a read cannot look cleaner than the scan of the same bytes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import text

from .record_guard import RECORD_MODELS
from .record_inventory import (
    DERIVED_COLUMNS,
    RECORD_SCAN_PLAN,
    SCANNED_TABLES,
    ScanClass,
)
from .record_rows import (
    AUDIT_COLUMN_KEYS,
    COLUMN_MISMATCH,
    TRACE_COLUMN_KEYS,
    class_findings,
    row_mapping,
)
from .run_errors import RepositoryError

SCAN_VERSION = "review-field-scan.v1"


class FieldScanError(RepositoryError):
    """The scan could not cover what the plan declares."""


@dataclass(slots=True)
class ClassScan:
    """What one declared class actually saw."""

    name: str
    table: str
    kind: str
    rows: int = 0
    scanned: int = 0
    refused: int = 0
    categories: list[str] = field(default_factory=list)
    zero_row_reason: str | None = None

    @property
    def covered(self) -> bool:
        return self.scanned == self.rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "table": self.table,
            "kind": self.kind,
            "rows": self.rows,
            "scanned": self.scanned,
            "refused": self.refused,
            "categories": sorted(set(self.categories)),
            "zero_row_reason": self.zero_row_reason if self.rows == 0 else None,
        }


@dataclass(slots=True)
class FieldScanReport:
    """The whole scan: one entry per class plus the coverage verdict."""

    classes: list[ClassScan] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    tables: dict[str, int] = field(default_factory=dict)

    @property
    def scanned(self) -> int:
        return sum(item.scanned for item in self.classes)

    @property
    def refused(self) -> int:
        return sum(item.refused for item in self.classes)

    @property
    def clean(self) -> bool:
        return not self.problems and self.refused == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "scan_version": SCAN_VERSION,
            "tables": dict(sorted(self.tables.items())),
            "classes": [item.as_dict() for item in self.classes],
            "scanned": self.scanned,
            "refused": self.refused,
            "problems": sorted(self.problems),
        }


def table_row_counts(connection: Any) -> dict[str, int]:
    """The committed row count of every declared table, or an explicit failure."""

    counts: dict[str, int] = {}
    present = {
        str(row[0])
        for row in connection.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
        ).all()
    }
    missing = [table for table in SCANNED_TABLES if table not in present]
    if missing:
        raise FieldScanError("declared tables are missing: " + ", ".join(missing))
    for table in SCANNED_TABLES:
        counts[table] = int(
            connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
        )
    return counts


def scan_persisted_records(
    connection: Any,
    *,
    run_id: str | None = None,
    plan: Sequence[ScanClass] = RECORD_SCAN_PLAN,
) -> FieldScanReport:
    """Scan every class in the plan and report its own numbers.

    ``run_id`` scopes the scan to one Run (the live scenario); without it the whole
    database is scanned (the offline inventory).  A class whose row count does not match
    what it scanned, a raised validation error and an unknown record kind are all recorded
    as problems, so a scan can never silently cover less than it declares.
    """

    report = FieldScanReport()
    report.tables = table_row_counts(connection)
    for item in plan:
        if item.kind not in RECORD_MODELS:
            report.problems.append(f"{item.name}: no closed contract for {item.kind}")
            continue
        entry = ClassScan(
            name=item.name,
            table=item.table,
            kind=item.kind,
            zero_row_reason=item.zero_row_reason,
        )
        scope = ""
        parameters: dict[str, Any] = {}
        if run_id is not None:
            scope = " WHERE run_id = :run_id"
            parameters["run_id"] = run_id
        entry.rows = int(
            connection.execute(
                text(f"SELECT count(*) FROM {item.table}{scope}"), parameters
            ).scalar_one()
        )
        order = " ORDER BY 1"
        rows = connection.execute(
            text(f"SELECT * FROM {item.table}{scope}{item.row_filter}{order}"),
            parameters,
        ).all()
        for row in rows:
            entry.scanned += 1
            # One shared rule per row: the same judgment an online reader applies, over
            # every column and every JSON payload the row carries.
            findings = class_findings(item, row_mapping(row))
            if findings:
                entry.refused += 1
                entry.categories.extend(findings)
        if not entry.covered:
            report.problems.append(
                f"{item.name}: scanned {entry.scanned} of {entry.rows} rows"
            )
        report.classes.append(entry)
    return report


def coverage_problems(
    report: FieldScanReport, plan: Sequence[ScanClass] = RECORD_SCAN_PLAN
) -> list[str]:
    """Declared classes the report does not cover, and classes it covers undeclared."""

    covered = {item.name for item in report.classes}
    declared = {item.name for item in plan}
    return [f"missing class: {name}" for name in sorted(declared - covered)] + [
        f"undeclared class: {name}" for name in sorted(covered - declared)
    ]


def require_complete_scan(report: FieldScanReport) -> None:
    """Fail closed when the scan did not cover what it declares."""

    if report.problems:
        raise FieldScanError("; ".join(sorted(report.problems)))


def field_scan_scenario(
    fixture: Any,
    *,
    repeat_index: int,
    work_dir: Path,
    start_session: Callable[[Any, Path], tuple[Any, Any, str]],
    observation: Callable[..., Any],
    report_filename: str,
    observations_filename: str,
) -> Any:
    """OBS gate: sensitive-pattern scans, the complete record scan and file modes.

    The scenario keeps its original assertions — the API log carries no sensitive pattern,
    the scanner still detects every synthetic canary, the write guard still refuses a
    poisoned payload and the artifact permissions hold — and adds the complete persisted
    scan: every declared class reports its own row, scanned and refused counts, a class
    that scanned fewer rows than it holds fails the scenario, and a zero-row class has to
    carry its declared reason.

    The runner's own helpers arrive as arguments, so this module never imports the suite
    that calls it and one process cannot end up with two copies of that suite.
    """

    from .field_guard import canary_values, scan_text
    from .record_guard import guard_record
    from .review_repository import FieldGuardError, guard_payload

    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir.parent
    process, session, token = start_session(fixture, work_dir)
    checks: dict[str, bool] = {}
    counters: dict[str, Any] = {}
    run_id = ""
    try:
        created = session.create(key=f"field-{repeat_index}")
        run_id = created["run_id"]
        body = session.wait_for(
            run_id, "pending_confirmation", "pending_human_review", timeout=45.0
        )
        # A decision records the receipt, so the scan covers that class with a real row
        # instead of an admitted zero.
        decided = session.confirm(run_id, body, key=f"field-confirm-{repeat_index}")
        assert decided.status_code == 200, decided.text
    finally:
        session.close()
        process.stop()

    log_text = Path(process.log_path).read_text(encoding="utf-8", errors="replace")
    log_report = scan_text(log_text, location="api_log")
    counters["api_log_findings"] = len(log_report.findings)
    checks["no_sensitive_pattern_in_api_log"] = log_report.clean

    with fixture.store.transaction() as connection:
        scan = scan_persisted_records(connection, run_id=run_id)
    counters["field_scan"] = scan.as_dict()
    counters["persisted_records_scanned"] = scan.scanned
    counters["persisted_value_findings"] = scan.refused
    counters["field_scan_problems"] = sorted(scan.problems)
    zero_rows = [
        item.name for item in scan.classes if item.rows == 0 and item.zero_row_reason
    ]
    counters["field_scan_zero_rows"] = sorted(zero_rows)
    counters["field_scan_classes"] = len(scan.classes)
    checks["field_scan_covers_every_declared_class"] = (
        len(scan.classes) == len(RECORD_SCAN_PLAN)
        and not coverage_problems(scan)
        and not scan.problems
    )
    checks["field_scan_scanned_at_least_one_row"] = scan.scanned > 0
    checks["every_zero_row_class_declares_its_reason"] = all(
        item.rows > 0 or item.zero_row_reason for item in scan.classes
    )
    checks["no_forbidden_key_in_persisted_payloads"] = scan.refused == 0

    detected = 0
    for category, canary in canary_values():
        report = scan_text(f"payload={canary}", location="canary")
        if any(item.category == category for item in report.findings):
            detected += 1
    counters["canaries_detected"] = detected
    counters["canaries_total"] = len(canary_values())
    checks["canaries_are_detected_by_the_scanner"] = detected == len(canary_values())

    rejected = 0
    for poisoned in (
        {"human_actor": "x", "idempotency_key": "raw-key"},
        {"note": "Bearer canarytoken0000000000000000"},
        {"note": "postgresql://canary:canary@127.0.0.1:5432/canary"},
    ):
        try:
            guard_payload(poisoned, location="poisoned")
        except FieldGuardError:
            rejected += 1
    counters["poisoned_payloads_rejected"] = rejected
    checks["write_guard_rejects_a_poisoned_payload"] = rejected == 3

    refused_record = 0
    for kind, payload in (
        (
            "trace",
            {"event_id": "event-run-" + "0" * 32 + "-0001", "error": "13800001234"},
        ),
        ("run", {"run_id": "13800001234"}),
    ):
        try:
            guard_record(kind, payload, location="poisoned")
        except FieldGuardError:
            refused_record += 1
    counters["poisoned_records_rejected"] = refused_record
    checks["record_guard_rejects_a_poisoned_record"] = refused_record == 2

    report_file = output / report_filename
    modes = {
        "output_dir": oct(output.stat().st_mode & 0o777),
        "observations": oct((output / observations_filename).stat().st_mode & 0o777)
        if (output / observations_filename).exists()
        else "missing",
        "report": oct(report_file.stat().st_mode & 0o777)
        if report_file.exists()
        else "missing",
    }
    counters["modes"] = modes
    checks["artifact_permissions_are_0700_and_0600"] = modes[
        "output_dir"
    ] == "0o700" and (modes["report"] in {"0o600", "missing"})
    counters["credential_presented"] = bool(token)
    passed = all(checks.values())
    return observation(
        scenario="FIELD-SCAN",
        repeat_index=repeat_index,
        expected=(
            "no sensitive value or forbidden key in logs/records, the complete record "
            "scan covers every declared class, guard and modes hold"
        ),
        actual="ok" if passed else "mismatch",
        passed=passed,
        counters=counters,
        notes={"checks": checks},
    )


__all__ = [
    "AUDIT_COLUMN_KEYS",
    "COLUMN_MISMATCH",
    "DERIVED_COLUMNS",
    "RECORD_SCAN_PLAN",
    "SCANNED_TABLES",
    "SCAN_VERSION",
    "TRACE_COLUMN_KEYS",
    "ClassScan",
    "FieldScanError",
    "FieldScanReport",
    "ScanClass",
    "coverage_problems",
    "require_complete_scan",
    "field_scan_scenario",
    "scan_persisted_records",
    "table_row_counts",
]
