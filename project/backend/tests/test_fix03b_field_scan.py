"""FIX-03B V-03B-02 and V-03B-03: the complete offline field scan.

The scan is judged from the schema itself.  Every column of every written table has to be
carried by a declared class, every class reports the rows the table holds versus the rows
it scanned, and a class that legitimately holds no row has to say why.  Each class then
gets a polluted counterexample: one committed row is damaged and the scan has to refuse
*that* class with a category, keep the other classes clean and leave every stored byte
untouched.

The last group is the negative control the plan asks for: a deliberately omitted class, a
scan predicate that hides a row, an unexpected validation error and a missing table each
have to fail explicitly, so "four normal JSON payloads scanned clean" can never be
mistaken for complete coverage.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from sqlalchemy import text

from procurement_review.field_scan import (
    DERIVED_COLUMNS,
    RECORD_SCAN_PLAN,
    SCANNED_TABLES,
    FieldScanError,
    ScanClass,
    coverage_problems,
    scan_persisted_records,
    table_row_counts,
)

from fix03a_support import ledger_snapshot
from test_fix03b_read_guard import CANARY, commit_run
from test_fix01b_record_integrity import api_node, database

#: One poison recipe per scanned class, keyed by the class name.
CLASS_POISON: dict[str, dict[str, Any]] = {
    "run": {"column": "tenant_id", "value": CANARY, "scope": ""},
    "isolation": {"column": "isolation", "value": CANARY, "scope": ""},
    "generation": {"column": "owner_id", "value": CANARY, "scope": ""},
    "attempt": {"column": "last_error", "value": CANARY, "scope": ""},
    "checkpoint": {"column": "result_payload", "value": CANARY, "scope": ""},
    "summary": {"column": "summary_payload", "value": CANARY, "scope": ""},
    "execution": {"column": "execution_payload", "value": CANARY, "scope": ""},
    "idempotency": {"column": "failure_code", "value": CANARY, "scope": ""},
    "receipt": {"column": "human_actor", "value": CANARY, "scope": ""},
    "receipt_payload": {"column": "receipt_payload", "value": CANARY, "scope": ""},
    "trace": {"column": "worker", "value": CANARY, "scope": ""},
    "audit": {"column": "actor", "value": CANARY, "scope": ""},
    "checkpoint_columns": {"column": "contract_checksum", "value": CANARY, "scope": ""},
    "summary_columns": {"column": "human_gate_status", "value": CANARY, "scope": ""},
}

#: Which JSON path inside a payload column the value is written to, where the column is a
#: record of its own.
PAYLOAD_PATH: dict[str, str] = {
    "isolation": "purchase_order_id",
    "checkpoint": "result,outcome",
    "summary": "failure_reasons",
    "execution": "branch_results,0,failure_detail",
    "receipt_payload": "human_actor",
}


def poison_class(fixture: Any, run_id: str, name: str) -> None:
    """Damage exactly one row of one scanned class, on an independent connection."""

    recipe = CLASS_POISON[name]
    table = next(item.table for item in RECORD_SCAN_PLAN if item.name == name)
    column = recipe["column"]
    path = PAYLOAD_PATH.get(name)
    if path is None:
        statement = (
            f"UPDATE {table} SET {column} = :v "
            "WHERE run_id = :r AND ctid = ("
            f"SELECT ctid FROM {table} WHERE run_id = :r LIMIT 1)"
        )
    else:
        keys = "{" + path + "}"
        statement = (
            f"UPDATE {table} SET {column} = jsonb_set({column}, '{keys}', "
            "to_jsonb(CAST(:v AS text))) WHERE run_id = :r AND ctid = ("
            f"SELECT ctid FROM {table} WHERE run_id = :r LIMIT 1)"
        )
    with fixture.driver.transaction() as connection:
        connection.execute(text(statement), {"v": recipe["value"], "r": run_id})


def scan_rows(fixture: Any, run_id: str) -> dict[str, tuple[int, int, int]]:
    with fixture.driver.transaction() as connection:
        report = scan_persisted_records(connection, run_id=run_id)
    return {
        item.name: (item.rows, item.scanned, item.refused) for item in report.classes
    }


# ------------------------------------------------------------------ V-03B-02 coverage


def test_every_declared_class_is_scanned_with_its_own_counts() -> None:
    """V-03B-02: each class reports the rows it holds, scanned and refused."""

    with database("v_03b_02_plan") as fixture:
        node = api_node(fixture, "v03b_plan")
        try:
            run_id, _ident = commit_run(fixture, node, key="v-03b-02-plan")
            with fixture.driver.transaction() as connection:
                report = scan_persisted_records(connection, run_id=run_id)
            assert coverage_problems(report) == []
            assert report.problems == []
            assert report.refused == 0
            assert report.scanned > 0
            for entry in report.classes:
                assert entry.scanned == entry.rows, entry.as_dict()
                assert entry.rows > 0, entry.as_dict()
            assert sorted(item.name for item in report.classes) == sorted(
                item.name for item in RECORD_SCAN_PLAN
            )
        finally:
            node.close()


def test_every_persisted_column_is_carried_by_a_declared_class() -> None:
    """V-03B-02: the plan covers every column of every written table."""

    with database("v_03b_02_columns") as fixture:
        with fixture.driver.transaction() as connection:
            for table in SCANNED_TABLES:
                observed = {
                    str(row[0])
                    for row in connection.execute(
                        text(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_name = :t"
                        ),
                        {"t": table},
                    ).all()
                }
                declared = {
                    column
                    for item in RECORD_SCAN_PLAN
                    if item.table == table
                    for column in item.columns
                }
                # A column is covered when a class reads it, or when it is a declared
                # derived value.  ``SELECT *`` is what every class reads, so naming the
                # columns is what makes the inventory auditable rather than incidental.
                missing = observed - declared - set(DERIVED_COLUMNS.get(table, {}))
                assert not missing, (table, sorted(missing))


@pytest.mark.parametrize("name", sorted(CLASS_POISON))
def test_every_class_is_refused_when_its_record_is_polluted(name: str) -> None:
    """V-03B-02: a polluted class is refused with a category, and only that class."""

    with database(f"v_03b_02_{name}") as fixture:
        node = api_node(fixture, f"v03b_pollute_{name}")
        try:
            run_id, _ident = commit_run(fixture, node, key=f"v-03b-02-{name}")
            clean = scan_rows(fixture, run_id)
            assert clean[name][2] == 0
            poison_class(fixture, run_id, name)
            before = ledger_snapshot(fixture, run_id)
            with fixture.driver.transaction() as connection:
                report = scan_persisted_records(connection, run_id=run_id)
            refused = {item.name: item for item in report.classes if item.refused}
            assert name in refused, (name, report.as_dict())
            assert refused[name].categories, refused[name].as_dict()
            assert set(refused) == {name}, sorted(refused)
            # Scanning is a read: not one committed byte moved.
            assert ledger_snapshot(fixture, run_id) == before
            # The class still covered its rows; it refused one of them.
            assert refused[name].scanned == refused[name].rows
        finally:
            node.close()


def test_the_scan_reports_zero_rows_with_their_declared_reason() -> None:
    """V-03B-02: a class with no row is an admitted zero, not a quiet one."""

    with database("v_03b_02_zero") as fixture:
        node = api_node(fixture, "v03b_zero")
        ident = fixture.identity()
        try:
            # A Run that never reached a decision has no receipt and no approval audit.
            from test_fix01b_record_integrity import create_run
            from fix01a_support import settle

            run_id = create_run(node, ident, key="v-03b-02-zero")
            settle(fixture, run_id, "pending_confirmation")
            with fixture.driver.transaction() as connection:
                report = scan_persisted_records(connection, run_id=run_id)
            empty = {item.name: item for item in report.classes if item.rows == 0}
            assert set(empty) == {"receipt", "receipt_payload", "audit"}, sorted(empty)
            for item in empty.values():
                assert item.zero_row_reason, item.as_dict()
                assert item.as_dict()["zero_row_reason"] == item.zero_row_reason
            assert report.refused == 0 and report.problems == []
        finally:
            node.close()


def test_a_global_scan_covers_every_run() -> None:
    """V-03B-02: without a Run scope the whole database is scanned."""

    with database("v_03b_02_global") as fixture:
        node = api_node(fixture, "v03b_global")
        try:
            first, _ident = commit_run(fixture, node, key="v-03b-02-global-a")
            second, _ident = commit_run(fixture, node, key="v-03b-02-global-b")
            with fixture.driver.transaction() as connection:
                scoped = scan_persisted_records(connection, run_id=first)
                whole = scan_persisted_records(connection)
            assert whole.scanned > scoped.scanned
            assert whole.refused == 0 and whole.problems == []
            with fixture.driver.transaction() as connection:
                tables = table_row_counts(connection)
            assert tables["review_run"] == 2
            assert whole.tables["review_run"] == 2
            assert second != first
        finally:
            node.close()


# ------------------------------------------------------------------ negative controls


def test_a_deliberately_omitted_class_fails_the_coverage_check() -> None:
    """V-03B-02 control: scanning four classes instead of twelve is a failure."""

    with database("v_03b_02_omit") as fixture:
        node = api_node(fixture, "v03b_omit")
        try:
            run_id, _ident = commit_run(fixture, node, key="v-03b-02-omit")
            partial = tuple(
                item for item in RECORD_SCAN_PLAN if item.name in {"summary", "audit"}
            )
            with fixture.driver.transaction() as connection:
                report = scan_persisted_records(connection, run_id=run_id, plan=partial)
            problems = coverage_problems(report)
            assert problems
            assert any("trace" in item for item in problems)
            assert any("receipt" in item for item in problems)
            # The same scan judged against its own plan looks complete, which is exactly
            # why the review is against the declared inventory and not against itself.
            assert coverage_problems(report, partial) == []
        finally:
            node.close()


def test_a_scan_predicate_that_hides_a_row_fails_the_coverage_check() -> None:
    """V-03B-02 control: skipping a bad row is detected, not tolerated."""

    with database("v_03b_02_skip") as fixture:
        node = api_node(fixture, "v03b_skip")
        try:
            run_id, _ident = commit_run(fixture, node, key="v-03b-02-skip")
            degraded = tuple(
                dataclasses.replace(item, row_filter=" AND sequence > 1")
                if item.name == "audit"
                else item
                for item in RECORD_SCAN_PLAN
            )
            with fixture.driver.transaction() as connection:
                report = scan_persisted_records(
                    connection, run_id=run_id, plan=degraded
                )
            assert any("audit: scanned" in item for item in report.problems), (
                report.problems
            )
        finally:
            node.close()


def test_a_validation_error_is_counted_instead_of_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V-03B-02 control: an unexpected scan error refuses the row, it never passes it."""

    from procurement_review import field_scan, record_guard

    def exploding(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("scanner exploded")

    with database("v_03b_02_raise") as fixture:
        node = api_node(fixture, "v03b_raise")
        try:
            run_id, _ident = commit_run(fixture, node, key="v-03b-02-raise")
            monkeypatch.setattr(record_guard, "scan_persisted", exploding)
            with fixture.driver.transaction() as connection:
                report = scan_persisted_records(connection, run_id=run_id)
            assert report.refused == report.scanned
            assert all(
                any(item.startswith("scan_error:") for item in entry.categories)
                for entry in report.classes
            )
            assert field_scan.SCAN_VERSION == "review-field-scan.v1"
        finally:
            node.close()


def test_a_missing_table_fails_the_scan_explicitly() -> None:
    """V-03B-02 control: a declared table that is gone is an explicit failure."""

    with database("v_03b_02_missing") as fixture:
        with fixture.driver.transaction() as connection:
            connection.execute(text("DROP TABLE review_audit_event"))
        with fixture.driver.transaction() as connection:
            with pytest.raises(FieldScanError) as raised:
                scan_persisted_records(connection)
        assert "review_audit_event" in str(raised.value)


def test_the_scan_plan_is_the_declared_inventory() -> None:
    """V-03B-02: the plan is data a reader can audit, not a code path to trust."""

    names = [item.name for item in RECORD_SCAN_PLAN]
    assert len(names) == len(set(names))
    assert set(SCANNED_TABLES) == {item.table for item in RECORD_SCAN_PLAN}
    for item in RECORD_SCAN_PLAN:
        assert isinstance(item, ScanClass)
        assert item.kind
        assert item.columns
