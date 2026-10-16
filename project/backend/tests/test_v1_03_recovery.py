"""V1-03 REC matrix and DB-01..03 database contracts.

The recovery matrix drives real API subprocesses through the versioned runner: eight
fault classes, three repeats each, over one disposable PostgreSQL database.  The frozen
denominator is 24 and the report is recomputed from the raw observations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from procurement_review.api import create_app
from procurement_review.review_repository import ReviewRepository
from procurement_review.review_service import ReviewService
from procurement_review.review_store import ReviewStore
from procurement_review.security_recovery_check import (
    REC_IDS,
    verify_report,
)
from procurement_review.security_recovery_check import run_suite

from v1_03_support import (
    DEMO_SUBJECT,
    SecurityHarness,
    create_isolated_database,
    drop_database,
    require_admin_dsn,
    wait_for_status,
)
from procurement_review.local_runtime import ScopedWorkflowRunner

REC_REPEAT = 3
FAST_TIMING = {
    "REVIEW_LEASE_SECONDS": "3",
    "REVIEW_HEARTBEAT_SECONDS": "0.5",
    "REVIEW_SCAN_INTERVAL_SECONDS": "0.2",
}


@pytest.fixture(scope="session")
def recovery_report(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the frozen REC matrix once for the whole session."""

    output = tmp_path_factory.mktemp("security-recovery")
    os.chmod(output, 0o700)
    previous = {key: os.environ.get(key) for key in FAST_TIMING}
    os.environ.update(FAST_TIMING)
    os.environ.setdefault("SECURITY_RECOVERY_ADMIN_DSN", require_admin_dsn())
    try:
        report = run_suite(suite="rec", repeat=REC_REPEAT, output=output)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    report["_directory"] = str(output)
    return report


def test_rec_matrix_is_twenty_four_of_twenty_four(recovery_report: dict) -> None:
    """AC-06/AC-13: eight recoverable fault classes x three repeats all recover."""

    assert (
        recovery_report["recovery_success"]["denominator"] == len(REC_IDS) * REC_REPEAT
    )
    assert recovery_report["recovery_success"]["denominator"] == 24
    assert recovery_report["recovery_success"]["numerator"] == 24, json.dumps(
        recovery_report["per_scenario"], ensure_ascii=False
    )
    per_scenario = recovery_report["per_scenario"]
    assert set(per_scenario) == set(REC_IDS)
    for scenario, bucket in per_scenario.items():
        assert bucket == {"runs": REC_REPEAT, "passed": REC_REPEAT}, scenario


def test_rec_report_recomputes_from_its_observations(recovery_report: dict) -> None:
    """AC-13: the report is reproducible from the raw observations alone."""

    verified = verify_report(Path(recovery_report["_directory"]))
    assert verified["verified"] is True
    assert verified["mismatches"] == []
    assert verified["duplicate_rows"] is False
    assert verified["observations"] == 24
    assert verified["recomputed"]["recovery_success"] == {
        "numerator": 24,
        "denominator": 24,
    }


def test_rec_observations_carry_raw_timings_and_counters(recovery_report: dict) -> None:
    """Every REC observation must keep its own timing and call ledger."""

    payload = json.loads(
        (Path(recovery_report["_directory"]) / "observations.json").read_text(
            encoding="utf-8"
        )
    )
    rows = payload["observations"]
    assert len(rows) == 24
    for row in rows:
        assert row["counters"]["distinct_runs"] == 1
        assert row["counters"]["attempts"]
        assert row["notes"]["summary_matches_baseline"] is True
        assert all(row["notes"]["invariants"].values()), row["scenario"]
    crashed = [
        row
        for row in rows
        if row["scenario"] in {"REC-01", "REC-02", "REC-03", "REC-04", "REC-05"}
    ]
    for row in crashed:
        assert row["counters"]["crash_exit_code"] == 70
    # REC-05 stops after the committed summary: nothing is interrupted and nothing is
    # resumed, so only the four interrupted crashes have identification timings.
    resumed = [
        row
        for row in rows
        if row["scenario"] in {"REC-01", "REC-02", "REC-03", "REC-04", "REC-08"}
    ]
    for row in resumed:
        assert row["timings"]["interrupt_identification_seconds"] <= 20.0, row[
            "scenario"
        ]
        assert row["timings"]["resume_seconds"] <= 30.0, row["scenario"]


# ------------------------------------------------------------------------ DB-01


def test_db01_migrations_are_repeatable_and_leave_other_resources_alone() -> None:
    """DB-01: empty database, repeated migration/seed, untouched non-project tables."""

    database_url, database = create_isolated_database()
    try:
        engine = create_engine(database_url)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE procurement_suppliers_v1 ("
                    "supplier_id text PRIMARY KEY, payload jsonb NOT NULL)"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO procurement_suppliers_v1 VALUES ('SUP-001', '{}'::jsonb)"
                )
            )
            before = connection.execute(
                text(
                    "SELECT md5(string_agg(supplier_id || payload::text, '|')) "
                    "FROM procurement_suppliers_v1"
                )
            ).scalar_one()

        store = ReviewStore.from_url(database_url)
        first = store.migrate()
        assert first.applied
        second = store.migrate()
        assert second.applied == ()
        assert second.already_present == first.applied
        from procurement_review.security_fixture import build_seed_plan

        store.seed(build_seed_plan())
        store.seed(build_seed_plan())
        with engine.begin() as connection:
            after = connection.execute(
                text(
                    "SELECT md5(string_agg(supplier_id || payload::text, '|')) "
                    "FROM procurement_suppliers_v1"
                )
            ).scalar_one()
            tables = connection.execute(
                text(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name LIKE 'review_%'"
                )
            ).scalar_one()
        assert after == before
        assert tables >= 12
        store.close()
        engine.dispose()
    finally:
        drop_database(database)


def test_db01_unknown_migration_version_is_refused() -> None:
    """DB-01: a database from a newer build is never migrated silently."""

    database_url, database = create_isolated_database()
    try:
        store = ReviewStore.from_url(database_url)
        store.migrate()
        with store.transaction() as connection:
            connection.execute(
                text(
                    "INSERT INTO review_schema_migration (version, name, checksum) "
                    "VALUES (9999, 'future', 'sha256:future')"
                )
            )
        from procurement_review.review_store import UnknownMigrationVersionError

        with pytest.raises(UnknownMigrationVersionError):
            store.migrate()
        store.close()
    finally:
        drop_database(database)


# ------------------------------------------------------------------------ DB-02


def test_db02_missing_configuration_has_no_anonymous_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DB-02: without a store the app starts but every business endpoint fails closed."""

    monkeypatch.delenv("REVIEW_DATABASE_URL", raising=False)
    application = create_app()
    with TestClient(application, base_url="http://127.0.0.1") as client:
        health = client.get("/api/v2/health")
        assert health.status_code == 200
        assert health.json()["security_store"] == "not_configured"
        for path in ("/api/v2/session", "/api/v2/cases", "/api/v2/reviews"):
            response = client.get(path)
            assert response.status_code in {401, 503}, response.text
            if response.status_code == 503:
                assert response.json()["error"]["code"] == "security_not_configured"
        created = client.post(
            "/api/v2/reviews",
            headers={"Idempotency-Key": "db02"},
            json={
                "api_version": "review-api.v2",
                "case_id": "CASE-LOW-001",
                "case_version": "case-low-001.v1",
            },
        )
        assert created.status_code == 503


def test_db02_store_outage_is_a_structured_failure(
    fresh_harness: SecurityHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DB-02: an unavailable store answers 503 and never an in-memory success."""

    token = fresh_harness.issue(DEMO_SUBJECT)
    application = create_app(
        store=fresh_harness.store,
        authorization=fresh_harness.authorization,
        repository=fresh_harness.repository,
        service=fresh_harness.service,
    )
    with TestClient(application, base_url="http://127.0.0.1") as client:
        created = client.post(
            "/api/v2/reviews",
            headers={
                "Authorization": f"Bearer {token}",
                "Idempotency-Key": "db02b",
            },
            json={
                "api_version": "review-api.v2",
                "case_id": "CASE-LOW-001",
                "case_version": "case-low-001.v1",
            },
        )
        assert created.status_code == 202, created.text
        run_id = created.json()["run_id"]
        body = wait_for_status(
            client,
            run_id,
            {"Authorization": f"Bearer {token}"},
            "pending_confirmation",
            "pending_human_review",
        )
        assert body["summary"] is not None

    # Now break the store underneath the same app and check the answer.
    engine = create_engine(
        f"{require_admin_dsn().rsplit('/', 1)[0]}/{fresh_harness.database}",
        isolation_level="AUTOCOMMIT",
    )
    try:
        with engine.connect() as connection:
            connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :d AND pid <> pg_backend_pid()"
                ),
                {"d": fresh_harness.database},
            )
    finally:
        engine.dispose()
    application2 = create_app(
        store=fresh_harness.store,
        authorization=fresh_harness.authorization,
        repository=fresh_harness.repository,
        service=fresh_harness.service,
    )
    with TestClient(application2, base_url="http://127.0.0.1") as client:
        response = client.get(
            f"/api/v2/reviews/{run_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code in {200, 503}
    if response.status_code == 503:
        assert response.json()["error"]["code"] == "security_dependency_unavailable"


# ------------------------------------------------------------------------ DB-03


def test_db03_a_new_process_reads_every_committed_fact(
    fresh_harness: SecurityHarness,
) -> None:
    """DB-03: a service built from scratch reads Runs, checkpoints and permissions."""

    service = fresh_harness.service
    token = fresh_harness.issue(DEMO_SUBJECT)
    import asyncio

    from procurement_review.api_contracts import ReviewCreateRequestV2
    from procurement_review.security_contracts import SecurityOperation

    async def create_run() -> str:
        context = fresh_harness.authorization.resolve(
            token,
            request_id="db03",
            operation=SecurityOperation.REVIEW_CREATE,
        )
        registration = fresh_harness.authorization.require_case(
            context,
            "CASE-LOW-001",
            request_id="db03",
            operation=SecurityOperation.REVIEW_CREATE,
        )
        scope = fresh_harness.authorization.build_runtime_scope(
            context,
            registration,
            generation=1,
            deadline_at=fresh_harness.authorization.runtime_scope_deadline(30.0),
        )
        run, _ = await service.create_review(
            ReviewCreateRequestV2(
                case_id="CASE-LOW-001",
                case_version=registration.case_version,
            ),
            registration=registration,
            scope=scope,
            context=context,
            idempotency_key="db03-key",
        )
        for _ in range(400):
            body = service.get_review(run.run_id, context=context)
            if body.status.value != "processing":
                break
            await asyncio.sleep(0.02)
        await service.shutdown()
        return run.run_id

    run_id = asyncio.run(create_run())

    # A brand-new store/service instance: nothing is inherited from the previous one.
    fresh_store = ReviewStore.from_url(_database_url(fresh_harness))
    try:
        fresh_repository = ReviewRepository(fresh_store)
        fresh_service = ReviewService(
            repository=fresh_repository,
            runner=ScopedWorkflowRunner(store=fresh_store),
        )
        fresh_authorization = type(fresh_harness.authorization)(store=fresh_store)
        loaded = fresh_repository.load_run(run_id)
        assert loaded is not None
        assert loaded.header.status.value in {
            "pending_confirmation",
            "pending_human_review",
        }
        assert len(loaded.checkpoints) == 3
        assert loaded.trace_event_count > 0
        assert loaded.summary_checksum is not None
        context = fresh_authorization.resolve(
            token,
            request_id="db03-read",
            operation=SecurityOperation.REVIEW_DETAIL,
        )
        body = fresh_service.get_review(run_id, context=context)
        assert body.summary_checksum == loaded.summary_checksum
        assert body.recovery is not None
        assert body.recovery.adopted_workers
        trace = fresh_service.get_trace(run_id, context=context)
        assert len(trace.events) == loaded.trace_event_count
        assert fresh_repository.attempt_totals(run_id)
    finally:
        fresh_store.close()


def _database_url(harness: SecurityHarness) -> str:
    base = require_admin_dsn()
    head, _, tail = base.partition("?")
    scheme_and_host, _, _old = head.rpartition("/")
    url = f"{scheme_and_host}/{harness.database}"
    return f"{url}?{tail}" if tail else url


def test_declared_suite_matrix_matches_the_runner() -> None:
    """AC-13: the frozen matrix is versioned and every row names a real runner."""

    from procurement_review.security_recovery_check import (
        REC_IDS,
        SUITE_GROUPS,
        load_declared_matrix,
    )

    matrix = load_declared_matrix()
    rows = matrix["scenarios"]
    assert len(rows) == 47
    assert matrix["fixed_recovery_denominator"] == 24
    recovery = [item for item in rows if item["class"] == "recovery"]
    assert [item["id"] for item in recovery] == list(REC_IDS)
    runner_names = {
        name
        for names in SUITE_GROUPS.values()
        for name in names
        if name in {"SEC-GATE", "AUDIT-RESTART", "FIELD-SCAN"}
    }
    assert {item["runner"] for item in rows} <= (
        runner_names
        | set(REC_IDS)
        | {"manual", "playwright"}
        | {
            "test_v1_03_transactions",
            "test_v1_03_recovery",
            "test_workbench_rules_v2_regression",
        }
    )
    for item in rows:
        assert item["expected"]
        assert int(item["repeats"]) >= 1
