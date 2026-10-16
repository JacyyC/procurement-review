"""FIX-03B V-03B-01 and V-03B-03: the read/export guard and the controlled refusal.

Every scenario commits a real Run through the product's own API, poisons exactly one
committed record with an independent administrator connection, and then asks the real HTTP
surface again.  What is asserted is the whole answer: the status code, the fixed error
code, the absence of the canary, of the validation input, of the SQL and of the original
exception, the fact that the read changed neither the stored bytes nor the Run's state,
and that a legal historical record still reads normally.

The authorization check always comes first: another tenant's Run answers 404 exactly as
before, and a revoked credential answers 401 — a damaged record never becomes a way to
learn that a foreign Run exists.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from sqlalchemy import text

from procurement_review import review_repository
from procurement_review.field_scan import scan_persisted_records
from procurement_review.run_errors import RecordUnavailableError

from fix01a_support import Fix01aFixture, settle
from test_fix01b_record_integrity import (
    API,
    API_VERSION_V2,
    ApiNode,
    api_node,
    create_run,
    database,
    headers,
)

CANARY = "13800001234"
CREDENTIAL_CANARY = "rvk1.cred-canary.ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
LEAK_MARKERS = (CANARY, CREDENTIAL_CANARY, "ValidationError", "psycopg", "SELECT")


# --------------------------------------------------------------------------- helpers


def commit_run(fixture: Fix01aFixture, node: ApiNode, *, key: str) -> tuple[str, Any]:
    """One Run that reached the human gate and was decided by its operator.

    The decision is what records the receipt and the approval audit, so every record class
    a read path may consume exists and can be polluted in turn.
    """

    ident = fixture.identity()
    run_id = create_run(node, ident, key=key)
    settle(fixture, run_id, "pending_confirmation")
    body = node.client.get(
        f"{API}/reviews/{run_id}", headers=headers(ident.token)
    ).json()
    decided = node.client.post(
        f"{API}/reviews/{run_id}/confirm",
        headers=headers(ident.token, f"{key}-decide"),
        json={
            "api_version": API_VERSION_V2,
            "expected_version": int(body["version"]),
            "expected_summary_checksum": body["summary_checksum"],
        },
    )
    assert decided.status_code == 200, decided.text
    return run_id, ident


def stored_bytes(fixture: Fix01aFixture, run_id: str) -> dict[str, str]:
    """A digest of every committed row of one Run, plus its scalar state."""

    digest: dict[str, str] = {}
    tables = (
        "review_run",
        "review_run_generation",
        "review_worker_attempt",
        "review_branch_checkpoint",
        "review_run_summary",
        "review_idempotency",
        "review_receipt",
        "review_trace_event",
        "review_audit_event",
    )
    with fixture.driver.transaction() as connection:
        for table in tables:
            rows = [
                json.dumps(row[0], sort_keys=True, default=str)
                for row in connection.execute(
                    text(
                        f"SELECT to_jsonb(t) FROM {table} t WHERE run_id = :r ORDER BY 1"
                    ),
                    {"r": run_id},
                ).all()
            ]
            digest[table] = hashlib.sha256("\n".join(rows).encode()).hexdigest()
        state = connection.execute(
            text(
                "SELECT status, revision, resume_count, error_code FROM review_run "
                "WHERE run_id = :r"
            ),
            {"r": run_id},
        ).one()
    digest["state"] = (
        f"{state.status}/{state.revision}/{state.resume_count}/{state.error_code}"
    )
    return digest


def poison(fixture: Fix01aFixture, run_id: str, *, what: str, value: str) -> None:
    """Commit one polluted record through the independent administrator connection."""

    statements: dict[str, tuple[str, dict[str, Any]]] = {
        "trace_error": (
            "UPDATE review_trace_event SET error = :v, event_payload = "
            "jsonb_set(event_payload, '{error}', to_jsonb(CAST(:v AS text))) "
            "WHERE run_id = :r AND event_ordinal = 1",
            {"v": value, "r": run_id},
        ),
        "trace_summary": (
            "UPDATE review_trace_event SET output_summary = "
            "jsonb_set(output_summary, '{status}', to_jsonb(CAST(:v AS text))), "
            "event_payload = jsonb_set(event_payload, '{output_summary,status}', "
            "to_jsonb(CAST(:v AS text))) "
            "WHERE run_id = :r AND event_ordinal = 2",
            {"v": value, "r": run_id},
        ),
        "summary": (
            "UPDATE review_run_summary SET summary_payload = "
            "jsonb_set(summary_payload, '{failure_reasons}', "
            "to_jsonb(ARRAY[CAST(:v AS text)])) WHERE run_id = :r",
            {"v": value, "r": run_id},
        ),
        "summary_key": (
            "UPDATE review_run_summary SET summary_payload = "
            'summary_payload || \'{"customer_notes": {"freeform": "text"}}\'::jsonb '
            "WHERE run_id = :r",
            {"v": value, "r": run_id},
        ),
        "receipt": (
            "UPDATE review_receipt SET receipt_payload = "
            "jsonb_set(receipt_payload, '{human_actor}', to_jsonb(CAST(:v AS text))) "
            "WHERE run_id = :r",
            {"v": value, "r": run_id},
        ),
        "checkpoint": (
            "UPDATE review_branch_checkpoint SET result_payload = "
            "jsonb_set(result_payload, '{result,outcome}', to_jsonb(CAST(:v AS text))) "
            "WHERE run_id = :r AND worker = 'policy'",
            {"v": value, "r": run_id},
        ),
        "audit": (
            "UPDATE review_audit_event SET event_payload = "
            "jsonb_set(event_payload, '{detail_code}', to_jsonb(CAST(:v AS text))) "
            "WHERE run_id = :r",
            {"v": value, "r": run_id},
        ),
        "isolation": (
            "UPDATE review_run SET isolation = "
            "jsonb_set(isolation, '{purchase_order_id}', to_jsonb(CAST(:v AS text))) "
            "WHERE run_id = :r",
            {"v": value, "r": run_id},
        ),
        "run_error_message": (
            "UPDATE review_run SET status = 'failed', error_message = :v "
            "WHERE run_id = :r",
            {"v": value, "r": run_id},
        ),
    }
    statement, parameters = statements[what]
    with fixture.driver.transaction() as connection:
        connection.execute(text(statement), parameters)


def assert_controlled_refusal(response: Any) -> None:
    """The one safe 503: fixed code, fixed message, no internals."""

    assert response.status_code == 503, response.text
    body = response.json()
    assert body["error"]["code"] == "review_record_unavailable", body
    assert body["error"]["message"] == (
        "The stored review record is not available for reading."
    )
    assert body["error"]["details"] == {}
    for marker in LEAK_MARKERS:
        assert marker not in response.text, marker


# ------------------------------------------------------------------- V-03B-01 reads


@pytest.mark.parametrize(
    "record,paths",
    [
        ("trace_error", ("trace",)),
        ("trace_summary", ("trace",)),
        ("summary", ("detail", "list", "trace", "audit")),
        ("summary_key", ("detail", "list")),
        ("receipt", ("detail", "list")),
        ("checkpoint", ("detail", "list")),
        ("audit", ("audit",)),
        ("isolation", ("detail", "list")),
        ("run_error_message", ("detail", "list")),
    ],
)
def test_a_polluted_record_is_refused_on_its_real_consumption_paths(
    record: str, paths: tuple[str, ...]
) -> None:
    """V-03B-01: every export path that consumes the record fails closed and safely."""

    with database(f"v_03b_01_{record}") as fixture:
        node = api_node(fixture, f"v03b_{record}")
        try:
            run_id, ident = commit_run(fixture, node, key=f"v-03b-01-{record}")
            before = stored_bytes(fixture, run_id)
            poison(fixture, run_id, what=record, value=CANARY)
            poisoned = stored_bytes(fixture, run_id)
            assert poisoned != before

            targets = {
                "detail": f"{API}/reviews/{run_id}",
                "list": f"{API}/reviews",
                "trace": f"{API}/reviews/{run_id}/trace",
                "audit": f"{API}/reviews/{run_id}/audit",
            }
            for name in paths:
                response = node.client.get(targets[name], headers=headers(ident.token))
                assert_controlled_refusal(response)
            # The refusal is a read: nothing about the record or the Run changed.
            assert stored_bytes(fixture, run_id) == poisoned
        finally:
            node.close()


def test_a_polluted_record_never_leaks_through_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """V-03B-01: the refusal logs the record kind, never the value or the path."""

    import logging

    with database("v_03b_01_log") as fixture:
        node = api_node(fixture, "v03b_log")
        try:
            run_id, ident = commit_run(fixture, node, key="v-03b-01-log")
            poison(fixture, run_id, what="trace_error", value=CREDENTIAL_CANARY)
            with caplog.at_level(logging.INFO):
                response = node.client.get(
                    f"{API}/reviews/{run_id}/trace", headers=headers(ident.token)
                )
            assert_controlled_refusal(response)
            assert CREDENTIAL_CANARY not in caplog.text
        finally:
            node.close()


def test_authorization_precedes_the_record_check() -> None:
    """V-03B-01: a foreign or unauthorized read keeps its own answer, not a 503."""

    with database("v_03b_01_authz") as fixture:
        node = api_node(fixture, "v03b_authz")
        try:
            run_id, ident = commit_run(fixture, node, key="v-03b-01-authz")
            poison(fixture, run_id, what="trace_error", value=CANARY)
            # No credential at all.
            assert node.client.get(f"{API}/reviews/{run_id}").status_code == 401
            # A credential of another tenant: the same 404 as an unknown Run.
            other = fixture.identity(
                subject_id="subject-alpha-operator",
                case_id="CASE-ALPHA-001",
                tenant_id="tenant-alpha",
            )
            for path in (
                f"{API}/reviews/{run_id}",
                f"{API}/reviews/{run_id}/trace",
                f"{API}/reviews/{run_id}/audit",
            ):
                response = node.client.get(path, headers=headers(other.token))
                # The authorization answer (no scope, or the same 404 an unknown Run
                # gets) — never the record refusal, and never a byte of the record.
                assert response.status_code in {403, 404}, response.text
                assert CANARY not in response.text
            # A revoked credential stops the read before the record is looked at.
            fixture.revoke_credential(ident.credential_id)
            revoked = node.client.get(
                f"{API}/reviews/{run_id}", headers=headers(ident.token)
            )
            assert revoked.status_code == 401, revoked.text
        finally:
            node.close()


def test_a_clean_run_still_reads_and_history_is_untouched() -> None:
    """V-03B-01: the closed read contract does not block a normal record."""

    with database("v_03b_01_clean") as fixture:
        node = api_node(fixture, "v03b_clean")
        try:
            run_id, ident = commit_run(fixture, node, key="v-03b-01-clean")
            before = stored_bytes(fixture, run_id)
            detail = node.client.get(
                f"{API}/reviews/{run_id}", headers=headers(ident.token)
            )
            assert detail.status_code == 200, detail.text
            assert detail.json()["summary"]["findings"]
            trace = node.client.get(
                f"{API}/reviews/{run_id}/trace", headers=headers(ident.token)
            )
            assert trace.status_code == 200, trace.text
            assert trace.json()["trace"]["events"]
            audit = node.client.get(
                f"{API}/reviews/{run_id}/audit", headers=headers(ident.token)
            )
            assert audit.status_code == 200, audit.text
            listing = node.client.get(f"{API}/reviews", headers=headers(ident.token))
            assert listing.status_code == 200, listing.text
            # Reading is pure: not one committed byte moved.
            assert stored_bytes(fixture, run_id) == before
        finally:
            node.close()


def test_a_legal_legacy_trace_row_still_reads() -> None:
    """V-03B-03: a Trace row committed before the payload column still reads complete."""

    with database("v_03b_03_trace_v0") as fixture:
        node = api_node(fixture, "v03b_trace_v0")
        ident = fixture.identity()
        token = ident.token
        try:
            run_id = create_run(node, ident, key="v-03b-03-trace-v0")
            settle(fixture, run_id, "pending_confirmation")
            # The declared v0 shape: the earliest Trace row predates the payload column.
            with fixture.driver.transaction() as connection:
                connection.execute(
                    text(
                        "UPDATE review_trace_event SET event_payload = '{}'::jsonb "
                        "WHERE run_id = :r AND event_ordinal = 1"
                    ),
                    {"r": run_id},
                )
            trace = node.client.get(
                f"{API}/reviews/{run_id}/trace", headers=headers(token)
            )
            assert trace.status_code == 200, trace.text
            first = next(
                item
                for item in trace.json()["trace"]["events"]
                if item["event_id"].endswith("0001")
            )
            # The row is answered with its complete projected record, not a dropped one.
            assert first["step"] == "review_run"
            assert first["source_versions"]["api"] == "review-api.v2"
            assert first["source_versions"]["dataset"]
            assert first["input_checksum"].startswith("sha256:")
            assert first["redaction_status"] == "not_required"
            # The offline scan accepts the same row as its declared version.
            with fixture.driver.transaction() as connection:
                scan = scan_persisted_records(connection, run_id=run_id)
            assert scan.problems == [] and scan.refused == 0
        finally:
            node.close()


def test_a_legal_legacy_isolation_still_reads() -> None:
    """V-03B-03: the v0 Case input stays readable through the shared repository read."""

    with database("v_03b_03_isolation_v0") as fixture:
        node = api_node(fixture, "v03b_isolation_v0")
        ident = fixture.identity()
        try:
            run_id = create_run(node, ident, key="v-03b-03-isolation-v0")
            settle(fixture, run_id, "pending_confirmation")
            with fixture.driver.transaction() as connection:
                connection.execute(
                    text(
                        "UPDATE review_run SET isolation = '{}'::jsonb WHERE run_id = :r"
                    ),
                    {"r": run_id},
                )
            run = node.repository.load_run(run_id)
            assert run is not None and dict(run.header.isolation) == {}
            # The repository read verifies the whole Run, so the declared v0 version is
            # what makes this succeed — with a populated but damaged input it refuses.
            events, anomalies = node.repository.audit_projection(run_id)
            assert events == () and anomalies == ()
            with fixture.driver.transaction() as connection:
                scan = scan_persisted_records(connection, run_id=run_id)
            assert scan.problems == [] and scan.refused == 0
            poison(fixture, run_id, what="isolation", value=CANARY)
            with pytest.raises(RecordUnavailableError):
                node.repository.audit_projection(run_id)
        finally:
            node.close()


def test_the_v0_version_cannot_be_used_to_hide_a_damaged_input() -> None:
    """V-03B-03: a populated but damaged Case input cannot downgrade into the v0 shape."""

    from procurement_review.record_reads import isolation_kind

    assert isolation_kind({}) == "isolation_v0"
    assert isolation_kind({"case_type": "purchase_order"}) == "isolation"
    with database("v_03b_03_downgrade") as fixture:
        node = api_node(fixture, "v03b_downgrade")
        try:
            run_id, ident = commit_run(fixture, node, key="v-03b-03-downgrade")
            poison(fixture, run_id, what="isolation", value=CANARY)
            response = node.client.get(
                f"{API}/reviews/{run_id}", headers=headers(ident.token)
            )
            assert_controlled_refusal(response)
        finally:
            node.close()


# ------------------------------------------------------------------ negative control


def test_the_pre_read_guard_exports_the_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V-03B-01 control: without the read guard the same record exports at 200.

    The pre-fix reader was ``RunEvent.model_validate`` plus the raw projection; restoring
    exactly that (no record verification on the read path) makes the canary reach the
    response and the assertion above fail, so the assertion binds the fix.
    """

    with database("v_03b_ctl_read") as fixture:
        node = api_node(fixture, "v03b_ctl_read")
        try:
            run_id, ident = commit_run(fixture, node, key="v-03b-01-ctl")
            poison(fixture, run_id, what="trace_error", value=CANARY)
            assert_controlled_refusal(
                node.client.get(
                    f"{API}/reviews/{run_id}/trace", headers=headers(ident.token)
                )
            )
            monkeypatch.setattr(
                review_repository, "verify_run_record", lambda *a, **k: None
            )
            monkeypatch.setattr(
                review_repository, "verify_trace_records", lambda *a, **k: None
            )
            exported = node.client.get(
                f"{API}/reviews/{run_id}/trace", headers=headers(ident.token)
            )
            assert exported.status_code == 200, exported.text
            assert CANARY in exported.text
        finally:
            node.close()


# ---------------------------------------------------------------- repository surface


def test_the_repository_read_helpers_refuse_a_polluted_record() -> None:
    """V-03B-01: the shared read helpers refuse, so every consumer inherits the guard."""

    with database("v_03b_01_repo") as fixture:
        node = api_node(fixture, "v03b_repo")
        try:
            run_id, _ident = commit_run(fixture, node, key="v-03b-01-repo")
            poison(fixture, run_id, what="audit", value=CANARY)
            with pytest.raises(RecordUnavailableError) as raised:
                node.repository.audit_projection(run_id)
            assert raised.value.record == "review_audit"
            assert raised.value.categories == ("cn_mobile_pii",)
            assert CANARY not in str(raised.value)
        finally:
            node.close()


def test_a_damaged_checkpoint_is_not_adopted_on_recovery() -> None:
    """V-03B-03: recovery verifies the checkpoint record before it reuses anything."""

    with database("v_03b_03_recovery") as fixture:
        node = api_node(fixture, "v03b_recovery")
        ident = fixture.identity()
        token = ident.token
        try:
            run_id = create_run(node, ident, key="v-03b-03-recovery")
            settle(fixture, run_id, "pending_confirmation")
            with fixture.driver.transaction() as connection:
                connection.execute(
                    text(
                        "UPDATE review_branch_checkpoint SET result_payload = "
                        "jsonb_set(result_payload, '{result,outcome}', "
                        "to_jsonb(CAST(:v AS text))) WHERE run_id = :r AND worker='policy'"
                    ),
                    {"v": CANARY, "r": run_id},
                )
                connection.execute(
                    text(
                        "UPDATE review_run SET status = 'interrupted', "
                        "revision = revision + 1, error_code = 'storage_unavailable' "
                        "WHERE run_id = :r"
                    ),
                    {"r": run_id},
                )
                connection.execute(
                    text(
                        "UPDATE review_run_generation SET status = 'expired' "
                        "WHERE run_id = :r AND status = 'active'"
                    ),
                    {"r": run_id},
                )
                version = int(
                    connection.execute(
                        text("SELECT revision FROM review_run WHERE run_id = :r"),
                        {"r": run_id},
                    ).scalar_one()
                )
            before = stored_bytes(fixture, run_id)
            response = node.client.post(
                f"{API}/reviews/{run_id}/resume",
                headers=headers(token, "v-03b-03-resume"),
                json={"api_version": API_VERSION_V2, "expected_version": version},
            )
            # The existing permanent-rejection contract, not a new failure mode.
            assert response.status_code == 409, response.text
            assert response.json()["error"]["code"] == "review_not_recoverable"
            assert CANARY not in response.text
            # The refused resume consumed nothing and adopted nothing.
            after = stored_bytes(fixture, run_id)
            assert (
                after["review_branch_checkpoint"] == before["review_branch_checkpoint"]
            )
            assert after["review_run_generation"] == before["review_run_generation"]
        finally:
            node.close()
