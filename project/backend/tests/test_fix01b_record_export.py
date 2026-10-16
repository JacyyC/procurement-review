"""FIX-01B rework: the six B-AUD2 gaps closed on their real consumption paths.

The second audit of FIX-01B (EV-AUDIT-B-02) reproduced six ways a committed record could
still reach a reader, or a reader's failure could still speak: a create replay that skipped
the export guard, a non-empty Trace payload that could downgrade itself into the legacy
shape, a read model that validated only the fields it had already kept, parse errors that
reached the real API log as their own values, a per-Case grant check that ran *after* the
record check, and a JSON/type surface that converted a boolean into a legal integer.

Every scenario commits a real Run through the product's own API on its own disposable
PostgreSQL database, damages exactly one committed record with an independent administrator
connection, and then asks the real HTTP surface — a TestClient for the read contracts, a
real Uvicorn subprocess for the error envelope and its log.  Nothing is asserted about
internal function calls: the assertion is the status code, the fixed error body, the absence
of the canary and of any exception text, and the byte-for-byte identity of the nine business
tables before and after the refused read.

The negative controls re-apply the exact pre-fix rule in process, so each group also proves
that its own assertion would fail on the behaviour it replaced.
"""

from __future__ import annotations

import contextlib
import http
import importlib
import json
import logging
import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import text

from procurement_review import record_guard
from procurement_review.contracts import ReviewLifecycleStatus, RunEvent
from procurement_review.field_scan import scan_persisted_records
from procurement_review.record_guard import guard_record, scan_persisted
from procurement_review.run_records import trace_row_version
from procurement_review.run_errors import RecordUnavailableError

from fix01a_support import Fix01aFixture, settle
from test_fix01b_record_integrity import (
    API,
    API_VERSION_V2,
    ApiNode,
    api_node,
    create_body,
    create_run,
    database,
    headers,
)
from test_fix03b_read_guard import (
    CANARY,
    CREDENTIAL_CANARY,
    LEAK_MARKERS,
    assert_controlled_refusal,
    stored_bytes,
)

#: A value that is legal for its column but disagrees with the payload stored beside it.
DIVERGENT_STEP = "review_workflow_diverged"
DIVERGENT_ACTOR = "subject-diverged"
#: A duration the strict rules must keep legal.
LEGAL_DURATION = 7

TRACE_TABLE = "review_trace_event"
FIRST_EVENT = "run_id = :r AND event_ordinal = 1"
APPROVAL_EVENT = "run_id = :r AND event_class = 'approval'"

#: Everything a refused read must keep out of a response and out of the server log.
FORBIDDEN_OUTPUT = LEAK_MARKERS + (
    "input_value",
    "ValueError",
    "KeyError",
    "TypeError",
    "Traceback",
    "psycopg",
    "SELECT",
)


# --------------------------------------------------------------------------- poisons


#: Each recipe damages exactly one committed fact through the admin connection, so a
#: scenario proves one record rather than a whole table.
POISONS: dict[str, tuple[str, dict[str, Any]]] = {
    # --- B-AUD2-01: a record a replay hands straight back.
    "receipt_actor": (
        "UPDATE review_receipt SET receipt_payload = jsonb_set("
        "receipt_payload, '{human_actor}', to_jsonb(CAST(:v AS text))) WHERE run_id = :r",
        {"v": CANARY},
    ),
    "receipt_column": (
        "UPDATE review_receipt SET human_actor = :v WHERE run_id = :r",
        {"v": CANARY},
    ),
    # --- B-AUD2-02: the Trace payload's own shape.
    "trace_without_event_id": (
        f"UPDATE {TRACE_TABLE} SET event_payload = event_payload - 'event_id' "
        f"WHERE {FIRST_EVENT}",
        {},
    ),
    "trace_without_event_id_canary": (
        f"UPDATE {TRACE_TABLE} SET event_payload = jsonb_set("
        f"event_payload - 'event_id', '{{error}}', to_jsonb(CAST(:v AS text))) "
        f"WHERE {FIRST_EVENT}",
        {"v": CANARY},
    ),
    "trace_payload_array": (
        f"UPDATE {TRACE_TABLE} SET event_payload = '[]'::jsonb WHERE {FIRST_EVENT}",
        {},
    ),
    "trace_payload_null": (
        f"UPDATE {TRACE_TABLE} SET event_payload = 'null'::jsonb WHERE {FIRST_EVENT}",
        {},
    ),
    "trace_payload_text": (
        "UPDATE review_trace_event SET event_payload = to_jsonb(CAST('plain text' AS text))"
        f" WHERE {FIRST_EVENT}",
        {},
    ),
    "trace_v0_column": (
        f"UPDATE {TRACE_TABLE} SET worker = :v WHERE {FIRST_EVENT}",
        {"v": CANARY},
    ),
    # --- B-AUD2-03: relation columns, and the payload beside them.
    "trace_column_error": (
        f"UPDATE {TRACE_TABLE} SET error = :v WHERE {FIRST_EVENT}",
        {"v": CANARY},
    ),
    "trace_column_worker": (
        f"UPDATE {TRACE_TABLE} SET worker = :v WHERE {FIRST_EVENT}",
        {"v": CANARY},
    ),
    "trace_column_diverges": (
        f"UPDATE {TRACE_TABLE} SET step = :v WHERE {FIRST_EVENT}",
        {"v": DIVERGENT_STEP},
    ),
    "checkpoint_unknown_key": (
        "UPDATE review_branch_checkpoint SET result_payload = result_payload || "
        "jsonb_build_object('customer_notes', "
        "jsonb_build_object('freeform', CAST(:v AS text))) WHERE run_id = :r",
        {"v": CANARY},
    ),
    "checkpoint_without_result": (
        "UPDATE review_branch_checkpoint SET result_payload = result_payload - 'result' "
        "WHERE run_id = :r AND worker = 'policy'",
        {},
    ),
    "audit_payload_array": (
        "UPDATE review_audit_event SET event_payload = '[]'::jsonb "
        f"WHERE {APPROVAL_EVENT}",
        {},
    ),
    "audit_sequence_text": (
        "UPDATE review_audit_event SET event_payload = jsonb_set(event_payload, "
        "'{sequence}', to_jsonb(CAST(:v AS text))) "
        f"WHERE {APPROVAL_EVENT}",
        {"v": CANARY},
    ),
    "audit_column_diverges": (
        f"UPDATE review_audit_event SET actor = :v WHERE {APPROVAL_EVENT}",
        {"v": DIVERGENT_ACTOR},
    ),
    # --- B-AUD2-04: values that used to escape as an unhandled error.
    "summary_checksum": (
        "UPDATE review_run_summary SET summary_checksum = :v WHERE run_id = :r",
        {"v": CANARY},
    ),
    # --- B-AUD2-06: a JSON boolean where the contract declares a non-negative integer.
    "trace_duration_boolean": (
        f"UPDATE {TRACE_TABLE} SET event_payload = jsonb_set(event_payload, "
        f"'{{duration_ms}}', 'true'::jsonb) WHERE {FIRST_EVENT}",
        {},
    ),
}


def poison(fixture: Fix01aFixture, run_id: str, what: str) -> None:
    """Apply one named recipe on the independent administrator connection."""

    statement, extra = POISONS[what]
    with fixture.driver.transaction() as connection:
        result = connection.execute(text(statement), {"r": run_id, **extra})
    assert result.rowcount >= 1, f"the {what} poison matched no row"


def read_sql(fixture: Fix01aFixture, statement: str, **parameters: Any) -> Any:
    with fixture.driver.transaction() as connection:
        return connection.execute(text(statement), parameters).scalar()


# ------------------------------------------------------------------ real HTTP server


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(slots=True)
class UvicornNode:
    """One real Uvicorn subprocess over the scenario's own database and log file."""

    base_url: str
    process: subprocess.Popen[bytes]
    log_path: Path
    client: httpx.Client

    @property
    def log_text(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace")

    def get(self, path: str, token: str) -> httpx.Response:
        return self.client.get(path, headers=headers(token))


@contextlib.contextmanager
def uvicorn_node(fixture: Fix01aFixture, tag: str) -> Any:
    """Start and stop one real API process; the log file is what the assertions read."""

    work_dir = Path(os.environ.get("FIX01B2_TMP", "/tmp")) / f"fix01b2-api-{tag}"
    work_dir.mkdir(parents=True, exist_ok=True)
    for stale in work_dir.glob("api-*.log"):
        stale.unlink()
    log_path = work_dir / f"api-{secrets.token_hex(3)}.log"
    port = _free_port()
    environment = {
        key: value
        for key, value in os.environ.items()
        if key != "TEST_POSTGRES_DSN" and not key.lower().endswith("_proxy")
    }
    environment["REVIEW_DATABASE_URL"] = fixture.url_for(f"fix01b2_{tag}")
    environment["REVIEW_ALLOWED_ORIGINS"] = ""
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    with log_path.open("wb") as handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "procurement_review.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "info",
            ],
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
    node = UvicornNode(
        base_url=f"http://127.0.0.1:{port}",
        process=process,
        log_path=log_path,
        client=httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0),
    )
    try:
        _wait_ready(node)
        yield node
    finally:
        node.client.close()
        if process.poll() is None:
            process.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=30)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)


def _wait_ready(node: UvicornNode, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if node.process.poll() is not None:
            raise AssertionError(
                f"the API process exited early:\n{node.log_text[-2000:]}"
            )
        try:
            response = node.client.get("/api/v2/health")
        except httpx.HTTPError:
            time.sleep(0.05)
            continue
        if response.status_code == 200:
            return
        time.sleep(0.05)
    raise AssertionError("the API process never became ready")


# --------------------------------------------------------------------- shared helpers


def decided_run(
    fixture: Fix01aFixture, node: ApiNode, ident: Any, *, key: str
) -> dict[str, Any]:
    """One Run that reached the human gate and was confirmed by its own operator.

    The decision records the receipt and the approval audit, and the body the operator sent
    comes back so a replay of that very operation stays reproducible.
    """

    run_id = create_run(node, ident, key=key)
    settle(fixture, run_id, "pending_confirmation")
    body = get(node, ident.token, f"/reviews/{run_id}").json()
    action = {
        "api_version": API_VERSION_V2,
        "expected_version": int(body["version"]),
        "expected_summary_checksum": body["summary_checksum"],
    }
    decided = node.client.post(
        f"{API}/reviews/{run_id}/confirm",
        headers=headers(ident.token, f"{key}-decide"),
        json=action,
    )
    assert decided.status_code == 200, decided.text
    return {"run_id": run_id, "key": f"{key}-decide", "action": action}


def node_run(fixture: Fix01aFixture, node: ApiNode, *, key: str) -> tuple[str, Any]:
    """A committed, decided Run plus the identity that created it."""

    ident = fixture.identity()
    created = decided_run(fixture, node, ident, key=key)
    return created["run_id"], ident


def get(node: ApiNode, token: str, path: str) -> Any:
    return node.client.get(f"{API}{path}", headers=headers(token))


ENDPOINTS = {
    "detail": "/reviews/{run_id}",
    "list": "/reviews",
    "trace": "/reviews/{run_id}/trace",
    "audit": "/reviews/{run_id}/audit",
}


def read_back(node: ApiNode, token: str, run_id: str) -> dict[str, Any]:
    """Every read/export surface of one Run, keyed by a short name."""

    return {
        name: get(node, token, path.format(run_id=run_id))
        for name, path in ENDPOINTS.items()
    }


def assert_scan_refuses(fixture: Fix01aFixture, run_id: str) -> list[str]:
    """The offline scan sees the same damage the online reader refuses."""

    with fixture.driver.transaction() as connection:
        report = scan_persisted_records(connection, run_id=run_id)
    assert report.problems == [], report.as_dict()
    refused = [item.name for item in report.classes if item.refused]
    assert refused, report.as_dict()
    return refused


def scan_is_clean(fixture: Fix01aFixture, run_id: str) -> None:
    with fixture.driver.transaction() as connection:
        report = scan_persisted_records(connection, run_id=run_id)
    assert report.refused == 0 and report.problems == [], report.as_dict()


def altered(base: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    """A deep copy of one stored document with a single field replaced."""

    value = json.loads(json.dumps(dict(base)))
    value.update(overrides)
    return value


# ------------------------------------------------------- B-AUD2-01: the replay export


@pytest.mark.parametrize("what", ["receipt_actor", "receipt_column"])
def test_a_polluted_run_is_refused_by_its_create_replay(what: str) -> None:
    """B-AUD2-01: a same-key create replay is an export, so it passes the same guard."""

    with database(f"aud2_01_{what}") as fixture:
        node = api_node(fixture, f"aud2_01_{what}")
        try:
            ident = fixture.identity()
            created = decided_run(fixture, node, ident, key="aud2-01")
            run_id = created["run_id"]
            clean = stored_bytes(fixture, run_id)
            body = create_body(
                ident.registration.case_id, ident.registration.case_version
            )
            assert (
                node.client.post(
                    f"{API}/reviews", headers=headers(ident.token, "aud2-01"), json=body
                ).status_code
                == 200
            )
            poison(fixture, run_id, what)
            poisoned = stored_bytes(fixture, run_id)
            assert poisoned != clean

            assert_controlled_refusal(get(node, ident.token, f"/reviews/{run_id}"))
            repeated = node.client.post(
                f"{API}/reviews", headers=headers(ident.token, "aud2-01"), json=body
            )
            assert_controlled_refusal(repeated)
            # The refused replay wrote and rewrote nothing.
            assert stored_bytes(fixture, run_id) == poisoned
        finally:
            node.close()


def test_a_clean_create_replay_still_replays_with_zero_new_rows() -> None:
    """B-AUD2-01: the shared outlet keeps the replay contract it already had."""

    with database("aud2_01_clean") as fixture:
        node = api_node(fixture, "aud2_01_clean")
        try:
            ident = fixture.identity()
            created = decided_run(fixture, node, ident, key="aud2-01-clean")
            run_id = created["run_id"]
            clean = stored_bytes(fixture, run_id)
            ledger = fixture.ledger(run_id)
            body = create_body(
                ident.registration.case_id, ident.registration.case_version
            )
            repeated = node.client.post(
                f"{API}/reviews",
                headers=headers(ident.token, "aud2-01-clean"),
                json=body,
            )
            assert repeated.status_code == 200, repeated.text
            assert repeated.json()["run_id"] == run_id
            assert fixture.ledger(run_id) == ledger
            assert stored_bytes(fixture, run_id) == clean
        finally:
            node.close()


def test_a_polluted_run_is_refused_by_its_action_replay() -> None:
    """B-AUD2-01: the confirm replay returns a Run, so it passes the same guard."""

    with database("aud2_01_action") as fixture:
        node = api_node(fixture, "aud2_01_action")
        try:
            ident = fixture.identity()
            created = decided_run(fixture, node, ident, key="aud2-01-action")
            run_id = created["run_id"]
            request = {
                "method": "post",
                "url": f"{API}/reviews/{run_id}/confirm",
                "headers": headers(ident.token, created["key"]),
                "json": created["action"],
            }
            replayed = node.client.request(**request)
            assert replayed.status_code == 200, replayed.text
            poison(fixture, run_id, what="receipt_actor")
            poisoned = stored_bytes(fixture, run_id)
            assert_controlled_refusal(node.client.request(**request))
            assert stored_bytes(fixture, run_id) == poisoned
        finally:
            node.close()


def test_the_pre_fix_replay_outlet_exported_the_polluted_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: without the shared outlet the same request answers 200 + canary."""

    with database("aud2_01_control") as fixture:
        node = api_node(fixture, "aud2_01_control")
        try:
            ident = fixture.identity()
            created = decided_run(fixture, node, ident, key="aud2-01-control")
            run_id = created["run_id"]
            body = create_body(
                ident.registration.case_id, ident.registration.case_version
            )
            poison(fixture, run_id, what="receipt_actor")
            assert_controlled_refusal(
                node.client.post(
                    f"{API}/reviews",
                    headers=headers(ident.token, "aud2-01-control"),
                    json=body,
                )
            )
            # The pre-fix outlet was the bare projection: no verification of the Run it
            # hands back, so the stored receipt went straight into the response.
            from procurement_review import review_service

            monkeypatch.setattr(
                review_service,
                "verify_run_record",
                lambda *args, **kwargs: None,
            )
            exported = node.client.post(
                f"{API}/reviews",
                headers=headers(ident.token, "aud2-01-control"),
                json=body,
            )
            assert exported.status_code == 200, exported.text
            assert CANARY in exported.text
        finally:
            node.close()


# ------------------------------------------------- B-AUD2-02: the legacy row version


def test_trace_row_version_only_accepts_the_registered_empty_shape() -> None:
    """B-AUD2-02: the version is decided by the whole shape, never by one key."""

    assert trace_row_version({}) == "trace-row.v0"
    for payload in (
        {"event_id": "x"},
        {"run_id": "r", "error": "e"},
        {"event_id": ""},
        None,
        [],
        ["x"],
        "",
        "text",
        0,
        1,
        True,
    ):
        assert trace_row_version(payload) == "trace-row.v1", payload


@pytest.mark.parametrize(
    "what",
    [
        "trace_without_event_id",
        "trace_without_event_id_canary",
        "trace_payload_array",
        "trace_payload_null",
        "trace_payload_text",
    ],
)
def test_a_non_empty_trace_payload_cannot_downgrade_itself(what: str) -> None:
    """B-AUD2-02: a damaged non-empty payload is a refusal, not a rebuild from columns."""

    with database(f"aud2_02_{what}") as fixture:
        node = api_node(fixture, f"aud2_02_{what}")
        try:
            run_id, ident = node_run(fixture, node, key=f"aud2-02-{what}")
            clean = stored_bytes(fixture, run_id)
            poison(fixture, run_id, what)
            poisoned = stored_bytes(fixture, run_id)
            assert poisoned != clean
            trace = get(node, ident.token, f"/reviews/{run_id}/trace")
            assert_controlled_refusal(trace)
            # Even with no canary in the value, the shape alone refuses, and the answer is
            # never a relation-column rebuild of the event.
            assert "event_id" not in trace.text
            assert_scan_refuses(fixture, run_id)
            assert stored_bytes(fixture, run_id) == poisoned
        finally:
            node.close()


def test_a_legacy_empty_trace_payload_still_reads_and_its_columns_are_still_scanned() -> (
    None
):
    """B-AUD2-02: the registered v0 shape stays readable; a damaged v0 column is refused."""

    with database("aud2_02_v0") as fixture:
        node = api_node(fixture, "aud2_02_v0")
        ident = fixture.identity()
        try:
            run_id = create_run(node, ident, key="aud2-02-v0")
            settle(fixture, run_id, "pending_confirmation")
            with fixture.driver.transaction() as connection:
                connection.execute(
                    text(
                        f"UPDATE {TRACE_TABLE} SET event_payload = '{{}}'::jsonb "
                        f"WHERE {FIRST_EVENT}"
                    ),
                    {"r": run_id},
                )
            trace = get(node, ident.token, f"/reviews/{run_id}/trace")
            assert trace.status_code == 200, trace.text
            first = _event_by_suffix(trace.json()["trace"]["events"], "0001")
            assert first["step"] == "review_run"
            assert first["redaction_status"] == "not_required"
            assert first["source_versions"]["api"] == API_VERSION_V2
            scan_is_clean(fixture, run_id)

            # A damaged relation column of that same v0 row is still refused.
            poison(fixture, run_id, what="trace_v0_column")
            assert_controlled_refusal(
                get(node, ident.token, f"/reviews/{run_id}/trace")
            )
            assert_scan_refuses(fixture, run_id)
        finally:
            node.close()


def _event_by_suffix(events: list[dict[str, Any]], suffix: str) -> dict[str, Any]:
    return next(item for item in events if str(item["event_id"]).endswith(suffix))


def test_the_pre_fix_version_rule_rebuilt_the_damaged_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: the old predicate downgraded any payload without an event_id."""

    def legacy_version(payload: Any) -> str:
        if isinstance(payload, dict) and payload.get("event_id"):
            return "trace-row.v1"
        return "trace-row.v0"

    record_rows = importlib.import_module("procurement_review.record_rows")
    with database("aud2_02_control") as fixture:
        node = api_node(fixture, "aud2_02_control")
        try:
            run_id, ident = node_run(fixture, node, key="aud2-02-control")
            # The damage is only the payload's shape: no sensitive value anywhere.
            poison(fixture, run_id, what="trace_without_event_id")
            assert_controlled_refusal(
                get(node, ident.token, f"/reviews/{run_id}/trace")
            )
            monkeypatch.setattr(record_rows, "trace_row_version", legacy_version)
            exported = get(node, ident.token, f"/reviews/{run_id}/trace")
            assert exported.status_code == 200, exported.text
            # The old reader answered with a record it rebuilt from the columns.
            assert _event_by_suffix(exported.json()["trace"]["events"], "0001")["step"]
        finally:
            node.close()


# --------------------------------------------- B-AUD2-03: complete raw records first


@pytest.mark.parametrize(
    "what,paths",
    [
        ("trace_column_error", ("trace",)),
        ("trace_column_worker", ("trace",)),
        ("trace_column_diverges", ("trace",)),
        ("checkpoint_unknown_key", ("detail", "list")),
        ("checkpoint_without_result", ("detail", "list")),
        ("audit_payload_array", ("audit",)),
        ("audit_column_diverges", ("audit",)),
    ],
)
def test_a_damaged_relation_column_or_dropped_field_is_refused(
    what: str, paths: tuple[str, ...]
) -> None:
    """B-AUD2-03: nothing is trimmed, hidden or rebuilt before its contract is applied."""

    with database(f"aud2_03_{what}") as fixture:
        node = api_node(fixture, f"aud2_03_{what}")
        try:
            run_id, ident = node_run(fixture, node, key=f"aud2-03-{what}")
            clean = stored_bytes(fixture, run_id)
            poison(fixture, run_id, what)
            poisoned = stored_bytes(fixture, run_id)
            assert poisoned != clean
            for name in paths:
                assert_controlled_refusal(read_back(node, ident.token, run_id)[name])
            if what == "audit_payload_array":
                # A damaged audit row is never silently dropped from the event list.
                audit = get(node, ident.token, f"/reviews/{run_id}/audit")
                assert '"events":[]' not in audit.text.replace(" ", ""), audit.text
            assert_scan_refuses(fixture, run_id)
            assert stored_bytes(fixture, run_id) == poisoned
        finally:
            node.close()


def test_a_damaged_checkpoint_record_takes_the_existing_permanent_rejection() -> None:
    """B-AUD2-03: recovery reads the same complete record before adopting anything."""

    with database("aud2_03_resume") as fixture:
        node = api_node(fixture, "aud2_03_resume")
        ident = fixture.identity()
        try:
            run_id = create_run(node, ident, key="aud2-03-resume")
            settle(fixture, run_id, "pending_confirmation")
            poison(fixture, run_id, what="checkpoint_without_result")
            _interrupt(fixture, run_id)
            refused = node.client.post(
                f"{API}/reviews/{run_id}/resume",
                headers=headers(ident.token, "aud2-03-resume-decide"),
                json={
                    "api_version": API_VERSION_V2,
                    "expected_version": _revision(fixture, run_id),
                },
            )
            assert refused.status_code == 409, refused.text
            assert refused.json()["error"]["code"] == "review_not_recoverable"
            assert CANARY not in refused.text
        finally:
            node.close()


def _interrupt(fixture: Fix01aFixture, run_id: str) -> None:
    with fixture.driver.transaction() as connection:
        connection.execute(
            text(
                "UPDATE review_run SET status = 'interrupted', revision = revision + 1, "
                "error_code = 'storage_unavailable' WHERE run_id = :r"
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


def _revision(fixture: Fix01aFixture, run_id: str) -> int:
    return int(
        read_sql(fixture, "SELECT revision FROM review_run WHERE run_id = :r", r=run_id)
    )


def test_the_read_model_carries_the_complete_raw_row_before_any_projection() -> None:
    """B-AUD2-03: ``load_run`` makes the whole committed record available to the guard."""

    with database("aud2_03_raw") as fixture:
        node = api_node(fixture, "aud2_03_raw")
        ident = fixture.identity()
        try:
            run_id = create_run(node, ident, key="aud2-03-raw")
            settle(fixture, run_id, "pending_confirmation")
            run = node.repository.load_run(run_id)
            assert run is not None
            raw = run.raw
            assert set(raw.run) == {
                "run_id",
                "tenant_id",
                "case_id",
                "case_version",
                "status",
                "revision",
                "resume_count",
                "created_by_subject",
                "current_subject_id",
                "request_checksum",
                "document_version",
                "dataset_version",
                "index_version",
                "rules_version",
                "isolation",
                "error_code",
                "error_message",
                "created_at",
                "updated_at",
                "trace_id",
            }
            assert raw.generation is not None and set(raw.generation) == {
                "run_id",
                "generation",
                "subject_id",
                "credential_id",
                "authorization_revision",
                "owner_id",
                "status",
                "heartbeat_at",
                "lease_expires_at",
                "deadline_at",
                "created_at",
            }
            assert raw.attempts and set(raw.attempts[0]) == {
                "run_id",
                "generation",
                "worker",
                "attempt_no",
                "started_at",
                "finished_at",
                "outcome",
                "first_error",
                "last_error",
                "abandoned",
                "discarded_late",
            }
            assert all(
                {
                    "worker",
                    "generation",
                    "attempt_no",
                    "result_checksum",
                    "result_payload",
                    "adopted_at",
                }
                <= set(item)
                for item in raw.checkpoints
            )
            assert raw.summary is not None and {
                "summary_payload",
                "execution_payload",
                "human_gate_status",
                "summary_checksum",
                "committed_revision",
            } <= set(raw.summary)
            assert raw.receipt is None or "receipt_payload" in raw.receipt
            assert raw.idempotency and "request_digest" in raw.idempotency[0]
            assert all("event_payload" in item for item in raw.trace_rows)
            assert all("event_payload" in item for item in raw.audit_rows)
            assert raw.refusal is None
            # The trimmed model still answers the same business facts.
            assert run.header.status is ReviewLifecycleStatus.PENDING_CONFIRMATION
            assert sorted(run.checkpoints) == ["contract", "policy", "procurement_data"]

            poison(fixture, run_id, what="trace_column_worker")
            damaged = node.repository.load_run(run_id)
            assert damaged is not None
            assert damaged.raw.refusal is not None
            assert damaged.raw.refusal.categories
            # The refusal names a record class and a fixed category; the stored value stays
            # in the row it was written to and goes no further than the reader that refuses.
            assert CANARY not in str(damaged.raw.refusal)
            with pytest.raises(RecordUnavailableError) as raised:
                node.repository.trace_envelope(run_id, complete=False)
            assert raised.value.record
            assert CANARY not in str(raised.value)
        finally:
            node.close()


# ------------------------------------------------ B-AUD2-04: one safe answer, no value


@pytest.mark.parametrize(
    "what",
    [
        "summary_checksum",
        "checkpoint_without_result",
        "audit_sequence_text",
        "audit_payload_array",
    ],
)
def test_a_record_that_breaks_the_parser_answers_the_fixed_refusal(what: str) -> None:
    """B-AUD2-04: a parse failure is a refusal, never a 500 and never its own value."""

    with database(f"aud2_04_{what}") as fixture:
        node = api_node(fixture, f"aud2_04_{what}")
        try:
            run_id, ident = node_run(fixture, node, key=f"aud2-04-{what}")
            poison(fixture, run_id, what)
            poisoned = stored_bytes(fixture, run_id)
            responses = read_back(node, ident.token, run_id)
            damaged = [item for item in responses.values() if item.status_code != 200]
            assert damaged, {name: item.text for name, item in responses.items()}
            for response in damaged:
                assert_controlled_refusal(response)
            assert stored_bytes(fixture, run_id) == poisoned
        finally:
            node.close()


@pytest.mark.parametrize(
    "what,paths",
    [
        ("summary_checksum", ("detail", "list", "audit")),
        ("checkpoint_without_result", ("detail", "list", "trace")),
        ("audit_sequence_text", ("audit",)),
        ("audit_payload_array", ("audit",)),
        ("trace_duration_boolean", ("trace",)),
    ],
)
def test_the_real_api_server_answers_json_and_logs_nothing_of_the_record(
    what: str, paths: tuple[str, ...]
) -> None:
    """B-AUD2-04: over real Uvicorn the envelope is JSON and the log carries no value."""

    with database(f"aud2_04_http_{what}") as fixture:
        node = api_node(fixture, f"aud2_04_http_{what}")
        try:
            run_id, ident = node_run(fixture, node, key=f"aud2-04-http-{what}")
            poison(fixture, run_id, what)
        finally:
            node.close()
        with uvicorn_node(fixture, f"http-{what}") as server:
            for name in paths:
                path = ENDPOINTS[name].format(run_id=run_id)
                response = server.get(f"/api/v2{path}", ident.token)
                assert response.status_code == http.HTTPStatus.SERVICE_UNAVAILABLE, (
                    name,
                    response.text,
                )
                assert response.headers["content-type"].startswith("application/json")
                body = response.json()
                assert body["error"]["code"] == "review_record_unavailable", body
                assert body["error"]["message"] == (
                    "The stored review record is not available for reading."
                )
                assert body["error"]["details"] == {}
                for marker in FORBIDDEN_OUTPUT + ("UPDATE", "INSERT"):
                    assert marker not in response.text, (name, marker)
            log = server.log_text
            for marker in FORBIDDEN_OUTPUT + ("postgresql", "rvk1."):
                assert marker not in log, marker
            # The refusal really is the fixed one, and it is visible on the diagnostic face.
            assert "review_record_unavailable" in log


def test_a_storage_outage_stays_a_dependency_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B-AUD2-04: an unavailable store is never reported as a damaged record."""

    from sqlalchemy.exc import OperationalError

    with database("aud2_04_store") as fixture:
        node = api_node(fixture, "aud2_04_store")
        try:
            run_id, ident = node_run(fixture, node, key="aud2-04-store")

            def exploding(*args: Any, **kwargs: Any) -> None:
                raise OperationalError(
                    "SELECT review_run.run_id FROM review_run", {}, Exception("down")
                )

            monkeypatch.setattr(node.repository, "load_run", exploding)
            response = get(node, ident.token, f"/reviews/{run_id}")
            assert response.status_code == 503, response.text
            assert response.json()["error"]["code"] == "security_dependency_unavailable"
        finally:
            node.close()


def test_the_refusal_logs_only_a_fixed_kind_and_category(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """B-AUD2-04: the internal log names the record class, never the value or path."""

    with database("aud2_04_log") as fixture:
        node = api_node(fixture, "aud2_04_log")
        try:
            run_id, ident = node_run(fixture, node, key="aud2-04-log")
            poison(fixture, run_id, what="trace_column_error")
            with caplog.at_level(logging.INFO):
                assert_controlled_refusal(
                    get(node, ident.token, f"/reviews/{run_id}/trace")
                )
            assert caplog.text
            for marker in (CANARY, CREDENTIAL_CANARY, "Traceback", "SELECT", "$."):
                assert marker not in caplog.text, marker
        finally:
            node.close()


# ------------------------------------------------ B-AUD2-05: authorization comes first


@pytest.mark.parametrize(
    "what",
    [None, "summary_checksum", "trace_column_error", "checkpoint_without_result"],
)
def test_the_case_grant_is_decided_before_the_record_is_read(what: str | None) -> None:
    """B-AUD2-05: a subject without the grant keeps its own answer, damaged record or not."""

    with database(f"aud2_05_{what}") as fixture:
        node = api_node(fixture, f"aud2_05_{what}")
        ident = fixture.identity()
        try:
            run_id = create_run(node, ident, key=f"aud2-05-{what}")
            settle(fixture, run_id, "pending_confirmation")
            assert get(node, ident.token, f"/reviews/{run_id}").status_code == 200
            if what is not None:
                poison(fixture, run_id, what)
                assert_controlled_refusal(get(node, ident.token, f"/reviews/{run_id}"))
            # Withdraw the per-Case grant: every run-scoped read is now the 404 this
            # caller always got, and a damaged record cannot turn it into a 503.
            fixture.revoke_case_grant(
                ident.context.subject_id, ident.registration.case_id
            )
            for path in (
                f"/reviews/{run_id}",
                f"/reviews/{run_id}/trace",
                f"/reviews/{run_id}/audit",
            ):
                response = get(node, ident.token, path)
                assert response.status_code == 404, (path, response.text)
                assert response.json()["error"]["code"] == "resource_not_found"
            listing = get(node, ident.token, "/reviews")
            assert listing.status_code == 200, listing.text
            assert all(item["run_id"] != run_id for item in listing.json()["runs"])
        finally:
            node.close()


def test_an_unauthorized_read_never_parses_the_business_record() -> None:
    """B-AUD2-05: the pre-grant read is minimal, so a broken record cannot reach it."""

    with database("aud2_05_minimal") as fixture:
        node = api_node(fixture, "aud2_05_minimal")
        ident = fixture.identity()
        try:
            run_id = create_run(node, ident, key="aud2-05-minimal")
            settle(fixture, run_id, "pending_confirmation")
            # A record the pre-grant read used to fail on before deciding anything.
            poison(fixture, run_id, what="checkpoint_without_result")
            fixture.revoke_case_grant(
                ident.context.subject_id, ident.registration.case_id
            )
            for path in (
                f"/reviews/{run_id}",
                f"/reviews/{run_id}/trace",
                f"/reviews/{run_id}/audit",
            ):
                response = get(node, ident.token, path)
                assert response.status_code == 404, (path, response.text)
            # A decision is a write: it keeps its own 404 for the withdrawn grant.
            refused = node.client.post(
                f"{API}/reviews/{run_id}/confirm",
                headers=headers(ident.token, "aud2-05-confirm"),
                json={"api_version": API_VERSION_V2, "expected_version": 2},
            )
            assert refused.status_code == 404, refused.text
            assert CANARY not in refused.text
        finally:
            node.close()


def test_cross_tenant_and_anonymous_readers_keep_their_own_answers() -> None:
    """B-AUD2-05: the existing rejection contracts are unchanged by the reordering."""

    with database("aud2_05_foreign") as fixture:
        node = api_node(fixture, "aud2_05_foreign")
        try:
            run_id, ident = node_run(fixture, node, key="aud2-05-foreign")
            poison(fixture, run_id, what="summary_checksum")
            for path in (
                f"{API}/reviews/{run_id}",
                f"{API}/reviews/{run_id}/trace",
                f"{API}/reviews/{run_id}/audit",
            ):
                assert node.client.get(path).status_code == 401, path
            other = fixture.identity(
                subject_id="subject-alpha-operator",
                case_id="CASE-ALPHA-001",
                tenant_id="tenant-alpha",
            )
            for path in (
                f"/reviews/{run_id}",
                f"/reviews/{run_id}/trace",
                f"/reviews/{run_id}/audit",
            ):
                response = get(node, other.token, path)
                assert response.status_code in {403, 404}, (path, response.text)
                assert CANARY not in response.text
            unknown = get(node, ident.token, "/reviews/run-" + "0" * 32)
            assert unknown.status_code == 404, unknown.text
        finally:
            node.close()


# ------------------------------------------- B-AUD2-06: strict JSON and nested models


@pytest.fixture(scope="module")
def trace_document() -> dict[str, Any]:
    """One real committed Trace payload, read once from its own database."""

    with database("aud2_06_document") as fixture:
        node = api_node(fixture, "aud2_06_document")
        try:
            run_id = create_run(node, fixture.identity(), key="aud2-06-document")
            settle(fixture, run_id, "pending_confirmation")
            payload = read_sql(
                fixture,
                f"SELECT event_payload FROM {TRACE_TABLE} WHERE {FIRST_EVENT}",
                r=run_id,
            )
        finally:
            node.close()
    return json.loads(json.dumps(payload))


@pytest.mark.parametrize(
    "value,accepted",
    [
        (1, True),
        (0, True),
        ("1", False),
        (True, False),
        (False, False),
        (-1, False),
        (1.5, False),
        (None, False),
    ],
)
def test_the_stored_json_document_is_judged_by_strict_json_types(
    trace_document: dict[str, Any], value: Any, accepted: bool
) -> None:
    """B-AUD2-06: no boolean, string or float converts into a legal integer."""

    report = scan_persisted(
        altered(trace_document, duration_ms=value), kind="trace", location="trace"
    )
    assert bool(report.findings) is not accepted, [
        item.category for item in report.findings
    ]


def test_a_legal_json_document_still_validates_strictly(
    trace_document: dict[str, Any],
) -> None:
    """B-AUD2-06: dates, enums and arrays keep their legal JSON representations."""

    base = altered(trace_document, duration_ms=LEGAL_DURATION)
    assert isinstance(base["occurred_at"], str)
    assert isinstance(base["event_type"], str)
    assert isinstance(base["source_versions"], dict)
    assert isinstance(base["input_summary"], dict)
    report = scan_persisted(base, kind="trace", location="trace")
    assert report.findings == [], [item.as_dict() for item in report.findings]
    nested = altered(
        base,
        output_summary={"status": "processing", "finding_count": 3},
        input_checksum=base["output_checksum"],
    )
    assert scan_persisted(nested, kind="trace", location="trace").findings == []


@pytest.mark.parametrize("value", ["1", -1, True, 1.5])
def test_a_nested_model_instance_is_revalidated_from_its_raw_values(
    trace_document: dict[str, Any], value: Any
) -> None:
    """B-AUD2-06: ``model_copy``/``model_construct`` cannot smuggle a nested value."""

    event = RunEvent.model_validate_json(
        json.dumps(altered(trace_document, duration_ms=1)), strict=True
    )
    row = {
        "event_ordinal": 1,
        "generation": None,
        "attempt": None,
        "worker": None,
        "event": event.model_copy(update={"duration_ms": value}),
    }
    with pytest.raises(Exception) as raised:
        guard_record("trace_row", row, location="trace")
    assert "trace" in str(raised.value)
    constructed = RunEvent.model_construct(
        **{**event.model_dump(), "duration_ms": value}
    )
    with pytest.raises(Exception):
        guard_record("trace_row", {**row, "event": constructed}, location="trace")
    # The same value as a stored document is refused by the reader too.
    assert scan_persisted(
        altered(trace_document, duration_ms=value), kind="trace", location="trace"
    ).findings


def test_both_entry_points_judge_a_type_the_same_way(
    trace_document: dict[str, Any],
) -> None:
    """B-AUD2-06: the Python and JSON entries agree on numbers, booleans and strings."""

    event = RunEvent.model_validate_json(
        json.dumps(altered(trace_document, duration_ms=1)), strict=True
    )
    for value, accepted in (
        (1, True),
        (0, True),
        ("1", False),
        (True, False),
        (1.5, False),
        (-1, False),
    ):
        findings = scan_persisted(
            altered(trace_document, duration_ms=value), kind="trace", location="trace"
        ).findings
        error: Exception | None = None
        try:
            guard_record(
                "trace",
                event.model_copy(update={"duration_ms": value}),
                location="trace",
            )
        except Exception as raised:  # noqa: BLE001 - the entry point refuses by raising
            error = raised
        assert bool(findings) is not accepted, (value, [i.category for i in findings])
        assert (error is None) is accepted, (value, type(error).__name__)


def test_a_boolean_duration_in_a_stored_row_is_refused_on_http_and_scan() -> None:
    """B-AUD2-06: the real Trace export never coerces the stored value."""

    with database("aud2_06_http") as fixture:
        node = api_node(fixture, "aud2_06_http")
        try:
            run_id, ident = node_run(fixture, node, key="aud2-06-http")
            clean = stored_bytes(fixture, run_id)
            poison(fixture, run_id, what="trace_duration_boolean")
            poisoned = stored_bytes(fixture, run_id)
            assert poisoned != clean
            assert_controlled_refusal(
                get(node, ident.token, f"/reviews/{run_id}/trace")
            )
            assert_scan_refuses(fixture, run_id)
            assert stored_bytes(fixture, run_id) == poisoned
        finally:
            node.close()


def test_the_pre_fix_lenient_validation_accepted_the_boolean(
    trace_document: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: the old ``strict=False`` conversion is what let it through."""

    payload = altered(trace_document, duration_ms=True)
    assert scan_persisted(payload, kind="trace", location="trace").findings

    def lax(kind: str, value: Any) -> Any:
        return record_guard.RECORD_MODELS[kind].model_validate(value, strict=False)

    monkeypatch.setattr(record_guard, "validate_stored_record", lax)
    assert scan_persisted(payload, kind="trace", location="trace").findings == []
