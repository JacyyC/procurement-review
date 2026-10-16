"""Transport-contract suite for the authenticated ``review-api.v2`` surface.

The unauthenticated ``review-api.v1`` business surface is retired: every ``/api/v1``
path returns a fixed, data-free 410.  This module therefore drives the real
authenticated app (``create_app`` bound to the isolated PostgreSQL security store from
``v1_03_support``) over HTTP and asserts the *transport* contract only: status codes,
error codes, envelope shape, tenant/Case isolation, idempotency and credential
handling.  Business rule semantics belong to the review and rules suites.

Every scenario here is required: nothing skips or xfails, and every credential is a
real store record resolved by the real authorization service.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from procurement_review.trace import canonical_checksum

from v1_03_support import (
    ALPHA_APPROVER,
    ALPHA_AUDITOR,
    ALPHA_DISABLED,
    ALPHA_NO_GRANT,
    ALPHA_OPERATOR,
    ALPHA_READER,
    ALPHA_TENANT,
    BETA_OPERATOR,
    DEMO_NO_GRANT,
    DEMO_READER,
    DEMO_SUBJECT,
    DEMO_TENANT,
    SecurityHarness,
    wait_for_status,
)

API = "/api/v2"
API_VERSION_V2 = "review-api.v2"
SHA256_ZERO = "sha256:" + "0" * 64
UNKNOWN_RUN = "run-does-not-exist"

DEMO_CASE_IDS: tuple[str, ...] = (
    "CASE-LOW-001",
    "CASE-HIGH-001",
    "CASE-CONFLICT-001",
    "CASE-UNKNOWN-001",
    "CASE-PARTIAL-FAILURE-001",
    "CASE-ALL-FAILURE-001",
    "CASE-VERSION-001",
    "CASE-CANCEL-001",
    "CASE-RUNTIME-FAILURE-001",
)
FROZEN_SCENARIOS = frozenset(
    {
        "low",
        "high",
        "conflict",
        "unknown",
        "partial_failure",
        "all_failure",
        "version_mismatch",
        "cancellable",
        "runtime_failure",
    }
)
CASE_KEYS = frozenset(
    {
        "api_version",
        "case_id",
        "case_version",
        "label",
        "case_type",
        "scenario",
        "document_version",
        "dataset_version",
        "index_version",
        "rules_version",
    }
)
LOW_CASE = "CASE-LOW-001"
LOW_VERSION = "case-low-001.v1"
CANCEL_CASE = "CASE-CANCEL-001"
CANCEL_VERSION = "case-cancel-001.v1"
ALPHA_CASE_ONE = "CASE-ALPHA-001"
ALPHA_CASE_ONE_VERSION = "case-alpha-001.v1"
ALPHA_CASE_TWO = "CASE-ALPHA-002"
BETA_CASE_ONE_VERSION = "case-beta-001.v1"
BETA_CASE_TWO = "CASE-BETA-002"

ROLE_SUBJECTS: tuple[str, ...] = (
    ALPHA_READER,
    ALPHA_OPERATOR,
    ALPHA_APPROVER,
    ALPHA_AUDITOR,
)

# The exact scope sets the seeded roles grant: a reader never gains review:create,
# review:cancel, review:decide or review:resume; an operator never gains
# review:decide or audit:read; an approver never gains review:create/review:cancel;
# an auditor never gains review:create, review:cancel or review:decide.
ROLE_SCOPES: dict[str, tuple[str, ...]] = {
    ALPHA_READER: ("case:read", "review:read"),
    ALPHA_OPERATOR: (
        "case:read",
        "review:cancel",
        "review:create",
        "review:read",
        "review:resume",
        "trace:read",
    ),
    ALPHA_APPROVER: ("audit:read", "case:read", "review:decide", "review:read"),
    ALPHA_AUDITOR: ("audit:read", "case:read", "review:read", "trace:read"),
}
DEMO_SCOPES = (
    "audit:read",
    "case:read",
    "review:cancel",
    "review:create",
    "review:decide",
    "review:read",
    "review:resume",
    "trace:read",
)

BUSINESS_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("GET", f"{API}/session"),
    ("GET", f"{API}/cases"),
    ("GET", f"{API}/reviews"),
    ("POST", f"{API}/reviews"),
    ("GET", f"{API}/reviews/{UNKNOWN_RUN}"),
    ("POST", f"{API}/reviews/{UNKNOWN_RUN}/confirm"),
    ("POST", f"{API}/reviews/{UNKNOWN_RUN}/reject"),
    ("POST", f"{API}/reviews/{UNKNOWN_RUN}/cancel"),
    ("GET", f"{API}/reviews/{UNKNOWN_RUN}/trace"),
    ("GET", f"{API}/reviews/{UNKNOWN_RUN}/audit"),
)

V1_RETIRED_PATHS: tuple[str, ...] = (
    "/api/v1",
    "/api/v1/health",
    "/api/v1/cases",
    "/api/v1/reviews",
    f"/api/v1/reviews/{UNKNOWN_RUN}",
    f"/api/v1/reviews/{UNKNOWN_RUN}/confirm",
    f"/api/v1/reviews/{UNKNOWN_RUN}/reject",
    f"/api/v1/reviews/{UNKNOWN_RUN}/cancel",
    f"/api/v1/reviews/{UNKNOWN_RUN}/trace",
    f"/api/v1/reviews/{UNKNOWN_RUN}/audit",
    "/api/v1/openapi.json",
)

# --------------------------------------------------------------------------- helpers


def _headers(
    harness: SecurityHarness,
    token: str,
    key: str | None = None,
) -> dict[str, str]:
    """One credential header, optionally carrying an idempotency key."""

    headers = harness.authorization_header(token)
    if key is not None:
        headers["Idempotency-Key"] = key
    return headers


def _create_body(case_id: str, case_version: str) -> dict[str, str]:
    return {
        "api_version": API_VERSION_V2,
        "case_id": case_id,
        "case_version": case_version,
    }


def _action_body(version: int, checksum: str | None) -> dict[str, object]:
    body: dict[str, object] = {
        "api_version": API_VERSION_V2,
        "expected_version": version,
    }
    if checksum is not None:
        body["expected_summary_checksum"] = checksum
    return body


def _create(
    client: TestClient,
    harness: SecurityHarness,
    token: str,
    *,
    case_id: str,
    case_version: str,
    key: str,
) -> Response:
    return client.post(
        f"{API}/reviews",
        headers=_headers(harness, token, key),
        json=_create_body(case_id, case_version),
    )


def _create_run(
    client: TestClient,
    harness: SecurityHarness,
    token: str,
    *,
    case_id: str,
    case_version: str,
    key: str,
) -> str:
    response = _create(
        client,
        harness,
        token,
        case_id=case_id,
        case_version=case_version,
        key=key,
    )
    assert response.status_code == 202, response.text
    return str(response.json()["run_id"])


def _action(
    client: TestClient,
    harness: SecurityHarness,
    token: str,
    run_id: str,
    action: str,
    *,
    key: str,
    version: int,
    checksum: str | None = None,
) -> Response:
    return client.post(
        f"{API}/reviews/{run_id}/{action}",
        headers=_headers(harness, token, key),
        json=_action_body(version, checksum),
    )


def _body(response: Response) -> dict[str, Any]:
    payload = response.json()
    assert isinstance(payload, dict), response.text
    return payload


def _error(response: Response) -> dict[str, Any]:
    """Return the error object of one v2 error envelope."""

    envelope = _body(response)
    assert set(envelope) == {"error"}, response.text
    error = envelope["error"]
    assert isinstance(error, dict)
    return error


def _expect_error(response: Response, status_code: int, code: str) -> dict[str, Any]:
    assert response.status_code == status_code, response.text
    error = _error(response)
    assert set(error) == {"code", "message", "request_id", "details"}, response.text
    assert error["code"] == code, response.text
    assert isinstance(error["message"], str) and error["message"], response.text
    assert str(error["request_id"]).startswith("request-"), response.text
    assert isinstance(error["details"], dict), response.text
    return error


def _without_request_id(response: Response) -> dict[str, Any]:
    error = dict(_error(response))
    error.pop("request_id")
    return error


def _case_index(response: Response) -> dict[str, dict[str, Any]]:
    assert response.status_code == 200, response.text
    return {str(item["case_id"]): item for item in _body(response)["cases"]}


# -------------------------------------------------------------------------- health


def test_health_is_anonymous_and_reports_security_store_state(
    fresh_harness: SecurityHarness,
) -> None:
    with fresh_harness.client() as client:
        response = client.get(f"{API}/health")

    assert response.status_code == 200
    assert _body(response) == {
        "api_version": "review-api.v2",
        "service": "review-api",
        "status": "ok",
        "execution_mode": "test_only",
        "security_store": "ready",
    }
    assert response.headers["x-execution-mode"] == "test_only"
    assert response.headers["x-request-id"].startswith("request-")


def test_every_business_endpoint_requires_a_credential(
    fresh_harness: SecurityHarness,
) -> None:
    with fresh_harness.client() as client:
        for method, path in BUSINESS_ENDPOINTS:
            payload: dict[str, Any] | None = None
            if method == "POST":
                payload = (
                    _create_body(ALPHA_CASE_ONE, ALPHA_CASE_ONE_VERSION)
                    if path.endswith("/reviews")
                    else _action_body(1, SHA256_ZERO)
                )
            response = client.request(method, path, json=payload)
            _expect_error(response, 401, "unauthenticated")
            assert _error(response)["details"] == {}
            for forbidden in ("run_id", "runs", "summary"):
                assert forbidden not in response.text


def test_openapi_document_describes_only_the_authenticated_surface(
    fresh_harness: SecurityHarness,
) -> None:
    with fresh_harness.client() as client:
        document = client.get(f"{API}/openapi.json")
        retired = client.get("/api/v1/openapi.json")

    assert document.status_code == 200, document.text
    schema = _body(document)
    assert schema["info"]["version"] == API_VERSION_V2
    paths = set(schema["paths"])
    assert {path for path in paths if path.startswith(API)} == {
        f"{API}/health",
        f"{API}/session",
        f"{API}/cases",
        f"{API}/reviews",
        f"{API}/reviews/{{run_id}}",
        f"{API}/reviews/{{run_id}}/confirm",
        f"{API}/reviews/{{run_id}}/reject",
        f"{API}/reviews/{{run_id}}/cancel",
        f"{API}/reviews/{{run_id}}/resume",
        f"{API}/reviews/{{run_id}}/trace",
        f"{API}/reviews/{{run_id}}/audit",
    }
    # The only surviving v1 entries are the two retired catch-alls.
    assert {path for path in paths if path.startswith("/api/v1")} == {
        "/api/v1",
        "/api/v1/{retired_path}",
    }
    assert retired.status_code == 410, retired.text


# ------------------------------------------------------------------------ identity


def test_session_returns_only_the_callers_own_verified_identity(
    fresh_harness: SecurityHarness,
) -> None:
    demo_token = fresh_harness.issue(DEMO_SUBJECT)
    alpha_token = fresh_harness.issue(ALPHA_OPERATOR)

    with fresh_harness.client() as client:
        demo = client.get(f"{API}/session", headers=_headers(fresh_harness, demo_token))
        alpha = client.get(
            f"{API}/session", headers=_headers(fresh_harness, alpha_token)
        )

    assert demo.status_code == 200, demo.text
    payload = _body(demo)
    assert payload["subject_id"] == DEMO_SUBJECT
    assert payload["tenant_id"] == DEMO_TENANT
    assert payload["credential_id"] == fresh_harness.credential_id_of(demo_token)
    assert payload["authorization_revision"] >= 1
    assert set(payload["roles"]) == {"operator", "approver"}
    assert tuple(payload["scopes"]) == DEMO_SCOPES
    issued_at = datetime.fromisoformat(payload["issued_at"])
    expires_at = datetime.fromisoformat(payload["expires_at"])
    assert expires_at > issued_at
    assert demo_token not in demo.text
    assert "token_digest" not in demo.text

    assert alpha.status_code == 200, alpha.text
    alpha_payload = _body(alpha)
    assert alpha_payload["subject_id"] == ALPHA_OPERATOR
    assert alpha_payload["tenant_id"] == ALPHA_TENANT
    assert alpha_payload["credential_id"] == fresh_harness.credential_id_of(alpha_token)
    assert alpha_payload["tenant_id"] != payload["tenant_id"]
    assert alpha_token not in alpha.text


def test_a_missing_case_grant_is_never_an_authentication_failure(
    fresh_harness: SecurityHarness,
) -> None:
    token = fresh_harness.issue(ALPHA_NO_GRANT)

    with fresh_harness.client() as client:
        session = client.get(f"{API}/session", headers=_headers(fresh_harness, token))

    assert session.status_code == 200, session.text
    payload = _body(session)
    assert payload["subject_id"] == ALPHA_NO_GRANT
    assert payload["tenant_id"] == ALPHA_TENANT
    assert payload["scopes"] == ["case:read", "review:read"]


# --------------------------------------------------------------------- credentials


def test_unusable_credentials_are_always_401_unauthenticated(
    fresh_harness: SecurityHarness,
) -> None:
    valid = fresh_harness.issue(DEMO_SUBJECT)
    expired = fresh_harness.issue(DEMO_SUBJECT, ttl_seconds=-60)
    revoked = fresh_harness.issue(DEMO_SUBJECT)
    assert fresh_harness.store.revoke_credential(
        credential_id=fresh_harness.credential_id_of(revoked),
        revoked_at=datetime.now(UTC),
        reason="transport-contract-test",
    )
    disabled = fresh_harness.issue(ALPHA_DISABLED)
    unknown = f"rvk1.cred-{'0' * 32}.{secrets.token_urlsafe(24)}"

    probes: tuple[tuple[str, str | None, dict[str, str]], ...] = (
        ("missing", None, {}),
        ("empty", None, {"Authorization": "Bearer "}),
        ("wrong-scheme", valid, {"Authorization": f"Token {valid}"}),
        ("malformed", "nonsense", {"Authorization": "Bearer nonsense"}),
        ("unknown", unknown, {"Authorization": f"Bearer {unknown}"}),
        ("expired", expired, {"Authorization": f"Bearer {expired}"}),
        ("revoked", revoked, {"Authorization": f"Bearer {revoked}"}),
        ("disabled-subject", disabled, {"Authorization": f"Bearer {disabled}"}),
    )

    with fresh_harness.client() as client:
        responses = [
            (label, presented, client.get(f"{API}/cases", headers=header))
            for label, presented, header in probes
        ]
        still_valid = client.get(
            f"{API}/session", headers=_headers(fresh_harness, valid)
        )

    for label, presented, response in responses:
        _expect_error(response, 401, "unauthenticated")
        assert _error(response)["details"] == {}
        if presented is not None:
            assert presented not in response.text, label
    assert still_valid.status_code == 200, still_valid.text
    assert _body(still_valid)["subject_id"] == DEMO_SUBJECT


# --------------------------------------------------------------------------- roles


ROLE_DENIALS = (
    pytest.param(ALPHA_READER, "POST", f"{API}/reviews", id="reader-cannot-create"),
    pytest.param(
        ALPHA_READER,
        "POST",
        f"{API}/reviews/{{run_id}}/confirm",
        id="reader-cannot-decide",
    ),
    pytest.param(
        ALPHA_READER,
        "POST",
        f"{API}/reviews/{{run_id}}/reject",
        id="reader-cannot-reject",
    ),
    pytest.param(
        ALPHA_READER,
        "POST",
        f"{API}/reviews/{{run_id}}/cancel",
        id="reader-cannot-cancel",
    ),
    pytest.param(
        ALPHA_READER,
        "POST",
        f"{API}/reviews/{{run_id}}/resume",
        id="reader-cannot-resume",
    ),
    pytest.param(
        ALPHA_APPROVER,
        "POST",
        f"{API}/reviews/{{run_id}}/resume",
        id="approver-cannot-resume",
    ),
    pytest.param(
        ALPHA_OPERATOR,
        "POST",
        f"{API}/reviews/{{run_id}}/confirm",
        id="operator-cannot-decide",
    ),
    pytest.param(
        ALPHA_OPERATOR,
        "GET",
        f"{API}/reviews/{{run_id}}/audit",
        id="operator-cannot-read-audit",
    ),
    pytest.param(ALPHA_APPROVER, "POST", f"{API}/reviews", id="approver-cannot-create"),
    pytest.param(
        ALPHA_APPROVER,
        "POST",
        f"{API}/reviews/{{run_id}}/cancel",
        id="approver-cannot-cancel",
    ),
    pytest.param(
        ALPHA_APPROVER,
        "GET",
        f"{API}/reviews/{{run_id}}/trace",
        id="approver-cannot-read-trace",
    ),
    pytest.param(ALPHA_AUDITOR, "POST", f"{API}/reviews", id="auditor-cannot-create"),
    pytest.param(
        ALPHA_AUDITOR,
        "POST",
        f"{API}/reviews/{{run_id}}/resume",
        id="auditor-cannot-resume",
    ),
    pytest.param(
        ALPHA_AUDITOR,
        "POST",
        f"{API}/reviews/{{run_id}}/cancel",
        id="auditor-cannot-cancel",
    ),
    pytest.param(
        ALPHA_AUDITOR,
        "POST",
        f"{API}/reviews/{{run_id}}/confirm",
        id="auditor-cannot-decide",
    ),
)


@pytest.mark.parametrize(("subject_id", "method", "path"), ROLE_DENIALS)
def test_unentitled_roles_receive_403_insufficient_scope(
    fresh_harness: SecurityHarness,
    subject_id: str,
    method: str,
    path: str,
) -> None:
    """Scope is checked before the target object exists, so the run id is irrelevant."""

    token = fresh_harness.issue(subject_id)
    payload: dict[str, Any] | None = None
    if method == "POST":
        if path.endswith("/reviews"):
            payload = _create_body(ALPHA_CASE_ONE, ALPHA_CASE_ONE_VERSION)
        elif path.endswith("/resume"):
            payload = {
                "api_version": API_VERSION_V2,
                "expected_version": 1,
                "idempotency_key": "role-denial-resume",
            }
        else:
            payload = _action_body(1, SHA256_ZERO)

    with fresh_harness.client() as client:
        response = client.request(
            method,
            path.format(run_id=UNKNOWN_RUN),
            headers=_headers(fresh_harness, token, "role-denial"),
            json=payload,
        )

    error = _expect_error(response, 403, "insufficient_scope")
    assert error["details"] == {}
    assert "run_id" not in response.text


def test_each_seeded_role_reaches_exactly_its_entitled_surface(
    fresh_harness: SecurityHarness,
) -> None:
    tokens = {subject: fresh_harness.issue(subject) for subject in ROLE_SUBJECTS}

    with fresh_harness.client() as client:
        for subject_id in ROLE_SUBJECTS:
            session = client.get(
                f"{API}/session", headers=_headers(fresh_harness, tokens[subject_id])
            )
            assert session.status_code == 200, session.text
            assert tuple(_body(session)["scopes"]) == ROLE_SCOPES[subject_id]
        assert {
            subject
            for subject, scopes in ROLE_SCOPES.items()
            if "review:resume" in scopes
        } == {ALPHA_OPERATOR}

        for subject_id in ROLE_SUBJECTS:
            cases = client.get(
                f"{API}/cases", headers=_headers(fresh_harness, tokens[subject_id])
            )
            assert cases.status_code == 200, cases.text
            assert set(_case_index(cases)) == {ALPHA_CASE_ONE, ALPHA_CASE_TWO}

        operator = _headers(fresh_harness, tokens[ALPHA_OPERATOR])
        created = client.post(
            f"{API}/reviews",
            headers={**operator, "Idempotency-Key": "entitled-create"},
            json=_create_body(ALPHA_CASE_ONE, ALPHA_CASE_ONE_VERSION),
        )
        assert created.status_code == 202, created.text
        run_id = str(_body(created)["run_id"])
        assert _body(created)["request"]["initiator"] == ALPHA_OPERATOR

        detail = client.get(f"{API}/reviews/{run_id}", headers=operator)
        assert detail.status_code == 200, detail.text
        review_list = client.get(f"{API}/reviews", headers=operator)
        assert review_list.status_code == 200, review_list.text
        assert run_id in {str(item["run_id"]) for item in _body(review_list)["runs"]}
        operator_trace = client.get(f"{API}/reviews/{run_id}/trace", headers=operator)
        assert operator_trace.status_code == 200, operator_trace.text

        for subject_id in (ALPHA_APPROVER, ALPHA_AUDITOR):
            audit = client.get(
                f"{API}/reviews/{run_id}/audit",
                headers=_headers(fresh_harness, tokens[subject_id]),
            )
            assert audit.status_code == 200, audit.text
            assert _body(audit)["run_id"] == run_id
        auditor_trace = client.get(
            f"{API}/reviews/{run_id}/trace",
            headers=_headers(fresh_harness, tokens[ALPHA_AUDITOR]),
        )
        assert auditor_trace.status_code == 200, auditor_trace.text

        demo_reader = client.get(
            f"{API}/cases",
            headers=_headers(fresh_harness, fresh_harness.issue(DEMO_READER)),
        )
        assert set(_case_index(demo_reader)) == set(DEMO_CASE_IDS)

        no_grant = _headers(fresh_harness, fresh_harness.issue(DEMO_NO_GRANT))
        assert _body(client.get(f"{API}/cases", headers=no_grant))["cases"] == []
        assert _body(client.get(f"{API}/reviews", headers=no_grant))["runs"] == []


# --------------------------------------------------------------------------- cases


def test_case_listing_is_grant_scoped_and_keeps_tenants_apart(
    fresh_harness: SecurityHarness,
) -> None:
    demo = _headers(fresh_harness, fresh_harness.issue(DEMO_SUBJECT))
    alpha = _headers(fresh_harness, fresh_harness.issue(ALPHA_OPERATOR))
    beta = _headers(fresh_harness, fresh_harness.issue(BETA_OPERATOR))

    with fresh_harness.client() as client:
        demo_response = client.get(f"{API}/cases", headers=demo)
        alpha_cases = _case_index(client.get(f"{API}/cases", headers=alpha))
        beta_cases = _case_index(client.get(f"{API}/cases", headers=beta))
        cross_tenant_version = client.post(
            f"{API}/reviews",
            headers={**beta, "Idempotency-Key": "beta-alpha-version"},
            json=_create_body(ALPHA_CASE_ONE, ALPHA_CASE_ONE_VERSION),
        )

    assert demo_response.status_code == 200, demo_response.text
    demo_body = _body(demo_response)
    assert demo_body["api_version"] == API_VERSION_V2
    assert demo_body["execution_mode"] == "test_only"
    assert len(demo_body["cases"]) == 9
    demo_cases = {str(item["case_id"]): item for item in demo_body["cases"]}
    assert set(demo_cases) == set(DEMO_CASE_IDS)
    for case_id, item in demo_cases.items():
        assert set(item) == CASE_KEYS
        assert item["case_version"] == f"{case_id.lower()}.v1"
        assert item["rules_version"] == "2.0.0"
        assert item["document_version"] == "retrieval-fixture-v1"
        assert item["dataset_version"] == "synthetic-procurement.v1"
        assert item["index_version"] == "retrieval-index-v1"
        assert item["scenario"] in FROZEN_SCENARIOS
        assert item["case_type"] in {"purchase_order", "contract_change"}

    assert set(alpha_cases) == {ALPHA_CASE_ONE, ALPHA_CASE_TWO}
    assert alpha_cases[ALPHA_CASE_ONE]["case_version"] == ALPHA_CASE_ONE_VERSION
    assert set(beta_cases) == {ALPHA_CASE_ONE, BETA_CASE_TWO}
    assert beta_cases[ALPHA_CASE_ONE]["case_version"] == BETA_CASE_ONE_VERSION
    assert (
        beta_cases[ALPHA_CASE_ONE]["case_version"]
        != alpha_cases[ALPHA_CASE_ONE]["case_version"]
    )
    assert beta_cases[ALPHA_CASE_ONE]["label"] != alpha_cases[ALPHA_CASE_ONE]["label"]
    assert not {ALPHA_CASE_ONE, ALPHA_CASE_TWO, BETA_CASE_TWO} & set(demo_cases)

    error = _expect_error(cross_tenant_version, 409, "case_version_mismatch")
    assert error["details"] == {"registered_case_version": BETA_CASE_ONE_VERSION}


# ----------------------------------------------------------------------- not found


def test_review_list_is_filtered_by_tenant_and_case_grant(
    fresh_harness: SecurityHarness,
) -> None:
    demo = _headers(fresh_harness, fresh_harness.issue(DEMO_SUBJECT))
    alpha_token = fresh_harness.issue(ALPHA_OPERATOR)
    alpha = _headers(fresh_harness, alpha_token)
    reader = _headers(fresh_harness, fresh_harness.issue(ALPHA_READER))
    no_grant = _headers(fresh_harness, fresh_harness.issue(ALPHA_NO_GRANT))

    with fresh_harness.client() as client:
        opening_lists = {
            "demo": client.get(f"{API}/reviews", headers=demo),
            "alpha": client.get(f"{API}/reviews", headers=alpha),
        }
        run_id = _create_run(
            client,
            fresh_harness,
            alpha_token,
            case_id=ALPHA_CASE_ONE,
            case_version=ALPHA_CASE_ONE_VERSION,
            key="list-scope-run",
        )
        later = {
            "demo": client.get(f"{API}/reviews", headers=demo),
            "alpha": client.get(f"{API}/reviews", headers=alpha),
            "granted-reader": client.get(f"{API}/reviews", headers=reader),
            "ungranted-reader": client.get(f"{API}/reviews", headers=no_grant),
        }

    for label, response in opening_lists.items():
        assert response.status_code == 200, label
        assert _body(response)["runs"] == [], label
        assert set(_body(response)) == {"api_version", "runs"}, label

    for label, response in later.items():
        assert response.status_code == 200, label
    for label in ("alpha", "granted-reader"):
        assert run_id in {str(item["run_id"]) for item in _body(later[label])["runs"]}
    assert run_id not in {str(item["run_id"]) for item in _body(later["demo"])["runs"]}
    assert _body(later["ungranted-reader"])["runs"] == []


def test_missing_foreign_and_ungranted_objects_share_one_404_body(
    fresh_harness: SecurityHarness,
) -> None:
    demo_token = fresh_harness.issue(DEMO_SUBJECT)
    alpha_token = fresh_harness.issue(ALPHA_OPERATOR)
    demo = _headers(fresh_harness, demo_token)

    with fresh_harness.client() as client:
        alpha_run = _create_run(
            client,
            fresh_harness,
            alpha_token,
            case_id=ALPHA_CASE_ONE,
            case_version=ALPHA_CASE_ONE_VERSION,
            key="not-found-alpha-run",
        )
        probes = {
            "unknown-run": client.get(f"{API}/reviews/{UNKNOWN_RUN}", headers=demo),
            "foreign-run": client.get(f"{API}/reviews/{alpha_run}", headers=demo),
            "foreign-case": _create(
                client,
                fresh_harness,
                demo_token,
                case_id=ALPHA_CASE_ONE,
                case_version=ALPHA_CASE_ONE_VERSION,
                key="not-found-foreign-case",
            ),
            "unknown-case": _create(
                client,
                fresh_harness,
                demo_token,
                case_id="CASE-DOES-NOT-EXIST",
                case_version="case-does-not-exist.v1",
                key="not-found-unknown-case",
            ),
        }
        fresh_harness.store.set_case_grant(
            tenant_id=ALPHA_TENANT,
            case_id=ALPHA_CASE_ONE,
            subject_id=ALPHA_OPERATOR,
            granted=False,
            occurred_at=datetime.now(UTC),
        )
        try:
            probes["ungranted-case"] = _create(
                client,
                fresh_harness,
                alpha_token,
                case_id=ALPHA_CASE_ONE,
                case_version=ALPHA_CASE_ONE_VERSION,
                key="not-found-ungranted-case",
            )
        finally:
            fresh_harness.store.set_case_grant(
                tenant_id=ALPHA_TENANT,
                case_id=ALPHA_CASE_ONE,
                subject_id=ALPHA_OPERATOR,
                granted=True,
                occurred_at=datetime.now(UTC),
            )

    shapes: dict[str, dict[str, Any]] = {}
    request_ids: set[str] = set()
    for label, response in probes.items():
        error = _expect_error(response, 404, "resource_not_found")
        assert error["details"] == {}, label
        request_ids.add(str(error["request_id"]))
        shapes[label] = _without_request_id(response)
    assert len(request_ids) == len(probes)
    reference = shapes["unknown-run"]
    for label, shape in shapes.items():
        assert shape == reference, label


def test_an_ungranted_case_is_not_reported_as_an_authentication_failure(
    fresh_harness: SecurityHarness,
) -> None:
    """An ungranted Case in the caller's own tenant must stay a 404, never a 401.

    ``subject-alpha-nogrant`` holds ``review:read`` but no Case grant, so reading a Run
    of ``CASE-ALPHA-001`` must be indistinguishable from reading an unknown Run.
    """

    alpha_token = fresh_harness.issue(ALPHA_OPERATOR)
    nogrant_token = fresh_harness.issue(ALPHA_NO_GRANT)
    nogrant = _headers(fresh_harness, nogrant_token)

    with fresh_harness.client() as client:
        alpha_run = _create_run(
            client,
            fresh_harness,
            alpha_token,
            case_id=ALPHA_CASE_ONE,
            case_version=ALPHA_CASE_ONE_VERSION,
            key="nogrant-alpha-run",
        )
        ungranted = client.get(f"{API}/reviews/{alpha_run}", headers=nogrant)
        unknown = client.get(f"{API}/reviews/{UNKNOWN_RUN}", headers=nogrant)

    assert _error(ungranted)["code"] != "unauthenticated", (
        "an ungranted Case in the caller's own tenant was reported as a credential "
        f"failure instead of a missing resource: {ungranted.text}"
    )
    _expect_error(ungranted, 404, "resource_not_found")
    _expect_error(unknown, 404, "resource_not_found")
    assert _without_request_id(ungranted) == _without_request_id(unknown)


# --------------------------------------------------------------------- idempotency


def test_create_replay_and_idempotency_conflict(
    fresh_harness: SecurityHarness,
) -> None:
    token = fresh_harness.issue(DEMO_SUBJECT)
    body = _create_body(LOW_CASE, LOW_VERSION)
    headers = _headers(fresh_harness, token)

    with fresh_harness.client() as client:
        created = _create(
            client,
            fresh_harness,
            token,
            case_id=LOW_CASE,
            case_version=LOW_VERSION,
            key="create-replay",
        )
        replay = _create(
            client,
            fresh_harness,
            token,
            case_id=LOW_CASE,
            case_version=LOW_VERSION,
            key="create-replay",
        )
        conflicting = _create(
            client,
            fresh_harness,
            token,
            case_id="CASE-HIGH-001",
            case_version="case-high-001.v1",
            key="create-replay",
        )
        body_keyed = client.post(
            f"{API}/reviews",
            headers=headers,
            json={**body, "idempotency_key": "body-key-only"},
        )
        body_keyed_replay = client.post(
            f"{API}/reviews",
            headers=headers,
            json={**body, "idempotency_key": "body-key-only"},
        )
        mismatched_key = client.post(
            f"{API}/reviews",
            headers=_headers(fresh_harness, token, "header-key"),
            json={**body, "idempotency_key": "body-key-only"},
        )

    assert created.status_code == 202, created.text
    assert created.headers["location"] == f"{API}/reviews/{_body(created)['run_id']}"
    assert replay.status_code == 200, replay.text
    assert _body(replay)["run_id"] == _body(created)["run_id"]
    assert _body(replay)["trace_id"] == _body(created)["trace_id"]

    _expect_error(conflicting, 409, "idempotency_conflict")
    assert body_keyed.status_code == 202, body_keyed.text
    assert body_keyed_replay.status_code == 200, body_keyed_replay.text
    assert _body(body_keyed_replay)["run_id"] == _body(body_keyed)["run_id"]
    _expect_error(mismatched_key, 409, "idempotency_conflict")


def test_create_rejects_wrong_versions_reserved_fields_and_bad_requests(
    fresh_harness: SecurityHarness,
) -> None:
    token = fresh_harness.issue(DEMO_SUBJECT)
    body = _create_body(LOW_CASE, LOW_VERSION)
    headers = _headers(fresh_harness, token)

    with fresh_harness.client() as client:
        wrong_version = _create(
            client,
            fresh_harness,
            token,
            case_id=LOW_CASE,
            case_version="case-low-001.v999",
            key="wrong-case-version",
        )
        missing_key = client.post(f"{API}/reviews", headers=headers, json=body)
        wrong_api = client.post(
            f"{API}/reviews",
            headers=_headers(fresh_harness, token, "wrong-api-version"),
            json={**body, "api_version": "review-api.v999"},
        )
        reserved_initiator = client.post(
            f"{API}/reviews",
            headers=_headers(fresh_harness, token, "reserved-initiator"),
            json={**body, "initiator": "mallory"},
        )
        run_id = _create_run(
            client,
            fresh_harness,
            token,
            case_id=LOW_CASE,
            case_version=LOW_VERSION,
            key="reserved-action-run",
        )
        pending = wait_for_status(client, run_id, headers, "pending_confirmation")
        reserved_actor = client.post(
            f"{API}/reviews/{run_id}/confirm",
            headers=_headers(fresh_harness, token, "reserved-actor"),
            json={
                "api_version": API_VERSION_V2,
                "human_actor": "mallory",
                "expected_version": pending["version"],
                "expected_summary_checksum": pending["summary_checksum"],
            },
        )

    error = _expect_error(wrong_version, 409, "case_version_mismatch")
    assert error["details"] == {"registered_case_version": LOW_VERSION}
    _expect_error(missing_key, 422, "idempotency_key_required")
    _expect_error(wrong_api, 409, "api_version_mismatch")

    _expect_error(reserved_initiator, 422, "reserved_field_rejected")
    assert "initiator" not in reserved_initiator.text
    assert "mallory" not in reserved_initiator.text
    _expect_error(reserved_actor, 422, "reserved_field_rejected")
    assert "human_actor" not in reserved_actor.text
    assert "mallory" not in reserved_actor.text


# --------------------------------------------------------------------- happy path


def test_happy_path_create_wait_decide_records_one_receipt_and_one_audit_event(
    fresh_harness: SecurityHarness,
) -> None:
    token = fresh_harness.issue(DEMO_SUBJECT)
    headers = _headers(fresh_harness, token)

    with fresh_harness.client() as client:
        created = _create(
            client,
            fresh_harness,
            token,
            case_id=LOW_CASE,
            case_version=LOW_VERSION,
            key="happy-create",
        )
        assert created.status_code == 202, created.text
        run_id = str(_body(created)["run_id"])
        pending = wait_for_status(client, run_id, headers, "pending_confirmation")
        confirmed = _action(
            client,
            fresh_harness,
            token,
            run_id,
            "confirm",
            key="happy-confirm",
            version=pending["version"],
            checksum=pending["summary_checksum"],
        )
        audit = client.get(f"{API}/reviews/{run_id}/audit", headers=headers)
        trace = client.get(f"{API}/reviews/{run_id}/trace", headers=headers)
        listed = client.get(f"{API}/reviews", headers=headers)

    opening = _body(created)
    assert opening["status"] == "processing"
    assert opening["version"] == opening["revision"] == 1
    assert opening["execution_mode"] == "test_only"
    assert opening["summary"] is None
    assert opening["summary_checksum"] is None
    assert opening["receipt"] is None
    assert opening["request"]["initiator"] == DEMO_SUBJECT
    assert opening["request"]["case_id"] == LOW_CASE
    assert opening["request"]["document_version"] == "retrieval-fixture-v1"
    assert opening["request"]["dataset_version"] == "synthetic-procurement.v1"
    assert opening["request"]["purchase_order_id"] == "PO-001"

    assert pending["status"] == "pending_confirmation"
    assert pending["version"] == pending["revision"] == 2
    assert pending["summary"]["recommendation"] == "recommend_approve"
    assert pending["summary"]["rules_version"] == "2.0.0"
    assert pending["summary"]["is_final_decision"] is False
    assert pending["summary"]["human_review_status"] == "pending_confirmation"
    assert len(pending["evidence_refs"]) == 2
    assert len(pending["business_facts"]) == 5
    assert pending["branch_failures"] == []
    assert pending["receipt"] is None
    assert pending["summary_checksum"] == canonical_checksum(pending["summary"])

    assert confirmed.status_code == 200, confirmed.text
    decision = _body(confirmed)
    assert decision["status"] == "confirmed"
    assert decision["version"] == decision["revision"] == 3
    assert decision["summary_checksum"] == pending["summary_checksum"]
    assert decision["audit_event_count"] == 1
    receipt = decision["receipt"]
    assert receipt["run_id"] == run_id
    assert receipt["case_id"] == LOW_CASE
    assert receipt["human_actor"] == DEMO_SUBJECT
    assert receipt["decision"] == "confirm_recommendation"
    assert receipt["reviewed_recommendation"] == "recommend_approve"
    assert receipt["simulation_mode"] == "local_test_only"
    assert receipt["simulation_status"] == "local_simulation_recorded"
    assert receipt["external_write_performed"] is False
    assert receipt["idempotency_key_digest"].startswith("sha256:")
    assert "happy-confirm" not in confirmed.text
    # A v2 Run payload never advertises the retired API version.
    assert "review-api.v1" not in confirmed.text

    assert audit.status_code == 200, audit.text
    events = _body(audit)["events"]
    assert len(events) == 1
    assert events[0]["actor"] == DEMO_SUBJECT
    assert events[0]["action_type"] == "confirm"
    assert events[0]["receipt_id"] == receipt["receipt_id"]
    assert events[0]["previous_status"] == "pending_confirmation"
    assert events[0]["new_status"] == "confirmed"
    assert events[0]["external_write_performed"] is False

    assert trace.status_code == 200, trace.text
    envelope = _body(trace)["trace"]
    assert envelope["complete"] is True
    assert envelope["events"][-1]["event_type"] == "run_completed"
    assert "confirmation_recorded" in {
        event["event_type"] for event in envelope["events"]
    }

    assert run_id in {str(item["run_id"]) for item in _body(listed)["runs"]}
    assert token not in confirmed.text
    assert token not in audit.text


def test_decision_replay_does_not_duplicate_receipts_audit_or_revision(
    fresh_harness: SecurityHarness,
) -> None:
    token = fresh_harness.issue(DEMO_SUBJECT)
    headers = _headers(fresh_harness, token)

    with fresh_harness.client() as client:
        run_id = _create_run(
            client,
            fresh_harness,
            token,
            case_id=LOW_CASE,
            case_version=LOW_VERSION,
            key="replay-create",
        )
        pending = wait_for_status(client, run_id, headers, "pending_confirmation")
        first = _action(
            client,
            fresh_harness,
            token,
            run_id,
            "confirm",
            key="replay-decide",
            version=pending["version"],
            checksum=pending["summary_checksum"],
        )
        assert first.status_code == 200, first.text
        replay = _action(
            client,
            fresh_harness,
            token,
            run_id,
            "confirm",
            key="replay-decide",
            version=pending["version"],
            checksum=pending["summary_checksum"],
        )
        conflicting_replay = _action(
            client,
            fresh_harness,
            token,
            run_id,
            "confirm",
            key="replay-decide",
            version=pending["version"],
            checksum=SHA256_ZERO,
        )
        new_key = _action(
            client,
            fresh_harness,
            token,
            run_id,
            "confirm",
            key="replay-decide-again",
            version=int(_body(first)["version"]),
            checksum=pending["summary_checksum"],
        )
        reject_terminal = _action(
            client,
            fresh_harness,
            token,
            run_id,
            "reject",
            key="replay-reject-terminal",
            version=int(_body(first)["version"]),
            checksum=pending["summary_checksum"],
        )
        audit = client.get(f"{API}/reviews/{run_id}/audit", headers=headers)

    assert replay.status_code == 200, replay.text
    decided = _body(first)
    replayed = _body(replay)
    assert replayed["receipt"] == decided["receipt"]
    assert replayed["version"] == decided["version"]
    assert replayed["revision"] == decided["revision"]
    assert replayed["audit_event_count"] == decided["audit_event_count"] == 1

    events = _body(audit)["events"]
    assert len(events) == 1
    assert [event["receipt_id"] for event in events if event["receipt_id"]] == [
        decided["receipt"]["receipt_id"]
    ]

    _expect_error(conflicting_replay, 409, "idempotency_conflict")
    _expect_error(new_key, 409, "invalid_state_transition")
    _expect_error(reject_terminal, 409, "invalid_state_transition")


def test_decision_guards_reject_stale_inputs_without_mutating_the_run(
    fresh_harness: SecurityHarness,
) -> None:
    token = fresh_harness.issue(DEMO_SUBJECT)
    headers = _headers(fresh_harness, token)

    with fresh_harness.client() as client:
        run_id = _create_run(
            client,
            fresh_harness,
            token,
            case_id=LOW_CASE,
            case_version=LOW_VERSION,
            key="guards-create",
        )
        pending = wait_for_status(client, run_id, headers, "pending_confirmation")
        stale_checksum = _action(
            client,
            fresh_harness,
            token,
            run_id,
            "reject",
            key="guards-stale-checksum",
            version=pending["version"],
            checksum=SHA256_ZERO,
        )
        no_checksum = _action(
            client,
            fresh_harness,
            token,
            run_id,
            "reject",
            key="guards-no-checksum",
            version=pending["version"],
        )
        no_version = client.post(
            f"{API}/reviews/{run_id}/cancel",
            headers=_headers(fresh_harness, token, "guards-no-version"),
            json={"api_version": API_VERSION_V2},
        )
        stale_version = _action(
            client,
            fresh_harness,
            token,
            run_id,
            "cancel",
            key="guards-stale-version",
            version=pending["version"] - 1,
        )
        wrong_type = client.post(
            f"{API}/reviews/{run_id}/reject",
            headers=_headers(fresh_harness, token, "guards-wrong-type"),
            json={
                "api_version": API_VERSION_V2,
                "expected_version": str(pending["version"]),
                "expected_summary_checksum": pending["summary_checksum"],
            },
        )
        after = client.get(f"{API}/reviews/{run_id}", headers=headers)
        confirmed = _action(
            client,
            fresh_harness,
            token,
            run_id,
            "confirm",
            key="guards-confirm",
            version=pending["version"],
            checksum=pending["summary_checksum"],
        )

    error = _expect_error(stale_checksum, 409, "summary_checksum_mismatch")
    assert error["details"] == {"summary_checksum": pending["summary_checksum"]}
    _expect_error(no_checksum, 422, "expected_summary_checksum_required")
    _expect_error(no_version, 422, "expected_version_required")
    error = _expect_error(stale_version, 409, "version_conflict")
    assert error["details"] == {
        "expected_version": pending["version"] - 1,
        "current_version": pending["version"],
    }
    _expect_error(wrong_type, 422, "invalid_request")

    untouched = _body(after)
    assert untouched["status"] == "pending_confirmation"
    assert untouched["version"] == pending["version"]
    assert untouched["audit_event_count"] == 0
    assert confirmed.status_code == 200, confirmed.text
    assert _body(confirmed)["status"] == "confirmed"


# ------------------------------------------------------------------------- cancel


def test_cancel_while_processing_never_records_a_receipt(
    fresh_harness: SecurityHarness,
) -> None:
    token = fresh_harness.issue(DEMO_SUBJECT)
    headers = _headers(fresh_harness, token)

    with fresh_harness.client() as client:
        created = _create(
            client,
            fresh_harness,
            token,
            case_id=CANCEL_CASE,
            case_version=CANCEL_VERSION,
            key="cancel-create",
        )
        assert created.status_code == 202, created.text
        run_id = str(_body(created)["run_id"])
        assert _body(created)["status"] == "processing"

        in_flight = _body(client.get(f"{API}/reviews/{run_id}", headers=headers))
        assert in_flight["status"] == "processing"
        assert in_flight["receipt"] is None

        cancelled = _action(
            client,
            fresh_harness,
            token,
            run_id,
            "cancel",
            key="cancel-action",
            version=in_flight["version"],
        )
        confirmed = _action(
            client,
            fresh_harness,
            token,
            run_id,
            "confirm",
            key="cancel-confirm",
            version=in_flight["version"] + 1,
            checksum=SHA256_ZERO,
        )
        audit = client.get(f"{API}/reviews/{run_id}/audit", headers=headers)
        trace = client.get(f"{API}/reviews/{run_id}/trace", headers=headers)
        settled = client.get(f"{API}/reviews/{run_id}", headers=headers)

    assert cancelled.status_code == 200, cancelled.text
    body = _body(cancelled)
    assert body["status"] == "cancelled"
    assert body["receipt"] is None
    assert body["version"] == body["revision"] == in_flight["version"] + 1
    assert body["audit_event_count"] == 1
    assert body["error_code"] is None

    _expect_error(confirmed, 409, "invalid_state_transition")

    events = _body(audit)["events"]
    assert len(events) == 1
    assert events[0]["action_type"] == "cancel"
    assert events[0]["actor"] == DEMO_SUBJECT
    assert events[0]["receipt_id"] is None
    assert events[0]["previous_status"] == "processing"
    assert events[0]["new_status"] == "cancelled"

    envelope = _body(trace)["trace"]
    assert envelope["complete"] is True
    assert envelope["events"][-1]["event_type"] == "run_cancelled"
    assert "step_cancelled" in {event["event_type"] for event in envelope["events"]}

    assert _body(settled)["status"] == "cancelled"
    assert _body(settled)["receipt"] is None


def test_failure_scenarios_stay_visible_and_never_record_receipts(
    fresh_harness: SecurityHarness,
) -> None:
    token = fresh_harness.issue(DEMO_SUBJECT)
    headers = _headers(fresh_harness, token)

    with fresh_harness.client() as client:
        runs: dict[str, dict[str, Any]] = {}
        for case_id, case_version in (
            ("CASE-PARTIAL-FAILURE-001", "case-partial-failure-001.v1"),
            ("CASE-ALL-FAILURE-001", "case-all-failure-001.v1"),
            ("CASE-VERSION-001", "case-version-001.v1"),
        ):
            run_id = _create_run(
                client,
                fresh_harness,
                token,
                case_id=case_id,
                case_version=case_version,
                key=f"failure-{case_id}",
            )
            runs[case_id] = wait_for_status(
                client, run_id, headers, "pending_human_review"
            )

    partial = runs["CASE-PARTIAL-FAILURE-001"]
    partial_workers = {item["worker"] for item in partial["branch_failures"]}
    assert partial["status"] == "pending_human_review"
    assert partial["summary"]["recommendation"] == "insufficient_information"
    assert "contract" in partial_workers
    assert "policy" not in partial_workers
    assert partial["receipt"] is None
    assert partial["summary_checksum"] == canonical_checksum(partial["summary"])

    everything = runs["CASE-ALL-FAILURE-001"]
    assert everything["status"] == "pending_human_review"
    assert everything["summary"]["recommendation"] == "insufficient_information"
    assert {item["worker"] for item in everything["branch_failures"]} == {
        "contract",
        "policy",
        "procurement_data",
    }
    assert everything["receipt"] is None
    assert everything["evidence_refs"] == []

    mismatched = runs["CASE-VERSION-001"]
    assert mismatched["status"] == "pending_human_review"
    assert mismatched["summary"]["recommendation"] == "insufficient_information"
    assert mismatched["branch_failures"]
    assert mismatched["receipt"] is None


def test_cross_tenant_idempotency_keys_are_isolated(
    fresh_harness: SecurityHarness,
) -> None:
    demo_token = fresh_harness.issue(DEMO_SUBJECT)
    alpha_token = fresh_harness.issue(ALPHA_OPERATOR)
    beta_token = fresh_harness.issue(BETA_OPERATOR)

    with fresh_harness.client() as client:
        demo_run = _create(
            client,
            fresh_harness,
            demo_token,
            case_id=LOW_CASE,
            case_version=LOW_VERSION,
            key="shared-key",
        )
        alpha_run = _create(
            client,
            fresh_harness,
            alpha_token,
            case_id=ALPHA_CASE_ONE,
            case_version=ALPHA_CASE_ONE_VERSION,
            key="shared-key",
        )
        beta_run = _create(
            client,
            fresh_harness,
            beta_token,
            case_id=ALPHA_CASE_ONE,
            case_version=BETA_CASE_ONE_VERSION,
            key="shared-key",
        )

    for response in (demo_run, alpha_run, beta_run):
        assert response.status_code == 202, response.text
    run_ids = {str(_body(item)["run_id"]) for item in (demo_run, alpha_run, beta_run)}
    assert len(run_ids) == 3


# ------------------------------------------------------------------ retired v1 API


def test_every_v1_path_is_retired_without_business_data(
    fresh_harness: SecurityHarness,
) -> None:
    token = fresh_harness.issue(DEMO_SUBJECT)
    headers = _headers(fresh_harness, token)

    with fresh_harness.client() as client:
        responses: list[tuple[str, Response]] = []
        for path in V1_RETIRED_PATHS:
            responses.append((f"anonymous GET {path}", client.get(path)))
            responses.append(
                (f"authenticated GET {path}", client.get(path, headers=headers))
            )
            responses.append(
                (
                    f"authenticated POST {path}",
                    client.post(
                        path,
                        headers={**headers, "Idempotency-Key": "v1-retired"},
                        json=_create_body(LOW_CASE, LOW_VERSION),
                    ),
                )
            )

    for label, response in responses:
        assert response.status_code == 410, f"{label}: {response.text}"
        error = _error(response)
        assert error["code"] == "api_version_retired", label
        assert set(error) == {"code", "message", "request_id", "details"}, label
        assert error["request_id"] == "retired", label
        assert error["details"] == {"current_api_version": API_VERSION_V2}, label
        for forbidden in ('"run_id"', '"runs"', '"summary"', '"cases"', '"receipt"'):
            assert forbidden not in response.text, label


def test_no_response_body_ever_echoes_the_presented_credential(
    fresh_harness: SecurityHarness,
) -> None:
    token = fresh_harness.issue(DEMO_SUBJECT)
    secret = token.split(".")[2]
    headers = _headers(fresh_harness, token)

    with fresh_harness.client() as client:
        run_id = _create_run(
            client,
            fresh_harness,
            token,
            case_id=LOW_CASE,
            case_version=LOW_VERSION,
            key="leak-scan-run",
        )
        responses = [
            client.get(f"{API}/health"),
            client.get(f"{API}/session", headers=headers),
            client.get(f"{API}/cases", headers=headers),
            client.get(f"{API}/reviews", headers=headers),
            client.get(f"{API}/reviews/{run_id}", headers=headers),
            client.get(f"{API}/reviews/{run_id}/trace", headers=headers),
            client.get(f"{API}/reviews/{run_id}/audit", headers=headers),
            _action(
                client,
                fresh_harness,
                token,
                run_id,
                "cancel",
                key="leak-scan-cancel",
                version=1,
            ),
            _create(
                client,
                fresh_harness,
                token,
                case_id=LOW_CASE,
                case_version=LOW_VERSION,
                key="leak-scan-run",
            ),
            _create(
                client,
                fresh_harness,
                token,
                case_id=LOW_CASE,
                case_version=LOW_VERSION,
                key="leak-scan-second-run",
            ),
            client.post(
                f"{API}/reviews",
                headers=_headers(fresh_harness, token, "leak-scan-authority"),
                json={**_create_body(LOW_CASE, LOW_VERSION), "initiator": "mallory"},
            ),
            client.get(f"{API}/reviews/{UNKNOWN_RUN}", headers=headers),
            client.get("/api/v1/health", headers=headers),
        ]

    for response in responses:
        assert token not in response.text
        assert secret not in response.text
        for value in response.headers.values():
            assert token not in value
