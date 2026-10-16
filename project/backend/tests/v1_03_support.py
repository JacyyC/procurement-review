"""Shared fixtures for the V1-03 security, persistence and recovery suites.

Every V1-03 database run uses its own uniquely named PostgreSQL database and marks its
connections with the V1-03 test flag, so no test can read or damage a user database.
The required PostgreSQL scenarios fail (they never skip) when no DSN is provided: a
skipped security scenario is not evidence.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from procurement_review.api import create_app
from procurement_review.authorization import AuthorizationService, generate_credential
from procurement_review.local_runtime import ScopedWorkflowRunner
from procurement_review.review_repository import ReviewRepository
from procurement_review.review_service import ReviewService
from procurement_review.review_store import ReviewStore
from procurement_review.security_fixture import build_seed_plan
from procurement_review.security_contracts import Scope, SecurityOperation

ADMIN_DSN_ENV = "TEST_POSTGRES_DSN"
V1_03_TEST_DATABASE_PREFIX = "v1_03_test"
V1_03_TEST_FLAG = "v1_03_security_recovery"
API_BASE_URL = "http://127.0.0.1"

DEMO_TENANT = "tenant-local-demo"
DEMO_SUBJECT = "subject-local-demo"
DEMO_READER = "subject-local-demo-reader"
DEMO_AUDITOR = "subject-local-demo-auditor"
DEMO_NO_GRANT = "subject-local-demo-nogrant"
ALPHA_OPERATOR = "subject-alpha-operator"
ALPHA_READER = "subject-alpha-reader"
ALPHA_APPROVER = "subject-alpha-approver"
ALPHA_AUDITOR = "subject-alpha-auditor"
ALPHA_NO_GRANT = "subject-alpha-nogrant"
ALPHA_DISABLED = "subject-alpha-disabled"
BETA_OPERATOR = "subject-beta-operator"
ALPHA_TENANT = "tenant-alpha"
BETA_TENANT = "tenant-beta"


class V1_03EnvironmentError(RuntimeError):
    """Raised when the V1-03 test environment is not available."""


def require_admin_dsn() -> str:
    value = os.environ.get(ADMIN_DSN_ENV)
    if not value:
        raise V1_03EnvironmentError(
            f"{ADMIN_DSN_ENV} is required for the V1-03 PostgreSQL scenarios; "
            "these checks must never be skipped"
        )
    return value


def _url_for_database(base_dsn: str, database: str) -> str:
    head, _, tail = base_dsn.partition("?")
    scheme_and_host, _, _old_database = head.rpartition("/")
    url = f"{scheme_and_host}/{database}"
    query = f"application_name={V1_03_TEST_FLAG}"
    if tail:
        query = f"{tail}&{query}"
    return f"{url}?{query}"


def _admin_engine(dsn: str):
    return create_engine(dsn, isolation_level="AUTOCOMMIT")


def create_isolated_database() -> tuple[str, str]:
    """Create one uniquely named database and return ``(dsn, database_name)``."""

    base = require_admin_dsn()
    database = f"{V1_03_TEST_DATABASE_PREFIX}_{secrets.token_hex(6)}"
    engine = _admin_engine(base)
    try:
        with engine.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{database}"'))
    finally:
        engine.dispose()
    return _url_for_database(base, database), database


def drop_database(database: str) -> None:
    engine = _admin_engine(require_admin_dsn())
    try:
        with engine.connect() as connection:
            connection.execute(
                text(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
            )
    finally:
        engine.dispose()


@dataclass(slots=True)
class SecurityHarness:
    """One isolated store plus the API surface bound to it."""

    store: ReviewStore
    authorization: AuthorizationService
    repository: ReviewRepository
    service: ReviewService
    database: str

    def issue(
        self,
        subject_id: str,
        *,
        ttl_seconds: int = 3_600,
        tenant_id: str | None = None,
    ) -> str:
        token, credential_id, digest = generate_credential()
        tenant = tenant_id or self.tenant_of(subject_id)
        issued_at = datetime.now(UTC)
        self.store.issue_credential(
            credential_id=credential_id,
            tenant_id=tenant,
            subject_id=subject_id,
            token_digest=digest,
            issued_at=issued_at,
            expires_at=issued_at + timedelta(seconds=ttl_seconds),
        )
        return token

    def tenant_of(self, subject_id: str) -> str:
        for tenant_id in self.store.tenant_ids():
            if subject_id in self.store.subject_ids(tenant_id):
                return tenant_id
        raise V1_03EnvironmentError(f"unknown seeded subject {subject_id}")

    def credential_id_of(self, token: str) -> str:
        from procurement_review.authorization import parse_token

        credential_id = parse_token(token)
        assert credential_id is not None
        return credential_id

    def client(self) -> TestClient:
        application = create_app(
            store=self.store,
            authorization=self.authorization,
            repository=self.repository,
            service=self.service,
        )
        return TestClient(application, base_url=API_BASE_URL)

    def authorization_header(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}


def build_service(
    store: ReviewStore,
    *,
    fault: object | None = None,
    resume_limit: int = 2,
    **kwargs: object,
) -> ReviewService:
    return ReviewService(
        repository=ReviewRepository(store),
        runner=ScopedWorkflowRunner(store=store),
        fault=fault,
        resume_limit=resume_limit,
        **kwargs,  # type: ignore[arg-type]
    )


def _build_harness(database_url: str, database: str) -> SecurityHarness:
    store = ReviewStore.from_url(database_url)
    report = store.migrate()
    assert report.total == len(report.applied) + len(report.already_present)
    store.seed(build_seed_plan())
    authorization = AuthorizationService(store=store)
    repository = ReviewRepository(store)
    service = ReviewService(
        repository=repository,
        runner=ScopedWorkflowRunner(store=store),
    )
    return SecurityHarness(
        store=store,
        authorization=authorization,
        repository=repository,
        service=service,
        database=database,
    )


@pytest.fixture(scope="session")
def v1_03_database() -> Iterator[tuple[str, str]]:
    database_url, database = create_isolated_database()
    try:
        yield database_url, database
    finally:
        drop_database(database)


@pytest.fixture(scope="session")
def v1_03_harness(v1_03_database: tuple[str, str]) -> Iterator[SecurityHarness]:
    database_url, database = v1_03_database
    harness = _build_harness(database_url, database)
    try:
        yield harness
    finally:
        harness.store.close()


@pytest.fixture
def fresh_harness() -> Iterator[SecurityHarness]:
    """A per-test harness on its own uniquely named database.

    Persisted Run state is real, so sharing one database across tests would make list
    and audit assertions order dependent.  Each test therefore gets a disposable
    database that is dropped afterwards.
    """

    database_url, database = create_isolated_database()
    harness = _build_harness(database_url, database)
    try:
        yield harness
    finally:
        harness.store.close()
        drop_database(database)


def scope_values(*scopes: Scope) -> tuple[str, ...]:
    return tuple(sorted(item.value for item in scopes))


def operation_of(name: str) -> SecurityOperation:
    return SecurityOperation(name)


def wait_for_status(
    client: TestClient,
    run_id: str,
    headers: dict[str, str],
    *statuses: str,
    timeout_seconds: float = 10.0,
) -> dict:
    import time

    deadline = time.monotonic() + timeout_seconds
    body: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v2/reviews/{run_id}", headers=headers)
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in statuses:
            return body
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not reach {statuses}: {body.get('status')}")
