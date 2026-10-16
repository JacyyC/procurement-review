"""V1-03 store, migration, registry and maintenance-CLI contracts.

These checks prove the schema is explicit and re-runnable, that the registry seed is
idempotent and non-secret, and that the maintenance CLI writes credentials only to a
protected file.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, text

from procurement_review import review_admin
from procurement_review.authorization import generate_credential
from procurement_review.review_store import (
    MigrationError,
    ReviewStore,
    UnknownMigrationVersionError,
    UnsupportedReviewStorageError,
)
from procurement_review.security_contracts import (
    CredentialState,
    DenialReason,
    SecurityOperation,
)
from procurement_review.security_fixture import (
    SECURITY_FIXTURE_VERSION,
    build_seed_plan,
    fixture_checksum,
)

from v1_03_support import (
    ALPHA_NO_GRANT,
    ALPHA_OPERATOR,
    ALPHA_TENANT,
    DEMO_NO_GRANT,
    DEMO_SUBJECT,
    DEMO_TENANT,
    SecurityHarness,
    _url_for_database,
    require_admin_dsn,
)

SECURITY_CASE = "CASE-ALPHA-001"
SECOND_SECURITY_CASE = "CASE-ALPHA-002"


def test_migrations_are_versioned_and_re_runnable(
    fresh_harness: SecurityHarness,
) -> None:
    report = fresh_harness.store.migrate()
    assert report.applied == ()
    assert report.total == len(report.already_present)
    assert report.total >= 1
    assert fresh_harness.store.applied_migrations() == report.already_present


def test_unknown_migration_version_is_refused(fresh_harness: SecurityHarness) -> None:
    with fresh_harness.store.transaction() as connection:
        connection.execute(
            text(
                "INSERT INTO review_schema_migration (version, name, checksum) "
                "VALUES (9999, 'from_the_future', 'sha256:unknown')"
            )
        )
    try:
        with pytest.raises(UnknownMigrationVersionError):
            fresh_harness.store.migrate()
    finally:
        with fresh_harness.store.transaction() as connection:
            connection.execute(
                text("DELETE FROM review_schema_migration WHERE version = 9999")
            )


def test_ensure_migrated_rejects_an_empty_database() -> None:
    from v1_03_support import create_isolated_database, drop_database

    database_url, database = create_isolated_database()
    store = ReviewStore.from_url(database_url)
    try:
        assert store.applied_migrations() == ()
        with pytest.raises(MigrationError):
            store.ensure_migrated()
    finally:
        store.close()
        drop_database(database)


def test_non_postgresql_storage_is_rejected() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    try:
        with pytest.raises(UnsupportedReviewStorageError):
            ReviewStore(engine)
    finally:
        engine.dispose()


def test_seed_is_idempotent_and_matches_the_fixture(
    fresh_harness: SecurityHarness,
) -> None:
    plan = build_seed_plan()
    assert plan.fixture_version == SECURITY_FIXTURE_VERSION
    assert fixture_checksum().startswith("sha256:")
    report = fresh_harness.store.seed(plan)
    assert report.already_present is True
    assert report.cases == len(plan.cases)
    assert report.subjects == len(plan.subjects)
    assert report.tenants == len(plan.tenants)
    # Re-seeding must not duplicate rows.
    with fresh_harness.store.transaction() as connection:
        counts = {
            table: connection.execute(
                text(f"SELECT count(*) FROM {table}")
            ).scalar_one()
            for table in (
                "review_tenant",
                "review_subject",
                "review_case_registry",
                "review_document_registry",
                "review_case_grant",
                "review_case_procurement",
                "review_procurement_record",
                "review_retrieval_score",
            )
        }
    fresh_harness.store.seed(plan)
    with fresh_harness.store.transaction() as connection:
        after = {
            table: connection.execute(
                text(f"SELECT count(*) FROM {table}")
            ).scalar_one()
            for table in counts
        }
    assert counts == after


def test_seed_rejects_a_different_definition(fresh_harness: SecurityHarness) -> None:
    plan = build_seed_plan()
    with fresh_harness.store.transaction() as connection:
        connection.execute(
            text(
                "UPDATE review_seed_marker SET checksum = :other "
                "WHERE fixture_version = :v"
            ),
            {"other": "sha256:" + "9" * 64, "v": plan.fixture_version},
        )
    try:
        with pytest.raises(Exception, match="different definition"):
            fresh_harness.store.seed(plan)
    finally:
        with fresh_harness.store.transaction() as connection:
            connection.execute(
                text(
                    "UPDATE review_seed_marker SET checksum = :good "
                    "WHERE fixture_version = :v"
                ),
                {"good": plan.checksum(), "v": plan.fixture_version},
            )


def test_grants_control_case_visibility(fresh_harness: SecurityHarness) -> None:
    granted = fresh_harness.store.list_granted_cases(
        tenant_id=ALPHA_TENANT, subject_id=ALPHA_OPERATOR
    )
    assert {item.case_id for item in granted} == {SECURITY_CASE, SECOND_SECURITY_CASE}
    denied = fresh_harness.store.list_granted_cases(
        tenant_id=ALPHA_TENANT, subject_id=ALPHA_NO_GRANT
    )
    assert denied == ()
    assert fresh_harness.store.case_granted(
        tenant_id=ALPHA_TENANT, case_id=SECURITY_CASE, subject_id=ALPHA_OPERATOR
    )
    assert not fresh_harness.store.case_granted(
        tenant_id=ALPHA_TENANT, case_id=SECURITY_CASE, subject_id=ALPHA_NO_GRANT
    )

    assert fresh_harness.store.set_case_grant(
        tenant_id=ALPHA_TENANT,
        case_id=SECURITY_CASE,
        subject_id=ALPHA_OPERATOR,
        granted=False,
        occurred_at=datetime.now(UTC),
    )
    assert not fresh_harness.store.case_granted(
        tenant_id=ALPHA_TENANT, case_id=SECURITY_CASE, subject_id=ALPHA_OPERATOR
    )
    assert (
        fresh_harness.store.load_case(
            tenant_id=ALPHA_TENANT, case_id=SECURITY_CASE, subject_id=ALPHA_OPERATOR
        )
        is None
    )
    fresh_harness.store.set_case_grant(
        tenant_id=ALPHA_TENANT,
        case_id=SECURITY_CASE,
        subject_id=ALPHA_OPERATOR,
        granted=True,
        occurred_at=datetime.now(UTC),
    )


def test_authorization_revision_bumps_on_every_authority_change(
    fresh_harness: SecurityHarness,
) -> None:
    before = fresh_harness.store.authorization_revision(ALPHA_TENANT)
    token, credential_id, digest = generate_credential()
    fresh_harness.store.issue_credential(
        credential_id=credential_id,
        tenant_id=ALPHA_TENANT,
        subject_id=ALPHA_OPERATOR,
        token_digest=digest,
        issued_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    assert fresh_harness.store.authorization_revision(ALPHA_TENANT) == before

    assert fresh_harness.store.revoke_credential(
        credential_id=credential_id,
        revoked_at=datetime.now(UTC),
        reason="test_revocation",
    )
    after_revoke = fresh_harness.store.authorization_revision(ALPHA_TENANT)
    assert after_revoke > before
    state, _row = fresh_harness.store.classify_credential(digest, now=datetime.now(UTC))
    assert state is CredentialState.REVOKED

    assert (
        fresh_harness.store.set_subject_disabled(
            subject_id=ALPHA_OPERATOR, disabled=True
        )
        == 1
    )
    assert fresh_harness.store.authorization_revision(ALPHA_TENANT) > after_revoke
    fresh_harness.store.set_subject_disabled(subject_id=ALPHA_OPERATOR, disabled=False)
    assert not fresh_harness.store.revoke_credential(
        credential_id=credential_id,
        revoked_at=datetime.now(UTC),
        reason="already_revoked",
    )


def test_credential_classification_covers_expiry_and_disablement(
    fresh_harness: SecurityHarness,
) -> None:
    expired, _cid, expired_digest = generate_credential()
    fresh_harness.store.issue_credential(
        credential_id=_cid,
        tenant_id=DEMO_TENANT,
        subject_id=DEMO_SUBJECT,
        token_digest=expired_digest,
        issued_at=datetime.now(UTC) - timedelta(hours=2),
        expires_at=datetime.now(UTC) - timedelta(hours=1),
    )
    state, _ = fresh_harness.store.classify_credential(
        expired_digest, now=datetime.now(UTC)
    )
    assert state is CredentialState.EXPIRED

    unknown_state, row = fresh_harness.store.classify_credential(
        "0" * 64, now=datetime.now(UTC)
    )
    assert unknown_state is CredentialState.UNKNOWN
    assert row is None


def test_denial_ledger_is_append_only_and_ordered(
    fresh_harness: SecurityHarness,
) -> None:
    before = fresh_harness.store.denial_count()
    first = fresh_harness.authorization.record_denial(
        request_id="request-denial-1",
        operation=SecurityOperation.REVIEW_DETAIL,
        reason=DenialReason.CASE_NOT_AUTHORIZED,
        actor_subject_id=ALPHA_OPERATOR,
        actor_tenant_id=ALPHA_TENANT,
        case_id=SECURITY_CASE,
        target_digest="sha256:" + "a" * 64,
    )
    second = fresh_harness.authorization.record_denial(
        request_id="request-denial-2",
        operation=SecurityOperation.SESSION,
        reason=DenialReason.MISSING_CREDENTIAL,
        actor_subject_id=None,
        actor_tenant_id=None,
    )
    assert first.sequence < second.sequence
    assert fresh_harness.store.denial_count() == before + 2

    anchored = fresh_harness.store.list_case_denials(
        tenant_id=ALPHA_TENANT, case_id=SECURITY_CASE
    )
    anchored_ids = [row.denial_id for row in anchored]
    # The ledger is shared by the whole isolated run, so other scenarios may already
    # have anchored events on the same Case; this event must be present exactly once
    # and its sequence must be strictly increasing.
    assert anchored_ids.count(first.denial_id) == 1
    sequences = [row.sequence for row in anchored]
    assert sequences == sorted(sequences)
    unanchored = fresh_harness.store.list_unanchored_denials()
    assert second.denial_id in {row.denial_id for row in unanchored}
    assert first.denial_id not in {row.denial_id for row in unanchored}


def test_denial_sequences_survive_a_new_store_instance(
    fresh_harness: SecurityHarness,
) -> None:
    def record(request_id: str):
        return fresh_harness.authorization.record_denial(
            request_id=request_id,
            operation=SecurityOperation.MCP_TOOL,
            reason=DenialReason.INSUFFICIENT_SCOPE,
            actor_subject_id=DEMO_SUBJECT,
            actor_tenant_id=DEMO_TENANT,
        )

    first = record("restart-1")
    fresh_harness.store.close()
    reopened = ReviewStore.from_url(_harness_url(fresh_harness))
    try:
        assert reopened.denial_count() >= 1
        from procurement_review.authorization import AuthorizationService

        second = AuthorizationService(store=reopened).record_denial(
            request_id="restart-2",
            operation=SecurityOperation.MCP_TOOL,
            reason=DenialReason.INSUFFICIENT_SCOPE,
            actor_subject_id=DEMO_SUBJECT,
            actor_tenant_id=DEMO_TENANT,
        )
        assert second.sequence > first.sequence
    finally:
        reopened.close()


# ------------------------------------------------------------------ admin CLI


def _harness_url(harness: SecurityHarness) -> str:
    return _url_for_database(require_admin_dsn(), harness.database)


@pytest.fixture
def admin_env(fresh_harness: SecurityHarness, monkeypatch: pytest.MonkeyPatch) -> str:
    url = _harness_url(fresh_harness)
    monkeypatch.setenv(review_admin.ADMIN_DATABASE_URL_ENV, url)
    monkeypatch.setenv("REVIEW_DATABASE_URL", url)
    return url


def test_admin_migrate_and_seed_are_re_runnable(
    admin_env: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert review_admin.main(["migrate"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "migrate"
    assert payload["applied"] == []

    assert review_admin.main(["seed", "--fixture", SECURITY_FIXTURE_VERSION]) == 0
    seeded = json.loads(capsys.readouterr().out)
    assert seeded["already_present"] is True
    assert seeded["fixture_checksum"].startswith("sha256:")

    assert review_admin.main(["seed", "--fixture", "unknown-fixture.v9"]) == 2


def test_admin_issues_credentials_only_into_a_protected_file(
    admin_env: str,
    fresh_harness: SecurityHarness,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directory = tmp_path / "secrets"
    output = directory / "demo.token"
    assert (
        review_admin.main(
            [
                "credentials",
                "issue",
                "--subject",
                DEMO_SUBJECT,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["token_printed"] is False
    assert payload["subject_id"] == DEMO_SUBJECT
    assert payload["tenant_id"] == DEMO_TENANT
    token = output.read_text(encoding="utf-8").strip()
    assert token.startswith("rvk1.")
    assert token not in json.dumps(payload)
    assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(output).st_mode) == 0o600

    # Re-issuing to the same path must not silently overwrite a credential file.
    assert (
        review_admin.main(
            [
                "credentials",
                "issue",
                "--subject",
                DEMO_SUBJECT,
                "--output",
                str(output),
            ]
        )
        == 2
    )

    credential_id = payload["credential_id"]
    assert (
        review_admin.main(["credentials", "revoke", "--credential-id", credential_id])
        == 0
    )
    revoked = json.loads(capsys.readouterr().out)
    assert revoked["revoked"] is True
    assert (
        fresh_harness.store.credential_metadata(credential_id)["revoked_at"] is not None
    )


def test_admin_grant_subject_and_denial_commands(
    admin_env: str, fresh_harness: SecurityHarness, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fresh_harness.store.case_granted(
        tenant_id=DEMO_TENANT, case_id="CASE-LOW-001", subject_id=DEMO_SUBJECT
    )
    assert (
        review_admin.main(
            [
                "grant",
                "--tenant",
                DEMO_TENANT,
                "--case",
                "CASE-LOW-001",
                "--subject",
                DEMO_SUBJECT,
                "--revoke",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert not fresh_harness.store.case_granted(
        tenant_id=DEMO_TENANT, case_id="CASE-LOW-001", subject_id=DEMO_SUBJECT
    )
    assert (
        review_admin.main(
            [
                "grant",
                "--tenant",
                DEMO_TENANT,
                "--case",
                "CASE-LOW-001",
                "--subject",
                DEMO_SUBJECT,
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert fresh_harness.store.case_granted(
        tenant_id=DEMO_TENANT, case_id="CASE-LOW-001", subject_id=DEMO_SUBJECT
    )

    assert review_admin.main(["subject", "disable", "--subject", DEMO_NO_GRANT]) == 0
    capsys.readouterr()
    assert review_admin.main(["subject", "enable", "--subject", DEMO_NO_GRANT]) == 0
    capsys.readouterr()

    assert (
        review_admin.main(
            ["denials", "--tenant", ALPHA_TENANT, "--case", SECURITY_CASE]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "denials"
    assert isinstance(payload["rows"], list)

    assert review_admin.main(["denials", "--tenant", ALPHA_TENANT]) == 2
    capsys.readouterr()


def test_admin_writes_a_protected_stdio_startup_context(
    admin_env: str,
    fresh_harness: SecurityHarness,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    token = fresh_harness.issue(ALPHA_OPERATOR, tenant_id=ALPHA_TENANT)
    context = fresh_harness.authorization.resolve(
        token,
        request_id="startup",
        operation=SecurityOperation.REVIEW_CREATE,
    )
    registration = fresh_harness.store.load_case(
        tenant_id=ALPHA_TENANT, case_id=SECURITY_CASE, subject_id=ALPHA_OPERATOR
    )
    assert registration is not None
    scope = fresh_harness.authorization.build_runtime_scope(
        context,
        registration,
        generation=1,
        deadline_at=fresh_harness.authorization.runtime_scope_deadline(30.0),
    )
    scope_file = tmp_path / "scope.json"
    scope_file.write_text(scope.model_dump_json(), encoding="utf-8")
    output = tmp_path / "startup" / "context.json"
    assert (
        review_admin.main(
            [
                "startup-context",
                "--scope-file",
                str(scope_file),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert stat.S_IMODE(os.stat(output).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(output.parent).st_mode) == 0o700
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["scope"]["case_id"] == SECURITY_CASE
    assert token not in json.dumps(payload)

    from procurement_review.mcp_startup import load_startup_context

    loaded = load_startup_context(output)
    try:
        assert loaded.scope.rules_version == scope.rules_version
    finally:
        loaded.close()
