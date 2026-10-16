"""The versioned PostgreSQL schema of the review and authorization registry.

Migration statements are data here, so the store that applies them stays about
operations.  A migration's recorded checksum is computed from exactly these
statement tuples, which is why a released version is never edited in place: a change
to the persisted shape adds a new migration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


_MIGRATION_1_IDENTITY = (
    """
    CREATE TABLE IF NOT EXISTS review_schema_migration (
        version integer PRIMARY KEY,
        name text NOT NULL,
        checksum text NOT NULL,
        applied_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_tenant (
        tenant_id text PRIMARY KEY,
        display_name text NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_subject (
        tenant_id text NOT NULL REFERENCES review_tenant(tenant_id),
        subject_id text NOT NULL,
        display_name text NOT NULL,
        disabled boolean NOT NULL DEFAULT false,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (tenant_id, subject_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_subject_role (
        tenant_id text NOT NULL,
        subject_id text NOT NULL,
        role text NOT NULL,
        PRIMARY KEY (tenant_id, subject_id, role),
        FOREIGN KEY (tenant_id, subject_id)
            REFERENCES review_subject(tenant_id, subject_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_authorization_revision (
        tenant_id text PRIMARY KEY REFERENCES review_tenant(tenant_id),
        revision bigint NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_credential (
        credential_id text PRIMARY KEY,
        tenant_id text NOT NULL,
        subject_id text NOT NULL,
        token_digest text NOT NULL,
        digest_algorithm text NOT NULL,
        issued_at timestamptz NOT NULL,
        expires_at timestamptz NOT NULL,
        revoked_at timestamptz,
        revoked_reason text,
        UNIQUE (token_digest),
        FOREIGN KEY (tenant_id, subject_id)
            REFERENCES review_subject(tenant_id, subject_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_case_registry (
        tenant_id text NOT NULL REFERENCES review_tenant(tenant_id),
        case_id text NOT NULL,
        label text NOT NULL,
        case_type text NOT NULL,
        scenario text NOT NULL,
        purchase_order_id text NOT NULL,
        case_version text NOT NULL,
        document_version text NOT NULL,
        dataset_version text NOT NULL,
        index_version text NOT NULL,
        rules_version text NOT NULL,
        checksum text NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (tenant_id, case_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_case_grant (
        tenant_id text NOT NULL,
        case_id text NOT NULL,
        subject_id text NOT NULL,
        granted_at timestamptz NOT NULL DEFAULT now(),
        revoked_at timestamptz,
        PRIMARY KEY (tenant_id, case_id, subject_id),
        FOREIGN KEY (tenant_id, case_id)
            REFERENCES review_case_registry(tenant_id, case_id) ON DELETE CASCADE,
        FOREIGN KEY (tenant_id, subject_id)
            REFERENCES review_subject(tenant_id, subject_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_document_registry (
        registry_id text PRIMARY KEY,
        document_id text NOT NULL,
        ownership text NOT NULL,
        tenant_id text,
        case_id text,
        category text NOT NULL,
        source_id text NOT NULL,
        source_version_id text NOT NULL,
        document_version text NOT NULL,
        index_version text NOT NULL,
        content_checksum text NOT NULL,
        test_only boolean NOT NULL DEFAULT true,
        chunk_payload jsonb NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT review_document_ownership_shape CHECK (
            (ownership = 'shared_fixture' AND tenant_id IS NULL AND case_id IS NULL)
            OR (ownership = 'case_document' AND tenant_id IS NOT NULL
                AND case_id IS NOT NULL)
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_case_document (
        tenant_id text NOT NULL,
        case_id text NOT NULL,
        document_id text NOT NULL,
        ownership text NOT NULL,
        PRIMARY KEY (tenant_id, case_id, document_id, ownership),
        FOREIGN KEY (tenant_id, case_id)
            REFERENCES review_case_registry(tenant_id, case_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_case_procurement (
        tenant_id text NOT NULL,
        case_id text NOT NULL,
        record_kind text NOT NULL,
        record_id text NOT NULL,
        PRIMARY KEY (tenant_id, case_id, record_kind, record_id),
        FOREIGN KEY (tenant_id, case_id)
            REFERENCES review_case_registry(tenant_id, case_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_procurement_record (
        tenant_id text NOT NULL,
        case_id text NOT NULL,
        record_kind text NOT NULL,
        record_id text NOT NULL,
        dataset_version text NOT NULL,
        as_of date NOT NULL,
        payload jsonb NOT NULL,
        PRIMARY KEY (tenant_id, case_id, record_kind, record_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_retrieval_score (
        document_version text NOT NULL,
        index_version text NOT NULL,
        query_text text NOT NULL,
        chunk_id text NOT NULL,
        score double precision NOT NULL,
        PRIMARY KEY (document_version, index_version, query_text, chunk_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_access_denial (
        sequence bigserial PRIMARY KEY,
        denial_id text NOT NULL UNIQUE,
        occurred_at timestamptz NOT NULL,
        request_id text NOT NULL,
        operation text NOT NULL,
        reason text NOT NULL,
        actor_subject_id text,
        actor_tenant_id text,
        case_id text,
        target_digest text
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS review_access_denial_case_idx
        ON review_access_denial (actor_tenant_id, case_id, sequence)
    """,
    """
    CREATE TABLE IF NOT EXISTS review_seed_marker (
        fixture_version text PRIMARY KEY,
        checksum text NOT NULL,
        applied_at timestamptz NOT NULL DEFAULT now()
    )
    """,
)

_MIGRATION_2_RUN_STATE = (
    """
    CREATE TABLE IF NOT EXISTS review_run (
        run_id text PRIMARY KEY,
        tenant_id text NOT NULL,
        case_id text NOT NULL,
        case_version text NOT NULL,
        status text NOT NULL,
        revision integer NOT NULL DEFAULT 1,
        resume_count integer NOT NULL DEFAULT 0,
        created_by_subject text NOT NULL,
        current_subject_id text NOT NULL,
        request_checksum text NOT NULL,
        document_version text NOT NULL,
        dataset_version text NOT NULL,
        index_version text NOT NULL,
        rules_version text NOT NULL,
        isolation jsonb NOT NULL,
        error_code text,
        error_message text,
        created_at timestamptz NOT NULL,
        updated_at timestamptz NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS review_run_scope_idx
        ON review_run (tenant_id, case_id, created_at DESC, run_id DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS review_run_generation (
        run_id text NOT NULL,
        generation integer NOT NULL,
        subject_id text NOT NULL,
        credential_id text NOT NULL,
        authorization_revision bigint NOT NULL,
        owner_id text NOT NULL,
        status text NOT NULL,
        heartbeat_at timestamptz NOT NULL,
        lease_expires_at timestamptz NOT NULL,
        deadline_at timestamptz NOT NULL,
        created_at timestamptz NOT NULL,
        PRIMARY KEY (run_id, generation),
        FOREIGN KEY (run_id) REFERENCES review_run(run_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_worker_attempt (
        run_id text NOT NULL,
        generation integer NOT NULL,
        worker text NOT NULL,
        attempt_no integer NOT NULL,
        started_at timestamptz NOT NULL,
        finished_at timestamptz,
        outcome text,
        first_error text,
        last_error text,
        abandoned boolean NOT NULL DEFAULT false,
        discarded_late boolean NOT NULL DEFAULT false,
        PRIMARY KEY (run_id, generation, worker, attempt_no),
        FOREIGN KEY (run_id) REFERENCES review_run(run_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_branch_checkpoint (
        run_id text NOT NULL,
        worker text NOT NULL,
        generation integer NOT NULL,
        attempt_no integer NOT NULL,
        input_checksum text NOT NULL,
        contract_checksum text NOT NULL,
        result_checksum text NOT NULL,
        result_payload jsonb NOT NULL,
        adopted_at timestamptz NOT NULL,
        PRIMARY KEY (run_id, worker),
        FOREIGN KEY (run_id) REFERENCES review_run(run_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_run_summary (
        run_id text PRIMARY KEY,
        summary_checksum text NOT NULL,
        summary_payload jsonb NOT NULL,
        execution_payload jsonb NOT NULL,
        human_gate_status text NOT NULL,
        committed_revision integer NOT NULL,
        created_at timestamptz NOT NULL,
        FOREIGN KEY (run_id) REFERENCES review_run(run_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_idempotency (
        tenant_id text NOT NULL,
        subject_id text NOT NULL,
        operation_family text NOT NULL,
        key_digest text NOT NULL,
        request_digest text NOT NULL,
        run_id text,
        action text,
        receipt_id text,
        first_revision integer,
        created_at timestamptz NOT NULL,
        PRIMARY KEY (tenant_id, subject_id, operation_family, key_digest)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_receipt (
        receipt_id text PRIMARY KEY,
        run_id text NOT NULL UNIQUE,
        case_id text NOT NULL,
        human_actor text NOT NULL,
        decision text NOT NULL,
        reviewed_recommendation text NOT NULL,
        recorded_at timestamptz NOT NULL,
        idempotency_digest text NOT NULL,
        receipt_payload jsonb NOT NULL,
        FOREIGN KEY (run_id) REFERENCES review_run(run_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_audit_event (
        sequence bigserial PRIMARY KEY,
        audit_id text NOT NULL UNIQUE,
        run_id text,
        case_id text,
        event_class text NOT NULL,
        action_type text,
        actor text,
        authorization_revision bigint,
        previous_status text,
        new_status text,
        previous_revision integer,
        new_revision integer,
        result_checksum text,
        idempotency_digest text,
        receipt_id text,
        api_version text NOT NULL,
        occurred_at timestamptz NOT NULL,
        event_payload jsonb NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS review_audit_run_idx
        ON review_audit_event (run_id, sequence)
    """,
    """
    CREATE TABLE IF NOT EXISTS review_trace_event (
        sequence bigserial PRIMARY KEY,
        event_ordinal integer NOT NULL,
        event_id text NOT NULL UNIQUE,
        trace_id text NOT NULL,
        run_id text NOT NULL,
        event_type text NOT NULL,
        step text NOT NULL,
        occurred_at timestamptz NOT NULL,
        input_summary jsonb NOT NULL,
        output_summary jsonb NOT NULL,
        input_checksum text NOT NULL,
        output_checksum text NOT NULL,
        source_versions jsonb NOT NULL,
        duration_ms integer NOT NULL DEFAULT 0,
        error text,
        generation integer,
        attempt integer,
        worker text,
        event_payload jsonb NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS review_trace_run_idx
        ON review_trace_event (run_id, sequence)
    """,
)


_MIGRATION_3_IDEMPOTENT_OUTCOME = (
    """
    ALTER TABLE review_idempotency ADD COLUMN IF NOT EXISTS outcome text
    """,
    """
    ALTER TABLE review_idempotency ADD COLUMN IF NOT EXISTS failure_code text
    """,
)


_MIGRATION_4_RECORD_IDENTITY_AND_SEQUENCE = (
    # One allocator row per stream.  The allocator is taken with a row lock that is held
    # to commit, so the assigned number always follows the committed order of the events
    # and never the order in which two transactions happened to read a maximum.  It is
    # seeded from the rows already present, so an upgrade continues the existing history
    # instead of restarting it; gaps are allowed and are never filled in.
    """
    CREATE TABLE IF NOT EXISTS review_sequence_allocator (
        stream text PRIMARY KEY,
        next_value bigint NOT NULL
    )
    """,
    """
    INSERT INTO review_sequence_allocator (stream, next_value)
    SELECT 'audit', coalesce(max(sequence), 0) + 1 FROM review_audit_event
    ON CONFLICT (stream) DO NOTHING
    """,
    # The Run record owns its Trace identity: a Run is created with it in the same
    # transaction, so no reader has to invent one from the Run ID, and a list view that
    # does not load events still reports the persisted identity.  Rows committed before
    # this migration take the identity their own earliest event already carries; a row
    # with neither is left without one and the read projection fails closed rather than
    # fabricating a different identity.
    """
    ALTER TABLE review_run ADD COLUMN IF NOT EXISTS trace_id text
    """,
    """
    UPDATE review_run r
    SET trace_id = (
        SELECT t.trace_id FROM review_trace_event t
        WHERE t.run_id = r.run_id ORDER BY t.sequence LIMIT 1
    )
    WHERE r.trace_id IS NULL
    """,
)


REVIEW_MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="identity_and_resource_registry",
        statements=_MIGRATION_1_IDENTITY,
    ),
    Migration(version=2, name="review_run_state", statements=_MIGRATION_2_RUN_STATE),
    Migration(
        version=3,
        name="idempotent_operation_outcome",
        statements=_MIGRATION_3_IDEMPOTENT_OUTCOME,
    ),
    Migration(
        version=4,
        name="record_identity_and_audit_sequence",
        statements=_MIGRATION_4_RECORD_IDENTITY_AND_SEQUENCE,
    ),
)


def migration_checksum(migration: Migration) -> str:
    from .trace import canonical_checksum

    return canonical_checksum(list(migration.statements))
