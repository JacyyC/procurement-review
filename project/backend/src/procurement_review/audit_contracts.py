"""Persisted product records for V1-03: approval audit v2, receipts and denials.

`approval-audit.v1` stays frozen as the V1-01 domain record.  The product runtime writes
the versioned v2 record, which adds the tenant/Case anchor, the database-assigned
sequence, the authorization revision, the action's before/after revision, the result
checksum and the idempotency *digest* (never the raw key).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .contracts import (
    ApprovalActionType,
    Checksum,
    Identifier,
    ReviewLifecycleStatus,
    StrictContract,
)

APPROVAL_AUDIT_V2 = "approval-audit.v2"
RUN_RECORD_VERSION = "review-run.v1"
CHECKPOINT_VERSION = "review-checkpoint.v1"


class AuditEventClass(StrEnum):
    """Independent event categories; a denial never pretends to be an approval."""

    APPROVAL = "approval"
    RECOVERY = "recovery"
    LIFECYCLE = "lifecycle"
    ACCESS_DENIAL = "access_denial"


class RunRecordStatus(StrEnum):
    """Terminal/intermediate classification used when reading a persisted Run."""

    ACTIVE = "active"
    FINAL = "final"


class ApprovalAuditEventV2(StrictContract):
    """One persisted product audit event with a database-assigned sequence."""

    audit_schema_version: Literal["approval-audit.v2"] = APPROVAL_AUDIT_V2
    audit_id: Identifier
    sequence: Annotated[int, Field(strict=True, ge=1)]
    tenant_id: Identifier | None = None
    case_id: Identifier | None = None
    run_id: Identifier | None = None
    event_class: AuditEventClass
    action_type: ApprovalActionType | None = None
    actor: Identifier | None = None
    authorization_revision: Annotated[int, Field(strict=True, ge=1)] | None = None
    previous_status: ReviewLifecycleStatus | None = None
    new_status: ReviewLifecycleStatus | None = None
    previous_revision: Annotated[int, Field(strict=True, ge=0)] | None = None
    new_revision: Annotated[int, Field(strict=True, ge=1)] | None = None
    result_checksum: Checksum | None = None
    idempotency_digest: Checksum | None = None
    receipt_id: Identifier | None = None
    api_version: Literal["review-api.v2"] = "review-api.v2"
    occurred_at: datetime
    detail_code: Identifier | None = None
    simulation_only: Literal[True] = True
    external_write_performed: Literal[False] = False

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def enforce_event_semantics(self) -> ApprovalAuditEventV2:
        if self.event_class is AuditEventClass.APPROVAL:
            if self.action_type is None or self.actor is None:
                raise ValueError(
                    "approval events require an action and a trusted actor"
                )
            if self.previous_status is None or self.new_status is None:
                raise ValueError("approval events require both lifecycle statuses")
            if self.action_type is ApprovalActionType.CONFIRM:
                if self.new_status is not ReviewLifecycleStatus.CONFIRMED:
                    raise ValueError("confirm must produce confirmed")
                if self.receipt_id is None:
                    raise ValueError("confirm must reference a receipt")
            elif self.action_type is ApprovalActionType.REJECT:
                if self.new_status is not ReviewLifecycleStatus.REJECTED:
                    raise ValueError("reject must produce rejected")
                if self.receipt_id is None:
                    raise ValueError("reject must reference a receipt")
            else:
                if self.new_status is not ReviewLifecycleStatus.CANCELLED:
                    raise ValueError("cancel must produce cancelled")
                if self.receipt_id is not None:
                    raise ValueError("cancel cannot reference a receipt")
        elif self.receipt_id is not None:
            raise ValueError("only approval events may reference a receipt")
        return self


class RunSummaryRecord(StrictContract):
    run_id: Identifier
    summary_checksum: Checksum
    human_gate_status: ReviewLifecycleStatus
    committed_revision: Annotated[int, Field(strict=True, ge=1)]


class RunRecord(StrictContract):
    """Persisted Run header; the fields recovery depends on are never recomputed."""

    run_record_version: Literal["review-run.v1"] = RUN_RECORD_VERSION
    run_id: Identifier
    tenant_id: Identifier
    case_id: Identifier
    case_version: Identifier
    status: ReviewLifecycleStatus
    revision: Annotated[int, Field(strict=True, ge=1)]
    resume_count: Annotated[int, Field(strict=True, ge=0)]
    created_by_subject: Identifier
    current_subject_id: Identifier
    request_checksum: Checksum
    document_version: Identifier
    dataset_version: Identifier
    index_version: Identifier
    rules_version: Identifier
    created_at: datetime
    updated_at: datetime
    error_code: Identifier | None = None
    error_message: Annotated[str, Field(min_length=1, max_length=500)] | None = None


class CheckpointRecord(StrictContract):
    """One adopted Worker branch result; at most one per Worker per Run."""

    checkpoint_version: Literal["review-checkpoint.v1"] = CHECKPOINT_VERSION
    run_id: Identifier
    worker: Identifier
    generation: Annotated[int, Field(strict=True, ge=1)]
    attempt_no: Annotated[int, Field(strict=True, ge=1)]
    input_checksum: Checksum
    contract_checksum: Checksum
    result_checksum: Checksum
