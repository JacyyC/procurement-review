"""The explicit failure vocabulary of the persisted review run.

Every persisted step answers with one of these types, so a caller decides on a
rejection from the contract instead of reading a database error.
"""

from __future__ import annotations


class RepositoryError(RuntimeError):
    """Base class for explicit persistence failures."""


class RunNotFoundError(RepositoryError):
    """The requested Run does not exist in the review store."""


class GrantRevokedError(RepositoryError):
    """The acting subject no longer holds a live Case grant."""


class StaleAuthorizationError(RepositoryError):
    """The tenant authorization revision moved after the request was authorized."""


class VersionConflictError(RepositoryError):
    """The presented Run revision is not the committed one."""

    def __init__(self, *, expected: int, current: int) -> None:
        super().__init__("run revision conflict")
        self.expected, self.current = expected, current


class IllegalTransitionError(RepositoryError):
    """The requested action is not legal from the committed status."""

    def __init__(self, *, action: str, status: str) -> None:
        super().__init__("illegal lifecycle transition")
        self.action, self.status = action, status


class SummaryChecksumError(RepositoryError):
    """The presented summary checksum is not the committed one."""

    def __init__(self, *, current: str) -> None:
        super().__init__("summary checksum mismatch")
        self.current = current


class IdempotencyConflictError(RepositoryError):
    """The same idempotency key was already used with a different request digest."""


class FieldGuardError(RepositoryError):
    """A record would persist a forbidden field or a sensitive value."""


class RecordContractError(FieldGuardError):
    """A record does not satisfy its closed contract; ``categories`` name the refusal."""

    def __init__(self, message: str, categories: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.categories = categories


class RecordIncompleteError(RepositoryError):
    """A committed record is missing an identity the read projection must report."""


class RecordUnavailableError(RepositoryError):
    """A committed record no longer satisfies the contract its readers enforce.

    Only the record kind and the fixed finding categories travel, never the value.
    """

    def __init__(self, *, record: str, categories: tuple[str, ...] = ()) -> None:
        super().__init__(f"the committed {record} record is unavailable")
        self.record, self.categories = record, categories


__all__ = [
    "FieldGuardError",
    "GrantRevokedError",
    "IdempotencyConflictError",
    "IllegalTransitionError",
    "RecordContractError",
    "RecordIncompleteError",
    "RecordUnavailableError",
    "RepositoryError",
    "RunNotFoundError",
    "StaleAuthorizationError",
    "SummaryChecksumError",
    "VersionConflictError",
]
