"""The API-facing error type of the persisted review service.

It lives on its own so the read-side record guard can build the same controlled refusal
without importing the service it protects.
"""

from __future__ import annotations


class ReviewServiceError(RuntimeError):
    """Expected API-facing error with a stable code and HTTP status."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


__all__ = ["ReviewServiceError"]
