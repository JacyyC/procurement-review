"""Official-SDK MCP server exposing only synthetic procurement read tools."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, time
from enum import StrEnum
from typing import Annotated, Literal, Self

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import BusinessFact, FactStatus
from .procurement_store import (
    DatasetVersionMismatchError,
    InvalidProcurementQueryError,
    ProcurementReader,
    ProcurementRecord,
    ProcurementScopeError,
    ProcurementStoreError,
)
from .security_contracts import RuntimeScope

MCP_SERVER_NAME = "procurement-readonly"
MCP_SERVER_VERSION = "procurement-readonly-mcp.v1"
MCP_RESPONSE_SCHEMA_VERSION = "procurement-mcp-response.v1"
GET_SUPPLIER_TOOL = "get_supplier"
GET_PURCHASE_ORDER_TOOL = "get_purchase_order"
LIST_HISTORICAL_TRANSACTIONS_TOOL = "list_historical_transactions"
READ_ONLY_TOOL_NAMES = (
    GET_SUPPLIER_TOOL,
    GET_PURCHASE_ORDER_TOOL,
    LIST_HISTORICAL_TRANSACTIONS_TOOL,
)

_STABLE_RECORD_ID = re.compile(r"^[A-Z]+-[0-9]{3}$")
_DATASET_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}$")

ResponseIdentifier = Annotated[
    str,
    Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$"),
]
ErrorMessage = Annotated[str, Field(min_length=1, max_length=500)]


class ProcurementToolOutcome(StrEnum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    UPSTREAM_FAILURE = "UPSTREAM_FAILURE"
    SCOPE_DENIED = "SCOPE_DENIED"


class ProcurementToolErrorCode(StrEnum):
    NOT_FOUND = "NOT_FOUND"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    UPSTREAM_FAILURE = "UPSTREAM_FAILURE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class ProcurementToolResponse(BaseModel):
    """Stable structured result returned by every procurement MCP tool."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    response_schema_version: Literal["procurement-mcp-response.v1"] = (
        MCP_RESPONSE_SCHEMA_VERSION
    )
    tool_name: ResponseIdentifier
    outcome: ProcurementToolOutcome
    dataset_version: ResponseIdentifier
    as_of: datetime
    facts: tuple[BusinessFact, ...] = ()
    error_code: ProcurementToolErrorCode | None = None
    error_message: ErrorMessage | None = None
    read_only: Literal[True] = True

    @model_validator(mode="after")
    def enforce_outcome(self) -> Self:
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")

        if self.outcome is ProcurementToolOutcome.KNOWN:
            if not self.facts or any(
                fact.status is not FactStatus.KNOWN for fact in self.facts
            ):
                raise ValueError("KNOWN responses require known facts")
            if self.error_code is not None or self.error_message is not None:
                raise ValueError("KNOWN responses cannot carry an error")
            return self

        if self.outcome is ProcurementToolOutcome.UNKNOWN:
            if len(self.facts) != 1 or self.facts[0].status is not FactStatus.UNKNOWN:
                raise ValueError("UNKNOWN responses require one unknown fact")
            expected_error = ProcurementToolErrorCode.NOT_FOUND
        else:
            if self.facts:
                raise ValueError("failed responses cannot carry facts")
            expected_error = (
                ProcurementToolErrorCode.OUT_OF_SCOPE
                if self.outcome is ProcurementToolOutcome.SCOPE_DENIED
                else ProcurementToolErrorCode(self.outcome.value)
            )

        if self.error_code is not expected_error or self.error_message is None:
            raise ValueError("response error does not match outcome")
        return self


def _as_of_datetime(reader: ProcurementReader) -> datetime:
    return datetime.combine(reader.identity.as_of, time.min, tzinfo=UTC)


def _known_fact(tool_name: str, record: ProcurementRecord) -> BusinessFact:
    return BusinessFact(
        tool_name=tool_name,
        record_id=record.record_id,
        dataset_version=record.dataset_version,
        as_of=datetime.combine(record.as_of, time.min, tzinfo=UTC),
        status=FactStatus.KNOWN,
        result=record.payload,
    )


def _unknown_fact(
    *,
    reader: ProcurementReader,
    tool_name: str,
    record_id: str,
) -> BusinessFact:
    return BusinessFact(
        tool_name=tool_name,
        record_id=record_id,
        dataset_version=reader.identity.dataset_version,
        as_of=_as_of_datetime(reader),
        status=FactStatus.UNKNOWN,
        failure_reason="NOT_FOUND: no matching synthetic procurement record",
    )


def _known_response(
    reader: ProcurementReader,
    tool_name: str,
    records: tuple[ProcurementRecord, ...],
) -> ProcurementToolResponse:
    return ProcurementToolResponse(
        tool_name=tool_name,
        outcome=ProcurementToolOutcome.KNOWN,
        dataset_version=reader.identity.dataset_version,
        as_of=_as_of_datetime(reader),
        facts=tuple(_known_fact(tool_name, record) for record in records),
    )


def _unknown_response(
    reader: ProcurementReader,
    tool_name: str,
    record_id: str,
) -> ProcurementToolResponse:
    return ProcurementToolResponse(
        tool_name=tool_name,
        outcome=ProcurementToolOutcome.UNKNOWN,
        dataset_version=reader.identity.dataset_version,
        as_of=_as_of_datetime(reader),
        facts=(
            _unknown_fact(
                reader=reader,
                tool_name=tool_name,
                record_id=record_id,
            ),
        ),
        error_code=ProcurementToolErrorCode.NOT_FOUND,
        error_message="No matching synthetic procurement record was found.",
    )


def _error_response(
    reader: ProcurementReader,
    tool_name: str,
    outcome: ProcurementToolOutcome,
    message: str,
) -> ProcurementToolResponse:
    error_code = (
        ProcurementToolErrorCode.OUT_OF_SCOPE
        if outcome is ProcurementToolOutcome.SCOPE_DENIED
        else ProcurementToolErrorCode(outcome.value)
    )
    return ProcurementToolResponse(
        tool_name=tool_name,
        outcome=outcome,
        dataset_version=reader.identity.dataset_version,
        as_of=_as_of_datetime(reader),
        error_code=error_code,
        error_message=message,
    )


def _validate_record_id(value: str, field_name: str) -> str | None:
    if not _STABLE_RECORD_ID.fullmatch(value):
        return f"{field_name} must be a stable synthetic record ID"
    return None


def _validate_dataset_version(
    reader: ProcurementReader,
    tool_name: str,
    dataset_version: str,
) -> ProcurementToolResponse | None:
    if not _DATASET_VERSION.fullmatch(dataset_version):
        return _error_response(
            reader,
            tool_name,
            ProcurementToolOutcome.INVALID_ARGUMENT,
            "dataset_version has an invalid format.",
        )
    if dataset_version != reader.identity.dataset_version:
        return _error_response(
            reader,
            tool_name,
            ProcurementToolOutcome.VERSION_MISMATCH,
            "The requested dataset version is not available.",
        )
    return None


def _note_scope_denial(reader: ProcurementReader) -> None:
    """Count one rejection on readers that expose the optional telemetry hook."""

    hook = getattr(reader, "note_scope_denial", None)
    if callable(hook):
        hook()


def _scope_denied_response(
    reader: ProcurementReader,
    tool_name: str,
    message: str = "The requested record is outside the runtime scope.",
) -> ProcurementToolResponse:
    """Return the single controlled rejection shape for an out-of-scope read."""

    _note_scope_denial(reader)
    return _error_response(
        reader,
        tool_name,
        ProcurementToolOutcome.SCOPE_DENIED,
        message,
    )


def _query_one(
    *,
    reader: ProcurementReader,
    scope: RuntimeScope,
    tool_name: str,
    record_id: str,
    dataset_version: str,
    getter: Callable[[str, str], ProcurementRecord | None],
    allowed_ids: frozenset[str],
) -> ProcurementToolResponse:
    version_error = _validate_dataset_version(reader, tool_name, dataset_version)
    if version_error is not None:
        return version_error
    id_error = _validate_record_id(record_id, "record_id")
    if id_error is not None:
        return _error_response(
            reader,
            tool_name,
            ProcurementToolOutcome.INVALID_ARGUMENT,
            id_error,
        )
    if record_id not in allowed_ids:
        # Scope is checked before any data access, not after reading a record.
        return _scope_denied_response(reader, tool_name)

    try:
        record = getter(record_id, dataset_version)
    except DatasetVersionMismatchError:
        return _error_response(
            reader,
            tool_name,
            ProcurementToolOutcome.VERSION_MISMATCH,
            "The requested dataset version is not available.",
        )
    except ProcurementScopeError:
        return _scope_denied_response(reader, tool_name)
    except ProcurementStoreError:
        return _error_response(
            reader,
            tool_name,
            ProcurementToolOutcome.UPSTREAM_FAILURE,
            "The procurement store could not complete the read.",
        )

    if record is None:
        return _unknown_response(reader, tool_name, record_id)
    return _known_response(reader, tool_name, (record,))


def create_procurement_mcp_server(
    reader: ProcurementReader,
    *,
    scope: RuntimeScope,
) -> MCPServer:
    """Build the official-SDK server bound to one immutable runtime scope.

    The factory has no unrestricted default: a caller must supply the scope the
    trusted composition root derived for this tenant, Case and execution generation.
    """

    if not isinstance(scope, RuntimeScope):
        raise TypeError("create_procurement_mcp_server requires a RuntimeScope")
    allowed_orders = frozenset(scope.allowed_order_ids)
    allowed_suppliers = frozenset(scope.allowed_supplier_ids)
    allowed_transactions = frozenset(scope.allowed_transaction_ids)

    server = MCPServer(
        MCP_SERVER_NAME,
        version=MCP_SERVER_VERSION,
        instructions="Read-only access to the versioned synthetic procurement fixture.",
    )
    read_only = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    @server.tool(
        name=GET_SUPPLIER_TOOL,
        annotations=read_only,
        structured_output=True,
    )
    def get_supplier(
        supplier_id: str,
        dataset_version: str,
    ) -> ProcurementToolResponse:
        """Read one supplier from a pinned synthetic dataset version."""

        return _query_one(
            reader=reader,
            scope=scope,
            tool_name=GET_SUPPLIER_TOOL,
            record_id=supplier_id,
            dataset_version=dataset_version,
            getter=reader.get_supplier,
            allowed_ids=allowed_suppliers,
        )

    @server.tool(
        name=GET_PURCHASE_ORDER_TOOL,
        annotations=read_only,
        structured_output=True,
    )
    def get_purchase_order(
        purchase_order_id: str,
        dataset_version: str,
    ) -> ProcurementToolResponse:
        """Read one purchase order from a pinned synthetic dataset version."""

        return _query_one(
            reader=reader,
            scope=scope,
            tool_name=GET_PURCHASE_ORDER_TOOL,
            record_id=purchase_order_id,
            dataset_version=dataset_version,
            getter=reader.get_purchase_order,
            allowed_ids=allowed_orders,
        )

    @server.tool(
        name=LIST_HISTORICAL_TRANSACTIONS_TOOL,
        annotations=read_only,
        structured_output=True,
    )
    def list_historical_transactions(
        dataset_version: str,
        purchase_order_id: str | None = None,
        supplier_id: str | None = None,
    ) -> ProcurementToolResponse:
        """Read transactions for exactly one purchase order or supplier."""

        version_error = _validate_dataset_version(
            reader,
            LIST_HISTORICAL_TRANSACTIONS_TOOL,
            dataset_version,
        )
        if version_error is not None:
            return version_error
        if (purchase_order_id is None) == (supplier_id is None):
            return _error_response(
                reader,
                LIST_HISTORICAL_TRANSACTIONS_TOOL,
                ProcurementToolOutcome.INVALID_ARGUMENT,
                "Provide exactly one of purchase_order_id or supplier_id.",
            )

        scope_name = (
            "purchase_order_id" if purchase_order_id is not None else "supplier_id"
        )
        scope_id = purchase_order_id or supplier_id
        assert scope_id is not None
        id_error = _validate_record_id(scope_id, scope_name)
        if id_error is not None:
            return _error_response(
                reader,
                LIST_HISTORICAL_TRANSACTIONS_TOOL,
                ProcurementToolOutcome.INVALID_ARGUMENT,
                id_error,
            )
        allowed_ids = (
            allowed_orders if purchase_order_id is not None else allowed_suppliers
        )
        if scope_id not in allowed_ids:
            return _scope_denied_response(reader, LIST_HISTORICAL_TRANSACTIONS_TOOL)

        try:
            records = reader.list_historical_transactions(
                dataset_version=dataset_version,
                purchase_order_id=purchase_order_id,
                supplier_id=supplier_id,
            )
        except InvalidProcurementQueryError:
            return _error_response(
                reader,
                LIST_HISTORICAL_TRANSACTIONS_TOOL,
                ProcurementToolOutcome.INVALID_ARGUMENT,
                "Provide exactly one of purchase_order_id or supplier_id.",
            )
        except DatasetVersionMismatchError:
            return _error_response(
                reader,
                LIST_HISTORICAL_TRANSACTIONS_TOOL,
                ProcurementToolOutcome.VERSION_MISMATCH,
                "The requested dataset version is not available.",
            )
        except ProcurementScopeError:
            return _scope_denied_response(reader, LIST_HISTORICAL_TRANSACTIONS_TOOL)
        except ProcurementStoreError:
            return _error_response(
                reader,
                LIST_HISTORICAL_TRANSACTIONS_TOOL,
                ProcurementToolOutcome.UPSTREAM_FAILURE,
                "The procurement store could not complete the read.",
            )

        # Even an allowed supplier query only ever returns this Case's registered
        # transactions, so a shared supplier cannot expose another Case's history.
        scoped_records = tuple(
            record for record in records if record.record_id in allowed_transactions
        )
        if not scoped_records:
            query_record_id = f"transaction-query:{scope_name}:{scope_id}"
            return _unknown_response(
                reader,
                LIST_HISTORICAL_TRANSACTIONS_TOOL,
                query_record_id,
            )
        return _known_response(
            reader,
            LIST_HISTORICAL_TRANSACTIONS_TOOL,
            scoped_records,
        )

    return server


def main() -> int:
    """Run the read-only server over stdio using an explicit startup context.

    The stdio entry point refuses to start without a trusted launch context: the
    process must receive the protected startup file produced for one execution
    generation, and it never loads the fixture or falls back to a migration role.
    """

    from .mcp_startup import load_stdio_startup_context

    context = load_stdio_startup_context()
    server = create_procurement_mcp_server(context.reader, scope=context.scope)
    try:
        server.run(transport="stdio")
    finally:
        context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
