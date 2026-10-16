"""Official-SDK client adapter for the fixed procurement read-only tool set."""

from __future__ import annotations

import json
from collections.abc import Mapping

from mcp import Client

from .contracts import BusinessFact
from .mcp_server import (
    GET_PURCHASE_ORDER_TOOL,
    GET_SUPPLIER_TOOL,
    LIST_HISTORICAL_TRANSACTIONS_TOOL,
    MCP_SERVER_VERSION,
    READ_ONLY_TOOL_NAMES,
    ProcurementToolOutcome,
    ProcurementToolResponse,
)
from .worker_contracts import (
    ProcurementReadResult,
    ProcurementReadStatus,
    ProcurementWorkerInput,
)


class OfficialProcurementMCPClient:
    """Collect review facts by calling the three existing read tools in order."""

    def __init__(self, client: Client) -> None:
        if not isinstance(client, Client):
            raise TypeError("OfficialProcurementMCPClient requires mcp.Client")
        self._client = client

    @staticmethod
    def _failed(
        request: ProcurementWorkerInput,
        status: ProcurementReadStatus,
        detail: str,
        *,
        facts: tuple[BusinessFact, ...] = (),
    ) -> ProcurementReadResult:
        return ProcurementReadResult(
            status=status,
            requested_dataset_version=request.dataset_version,
            facts=facts,
            detail=detail,
        )

    async def _tool_contract_is_read_only(self) -> bool:
        server_info = self._client.server_info
        if server_info is None or server_info.version != MCP_SERVER_VERSION:
            return False
        result = await self._client.list_tools()
        if tuple(tool.name for tool in result.tools) != READ_ONLY_TOOL_NAMES:
            return False
        return all(
            tool.annotations is not None
            and tool.annotations.read_only_hint is True
            and tool.annotations.destructive_hint is False
            and tool.annotations.idempotent_hint is True
            and tool.annotations.open_world_hint is False
            for tool in result.tools
        )

    async def _call(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
    ) -> ProcurementToolResponse | None:
        result = await self._client.call_tool(tool_name, dict(arguments))
        if result.is_error or result.structured_content is None:
            return None
        try:
            response = ProcurementToolResponse.model_validate_json(
                json.dumps(
                    result.structured_content,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError):
            return None
        if response.tool_name != tool_name:
            return None
        if any(fact.tool_name != tool_name for fact in response.facts):
            return None
        return response

    @staticmethod
    def _terminal_response(
        request: ProcurementWorkerInput,
        response: ProcurementToolResponse,
        prior_facts: tuple[BusinessFact, ...],
    ) -> ProcurementReadResult | None:
        facts = (*prior_facts, *response.facts)
        if response.outcome is ProcurementToolOutcome.KNOWN:
            if response.dataset_version != request.dataset_version or any(
                fact.dataset_version != request.dataset_version
                for fact in response.facts
            ):
                return OfficialProcurementMCPClient._failed(
                    request,
                    ProcurementReadStatus.VERSION_MISMATCH,
                    "mcp_known_response_version_mismatch",
                    facts=prior_facts,
                )
            return None
        if response.outcome is ProcurementToolOutcome.UNKNOWN:
            return OfficialProcurementMCPClient._failed(
                request,
                ProcurementReadStatus.UNKNOWN,
                "mcp_record_not_found",
                facts=facts,
            )
        if response.outcome is ProcurementToolOutcome.INVALID_ARGUMENT:
            status = ProcurementReadStatus.INVALID_INPUT
            detail = "mcp_invalid_argument"
        elif response.outcome is ProcurementToolOutcome.VERSION_MISMATCH:
            status = ProcurementReadStatus.VERSION_MISMATCH
            detail = "mcp_dataset_version_mismatch"
        else:
            status = ProcurementReadStatus.UPSTREAM_FAILURE
            detail = "mcp_upstream_failure"
        return OfficialProcurementMCPClient._failed(
            request,
            status,
            detail,
            facts=prior_facts,
        )

    async def collect_review_facts(
        self,
        request: ProcurementWorkerInput,
    ) -> ProcurementReadResult:
        """Call order, supplier and history tools without exposing storage APIs."""

        try:
            if not await self._tool_contract_is_read_only():
                return self._failed(
                    request,
                    ProcurementReadStatus.UPSTREAM_FAILURE,
                    "mcp_read_only_tool_contract_mismatch",
                )

            order_response = await self._call(
                GET_PURCHASE_ORDER_TOOL,
                {
                    "purchase_order_id": request.purchase_order_id,
                    "dataset_version": request.dataset_version,
                },
            )
            if order_response is None:
                return self._failed(
                    request,
                    ProcurementReadStatus.UPSTREAM_FAILURE,
                    "mcp_purchase_order_response_invalid",
                )
            terminal = self._terminal_response(request, order_response, ())
            if terminal is not None:
                return terminal
            if len(order_response.facts) != 1:
                return self._failed(
                    request,
                    ProcurementReadStatus.UPSTREAM_FAILURE,
                    "mcp_purchase_order_cardinality_invalid",
                )

            order_fact = order_response.facts[0]
            supplier_id = order_fact.result.get("supplier_id")
            if not isinstance(supplier_id, str) or not supplier_id:
                return self._failed(
                    request,
                    ProcurementReadStatus.UPSTREAM_FAILURE,
                    "mcp_purchase_order_supplier_invalid",
                    facts=(order_fact,),
                )

            supplier_response = await self._call(
                GET_SUPPLIER_TOOL,
                {
                    "supplier_id": supplier_id,
                    "dataset_version": request.dataset_version,
                },
            )
            if supplier_response is None:
                return self._failed(
                    request,
                    ProcurementReadStatus.UPSTREAM_FAILURE,
                    "mcp_supplier_response_invalid",
                    facts=(order_fact,),
                )
            terminal = self._terminal_response(
                request,
                supplier_response,
                (order_fact,),
            )
            if terminal is not None:
                return terminal
            if len(supplier_response.facts) != 1:
                return self._failed(
                    request,
                    ProcurementReadStatus.UPSTREAM_FAILURE,
                    "mcp_supplier_cardinality_invalid",
                    facts=(order_fact,),
                )

            known_facts = (order_fact, supplier_response.facts[0])
            history_response = await self._call(
                LIST_HISTORICAL_TRANSACTIONS_TOOL,
                {
                    "purchase_order_id": request.purchase_order_id,
                    "dataset_version": request.dataset_version,
                },
            )
            if history_response is None:
                return self._failed(
                    request,
                    ProcurementReadStatus.UPSTREAM_FAILURE,
                    "mcp_history_response_invalid",
                    facts=known_facts,
                )
            terminal = self._terminal_response(
                request,
                history_response,
                known_facts,
            )
            if terminal is not None:
                return terminal

            return ProcurementReadResult(
                status=ProcurementReadStatus.KNOWN,
                requested_dataset_version=request.dataset_version,
                facts=(*known_facts, *history_response.facts),
            )
        except Exception:  # noqa: BLE001 - the external client is fail-closed
            return self._failed(
                request,
                ProcurementReadStatus.UPSTREAM_FAILURE,
                "official_mcp_client_failure",
            )
