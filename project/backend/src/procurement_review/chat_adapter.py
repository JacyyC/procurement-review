"""Fail-closed OpenAI-compatible Chat adapter for the V1-02 local experiment.

Only bounded execution metadata is retained.  Prompt content, model output
text, authorization headers, and model reasoning are never written to the
adapter record or an audit sink.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
import re
import time
from enum import StrEnum
from typing import Any, Literal, NoReturn
from urllib.parse import urlsplit

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)

from .contracts import (
    BusinessFact,
    EvidenceRef,
    Finding,
    FindingAssertion,
    FindingStatus,
    Identifier,
    NonEmptyText,
    RiskCategory,
    RiskLevel,
    WorkerKind,
)
from .rules import RULE_SET
from .worker_contracts import SingleAgentModelRequest, WorkerModelRequest

CHAT_ADAPTER_VERSION = "openai-compatible-chat-adapter.v1"
CHAT_FINDING_CONTRACT_VERSION = "chat-finding-contract.v1"
CHAT_RESPONSE_SCHEMA_NAME = "procurement_finding_envelope_v1"
CHAT_RECORD_SCHEMA_VERSION = "chat-call-record.v1"
EMBEDDING_STATUS = "not_supported"
RERANK_STATUS = "not_supported"

_SAFE_HASH = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_CONTEXT_KEYS = {
    "api_key",
    "authorization",
    "credential",
    "password",
    "gold_label",
    "label_id",
    "label_version",
    "label_checksum",
    "expected_recommendation",
    "expected_risk_level",
    "expected_review_state",
    "rule_ids",
    "adjudication_status",
    "acceptable_alternatives",
    "case_family_id",
    "scenario_tags",
    "split",
    "model_fixture_id",
    "failure_mode",
    "failure_metadata",
    "scenario",
    "scenario_tags",
}

_FORBIDDEN_OUTPUT_MARKERS = (
    "synthetic_secret_placeholder",
    "external_sink",
    "write_to_external",
    "external_write_tool",
    "send_to_external",
    "调用外部写入",
    "执行外部写入",
    "外部写入工具",
)


class ChatErrorCode(StrEnum):
    """Stable, non-sensitive failure categories emitted by the adapter."""

    CONFIGURATION = "configuration"
    SECURITY_BOUNDARY = "security_boundary"
    TIMEOUT = "timeout"
    TRANSPORT = "transport"
    AUTHENTICATION = "authentication"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    HTTP_ERROR = "http_error"
    INVALID_JSON = "invalid_json"
    OUTPUT_SCHEMA = "output_schema"
    MISSING_REQUEST_ID = "missing_request_id"
    MISSING_MODEL = "missing_model"
    MISSING_USAGE = "missing_usage"
    AUDIT_FAILURE = "audit_failure"


@dataclass(frozen=True, slots=True)
class ChatAdapterConfig:
    """Non-secret endpoint settings plus an in-memory secret wrapper."""

    base_url: str
    model_name: str
    api_key: SecretStr
    chat_path: str = "/chat/completions"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = 20260901
    max_output_tokens: int = 1_536
    timeout_seconds: float = 30.0
    max_retries: int = 1
    # Off by default: the V1-02B Smoke payload stays unchanged. The V1-02C Eval
    # Runner turns it on for both topologies so the required-finding vocabulary is
    # measurable instead of unknowable.
    include_required_findings: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("MODEL_BASE_URL is required")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("MODEL_BASE_URL must be an HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("MODEL_BASE_URL cannot contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("MODEL_BASE_URL cannot contain query or fragment data")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("MODEL_NAME is required")
        if any(char.isspace() for char in self.model_name):
            raise ValueError("MODEL_NAME cannot contain whitespace")
        if (
            not isinstance(self.api_key, SecretStr)
            or not self.api_key.get_secret_value()
        ):
            raise ValueError("MODEL_API_KEY is required")
        path = self.chat_path
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("MODEL_CHAT_PATH must start with '/'")
        parsed_path = urlsplit(path)
        if parsed_path.query or parsed_path.fragment or parsed_path.netloc:
            raise ValueError("MODEL_CHAT_PATH must be a path only")
        if type(self.temperature) is not float or self.temperature < 0.0:
            raise ValueError("MODEL_TEMPERATURE must be a non-negative float")
        if type(self.top_p) is not float or not 0.0 < self.top_p <= 1.0:
            raise ValueError("MODEL_TOP_P must be in (0, 1]")
        if self.seed is not None and (
            isinstance(self.seed, bool) or type(self.seed) is not int or self.seed < 0
        ):
            raise ValueError("MODEL_SEED must be a non-negative integer")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("MODEL_MAX_OUTPUT_TOKENS must be positive")
        if type(self.timeout_seconds) is not float or self.timeout_seconds <= 0.0:
            raise ValueError("MODEL_TIMEOUT_SECONDS must be positive")
        if type(self.max_retries) is not int or self.max_retries not in {0, 1}:
            raise ValueError("MODEL_MAX_RETRIES must be 0 or 1")

    @property
    def endpoint_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.chat_path.lstrip('/')}"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ChatAdapterConfig:
        values = os.environ if environ is None else environ

        def required(name: str) -> str:
            value = values.get(name, "").strip()
            if not value:
                raise ValueError(f"{name} is required")
            return value

        def float_value(name: str, default: float) -> float:
            raw = values.get(name)
            if raw is None or not raw.strip():
                return default
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError(f"{name} is invalid") from exc

        def int_value(name: str, default: int) -> int:
            raw = values.get(name)
            if raw is None or not raw.strip():
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} is invalid") from exc

        raw_seed = values.get("MODEL_SEED")
        seed = (
            None
            if raw_seed is not None and not raw_seed.strip()
            else int_value("MODEL_SEED", 20260901)
        )
        return cls(
            base_url=required("MODEL_BASE_URL"),
            model_name=required("MODEL_NAME"),
            api_key=SecretStr(required("MODEL_API_KEY")),
            chat_path=values.get("MODEL_CHAT_PATH", "/chat/completions").strip()
            or "/chat/completions",
            temperature=float_value("MODEL_TEMPERATURE", 0.0),
            top_p=float_value("MODEL_TOP_P", 1.0),
            seed=seed,
            max_output_tokens=int_value("MODEL_MAX_OUTPUT_TOKENS", 1_536),
            timeout_seconds=float_value("MODEL_TIMEOUT_SECONDS", 30.0),
            max_retries=int_value("MODEL_MAX_RETRIES", 1),
        )


class ChatCallRecord(BaseModel):
    """Safe request metadata suitable for a local Trace or test sink."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: Literal["chat-call-record.v1"] = CHAT_RECORD_SCHEMA_VERSION
    case_id: str = Field(min_length=1, max_length=160)
    agent_topology: Literal["worker_branch", "single_agent_branch"] = "worker_branch"
    worker: WorkerKind | None = None
    requested_model_id: str = Field(min_length=1, max_length=300)
    resolved_model_id: str | None = Field(default=None, max_length=300)
    request_id_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    http_status: int | None = Field(default=None, ge=100, le=599)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int = Field(ge=0)
    retry_count: int = Field(ge=0, le=1)
    error_code: str | None = Field(default=None, min_length=1, max_length=80)
    error_detail: str | None = Field(default=None, min_length=1, max_length=160)
    error_reason: str | None = Field(
        default=None, min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.:/$\[\]-]+$"
    )
    redaction_status: Literal["redacted"] = "redacted"

    @model_validator(mode="after")
    def topology_matches_worker(self) -> ChatCallRecord:
        if self.agent_topology == "worker_branch" and self.worker is None:
            raise ValueError("worker-branch records require a worker")
        if self.agent_topology == "single_agent_branch" and self.worker is not None:
            raise ValueError("single-agent records must not claim one worker")
        return self


class ChatFinding(BaseModel):
    """Wire Finding shape exposed to the model without integrity metadata."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: Literal["1.0"] = "1.0"
    worker: WorkerKind
    finding_key: Identifier
    category: RiskCategory
    risk_level: RiskLevel
    assertion: FindingAssertion
    statement: NonEmptyText
    status: FindingStatus
    evidence_refs: tuple[Identifier, ...] = ()
    business_fact_refs: tuple[Identifier, ...] = ()
    missing_information: tuple[NonEmptyText, ...] = ()
    recommended_action: NonEmptyText


class FindingEnvelope(BaseModel):
    """The only response shape accepted from the remote Chat endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    findings: tuple[ChatFinding, ...] = Field(min_length=1)


CHAT_RESPONSE_JSON_SCHEMA = FindingEnvelope.model_json_schema()


class ChatAdapterError(RuntimeError):
    """A safe, categorized adapter failure with no response-body payload."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        record: ChatCallRecord | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.record = record
        super().__init__(f"{code}:{detail}")


class _ProtocolIssue(Exception):
    def __init__(
        self,
        code: str,
        detail: str,
        *,
        resolved_model_id: str | None = None,
        request_id_hash: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.resolved_model_id = resolved_model_id
        self.request_id_hash = request_id_hash
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens


def _hash_identifier(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: BaseModel) -> str:
    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _safe_header_request_id(headers: Mapping[str, str]) -> str | None:
    for name in ("x-request-id", "request-id"):
        value = headers.get(name)
        if value and value.strip():
            return _hash_identifier(value.strip())
    return None


_ERROR_SIGNAL_KEYS = ("type", "code", "param")
_SAFE_ERROR_TOKEN = re.compile(r"^[A-Za-z0-9_.:$\[\]-]{1,40}$")


def _safe_error_reason(response: httpx.Response) -> str | None:
    """Return a bounded server error identifier, never the response body."""

    if len(response.content) > 4_096:
        return None
    try:
        payload = json.loads(response.content)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    sources = (error, payload) if isinstance(error, Mapping) else (payload,)
    for source in sources:
        parts = [
            value.strip()
            for key in _ERROR_SIGNAL_KEYS
            if isinstance(value := source.get(key), str)
            and _SAFE_ERROR_TOKEN.fullmatch(value.strip())
        ]
        if parts:
            return "/".join(parts)[:80]
    return None


def _walk_forbidden_keys(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if (
                normalized in _FORBIDDEN_CONTEXT_KEYS
                or normalized.startswith("expected_")
                or "checksum" in normalized
                or normalized in {"label", "labels", "family", "split"}
            ):
                return True
            if _walk_forbidden_keys(child):
                return True
    elif isinstance(value, (tuple, list)):
        return any(_walk_forbidden_keys(item) for item in value)
    return False


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _strict_int(value: Any) -> int | None:
    if type(value) is not int or value < 0:
        return None
    return value


def _wire_evidence_context(context: Any) -> dict[str, Any]:
    """Project an evidence context without checksum or other test metadata."""

    reference = context.evidence_ref
    payload: dict[str, Any] = {
        "ref_id": reference.source_id,
        "source_kind": reference.source_kind.value,
        "version": reference.version,
        "content": context.content,
    }
    if reference.page_number is not None:
        payload["page_number"] = reference.page_number
    if reference.section is not None:
        payload["section"] = reference.section
    if reference.chunk_id is not None:
        payload["chunk_id"] = reference.chunk_id
    return payload


def _wire_business_fact(fact: BusinessFact) -> dict[str, Any]:
    """Project a known business fact without failure/checksum metadata."""

    return {
        "record_id": fact.record_id,
        "tool_name": fact.tool_name,
        "dataset_version": fact.dataset_version,
        "as_of": fact.as_of.isoformat(),
        "status": fact.status.value,
        "result": fact.result,
        "read_only": fact.read_only,
    }


def _wire_finding_text(finding: Finding) -> str:
    return json.dumps(
        finding.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def required_findings_for(
    workers: Sequence[WorkerKind],
) -> list[dict[str, str]]:
    """The public Demo Policy finding vocabulary for the workers in scope.

    Without it a model cannot be expected to name the required ``finding_key``, and
    required-finding coverage would measure the prompt instead of the review.
    """

    wanted = set(workers)
    return sorted(
        (
            {
                "worker": contract.worker.value,
                "finding_key": finding_key,
                "category": contract.category.value,
            }
            for finding_key, contract in RULE_SET.required_finding_contracts.items()
            if contract.worker in wanted
        ),
        key=lambda item: item["finding_key"],
    )


def _record_scope(
    request: WorkerModelRequest | SingleAgentModelRequest,
) -> tuple[Literal["worker_branch", "single_agent_branch"], WorkerKind | None]:
    """Map a request onto the record fields that identify its topology."""

    if isinstance(request, SingleAgentModelRequest):
        return "single_agent_branch", None
    return "worker_branch", request.worker


class OpenAICompatibleChatAdapter:
    """Real HTTP Chat adapter with a replaceable transport for offline tests."""

    adapter_version = CHAT_ADAPTER_VERSION
    fixture_version = CHAT_FINDING_CONTRACT_VERSION
    test_only = False
    real_model = True
    model_contract_version = CHAT_FINDING_CONTRACT_VERSION

    def __init__(
        self,
        config: ChatAdapterConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        audit_sink: Callable[[ChatCallRecord], None] | None = None,
    ) -> None:
        if client is not None and transport is not None:
            raise ValueError("provide either client or transport, not both")
        self.config = config or ChatAdapterConfig.from_env()
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_seconds),
            transport=transport,
        )
        self._owns_client = client is None
        self._sleep = sleep
        self._clock = clock
        self._audit_sink = audit_sink
        self._records: list[ChatCallRecord] = []

    @property
    def records(self) -> tuple[ChatCallRecord, ...]:
        return tuple(self._records)

    @property
    def capabilities(self) -> Mapping[str, str]:
        return {
            "chat": "supported",
            "embedding": EMBEDDING_STATUS,
            "rerank": RERANK_STATUS,
        }

    async def __aenter__(self) -> OpenAICompatibleChatAdapter:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _emit(self, record: ChatCallRecord) -> None:
        self._records.append(record)
        if self._audit_sink is None:
            return
        try:
            self._audit_sink(record)
        except Exception:  # noqa: BLE001 - audit sinks are untrusted ports
            raise ChatAdapterError(
                ChatErrorCode.AUDIT_FAILURE,
                "audit_sink_failure",
                record=record,
            ) from None

    def _context_payload(
        self, request: WorkerModelRequest | SingleAgentModelRequest
    ) -> dict[str, Any]:
        if isinstance(request, SingleAgentModelRequest):
            payload: dict[str, Any] = {
                "case_id": request.case_id,
                "agent_topology": "single_agent",
                "workers": [worker.value for worker in request.workers],
                "question": request.question,
                "evidence_contexts": [
                    _wire_evidence_context(context)
                    for context in request.evidence_contexts
                ],
                "business_facts": [
                    _wire_business_fact(fact) for fact in request.business_facts
                ],
            }
        else:
            if not isinstance(request, WorkerModelRequest):
                raise ChatAdapterError(
                    ChatErrorCode.CONFIGURATION, "worker_request_schema_invalid"
                )
            payload = {
                "case_id": request.case_id,
                "worker": request.worker.value,
            }
            if request.question is not None:
                payload["question"] = request.question
            if request.worker in {WorkerKind.POLICY, WorkerKind.CONTRACT}:
                payload["evidence_contexts"] = [
                    _wire_evidence_context(context)
                    for context in request.evidence_contexts
                ]
            else:
                payload["business_facts"] = [
                    _wire_business_fact(fact) for fact in request.business_facts
                ]
        if self.config.include_required_findings:
            workers = (
                request.workers
                if isinstance(request, SingleAgentModelRequest)
                else (request.worker,)
            )
            payload["required_findings"] = required_findings_for(workers)
        if _walk_forbidden_keys(payload):
            raise ChatAdapterError(
                ChatErrorCode.SECURITY_BOUNDARY,
                "model_context_contains_forbidden_metadata",
            )
        encoded = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, sort_keys=True
        )
        api_key = self.config.api_key.get_secret_value()
        if api_key and api_key in encoded:
            raise ChatAdapterError(
                ChatErrorCode.SECURITY_BOUNDARY,
                "model_context_contains_credential",
            )
        return payload

    def _request_body(
        self, request: WorkerModelRequest | SingleAgentModelRequest
    ) -> dict[str, Any]:
        context = self._context_payload(request)
        body: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only JSON matching the supplied Finding envelope schema. "
                        "Use only the provided context, cite evidence_refs by ref_id and "
                        "business_fact_refs by record_id from that context, "
                        "and never use tools or perform external writes."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        context,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                },
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_output_tokens,
            "stream": False,
            "n": 1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": CHAT_RESPONSE_SCHEMA_NAME,
                    "strict": True,
                    "schema": CHAT_RESPONSE_JSON_SCHEMA,
                },
            },
        }
        if self.config.seed is not None:
            body["seed"] = self.config.seed
        return body

    @staticmethod
    def _validate_citations(
        envelope: FindingEnvelope,
        request: WorkerModelRequest | SingleAgentModelRequest,
        *,
        credential: str = "",
    ) -> tuple[Finding, ...]:
        single_agent = isinstance(request, SingleAgentModelRequest)
        evidence_by_id = {
            context.evidence_ref.source_id: context.evidence_ref
            for context in request.evidence_contexts
        }
        facts_by_id = {fact.record_id: fact for fact in request.business_facts}
        hydrated: list[Finding] = []
        for wire_finding in envelope.findings:
            if single_agent:
                if wire_finding.worker not in request.workers:
                    raise _ProtocolIssue(
                        ChatErrorCode.OUTPUT_SCHEMA,
                        "finding_worker_outside_request_scope",
                    )
            elif wire_finding.worker is not request.worker:
                raise _ProtocolIssue(
                    ChatErrorCode.OUTPUT_SCHEMA,
                    "finding_worker_boundary_violation",
                )
            evidence_refs: list[EvidenceRef] = []
            for ref_id in wire_finding.evidence_refs:
                reference = evidence_by_id.get(ref_id)
                if reference is None:
                    raise _ProtocolIssue(
                        ChatErrorCode.OUTPUT_SCHEMA,
                        "evidence_reference_out_of_context",
                    )
                evidence_refs.append(reference)
            business_fact_refs: list[BusinessFact] = []
            for ref_id in wire_finding.business_fact_refs:
                fact = facts_by_id.get(ref_id)
                if fact is None:
                    raise _ProtocolIssue(
                        ChatErrorCode.OUTPUT_SCHEMA,
                        "business_fact_reference_out_of_context",
                    )
                business_fact_refs.append(fact)
            if wire_finding.worker in {WorkerKind.POLICY, WorkerKind.CONTRACT}:
                if business_fact_refs:
                    raise _ProtocolIssue(
                        ChatErrorCode.OUTPUT_SCHEMA,
                        "document_worker_cited_business_fact",
                    )
            elif evidence_refs:
                raise _ProtocolIssue(
                    ChatErrorCode.OUTPUT_SCHEMA,
                    "procurement_worker_cited_document",
                )
            finding_payload = wire_finding.model_dump(mode="python")
            finding_payload["evidence_refs"] = tuple(evidence_refs)
            finding_payload["business_fact_refs"] = tuple(business_fact_refs)
            try:
                finding = Finding.model_validate(finding_payload, strict=True)
            except ValidationError as exc:
                raise _ProtocolIssue(
                    ChatErrorCode.OUTPUT_SCHEMA,
                    "finding_schema_invalid",
                ) from exc
            rendered = _wire_finding_text(finding)
            rendered_lower = rendered.lower()
            if (len(credential) >= 8 and credential in rendered) or any(
                marker in rendered_lower for marker in _FORBIDDEN_OUTPUT_MARKERS
            ):
                raise _ProtocolIssue(
                    ChatErrorCode.SECURITY_BOUNDARY,
                    "model_output_security_boundary",
                )
            hydrated.append(finding)
        return tuple(hydrated)

    @staticmethod
    def _parse_response(
        response: httpx.Response,
        *,
        request: WorkerModelRequest | SingleAgentModelRequest,
        credential: str = "",
    ) -> tuple[tuple[Finding, ...], str, str, int, int, int]:
        header_request_id = _safe_header_request_id(response.headers)
        try:
            payload = json.loads(
                response.content,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except json.JSONDecodeError as exc:
            raise _ProtocolIssue(
                ChatErrorCode.INVALID_JSON,
                "response_json_invalid",
                request_id_hash=header_request_id,
            ) from exc
        except UnicodeDecodeError as exc:
            raise _ProtocolIssue(
                ChatErrorCode.INVALID_JSON,
                "response_json_invalid",
                request_id_hash=header_request_id,
            ) from exc
        except ValueError:
            raise _ProtocolIssue(
                ChatErrorCode.OUTPUT_SCHEMA,
                "response_json_duplicate_key",
                request_id_hash=header_request_id,
            ) from None
        if not isinstance(payload, Mapping):
            raise _ProtocolIssue(
                ChatErrorCode.OUTPUT_SCHEMA,
                "response_envelope_invalid",
                request_id_hash=header_request_id,
            )
        raw_id = payload.get("id")
        request_id_hash = (
            _hash_identifier(raw_id.strip())
            if isinstance(raw_id, str) and raw_id.strip()
            else header_request_id
        )
        if request_id_hash is None or not _SAFE_HASH.fullmatch(request_id_hash):
            raise _ProtocolIssue(
                ChatErrorCode.MISSING_REQUEST_ID,
                "response_request_id_missing",
            )
        resolved_model = payload.get("model")
        if not isinstance(resolved_model, str) or not resolved_model.strip():
            raise _ProtocolIssue(
                ChatErrorCode.MISSING_MODEL,
                "response_model_id_missing",
                request_id_hash=request_id_hash,
            )
        usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            raise _ProtocolIssue(
                ChatErrorCode.MISSING_USAGE,
                "response_usage_missing",
                resolved_model_id=resolved_model.strip(),
                request_id_hash=request_id_hash,
            )
        input_tokens = _strict_int(
            usage.get("prompt_tokens", usage.get("input_tokens"))
        )
        output_tokens = _strict_int(
            usage.get("completion_tokens", usage.get("output_tokens"))
        )
        total_tokens = _strict_int(usage.get("total_tokens"))
        if input_tokens is None or output_tokens is None or total_tokens is None:
            raise _ProtocolIssue(
                ChatErrorCode.MISSING_USAGE,
                "response_usage_incomplete",
                resolved_model_id=resolved_model.strip(),
                request_id_hash=request_id_hash,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        choices = payload.get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], Mapping)
        ):
            raise _ProtocolIssue(
                ChatErrorCode.OUTPUT_SCHEMA,
                "response_choices_invalid",
                resolved_model_id=resolved_model.strip(),
                request_id_hash=request_id_hash,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        choice = choices[0]
        message = choice.get("message")
        if (
            choice.get("finish_reason") == "tool_calls"
            or isinstance(message, Mapping)
            and any(key in message for key in ("tool_calls", "function_call"))
        ):
            raise _ProtocolIssue(
                ChatErrorCode.OUTPUT_SCHEMA,
                "model_tool_call_not_allowed",
                resolved_model_id=resolved_model.strip(),
                request_id_hash=request_id_hash,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise _ProtocolIssue(
                ChatErrorCode.OUTPUT_SCHEMA,
                "response_message_content_invalid",
                resolved_model_id=resolved_model.strip(),
                request_id_hash=request_id_hash,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        try:
            json.loads(content, object_pairs_hook=_reject_duplicate_json_keys)
        except json.JSONDecodeError as exc:
            raise _ProtocolIssue(
                ChatErrorCode.INVALID_JSON,
                "model_content_json_invalid",
                resolved_model_id=resolved_model.strip(),
                request_id_hash=request_id_hash,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            ) from exc
        except ValueError:
            raise _ProtocolIssue(
                ChatErrorCode.OUTPUT_SCHEMA,
                "model_content_json_duplicate_key",
                resolved_model_id=resolved_model.strip(),
                request_id_hash=request_id_hash,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            ) from None
        try:
            envelope = FindingEnvelope.model_validate_json(content, strict=True)
            findings = OpenAICompatibleChatAdapter._validate_citations(
                envelope, request, credential=credential
            )
        except _ProtocolIssue as exc:
            if exc.resolved_model_id is None:
                exc.resolved_model_id = resolved_model.strip()
            if exc.request_id_hash is None:
                exc.request_id_hash = request_id_hash
            if exc.input_tokens is None:
                exc.input_tokens = input_tokens
            if exc.output_tokens is None:
                exc.output_tokens = output_tokens
            if exc.total_tokens is None:
                exc.total_tokens = total_tokens
            raise
        except ValidationError as exc:
            raise _ProtocolIssue(
                ChatErrorCode.OUTPUT_SCHEMA,
                "finding_schema_invalid",
                resolved_model_id=resolved_model.strip(),
                request_id_hash=request_id_hash,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            ) from exc
        return (
            findings,
            resolved_model.strip(),
            request_id_hash,
            input_tokens,
            output_tokens,
            total_tokens,
        )

    def _error_with_record(
        self,
        *,
        request: WorkerModelRequest | SingleAgentModelRequest,
        started: float,
        retry_count: int,
        code: str,
        detail: str,
        http_status: int | None = None,
        error_reason: str | None = None,
        resolved_model_id: str | None = None,
        request_id_hash: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> NoReturn:
        topology, worker = _record_scope(request)
        record = ChatCallRecord(
            case_id=request.case_id,
            agent_topology=topology,
            worker=worker,
            requested_model_id=self.config.model_name,
            resolved_model_id=resolved_model_id,
            request_id_hash=request_id_hash,
            http_status=http_status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=max(0, round((self._clock() - started) * 1_000)),
            retry_count=retry_count,
            error_code=code,
            error_detail=detail,
            error_reason=error_reason,
        )
        self._emit(record)
        raise ChatAdapterError(code, detail, record=record) from None

    async def generate(
        self, request: WorkerModelRequest | SingleAgentModelRequest
    ) -> object:
        body = self._request_body(request)
        headers = {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        started = self._clock()
        retry_count = 0
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._client.post(
                    self.config.endpoint_url,
                    headers=headers,
                    json=body,
                )
            except httpx.TimeoutException:
                self._error_with_record(
                    request=request,
                    started=started,
                    retry_count=retry_count,
                    code=ChatErrorCode.TIMEOUT,
                    detail="request_timeout",
                )
            except httpx.RequestError:
                self._error_with_record(
                    request=request,
                    started=started,
                    retry_count=retry_count,
                    code=ChatErrorCode.TRANSPORT,
                    detail="request_transport_failure",
                )

            status = response.status_code
            if status == 429 or 500 <= status <= 599:
                if attempt < self.config.max_retries:
                    retry_count += 1
                    await self._sleep(min(2.0, 0.25 * (2**attempt)))
                    continue
                self._error_with_record(
                    request=request,
                    started=started,
                    retry_count=retry_count,
                    code=ChatErrorCode.RATE_LIMITED
                    if status == 429
                    else ChatErrorCode.SERVER_ERROR,
                    detail="http_status_429" if status == 429 else "http_status_5xx",
                    http_status=status,
                    error_reason=_safe_error_reason(response),
                    request_id_hash=_safe_header_request_id(response.headers),
                )
            if status < 200 or status >= 300:
                self._error_with_record(
                    request=request,
                    started=started,
                    retry_count=retry_count,
                    code=ChatErrorCode.AUTHENTICATION
                    if status in {401, 403}
                    else ChatErrorCode.HTTP_ERROR,
                    detail="http_authentication_failure"
                    if status in {401, 403}
                    else "http_status_4xx_or_3xx",
                    http_status=status,
                    error_reason=_safe_error_reason(response),
                    request_id_hash=_safe_header_request_id(response.headers),
                )
            try:
                (
                    findings,
                    resolved_model_id,
                    request_id_hash,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                ) = self._parse_response(
                    response,
                    request=request,
                    credential=self.config.api_key.get_secret_value(),
                )
            except _ProtocolIssue as issue:
                self._error_with_record(
                    request=request,
                    started=started,
                    retry_count=retry_count,
                    code=issue.code,
                    detail=issue.detail,
                    http_status=status,
                    resolved_model_id=issue.resolved_model_id,
                    request_id_hash=issue.request_id_hash,
                    input_tokens=issue.input_tokens,
                    output_tokens=issue.output_tokens,
                    total_tokens=issue.total_tokens,
                )
            topology, worker = _record_scope(request)
            record = ChatCallRecord(
                case_id=request.case_id,
                agent_topology=topology,
                worker=worker,
                requested_model_id=self.config.model_name,
                resolved_model_id=resolved_model_id,
                request_id_hash=request_id_hash,
                http_status=status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=max(0, round((self._clock() - started) * 1_000)),
                retry_count=retry_count,
            )
            self._emit(record)
            return findings
        raise AssertionError("unreachable retry loop")


ChatAdapter = OpenAICompatibleChatAdapter
RealChatAdapter = OpenAICompatibleChatAdapter
RealChatModelAdapter = OpenAICompatibleChatAdapter
ChatModelAdapter = OpenAICompatibleChatAdapter
