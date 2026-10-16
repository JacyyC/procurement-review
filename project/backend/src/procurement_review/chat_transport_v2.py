"""A single physical Chat attempt, with no hidden HTTP retries or tools."""

from __future__ import annotations

import asyncio
import json
import hashlib
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import Field, SecretStr, model_validator

from .eval_v2_contracts import EvalV2Model, _assert_no_secret_tree
from .gold_v3_contracts import payload_sha256
from .model_wire_v2 import (
    BranchRequestV2,
    PROMPT_VERSION,
    WIRE_VERSION,
    ValidatedFindingV2,
    WireIssue,
    messages,
    parse_findings,
    response_schema,
    strict_json,
)
from .run_guard import secret_value_paths


#: The accepted values of the supplier's ``thinking.type`` object.  The wire convention
#: (``{"thinking": {"type": ...}}``) is the one the endpoint's public documentation
#: declares for the registered model family and the one the existing probe registrations
#: already used for ``disabled``; ``enabled`` is the frozen V1-08 candidate.  ``None``
#: means the parameter is not sent at all, and stays a distinct, readable state.
ThinkingMode = Literal["disabled", "enabled"]
THINKING_MODES: tuple[str, ...] = ("disabled", "enabled")

#: The only fields of a refusal body this transport may read: the supplier's fixed error
#: code and category.  They are compared against the fixed vocabulary below and never
#: persisted, quoted or logged; the records carry this project's own classification code.
PROVIDER_ERROR_CODE_FIELDS: tuple[str, ...] = ("code", "type")
#: The fixed subscription/quota vocabulary, normalized to lowercase alphanumerics (the
#: historical ``InvalidSubscription`` refusal is the first entry).  A match classifies the
#: refusal as account exhaustion, which stops the batch immediately and is never retried.
ACCOUNT_QUOTA_SIGNALS = frozenset(
    {
        "accountdeactivated",
        "insufficientquota",
        "invalidsubscription",
        "quotaexhausted",
        "subscriptionexpired",
        "subscriptioninactive",
    }
)


def provider_error_signal(raw: object) -> str | None:
    """The normalized fixed error code/category of one refusal body, or ``None``.

    Only the registered ``error.code``/``error.type`` fields are read, rendered as a
    lowercase alphanumeric token; the supplier's message, credential or any other free
    text can never enter the classification or the records.
    """

    if type(raw) is not dict:
        return None
    error = raw.get("error")
    if type(error) is not dict:
        return None
    for field in PROVIDER_ERROR_CODE_FIELDS:
        value = error.get(field)
        if type(value) is str and value.strip():
            token = re.sub(r"[^a-z0-9]", "", value.strip().lower())
            if token:
                return token
    return None


class EndpointConfigV2(EvalV2Model):
    endpoint: str
    requested_model: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,160}$")
    resolved_model: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,160}$")
    mode: Literal["json_schema", "json_object", "text"]
    thinking: ThinkingMode | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, gt=0)
    single_attempt_seconds: float = Field(default=45.0, gt=0, le=90)

    @model_validator(mode="after")
    def endpoint_is_not_a_secret(self):
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("invalid endpoint")
        return self

    @property
    def fingerprint(self):
        return payload_sha256(
            {
                "endpoint_config": self.model_dump(mode="json"),
                "wire": WIRE_VERSION,
                "prompt": PROMPT_VERSION,
                "implementation": {
                    name: hashlib.sha256(
                        Path(__file__).with_name(name).read_bytes()
                    ).hexdigest()
                    for name in (
                        "model_wire_v2.py",
                        "chat_transport_v2.py",
                        "model_attempt_v2.py",
                        "model_attempt_projection.py",
                        "model_probe_run.py",
                        "model_probe_v2.py",
                        "eval_attempt_ledger.py",
                        "eval_attempt_schema.py",
                        "eval_attempt_budget.py",
                        "eval_attempt_manifest.py",
                        "gold_v31_contracts.py",
                        "eval_v2_contracts.py",
                    )
                },
            }
        )


def request_payload(
    config: EndpointConfigV2,
    request: BranchRequestV2,
    correction: str | None = None,
    correction_fields: tuple[str, ...] = (),
) -> dict:
    payload = {
        "model": config.requested_model,
        "messages": messages(request, correction, correction_fields),
    }
    if config.mode == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "finding_envelope_v2",
                "strict": True,
                "schema": response_schema(request),
            },
        }
    elif config.mode == "json_object":
        payload["response_format"] = {"type": "json_object"}
    if config.thinking is not None:
        payload["thinking"] = {"type": config.thinking}
    for key in ("temperature", "top_p", "seed", "max_tokens"):
        if (value := getattr(config, key)) is not None:
            payload[key] = value
    return payload


@dataclass(frozen=True)
class AttemptResult:
    findings: tuple[ValidatedFindingV2, ...] = ()
    error_code: str | None = None
    correctable: bool = False
    retryable: bool = False
    http_status: int | None = None
    resolved_model_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_bytes: int | None = None
    reasoning_tokens: int | None = None
    reasoning_bytes: int | None = None
    finish_reason: str | None = None
    received_body: bool = False
    format_diagnostic: str | None = None
    format_fields: tuple[str, ...] = ()
    #: Bounded location tokens of one refused answer: slot/index, authorized duty,
    #: validation category, expected word, the basic JSON type actually found and a
    #: bounded size.  Never a value, an answer text or a prompt.
    format_diagnostics: tuple[str, ...] = ()
    retry_after: float = 0.0

    @property
    def usage_unknown(self):
        return self.input_tokens is None or self.output_tokens is None


# Preserve the existing conservative output boundary in C. E owns the separate
# refusal/attack event semantics; neither model prose nor this transport grants tools.
_LEGACY_MARKERS = (
    "synthetic_secret_placeholder",
    "synthetic_secret_to_external_sink",
    "external_sink",
    "write_to_external",
    "external_write_tool",
    "send_to_external",
    "调用外部写入",
    "执行外部写入",
    "外部写入工具",
)


def assert_safe_output(value: object, credential: str = "") -> None:
    rendered = json.dumps(value, ensure_ascii=False)
    if (
        (credential and credential in rendered)
        or any(marker in rendered.lower() for marker in _LEGACY_MARKERS)
        or secret_value_paths(value, credential=credential)
    ):
        raise WireIssue("security_boundary")
    try:
        _assert_no_secret_tree(value)
    except ValueError:
        raise WireIssue("security_boundary") from None


def _tokens(value):
    return value if type(value) is int and value >= 0 else None


def format_diagnostic(content: str) -> str:
    try:
        strict_json(content)
    except ValueError as exc:
        if str(exc) == "duplicate key":
            return "duplicate_key"
        if str(exc) == "non-json number":
            return "invalid_number"
        if content.lstrip().startswith("```"):
            return "markdown_fence"
        return "json_syntax" if content.lstrip().startswith("{") else "non_json_text"
    return "valid_json"


def decode_response(
    response: httpx.Response,
    request: BranchRequestV2,
    config: EndpointConfigV2,
    credential: str,
    *,
    output_guard=assert_safe_output,
    parse_output=parse_findings,
) -> AttemptResult:
    meta = {
        "http_status": response.status_code,
        "response_bytes": len(response.content),
    }
    try:
        raw = strict_json(response.content)
    except (ValueError, UnicodeError):
        raw = None
    if isinstance(raw, dict):
        usage = raw.get("usage")
        if isinstance(usage, dict):
            inp = _tokens(usage.get("prompt_tokens", usage.get("input_tokens")))
            out = _tokens(usage.get("completion_tokens", usage.get("output_tokens")))
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict):
                meta["reasoning_tokens"] = _tokens(details.get("reasoning_tokens"))
            if inp is not None and out is not None:
                meta.update(input_tokens=inp, output_tokens=out)
        if raw.get("model") == config.resolved_model:
            meta["resolved_model_id"] = config.resolved_model
        elif raw.get("model") is not None:
            # Never persist a server-controlled model string as a system identity.
            return AttemptResult(error_code="model_identity_drift", **meta)
    if response.status_code != 200:
        status = response.status_code
        signal = provider_error_signal(raw)
        if signal is not None and signal in ACCOUNT_QUOTA_SIGNALS:
            # The account or subscription behind this batch is exhausted: the refusal has
            # its own fixed classification, is never retried and never waits for the
            # dense-429 window (Spec P-05).
            code = "account_quota_exhausted"
        elif status in {401, 403}:
            code = "authentication"
        elif status == 429:
            code = "rate_limited"
        elif status in {400, 404, 422}:
            # A refused request with no readable provider code stays this branch's own
            # registered outcome.  A refusal whose code is outside the fixed vocabulary
            # cannot be classified from what may be read, so it becomes the versioned
            # unregistered error and stops the batch instead of passing as an ordinary
            # business failure.
            code = "unregistered_provider_error" if signal else "unsupported_request"
        elif 500 <= status <= 599:
            code = "server_error"
        else:
            code = "unregistered_provider_error"
        try:
            delay = max(0.0, float(response.headers.get("retry-after", "0")))
            if not 0 <= delay <= 10:
                delay = 11.0  # larger/unbounded values prevent retry within this policy
        except ValueError:
            delay = 11.0
        return AttemptResult(
            error_code=code,
            retryable=status == 429 or status in {500, 502, 503, 504},
            retry_after=delay,
            **meta,
        )
    if not isinstance(raw, dict) or "resolved_model_id" not in meta:
        return AttemptResult(error_code="response_envelope_invalid", **meta)
    choices = raw.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
    ):
        return AttemptResult(error_code="response_envelope_invalid", **meta)
    choice = choices[0]
    finish = choice.get("finish_reason")
    meta["finish_reason"] = (
        finish
        if isinstance(finish, str)
        and finish in {"stop", "length", "content_filter", "tool_calls"}
        else "other"
    )
    message = choice.get("message")
    if not isinstance(message, dict):
        return AttemptResult(error_code="response_envelope_invalid", **meta)
    if (
        message.get("tool_calls")
        or message.get("function_call")
        or finish == "tool_calls"
    ):
        return AttemptResult(error_code="tool_call_forbidden", **meta)
    if message.get("refusal") or finish == "content_filter":
        return AttemptResult(error_code="safety_refusal", **meta)
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str):
        meta["reasoning_bytes"] = len(reasoning.encode())
    content = message.get("content")
    meta["received_body"] = isinstance(content, str) and bool(content.strip())
    if isinstance(content, str):
        try:
            # Decode escaped JSON before scanning: escaping is not a safe exit.
            try:
                scanned = strict_json(content)
            except ValueError:
                scanned = content
            output_guard(scanned, credential)
        except WireIssue:
            return AttemptResult(error_code="security_boundary", **meta)
    if finish == "length":
        return AttemptResult(error_code="length_truncated", **meta)
    if finish != "stop" or not meta["received_body"]:
        return AttemptResult(error_code="response_envelope_invalid", **meta)
    try:
        findings = parse_output(content, request)
    except WireIssue as issue:
        return AttemptResult(
            error_code=issue.code,
            correctable=issue.correctable,
            format_diagnostic=format_diagnostic(content),
            format_fields=issue.fields,
            format_diagnostics=issue.details,
            **meta,
        )
    return AttemptResult(findings=findings, **meta)


class ChatTransportV2:
    def __init__(
        self,
        config: EndpointConfigV2,
        credential: SecretStr,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config = config
        self.credential = credential
        self.transport = transport

    @property
    def configuration_fingerprint(self):
        return self.config.fingerprint

    def decode(self, response, request, credential):
        return decode_response(response, request, self.config, credential)

    async def post_once(self, payload: dict, seconds: float):
        """Perform exactly one physical send, with no retry and no decoding.

        The caller owns what happens to the receipt: the frozen line decodes it right
        away, and a line that has to admit the result before it is read can do so between
        the response and its decoding.
        """

        key = self.credential.get_secret_value()
        if key and key in json.dumps(payload, ensure_ascii=False):
            return AttemptResult(error_code="credential_in_request")
        try:
            async with asyncio.timeout(
                min(seconds, self.config.single_attempt_seconds)
            ):
                async with httpx.AsyncClient(
                    transport=self.transport,
                    trust_env=False,
                    follow_redirects=False,
                    timeout=seconds,
                ) as client:
                    return await client.post(
                        self.config.endpoint,
                        json=payload,
                        headers={"Authorization": "Bearer " + key},
                    )
        except (TimeoutError, httpx.TimeoutException):
            return AttemptResult(error_code="timeout", retryable=True)
        except httpx.TransportError:
            return AttemptResult(error_code="transport", retryable=True)

    async def send(
        self, payload: dict, request: BranchRequestV2, seconds: float
    ) -> AttemptResult:
        receipt = await self.post_once(payload, seconds)
        if isinstance(receipt, AttemptResult):
            return receipt
        return self.decode(receipt, request, self.credential.get_secret_value())
