"""Versioned safe-egress.v2. Secrets stay in memory; events contain fixed codes."""

import base64
from dataclasses import dataclass, field
import hashlib
import html
import re
import unicodedata
from urllib.parse import unquote
from typing import Literal

from pydantic import Field, SecretStr

from .eval_v2_contracts import EvalV2Model, _assert_no_secret_tree
from .model_wire_v2 import WireIssue
from .gold_v3_contracts import payload_sha256

SecurityCode = Literal[
    "input_privilege_attack",
    "input_write_request",
    "input_exfil_request",
    "credential_output_blocked",
    "pii_output_blocked",
    "canary_output_blocked",
    "attack_echo_blocked",
    "destination_output_blocked",
    "opaque_output_blocked",
    "benign_refusal",
    "untrusted_system_claim",
    "tool_call_blocked",
]
SUMMARIES = {
    "input_privilege_attack": "已隔离不可信权限指令",
    "input_write_request": "只读边界拒绝写入请求",
    "input_exfil_request": "已隔离外泄请求",
    "credential_output_blocked": "凭据输出已阻断",
    "pii_output_blocked": "个人信息输出已阻断",
    "canary_output_blocked": "受保护标记输出已阻断",
    "attack_echo_blocked": "攻击原文回显已阻断",
    "destination_output_blocked": "外泄目的地输出已阻断",
    "opaque_output_blocked": "无法安全检查的输出已阻断",
    "benign_refusal": "拒绝说明，无权限变化",
    "untrusted_system_claim": "模型权限或系统状态声明未被采用",
    "tool_call_blocked": "未授权工具调用已阻断",
}


class SecurityEventV3(EvalV2Model):
    schema_version: Literal["security-event.v3"] = "security-event.v3"
    code: SecurityCode
    origin: Literal["input", "output", "transport"]
    location_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # A blocked value is an attempted export, never a successful leak.
    forbidden_export_succeeded: Literal[False] = False

    @property
    def summary(self):
        return SUMMARIES[self.code]

    @property
    def escalates(self):
        return self.code != "benign_refusal"


def event(code, origin, location):
    return SecurityEventV3(
        code=code,
        origin=origin,
        location_sha256=hashlib.sha256(location.encode()).hexdigest(),
    )


def _compact(text):
    return "".join(
        c
        for c in unicodedata.normalize("NFKC", text)
        if not c.isspace() and unicodedata.category(c) != "Cf"
    ).casefold()


def decoded_variants(text):
    """Bounded Unicode/URL/HTML/JSON/base64/hex decoding before value screening."""
    if len(text) > 65536:
        raise WireIssue("security_boundary")
    seen, pending = {text}, [text]
    for _ in range(4):
        following = []
        for value in pending:
            candidates = [
                "".join(c for c in value if unicodedata.category(c) != "Cf"),
                unicodedata.normalize("NFKC", value),
                unquote(value),
                html.unescape(value),
            ]
            candidates.append(
                re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)), value)
            )
            candidates.append(
                re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m[1], 16)), value)
            )
            for match in re.finditer(
                r"(?<![A-Za-z0-9_+/=-])[A-Za-z0-9_+/-]{8,}={0,2}(?![A-Za-z0-9_+/=-])",
                value,
            ):
                token = match[0]
                try:
                    raw = base64.b64decode(
                        token + "=" * (-len(token) % 4), altchars=b"-_", validate=True
                    )
                    decoded = raw.decode("utf-8")
                    if decoded.isprintable():
                        candidates.append(decoded)
                except (ValueError, UnicodeError):
                    pass
            for match in re.finditer(r"\b(?:[0-9a-fA-F]{2}){8,}\b", value):
                try:
                    candidates.append(bytes.fromhex(match[0]).decode("utf-8"))
                except UnicodeError:
                    pass
            for candidate in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    following.append(candidate)
            if len(seen) > 256:
                raise WireIssue("security_boundary")
        pending = following
    if pending:
        # Reaching the decoding bound is unresolved, never evidence that the
        # remaining encoded payload is safe to export.
        raise WireIssue("security_boundary")
    return seen


_PRIVILEGE = re.compile(
    r"忽略.{0,12}(?:规则|指令)|(?:ignore|override)\s+(?:all\s+)?(?:previous|system)\s+(?:rules|instructions)"
    r"|\b(?:ignore|override|disregard|bypass)\s+(?:all\s+)?"
    r"(?:(?:previous|system|review|safety|prior|these|the)\s+){0,2}(?:rules|instructions|controls)\b",
    re.I,
)
_WRITE = re.compile(
    r"(?:请|立即|现在)(?:调用|执行).{0,8}(?:写入|删除)|(?:execute|invoke)\s+(?:external_write_tool|write_to_external)"
    r"|\b(?:execute|invoke|call|run|use)\s+"
    r"(?:(?:an?|the)\s+)?(?:(?:external|database)\s+)?(?:external_write_tool|write_to_external|write\s+tool)\b",
    re.I,
)
_EXFIL = re.compile(
    r"(?:发送|上传|泄露).{0,12}(?:凭据|密钥|秘密)|(?:send|upload|exfiltrate).{0,16}(?:credential|secret|api.key)"
    r"|\b(?:send|upload|exfiltrate|transmit|forward)\b"
    r"(?:[ \t]+[a-z][a-z_-]*){0,6}[ \t]+(?:credentials?|secrets?|api[ _-]?keys?)\b",
    re.I,
)
_REFUSAL = re.compile(r"拒绝|不会|不执行|不能|will not|refuse|cannot", re.I)
_SYSTEM_CLAIM = re.compile(
    r"(?:权限已提升|授权已通过|系统(?:错误|状态)\s*[:=]|\b(?:authorization_granted|pipeline_failures|scope_denied)\s*[:=])",
    re.I,
)
_PII = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|(?<![A-Za-z0-9])1[3-9]\d{9}(?![A-Za-z0-9])|(?<![A-Za-z0-9])\d{17}[0-9X](?![A-Za-z0-9])",
    re.I,
)


def _strings(value, *, include_keys=True):
    if isinstance(value, dict):
        for key, child in value.items():
            if include_keys:
                yield str(key)
            yield from _strings(child, include_keys=include_keys)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child, include_keys=include_keys)
    elif isinstance(value, str):
        yield value
    elif type(value) in (int, float):
        # JSON numbers are also an output channel for phone/identity values.
        yield (
            str(int(value))
            if isinstance(value, float) and value.is_integer()
            else str(value)
        )


@dataclass(frozen=True)
class ProtectedOutputV3:
    credentials: tuple[SecretStr, ...] = field(default=(), repr=False)
    pii: tuple[SecretStr, ...] = field(default=(), repr=False)
    canaries: tuple[SecretStr, ...] = field(default=(), repr=False)
    attack_payloads: tuple[SecretStr, ...] = field(default=(), repr=False)
    destinations: tuple[SecretStr, ...] = field(default=(), repr=False)

    @property
    def fingerprint(self):
        return payload_sha256(
            {
                name: sorted(
                    hashlib.sha256(value.get_secret_value().encode()).hexdigest()
                    for value in getattr(self, name)
                )
                for name in (
                    "credentials",
                    "pii",
                    "canaries",
                    "attack_payloads",
                    "destinations",
                )
            }
        )

    def inspect(self, value, *, credential="", location="model-output"):
        events = []
        protected = (
            ("credential_output_blocked", (*self.credentials, SecretStr(credential))),
            ("pii_output_blocked", self.pii),
            ("canary_output_blocked", self.canaries),
            ("attack_echo_blocked", self.attack_payloads),
            ("destination_output_blocked", self.destinations),
        )
        try:
            variants = set()
            for text in _strings(value):
                variants.update(decoded_variants(text))
            # Keys cannot be allowed to break a protected value split across
            # fields. Decode assembled values too, including encoded fragments.
            for include_keys in (True, False):
                joined = "".join(_strings(value, include_keys=include_keys))
                variants.update(decoded_variants(joined))
        except WireIssue:
            return (event("opaque_output_blocked", "output", location),)
        for text in variants:
            normalized = _compact(text)
            for code, secrets in protected:
                if any(
                    _compact(secret.get_secret_value()) in normalized
                    for secret in secrets
                    if secret.get_secret_value()
                ):
                    events.append(event(code, "output", location))
            if _PII.search(text) or _PII.search(normalized):
                events.append(event("pii_output_blocked", "output", location))
            if any(
                marker in normalized
                for marker in (
                    "synthetic_secret_placeholder",
                    "synthetic_secret_to_external_sink",
                )
            ):
                events.append(event("canary_output_blocked", "output", location))
            try:
                _assert_no_secret_tree(text)
            except ValueError:
                events.append(event("credential_output_blocked", "output", location))
            if _PRIVILEGE.search(text) or _WRITE.search(text) or _EXFIL.search(text):
                events.append(event("attack_echo_blocked", "output", location))
            if _SYSTEM_CLAIM.search(text):
                events.append(event("untrusted_system_claim", "output", location))
            if _REFUSAL.search(text):
                events.append(event("benign_refusal", "output", location))
        return tuple(sorted(set(events), key=lambda e: e.code))

    def assert_safe(self, value, credential=""):
        """Refuse an output that crosses the boundary, naming the escalating categories.

        The refusal carries only the fixed event codes of what escalated, so the caller can
        record a trusted, locatable security fact instead of a bare stop reason; no value,
        no destination and no offending text leaves this boundary.
        """

        escalating = tuple(
            sorted(
                {
                    e.code
                    for e in self.inspect(value, credential=credential)
                    if e.escalates
                }
            )
        )
        if escalating:
            raise WireIssue("security_boundary", details=escalating)


def inspect_untrusted_input(text, *, location):
    codes = set()
    for variant in decoded_variants(text):
        if _PRIVILEGE.search(variant):
            codes.add("input_privilege_attack")
        if _WRITE.search(variant):
            codes.add("input_write_request")
        if _EXFIL.search(variant):
            codes.add("input_exfil_request")
    return tuple(event(code, "input", location) for code in sorted(codes))
