"""Closed-field projection and recursive guards for V1-03 persisted records.

Every persisted business/observability record passes through one versioned allow-list
before it is written, and the same rules drive the offline scanner.  Detection is
pattern/key based and deliberately never claims to be exhaustive: it is one half of the
control, the closed allow-list is the other.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

FIELD_GUARD_VERSION = "review-field-guard.v1"

FORBIDDEN_KEY_PARTS = (
    "authorization",
    "bearer",
    "chain_of_thought",
    "credential_secret",
    "dsn",
    "idempotency_key",
    "password",
    "prompt",
    "reasoning",
    "secret",
    "thought",
    "token",
)
SENSITIVE_VALUE_PATTERNS = (
    ("local_credential", re.compile(r"rvk1\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{16,}")),
    ("bearer_header", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{16,}")),
    (
        "dsn_with_secret",
        re.compile(r"(?i)\b(?:postgres|postgresql|psycopg)://[^\s:@/]+:[^\s@/]+@"),
    ),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("email_pii", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("cn_mobile_pii", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("national_id_pii", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
)

#: Product-minted identity and digest formats (AC-04.4).
#:
#: A hex identity or digest legitimately contains digit runs that *look* like a phone
#: number or a national ID — ``run-19436117970e4786bf273aa093fa7332`` is one real example
#: (X-8).  Such a value is only exempt from the value patterns when the record declares the
#: slot as an identifier/digest **and** the value matches one of these strict, registered
#: formats; the format and its minting site are the proof, not the field name.  Anything
#: else that matches a sensitive pattern is still rejected, including a value that only
#: pretends to be one of these identities.
TRUSTED_VALUE_FORMATS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("run_id", re.compile(r"run-[0-9a-f]{32}")),
    ("trace_id", re.compile(r"trace-[0-9a-f]{32}")),
    ("event_id", re.compile(r"event-run-[0-9a-f]{32}-[0-9]{4}")),
    ("audit_id", re.compile(r"audit-[0-9a-f]{32}")),
    ("receipt_id", re.compile(r"receipt-[0-9a-f]{32}")),
    ("credential_id", re.compile(r"cred-[0-9a-f]{32}")),
    ("denial_id", re.compile(r"denial-[0-9a-f]{32}")),
    ("owner_id", re.compile(r"owner-[0-9a-f]{32}")),
    ("document_id", re.compile(r"document-[0-9a-f]{64}")),
    ("chunk_id", re.compile(r"chunk-[0-9a-f]{64}")),
    ("checksum", re.compile(r"sha256:[0-9a-f]{64}")),
    # The bare digest spelling this product's own contracts declare (``span_sha256``,
    # ``content_sha256``): a whole 64-hex value is a hash, never free text, so a digit
    # run inside it is not a phone number and refusing it would refuse the product's
    # own record.
    ("digest", re.compile(r"[0-9a-f]{64}")),
)

ALLOWED_PERSISTED_KEYS = frozenset(
    {
        # Run / generation / attempt
        "run_id",
        "trace_id",
        "tenant_id",
        "case_id",
        "case_version",
        "status",
        "revision",
        "resume_count",
        "created_by_subject",
        "current_subject_id",
        "request_checksum",
        "document_version",
        "dataset_version",
        "index_version",
        "rules_version",
        "isolation",
        "error_code",
        "error_message",
        "created_at",
        "updated_at",
        "generation",
        "attempt_no",
        "worker",
        "outcome",
        "abandoned",
        "discarded_late",
        "started_at",
        "finished_at",
        "first_error",
        "last_error",
        "owner_id",
        "subject_id",
        "credential_id",
        "authorization_revision",
        "heartbeat_at",
        "lease_expires_at",
        "deadline_at",
        "input_checksum",
        "contract_checksum",
        "result_checksum",
        "adopted_at",
        "result",
        # Summary / execution / domain contracts
        "summary",
        "execution",
        "schema_version",
        "execution_version",
        "state_version",
        "branch_results",
        "findings",
        "conflicts",
        "missing_information",
        "failure_reasons",
        "highest_risk",
        "recommendation",
        "human_review_status",
        "escalation",
        "finding_id",
        "finding_key",
        "worker_kind",
        "assertion",
        "risk",
        "statement",
        "recommended_action",
        "evidence_refs",
        "business_fact_refs",
        "source_kind",
        "source_id",
        "version",
        "page_number",
        "section",
        "chunk_id",
        "content_checksum",
        "tool_name",
        "record_id",
        "as_of",
        "fact_status",
        "failure_reason",
        "read_only",
        "model_fixture_id",
        "case_facts",
        "document_evidence",
        "conflict_reason",
        "cause_code",
        "generated_reason_codes",
        # Trace / audit
        "event_id",
        "event_type",
        "step",
        "occurred_at",
        "input_summary",
        "output_summary",
        "output_checksum",
        "source_versions",
        "duration_ms",
        "token_count",
        "cost_amount",
        "cost_currency",
        "redaction_status",
        "error",
        "sequence",
        "audit_id",
        "event_class",
        "action_type",
        "actor",
        "previous_status",
        "new_status",
        "previous_revision",
        "new_revision",
        "idempotency_digest",
        "receipt_id",
        "detail_code",
        "api_version",
        "simulation_only",
        "external_write_performed",
        "audit_schema_version",
        "human_gate_status",
        "committed_revision",
        "decision",
        "reviewed_recommendation",
        "recorded_at",
    }
)


@dataclass(slots=True)
class ScanFinding:
    category: str
    location: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "location": self.location,
            "detail": self.detail,
        }


@dataclass(slots=True)
class ScanReport:
    scanned_values: int = 0
    findings: list[ScanFinding] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings

    def as_dict(self) -> dict[str, Any]:
        return {
            "field_guard_version": FIELD_GUARD_VERSION,
            "scanned_values": self.scanned_values,
            "findings": [item.as_dict() for item in self.findings],
            "categories": sorted({item.category for item in self.findings}),
        }


def iter_leaves(payload: Any, *, path: str = "$") -> Iterable[tuple[str, Any]]:
    """Yield every scalar leaf with a stable JSON-ish path."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            yield from iter_leaves(value, path=f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            yield from iter_leaves(value, path=f"{path}[{index}]")
    else:
        yield path, payload


def iter_keys(payload: Any, *, path: str = "$") -> Iterable[str]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            yield str(key)
            yield from iter_keys(value, path=f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            yield from iter_keys(value)


def scan_payload(payload: Any, *, location: str) -> ScanReport:
    """Scan one structured value for forbidden keys and sensitive value shapes."""

    report = ScanReport()
    for leaf_path, value in iter_leaves(payload):
        report.scanned_values += 1
        if not isinstance(value, str):
            continue
        for category, pattern in SENSITIVE_VALUE_PATTERNS:
            if pattern.search(value):
                report.findings.append(
                    ScanFinding(
                        category=category,
                        location=f"{location}:{leaf_path}",
                        detail=f"{category} pattern matched",
                    )
                )
    return report


def scan_text(text: str, *, location: str) -> ScanReport:
    report = ScanReport()
    report.scanned_values = 1
    for category, pattern in SENSITIVE_VALUE_PATTERNS:
        if pattern.search(text):
            report.findings.append(
                ScanFinding(
                    category=category,
                    location=location,
                    detail=f"{category} pattern matched",
                )
            )
    return report


def trusted_value_kind(value: str) -> str | None:
    """Return the registered identity/digest kind of one value, or ``None``.

    Only whole values count: a substring that happens to look like a phone number never
    makes its container trusted, and a value that merely starts with ``run-`` is not an
    identity unless the rest of the registered format is there too.
    """

    for kind, pattern in TRUSTED_VALUE_FORMATS:
        if pattern.fullmatch(value):
            return kind
    return None


def scan_typed_value(value: str, *, kind: str, location: str, path: str) -> ScanReport:
    """Scan one leaf whose declared type is an identifier or a digest.

    A registered identity/digest format is trusted by type *and* format; every other
    value in such a slot is still screened, so a phone number pasted into an identifier
    field, or text pretending to be a ``run-`` identity, is rejected.
    """

    report = ScanReport()
    report.scanned_values = 1
    if (
        kind in {"any", "checksum", "identifier"}
        and trusted_value_kind(value) is not None
    ):
        return report
    for category, pattern in SENSITIVE_VALUE_PATTERNS:
        if pattern.search(value):
            report.findings.append(
                ScanFinding(
                    category=category,
                    location=f"{location}:{path}",
                    detail=f"{category} pattern matched",
                )
            )
    return report


def scan_forbidden_keys(payload: Any, *, location: str) -> ScanReport:
    """Reject documentation/trace records that carry a forbidden key name."""

    report = ScanReport()
    for key in iter_keys(payload):
        report.scanned_values += 1
        if key in ALLOWED_PERSISTED_KEYS:
            # The closed allow-list is the primary control; an allowed name such as
            # ``token_count`` is not a credential field.
            continue
        lowered = key.lower()
        for part in FORBIDDEN_KEY_PARTS:
            if part in lowered:
                report.findings.append(
                    ScanFinding(
                        category="forbidden_key",
                        location=f"{location}:{key}",
                        detail=f"key contains {part}",
                    )
                )
    return report


def merge(*reports: ScanReport) -> ScanReport:
    merged = ScanReport()
    for report in reports:
        merged.scanned_values += report.scanned_values
        merged.findings.extend(report.findings)
    return merged


def unknown_keys(
    payload: Mapping[str, Any], *, allow: Sequence[str] = ()
) -> tuple[str, ...]:
    """Return top-level keys outside the closed allow-list (for projections)."""

    permitted = set(ALLOWED_PERSISTED_KEYS) | set(allow)
    return tuple(sorted(key for key in payload if key not in permitted))


def canary_values() -> tuple[tuple[str, str], ...]:
    """Synthetic canaries used by the scanner; they are never real secrets."""

    return (
        ("local_credential", "rvk1.cred-canary.ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
        ("bearer_header", "Bearer canarytoken0000000000000000"),
        ("openai_key", "sk-" + "c" * 32),
        ("email_pii", "canary.person@example.invalid"),
        ("cn_mobile_pii", "13800001234"),
        ("national_id_pii", "110101199003070000"),
        ("dsn_with_secret", "postgresql://canary:canary@127.0.0.1:5432/canary"),
    )


def canary_digest(location: str) -> str:
    return hashlib.sha256(location.encode("utf-8")).hexdigest()[:16]


def json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


__all__ = [
    "ALLOWED_PERSISTED_KEYS",
    "FIELD_GUARD_VERSION",
    "FORBIDDEN_KEY_PARTS",
    "SENSITIVE_VALUE_PATTERNS",
    "TRUSTED_VALUE_FORMATS",
    "ScanFinding",
    "ScanReport",
    "canary_digest",
    "canary_values",
    "iter_keys",
    "iter_leaves",
    "json_text",
    "merge",
    "scan_forbidden_keys",
    "scan_payload",
    "scan_text",
    "scan_typed_value",
    "trusted_value_kind",
    "unknown_keys",
]
