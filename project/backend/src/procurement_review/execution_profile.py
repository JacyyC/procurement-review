"""Which execution profile this process runs, and how it is allowed to be wrong.

The workbench has exactly two execution profiles:

* ``test_only`` (the default) — the deterministic Fake line.  It needs no credential and
  its results are synthetic; every response says so.
* ``real`` — the precise line talking to the configured endpoint.  It is opt-in through
  ``REVIEW_EXECUTION_MODE=real`` and it is only usable when *all* of its settings are
  present and valid: the endpoint, the requested and resolved model identities, the model
  credential, the thinking value and an explicitly stated generation deadline (within the
  accepted candidate ceiling of 180 seconds; the product's 30-second default is never
  raised silently).

A half-configured real profile is a controlled failure, not a fallback: the runtime reports
``real_mode_not_configured`` and the business endpoints answer ``503``.  The credential is
read here, on the server, and never appears in a response, a log line or a Run record.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic import SecretStr

EXECUTION_MODE_ENV = "REVIEW_EXECUTION_MODE"
MODEL_ENDPOINT_ENV = "REVIEW_MODEL_ENDPOINT"
MODEL_REQUESTED_ENV = "REVIEW_MODEL_REQUESTED"
MODEL_RESOLVED_ENV = "REVIEW_MODEL_RESOLVED"
MODEL_CREDENTIAL_ENV = "REVIEW_MODEL_API_KEY"
THINKING_ENV = "REVIEW_MODEL_THINKING"
DEADLINE_ENV = "REVIEW_GENERATION_DEADLINE_SECONDS"

TEST_ONLY = "test_only"
REAL = "real"
EXECUTION_MODES = (TEST_ONLY, REAL)

#: The product default of a test-only composition.  A real profile has to state its own
#: deadline; the shorter registered ceiling still applies to every attempt.
DEFAULT_DEADLINE_SECONDS = 30.0
#: The largest Run deadline a composed profile may state.  The accepted demonstration
#: candidates are the Spec P-04 values up to this ceiling; a value beyond it (240 seconds
#: and above) needs its own decision and is refused rather than silently accepted.
MAX_ACCEPTED_DEADLINE_SECONDS = 180.0
#: The thinking value a real profile runs with when it does not state one.  It is the
#: historical, explicitly disabled configuration; ``enabled`` must be configured on
#: purpose, and an unusable value is refused instead of silently falling back to this one.
DEFAULT_THINKING = "disabled"


def execution_mode_value() -> str:
    """The configured execution mode, or a controlled failure for an unknown word."""

    value = (os.environ.get(EXECUTION_MODE_ENV) or TEST_ONLY).strip()
    return value if value in EXECUTION_MODES else TEST_ONLY


def is_real_mode() -> bool:
    return execution_mode_value() == REAL


@dataclass(frozen=True, slots=True)
class SyntheticProfile:
    """The default profile: the deterministic synthetic line, no credential involved."""

    generation_deadline_seconds: float

    def runner(self, store: object) -> object:
        from .local_runtime import ScopedWorkflowRunner

        return ScopedWorkflowRunner(store=store)


@dataclass(frozen=True, slots=True)
class RealProfile:
    """The settings a real profile must state, resolved from the environment."""

    endpoint: str
    requested_model: str
    resolved_model: str
    credential: SecretStr
    generation_deadline_seconds: float
    thinking: str = DEFAULT_THINKING

    def endpoint_config(self) -> object:
        from .business_v3_security import ProtectedOutputV3
        from .eval_f_contracts import FEndpointConfig

        return FEndpointConfig(
            endpoint=self.endpoint,
            requested_model=self.requested_model,
            resolved_model=self.resolved_model,
            mode="json_schema",
            thinking=self.thinking,
            temperature=0.0,
            single_attempt_seconds=90.0,
            prompt_style="baseline",
            protection_sha256=ProtectedOutputV3().fingerprint,
        )

    def runner(self, store: object) -> object:
        """The precise line, wired to the configured endpoint.

        The credential stays on the server: it is handed to the transport that performs
        the send and never reaches a response, a log line or a Run record.
        """

        from .business_v3_security import ProtectedOutputV3
        from .precise_run_runner import PreciseScopedRunner
        from .precise_run_transport import PreciseRunChatTransport

        transport = PreciseRunChatTransport(
            self.endpoint_config(),
            self.credential,
            catalog=None,
            protection=ProtectedOutputV3(),
        )
        return PreciseScopedRunner(
            store=store,
            transport=transport,
            mode=REAL,
            generation_deadline_seconds=self.generation_deadline_seconds,
        )

    def fingerprint(self) -> str:
        """What a Run created under this profile is frozen with, credential excluded."""

        from .gold_v3_contracts import payload_sha256

        return payload_sha256(
            {
                "mode": REAL,
                "endpoint": self.endpoint,
                "requested_model": self.requested_model,
                "resolved_model": self.resolved_model,
                "thinking": self.thinking,
                "generation_deadline_seconds": self.generation_deadline_seconds,
            }
        )


@dataclass(frozen=True, slots=True)
class RealProfileProblem:
    """Why the real profile cannot be used; the reason word is safe to report."""

    reason: str
    detail: str


def real_profile() -> RealProfile | RealProfileProblem:
    """Resolve the real profile, or state exactly what is missing.

    Nothing here raises: the caller decides whether an unusable profile is a startup
    failure (it is) or a degraded one (it is not).
    """

    endpoint = (os.environ.get(MODEL_ENDPOINT_ENV) or "").strip()
    requested = (os.environ.get(MODEL_REQUESTED_ENV) or "").strip()
    resolved = (os.environ.get(MODEL_RESOLVED_ENV) or "").strip()
    credential = (os.environ.get(MODEL_CREDENTIAL_ENV) or "").strip()
    raw_deadline = (os.environ.get(DEADLINE_ENV) or "").strip()
    thinking = (os.environ.get(THINKING_ENV) or DEFAULT_THINKING).strip()
    missing = [
        name
        for name, value in (
            (MODEL_ENDPOINT_ENV, endpoint),
            (MODEL_REQUESTED_ENV, requested),
            (MODEL_RESOLVED_ENV, resolved),
            (MODEL_CREDENTIAL_ENV, credential),
            (DEADLINE_ENV, raw_deadline),
        )
        if not value
    ]
    if missing:
        return RealProfileProblem(
            reason="real_mode_not_configured",
            detail="missing: " + ", ".join(missing),
        )
    from .chat_transport_v2 import THINKING_MODES

    if thinking not in THINKING_MODES:
        return RealProfileProblem(
            reason="real_mode_not_configured",
            detail=f"{THINKING_ENV} is not a registered thinking value",
        )
    try:
        deadline = float(raw_deadline)
    except ValueError:
        return RealProfileProblem(
            reason="real_mode_not_configured", detail="deadline is not a number"
        )
    if not 0 < deadline <= MAX_ACCEPTED_DEADLINE_SECONDS:
        return RealProfileProblem(
            reason="real_mode_not_configured",
            detail=(
                f"deadline must be within (0, {MAX_ACCEPTED_DEADLINE_SECONDS:g}] seconds"
            ),
        )
    if not endpoint.startswith(("https://", "http://")):
        return RealProfileProblem(
            reason="real_mode_not_configured", detail="endpoint is not an http(s) URL"
        )
    return RealProfile(
        endpoint=endpoint,
        requested_model=requested,
        resolved_model=resolved,
        credential=SecretStr(credential),
        generation_deadline_seconds=deadline,
        thinking=thinking,
    )


def execution_profile() -> SyntheticProfile | RealProfile | RealProfileProblem:
    """The profile this process runs under, resolved once per composition."""

    if not is_real_mode():
        return SyntheticProfile(
            generation_deadline_seconds=timing_override(DEFAULT_DEADLINE_SECONDS)
        )
    return real_profile()


def timing_override(default: float) -> float:
    """The explicitly configured deadline, or the product default.

    A configured value outside the accepted bound is clamped to the bound rather than
    silently accepted: the shorter limit always wins, and the ceiling is never widened.
    """

    raw = (os.environ.get(DEADLINE_ENV) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value <= 0:
        return default
    return min(value, MAX_ACCEPTED_DEADLINE_SECONDS)


__all__ = [
    "DEFAULT_DEADLINE_SECONDS",
    "DEFAULT_THINKING",
    "DEADLINE_ENV",
    "EXECUTION_MODES",
    "EXECUTION_MODE_ENV",
    "MAX_ACCEPTED_DEADLINE_SECONDS",
    "MODEL_CREDENTIAL_ENV",
    "MODEL_ENDPOINT_ENV",
    "MODEL_REQUESTED_ENV",
    "MODEL_RESOLVED_ENV",
    "REAL",
    "TEST_ONLY",
    "THINKING_ENV",
    "RealProfile",
    "RealProfileProblem",
    "SyntheticProfile",
    "execution_profile",
    "execution_mode_value",
    "is_real_mode",
    "real_profile",
]
