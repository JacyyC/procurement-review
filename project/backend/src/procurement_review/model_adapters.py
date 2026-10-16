"""Replaceable model adapters, including the V1-01 test fixture and V1-02 Chat adapter."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import Finding, Identifier, WorkerKind
from .worker_contracts import WorkerModelRequest

# Re-export the real adapter from the historical adapter module so callers
# have one stable import surface for both test and local experiment adapters.
from .chat_adapter import (  # noqa: E402
    CHAT_ADAPTER_VERSION,
    CHAT_FINDING_CONTRACT_VERSION,
    EMBEDDING_STATUS,
    RERANK_STATUS,
    ChatAdapter,
    ChatAdapterConfig,
    ChatAdapterError,
    ChatCallRecord,
    ChatModelAdapter,
    FindingEnvelope,
    OpenAICompatibleChatAdapter,
    RealChatAdapter,
    RealChatModelAdapter,
)

__all__ = [
    "CHAT_ADAPTER_VERSION",
    "CHAT_FINDING_CONTRACT_VERSION",
    "EMBEDDING_STATUS",
    "RERANK_STATUS",
    "ChatAdapter",
    "ChatAdapterConfig",
    "ChatAdapterError",
    "ChatCallRecord",
    "ChatModelAdapter",
    "DeterministicFakeModelAdapter",
    "FindingEnvelope",
    "OpenAICompatibleChatAdapter",
    "RealChatAdapter",
    "RealChatModelAdapter",
    "FAKE_MODEL_ADAPTER_VERSION",
    "FAKE_MODEL_FIXTURE_VERSION",
    "DEFAULT_WORKER_MODEL_FIXTURE_PATH",
    "load_worker_model_fixture",
]

FAKE_MODEL_ADAPTER_VERSION = "deterministic-fake-worker-model-v1"
FAKE_MODEL_FIXTURE_VERSION = "worker-model-fixture-v1"
DEFAULT_WORKER_MODEL_FIXTURE_PATH = (
    Path(__file__).parent / "resources" / "workers" / "fixture.v1.json"
)


class _FixtureRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    fixture_id: Identifier
    worker: WorkerKind
    findings: tuple[Finding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def findings_match_worker(self) -> _FixtureRecord:
        if any(finding.worker is not self.worker for finding in self.findings):
            raise ValueError("fixture findings must belong to the declared worker")
        return self


class _FixtureEnvelope(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: Literal["worker-model-fixture.v1"]
    fixture_version: Literal["worker-model-fixture-v1"]
    adapter_version: Literal["deterministic-fake-worker-model-v1"]
    test_only: Literal[True]
    fixtures: tuple[_FixtureRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def fixture_ids_are_unique(self) -> _FixtureEnvelope:
        fixture_ids = tuple(item.fixture_id for item in self.fixtures)
        if len(fixture_ids) != len(set(fixture_ids)):
            raise ValueError("worker model fixture IDs must be unique")
        return self


def load_worker_model_fixture(
    path: Path = DEFAULT_WORKER_MODEL_FIXTURE_PATH,
) -> _FixtureEnvelope:
    """Read and strictly validate the versioned test-only fixture."""

    return _FixtureEnvelope.model_validate_json(path.read_text(encoding="utf-8"))


class _FakeModelCallHandle:
    """The handle of one already-received fake call; only its result is asynchronous."""

    __slots__ = ("_findings",)

    def __init__(self, findings: tuple[Finding, ...]) -> None:
        self._findings = findings

    async def result(self) -> object:
        return self._findings


class DeterministicFakeModelAdapter:
    """Return only exact outputs declared in the package's test fixture."""

    adapter_version = FAKE_MODEL_ADAPTER_VERSION
    test_only = True

    def __init__(
        self,
        fixture_path: Path = DEFAULT_WORKER_MODEL_FIXTURE_PATH,
    ) -> None:
        envelope = load_worker_model_fixture(fixture_path)
        self.fixture_version = envelope.fixture_version
        self._fixtures = MappingProxyType(
            {item.fixture_id: item for item in envelope.fixtures}
        )

    def resolve(self, request: WorkerModelRequest) -> tuple[Finding, ...]:
        """Synchronously resolve one immutable request to its declared answer."""

        record = self._fixtures.get(request.model_fixture_id)
        if record is None:
            raise LookupError("model fixture ID is not declared")
        if record.worker is not request.worker:
            raise ValueError("model fixture belongs to a different worker")
        return record.findings

    def start(self, request: WorkerModelRequest) -> _FakeModelCallHandle:
        """Receive one request synchronously; the network-free answer comes later.

        This is the receive half the gated model-call boundary requires: the request is
        taken and resolved here, before the caller's transaction closes, and no wait of any
        kind is hidden behind a coroutine that has not started yet.
        """

        return _FakeModelCallHandle(self.resolve(request))

    async def generate(self, request: WorkerModelRequest) -> object:
        return await self.start(request).result()
