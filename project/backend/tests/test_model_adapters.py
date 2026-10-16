from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError

from procurement_review.contracts import WorkerKind
from procurement_review.model_adapters import (
    FAKE_MODEL_ADAPTER_VERSION,
    FAKE_MODEL_FIXTURE_VERSION,
    DeterministicFakeModelAdapter,
    load_worker_model_fixture,
)
from procurement_review.worker_contracts import EvidenceContext, WorkerModelRequest


def test_fake_model_is_versioned_test_only_and_returns_declared_output() -> None:
    envelope = load_worker_model_fixture()
    adapter = DeterministicFakeModelAdapter()
    fixture = next(
        item for item in envelope.fixtures if item.fixture_id == "policy-low"
    )
    evidence_ref = fixture.findings[0].evidence_refs[0]
    request = WorkerModelRequest(
        worker=WorkerKind.POLICY,
        case_id="CASE-001",
        model_fixture_id=fixture.fixture_id,
        evidence_contexts=(
            EvidenceContext(
                evidence_ref=evidence_ref,
                content="Competitive bidding is required above the procurement threshold.",
            ),
        ),
    )

    first = asyncio.run(adapter.generate(request))
    second = asyncio.run(adapter.generate(request))

    assert envelope.schema_version == "worker-model-fixture.v1"
    assert envelope.fixture_version == FAKE_MODEL_FIXTURE_VERSION
    assert envelope.adapter_version == FAKE_MODEL_ADAPTER_VERSION
    assert envelope.test_only is True
    assert adapter.adapter_version == FAKE_MODEL_ADAPTER_VERSION
    assert adapter.fixture_version == FAKE_MODEL_FIXTURE_VERSION
    assert adapter.test_only is True
    assert first == fixture.findings
    assert second == fixture.findings
    assert first == second


def test_fake_model_rejects_undeclared_or_cross_worker_fixture() -> None:
    adapter = DeterministicFakeModelAdapter()
    envelope = load_worker_model_fixture()
    evidence_ref = (
        next(item for item in envelope.fixtures if item.fixture_id == "policy-low")
        .findings[0]
        .evidence_refs[0]
    )

    unknown = WorkerModelRequest(
        worker=WorkerKind.POLICY,
        case_id="CASE-001",
        model_fixture_id="not-declared",
        evidence_contexts=(
            EvidenceContext(evidence_ref=evidence_ref, content="Declared context."),
        ),
    )
    cross_worker = WorkerModelRequest(
        worker=WorkerKind.CONTRACT,
        case_id="CASE-001",
        model_fixture_id="policy-low",
        evidence_contexts=(
            EvidenceContext(evidence_ref=evidence_ref, content="Declared context."),
        ),
    )

    with pytest.raises(LookupError, match="not declared"):
        asyncio.run(adapter.generate(unknown))
    with pytest.raises(ValueError, match="different worker"):
        asyncio.run(adapter.generate(cross_worker))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("test_only", False),
        ("schema_version", "worker-model-fixture.v999"),
        ("fixture_version", "worker-model-fixture-v999"),
        ("adapter_version", "fake-that-claims-real-inference"),
    ),
)
def test_fake_model_rejects_tampered_fixture_metadata(
    tmp_path, field: str, value: object
) -> None:
    envelope = load_worker_model_fixture()
    payload = envelope.model_dump(mode="json")
    payload[field] = value
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        DeterministicFakeModelAdapter(path)
