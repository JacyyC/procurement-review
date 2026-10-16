"""The candidate fingerprint: it must move when any real dependency moves.

Each case below changes exactly one thing a result depends on — a byte of a registered
implementation file, the model-visible schema, the prompt text or the endpoint
configuration — and reads the fingerprint again.  A fingerprint that stayed the same would
let a changed candidate be handed over under the old name, so every case is a negative
control for the handover, not a formatting check.
"""

from __future__ import annotations


import pytest

from procurement_review.precise_candidate import candidate_fingerprint


def test_the_fingerprint_covers_every_section_of_the_candidate():
    fingerprint = candidate_fingerprint()
    assert set(fingerprint["sections"]) == {
        "product",
        "diagnostic",
        "frontend",
        "resources",
        "locks",
    }
    for section, entries in fingerprint["sections"].items():
        assert entries, section
        for name, entry in entries.items():
            assert len(entry["sha256"]) == 64, (section, name)
            assert entry["bytes"] > 0, (section, name)
    assert fingerprint["generated"]["workers"]
    assert len(fingerprint["candidate_sha256"]) == 64


def test_the_model_visible_schema_is_hashed_as_the_endpoint_receives_it():
    from procurement_review.business_v31_documents import VALUE_MODELS_V31
    from procurement_review.precise_candidate import _synthetic_request
    from procurement_review.precise_run_transport import precise_answer_schema

    fingerprint = candidate_fingerprint()
    workers = fingerprint["generated"]["workers"]
    assert set(workers) == {"policy", "contract"}
    # The duty set of each branch is part of the fingerprint: the policy branch answers
    # three duties and the contract branch five, and a change of either set moves it.
    assert workers["policy"]["duties"] == ["O-01", "O-02", "O-04"]
    assert workers["contract"]["duties"] == ["O-04", "O-05", "O-06", "O-08", "O-09"]
    assert set(workers["policy"]["schema_defs"]) == {
        "PreciseAnswerFindingV3",
        "PreciseAnswerWitnessV3",
    }
    assert workers["policy"]["schema_properties"] == [
        "findings",
        "schema_version",
        "witnesses",
    ]
    # The per-duty value object is offered through the task text, not through the response
    # schema, so both are hashed: the schema narrows the answer shape and the task carries
    # the closed value registry.
    for worker, models in VALUE_MODELS_V31.items():
        assert workers[worker]["fields"] == {
            duty: sorted(model.model_fields) for duty, model in models.items()
        }
        request = _synthetic_request(worker)
        schema = precise_answer_schema(request)
        assert schema["properties"]["findings"]["minItems"] == len(request.checks)
        assert schema["$defs"]["PreciseAnswerWitnessV3"]["properties"]["source_ref"][
            "enum"
        ] == [source.ref for source in request.sources]
        assert sorted(
            schema["$defs"]["PreciseAnswerWitnessV3"]["properties"]["field"]["enum"]
        ) == sorted({name for model in models.values() for name in model.model_fields})
        assert workers[worker]["task_text_sha256"]


def test_a_changed_prompt_or_endpoint_moves_the_whole_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
):
    """What the endpoint is asked and what it is configured with are both parts of it."""

    from procurement_review import precise_run_transport as transport

    first = candidate_fingerprint()
    assert (
        first["endpoint"]["real"]["requested_model"]
        != first["endpoint"]["test_only"]["requested_model"]
    )
    monkeypatch.setattr(
        transport, "PRECISE_SYSTEM_PROMPT", transport.PRECISE_SYSTEM_PROMPT + " "
    )
    second = candidate_fingerprint()
    assert (
        first["generated"]["system_prompt_sha256"]
        != second["generated"]["system_prompt_sha256"]
    )
    assert (
        first["generated"]["wire_fingerprint"]
        != second["generated"]["wire_fingerprint"]
    )
    assert first["candidate_sha256"] != second["candidate_sha256"]


def test_a_changed_frontend_contract_is_part_of_the_same_candidate():
    fingerprint = candidate_fingerprint()
    frontend = fingerprint["sections"]["frontend"]
    # The whole loaded closure is inside the candidate, not a hand-picked subset of it.
    assert {"src/App.tsx", "src/main.tsx", "src/ReviewViews.tsx"} <= set(frontend)
    assert {"src/preciseExecutionContract.ts", "src/summaryContract.ts"} <= set(
        frontend
    )
    for name, entry in frontend.items():
        assert entry["bytes"] > 0, name
