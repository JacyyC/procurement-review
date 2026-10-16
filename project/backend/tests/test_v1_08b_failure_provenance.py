"""V1-08B: the failure path of one H Run keeps its own provenance (Spec HC-01).

A Run that ended without a committed summary is a legitimate measured outcome, and these
tests pin what its record must still prove: its frozen snapshot, its lifecycle event on
this Run's own Trace, its closed generation closure, the adoption ledger pairing its
checkpoints with the attempts they adopted together with the receipts that settled them,
its timing, and the sources the executed branches actually used.  Every directed damage is
applied to a disposable copy the test itself creates, with the outer observation commit
checksum recomputed the way an attacker could — so the refusal has to come from the
semantic layer, not from a digest mismatch — and each damage must surface by name through
the direct entry, the completion proof and the CLI, instead of being read as a
``product_run_*`` outcome.
"""

import asyncio
import copy
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from procurement_review.eval_h_cli import main as h_cli
from procurement_review.eval_h_consumption import (
    completion_proof,
    observation_failures,
)
from procurement_review.eval_h_contracts import H_PLAN_V2
from procurement_review.eval_h_fixture import make_disposable
from procurement_review.eval_h_mock import mock_transport
from procurement_review.eval_h_runner import run, setup
from procurement_review.eval_h_seal import HSeal
from procurement_review.gold_v3_contracts import payload_sha256


class SummaryFailure:
    """A fault controller that fails exactly the first summary commit."""

    enabled = True
    fired = False

    def barrier(self, site):
        if site == "during_summary_commit" and not self.fired:
            self.fired = True
            raise RuntimeError("controlled summary failure")


class ContractStorageOutage:
    """A fault controller that interrupts the contract branch of the first Run."""

    enabled = True
    fired = False

    def barrier(self, site):
        if (
            site == "during_model_entry_authorization:precise.contract"
            and not self.fired
        ):
            from procurement_review.checkpointing import StorageUnavailableSignal

            self.fired = True
            raise StorageUnavailableSignal("controlled-interruption")


V2 = {
    "plan_version": H_PLAN_V2,
    "thinking": "enabled",
    "generation_deadline_seconds": 120.0,
    "rate_limit_window_seconds": 10,
    "rate_limit_burst_threshold": 2,
}


@contextmanager
def runnable_batch(tmp_path, suffix, fault=None, count=1):
    """One generated, sealed disposable batch, run once under the optional fault."""

    value = make_disposable(
        tmp_path / ".local", count, batch_id="disposable-h-" + suffix, **V2
    )
    seal = HSeal(Path(value["root"]))
    seal.generate(
        seal.directory / "private/recipe.json",
        seal.directory / "private/seed.bin",
        seal.directory / "private/registry.json",
    )
    seal.seal()
    base = os.environ["TEST_POSTGRES_DSN"]
    name = "v107h_" + seal.plan().batch_id.replace("-", "_")
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(base).set(database=name).render_as_string(hide_password=False)
    try:
        setup(seal.directory, url)
        asyncio.run(
            run(
                seal.directory,
                transport_override=mock_transport("material"),
                fault=fault,
            )
        )
        yield seal
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def observations_of(seal):
    return {
        path.name: json.loads(path.read_bytes())
        for path in sorted((seal.directory / "observations").glob("*.json"))
    }


def case_of(seal, case_id):
    return next(item for item in seal.runner_cases() if item.case_id == case_id)


def resign(body):
    """Recompute the outer observation commit checksum after a mutation."""

    changed = {k: v for k, v in body.items() if k != "commit_sha256"}
    changed["commit_sha256"] = payload_sha256(changed)
    return changed


def damaged(body, mutate):
    changed = copy.deepcopy(body)
    mutate(changed)
    return resign(changed)


def drop_checkpoints(item):
    item["binding"]["checkpoints"] = []


def reparent_events(item):
    for event in item["binding"]["events"]:
        event["run_id"] = "run-00000000000000000000000000000000"
        event["trace_id"] = "trace-0000000000000000000000000000000"


def reparent_generations(item):
    for generation in item["binding"]["generations"]:
        generation["run_id"] = "run-00000000000000000000000000000000"


def reparent_attempt(item):
    item["binding"]["attempts"][0]["run_id"] = "run-00000000000000000000000000000000"


def drop_attempts_and_receipts(item):
    """Erase every attempt that was sent with its receipt, and sync both projections."""

    item["binding"]["attempts"] = []
    item["binding"]["events"] = [
        event
        for event in item["binding"]["events"]
        if event.get("step") != "model_attempt"
    ]
    item["attempts"] = []


def drop_retrieval(item):
    item["retrieval"] = {}


DAMAGES = {
    # label: (damage, the failure it must raise, whether the record still yields its
    # measured outcome — a damaged provenance commandeers the record, a damaged source
    # reading keeps the outcome visible while the batch stays incomplete)
    "delete-checkpoints": (
        drop_checkpoints,
        "product_failed_adoption_without_checkpoint",
        False,
    ),
    "event-parent-mismatch": (
        reparent_events,
        "product_failed_event_parent",
        False,
    ),
    "event-trace-mismatch": (
        reparent_events,
        "product_failed_event_trace",
        False,
    ),
    "generation-run-mismatch": (
        reparent_generations,
        "product_failed_generation_parent",
        False,
    ),
    "attempt-run-mismatch": (
        reparent_attempt,
        "product_failed_attempt_parent",
        False,
    ),
    "delete-attempts-and-receipts": (
        drop_attempts_and_receipts,
        "product_failed_checkpoint_attempt",
        False,
    ),
    "delete-retrieval": (
        drop_retrieval,
        "product_actual_model_sources:policy",
        True,
    ),
}


def test_legal_failed_run_keeps_its_measured_outcome(tmp_path):
    """The undisputed failure record is an outcome, and it names the branch's own code."""

    with runnable_batch(tmp_path, "failed-run", SummaryFailure()) as seal:
        name, body = next(
            (name, body)
            for name, body in observations_of(seal).items()
            if body["arm"] == "multi_agent" and body["row"]["status"] == "failed"
        )
        plan = seal.plan()
        assert observation_failures(plan, case_of(seal, body["case_id"]), body) == [
            "product_run_failed:review_execution_failed"
        ]
        proof = completion_proof(seal.directory)
        assert proof["complete"] is True
        assert proof["product_outcomes"] == [
            f"{name}:product_run_failed:review_execution_failed"
        ]
        # The legitimate record carries the adoption evidence the damages below erase.
        assert sorted(cp["worker"] for cp in body["binding"]["checkpoints"]) == [
            "contract",
            "policy",
        ]
        assert len(body["binding"]["attempts"]) == 2


@pytest.mark.parametrize("label", sorted(DAMAGES))
def test_damaged_failure_record_is_refused_by_name(tmp_path, label):
    """Each directed damage reaches the semantic layer and is refused there."""

    mutate, expected, keeps_outcome = DAMAGES[label]
    with runnable_batch(tmp_path, f"damage-{label}", SummaryFailure()) as seal:
        name, body = next(
            (name, body)
            for name, body in observations_of(seal).items()
            if body["arm"] == "multi_agent" and body["row"]["status"] == "failed"
        )
        plan = seal.plan()
        changed = damaged(body, mutate)
        failures = observation_failures(
            plan, case_of(seal, changed["case_id"]), changed
        )
        assert expected in failures, failures
        # The damaged record never reads as "just a product failure": the damage itself is
        # named, and a damaged provenance stops yielding the measured outcome entirely.
        assert [item for item in failures if not item.startswith("product_run_")], (
            failures
        )
        assert keeps_outcome == any(
            item.startswith("product_run_") for item in failures
        ), failures

        # The same damage reaches the completion proof through a copy of the batch: the
        # record is unreadable, not a measured product failure, and the batch is refused.
        copy_root = tmp_path / ".local" / ("copy-" + label)
        shutil.copytree(seal.directory, copy_root)
        path = copy_root / "observations" / name
        path.write_text(
            json.dumps(changed, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.chmod(path, 0o600)
        proof = completion_proof(copy_root)
        assert proof["complete"] is False
        assert f"{name}:{expected}" in proof["failures"], proof["failures"]
        if not keeps_outcome:
            assert proof["product_outcomes"] == []
        with pytest.raises(ValueError, match="h_run_incomplete"):
            h_cli(["complete", str(copy_root)])


def test_receipt_without_its_attempt_is_refused(tmp_path):
    """A receipt that proves a send whose ledger row is gone is an orphan, not evidence."""

    with runnable_batch(tmp_path, "orphan-receipt", SummaryFailure()) as seal:
        _name, body = next(
            (name, body)
            for name, body in observations_of(seal).items()
            if body["arm"] == "multi_agent" and body["row"]["status"] == "failed"
        )
        plan = seal.plan()

        def erase_without_receipts(item):
            item["binding"]["attempts"] = []
            item["attempts"] = []

        failures = observation_failures(
            plan,
            case_of(seal, body["case_id"]),
            damaged(body, erase_without_receipts),
        )
        assert "product_failed_receipt_without_attempt" in failures, failures


def test_retrieval_deletion_is_refused_without_a_pairing_witness(tmp_path):
    """Deleting the recorded sources fails on its own, not only through arm pairing."""

    with runnable_batch(tmp_path, "retrieval-orphan", SummaryFailure()) as seal:
        _name, body = next(
            (name, body)
            for name, body in observations_of(seal).items()
            if body["arm"] == "multi_agent" and body["row"]["status"] == "failed"
        )
        failures = observation_failures(
            seal.plan(), case_of(seal, body["case_id"]), damaged(body, drop_retrieval)
        )
        assert "retrieval_source_identity:policy:original" in failures, failures
        assert "retrieval_source_identity:contract:model_sources" in failures, failures
        assert "product_actual_model_sources:policy" in failures, failures


def test_interrupted_run_keeps_its_own_reading_and_adoption(tmp_path):
    """An interrupted Run stays terminal and measured, and its adopted branch stays proved."""

    with runnable_batch(
        tmp_path, "interrupted-provenance", ContractStorageOutage()
    ) as seal:
        plan = seal.plan()
        _name, body = next(
            (name, body)
            for name, body in observations_of(seal).items()
            if body["row"]["status"] == "interrupted"
        )
        assert observation_failures(plan, case_of(seal, body["case_id"]), body) == [
            "product_run_interrupted:storage_unavailable"
        ]
        # Deleting the adopted branch's checkpoint is refused here as well: the unfinished
        # branch is not an excuse for erasing the completed one.
        assert "product_failed_adoption_without_checkpoint" in observation_failures(
            plan, case_of(seal, body["case_id"]), damaged(body, drop_checkpoints)
        )
