"""V1-08B: a batch-level stop stays effective across processes and calls (Spec P-05).

The repair closes the hole the audit reproduced: a batch stopped for dense 429s was
resumed by simply calling the runner again, and the fresh in-memory controller started
sending the queued work.  These tests pin the durable stop — its record, the second witness
in the batch's own committed observations, the fail-closed reading of an unverifiable
record — and, on the other side, that an ordinary Run interruption is still recovered
inside its own batch with its committed branch reused and its remaining budget spent.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from procurement_review.eval_h_batch_stop import stop_record_path
from procurement_review.eval_h_consumption import completion_proof
from procurement_review.eval_h_contracts import H_PLAN_V2
from procurement_review.eval_h_fixture import make_disposable
from procurement_review.eval_h_mock import mock_transport
from procurement_review.eval_h_runner import run, setup
from procurement_review.eval_h_seal import HSeal
from procurement_review.eval_h_storage import read

V2 = {
    "plan_version": H_PLAN_V2,
    "thinking": "enabled",
    "generation_deadline_seconds": 120.0,
    "rate_limit_window_seconds": 10,
    "rate_limit_burst_threshold": 2,
}


@contextmanager
def batch(tmp_path, suffix, count=1):
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
        yield seal, url
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def rate_limited_wire(sent):
    def respond(request):
        payload = json.loads(request.content)
        user = json.loads(
            next(
                message["content"]
                for message in payload["messages"]
                if message["role"] == "user"
            )
        )
        sent.append("joint" if "tasks" in user else user["worker"])
        return httpx.Response(429, json={"error": "slow down"})

    return httpx.MockTransport(respond)


def digest_tree(root):
    """The bytes of a directory tree, so an untouched batch can be compared exactly."""

    digests = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return digests


def stop_a_batch(seal):
    """Run the batch into its dense-429 stop and return the sends it actually made."""

    sent = []
    result = asyncio.run(
        run(seal.directory, transport_override=rate_limited_wire(sent))
    )
    assert result["stopped"] is True
    assert result["stop_reason"] == "rate_limited_dense"
    assert sent == ["policy", "policy"], sent
    return result


def test_stopped_batch_refuses_every_later_call(tmp_path):
    """A new controller, a repeated call and the resume entrance all keep the stop."""

    with batch(tmp_path, "stop-reentry") as (seal, _url):
        stop_a_batch(seal)
        # The stop is durable and bound to the registered plan.
        record = read(stop_record_path(seal.directory))
        assert record["reason"] == "rate_limited_dense"
        assert record["schema_version"] == "v1-08h-batch-stop.v1"
        assert record["trigger"] == {
            "window_seconds": 10,
            "threshold": 2,
            "observed": 2,
        }
        assert stop_record_path(seal.directory).stat().st_mode & 0o077 == 0
        before = digest_tree(seal.directory)

        for extra in ({}, {"resume": True}):
            sent = []
            with pytest.raises(
                ValueError, match="h_batch_stopped_explicit_continuation_required"
            ):
                asyncio.run(
                    run(
                        seal.directory,
                        transport_override=rate_limited_wire(sent),
                        **extra,
                    )
                )
            # Nothing left the process, and the whole batch is byte-identical: the
            # committed observation, the consumed attempts and the unsent facts stay.
            assert sent == [], sent
            assert digest_tree(seal.directory) == before


def test_cross_process_reentry_cannot_send(tmp_path):
    """A fresh process entering the same batch is refused before any send."""

    with batch(tmp_path, "stop-subprocess") as (seal, _url):
        stop_a_batch(seal)
        before = digest_tree(seal.directory)
        environment = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        }
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "procurement_review.eval_h_cli",
                "run",
                str(seal.directory),
                "--mock",
                "material",
            ],
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
        assert completed.returncode != 0, completed.stdout
        assert "h_batch_stopped_explicit_continuation_required" in completed.stderr
        assert digest_tree(seal.directory) == before


def test_a_lost_record_file_does_not_reopen_a_stopped_batch(tmp_path):
    """The committed observation is the second witness of the same stop."""

    with batch(tmp_path, "stop-lost-record") as (seal, _url):
        stop_a_batch(seal)
        os.unlink(stop_record_path(seal.directory))
        sent = []
        with pytest.raises(
            ValueError, match="h_batch_stopped_explicit_continuation_required"
        ):
            asyncio.run(run(seal.directory, transport_override=rate_limited_wire(sent)))
        assert sent == []


def test_unverifiable_stop_record_closes_sending(tmp_path):
    """A record that cannot be verified is not read as a batch that never stopped."""

    with batch(tmp_path, "stop-unreadable") as (seal, _url):
        stop_a_batch(seal)
        path = stop_record_path(seal.directory)
        body = read(path)
        body["plan_sha256"] = "0" * 64
        path.write_text(
            json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.chmod(path, 0o600)
        sent = []
        with pytest.raises(ValueError, match="h_batch_stop_record_unreadable"):
            asyncio.run(run(seal.directory, transport_override=rate_limited_wire(sent)))
        assert sent == []


def test_normal_interruption_recovery_still_reuses_committed_branches(tmp_path):
    """A batch-level stop is not what an interrupted Run is: its own batch recovers it."""

    from pydantic import SecretStr

    from procurement_review.checkpointing import StorageUnavailableSignal
    from procurement_review.eval_h_product import multi
    from procurement_review.eval_h_storage import read as read_json
    from procurement_review.precise_run_transport import PreciseRunChatTransport
    from procurement_review.review_store import ReviewStore

    class Once:
        enabled = True
        fired = False

        def barrier(self, site):
            if (
                site == "during_model_entry_authorization:precise.contract"
                and not self.fired
            ):
                self.fired = True
                raise StorageUnavailableSignal("h-disposable-injected-storage-outage")

    with batch(tmp_path, "recovery-positive") as (seal, url):
        state = read_json(seal.directory / "private/runtime.json")
        store = ReviewStore.from_url(url)
        requests = []
        transport = PreciseRunChatTransport(
            seal.plan().configuration.model_copy(update={"seed": 101}),
            SecretStr("h-offline-only"),
            catalog=None,
            transport=mock_transport("material", requests),
        )
        case = seal.runner_cases()[0]
        try:
            _row, binding = asyncio.run(
                multi(
                    store,
                    state["token"],
                    seal.plan(),
                    case,
                    101,
                    transport,
                    resume=True,
                    fault=Once(),
                )
            )
        finally:
            store.close()
        # The interruption was recovered inside the Run: the committed branch was reused
        # and the failed one spent its own remaining budget.
        assert binding["resume_count"] == 1
        assert binding["execution"]["metering"]["reused_branches"] == 1
        assert len(requests) == 2, len(requests)
        # A normal Run interruption is not a batch-level stop: no record was written.
        assert stop_record_path(seal.directory).exists() is False
        sends = []
        result = asyncio.run(
            run(
                seal.directory,
                transport_override=mock_transport("material", sends),
                resume=True,
            )
        )
        assert result["stopped"] is False
        assert result["committed"] == 4
        assert completion_proof(seal.directory)["complete"] is True
        # Nothing was re-sent for the logical tasks this Run already committed: of the
        # seed it already ran, only the untried joint task left the process.
        already = []
        for payload in sends:
            if payload.get("seed") != 101:
                continue
            user = json.loads(
                next(
                    message["content"]
                    for message in payload["messages"]
                    if message["role"] == "user"
                )
            )
            already.append("joint" if "tasks" in user else user["worker"])
        assert already == ["joint"], already
