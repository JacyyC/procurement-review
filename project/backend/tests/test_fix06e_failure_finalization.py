"""FIX-06E: a failure finalization obeys the same lock and run identity as a success.

R-06 (the D-review counterexample) was that ``run_gate.stop`` wrote ``failure.json`` and
``run.json`` unconditionally: an *older* run that failed could overwrite the state a
newer, already published success owns — even while a third process held the work-tree
lock.  The success path had taken the shared lock and re-checked the run identity since
FIX-05; the failure path did neither.

Everything here drives the real tool: the shipped gate source is copied into a synthetic
repository with stub governance scripts (``Sandbox`` from the FIX-01D suite, re-exported
by the FIX-05 suite), the real lifecycle, lock, records and CLI write paths are the ones
under test, and only the expensive external sections are replaced inside the copy.

The cases follow the FIX-06E verification matrix:

* V-E-01 — an older failure (with and without a third process holding the lock) leaves
  the newer state byte-identical and keeps only its own run-identity diagnostic, and the
  latest success still promotes afterwards; the opposite completion order, a newer
  running run and a genuine new failure are covered too;
* V-E-02 — the publish path's own failure is recorded without re-entering the lock it
  already holds, a hard kill leaves no stale lock and no forged state, and a diagnostic
  that cannot be written still reports the original failure with a non-zero exit.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_SOURCE = REPO_ROOT / "project" / "scripts" / "quality_gate.py"
FIX05_SUITE = REPO_ROOT / "project" / "backend" / "tests" / "test_fix05_quality_gate.py"

LOCK_HOLDER = '''"""Hold the work-tree lock until the release file appears (a third process)."""

import importlib.util
import sys
import time
from pathlib import Path


def load(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


gate = load(Path(sys.argv[1]), "holder_gate")
entered = Path(sys.argv[2])
release = Path(sys.argv[3])
with gate.gate_lock("third-process"):
    entered.touch()
    deadline = time.monotonic() + 90
    while not release.exists():
        if time.monotonic() > deadline:
            break
        time.sleep(0.01)
'''


def _load(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


FIX05 = _load(FIX05_SUITE, "fix06e_fix05_suite")
Harness = FIX05.Harness
Sandbox = FIX05.Sandbox


class FailureHarness(Harness):
    """The FIX-05 harness plus one process that simply holds the work-tree lock."""

    def __init__(self, sandbox: Sandbox) -> None:
        super().__init__(sandbox)
        self.holder = self.scratch / "lock-holder.py"
        self.holder.write_text(LOCK_HOLDER, encoding="utf-8")

    def hold_lock(
        self, name: str, *, purpose: str = "update"
    ) -> tuple[subprocess.Popen, Path]:
        entered = self.scratch / f"{name}-locked"
        release = self.scratch / f"{name}-unlock"
        process = subprocess.Popen(
            [
                sys.executable,
                str(self.holder),
                str(self.sandbox.gate_path),
                str(entered),
                str(release),
                purpose,
            ],
            cwd=self.sandbox.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.wait(entered)
        assert self.lock_visibility() is False, "the third process must hold the lock"
        return process, release

    def parked_failing_run(self, name: str) -> tuple[subprocess.Popen, Path]:
        """A real run process stopped inside a section, set up to fail when released."""

        entered = self.scratch / f"{name}-entered"
        release = self.scratch / f"{name}-release"
        process = self.child("run", entered=entered, release=release, failure="backend")
        self.wait(entered)
        return process, release

    def diagnostic(self, run_id: str) -> Path:
        """The run-identity-isolated diagnostic an older run is allowed to keep."""

        return self.sandbox.state / "diagnostics" / f"{run_id}.failure.json"

    def read_diagnostic(self, run_id: str) -> dict:
        return json.loads(self.diagnostic(run_id).read_text(encoding="utf-8"))


@pytest.fixture()
def harness(tmp_path: Path) -> FailureHarness:
    return FailureHarness(Sandbox(tmp_path / "repo"))


# ================================================= V-E-01 the original R-06 window


def test_an_older_failure_cannot_overwrite_a_newer_success(
    harness: FailureHarness,
) -> None:
    """R-06: the failure finalization used to ignore both the lock and the run identity."""

    older, older_release = harness.parked_failing_run("older")
    older_id = harness.read_state("run.json")["run_id"]

    harness.install_workload()
    assert harness.module.run_gate() == 0
    newer = harness.read_state("run.json")
    assert newer["status"] == "success" and newer["run_id"] != older_id
    success_before = harness.state("success.json").read_bytes()
    run_before = harness.state("run.json").read_bytes()

    older_release.touch()
    code, stdout, stderr = harness.finish(older)
    assert code != 0

    # The newer state keeps its exact place: nothing shared was rewritten.
    assert harness.state("run.json").read_bytes() == run_before
    assert harness.state("success.json").read_bytes() == success_before
    assert not harness.state("failure.json").exists()
    # The older run keeps its own, run-identity-isolated diagnostic.
    assert harness.diagnostic(older_id).is_file()
    diagnostic = harness.read_diagnostic(older_id)
    assert diagnostic["run_id"] == older_id
    assert diagnostic["status"] == "failed"
    assert diagnostic["section"] == "backend"
    assert diagnostic["message"] == "injected external test failure"
    assert "superseded" in (stdout + stderr) or "diagnostic" in (stdout + stderr)


def test_an_older_failure_cannot_write_while_a_third_process_holds_the_lock(
    harness: FailureHarness,
) -> None:
    """The same window with the work tree locked by somebody else: busy, never a write."""

    older, older_release = harness.parked_failing_run("older")
    older_id = harness.read_state("run.json")["run_id"]
    holder, unlock = harness.hold_lock("third")
    try:
        run_before = harness.state("run.json").read_bytes()
        older_release.touch()
        code, stdout, stderr = harness.finish(older)
        assert code != 0
        assert harness.state("run.json").read_bytes() == run_before
        assert not harness.state("failure.json").exists()
        assert harness.lock_visibility() is False, (
            "the third process still owns the lock"
        )
        assert harness.diagnostic(older_id).is_file()
        assert "lock" in (stdout + stderr), stdout + stderr
    finally:
        unlock.touch()
        harness.finish(holder)
    assert harness.lock_visibility() is True


def test_a_gate_writer_holding_the_lock_keeps_a_failing_run_out_of_shared_state(
    harness: FailureHarness,
) -> None:
    """``update``/``recover`` and the failure finalization share one lock (V-E-01).

    The same lock serializes admission, so a real ``--update-baseline`` and a run
    mid-sections cannot overlap through the normal entry points: the update is refused
    while the run is the latest, and the run cannot start while the update holds the
    lock.  The residual window — a third process that owns the lock while a run that
    already started fails — is what the R-06 reading produced, so this holds the very
    same lock under the ``update`` purpose and releases the failure into it.
    """

    older, older_release = harness.parked_failing_run("older")
    older_id = harness.read_state("run.json")["run_id"]
    harness.install_workload()
    assert harness.module.run_gate() == 0
    # The normal entry point is refused while the run is the latest one.
    running, running_release = harness.parked_failing_run("running")
    harness.refuse("--update-baseline", expect="'running', not 'success'")

    holder, unlock = harness.hold_lock("writer", purpose="update")
    try:
        run_before = harness.state("run.json").read_bytes()
        older_release.touch()
        code, stdout, stderr = harness.finish(older)
        assert code != 0
        assert harness.state("run.json").read_bytes() == run_before
        assert not harness.state("failure.json").exists()
        assert harness.diagnostic(older_id).is_file()
        assert "lock" in (stdout + stderr), stdout + stderr
    finally:
        unlock.touch()
        harness.finish(holder)
        running_release.touch()
        harness.finish(running)
    assert harness.lock_visibility() is True


def test_the_latest_success_still_promotes_after_an_older_failure(
    harness: FailureHarness,
) -> None:
    """The superseded failure is not a new round: the measured success still moves it."""

    older, older_release = harness.parked_failing_run("older")
    harness.install_workload()
    assert harness.module.run_gate() == 0
    older_release.touch()
    assert harness.finish(older)[0] != 0

    harness.promote("--update-baseline")
    updated = json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))
    assert updated["git_head"] == harness.sandbox.head


def test_an_older_failure_cannot_overwrite_a_newer_running_run(
    harness: FailureHarness,
) -> None:
    """Neither a newer success nor a newer running state may be replaced by an old failure."""

    older, older_release = harness.parked_failing_run("older")
    newer, newer_release = harness.parked_failing_run("newer")
    newer_id = harness.read_state("run.json")["run_id"]

    older_release.touch()
    assert harness.finish(older)[0] != 0
    running = harness.read_state("run.json")
    assert running["run_id"] == newer_id and running["status"] == "running"
    assert not harness.state("failure.json").exists()

    newer_release.touch()
    assert harness.finish(newer)[0] != 0
    failed = harness.read_state("run.json")
    assert failed["run_id"] == newer_id and failed["status"] == "failed"
    assert harness.read_state("failure.json")["run_id"] == newer_id


def test_a_failure_that_lands_first_does_not_survive_a_later_success(
    harness: FailureHarness,
) -> None:
    """The opposite completion order: the old failure is legitimate, then loses."""

    older, older_release = harness.parked_failing_run("older")
    older_id = harness.read_state("run.json")["run_id"]
    older_release.touch()
    assert harness.finish(older)[0] != 0
    landed = harness.read_state("run.json")
    assert landed["run_id"] == older_id and landed["status"] == "failed"
    assert harness.read_state("failure.json")["run_id"] == older_id

    harness.install_workload()
    assert harness.module.run_gate() == 0
    newer = harness.read_state("run.json")
    assert newer["status"] == "success" and newer["run_id"] != older_id
    assert (
        json.loads(harness.state("success.json").read_text(encoding="utf-8"))["run_id"]
        == newer["run_id"]
    )
    harness.promote("--update-baseline")


def test_the_latest_failure_records_itself_and_blocks_promotion(
    harness: FailureHarness,
) -> None:
    """The positive control: a real, current failure still publishes its own record."""

    harness.install_workload(failure="backend")
    assert harness.module.run_gate() == 1
    run = harness.read_state("run.json")
    assert run["status"] == "failed" and run["failed_section"] == "backend"
    failure = harness.read_state("failure.json")
    assert failure["run_id"] == run["run_id"]
    assert not harness.state("success.json").exists()
    harness.refuse("--update-baseline", expect="not 'success'")
    assert not harness.diagnostic(run["run_id"]).exists()


def test_a_newer_running_run_still_blocks_promotion(harness: FailureHarness) -> None:
    running, release = harness.parked_failing_run("running")
    try:
        harness.refuse("--update-baseline", expect="'running', not 'success'")
    finally:
        release.touch()
        harness.finish(running)


# ============================================ V-E-02 the finalization's own edge cases


def test_the_publish_paths_own_failure_is_recorded_without_reentering_the_lock(
    harness: FailureHarness,
) -> None:
    """The inputs-changed failure happens inside the publish path; it must still land."""

    module = harness.module
    note = harness.sandbox.root / "project/frontend/src/App.tsx"

    def frontend(measured: dict) -> None:
        measured["frontend"] = {"line_percent": 96.0, "branch_percent": 90.0}
        # Editing a measurement input after the sections makes the final fingerprint
        # differ from the one the run started with.
        note.write_text("export const app = 2;\n", encoding="utf-8")

    module.SECTIONS = (
        ("gov", module.section_gov),
        ("ruff", lambda: None),
        (
            "backend",
            lambda measured: measured.update(
                backend={"line_percent": 87.5, "branch_percent": 72.5}
            ),
        ),
        ("frontend", frontend),
        ("ratchet", module.section_ratchet),
    )
    assert module.run_gate() == 1
    run = harness.read_state("run.json")
    assert run["status"] == "failed"
    assert run["failed_section"] == "inputs"
    assert harness.read_state("failure.json")["run_id"] == run["run_id"]
    assert not harness.state("success.json").exists()
    assert harness.lock_visibility() is True, (
        "the run must not deadlock on its own lock"
    )


def test_a_hard_kill_of_a_failing_run_leaves_no_lock_and_no_forged_state(
    harness: FailureHarness,
) -> None:
    killed, _ = harness.parked_failing_run("killed")
    run_before = harness.state("run.json").read_bytes()
    killed.kill()
    code, _, _ = harness.finish(killed)
    assert code != 0
    assert harness.lock_visibility() is True
    assert harness.state("run.json").read_bytes() == run_before
    assert not harness.state("failure.json").exists()

    harness.install_workload()
    assert harness.module.run_gate() == 0
    assert harness.read_state("run.json")["status"] == "success"


def test_a_sigterm_of_a_failing_run_leaves_no_lock_and_no_forged_state(
    harness: FailureHarness,
) -> None:
    import signal

    signalled, _ = harness.parked_failing_run("signalled")
    run_before = harness.state("run.json").read_bytes()
    signalled.send_signal(signal.SIGTERM)
    code, _, _ = harness.finish(signalled)
    assert code != 0
    assert harness.lock_visibility() is True
    assert harness.state("run.json").read_bytes() == run_before
    assert not harness.state("failure.json").exists()


def test_an_unwritable_diagnostic_still_reports_the_original_failure(
    harness: FailureHarness, capsys: pytest.CaptureFixture
) -> None:
    """An auxiliary write failure may not mask the reason, forge success or traceback."""

    failure_path = harness.state("failure.json")
    failure_path.mkdir(parents=True, exist_ok=True)
    (failure_path / "occupied").write_text("gone\n", encoding="utf-8")
    harness.install_workload(failure="backend")
    assert harness.module.run_gate() == 1
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Traceback" not in combined, combined
    assert "backend" in combined, combined
    assert "injected external test failure" in combined, combined
    assert not harness.state("success.json").exists()
    assert (
        json.loads(harness.state("run.json").read_text(encoding="utf-8"))["status"]
        != "success"
    )


def test_an_unexpected_section_error_is_finalized_as_a_failure(
    harness: FailureHarness,
) -> None:
    """An escaping exception is a failure of this run, not a run left "running" forever."""

    module = harness.module

    def exploding(measured: dict) -> None:
        raise RuntimeError("the backend section exploded")

    module.SECTIONS = (
        ("gov", module.section_gov),
        ("ruff", lambda: None),
        ("backend", exploding),
        (
            "frontend",
            lambda measured: measured.update(
                frontend={"line_percent": 96.0, "branch_percent": 90.0}
            ),
        ),
        ("ratchet", module.section_ratchet),
    )
    assert module.run_gate() == 1
    run = harness.read_state("run.json")
    assert run["status"] == "failed"
    assert run["failed_section"] == "backend"
    failure = harness.read_state("failure.json")
    assert failure["run_id"] == run["run_id"]
    assert "the backend section exploded" in failure["message"]
    assert not harness.state("success.json").exists()
    harness.refuse("--update-baseline", expect="not 'success'")
    assert harness.lock_visibility() is True


def test_an_unexpected_publish_error_is_finalized_as_a_failure(
    harness: FailureHarness,
) -> None:
    """A publish-step exception must not leave the run silently "running" either."""

    module = harness.module
    real = module.build_manifest
    calls = {"count": 0}

    def flaky_manifest() -> dict:
        calls["count"] += 1
        if calls["count"] >= 2:
            raise RuntimeError("the manifest build exploded")
        return real()

    module.build_manifest = flaky_manifest
    harness.install_workload()
    assert module.run_gate() == 1
    run = harness.read_state("run.json")
    assert run["status"] == "failed"
    assert run["failed_section"] == "publish"
    assert harness.read_state("failure.json")["run_id"] == run["run_id"]
    assert not harness.state("success.json").exists()
    assert harness.lock_visibility() is True
