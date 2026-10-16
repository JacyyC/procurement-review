"""FIX-06E (2026-09-17): the backend section's cap is a hang detector, not a verdict.

The backend section runs the entire suite under ``coverage run --branch`` inside one
``subprocess.run(timeout=...)`` call.  That cap used to be 3600s.  On 2026-09-16 the same
unmodified tree measured 1.8x-4.9x slower than its quiet-host baseline because other
workspaces were saturating the host, and the only window that was reached at all was cut
off at 3607.2s mid-suite: a *working* run was reported as a failure, and the only thing
distinguishing it from a real test failure was the ``TimeoutExpired`` text.

The property these cases pin down is the one that matters on a shared host: being slow
must never by itself be the verdict, while a genuinely hung suite must still end the run.
The counterexample is behavioural, not a comparison of constants -- the section is driven
with a stubbed subprocess layer whose suite call reports a duration, and the timeout
contract of ``subprocess.run`` is emulated faithfully (raise ``TimeoutExpired`` when the
suite outlasts the cap it was handed).

Coverage of the FIX-06E verification matrix (V-E-02):

* the cap a section hands to its child is the documented, finite one, and the shipped
  value clears the recorded quiet-host run with room to spare;
* the shape that failed on 2026-09-16 (7200s of real work, 4.26x the quiet-host run) now
  completes, while the superseded 3600s cap still cuts that same run off;
* a suite that outlasts the cap still ends the section, so the raised cap did not become
  "no cap".
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_SOURCE = REPO_ROOT / "project" / "scripts" / "quality_gate.py"

#: The recorded quiet-host duration of the backend section, and the run it came from.
QUIET_HOST_SECONDS = 1689.1
QUIET_HOST_RUN = "20260915T133535Z-1edd2ca5"
#: The cap that was proven insufficient on 2026-09-16 (see the module docstring).
SUPERSEDED_CAP_SECONDS = 3600
#: What the cut-off run had already spent when the superseded cap ended it (3607.2s),
#: rounded down to the 7200s a 4.26x-slower host needs for the whole suite.
CUT_OFF_SECTION_SECONDS = 7200.0


@pytest.fixture(name="gate")
def gate_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The shipped gate source, loaded from disk with its artifact directory redirected."""

    specification = importlib.util.spec_from_file_location(
        "gate_section_budget", GATE_SOURCE
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    monkeypatch.setattr(module, "GATE_DIR", tmp_path / "quality-gate")
    return module


class BackendStub:
    """Answer every subprocess the backend section starts.

    ``suite_seconds`` is what the coverage/pytest call reports as its duration.  The stub
    reproduces ``subprocess.run``'s timeout contract exactly: that call raises
    ``TimeoutExpired`` when the duration is longer than the cap it was handed, and no
    wall-clock time is spent doing it.
    """

    def __init__(self, gate: ModuleType, *, suite_seconds: float) -> None:
        self.gate = gate
        self.suite_seconds = suite_seconds
        self.suite_timeout: int | None = None
        self.suite_argv: list[str] | None = None
        self.calls: list[list[str]] = []

    def __call__(self, argv, *, cwd, env=None, timeout=3600):
        self.calls.append(list(argv))
        # The coverage subcommand, not any "run" that ``uv run`` happens to contribute.
        command = None
        if "coverage" in argv:
            index = argv.index("coverage")
            if index + 1 < len(argv):
                command = argv[index + 1]
        if command == "run":
            self.suite_argv = list(argv)
            self.suite_timeout = timeout
            if self.suite_seconds > timeout:
                raise subprocess.TimeoutExpired(argv, timeout)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if command == "json":
            target = Path(argv[argv.index("-o") + 1])
            target.write_text(
                json.dumps(
                    {
                        "totals": {
                            "covered_lines": 900,
                            "missing_lines": 100,
                            "num_branches": 200,
                            "covered_branches": 150,
                        }
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "", "")


@pytest.fixture(name="stub_factory")
def stub_factory_fixture(gate: ModuleType, monkeypatch: pytest.MonkeyPatch):
    def build(*, suite_seconds: float) -> BackendStub:
        stub = BackendStub(gate, suite_seconds=suite_seconds)
        monkeypatch.setattr(gate, "run", stub)
        return stub

    return build


def test_the_cap_clears_the_recorded_quiet_host_run_with_room(
    gate, stub_factory
) -> None:
    """A cap barely above the quiet-host time is what made a shared host unmeasurable."""

    stub = stub_factory(suite_seconds=0.0)
    gate.section_backend({})

    assert stub.suite_timeout is not None
    assert stub.suite_timeout > QUIET_HOST_SECONDS * 5, (
        f"the backend cap {stub.suite_timeout}s leaves less than 5x room over the recorded "
        f"quiet-host run {QUIET_HOST_SECONDS}s ({QUIET_HOST_RUN})"
    )


def test_a_slow_but_working_suite_is_no_longer_cut_off(gate, stub_factory) -> None:
    """The 2026-09-16 shape: 7200s of real work on a host running 4.26x slower."""

    stub = stub_factory(suite_seconds=CUT_OFF_SECTION_SECONDS)
    measured: dict = {}
    gate.section_backend(measured)

    assert stub.suite_timeout is not None
    assert stub.suite_timeout > CUT_OFF_SECTION_SECONDS
    assert measured["backend"] == {"line_percent": 90.0, "branch_percent": 75.0}


def test_the_superseded_cap_still_cuts_that_same_run_off(
    gate, stub_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original defect, kept as the counterexample the raised cap answers.

    On the pre-change source the section uses the literal 3600s, so setting the module
    attribute has no effect there and this case reproduces the cut-off exactly; after the
    change the attribute is what the section reads, and the same 3600s is restored here on
    purpose.
    """

    monkeypatch.setattr(
        gate, "BACKEND_SECTION_TIMEOUT_SECONDS", SUPERSEDED_CAP_SECONDS, raising=False
    )
    stub = stub_factory(suite_seconds=CUT_OFF_SECTION_SECONDS)
    with pytest.raises(subprocess.TimeoutExpired):
        gate.section_backend({})
    assert stub.suite_timeout == SUPERSEDED_CAP_SECONDS or stub.suite_timeout == 3600


def test_a_suite_that_outlasts_the_cap_still_ends_the_section(
    gate, stub_factory
) -> None:
    """The raised cap is still a cap: a suite that never ends does not run forever."""

    stub = stub_factory(suite_seconds=10**9)
    with pytest.raises(subprocess.TimeoutExpired):
        gate.section_backend({})
    assert stub.suite_timeout is not None


def test_the_cap_is_finite_and_bounded(gate, stub_factory) -> None:
    """A hang detector has to be a number, and a run cannot hold the work tree forever."""

    stub = stub_factory(suite_seconds=0.0)
    gate.section_backend({})

    assert isinstance(stub.suite_timeout, int)
    assert 0 < stub.suite_timeout <= 12 * 3600
