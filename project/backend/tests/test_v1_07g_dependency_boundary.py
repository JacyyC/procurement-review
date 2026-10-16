"""The candidate and Run-snapshot dependency boundaries, proved against real omissions.

Every case here is a negative control for a handover: it changes exactly one dependency the
audited implementation really executes, in a copy, and reads the digests again.  The files
chosen are the ones an earlier hand-kept list missed — the conflict pairing, the split-out
diagnostic modules, the split-out frontend validators and the aggregator/summary that
decide a Run's recommendation — so a boundary that silently stops covering them fails
here instead of passing a changed candidate under the old name.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from procurement_review import precise_candidate, precise_dependencies
from procurement_review.precise_candidate import candidate_fingerprint
from procurement_review.precise_dependencies import (
    DIAGNOSTIC_ROOTS,
    PRODUCT_ROOTS,
    MissingDependency,
    module_path,
    frontend_closure,
    inventory,
    package_closure,
    require_present,
    unassigned_modules,
)
from procurement_review.precise_run_scope import (
    SNAPSHOT_IMPLEMENTATION_FILES,
    implementation_fingerprint,
)

#: Files that were *not* covered before this repair, with the module that imports them.
AUDITED_OMISSIONS = (
    ("business_v3_conflicts.py", "business_v31_review"),
    ("precise_dev_paired_wire.py", "precise_dev_paired_run"),
    ("precise_dev_paired_measure.py", "precise_dev_paired_run"),
    ("precise_dev_paired_metering.py", "precise_dev_paired_run"),
    ("business_v3_aggregate.py", "business_v31_review"),
    ("precise_run_summary.py", "precise_run_runner"),
)


def _copy_subpackages(package: Path) -> None:
    """Copy every module that lives inside a sub-package, keeping the layout."""

    for entry in package_closure(PRODUCT_ROOTS) + package_closure(DIAGNOSTIC_ROOTS):
        if "." not in entry:
            continue
        path = module_path(entry)
        target = package.joinpath(*entry.split("."))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target.with_suffix(path.suffix))


def _copy_package(package: Path) -> None:
    """Copy the package's top-level modules and its sub-packages into ``package``."""

    for entry in package_closure(PRODUCT_ROOTS) + package_closure(DIAGNOSTIC_ROOTS):
        path = module_path(entry)
        target = package.joinpath(*entry.split("."))
        if path.name == "__init__.py":
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / "__init__.py")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target.with_suffix(".py"))


def test_the_inventory_refuses_a_missing_required_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A dependency that cannot be read is an error, not a candidate with fewer files."""

    package = tmp_path / "procurement_review"
    package.mkdir()
    _copy_package(package)
    # The copied tree is complete: it fingerprints.
    monkeypatch.setattr(precise_dependencies, "PACKAGE", package)
    assert package_closure(PRODUCT_ROOTS)
    (package / "business_v3_conflicts.py").unlink()
    with pytest.raises(MissingDependency) as refusal:
        package_closure(PRODUCT_ROOTS)
    assert "business_v3_conflicts" in str(refusal.value)


def test_every_module_of_each_closure_carries_a_responsibility(
    monkeypatch: pytest.MonkeyPatch,
):
    """A new import has to be assigned deliberately instead of silently entering a blob."""

    sections = inventory()
    assert unassigned_modules(sections["product"]) == ()
    assert unassigned_modules(sections["diagnostic"]) == ()
    # The audited omissions are inside the closures, not merely declared beside them.
    covered = set(sections["product"]) | set(sections["diagnostic"])
    for name, importer in AUDITED_OMISSIONS:
        assert Path(name).stem in covered, name
        assert Path(name).stem in package_closure((f"{importer}.py",)), name
    for name in ("src/preciseExecutionContract.ts", "src/summaryContract.ts"):
        assert name in sections["frontend"], name


def test_a_byte_change_in_each_audited_dependency_moves_the_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The omissions, re-checked on a copy: the digest of each one really is in the sum."""

    package = tmp_path / "procurement_review"
    frontend = tmp_path / "frontend" / "src"
    package.mkdir()
    frontend.mkdir(parents=True)
    sections = require_present()
    for entry in sections["product"] + sections["diagnostic"]:
        # Sub-package modules keep their layout: the closure follows them by dotted name.
        path = module_path(entry)
        base = package.joinpath(*entry.split("."))
        target = (
            base / "__init__.py"
            if path.name == "__init__.py"
            else base.with_suffix(".py")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for entry in sections["frontend"]:
        target = frontend / Path(entry).name
        shutil.copy2(precise_dependencies.FRONTEND / entry, target)
    monkeypatch.setattr(precise_dependencies, "PACKAGE", package)
    monkeypatch.setattr(precise_dependencies, "FRONTEND", tmp_path / "frontend")
    monkeypatch.setattr(precise_candidate, "PACKAGE", package)
    monkeypatch.setattr(precise_candidate, "FRONTEND", tmp_path / "frontend")

    def digests() -> dict:
        return precise_candidate.dependency_digests()

    def section_of(key: str) -> str:
        for section in ("product", "diagnostic"):
            if key in digests()[section]:
                return section
        raise AssertionError(f"{key} is not in any package section")

    before = digests()
    for name, _importer in AUDITED_OMISSIONS:
        target = package / name
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        after = digests()
        key = Path(name).stem
        section = section_of(key)
        assert after[section][key]["sha256"] != before[section][key]["sha256"], name
        target.write_bytes(original)
        assert digests()[section][key]["sha256"] == before[section][key]["sha256"]
    for name in ("src/preciseExecutionContract.ts", "src/summaryContract.ts"):
        target = frontend / Path(name).name
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        after = digests()
        assert (
            after["frontend"][name]["sha256"] != before["frontend"][name]["sha256"]
        ), name


def test_the_run_snapshot_covers_the_aggregator_and_the_summary_projection():
    """The two modules that decide a Run's own recommendation are inside its build."""

    assert "business_v3_aggregate" in SNAPSHOT_IMPLEMENTATION_FILES
    assert "precise_run_summary" in SNAPSHOT_IMPLEMENTATION_FILES
    assert "business_v3_conflicts" in SNAPSHOT_IMPLEMENTATION_FILES
    assert "business_v3_evidence" in SNAPSHOT_IMPLEMENTATION_FILES
    assert "api_routers.reviews" in SNAPSHOT_IMPLEMENTATION_FILES
    # The snapshot's own fingerprint is a digest over exactly that list.
    digests = {}
    for name in SNAPSHOT_IMPLEMENTATION_FILES:
        digests[name] = hashlib.sha256(module_path(name).read_bytes()).hexdigest()
    assert (
        hashlib.sha256(
            __import__("json").dumps(digests, sort_keys=True).encode()
        ).hexdigest()
        == implementation_fingerprint()
    )


def test_the_frontend_closure_is_derived_instead_of_listed():
    """A new module imported by the page enters the boundary without editing a list."""

    closure = frontend_closure(("src/main.tsx",))
    assert "src/App.tsx" in closure
    assert "src/ReviewViews.tsx" in closure
    assert "src/preciseExecutionContract.ts" in closure
    assert "src/summaryContract.ts" in closure
    assert "src/workbenchGuide.ts" in closure
    # Test modules are not part of the shipped candidate even when they sit beside it.
    assert not [name for name in closure if ".test." in name]


def test_the_candidate_fingerprint_reports_its_own_sections_and_layers():
    fingerprint = candidate_fingerprint()
    assert set(fingerprint["sections"]) == {
        "product",
        "diagnostic",
        "frontend",
        "resources",
        "locks",
    }
    assert set(fingerprint["file_counts"]) == set(fingerprint["sections"])
    layers = precise_candidate.responsibility_layers()
    assert "conflicts" in layers
    assert "business_v3_conflicts" in layers["conflicts"]
    assert "independent_review" in layers
    assert fingerprint["generated"]["workers"]
    assert len(fingerprint["candidate_sha256"]) == 64


def test_a_run_frozen_on_an_older_implementation_refuses_to_resume():
    """The snapshot's implementation digest is a resume condition, not a note.

    A Run created under an older build must stop on resume instead of finishing under the
    new one; the dimension that moved is named, so the refusal is attributable.
    """

    from types import SimpleNamespace

    from procurement_review.checkpointing import IncompatibleCheckpointError
    from procurement_review.precise_run_scope import (
        execution_snapshot,
        require_frozen_snapshot,
    )
    from procurement_review.precise_dev_paired import paired_configuration

    current = execution_snapshot(
        mode="real",
        config=paired_configuration(),
        registration=SimpleNamespace(
            rules_version="3.1.0",
            document_version="synthetic-gold-document.v3.1",
            dataset_version="synthetic-gold-procurement.v3",
            index_version="g-retrieval-index.v1",
        ),
        branch_timeout_seconds=30.0,
        max_attempts_per_branch=2,
        generation_deadline_seconds=90.0,
    )
    older = current.model_copy(update={"implementation_sha256": "0" * 64})
    run = SimpleNamespace(
        header=SimpleNamespace(
            isolation={"execution_snapshot": older.model_dump(mode="json")}
        ),
        generation=None,
    )
    with pytest.raises(IncompatibleCheckpointError) as refusal:
        require_frozen_snapshot(run, current)
    assert "implementation_digest" in str(refusal.value)
    # The same build resumes: the frozen snapshot is returned unchanged.
    matching = SimpleNamespace(
        header=SimpleNamespace(
            isolation={"execution_snapshot": current.model_dump(mode="json")}
        ),
        generation=None,
    )
    assert require_frozen_snapshot(matching, current) == current


def test_the_api_layer_is_inside_the_candidate_boundary():
    """The product is driven through its API, so a route is part of the same build."""

    sections = inventory()
    for name in (
        "api",
        "api_routers",
        "api_routers.reviews",
        "api_routers.actions",
        "logging_setup",
    ):
        assert name in sections["product"], name
    # A route module is a real file the digest covers, not a name in a list.
    assert module_path("api_routers.reviews").is_file()
    fingerprint = candidate_fingerprint()
    assert fingerprint["sections"]["product"]["api_routers.reviews"]["bytes"] > 0
