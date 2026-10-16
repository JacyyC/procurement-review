"""The authoritative dependency inventory of the precise candidate.

Two consumers need to know what the candidate *is*: the candidate fingerprint (the fixed
input a G verdict or an H comparison is about) and the Run execution snapshot (the build
one Run is frozen against, so a resume under a different build stops).  Both derive from
this one inventory instead of keeping separate hand-written lists that drift apart.

The inventory is computed, not typed: starting from the entry points below, the package's
own modules are followed through their imports, so a module split out of another one, or a
newly imported dependency, is inside the boundary without editing a list.  Package
internals never appear as bare names — the closure is over ``procurement_review`` modules
only — and a module that cannot be found is an error, never a silently absent digest.

The boundary is deliberately wider than any single request path:

* ``PRODUCT_ROOTS`` is what executes one Run: admission, transport, retrieval, MCP reads,
  grounding, the deterministic rules, persistence, the guard and the metering.
* ``DIAGNOSTIC_ROOTS`` is what produces the pre-registered DEV batch and judges it: the
  paired diagnostic, the attempt ledger, the independent scorer, the report and the
  audit consumers.
* ``FRONTEND_ROOTS`` is the workbench's own entry point; its modules are followed through
  their imports as well, so the pages that read a Run are part of the frozen candidate.

Data and configuration are inventoried too: the resources the pinned dataset versions
resolve to, and the lock files that pin the interpreter-visible dependency set.  A version
*name* is not a dependency — the digest is taken over the bytes.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Mapping

PACKAGE = Path(__file__).resolve().parent
BACKEND = PACKAGE.parents[1]
REPOSITORY = BACKEND.parents[1]
FRONTEND = REPOSITORY / "project" / "frontend"

#: The modules one product Run executes.  Everything they import transitively is inside
#: the execution boundary; a change there is a new build whatever the version names say.
PRODUCT_ROOTS: tuple[str, ...] = (
    "precise_run_runner.py",
    "precise_run_scope.py",
    "precise_run_transport.py",
    "precise_run_summary.py",
    "precise_run_state.py",
    "precise_branch_records.py",
    "precise_run_contracts.py",
    "execution_profile.py",
    "review_service.py",
    # The API the product is driven through is part of what one Run executes: a change in
    # a route, a request contract or the error envelope is a different product build.
    "api.py",
    "api_wiring.py",
    "api_routers",
)

#: The modules the pre-registered DEV batch, its scoring, its report and its audit use.
DIAGNOSTIC_ROOTS: tuple[str, ...] = (
    "precise_dev_paired_run.py",
    "precise_dev_paired.py",
    "precise_dev_paired_wire.py",
    "precise_dev_paired_measure.py",
    "precise_dev_paired_metering.py",
    "eval_g_score.py",
    "eval_g_report.py",
    "eval_g_comparison.py",
    "gold_v37_conditional.py",
    "gold_v38_dev_quality.py",
    "precise_candidate.py",
    "precise_dev_sweep.py",
    "precise_dev_score.py",
)

#: The workbench entry point the frontend closure starts from.
FRONTEND_ROOTS: tuple[str, ...] = ("src/main.tsx",)

#: The dataset and truth versions this candidate pins beside its configuration.
CANDIDATE_VERSIONS: tuple[str, ...] = (
    "gold-set.v3.7",
    "v1-07-truth.v1.7",
)

#: The resource files the pinned data resolves to, relative to the backend root.
RESOURCE_FILES: tuple[str, ...] = (
    "src/procurement_review/resources/gold_set/v3.1/cases.json",
    "src/procurement_review/resources/gold_set/v3.1/manifest.json",
    "src/procurement_review/resources/gold_set/v3.1/truth.json",
    "src/procurement_review/resources/public_sources",
    "src/procurement_review/resources/retrieval",
    "src/procurement_review/resources/rules",
    "src/procurement_review/resources/security",
    "src/procurement_review/resources/workers",
    "src/procurement_review/resources/synthetic",
)

#: The lock and build configuration that pin the dependency set of both halves.
LOCK_FILES: tuple[str, ...] = (
    "project/backend/uv.lock",
    "project/backend/pyproject.toml",
    "project/frontend/package-lock.json",
    "project/frontend/package.json",
    "project/frontend/tsconfig.app.json",
    "project/frontend/vite.config.ts",
)

#: Which responsibility each module carries.  It exists so a report can say *what* moved
#: (a rule, the grounding, the guard) instead of only *that* something moved, and so a
#: newly imported module has to be assigned deliberately: the coverage test fails while an
#: inventory module has no layer.
RESPONSIBILITY_LAYERS: Mapping[str, tuple[str, ...]] = {
    "model_visible_schema_and_prompt": (
        "business_v31_documents.py",
        "business_v3_documents.py",
        "model_wire_v2.py",
        "precise_answer_v3.py",
        "precise_run_transport.py",
        "precise_run_summary.py",
        "chat_transport_v2.py",
        "chat_transport_v3.py",
        "worker_contracts.py",
    ),
    "rules_and_decision": (
        "business_v3_rules.py",
        "business_v3_conditional.py",
        "business_v3_aggregate.py",
        "business_v3_types.py",
        "business_v3_precise_workers.py",
        "rules.py",
    ),
    "grounding_and_evidence": (
        "business_v31_identity.py",
        "business_v3_support.py",
        "business_v3_evidence.py",
        "business_v31_review.py",
        "business_v3_review.py",
        "business_v3_prose.py",
        "business_v3_text.py",
    ),
    "conflicts": ("business_v3_conflicts.py",),
    "security_and_guard": (
        "business_v3_security.py",
        "logging_setup.py",
        "security_contracts.py",
        "field_guard.py",
        "value_rules.py",
        "execution_guards.py",
        "service_errors.py",
    ),
    "retrieval_and_scope": (
        "retrieval.py",
        "scoped_runtime.py",
        "eval_g_retrieval.py",
        "eval_g_distinguishability.py",
        "ingestion.py",
        "public_corpus.py",
        "local_runtime.py",
        "case_catalog.py",
    ),
    "mcp_and_reads": (
        "mcp_server.py",
        "mcp_startup.py",
        "procurement_mcp_client.py",
        "procurement_store.py",
        "admitted_reads.py",
        "business_v3_procurement.py",
    ),
    "api_surface": (
        "api.py",
        "api_wiring.py",
        "api_support.py",
        "api_error_handlers.py",
        "api_routers",
        "api_routers.actions",
        "api_routers.cases",
        "api_routers.meta",
        "api_routers.reviews",
        "api_routers.session",
    ),
    "admission_and_execution": (
        "model_invocation.py",
        "result_admission.py",
        "checkpointing.py",
        "execution_profile.py",
        "precise_run_runner.py",
        "precise_run_scope.py",
        "precise_run_state.py",
        "precise_run_contracts.py",
        "precise_branch_records.py",
        "fault_hooks.py",
        "run_errors.py",
    ),
    "persistence_and_records": (
        "review_service.py",
        "review_repository.py",
        "review_store.py",
        "review_schema.py",
        "record_contracts.py",
        "record_guard.py",
        "record_inventory.py",
        "record_reads.py",
        "record_rows.py",
        "run_projection.py",
        "run_records.py",
        "run_recovery.py",
        "run_trace.py",
        "run_guard.py",
        "run_idempotency.py",
        "run_idempotency_namespace.py",
        "review_graph.py",
        "workers.py",
        "trace.py",
        "authorization.py",
        "audit.py",
        "audit_contracts.py",
        "run_audit.py",
        "contracts.py",
        "business_fact_contracts.py",
        "api_contracts.py",
        "api_support.py",
        "api_wiring.py",
        "api_error_handlers.py",
        "api.py",
    ),
    "generation_and_material": (
        "gold_v3.py",
        "gold_v3_oracle.py",
        "gold_v3_contracts.py",
        "gold_v31.py",
        "gold_v31_contracts.py",
        "gold_v31_oracle.py",
        "gold_v32_contracts.py",
        "gold_v33_contracts.py",
        "gold_v34_contracts.py",
        "gold_v35_contracts.py",
        "gold_v36_contracts.py",
        "gold_v37_conditional.py",
        "synthetic.py",
        "eval_f_context.py",
        "security_fixture.py",
        "eval_v2_contracts.py",
        "manifest.py",
    ),
    "attempt_ledger": (
        "model_attempt_v2.py",
        "model_attempt_projection.py",
        "eval_attempt_ledger.py",
        "eval_attempt_schema.py",
        "eval_attempt_budget.py",
        "eval_attempt_export.py",
        "eval_attempt_manifest.py",
        "eval_f_transport.py",
        "eval_f_contracts.py",
        "eval_f_run.py",
        "eval_f_budget.py",
        "eval_f_registration.py",
        "eval_f_stop.py",
        "model_adapters.py",
        "chat_adapter.py",
    ),
    "dev_quality_set_and_sweep": (
        "gold_v38_dev_quality.py",
        "precise_dev_sweep.py",
        "precise_dev_score.py",
    ),
    "scoring_and_metrics": (
        "eval_f_score.py",
        "eval_f_semantics.py",
        "eval_f_obligations.py",
        "eval_f_value_diagnostics.py",
        "eval_f_text.py",
        "eval_f_english.py",
        "eval_f_lineage.py",
        "eval_f_validation.py",
        "eval_g_consumption.py",
        "eval_g_product_summary.py",
        "eval_g_transactions.py",
        "eval_g_cli.py",
        "eval_g_reuse.py",
        "eval_g_score.py",
        "eval_v2_metrics.py",
        "eval_v2_report.py",
    ),
    "reporting_and_comparison": (
        "eval_g_report.py",
        "eval_g_comparison.py",
        "eval_g_plan.py",
        "eval_g_projection.py",
        "eval_g_exclusion.py",
        "eval_g_contracts.py",
        "eval_rescore.py",
        "eval_score.py",
    ),
    "independent_review": (
        "eval_audit.py",
        "precise_candidate.py",
        "precise_dev_paired.py",
        "precise_dev_paired_run.py",
        "precise_dev_paired_wire.py",
        "precise_dev_paired_measure.py",
        "precise_dev_paired_metering.py",
        "attempt_metering.py",
        "blind_seal.py",
    ),
    "model_probes": (
        "model_probe_v2.py",
        "model_probe_run.py",
        "model_probe_budget.py",
        "eval_f_budget.py",
    ),
    "dependency_inventory": ("precise_dependencies.py",),
}


class MissingDependency(RuntimeError):
    """A required file of the inventory is not present in the checkout."""


def _module_dependencies(path: Path) -> set[str]:
    """The package modules one module imports, including imports inside functions.

    Relative imports resolve against the importing module's own package, so a module
    inside a sub-package (``api_routers/reviews.py``) names its siblings the way the
    interpreter does instead of appearing as a top-level module of the package.
    """

    own = list(path.relative_to(PACKAGE).with_suffix("").parts)
    package_parts = own[:-1]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: len(package_parts) - (node.level - 1)]
                if node.module:
                    parts = [*base, *node.module.split(".")]
                    found.add(".".join(parts))
                else:
                    for alias in node.names:
                        found.add(".".join([*base, alias.name.split(".")[0]]))
            elif node.module and node.module.split(".")[0] == PACKAGE.name:
                parts = node.module.split(".")[1:]
                if parts:
                    found.add(".".join(parts))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == PACKAGE.name and len(parts) > 1:
                    found.add(".".join(parts[1:]))
    return {name for name in found if name}


def module_path(name: str) -> Path | None:
    """The file one module or sub-package lives in, if it exists."""

    base = PACKAGE.joinpath(*name.split("."))
    module = base.with_suffix(".py")
    if module.is_file():
        return module
    package = base / "__init__.py"
    if package.is_file():
        return package
    return None


def _module_path(name: str) -> Path | None:
    return module_path(name)


def package_closure(roots: Iterable[str]) -> tuple[str, ...]:
    """Every package module and sub-package reachable from ``roots`` through its imports."""

    pending = [Path(name).stem if name.endswith(".py") else name for name in roots]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if not name or name in seen:
            continue
        path = _module_path(name)
        if path is None:
            raise MissingDependency(f"procurement_review.{name} is listed but absent")
        seen.add(name)
        if path.name == "__init__.py":
            # A sub-package ships every module it holds: the routers of one package are
            # part of the build even when the entry point imports them lazily.
            prefix = name if (PACKAGE / name).is_dir() is False else name
            prefix = "" if name == PACKAGE.name else f"{name}."
            pending.extend(
                sorted(
                    f"{prefix}{item.stem}"
                    for item in path.parent.glob("*.py")
                    if item.name != "__init__.py"
                )
            )
        pending.extend(sorted(_module_dependencies(path) - seen))
    return tuple(sorted(seen))


def _resolve_frontend(specifier: str, importer: Path) -> Path | None:
    if not specifier.startswith("."):
        return None
    base = (importer.parent / specifier).resolve()
    candidates = [base, *(base.with_suffix(suffix) for suffix in (".ts", ".tsx"))]
    candidates += [base / f"index{suffix}" for suffix in (".ts", ".tsx")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _frontend_dependencies(path: Path) -> set[Path]:
    if path.suffix == ".css":
        return set()
    body = path.read_text(encoding="utf-8")
    found: set[Path] = set()
    # ``import ... from 'x'``, ``export ... from 'x'`` and side-effect ``import 'x'``.
    for marker in ("from ", "import "):
        offset = 0
        while True:
            index = body.find(marker, offset)
            if index < 0:
                break
            offset = index + len(marker)
            quote = body[offset : offset + 1]
            if quote not in {"'", '"'}:
                continue
            end = body.find(quote, offset + 1)
            if end < 0:
                continue
            specifier = body[offset + 1 : end]
            resolved = _resolve_frontend(specifier, path)
            if resolved is not None:
                found.add(resolved)
    return found


def frontend_closure(roots: Iterable[str]) -> tuple[str, ...]:
    """Every frontend module the workbench entry points really load, test files aside."""

    pending = [(FRONTEND / name).resolve() for name in roots]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file():
            raise MissingDependency(f"frontend module {path} is listed but absent")
        if ".test." in path.name:
            # A test file is not part of the shipped candidate even when the app imports
            # a fixture beside it.
            continue
        seen.add(path)
        pending.extend(sorted(_frontend_dependencies(path) - seen))
    return tuple(sorted(str(path.relative_to(FRONTEND)) for path in seen))


def resource_files() -> tuple[str, ...]:
    """The pinned data files and directories, expanded to individual files."""

    found: list[str] = []
    for entry in RESOURCE_FILES:
        path = BACKEND / entry
        if path.is_dir():
            found.extend(
                str(item.relative_to(BACKEND))
                for item in sorted(path.rglob("*"))
                if item.is_file() and "__pycache__" not in item.parts
            )
        elif path.is_file():
            found.append(entry)
    return tuple(found)


def inventory() -> dict[str, tuple[str, ...]]:
    """The whole candidate boundary, section by section."""

    return {
        "product": package_closure(PRODUCT_ROOTS),
        "diagnostic": package_closure(DIAGNOSTIC_ROOTS),
        "frontend": frontend_closure(FRONTEND_ROOTS),
        "resources": resource_files(),
        "locks": LOCK_FILES,
    }


def require_present() -> dict[str, tuple[str, ...]]:
    """The inventory, refusing to return while any required file is missing."""

    sections = inventory()
    missing: list[str] = []
    for section, entries in sections.items():
        for entry in entries:
            if section == "frontend":
                path = FRONTEND / entry
            elif section == "resources":
                path = BACKEND / entry
            elif section == "locks":
                path = REPOSITORY / entry
            else:
                path = module_path(entry)
            if path is None or not path.is_file():
                missing.append(f"{section}:{entry}")
    if missing:
        raise MissingDependency("missing candidate dependencies: " + ", ".join(missing))
    return sections


def layer_of(module: str) -> str | None:
    """The responsibility layer one module belongs to, or ``None`` when unassigned."""

    stem = Path(module).stem
    for layer, names in RESPONSIBILITY_LAYERS.items():
        if stem in {Path(name).stem for name in names}:
            return layer
    return None


def unassigned_modules(modules: Iterable[str]) -> tuple[str, ...]:
    """The modules of a closure no layer claims — a deliberate assignment is required."""

    return tuple(name for name in modules if layer_of(name) is None)


__all__ = [
    "CANDIDATE_VERSIONS",
    "DIAGNOSTIC_ROOTS",
    "FRONTEND_ROOTS",
    "FRONTEND",
    "LOCK_FILES",
    "PACKAGE",
    "PRODUCT_ROOTS",
    "REPOSITORY",
    "RESOURCE_FILES",
    "RESPONSIBILITY_LAYERS",
    "MissingDependency",
    "frontend_closure",
    "module_path",
    "inventory",
    "layer_of",
    "package_closure",
    "require_present",
    "resource_files",
    "unassigned_modules",
]
