"""Freeze H's actual import closure beside the product and semantic consumers."""

from pathlib import Path

from .gold_v3_contracts import payload_sha256
from .precise_candidate import candidate_fingerprint
from .precise_dependencies import package_closure, module_path
from .precise_run_scope import implementation_fingerprint
from .eval_h_storage import sha

H_ROOTS = (
    "eval_h_cli.py",
    "eval_h_seal.py",
    "eval_h_runner.py",
    "eval_h_score.py",
    "eval_h_audit.py",
)
CONSUMER_ROOTS = ("eval_h_score.py", "eval_h_audit.py", "eval_h_consumption.py")


def _digests(roots):
    return {name: sha(module_path(name)) for name in package_closure(roots)}


def identity() -> dict:
    base = candidate_fingerprint()
    dependencies = _digests(H_ROOTS)
    consumers = _digests(CONSUMER_ROOTS)
    script = Path(__file__).resolve().parents[3] / "scripts/comparison_report.py"
    dependencies["scripts/comparison_report.py"] = sha(script)
    consumers["scripts/comparison_report.py"] = sha(script)
    result = dict(
        schema_version="v1-07h-candidate.v1",
        base=base,
        h_dependencies=dependencies,
        consumer_dependencies=consumers,
        product_sha256=implementation_fingerprint(),
        consumer_sha256=payload_sha256(consumers),
    )
    result["candidate_sha256"] = payload_sha256(result)
    return result


def require_identity(plan) -> dict:
    current = identity()
    for field in ("candidate_sha256", "consumer_sha256", "product_sha256"):
        if getattr(plan, field) != current[field]:
            raise ValueError("h_frozen_identity_changed:" + field)
    return current
