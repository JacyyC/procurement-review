"""Explicit offline evidence reuse when only diagnostic ownership metadata moved."""

import ast
import hashlib
import json
from pathlib import Path


def inventory_semantics(text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        raise ValueError("product_reuse_inventory_syntax") from None
    owners = None
    kept = []
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "RESPONSIBILITY_LAYERS"
        ):
            try:
                owners = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                raise ValueError("product_reuse_responsibility_type") from None
        else:
            kept.append(node)
    tree.body = kept
    if type(owners) is not dict or any(
        type(key) is not str
        or type(value) is not tuple
        or any(type(name) is not str for name in value)
        for key, value in owners.items()
    ):
        raise ValueError("product_reuse_responsibility_type")
    return ast.dump(tree, include_attributes=False), owners


def reuse_implementation(registration):
    from .precise_dependencies import PACKAGE, PRODUCT_ROOTS, package_closure

    proof = registration["binding"].get("product_reuse")
    if type(proof) is not dict or proof.get("contract") != "v1-07g-product-reuse.v1":
        raise ValueError("product_reuse_contract")
    original = proof.get("inventory_source")
    prior = registration.get("implementation", {})
    if type(original) is not str or hashlib.sha256(
        original.encode()
    ).hexdigest() != prior.get("precise_dependencies"):
        raise ValueError("product_reuse_inventory_source")
    live = (PACKAGE / "precise_dependencies.py").read_text()
    before, prior_owners = inventory_semantics(original)
    after, owners = inventory_semantics(live)
    if before != after or set(prior_owners) != set(owners):
        raise ValueError("product_reuse_inventory_semantics")
    additions = set()
    for layer, names in owners.items():
        old = prior_owners[layer]
        added = set(names) - set(old)
        if tuple(name for name in names if name not in added) != old:
            raise ValueError("product_reuse_responsibility_changed")
        additions.update(Path(name).stem for name in added)
    diagnostics = set(package_closure(("precise_dev_score.py",)))
    if not additions <= diagnostics - set(prior):
        raise ValueError("product_reuse_non_diagnostic_addition")
    digests = {}
    for module in package_closure(PRODUCT_ROOTS):
        path = PACKAGE.joinpath(*module.split(".")).with_suffix(".py")
        if not path.is_file():
            path = PACKAGE.joinpath(*module.split("."), "__init__.py")
        value = prior.get(module)
        if type(value) is not str or len(value) != 64:
            raise ValueError("product_reuse_dependency_missing:" + module)
        if (
            module != "precise_dependencies"
            and hashlib.sha256(path.read_bytes()).hexdigest() != value
        ):
            raise ValueError("product_reuse_dependency_changed:" + module)
        digests[module] = value
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()


def attach_product_reuse(registration, source, inventory_path):
    registration["binding"]["product_reuse"] = {
        "contract": "v1-07g-product-reuse.v1",
        "inventory_source": Path(inventory_path).read_text(),
    }
    original = reuse_implementation(registration)
    if any(
        run.get("snapshot", {}).get("implementation_sha256") != original
        for run in source["runs"].values()
    ):
        raise ValueError("product_reuse_source_snapshot")
    registration["binding"]["product_implementation_sha256"] = original
