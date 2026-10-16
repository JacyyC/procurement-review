"""The complete dependency and schema fingerprint of the precise review candidate.

A version string is not a dependency.  What a Run's result actually depends on is the
implementation the Run executes, the model-visible schema and prompt each branch really
receives, the endpoint configuration, the pinned data and the lock files — so the
fingerprint is taken over exactly those, and a change in any of them is a different
candidate whatever the version names say.

The dependency boundary is not written here: it comes from
:mod:`procurement_review.precise_dependencies`, the one authoritative inventory both the
candidate fingerprint and the Run execution snapshot derive from, so a module split out of
another one or a newly imported dependency is inside the boundary without editing a list.
The generated artifacts are hashed from the live builders rather than from a hand-kept
list: the model-visible schemas come from the same function the request uses, and the
prompt text is the same constant the transport sends.  Nothing here sends, writes or reads
outside the repository.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .precise_dependencies import (
    CANDIDATE_VERSIONS,
    MissingDependency,
    FRONTEND,
    PACKAGE,
    REPOSITORY,
    layer_of,
    module_path,
    require_present,
)

from .eval_g_retrieval import G_DATASET_VERSION
from .gold_v37_conditional import as_gold_case
from .gold_v38_dev_quality import (
    DevQualityCaseV1,
    dev_quality_cases,
    dev_quality_truth,
    id_map,
    mapped_case,
)
from .ingestion import DEFAULT_CONFIG, LocatorKind, SourceLocator, make_chunk_id

BACKEND = PACKAGE.parents[1]


def path_of(section: str, entry: str) -> Path:
    """Where one inventory entry lives, by section."""

    if section == "frontend":
        return FRONTEND / entry
    if section == "resources":
        return BACKEND / entry
    if section == "locks":
        return REPOSITORY / entry
    path = module_path(entry)
    if path is None:  # pragma: no cover - require_present refuses first
        raise MissingDependency(f"procurement_review.{entry} is listed but absent")
    return path


def _digest(path: Path) -> dict[str, Any]:
    body = path.read_bytes()
    return {
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
    }


def dependency_digests() -> dict[str, dict[str, Any]]:
    """The digest of every file the candidate depends on, by inventory section.

    A missing required file raises instead of being recorded as absent: a candidate whose
    dependency cannot be read is not a candidate with one fewer dependency.
    """

    sections = require_present()
    return {
        section: {entry: _digest(path_of(section, entry)) for entry in entries}
        for section, entries in sections.items()
    }


def responsibility_layers() -> dict[str, tuple[str, ...]]:
    """Which modules carry which responsibility, for reporting what moved."""

    layers: dict[str, set[str]] = {}
    for section, entries in require_present().items():
        if section in {"resources", "locks"}:
            layers.setdefault(section, set()).update(entries)
            continue
        for entry in entries:
            name = entry if section == "frontend" else entry
            layers.setdefault(layer_of(entry) or section, set()).add(name)
    return {layer: tuple(sorted(names)) for layer, names in sorted(layers.items())}


def _synthetic_request(worker: str):
    """The branch request a fingerprint is taken over: the live builder, fixed inputs.

    It is the same function that renders the live task, so a change in the task text, the
    duty set, the field registry or the citation enumeration moves this digest.
    """

    from .business_v31_documents import document_request_v31
    from .business_v3_types import BusinessEvidenceIdentity
    from .model_wire_v2 import SourceSlice

    context = SimpleNamespace(
        scope_id="scope:candidate-fingerprint",
        purchase_order_id="PO-CANDIDATE",
        as_of=date(2026, 1, 1),
    )
    contents = tuple(f"candidate fingerprint source {index}" for index in range(2))
    sources = tuple(
        SourceSlice(
            scope_id=context.scope_id,
            worker=worker,
            identity=BusinessEvidenceIdentity(
                source_id=f"CANDIDATE-{worker.upper()}-{index}",
                version="candidate-fingerprint.v1",
                locator=f"chunk-candidate-{worker}-{index}",
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            ),
            content=content,
        )
        for index, content in enumerate(contents)
    )
    return document_request_v31(
        context,
        case_id="V107-D-0000",
        worker=worker,
        question="fingerprint",
        sources=sources,
    )


def generated_digests() -> dict[str, Any]:
    """The model-visible artifacts, hashed exactly as the endpoint would receive them."""

    from .business_v31_documents import VALUE_MODELS_V31
    from .precise_run_transport import (
        PRECISE_ANSWER_VERSION,
        PRECISE_SYSTEM_PROMPT,
        PRECISE_WIRE_SCHEMA,
        precise_answer_schema,
        precise_wire_fingerprint,
    )

    def digest(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    workers: dict[str, Any] = {}
    for worker in sorted(VALUE_MODELS_V31):
        request = _synthetic_request(worker)
        schema = precise_answer_schema(request)
        workers[worker] = {
            "schema_sha256": digest(schema),
            "task_text_sha256": digest(request.question),
            "duties": sorted(VALUE_MODELS_V31[worker]),
            "fields": {
                duty: sorted(model.model_fields)
                for duty, model in VALUE_MODELS_V31[worker].items()
            },
            "schema_properties": sorted(schema.get("properties", {})),
            "schema_defs": sorted(schema.get("$defs", {})),
        }
    return {
        "answer_version": PRECISE_ANSWER_VERSION,
        "wire_schema": PRECISE_WIRE_SCHEMA,
        "system_prompt_sha256": digest(PRECISE_SYSTEM_PROMPT),
        "system_prompt": PRECISE_SYSTEM_PROMPT,
        "wire_fingerprint": precise_wire_fingerprint(),
        "workers": workers,
    }


def candidate_fingerprint() -> dict[str, Any]:
    """The whole fixed input of one candidate, with one digest over all of it."""

    from .eval_f_registration import configuration
    from .precise_dev_paired import paired_configuration

    sections = dependency_digests()
    generated = generated_digests()
    endpoint = configuration().model_dump(mode="json")
    real_endpoint = paired_configuration().model_dump(mode="json")
    body = {
        "sections": sections,
        "generated": {
            key: value for key, value in generated.items() if key != "system_prompt"
        },
        "endpoint": {"test_only": endpoint, "real": real_endpoint},
        "versions": list(CANDIDATE_VERSIONS),
    }
    return {
        **body,
        "file_counts": {section: len(entries) for section, entries in sections.items()},
        "candidate_sha256": hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
    }


def canonical_json_sha256(value: Any) -> str:
    """The one canonical digest the product's payload hashes use (trailing newline kept)."""

    body = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def location_sha256(source: dict[str, str]) -> str:
    """The digest one input event carries for the material item it was inspected from.

    The product locates an input event by the digest of the source slice's rendered
    reference; reproducing that encoding here lets the independent truth fix the location
    the observed event has to name, instead of accepting any digest.
    """

    identity = {
        "source_id": source["source_id"],
        "version": source["version"],
        "locator": source["locator"],
        "content_sha256": source["content_sha256"],
    }
    reference = "ref-" + canonical_json_sha256(identity)
    return hashlib.sha256(reference.encode()).hexdigest()


def branch_source_identities(case: DevQualityCaseV1) -> dict[str, dict[str, str]]:
    """The chunk identity each branch's own document was registered under.

    Each seeded document is exactly one chunk, so the identity is a pure function of the
    case's mapped text and the pinned ingestion rules; the scorer compares claims against
    these values instead of recomputing product internals.
    """

    mapped = mapped_case(case)
    gold = as_gold_case(mapped)
    out: dict[str, dict[str, str]] = {}
    for domain in ("policy", "contract"):
        document = getattr(gold.model_input, f"{domain}_document")
        content = document.content
        checksum = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
        locator = SourceLocator(
            kind=LocatorKind.TEXT_LINES,
            line_start=1,
            line_end=max(1, content.count("\n") + 1),
        )
        chunk_id = make_chunk_id(
            document_id=f"DOC-{case.case_id}-{domain.upper()}",
            ordinal=1,
            locator=locator,
            content_checksum=checksum,
            chunk_rules_version=DEFAULT_CONFIG.chunk_rules_version,
            chunk_rules_checksum=DEFAULT_CONFIG.chunk_rules_checksum,
        )
        out[domain] = {
            "source_id": document.source_id,
            "version": document.version,
            "locator": chunk_id,
            "content_sha256": checksum.removeprefix("sha256:"),
        }
    return out


def truth_extension(case: DevQualityCaseV1) -> dict[str, Any]:
    """The verification metadata the scorer needs beside the pre-registered truth.

    Every entry is derived from the same authored case the truth was authored for: the
    expected object identity of the case's spans, the chunk identity each branch reads,
    the security events with the material locus they must locate, the conflict pairs with
    both sides' full identity and values, and the per-field support registry.  None of it
    changes a label: it is what lets the scorer verify the labels it was given.
    """

    truth = dev_quality_truth(case)
    replacements = id_map(case)

    def map_text(value: str) -> str:
        for source, target in sorted(
            replacements.items(), key=lambda item: -len(item[0])
        ):
            value = value.replace(source, target)
        return value

    gold = as_gold_case(mapped_case(case))
    identities = branch_source_identities(case)
    security = [
        {
            "code": event.code,
            "origin": event.origin,
            "document": event.document,
            "location_sha256": location_sha256(identities[event.document]),
        }
        for event in truth.security_events
    ]
    conflicts = []
    order_id = id_map(case)[str(case.facts["purchase_order_id"])]
    for pair in truth.conflict_pairs:
        sides = {}
        for name, side in (("left", pair.left), ("right", pair.right)):
            sides[name] = {
                "check_id": side.check_id,
                "subject_id": order_id,
                "field_path": side.field_path,
                "effective_at": gold.as_of.isoformat(),
                "source_version": G_DATASET_VERSION,
                "value": side.value,
                "source_id": order_id,
                "locator": side.field_path,
            }
        conflicts.append(
            {"relation": pair.relation, "left": sides["left"], "right": sides["right"]}
        )
    support: dict[str, dict[str, Any]] = {}
    for expectation in truth.expectations:
        fields = {}
        for key, descriptor in expectation.support.items():
            body = descriptor.model_dump(mode="json")
            body["needles"] = [map_text(item) for item in body["needles"]]
            if isinstance(body.get("value"), str):
                body["value"] = map_text(body["value"])
            fields[key] = body
        if fields:
            support[expectation.obligation_id] = fields
    return {
        "subject_id": order_id,
        "as_of": gold.as_of.isoformat(),
        "branch_sources": identities,
        "security_events": security,
        "conflict_pairs": conflicts,
        "support": support,
    }


def truth_extension_v3(case: DevQualityCaseV1) -> dict[str, Any]:
    """Full field requirements, derived from authored material before seeing output."""
    import re

    extension = truth_extension(case)
    mapped = mapped_case(case)
    policy, contract = mapped.policy_text, mapped.contract_text
    lines = {
        "applies": next(
            line for line in policy.splitlines() if line.startswith("本办法")
        ),
        "quotes": next(line for line in policy.splitlines() if "报价规则" in line),
        "limits": next(
            line for line in policy.splitlines() if line.startswith("交付不超过")
        ),
        "supplier": next(
            line for line in policy.splitlines() if line.startswith("供应商要求")
        ),
        "money": next(
            line for line in policy.splitlines() if line.startswith("订单与发票")
        ),
        "dates": next(
            line for line in policy.splitlines() if line.startswith("记账日期")
        ),
    }
    texts = {"policy": policy, "contract": contract}
    prefixes = {
        "exclusion_basis": "租赁类交易不适用",
        "excluded_categories": "租赁类交易不适用",
        "exception_basis": "唯一来源采购",
        "exceptions": "唯一来源采购",
        "exception_requires": "唯一来源采购",
        "category": "类交易（本次合同",
        "category_basis": "类交易（本次合同",
        "single_source": "本合同为单一来源",
        "written_justification": "书面说明:",
        "approval_reference": "审批授权:",
        "commitment_current": "原质保承诺已被",
        "superseding_amendment": "原质保承诺已被",
        "standard": "验收标准STANDARD:",
        "responsibility": "质保责任WARRANTY_RESPONSIBILITY:",
        "damages": "赔付DAMAGES:",
        "cap": "责任上限CAP:",
        "written": "变更须双方",
        "other": "本次只审",
        "amount": "价款记录:",
        "currency": "价款记录:",
        "subject": "买方BUYER:",
        "buyer": "买方BUYER:",
        "seller": "买方BUYER:",
        "order": "订单ORDER:",
        "contract": "订单ORDER:",
    }
    registry = {}
    for duty, fields in extension["support"].items():
        registry[duty] = {}
        for key, old in fields.items():
            worker, field = key.split(".")
            text = texts[worker]
            if worker == "policy" and field in {"applies", "basis", "exempt"}:
                clause = lines["applies"]
            elif worker == "policy" and field in {
                "quotes",
                "minimum_quotes",
                "applies_to",
            }:
                clause = lines["quotes"]
            elif worker == "policy" and field in {
                "delivery",
                "acceptance",
                "warranty",
                "special",
            }:
                clause = lines["limits"]
            elif worker == "policy" and field in lines:
                clause = lines[field]
            elif (
                field in {"excluded_categories", "exclusion_basis"}
                and not old["value"]
                and not old["needles"]
            ):
                clause = lines["quotes"]
            elif worker == "contract" and field in {
                "delivery",
                "acceptance",
                "warranty",
            }:
                clause = next(
                    line
                    for line in text.splitlines()
                    if line.startswith("交付DELIVERY")
                )
            else:
                clause = next(
                    line for line in text.splitlines() if prefixes[field] in line
                )
            value = old["value"]
            if old["kind"] == "needles":
                if field == "special":
                    numbers = re.findall(
                        r"(交付|验收|质保)(不超过|至少)(\d+)日", clause
                    )
                    names = {
                        "交付": "delivery",
                        "验收": "acceptance",
                        "质保": "warranty",
                    }
                    value = ";".join(
                        names[name] + ("<=" if op == "不超过" else ">=") + n
                        for name, op, n in numbers
                    )
                elif field == "subject":
                    value = re.search(r"标的SUBJECT:\s*(.+)", clause)[1]
                elif field in {"standard", "responsibility", "damages"}:
                    value = clause.split(":", 1)[1].strip()
                else:
                    value = clause
            registry[duty][key] = {
                "field": field,
                "value": value,
                "clauses": [clause],
                "requirement": "required",
            }
    optional = {
        "policy.exclusion_basis",
        "policy.exception_basis",
        "contract.category_basis",
        "contract.approval_reference",
        "contract.superseding_amendment",
    }
    for fields in registry.values():
        for key, descriptor in fields.items():
            if key in optional:
                descriptor["requirement"] = "optional"
    extension["support"] = registry
    facts = case.facts
    mapping = id_map(case)
    order_id = extension["subject_id"]
    supplier_id = mapping[str(facts["supplier_id"])]
    transaction_id = mapping[str(facts["transaction_id"])]
    order = {
        "purchase_order_id": order_id,
        "supplier_id": supplier_id,
        "competitive_quote_count": facts["competitive_quote_count"],
        "contract_reference": mapping[str(facts["contract_reference"])],
        "ordered_on": facts["ordered_on"],
        "invoice_amount_minor": facts["invoice_amount_minor"],
        "invoice_currency": facts["invoice_currency"],
        "amount_minor": facts["order_amount_minor"],
        "currency": facts["order_currency"],
    }
    if case.transaction_category is not None:
        order["transaction_category"] = case.transaction_category
    supplier = {
        name: facts[name]
        for name in (
            "registry_status",
            "procurement_status",
            "tax_identifier",
            "due_diligence_status",
            "sanctions_screening",
        )
    }
    supplier["supplier_id"] = supplier_id
    transaction = {
        "transaction_id": transaction_id,
        "purchase_order_id": order_id,
        "supplier_id": supplier_id,
        "counterparty_reference": mapping[str(facts["counterparty_reference"])],
        "booked_on": facts["booked_on"],
        "amount_minor": facts["transaction_amount_minor"],
        "currency": facts["transaction_currency"],
    }
    extension["scope_id"] = f"tenant-v1-07g-dev-quality:{case.case_id}"
    catalog = {}
    for kind, sid, record in (
        ("purchase_order", order_id, order),
        ("supplier", supplier_id, supplier),
        ("historical_transaction", transaction_id, transaction),
    ):
        for field, value in {**record, "identity": sid}.items():
            catalog[f"{kind}.{field}"] = {
                "subject_id": sid,
                "subject_type": "transaction"
                if kind == "historical_transaction"
                else kind,
                "effective_at": extension["as_of"],
                "source_id": sid,
                "version": G_DATASET_VERSION,
                "content_sha256": canonical_json_sha256(record),
                "value": value,
            }
    extension["facts"] = catalog
    extension["required_facts"] = {
        "O-03": ["supplier.registry_status", "supplier.procurement_status"],
        "O-06": [
            "purchase_order.currency",
            "purchase_order.invoice_currency",
            "purchase_order.amount_minor",
            "purchase_order.invoice_amount_minor",
            "transaction.currency",
            "transaction.amount_minor",
        ],
        "O-07": ["purchase_order.ordered_on", "transaction.booked_on"],
    }
    for pair in extension["conflict_pairs"]:
        for side in (pair["left"], pair["right"]):
            side.update(
                scope_id=extension["scope_id"],
                subject_type="purchase_order",
                content_sha256=catalog[side["locator"]]["content_sha256"],
            )
    extension["field_disposition"] = {
        duty: {key: desc["requirement"] for key, desc in registry.get(duty, {}).items()}
        for duty in (f"O-{n:02}" for n in range(1, 13))
    }
    from .business_v31_documents import VALUE_MODELS_V31

    for worker, contracts in VALUE_MODELS_V31.items():
        for duty, model in contracts.items():
            for field in model.model_fields:
                extension["field_disposition"][duty].setdefault(
                    f"{worker}.{field}", "not_applicable"
                )
    return extension


def request_identities_v3(case):
    """Rebuild request identity only, never a truth answer, using the pinned wire API."""
    from .business_v31_documents import document_request_v31
    from .business_v3_types import BusinessEvidenceIdentity
    from .model_wire_v2 import SourceSlice
    from .precise_run_scope import PRECISE_QUESTIONS

    mapped = mapped_case(case)
    context = SimpleNamespace(
        scope_id=f"tenant-v1-07g-dev-quality:{case.case_id}",
        purchase_order_id=id_map(case)[str(case.facts["purchase_order_id"])],
        as_of=as_gold_case(case).as_of,
    )
    identities = branch_source_identities(case)
    result = {}
    for worker in ("policy", "contract"):
        source = SourceSlice(
            scope_id=context.scope_id,
            worker=worker,
            identity=BusinessEvidenceIdentity(**identities[worker]),
            content=getattr(mapped, worker + "_text"),
        )
        request = document_request_v31(
            context,
            case_id=case.case_id,
            worker=worker,
            question=PRECISE_QUESTIONS[worker],
            sources=(source,),
        )
        result[worker] = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
    return result


def extend_registration_v3(base_path: Path, output_path: Path) -> dict[str, Any]:
    from .precise_dev_sweep import extend_registration

    result = extend_registration(base_path, output_path)
    body = json.loads(output_path.read_text())
    body["truth_extension"] = {
        case.case_id: truth_extension_v3(case)
        for case in dev_quality_cases()
        if case.case_id in body["case_ids"]
    }
    body["extends"]["extension_version"] = "v1-07-truth-extension.v2"
    body["binding"]["expected_requests"] = {
        case.case_id: request_identities_v3(case)
        for case in dev_quality_cases()
        if case.case_id in body["case_ids"]
    }
    body["binding"]["expected_snapshot_fields"] = expected_snapshot_fields_v3()
    output_path.write_text(
        json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return {**result, "extension_version": body["extends"]["extension_version"]}


def expected_snapshot_fields_v3():
    """The frozen G real profile, distinct from the observer's 240-second wait limit."""
    from .precise_run_transport import precise_wire_fingerprint

    return {
        "snapshot_version": "precise-run-snapshot.v1",
        "response_mode": "json_schema",
        "prompt_style": "baseline",
        "thinking": "disabled",
        "temperature": 0.0,
        "top_p": None,
        "seed": None,
        "max_tokens": None,
        "single_attempt_seconds": 90.0,
        "branch_timeout_seconds": 90.0,
        "generation_deadline_seconds": 90.0,
        "retrieval_vector_adapter": "no-vector-v1",
        "retrieval_lexical_adapter": "deterministic-lexical-idf-v1",
        "wire_version": "finding-envelope.v2",
        "prompt_version": "worker-output-prompt.v2",
        "schema_sha256": precise_wire_fingerprint(),
    }


def verify_registration_v3(registration):
    """All v3 consumers share independent truth and frozen request/profile checks."""
    failures = []
    binding = registration.get("binding", {})
    if binding.get("expected_snapshot_fields") != expected_snapshot_fields_v3():
        failures.append("snapshot_registration_mismatch")
    cases = {case.case_id: case for case in dev_quality_cases()}
    from .precise_dev_sweep import binding_block

    chosen = tuple(cases[key] for key in registration["case_ids"] if key in cases)
    if chosen:
        for key, expected in binding_block(chosen).items():
            if (
                key == "product_implementation_sha256"
                and binding.get("consumption_version") == "v1-07g-consumption.v5"
                and "product_reuse" in binding
            ):
                from .eval_g_reuse import reuse_implementation

                expected = reuse_implementation(registration)
            if (
                type(binding.get(key)) is not type(expected)
                or binding.get(key) != expected
            ):
                failures.append(f"binding_registration_mismatch:{key}")
    for case_id in registration["case_ids"]:
        if case_id not in cases:
            failures.append(f"unknown_registered_case:{case_id}")
            continue
        case = cases[case_id]
        if registration.get("truth_extension", {}).get(case_id) != truth_extension_v3(
            case
        ):
            failures.append(f"independent_extension_mismatch:{case_id}")
        if binding.get("expected_requests", {}).get(case_id) != request_identities_v3(
            case
        ):
            failures.append(f"request_registration_mismatch:{case_id}")
    return failures


def project_base(body: dict[str, Any]) -> dict[str, Any]:
    from .precise_dev_sweep import EXTENSION_KEYS

    return {key: value for key, value in body.items() if key not in EXTENSION_KEYS}


def registration_binding(registration: dict, observations: dict) -> dict:
    """The pre-registered base reproduced from the extended file, byte for byte."""

    from .precise_dev_sweep import project_base

    projection = json.dumps(
        project_base(registration), ensure_ascii=False, sort_keys=True, indent=2
    ).encode()
    base_sha = hashlib.sha256(projection).hexdigest()
    declared = (registration.get("extends") or {}).get("base_registration_sha256")
    observed = observations.get("registration_sha256")
    return {
        "base_registration_sha256": base_sha,
        "declared_base_registration_sha256": declared,
        "observed_registration_sha256": observed,
        "extension_version": (registration.get("extends") or {}).get(
            "extension_version"
        ),
        "matched": bool(base_sha == declared == observed),
    }


def _scorer_candidate_sha256() -> str | None:
    """The candidate fingerprint of the tree that produced this verdict.

    The observations are bound to the pre-registered product build; this field names the
    evaluation consumer that judged them, so a stored verdict whose scorer moved is
    visible instead of silently attributed to the old one.
    """

    try:
        return candidate_fingerprint()["candidate_sha256"]
    except Exception:  # pragma: no cover - a scorer run without the full tree
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", action="append", default=None)
    parser.add_argument("--layers", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.layers:
        print(json.dumps(responsibility_layers(), ensure_ascii=False, sort_keys=True))
        return
    fingerprint = candidate_fingerprint()
    if args.section:
        for section in args.section:
            print(
                json.dumps(
                    {section: fingerprint["sections"].get(section)},
                    ensure_ascii=False,
                )
            )
        return
    print(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, indent=2))


__all__ = [
    "candidate_fingerprint",
    "dependency_digests",
    "generated_digests",
    "main",
    "path_of",
    "responsibility_layers",
]


if __name__ == "__main__":
    main()
