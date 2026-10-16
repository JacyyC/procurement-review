"""Independent duty-to-evidence requirements from the frozen business contract.

These requirements describe what must be proved, not how production rules decide.
No production aggregator, procurement rule, or expected-output builder is called.
"""

import json as _json
import unicodedata as _unicodedata
from typing import Any

from .eval_f_semantics import fact_catalog, source_identity, evidence_valid
from .gold_v3_contracts import payload_sha256


def identity_key(value):
    return tuple(
        value[k] for k in ("source_id", "version", "locator", "content_sha256")
    )


def obligation_evidence_valid(case, target, observed, decision, conflicts):
    facts = fact_catalog(case)
    by_kind = {
        kind: (sid, fields, version) for sid, (kind, fields, version) in facts.items()
    }

    def fields(kind, *names):
        sid, body, version = by_kind[kind]
        return {
            (sid, version, kind + "." + name, payload_sha256(body)) for name in names
        }

    policy = {identity_key(source_identity(case.model_input.policy_document))}
    contract = {identity_key(source_identity(case.model_input.contract_document))}
    control = {
        identity_key(e.model_dump())
        for f in decision.business.findings
        for e in f.evidence
        if e.source_id == "trusted-review-control" and evidence_valid(case, e, decision)
    }
    requirements = {
        "O-01": policy,
        "O-02": policy,
        "O-03": policy
        | fields(
            "supplier",
            "registry_status",
            "procurement_status",
            "tax_identifier",
            "due_diligence_status",
            "sanctions_screening",
        ),
        "O-04": policy
        | fields("purchase_order", "competitive_quote_count", "contract_reference")
        | fields("historical_transaction", "counterparty_reference"),
        "O-05": contract
        | fields("purchase_order", "contract_reference")
        | fields("historical_transaction", "counterparty_reference"),
        "O-06": policy
        | contract
        | fields(
            "purchase_order",
            "amount_minor",
            "currency",
            "invoice_amount_minor",
            "invoice_currency",
        )
        | fields("historical_transaction", "amount_minor", "currency"),
        "O-07": policy
        | fields("purchase_order", "ordered_on")
        | fields("historical_transaction", "booked_on"),
        "O-08": policy | contract,
        "O-09": contract,
        "O-10": set().union(*(fields(kind, "identity") for kind in by_kind)),
        "O-11": control,
        "O-12": control,
    }
    if conflicts:
        pair_fields = set()
        for pair in conflicts:
            for side in (pair.left, pair.right):
                record = facts.get(side.subject_id)
                if record is None:
                    return False
                kind = record[0]
                name = side.field_path.split(".")[-1]
                if kind == "purchase_order" and name == "order_currency":
                    name = "currency"
                pair_fields |= fields(kind, name)
        requirements["O-11"] = pair_fields
    actual = {identity_key(e.model_dump()) for e in observed.evidence}
    required = requirements[target.obligation_id]
    if not all(evidence_valid(case, e, decision) for e in observed.evidence):
        return False
    if observed.state != "unknown":
        return bool(required) and actual == required
    # Missing information must identify the actual missing facts, not arbitrary text.
    if target.normalized_value == "quote_or_reference_unknown":
        f = case.model_input.business_facts
        missing = {
            name
            for name in (
                "competitive_quote_count",
                "contract_reference",
                "counterparty_reference",
            )
            if getattr(f, name) is None or getattr(f, name) == ""
        }
    elif target.normalized_value == "document_terms_insufficient":
        missing = {
            {
                "O-05": "parties_subject_references",
                "O-08": "delivery_acceptance_warranty",
                "O-09": "written_change_damages_cap",
            }.get(target.obligation_id, "")
        }
    else:
        return False
    return bool(missing) and set(observed.missing) == missing and actual <= required


# ------------------------------------------------------- G DEV claim verification v2
#
# The functions below judge one claimed proposition against the material and the
# registered expectation, independently of the product line: the claim has to equal what
# the material fixes, and the cited excerpt has to be the passage that says it.  Nothing
# here calls a production rule, aggregator, grounder or the scorer's own product.


def normalize_text(text: str) -> str:
    """The comparison form of a text: NFKC, no whitespace, no format characters."""

    return "".join(
        character
        for character in _unicodedata.normalize("NFKC", text)
        if not character.isspace() and _unicodedata.category(character) not in {"Cf"}
    )


def _strip_terminal(text: str) -> str:
    """Drop surrounding quotes and one sentence-ending mark the model may have carried."""

    stripped = text.strip().strip("\"'“”‘’").strip()
    return stripped.rstrip("。．.！!")


def canonical_claim(value):
    """A claim in comparable form: JSON text becomes its value, strings stay trimmed."""

    if isinstance(value, str):
        text = value.strip()
        try:
            return _json.loads(text)
        except ValueError:
            return _strip_terminal(text)
    return value


def claims_equal(left, right) -> bool:
    a, b = canonical_claim(left), canonical_claim(right)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return normalize_text(_strip_terminal(a)) == normalize_text(_strip_terminal(b))
    return a == b


def claim_text(value) -> str:
    """One claim as the text its needles have to be found in."""

    body = canonical_claim(value)
    if isinstance(body, str):
        return body
    return _json.dumps(body, ensure_ascii=False, sort_keys=True)


def text_carries(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return normalize_text(needle) in normalize_text(haystack)


def identity_verdict(span: dict, expected: dict) -> list[str]:
    """Which parts of a claim's citation identity are not the authorized ones."""

    failures = []
    if not span.get("source_id") or not span.get("version") or not span.get("locator"):
        failures.append("citation_identity_missing")
        return failures
    if span.get("source_id") != expected.get("source_id"):
        failures.append("wrong_source")
    if span.get("version") != expected.get("version") or (
        span.get("source_version")
        and span.get("source_version") != expected.get("version")
    ):
        failures.append("wrong_version")
    if span.get("locator") != expected.get("locator"):
        failures.append("wrong_locator")
    return failures


def judge_claim(descriptor: dict, claim, excerpt: str) -> dict:
    """One claim against its registered expectation and the excerpt it cites."""

    kind = descriptor.get("kind")
    failures: list[str] = []
    if kind == "literal":
        if not claims_equal(claim, descriptor.get("value")):
            failures.append("value_not_material")
        elif not text_carries(excerpt, str(descriptor.get("value"))):
            failures.append("excerpt_does_not_carry_value")
    elif kind == "token":
        if not claims_equal(claim, descriptor.get("value")):
            failures.append("value_not_material")
        for needle in descriptor.get("needles") or ():
            if not text_carries(excerpt, needle):
                failures.append("excerpt_does_not_carry_token")
                break
    elif kind == "needles":
        text = claim_text(claim)
        for needle in descriptor.get("needles") or ():
            if not text_carries(text, needle):
                failures.append("claim_missing_needle")
                break
        for needle in descriptor.get("needles") or ():
            if not text_carries(excerpt, needle):
                failures.append("excerpt_missing_needle")
                break
    else:
        failures.append("unregistered_field")
    return {
        "kind": kind,
        "failures": failures,
        "value_ok": not any(
            item in {"value_not_material", "claim_missing_needle"} for item in failures
        ),
        "support_ok": not any(
            item
            in {
                "excerpt_does_not_carry_value",
                "excerpt_does_not_carry_token",
                "excerpt_missing_needle",
            }
            for item in failures
        ),
        "ok": not failures,
    }


def excerpt_of(material: dict, case_id: str, span: dict) -> str:
    """The cited passage: the branch document's own text at the span's [start, end)."""

    document = (material.get(case_id) or {}).get(span.get("worker"))
    if not isinstance(document, str):
        return ""
    start = span.get("start")
    if not isinstance(start, int) or start < 0 or start > len(document):
        return ""
    end = span.get("end")
    return document[start : end if isinstance(end, int) else None]


def _expected_side_signature(side: dict) -> tuple | None:
    """One truth side as the exact identity a prediction has to reproduce."""

    identity = (
        side.get("subject_id"),
        side.get("check_id"),
        side.get("field_path"),
        side.get("effective_at"),
        side.get("source_version"),
    )
    if any(item in (None, "") for item in identity) or side.get("value") is None:
        return None
    evidence = {
        (side.get("source_id"), side.get("source_version"), side.get("locator"))
    }
    return (*identity, canonical_claim(side.get("value")), frozenset(evidence))


def _observed_side_signature(side: dict) -> tuple | None:
    """One predicted side, or ``None`` when it cannot carry a full identity."""

    proposition = side.get("proposition") or {}
    identity = (
        proposition.get("subject_id"),
        proposition.get("check_id"),
        proposition.get("field_path"),
        proposition.get("effective_at"),
        proposition.get("source_version"),
    )
    if any(item in (None, "") for item in identity) or side.get("value") is None:
        return None
    evidence = {
        (item.get("source_id"), item.get("version"), item.get("locator"))
        for item in side.get("evidence") or []
        if isinstance(item, dict)
    }
    if not evidence or any(None in item for item in evidence):
        return None
    return (*identity, canonical_claim(side.get("value")), frozenset(evidence))


#: A citation that left the branch's authorized context: never acceptable, counted from
#: the Run's own refusal codes and from every claimed identity that is not the authorized
#: one.
OUT_OF_SCOPE_IDENTITY = frozenset(
    {"citation_identity_missing", "wrong_source", "wrong_version", "wrong_locator"}
)


def judge_span(registration: dict, case_id: str, span: dict, material: dict) -> dict:
    """One claimed proposition against its authorized identity and the cited excerpt."""

    extension = registration["truth_extension"][case_id]
    worker = span.get("worker")
    sources = extension["branch_sources"].get(str(worker))
    descriptor = (
        registration.get("truth_extension", {})
        .get(case_id, {})
        .get("support", {})
        .get(str(span.get("check_id")), {})
        .get(f"{worker}.{span.get('field')}")
    )
    excerpt = excerpt_of(material, case_id, span)
    failures: list[str] = []
    identity_failures = list(identity_verdict(span, sources or {}))
    if sources is None:
        identity_failures = ["unknown_branch"]
    object_ok = span.get("subject_id") in (None, extension.get("subject_id"))
    if not object_ok:
        failures.append("wrong_object")
    if not excerpt.strip():
        failures.append("empty_excerpt")
    claim = None
    if descriptor is None:
        failures.append("unregistered_field")
    else:
        claim = judge_claim(descriptor, span.get("value"), excerpt)
        failures.extend(claim["failures"])
    failures = list(dict.fromkeys(identity_failures + failures))
    out_of_scope = bool(
        OUT_OF_SCOPE_IDENTITY.intersection(identity_failures)
        or "wrong_object" in failures
    )
    return {
        "worker": worker,
        "check_id": span.get("check_id"),
        "field": span.get("field"),
        "value": span.get("value"),
        "excerpt": excerpt,
        "identity_failures": identity_failures,
        "claim": claim,
        "failures": failures,
        "supported": not failures,
        "out_of_scope": out_of_scope,
        "value_ok": bool(claim and claim.get("value_ok")) if claim else False,
        "support_ok": bool(claim and claim.get("support_ok")) if claim else False,
    }


def judge_obligation(
    expectation: dict,
    verdicts: list[dict],
    finding: dict | None,
) -> dict[str, Any]:
    """One obligation: state, reason, values, clauses, locators and every claim it made."""

    obligation = expectation["obligation_id"]
    own = [item for item in verdicts if item.get("check_id") == obligation]
    supported = [item for item in own if item["supported"]]
    missing_clauses = [
        clause
        for clause in expectation.get("document_clauses", ())
        if not any(text_carries(item["excerpt"], clause) for item in supported)
    ]
    cited = {
        str(item.get("locator"))
        for item in (finding or {}).get("evidence", [])
        if isinstance(item, dict)
    }
    missing_locators = [
        locator
        for locator in expectation.get("fact_locators", ())
        if locator not in cited
    ]
    value_failures = []
    for field, expected in expectation.get("values", {}).items():
        if not any(
            item.get("field") == field and claims_equal(item.get("value"), expected)
            for item in supported
        ):
            value_failures.append(f"{field}:{expected!r}")
    contains_failures = []
    for field, needles in expectation.get("contains", {}).items():
        observed = [
            str(item.get("value")) for item in supported if item.get("field") == field
        ]
        if not observed or not all(
            any(text_carries(item, needle) for item in observed) for needle in needles
        ):
            contains_failures.append(f"{field}:{list(needles)}")
    registry = expectation.get("support", {})
    missing_claims = [
        key
        for key in registry
        if not any(f"{item.get('worker')}.{item.get('field')}" == key for item in own)
    ]
    claim_failures = [
        f"{item.get('worker')}.{item.get('field')}:{code}"
        for item in own
        for code in item["failures"]
    ]
    if finding is None:
        return {
            "obligation_id": obligation,
            "judged": False,
            "correct": False,
            "reason": "missing_from_decision",
            "expected_state": expectation["state"],
            "expected_reason": expectation["reason"],
        }
    state_ok = finding.get("state") == expectation["state"]
    reason_ok = finding.get("reason") == expectation["reason"]
    return {
        "obligation_id": obligation,
        "judged": True,
        "state_ok": state_ok,
        "reason_ok": reason_ok,
        "expected_state": expectation["state"],
        "observed_state": finding.get("state"),
        "expected_reason": expectation["reason"],
        "observed_reason": finding.get("reason"),
        "value_failures": value_failures,
        "contains_failures": contains_failures,
        "missing_clauses": missing_clauses,
        "missing_locators": missing_locators,
        "missing_claims": missing_claims,
        "claim_failures": claim_failures,
        "correct": bool(
            state_ok
            and reason_ok
            and not value_failures
            and not contains_failures
            and not missing_clauses
            and not missing_locators
            and not missing_claims
            and not claim_failures
        ),
    }


def conflict_pairs_matched(expected_pairs: list, observed: list) -> dict:
    """Truth pairs against predictions, matched one-to-one on the full identity.

    A prediction counts only when it names the same relation *and* both sides carry the
    object, the matter, the value, the time, the version and their own evidence the truth
    fixes; every prediction stays in the precision denominator, including an ill-formed
    one, so a wrong or unattributable pair cannot disappear from the reading.
    """

    expected_signatures = []
    for pair in expected_pairs:
        relation = pair.get("relation")
        left = _expected_side_signature(pair.get("left") or {})
        right = _expected_side_signature(pair.get("right") or {})
        expected_signatures.append((relation, left, right))
    used = [False] * len(expected_signatures)
    matched = 0
    well_formed = 0
    for pair in observed:
        left = _observed_side_signature(pair.get("left") or {})
        right = _observed_side_signature(pair.get("right") or {})
        if left is None or right is None:
            continue
        well_formed += 1
        relation = pair.get("relation")
        for index, (expected_relation, want_left, want_right) in enumerate(
            expected_signatures
        ):
            if used[index] or relation != expected_relation:
                continue
            direct = left == want_left and right == want_right
            crossed = left == want_right and right == want_left
            if direct or crossed:
                used[index] = True
                matched += 1
                break
    return {
        "expected": len(expected_signatures),
        "predicted": len(observed),
        "well_formed": well_formed,
        "matched": matched,
        "ill_formed": len(observed) - well_formed,
        "precision": None if not observed else matched / len(observed),
        "recall": None
        if not expected_signatures
        else matched / len(expected_signatures),
    }


def judge_span_v3(registration, case_id, span, material):
    from .eval_f_text import claim_supported_v3

    extension = registration["truth_extension"][case_id]
    worker, duty, field = span.get("worker"), span.get("check_id"), span.get("field")
    source = extension["branch_sources"].get(worker)
    descriptor = extension["support"].get(duty, {}).get(f"{worker}.{field}")
    identity_failures = identity_verdict(span, source or {})
    if source is None:
        identity_failures.append("unknown_branch")
    failures = list(identity_failures)
    if span.get("source_version") != (source or {}).get("version"):
        failures.append("wrong_source_version")
    if span.get("subject_id") != extension["subject_id"]:
        failures.append("wrong_object")
    if (
        span.get("field_path")
        != f"{worker}.{str(duty).lower().replace('-', '_')}.{field}"
    ):
        failures.append("wrong_field_path")
    if span.get("subject_type") != "purchase_order":
        failures.append("wrong_subject_type")
    for name, expected in (
        ("effective_at", extension["as_of"]),
        ("scope_id", extension["scope_id"]),
        ("content_sha256", (source or {}).get("content_sha256")),
    ):
        if span.get(name) != expected:
            failures.append(f"wrong_{name}")
    text = material.get(case_id, {}).get(worker, "")
    start, end = span.get("start"), span.get("end")
    bounds_ok = (
        type(start) is int and type(end) is int and 0 <= start < end <= len(text)
    )
    excerpt = text[start:end] if bounds_ok else ""
    if not bounds_ok:
        failures.append("invalid_excerpt_bounds")
    claim = (
        claim_supported_v3(descriptor, span.get("value"), excerpt)
        if descriptor
        else None
    )
    failures.extend(claim["failures"] if claim else ["unregistered_field"])
    return {
        "worker": worker,
        "check_id": duty,
        "field": field,
        "value": span.get("value"),
        "excerpt": excerpt,
        "identity_failures": identity_failures,
        "claim": claim,
        "failures": failures,
        "supported": not failures,
        "out_of_scope": bool(
            identity_failures
            or "wrong_object" in failures
            or "wrong_scope_id" in failures
        ),
        "value_ok": bool(claim and claim["value_ok"]),
        "support_ok": bool(claim and claim["support_ok"]),
    }


def judge_obligation_v3(expectation, verdicts, finding, extension):
    """Required fields come from the extension, even when no output mentions them."""
    obligation = expectation["obligation_id"]
    required = {
        key: value
        for key, value in extension["support"].get(obligation, {}).items()
        if value["requirement"] == "required"
    }
    result = judge_obligation(
        {**expectation, "support": required, "contains": {}, "values": {}},
        verdicts,
        finding,
    )
    failures = []
    if finding is not None:
        for field in ("risk", "blocking"):
            if field == "risk" and expectation["state"] == "unknown":
                continue  # UNKNOWN's authored labels specify state/reason, not a risk grade.
            if (
                type(finding.get(field)) is not type(expectation[field])
                or finding.get(field) != expectation[field]
            ):
                failures.append(f"wrong_{field}")
        own = [item for item in verdicts if item["check_id"] == obligation]
        keys = [(item["worker"], item["field"]) for item in own]
        if len(keys) != len(set(keys)):
            failures.append("duplicate_claim")
        seen = []
        allowed_facts = {
            "O-03": {
                "supplier." + name
                for name in (
                    "registry_status",
                    "procurement_status",
                    "tax_identifier",
                    "due_diligence_status",
                    "sanctions_screening",
                )
            },
            "O-04": {
                "purchase_order.competitive_quote_count",
                "purchase_order.contract_reference",
                "historical_transaction.counterparty_reference",
                "purchase_order.transaction_category",
            },
            "O-05": {
                "purchase_order.contract_reference",
                "historical_transaction.counterparty_reference",
            },
            "O-06": {
                "purchase_order." + name
                for name in (
                    "amount_minor",
                    "currency",
                    "invoice_amount_minor",
                    "invoice_currency",
                )
            }
            | {
                "historical_transaction.amount_minor",
                "historical_transaction.currency",
            },
            "O-07": {"purchase_order.ordered_on", "historical_transaction.booked_on"},
            "O-10": {
                kind + ".identity"
                for kind in ("purchase_order", "supplier", "historical_transaction")
            },
            "O-11": {"purchase_order.currency", "purchase_order.invoice_currency"},
        }
        document_workers = {
            "O-01": {"policy"},
            "O-02": {"policy"},
            "O-03": {"policy"},
            "O-04": {"policy", "contract"},
            "O-05": {"contract"},
            "O-06": {"policy", "contract"},
            "O-07": {"policy"},
            "O-08": {"policy", "contract"},
            "O-09": {"contract"},
        }
        for evidence in finding.get("evidence", []):
            locator = evidence.get("locator")
            descriptor = extension["facts"].get(locator)
            if descriptor is None:
                descriptor = next(
                    (
                        source
                        for worker, source in extension["branch_sources"].items()
                        if worker in document_workers.get(obligation, set())
                        and source["locator"] == locator
                    ),
                    None,
                )
            elif locator not in allowed_facts.get(obligation, set()):
                descriptor = None
            if (
                descriptor is None
                and locator == "source-and-output-inspection"
                and obligation in {"O-11", "O-12"}
            ):
                descriptor = {
                    "source_id": "trusted-review-control",
                    "version": "business-review.v3.1",
                }
            if descriptor is None or any(
                evidence.get(key) != descriptor.get(key)
                for key in ("source_id", "version")
            ):
                failures.append(f"fact_identity_mismatch:{locator}")
            elif "content_sha256" in evidence and evidence[
                "content_sha256"
            ] != descriptor.get("content_sha256"):
                if locator != "source-and-output-inspection":
                    failures.append(f"fact_content_mismatch:{locator}")
            seen.append((evidence.get("source_id"), evidence.get("version"), locator))
        if len(seen) != len(set(seen)):
            failures.append("duplicate_evidence")
    result["evidence_failures"] = failures
    result["correct"] = result["correct"] and not failures
    return result


def judge_fact_v3(extension, observed):
    """A deterministic fact proposition, with full scope/object/path/time/source binding."""
    proposition = observed.get("proposition", {})
    path = proposition.get("field_path", "")
    locator = (
        path.replace("transaction.", "historical_transaction.", 1)
        if path.startswith("transaction.")
        else path
    )
    fact = extension["facts"].get(locator)
    failures = []
    if fact is None:
        failures.append("unknown_fact_field")
    else:
        for key in ("subject_id", "subject_type", "effective_at"):
            if proposition.get(key) != fact[key]:
                failures.append(f"fact_{key}")
        if (
            proposition.get("scope_id") != extension["scope_id"]
            or proposition.get("source_version") != fact["version"]
        ):
            failures.append("fact_scope_or_version")
        if not claims_equal(observed.get("value"), fact["value"]):
            failures.append("fact_value_mismatch")
        evidence = observed.get("evidence", [])
        if (
            len(evidence) != 1
            or any(
                evidence[0].get(key) != fact[key]
                for key in ("source_id", "version", "content_sha256")
            )
            or evidence[0].get("locator") != locator
        ):
            failures.append("fact_evidence_identity")
        allowed = {"supplier": "O-03", "purchase_order": "O-06", "transaction": "O-06"}
        duty = (
            "O-07"
            if locator.endswith((".ordered_on", ".booked_on"))
            else allowed[fact["subject_type"]]
        )
        if proposition.get("check_id") != duty:
            failures.append("fact_check_identity")
    return {
        "failures": failures,
        "supported": not failures,
        "out_of_scope": any(code != "fact_value_mismatch" for code in failures),
    }


def conflict_pairs_matched_v3(expected_pairs, observed):
    """Retain every predicted pair, validating scope, object type and source contents."""
    import copy

    checked = copy.deepcopy(observed)
    outside = 0
    for pair in checked:
        for name in ("left", "right"):
            side = pair.get(name, {})
            candidates = [expected[name] for expected in expected_pairs] + [
                expected["right" if name == "left" else "left"]
                for expected in expected_pairs
            ]
            prop = side.get("proposition", {})
            signature = _observed_side_signature(side)
            if not any(
                signature is not None
                and signature[:5] == _expected_side_signature(expected)[:5]
                and signature[-1] == _expected_side_signature(expected)[-1]
                and prop.get("scope_id") == expected["scope_id"]
                and prop.get("subject_type") == expected["subject_type"]
                and len(side.get("evidence", [])) == 1
                and side["evidence"][0].get("content_sha256")
                == expected["content_sha256"]
                for expected in candidates
            ):
                side["proposition"] = {}
                outside += 1
    return {**conflict_pairs_matched(expected_pairs, checked), "out_of_scope": outside}
