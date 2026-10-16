"""Unordered, evidence-bearing pairs for exact propositions and named relations."""

from itertools import combinations
from typing import Literal

from .business_v3_types import BusinessObservationV3
from .eval_v2_contracts import EvalV2Model
from .gold_v3_contracts import payload_sha256


class ConflictPairV3(EvalV2Model):
    pair_id: str
    kind: Literal["business_conflict", "rule_inconsistency"]
    relation: Literal[
        "same_proposition_exclusive_values", "order_invoice_currency_equal"
    ]
    left: BusinessObservationV3
    right: BusinessObservationV3


def _known(value):
    return value is not None and not (
        isinstance(value, str)
        and value.casefold().strip() in {"", "unknown", "missing", "null", "unresolved"}
    )


def conflict_pairs_v3(observations):
    """Only consume validated observations; never a finding's prose status."""
    pairs = {}
    for a, b in combinations(observations, 2):
        if (
            not a.evidence
            or not b.evidence
            or not _known(a.value)
            or not _known(b.value)
        ):
            continue
        if type(a.value) is type(b.value) and a.value == b.value:
            continue
        if a.proposition == b.proposition:
            kind, relation = "business_conflict", "same_proposition_exclusive_values"
        else:
            ap, bp = a.proposition, b.proposition
            same_context = ap.model_dump(exclude={"field_path"}) == bp.model_dump(
                exclude={"field_path"}
            )
            currency_relation = (
                ap.subject_type == "purchase_order"
                and ap.check_id == "O-06"
                and {ap.field_path, bp.field_path}
                == {"purchase_order.currency", "purchase_order.invoice_currency"}
            )
            if (
                not same_context
                or not currency_relation
                or not (a.value in {"CNY", "USD"} and b.value in {"CNY", "USD"})
            ):
                continue
            kind, relation = "rule_inconsistency", "order_invoice_currency_equal"
        left, right = sorted(
            (a, b), key=lambda row: payload_sha256(row.model_dump(mode="json"))
        )
        key = payload_sha256(
            {
                "relation": relation,
                "sides": [left.model_dump(mode="json"), right.model_dump(mode="json")],
            }
        )
        pairs[key] = ConflictPairV3(
            pair_id="pair-" + key, kind=kind, relation=relation, left=left, right=right
        )
    return tuple(pairs[key] for key in sorted(pairs))
