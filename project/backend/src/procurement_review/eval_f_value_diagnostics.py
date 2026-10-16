"""Closed typed-error paths; never retain model values or arbitrary JSON keys."""

import json

from pydantic import ValidationError

from .business_v3_types import VALUE_MODELS
from .model_wire_v2 import WireIssue, strict_json

ERRORS = frozenset(
    {
        "missing",
        "string_type",
        "string_too_long",
        "string_too_short",
        "bool_type",
        "int_type",
        "literal_error",
        "greater_than_equal",
        "less_than_equal",
        "too_long",
        "tuple_type",
        "extra_forbidden",
        "model_type",
        "value_error",
    }
)


def typed_error_codes():
    """Finite vocabulary shared by diagnostics and F-only restart admission."""
    return frozenset(
        "typed_" + duty.lower().replace("-", "_") + "_" + field + "_" + error
        for models in VALUE_MODELS.values()
        for duty, model in models.items()
        for field in (*model.model_fields, "normalized_value")
        for error in ERRORS | {"invalid_json", "invalid"}
    )


def validate_typed_diagnostics(raw, worker):
    """Diagnostic precheck only; existing wire, semantics and evidence still run."""
    if not isinstance(raw, dict) or not isinstance(raw.get("findings"), list):
        return
    for item in raw["findings"]:
        if not isinstance(item, dict) or item.get("status") == "unknown":
            continue
        duty = item.get("obligation_id")
        if not isinstance(duty, str) or duty not in VALUE_MODELS[worker]:
            continue
        model = VALUE_MODELS[worker][duty]
        encoded = item.get("normalized_value")
        field, error = "normalized_value", None
        if not isinstance(encoded, str):
            error = "string_type"
        elif len(encoded) > 256:
            error = "string_too_long"
        else:
            try:
                value = strict_json(encoded)
            except (ValueError, TypeError):
                error = "invalid_json"
            else:
                try:
                    model.model_validate_json(json.dumps(value), strict=True)
                except ValidationError as exc:
                    entry = exc.errors(
                        include_input=False, include_context=False, include_url=False
                    )[0]
                    loc = entry["loc"]
                    if loc and loc[0] in model.model_fields:
                        field = loc[0]
                    error = entry["type"] if entry["type"] in ERRORS else "invalid"
        if error:
            raise WireIssue(
                "typed_" + duty.lower().replace("-", "_") + "_" + field + "_" + error,
                correctable=True,
                fields=("normalized_value",),
            )
