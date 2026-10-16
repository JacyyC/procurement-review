"""F pre-registration. No truth or production aggregation is read here."""

import hashlib
from pathlib import Path

from .business_v3_security import ProtectedOutputV3
from .eval_f_context import input_fingerprints, load_cases, stable_write, data_directory
from .eval_f_contracts import FEndpointConfig
from .gold_v3_contracts import payload_sha256

BACKEND = Path(__file__).resolve().parents[2]
LIVE_ROOT = BACKEND / ".local/v1-07f/live"
DATA_VERSION = "gold-set.v3.2"


def implementation_fingerprints():
    root = Path(__file__).parent
    names = {
        p.name
        for pattern in (
            "business_v3_*.py",
            "eval_f_*.py",
            "eval_attempt_*.py",
            "gold_v3*.py",
        )
        for p in root.glob(pattern)
    }
    names.update(
        (
            "chat_transport_v2.py",
            "chat_transport_v3.py",
            "model_attempt_v2.py",
            "model_attempt_projection.py",
            "model_wire_v2.py",
            "eval_v2_contracts.py",
        )
    )
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in sorted(names)
    }


def configuration(style="baseline", thinking="disabled"):
    return FEndpointConfig(
        endpoint="https://model.example.invalid/api/coding/v3/chat/completions",
        requested_model="deepseek-v4-flash",
        resolved_model="deepseek-v4-flash-ga-260731",
        mode="json_schema",
        thinking=thinking,
        temperature=0.0,
        single_attempt_seconds=90.0,
        prompt_style=style,
        protection_sha256=ProtectedOutputV3().fingerprint,
    )


def registration(diagnostic_predecessor=None, data_version="gold-set.v3.1"):
    dev = [c.case_id for c in load_cases(version=data_version)]
    validation = [c.case_id for c in load_cases("validation", data_version)]
    candidates = {}
    for name, style, thinking, concurrency, subset, repeats, dependency in (
        ("minimum", "baseline", "disabled", 1, dev[:6], [111], None),
        ("baseline", "baseline", "disabled", 1, dev[:12], [205], "minimum"),
        ("compact", "compact", "disabled", 1, dev[:12], [205], "baseline"),
        ("reasoning", "baseline", None, 1, dev[:12], [205], "baseline"),
        ("concurrency", "baseline", "disabled", 2, dev[:12], [205], "baseline"),
    ):
        candidates[name] = dict(
            configuration=configuration(style, thinking).model_dump(mode="json"),
            concurrency=concurrency,
            case_ids=subset,
            repeat_seeds=repeats,
            max_requests=len(subset) * len(repeats) * 2,
            max_attempts=1,
            dependency=dependency,
        )
    diagnostics = sum(c["max_requests"] for c in candidates.values())
    if diagnostic_predecessor:
        candidates["recovery"] = dict(
            configuration=configuration().model_dump(mode="json"),
            concurrency=1,
            case_ids=dev[:1],
            repeat_seeds=[401],
            max_requests=1,
            max_attempts=1,
            dependency=None,
        )
        diagnostics = 1
    for arm in ("baseline", "compact", "reasoning", "concurrency"):
        base = candidates[arm]
        for split, ids, dependency in (
            ("dev", dev, arm),
            ("validation", validation, "dev-" + arm),
        ):
            candidates[split + "-" + arm] = dict(
                configuration=base["configuration"],
                concurrency=base["concurrency"],
                case_ids=ids,
                repeat_seeds=[301, 302],
                max_requests=len(ids) * 8,
                max_attempts=2,
                dependency=dependency,
                selected_arm=arm,
            )
    return dict(
        contract="v1-07f-batch.v1",
        revision=17,
        diagnostic_predecessor=diagnostic_predecessor,
        stage="V1-07F",
        data=input_fingerprints(data_version),
        implementation=implementation_fingerprints(),
        gold_manifest_sha256=hashlib.sha256(
            (data_directory(data_version) / "manifest.json").read_bytes()
        ).hexdigest(),
        versions=[
            data_version,
            {
                "gold-set.v3.1": "v1-07-truth.v1.1",
                "gold-set.v3.2": "v1-07-truth.v1.2",
                "gold-set.v3.3": "v1-07-truth.v1.3",
                "gold-set.v3.4": "v1-07-truth.v1.4",
                "gold-set.v3.5": "v1-07-truth.v1.5",
                "gold-set.v3.6": "v1-07-truth.v1.6",
            }[data_version],
            "typed-business-values.v1",
            "finding-evidence.v1",
            "business-review.v3",
            "safe-egress.v2",
            "3.0.0",
        ],
        authorization="user-explicit-no-balance-limit",
        request_authorization="user_no_cumulative_request_limit_20260920",
        token_limit=None,
        cost_limit=None,
        prior=dict(
            c_requests=232,
            d_e_requests=0,
            known_input=761408,
            known_output=108517,
            c_unknown=14,
            historical_requests=[7857, 7871],
            historical_unknown=101,
            historical_inflight_upper=4,
            historical_unledgered=[125, 135],
        ),
        caps=dict(d_e_f=None, g=None, item=None, per_branch=2),
        reserve=dict(
            dev_requests=960,
            validation_requests=480,
            complete_requests=1440,
            diagnostics=None,
            registered_diagnostics=diagnostics,
        ),
        logical_deadline_seconds=120,
        batch_deadline_seconds=21600,
        diagnostic_dependency="completed prior experiment; failed quality cannot be selected but does not prevent comparing alternatives",
        diagnostic_attempt_policy="first-send screening only for all four arms; errors remain failures; full DEV/validation preserve two shared attempts per branch",
        hypotheses=dict(
            baseline="新精确合同语义与独立真值一致",
            compact="只压缩重复Schema和描述，先质量后延迟与Token",
            reasoning="仅移除thinking.disabled，测端点默认推理的质量和耗时",
            concurrency="只将同时执行Case上限由1改为2",
        ),
        candidates=candidates,
        dev=dev,
        validation=validation,
        repeats=[301, 302],
        stop_conditions=[
            "model_identity_drift",
            "security_boundary",
            "tool_call_forbidden",
            "authentication",
            "unsupported_request",
            "twelve_consecutive_transport_failures",
            "cancelled",
            "deadline",
            "configuration_changed",
        ],
    )


def prepare(path, diagnostic_predecessor=None, data_version="gold-set.v3.1"):
    value = registration(diagnostic_predecessor, data_version)
    stable_write(Path(path), value)
    return {
        "registration_sha256": payload_sha256(value),
        "network_calls": 0,
        "diagnostic_max": value["reserve"]["registered_diagnostics"],
        "complete_reserved": 1440,
    }


def fixed_registration(data_version, *, split="validation"):
    """Fixed settings; DEV confirmation requires an explicit post-selection request."""
    if split not in {"dev", "validation"}:
        raise ValueError("fixed evaluation requires DEV or validation")
    value = registration(data_version=data_version)
    name = split + "-concurrency"
    candidate = value["candidates"][name]
    candidate["dependency"] = None
    value.update(
        revision=18,
        fixed_selection="user_20260920_baseline_disabled_case_concurrency_2",
        selection_scope="settings fixed by user; repaired candidate quality pending fresh validation",
        stop_when_first_pass_unreachable=True,
        candidates={name: candidate},
        reserve=dict(
            dev_requests=candidate["max_requests"] if split == "dev" else 0,
            validation_requests=candidate["max_requests"]
            if split == "validation"
            else 0,
            complete_requests=candidate["max_requests"],
            diagnostics=None,
            registered_diagnostics=0,
        ),
        hypotheses=dict(validation="修复后的固定配置在新组满足原全部适用门槛"),
        diagnostic_dependency="completed comparisons and focused repair evidence; no repeated full DEV",
    )
    if split == "dev":
        value.update(
            confirmation_split="dev",
            selection_scope="user-requested full DEV confirmation of the fixed candidate",
            hypotheses=dict(dev="完整DEV复核修后候选；不重新配置选型或复用validation"),
            diagnostic_dependency="explicit post-validation user authorization for full DEV confirmation",
        )
    return value
