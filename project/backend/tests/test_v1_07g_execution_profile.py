"""V1-07G execution profiles: the opt-in real mode and its controlled failures.

The workbench has two profiles and no third: the deterministic synthetic line (the
default, which needs no credential) and the precise real line, which is only usable when
its endpoint, identities, credential and an explicit deadline are all configured.  A real
profile that is missing any of them is a recorded failure — never a silent fall back to
the synthetic line, and never a raised limit: the product's 30-second default is not
silently replaced by the registered 120-second ceiling.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from procurement_review.api import SecurityRuntime, create_app  # noqa: E402
from procurement_review.execution_profile import (  # noqa: E402
    DEFAULT_DEADLINE_SECONDS,
    MAX_ACCEPTED_DEADLINE_SECONDS,
    RealProfile,
    RealProfileProblem,
    SyntheticProfile,
    execution_mode_value,
    execution_profile,
)
from v1_03_support import create_isolated_database, drop_database  # noqa: E402

MODE_ENV = "REVIEW_EXECUTION_MODE"
PROFILE_ENV = (
    "REVIEW_MODEL_ENDPOINT",
    "REVIEW_MODEL_REQUESTED",
    "REVIEW_MODEL_RESOLVED",
    "REVIEW_MODEL_API_KEY",
    "REVIEW_MODEL_THINKING",
    "REVIEW_GENERATION_DEADLINE_SECONDS",
)


@pytest.fixture()
def profile_environment():
    """An environment with no profile settings of its own, restored afterwards."""

    saved = {name: os.environ.get(name) for name in (MODE_ENV, *PROFILE_ENV)}
    for name in saved:
        os.environ.pop(name, None)
    try:
        yield os.environ
    finally:
        for name, value in saved.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value


REAL_SETTINGS = {
    "REVIEW_MODEL_ENDPOINT": "https://example.invalid/v3/chat/completions",
    "REVIEW_MODEL_REQUESTED": "deepseek-v4.1-flash",
    "REVIEW_MODEL_RESOLVED": "deepseek-v4-1-flash",
    "REVIEW_MODEL_API_KEY": "local-test-credential-not-a-real-key",
    "REVIEW_GENERATION_DEADLINE_SECONDS": "120",
}


def test_the_default_profile_is_the_synthetic_line(profile_environment):
    profile = execution_profile()
    assert isinstance(profile, SyntheticProfile)
    assert profile.generation_deadline_seconds == DEFAULT_DEADLINE_SECONDS
    assert execution_mode_value() == "test_only"


def test_a_half_configured_real_profile_names_what_is_missing(profile_environment):
    profile_environment[MODE_ENV] = "real"
    for name in ("REVIEW_MODEL_ENDPOINT", "REVIEW_MODEL_API_KEY"):
        profile_environment[name] = REAL_SETTINGS[name]
    profile = execution_profile()
    assert isinstance(profile, RealProfileProblem)
    assert profile.reason == "real_mode_not_configured"
    assert "REVIEW_MODEL_REQUESTED" in profile.detail


def test_the_deadline_is_never_raised_silently(profile_environment):
    profile_environment[MODE_ENV] = "real"
    profile_environment.update(REAL_SETTINGS)
    profile_environment["REVIEW_GENERATION_DEADLINE_SECONDS"] = "600"
    profile = execution_profile()
    assert isinstance(profile, RealProfileProblem)
    assert "deadline" in profile.detail

    # The accepted V1-08 candidates (120, 180) and any shorter cap stay usable: the
    # shorter limit wins and the accepted ceiling is never widened.
    for value in ("90", "120", "180"):
        profile_environment["REVIEW_GENERATION_DEADLINE_SECONDS"] = value
        profile = execution_profile()
        assert isinstance(profile, RealProfile)
        assert profile.generation_deadline_seconds == float(value)
    assert MAX_ACCEPTED_DEADLINE_SECONDS == 180.0

    # 240 seconds is a boundary that needs its own decision and is refused here.
    profile_environment["REVIEW_GENERATION_DEADLINE_SECONDS"] = "240"
    assert isinstance(execution_profile(), RealProfileProblem)


def test_an_unusable_thinking_value_is_refused_not_silently_disabled(
    profile_environment,
):
    profile_environment[MODE_ENV] = "real"
    profile_environment.update(REAL_SETTINGS)
    # Not stated: the historical, explicitly disabled configuration.
    assert execution_profile().thinking == "disabled"
    for value in ("disabled", "enabled"):
        profile_environment["REVIEW_MODEL_THINKING"] = value
        profile = execution_profile()
        assert isinstance(profile, RealProfile)
        assert profile.thinking == value
        assert profile.endpoint_config().thinking == value
    profile_environment["REVIEW_MODEL_THINKING"] = "yes"
    profile = execution_profile()
    assert isinstance(profile, RealProfileProblem)
    assert "REVIEW_MODEL_THINKING" in profile.detail


def test_a_complete_real_profile_builds_the_precise_transport_without_the_credential(
    profile_environment,
):
    profile_environment[MODE_ENV] = "real"
    profile_environment.update(REAL_SETTINGS)
    profile = execution_profile()
    assert isinstance(profile, RealProfile)
    config = profile.endpoint_config()
    assert config.requested_model == "deepseek-v4.1-flash"
    assert config.resolved_model == "deepseek-v4-1-flash"
    assert config.mode == "json_schema" and config.thinking == "disabled"
    assert config.temperature == 0.0
    # The profile's own fingerprint and repr carry no credential value.
    secret = REAL_SETTINGS["REVIEW_MODEL_API_KEY"]
    assert secret not in profile.fingerprint()
    assert secret not in repr(profile)


def test_an_unusable_real_profile_fails_closed_instead_of_falling_back(
    profile_environment,
):
    database_url, database = create_isolated_database()
    try:
        profile_environment[MODE_ENV] = "real"
        profile_environment["REVIEW_MODEL_REQUESTED"] = REAL_SETTINGS[
            "REVIEW_MODEL_REQUESTED"
        ]
        from procurement_review.review_store import ReviewStore

        store = ReviewStore.from_url(database_url)
        store.migrate()
        runtime = SecurityRuntime.build(store)
        assert runtime.failure == "real_mode_not_configured"
        assert runtime.service is None
        from fastapi.testclient import TestClient

        application = create_app(runtime=runtime)
        with TestClient(application, base_url="http://127.0.0.1") as client:
            health = client.get("/api/v2/health")
            assert health.status_code == 200
            assert health.json()["execution_mode"] == "real"
            assert health.json()["security_store"] == "not_configured"
            refused = client.post(
                "/api/v2/reviews",
                headers={
                    "Authorization": "Bearer not-a-real-token",
                    "Idempotency-Key": "g-profile",
                },
                json={
                    "api_version": "review-api.v2",
                    "case_id": "CASE-ABSENT",
                    "case_version": "case.v1",
                },
            )
            assert refused.status_code == 503
            assert refused.json()["error"]["code"] == "security_not_configured"
        store.close()
    finally:
        drop_database(database)
