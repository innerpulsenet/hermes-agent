import math

import pytest

from agent.delegation_policy import normalize_delegation_override, resolve_delegation_policy


def test_normalize_accepts_allowed_values_and_serializes_timeout():
    override = normalize_delegation_override({
        "provider": " openrouter ", "model": " gpt-test ",
        "max_concurrent_children": 2, "max_iterations": 7,
        "child_timeout_seconds": 0,
    })
    assert override == {
        "provider": "openrouter", "model": "gpt-test",
        "max_concurrent_children": 2, "max_iterations": 7,
        "child_timeout_seconds": 0,
    }


def test_normalize_rejects_invalid_shape_and_partial_route():
    with pytest.raises(ValueError, match="unknown"):
        normalize_delegation_override({"wat": 1})
    with pytest.raises(ValueError, match="provider and model"):
        normalize_delegation_override({"provider": "p"})
    with pytest.raises(ValueError, match="boolean"):
        normalize_delegation_override({"max_iterations": True})
    with pytest.raises(ValueError, match="finite"):
        normalize_delegation_override({"child_timeout_seconds": math.inf})


def test_resolve_override_over_current_profile_and_reports_inherited():
    result = resolve_delegation_policy(
        {"provider": "p", "model": "m", "max_iterations": 7},
        {"provider": "old-p", "model": "old-m", "max_iterations": 3,
         "max_concurrent_children": 4, "child_timeout_seconds": 12},
    )
    assert result["override"] == {"provider": "p", "model": "m", "max_iterations": 7}
    assert result["effective"] == {
        "provider": "p", "model": "m", "max_iterations": 7,
        "max_concurrent_children": 4, "child_timeout_seconds": 12, "reasoning_effort": None,
    }
    assert result["inherited"] == {
        "provider": False, "model": False, "max_iterations": False,
        "max_concurrent_children": True, "child_timeout_seconds": True, "reasoning_effort": True,
    }
    assert result["global_max_concurrent_children"] == 4


def test_resolve_caps_effective_concurrency_at_global_ceiling():
    result = resolve_delegation_policy(
        {"max_concurrent_children": 9},
        {"max_concurrent_children": 4},
    )
    assert result["override"]["max_concurrent_children"] == 9
    assert result["effective"]["max_concurrent_children"] == 4


def test_reset_is_represented_by_none_and_defaults_are_stable():
    result = resolve_delegation_policy(None, {})
    assert result["override"] is None
    assert result["effective"]["max_concurrent_children"] == 10
    assert result["effective"]["max_iterations"] == 250
    assert result["effective"]["child_timeout_seconds"] == 0


def test_reasoning_effort_accepts_levels_and_disable_words():
    from agent.delegation_policy import normalize_delegation_override
    assert normalize_delegation_override({"reasoning_effort": "High"}) == {"reasoning_effort": "high"}
    assert normalize_delegation_override({"reasoning_effort": "ultra"}) == {"reasoning_effort": "ultra"}
    # A YAML ``false`` must disable thinking, not read as "unset".
    assert normalize_delegation_override({"reasoning_effort": False}) == {"reasoning_effort": "false"}
    assert normalize_delegation_override({"reasoning_effort": "none"}) == {"reasoning_effort": "none"}


def test_reasoning_effort_rejects_unknown_levels():
    import pytest
    from agent.delegation_policy import normalize_delegation_override
    for bad in ("turbo", "", "  ", 3, None, True):
        with pytest.raises(ValueError):
            normalize_delegation_override({"reasoning_effort": bad})


def test_reasoning_effort_resolves_over_profile_config():
    from agent.delegation_policy import resolve_delegation_policy
    profile = {"reasoning_effort": "low", "max_concurrent_children": 4}
    inherited = resolve_delegation_policy(None, profile)
    assert inherited["effective"]["reasoning_effort"] == "low"
    assert inherited["inherited"]["reasoning_effort"] is True
    overridden = resolve_delegation_policy({"reasoning_effort": "max"}, profile)
    assert overridden["effective"]["reasoning_effort"] == "max"
    assert overridden["inherited"]["reasoning_effort"] is False
    # Nothing configured anywhere means inherit the parent's level.
    assert resolve_delegation_policy(None, {})["effective"]["reasoning_effort"] is None


def test_unparseable_profile_reasoning_effort_inherits_instead_of_raising():
    from agent.delegation_policy import resolve_delegation_policy
    resolved = resolve_delegation_policy(None, {"reasoning_effort": "turbo"})
    assert resolved["effective"]["reasoning_effort"] is None
