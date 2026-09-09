"""Validation and resolution for session-scoped delegation policy."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

POLICY_KEYS = frozenset({
    "provider", "model", "max_concurrent_children", "max_iterations", "child_timeout_seconds",
    "reasoning_effort",
})
_STRING_KEYS = frozenset({"provider", "model"})
_INT_KEYS = frozenset({"max_concurrent_children", "max_iterations"})
# "none"/"false"/"disabled" all mean no thinking (parse_reasoning_effort's own vocabulary);
# a YAML ``false`` in profile config means the same and is coerced to "false" below.
_EFFORT_DISABLED = frozenset({"none", "false", "disabled"})


def _effort_levels() -> tuple[str, ...]:
    from hermes_constants import VALID_REASONING_EFFORTS
    return VALID_REASONING_EFFORTS


def _normalize_effort(raw: Any) -> str:
    """Canonical lowercase level, or a disable word. Raises ValueError on anything else."""
    if raw is False:
        return "false"
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("delegation reasoning_effort must be a nonempty string")
    value = raw.strip().lower()
    if value in _EFFORT_DISABLED or value in _effort_levels():
        return value
    allowed = ", ".join((*_effort_levels(), *sorted(_EFFORT_DISABLED)))
    raise ValueError(f"delegation reasoning_effort must be one of: {allowed}")

# These are deliberately local defaults: global config is consulted by callers and this module is usable
# without importing the CLI/config loader.
DEFAULT_POLICY = {
    "provider": None,
    "model": None,
    "max_concurrent_children": 10,
    "max_iterations": 250,
    "child_timeout_seconds": 0,
    # None = inherit the parent's level, which is what delegation has always done
    # when profile config says nothing.
    "reasoning_effort": None,
}


def normalize_delegation_override(value: Any) -> dict[str, Any] | None:
    """Return a copied, canonical override; ``None`` means reset/delete."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("delegation config must be an object")
    unknown = set(value) - POLICY_KEYS
    if unknown:
        raise ValueError(f"unknown delegation keys: {', '.join(sorted(unknown))}")
    result: dict[str, Any] = {}
    for key, raw in value.items():
        if key == "reasoning_effort":
            result[key] = _normalize_effort(raw)
        elif key in _STRING_KEYS:
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"delegation {key} must be a nonempty string")
            result[key] = raw.strip()
        elif key in _INT_KEYS:
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
                raise ValueError(f"delegation {key} must be an integer >= 1, not a boolean")
            result[key] = raw
        else:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError("delegation child_timeout_seconds must be a number")
            if not math.isfinite(raw) or raw < 0:
                raise ValueError("delegation child_timeout_seconds must be finite and >= 0")
            result[key] = raw
    if ("provider" in result) != ("model" in result):
        raise ValueError("delegation provider and model must be supplied together")
    return result


def _global_policy(global_config: Any) -> dict[str, Any]:
    source = global_config if isinstance(global_config, dict) else {}
    result = dict(DEFAULT_POLICY)
    for key in POLICY_KEYS:
        if key not in source:
            continue
        raw = source[key]
        if key == "reasoning_effort":
            # Profile config is best effort: an unparseable level inherits rather than raising,
            # matching how delegate_tool_config already treats a bad delegation.reasoning_effort.
            try:
                result[key] = _normalize_effort(raw)
            except ValueError:
                pass
        elif key in _STRING_KEYS:
            if isinstance(raw, str) and raw.strip():
                result[key] = raw.strip()
        elif key in _INT_KEYS:
            if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1:
                result[key] = raw
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw) and raw >= 0:
            result[key] = raw
    result["child_timeout_seconds"] = result["child_timeout_seconds"] or 0
    return result


def resolve_delegation_policy(override: Any, global_config: Any) -> dict[str, Any]:
    """Resolve an override over current profile config for an RPC-safe response."""
    normalized = normalize_delegation_override(override)
    global_policy = _global_policy(global_config)
    effective = dict(global_policy)
    if normalized:
        effective.update(normalized)
    effective["max_concurrent_children"] = min(
        effective["max_concurrent_children"], global_policy["max_concurrent_children"]
    )
    return {
        "override": deepcopy(normalized),
        "effective": effective,
        "inherited": {key: not normalized or key not in normalized for key in POLICY_KEYS},
        "global_max_concurrent_children": global_policy["max_concurrent_children"],
    }
