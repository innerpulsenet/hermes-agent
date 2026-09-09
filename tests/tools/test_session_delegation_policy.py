from types import SimpleNamespace
from unittest.mock import patch

from tools.delegate_tool_child_run import _effective_child_timeout
from tools.delegate_tool import _resolve_session_delegation_policy
from tools.delegate_tool import delegate_task
from tools.delegate_tool_dispatch import _background_scheduler_capacity


def test_explicit_zero_timeout_disables_profile_timeout():
    child = SimpleNamespace(_delegation_timeout=None)
    with patch("tools.delegate_tool._get_child_timeout", return_value=60):
        assert _effective_child_timeout(child) is None


def test_legacy_child_without_snapshot_uses_profile_timeout():
    child = SimpleNamespace()
    with patch("tools.delegate_tool._get_child_timeout", return_value=60):
        assert _effective_child_timeout(child) == 60


def test_runtime_policy_preserves_normalized_profile_knob_semantics():
    parent = SimpleNamespace(_session_delegation_override={"max_iterations": 7})
    raw = {"max_concurrent_children": "bad", "child_timeout_seconds": 1}
    with patch("tools.delegate_tool._get_max_concurrent_children", return_value=6), \
            patch("tools.delegate_tool._get_child_timeout", return_value=30):
        policy = _resolve_session_delegation_policy(parent, raw)

    assert policy["effective"]["max_concurrent_children"] == 6
    assert policy["effective"]["child_timeout_seconds"] == 30
    assert policy["effective"]["max_iterations"] == 7


def test_delegate_task_applies_one_session_policy_snapshot_end_to_end():
    parent = SimpleNamespace(
        _session_delegation_override={
            "provider": "session-provider", "model": "session-model",
            "max_concurrent_children": 3, "max_iterations": 7,
            "child_timeout_seconds": 0,
        },
        _delegate_depth=0,
    )
    captured = {}

    def resolve_route(config, _parent):
        captured["routing"] = config
        return {
            "provider": config["provider"], "model": config["model"],
            "base_url": None, "api_key": "test", "api_mode": "chat_completions",
        }

    def build_children(task_list, task_schemas, creds, **kwargs):
        captured["build"] = kwargs
        return [(0, task_list[0], object())], None

    def run_batch(batch, background):
        captured["batch"] = batch
        captured["background"] = background
        return "ok"

    with patch("tools.delegate_tool.is_spawn_paused", return_value=False), \
            patch("tools.delegate_tool._get_max_spawn_depth", return_value=1), \
            patch("tools.delegate_tool._load_config", return_value={
                "provider": "global-provider", "model": "global-model",
                "max_concurrent_children": 8, "max_iterations": 250,
                "child_timeout_seconds": 60,
            }), patch("tools.delegate_tool._get_max_concurrent_children", return_value=8), \
            patch("tools.delegate_tool._get_child_timeout", return_value=60), \
            patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=resolve_route), \
            patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], [])), \
            patch("tools.delegate_tool._announce_batch"), \
            patch("tools.delegate_tool._capture_origin", return_value=("", "", None, None)), \
            patch("tools.delegate_tool._build_children", side_effect=build_children), \
            patch("tools.delegate_tool._run_batch", side_effect=run_batch):
        result = delegate_task(tasks=[{"goal": "test"}], parent_agent=parent)

    assert result == "ok"
    assert captured["routing"]["provider"] == "session-provider"
    assert captured["routing"]["model"] == "session-model"
    assert captured["build"]["max_iterations"] == 7
    assert captured["build"]["delegation_timeout"] == 0
    assert captured["batch"].max_children == 3


def test_session_call_limit_does_not_shrink_shared_background_scheduler():
    batch = SimpleNamespace(max_children=2)
    with patch("tools.delegate_tool._get_max_async_children", return_value=8):
        assert _background_scheduler_capacity(batch) == 8
