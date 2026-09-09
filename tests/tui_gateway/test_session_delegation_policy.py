from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tui_gateway.server as server


@contextlib.contextmanager
def _db_scope(db):
    yield db


def _call(method: str, session_id: str, **params):
    return server._methods[method]("rid", {"session_id": session_id, **params})


def _session(key: str = "stored-a", *, agent=None):
    return {"session_key": key, "agent": agent, "transport": None}


def test_set_updates_only_target_lazy_session_without_creating_a_draft_row():
    db = MagicMock()
    db.get_session.return_value = None
    first = _session("stored-a")
    second = _session("stored-b")
    config = {"provider": "openrouter", "model": "m", "max_iterations": 7}
    with patch.dict(server._sessions, {"runtime-a": first, "runtime-b": second}, clear=True), \
            patch.object(server, "_session_db", return_value=_db_scope(db)), \
            patch.object(server, "_delegation_global_config", return_value={"max_iterations": 500}):
        response = _call("session.delegation.set", "runtime-a", config=config)

    assert response["result"]["override"] == config
    assert response["result"]["inherited"]["max_iterations"] is False
    assert first["delegation_override"] == config
    assert "delegation_override" not in second
    db.patch_session_model_config.assert_not_called()


def test_set_persists_before_mutating_live_state_and_rolls_back_on_failure():
    class BrokenDB:
        def get_session(self, _key):
            return {"id": "stored-a"}

        def patch_session_model_config(self, _key, _patch):
            raise OSError("disk full")

    old = {"max_iterations": 3}
    agent = SimpleNamespace(
        _session_delegation_override=dict(old),
        _session_init_model_config={"delegation": dict(old)},
    )
    session = _session(agent=agent)
    session["delegation_override"] = dict(old)
    with patch.dict(server._sessions, {"runtime-a": session}, clear=True), \
            patch.object(server, "_session_db", return_value=_db_scope(BrokenDB())):
        response = _call(
            "session.delegation.set", "runtime-a", config={"max_iterations": 9}
        )

    assert response["error"]["code"] == 5000
    assert session["delegation_override"] == old
    assert agent._session_delegation_override == old
    assert agent._session_init_model_config["delegation"] == old


def test_reset_deletes_persisted_and_live_override():
    db = MagicMock()
    db.get_session.return_value = {"id": "stored-a"}
    agent = SimpleNamespace(
        _session_delegation_override={"max_iterations": 3},
        _session_init_model_config={"delegation": {"max_iterations": 3}, "keep": True},
    )
    session = _session(agent=agent)
    session["delegation_override"] = {"max_iterations": 3}
    with patch.dict(server._sessions, {"runtime-a": session}, clear=True), \
            patch.object(server, "_session_db", return_value=_db_scope(db)), \
            patch.object(server, "_delegation_global_config", return_value={}):
        response = _call("session.delegation.reset", "runtime-a")

    assert response["result"]["override"] is None
    db.patch_session_model_config.assert_called_once_with("stored-a", {"delegation": None})
    assert "delegation_override" not in session
    assert agent._session_delegation_override is None
    assert agent._session_init_model_config == {"keep": True}


def test_set_rejects_active_child_without_writing_or_mutating():
    import tools.delegate_tool_registry as registry

    db = MagicMock()
    transport = object()
    session = _session()
    session["transport"] = transport
    record = {
        "owner_session_id": "runtime-a",
        "owner_transport": transport,
        "owner_session_record": session,
    }
    with patch.dict(server._sessions, {"runtime-a": session}, clear=True), \
            patch.dict(registry._active_subagents, {"child": record}, clear=True), \
            patch.object(server, "_session_db", return_value=_db_scope(db)):
        response = _call(
            "session.delegation.set", "runtime-a", config={"max_iterations": 9}
        )

    assert response["error"]["code"] == 4090
    assert "delegation_override" not in session
    db.patch_session_model_config.assert_not_called()


def test_set_rejects_child_attached_during_construction():
    db = MagicMock()
    agent = SimpleNamespace(
        _active_children=[object()],
        _active_children_lock=threading.Lock(),
    )
    session = _session(agent=agent)
    with patch.dict(server._sessions, {"runtime-a": session}, clear=True), \
            patch.object(server, "_session_db", return_value=_db_scope(db)):
        response = _call(
            "session.delegation.set", "runtime-a", config={"max_iterations": 9}
        )

    assert response["error"]["code"] == 4090
    db.patch_session_model_config.assert_not_called()


def test_reset_rejects_live_background_delegation_between_child_registry_events():
    db = MagicMock()
    session = _session()
    with patch.dict(server._sessions, {"runtime-a": session}, clear=True), \
            patch.object(server, "_session_db", return_value=_db_scope(db)), \
            patch("tools.async_delegation.has_live_for_session", return_value=True):
        response = _call("session.delegation.reset", "runtime-a")

    assert response["error"]["code"] == 4090
    db.patch_session_model_config.assert_not_called()


def test_attach_built_agent_reconciles_policy_set_during_build():
    override = {"max_iterations": 9, "child_timeout_seconds": 0}
    session = _session()
    session["delegation_override"] = override
    agent = SimpleNamespace(
        _session_delegation_override=None,
        _session_init_model_config={"keep": True},
    )

    with patch.object(server, "_session_todo_state"):
        server._attach_built_agent(session, agent)

    assert agent._session_delegation_override == override
    assert agent._session_delegation_override is not override
    assert agent._session_init_model_config["delegation"] == override


def test_resume_restores_delegation_even_when_model_follows_profile():
    override = {"provider": "openrouter", "model": "m", "max_iterations": 9}
    row = {
        "title": "Bot Chat",
        "model_config": {"follow_profile_config": True, "delegation": override},
    }

    restored = server._stored_session_runtime_overrides(row)

    assert restored["delegation_override"] == override


def test_desktop_branch_inherits_parent_policy_from_persisted_row():
    override = {"provider": "openrouter", "model": "m", "max_iterations": 9}
    db = MagicMock()
    db.get_session.return_value = {"model_config": {"delegation": override}}
    child = _session("child")

    inherited = server._branch_delegation_override(child, "parent", db)

    assert inherited == override
    assert inherited is not override
    assert child["delegation_override"] == override


def test_desktop_branch_prefers_live_parent_policy():
    db = MagicMock()
    live_override = {"max_iterations": 11}
    parent = _session("parent")
    parent["delegation_override"] = live_override
    child = _session("child")

    with patch.dict(server._sessions, {"parent-runtime": parent}, clear=True):
        inherited = server._branch_delegation_override(child, "parent", db)

    assert inherited == live_override
    assert inherited is not live_override
    db.get_session.assert_not_called()


def test_rpc_global_policy_uses_runtime_normalized_limits():
    with patch("hermes_cli.config.load_config_readonly", return_value={
            "delegation": {"max_concurrent_children": "bad", "child_timeout_seconds": 1}
         }), patch("tools.delegate_tool_config._get_max_concurrent_children", return_value=6), \
            patch("tools.delegate_tool_config._get_child_timeout", return_value=30):
        config = server._delegation_global_config()

    assert config["max_concurrent_children"] == 6
    assert config["child_timeout_seconds"] == 30
