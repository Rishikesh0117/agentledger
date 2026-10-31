"""Config tests: settings, roles, and .env secret loading."""

import json
from pathlib import Path

import pytest

from agentledger import (
    API_KEY_ENV_VAR,
    get_api_key,
    load_dotenv,
    load_roles,
    load_settings,
)


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return str(path)


def test_load_settings_full_file(tmp_path):
    path = _write(
        tmp_path / "settings.json",
        {
            "agent_model": "model-a",
            "guardian_model": "model-b",
            "model_api_base": "https://models.test/v1",
            "email_allowlist": ["acme.test"],
            "max_tool_calls": 3,
            "db_path": "custom.db",
        },
    )
    settings = load_settings(path, dotenv_path=None)
    assert settings.agent_model == "model-a"
    assert settings.guardian_model == "model-b"
    assert settings.model_api_base == "https://models.test/v1"
    assert settings.email_allowlist == ["acme.test"]
    assert settings.max_tool_calls == 3
    assert settings.db_path == "custom.db"


def test_load_settings_partial_file_fills_defaults(tmp_path):
    settings = load_settings(
        _write(
            tmp_path / "s.json",
            {"agent_model": "x", "guardian_model": "y", "model_api_base": "https://z.test"},
        ),
        dotenv_path=None,
    )
    assert settings.agent_model == "x"
    assert settings.guardian_model == "y"
    assert settings.model_api_base == "https://z.test"
    assert settings.email_allowlist == ["example.com"]
    assert settings.max_tool_calls == 10
    assert settings.db_path == "agentledger.db"
    assert settings.model_timeout == 30


def test_load_settings_missing_file():
    with pytest.raises(FileNotFoundError):
        load_settings("does/not/exist.json", dotenv_path=None)


def test_load_settings_missing_required_values_fails_clearly(tmp_path):
    path = _write(tmp_path / "s.json", {"agent_model": "x"})
    with pytest.raises(ValueError) as exc_info:
        load_settings(path, dotenv_path=None)
    message = str(exc_info.value)
    assert "guardian_model" in message and "model_api_base" in message
    assert "s.json" in message


def test_shipped_roles_match_plan():
    roles = load_roles("config/roles.json")
    assert roles.tools_for("reader") == ["read_document", "lookup_record"]
    assert roles.tools_for("operator") == [
        "read_document",
        "lookup_record",
        "draft_message",
        "send_email",
        "update_status",
    ]


def test_shipped_settings_load():
    settings = load_settings("config/settings.json", dotenv_path=None)
    assert settings.email_allowlist == ["example.com"]


def test_load_roles_rejects_unknown_tool(tmp_path):
    path = _write(tmp_path / "r.json", {"roles": {"reader": ["delete_everything"]}})
    with pytest.raises(ValueError, match="unknown tool"):
        load_roles(path)


def test_unknown_role_rejected(tmp_path):
    roles = load_roles(_write(tmp_path / "r.json", {"roles": {"reader": []}}))
    with pytest.raises(ValueError, match="unknown role"):
        roles.tools_for("admin")


def test_dotenv_provides_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n\nMETA_MODEL_API_KEY=test-key-123\n", encoding="utf-8"
    )
    load_dotenv(env_file)
    assert get_api_key() == "test-key-123"


def test_dotenv_never_overrides_real_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "real-env-wins")
    env_file = tmp_path / ".env"
    env_file.write_text("META_MODEL_API_KEY=from-file\n", encoding="utf-8")
    load_dotenv(env_file)
    assert get_api_key() == "real-env-wins"


def test_dotenv_missing_file_is_fine(tmp_path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    load_dotenv(tmp_path / "nope.env")
    assert get_api_key() is None


def test_dotenv_example_has_placeholders_only():
    for line in Path(".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition("=")
        assert key and value == "", f".env.example must not carry values: {stripped}"
