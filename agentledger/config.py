"""Configuration loading: JSON files for settings and roles, `.env` for secrets.

Model names and other non-secret settings live in ``config/settings.json``.
Role permissions live in ``config/roles.json``. API keys live only in the
environment (via a git-ignored ``.env`` file), never in code or config files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from agentledger.tools import TOOLS

API_KEY_ENV_VAR = "META_MODEL_API_KEY"


class Settings(BaseModel):
    """Model names and the API base URL have no defaults.

    They must come from the config file. Everything else still has a
    fallback default. API keys are never part of settings: they live
    only in the environment (via ``.env``).
    """

    agent_model: str
    guardian_model: str
    model_api_base: str
    email_allowlist: list[str] = Field(default_factory=lambda: ["example.com"])
    max_tool_calls: int = 10
    db_path: str = "agentledger.db"
    model_timeout: int = 30


class Roles(BaseModel):
    roles: dict[str, list[str]]

    def tools_for(self, role: str) -> list[str]:
        try:
            return self.roles[role]
        except KeyError:
            raise ValueError(f"unknown role: {role!r}") from None


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal `.env` reader (stdlib only): KEY=VALUE lines, `#` comments.

    Never overrides a variable that is already set in the environment.
    A missing file is fine: secrets may come from the real environment.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key(env_var: str = API_KEY_ENV_VAR) -> str | None:
    """API key from the environment, or None when not configured."""
    return os.environ.get(env_var) or None


def load_settings(
    path: str | Path = "config/settings.json",
    dotenv_path: str | Path | None = ".env",
) -> Settings:
    """Load settings from JSON. Missing keys fall back to defaults, except
    model names and the base URL, which are required.

    Also loads ``.env`` first (unless ``dotenv_path`` is None) so that
    :func:`get_api_key` works right after this call.
    """
    if dotenv_path is not None:
        load_dotenv(dotenv_path)
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} is not valid JSON: {e}") from None
    try:
        return Settings.model_validate(data)
    except ValidationError as e:
        details = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        raise ValueError(f"invalid settings in {path}: {details}") from None


def load_roles(path: str | Path = "config/roles.json") -> Roles:
    """Load roles and fail fast if a role lists a tool that does not exist."""
    with open(path, encoding="utf-8") as f:
        roles = Roles.model_validate(json.load(f))
    for role, tools in roles.roles.items():
        for tool in tools:
            if tool not in TOOLS:
                raise ValueError(f"role {role!r} lists unknown tool: {tool!r}")
    return roles
