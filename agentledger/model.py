"""Real model calls for the agent (Muse Spark via the Meta Model API).

The client speaks plain HTTP POST with stdlib ``urllib`` only, so this step
adds no dependencies. It assumes a chat-completions-style JSON API::

    POST {api_base}/chat/completions
    {"model": ..., "messages": [{"role": ..., "content": ...}], "tools": [...]}

    -> {"choices": [{"message": {"content": ..., "tool_calls": [...]}}]}

``api_base`` and ``model`` come from :class:`Settings`, the key from the
``META_MODEL_API_KEY`` environment variable. If the hackathon's real
endpoint or shape differs, only :class:`ModelClient` needs to change.

Native tool calling is unconfirmed, so :class:`ModelAgent` supports both:
with ``native_tools=True`` it sends tool schemas and prefers ``tool_calls``;
otherwise (the default) it asks for one strict ``{"tool", "args"}`` JSON
object and parses it with :func:`parse_action_json`. The model finishes a
run by proposing the ``finish`` tool, which the agent converts to ``None``
so the pipeline never sees it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from agentledger.agent import ProposedAction, parse_action_json
from agentledger.config import Settings, get_api_key
from agentledger.tools import TOOLS, ToolSpec

FINISH_TOOL = "finish"


class ModelError(Exception):
    """The model call failed: no key, network error, bad response, bad action."""


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class ModelClient:
    """Minimal chat client for the Meta Model API (stdlib only)."""

    def __init__(self, model: str, api_key: str, api_base: str, timeout: int = 30):
        if not api_key:
            raise ModelError("missing API key: set META_MODEL_API_KEY in .env")
        self.model = model
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings, api_key: str | None = None) -> ModelClient:
        """Build a client for the agent model, reading the key from .env."""
        return cls(
            model=settings.agent_model,
            api_key=api_key or get_api_key() or "",
            api_base=settings.model_api_base,
            timeout=settings.model_timeout,
        )

    def chat(
        self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> ChatResult:
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = tools
        request = urllib.request.Request(
            f"{self._api_base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise ModelError(f"model API HTTP {e.code}: {e.reason}") from None
        except urllib.error.URLError as e:
            raise ModelError(f"model API unreachable: {e.reason}") from None
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ModelError(f"model API returned invalid JSON: {e}") from None
        return self._parse_response(payload)

    @staticmethod
    def _parse_response(payload: Any) -> ChatResult:
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise ModelError("model API response has no choices[0].message") from None
        result = ChatResult(content=message.get("content") or "")
        for call in message.get("tool_calls") or []:
            try:
                function = call["function"]
                name = function["name"]
                raw_args = function.get("arguments", "{}")
            except (KeyError, TypeError):
                raise ModelError("model API tool_call is missing function name") from None
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                raise ModelError(f"tool_call {name!r} has invalid JSON arguments") from None
            if not isinstance(args, dict):
                raise ModelError(f"tool_call {name!r} arguments must be an object") from None
            result.tool_calls.append(ToolCall(name=name, args=args))
        return result


def extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply (strips ``` fences)."""
    text = text.strip()
    if text.startswith("```"):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ModelError("model reply has no JSON object inside code fences")
        return text[start : end + 1]
    return text


def _tool_schemas(specs: dict[str, ToolSpec]) -> list[dict[str, Any]]:
    schemas = [
        {
            "name": spec.name,
            "description": f"{spec.description} (risk: {spec.risk})",
            "parameters": spec.args_model.model_json_schema(),
        }
        for spec in specs.values()
    ]
    schemas.append(
        {
            "name": FINISH_TOOL,
            "description": "Call when the task is complete. No further actions.",
            "parameters": {"type": "object", "properties": {}},
        }
    )
    return schemas


def _system_prompt(task: str, role: str, specs: dict[str, ToolSpec]) -> str:
    lines = [
        f"You are an AI agent with role {role!r}. Complete this task: {task}",
        "",
        "Reply with exactly one JSON object per step: {\"tool\": ..., \"args\": {...}}.",
        f"When the task is complete, reply with {{\"tool\": \"{FINISH_TOOL}\", \"args\": {{}}}}.",
        "Never add explanations outside the JSON object.",
        "",
        "Available tools:",
    ]
    for spec in specs.values():
        lines.append(f"- {spec.name} ({spec.risk} risk): {spec.description}")
    return "\n".join(lines)


class ModelAgent:
    """A :class:`BaseAgent` that calls Muse Spark for every proposal."""

    def __init__(
        self,
        task: str,
        role: str,
        client: ModelClient,
        specs: dict[str, ToolSpec] | None = None,
        native_tools: bool = False,
        name: str | None = None,
    ):
        self.name = name or client.model
        self._client = client
        self._specs = specs if specs is not None else TOOLS
        self._native_tools = native_tools
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": _system_prompt(task, role, self._specs)}
        ]

    @property
    def messages(self) -> list[dict[str, str]]:
        """Conversation so far (read-only copy, useful for tests/debug)."""
        return list(self._messages)

    def propose(self) -> ProposedAction | None:
        schemas = _tool_schemas(self._specs) if self._native_tools else None
        result = self._client.chat(self._messages, tools=schemas)
        self._messages.append({"role": "assistant", "content": result.content})
        if self._native_tools and result.tool_calls:
            call = result.tool_calls[0]
            if call.name == FINISH_TOOL:
                return None
            return ProposedAction(tool=call.name, args=call.args)
        try:
            action = parse_action_json(extract_json(result.content))
        except ValueError as e:
            raise ModelError(f"model did not return a valid action: {e}") from None
        if action.tool == FINISH_TOOL:
            return None
        return action

    def observe(self, action: ProposedAction, outcome: dict[str, Any]) -> None:
        """Record what the pipeline did with an action, for the next proposal."""
        self._messages.append(
            {
                "role": "user",
                "content": json.dumps({"action": action.model_dump(), "outcome": outcome}),
            }
        )
