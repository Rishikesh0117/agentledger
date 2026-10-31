"""Agents: what proposes the next action.

Step 3 ships only a scripted fake agent, so the wrapper pipeline (and,
later, the policy engine) can be tested without any model API calls.
Step 4 adds a real agent that calls Muse Spark; it will implement the
same :class:`BaseAgent` protocol.

When native tool calling is unavailable, the model must answer with one
strict JSON object (``{"tool": ..., "args": {...}}``). Use
:func:`parse_action_json` to parse and validate that format.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ProposedAction(BaseModel):
    """One action the agent wants the pipeline to run."""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


def parse_action_json(text: str) -> ProposedAction:
    """Parse strict ``{"tool": ..., "args": {...}}`` JSON into an action.

    Raises ``ValueError`` (including Pydantic ``ValidationError``) when the
    text is not valid JSON, is not an object, or misses/mistypes fields.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"action is not valid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise ValueError("action JSON must be an object with 'tool' and 'args'")
    return ProposedAction.model_validate(obj)


class BaseAgent(Protocol):
    """Anything the pipeline can pull actions from, one at a time.

    Optional extras the pipeline honors when present:

    - ``observe(action, outcome)``: receives each action's outcome dict.
    - ``empty_proposal_is_call``: when True, a ``None`` return still
      records a ``model_call`` event, because the agent called the model
      and the model signaled completion.
    """

    name: str

    def propose(self) -> ProposedAction | None:
        """Return the next action, or None when the agent is finished."""
        ...


class ScriptedAgent:
    """Fake agent that replays a fixed list of actions, then finishes.

    Each entry may be a :class:`ProposedAction`, an args dict, or a strict
    JSON string (parsed with :func:`parse_action_json`).
    """

    def __init__(
        self,
        actions: list[ProposedAction | dict[str, Any] | str],
        name: str = "scripted-fake",
    ):
        self.name = name
        self._actions: list[ProposedAction] = [self._coerce(a) for a in actions]
        self._index = 0

    @staticmethod
    def _coerce(action: ProposedAction | dict[str, Any] | str) -> ProposedAction:
        if isinstance(action, ProposedAction):
            return action
        if isinstance(action, str):
            return parse_action_json(action)
        return ProposedAction.model_validate(action)

    def propose(self) -> ProposedAction | None:
        if self._index >= len(self._actions):
            return None
        action = self._actions[self._index]
        self._index += 1
        return action
