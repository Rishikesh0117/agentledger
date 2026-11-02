"""Guardian interface: second-model review of risky actions.

From PLAN.md, the guardian sees the original task, the proposed action
with arguments, and the recent trace, and answers with a verdict
(``allow``, ``flag``, ``block``) plus a plain-language reason.

This module holds the verdict type, the :class:`Guardian` protocol, and
a scripted :class:`FakeGuardian` for testing the pipeline with no API
calls. The real model-backed guardian comes in a later substep.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from agentledger.agent import ProposedAction
from agentledger.ledger import Event


class GuardianVerdict(BaseModel):
    verdict: Literal["allow", "flag", "block"]
    reason: str = ""


@runtime_checkable
class Guardian(Protocol):
    """Reviews one proposed action against the task and recent trace."""

    def review(self, task: str, action: ProposedAction, trace: list[Event]) -> GuardianVerdict: ...


class FakeGuardian:
    """Scripted guardian: replays canned verdicts and records what it saw.

    Each entry may be a :class:`GuardianVerdict`, a ``{"verdict",
    "reason"}`` dict, or a ``(verdict, reason)`` tuple. Once the script
    is exhausted, further reviews default to ``allow``. Every call is
    appended to :attr:`calls` as a ``{"task", "action", "trace"}`` dict.
    """

    def __init__(self, verdicts: list[GuardianVerdict | dict[str, str] | tuple[str, str]] | None = None):
        self._verdicts: list[GuardianVerdict] = [self._coerce(v) for v in verdicts or []]
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _coerce(value: GuardianVerdict | dict[str, str] | tuple[str, str]) -> GuardianVerdict:
        if isinstance(value, GuardianVerdict):
            return value
        if isinstance(value, tuple):
            verdict, reason = value
            return GuardianVerdict.model_validate({"verdict": verdict, "reason": reason})
        return GuardianVerdict.model_validate(value)

    def review(self, task: str, action: ProposedAction, trace: list[Event]) -> GuardianVerdict:
        self.calls.append({"task": task, "action": action, "trace": list(trace)})
        if self._index < len(self._verdicts):
            verdict = self._verdicts[self._index]
            self._index += 1
            return verdict
        return GuardianVerdict(verdict="allow", reason="fake guardian script exhausted: default allow")


class AllowAllGuardian:
    """Allow everything. Pipeline default; the approval gate stays the backstop."""

    def review(self, task: str, action: ProposedAction, trace: list[Event]) -> GuardianVerdict:
        return GuardianVerdict(verdict="allow", reason="no guardian wired up: default allow")
