"""Policy engine: pure rules, no DB, model, or tool calls.

Rules run in order; the first match decides:

1. unknown tool -> block
2. tool not allowed for the role -> block (unknown roles allow nothing)
3. ``send_email`` to a domain outside the allowlist -> block
   (a missing or malformed recipient blocks too)
4. tool call cap reached -> block (possible loop)
5. high-risk tool -> require human approval
6. anything else -> allow

``ROLE_TOOLS`` mirrors ``config/roles.json``; a test fails if they drift
apart. Email domains match exactly (case-insensitive): subdomains of an
allowed domain are not covered.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from agentledger.agent import ProposedAction
from agentledger.config import Settings
from agentledger.tools import TOOLS

_READER_TOOLS = frozenset({"read_document", "lookup_record"})
ROLE_TOOLS: dict[str, frozenset[str]] = {
    "reader": _READER_TOOLS,
    "operator": _READER_TOOLS | {"draft_message", "send_email", "update_status"},
}


class PolicyDecision(BaseModel):
    outcome: Literal["allow", "require_approval", "block"]
    reason: str = ""
    rule: str | None = None  # which rule fired; None when allowed by default


def _domain_of(recipient: Any) -> str | None:
    """Lowercased domain of a plain address, or None when malformed."""
    if not isinstance(recipient, str):
        return None
    local, sep, domain = recipient.strip().rpartition("@")
    if not sep or not local or not domain:
        return None
    return domain.lower()


def evaluate(
    action: ProposedAction, agent_role: str, settings: Settings, tool_call_count: int = 0
) -> PolicyDecision:
    """Decide an action against the policy rules, in order."""
    if action.tool not in TOOLS:
        return PolicyDecision(outcome="block", reason=f"unknown tool: {action.tool!r}", rule="unknown_tool")
    allowed = ROLE_TOOLS.get(agent_role)
    if allowed is None:
        return PolicyDecision(outcome="block", reason=f"unknown role: {agent_role!r}", rule="role_denied")
    if action.tool not in allowed:
        return PolicyDecision(
            outcome="block",
            reason=f"tool {action.tool!r} is not allowed for role {agent_role!r}",
            rule="role_denied",
        )
    if action.tool == "send_email":
        domain = _domain_of(action.args.get("to"))
        if domain is None:
            return PolicyDecision(
                outcome="block", reason="send_email has a missing or invalid recipient", rule="email_domain"
            )
        allowlist = {entry.lower() for entry in settings.email_allowlist}
        if domain not in allowlist:
            return PolicyDecision(
                outcome="block",
                reason=f"send_email domain {domain!r} is not on the allowlist",
                rule="email_domain",
            )
    if tool_call_count >= settings.max_tool_calls:
        return PolicyDecision(
            outcome="block",
            reason=f"loop cap reached: already made {tool_call_count} of {settings.max_tool_calls} allowed tool calls",
            rule="loop_cap",
        )
    if TOOLS[action.tool].risk == "high":
        return PolicyDecision(
            outcome="require_approval",
            reason=f"{action.tool!r} is high-risk and needs human approval",
            rule="high_risk",
        )
    return PolicyDecision(outcome="allow")
