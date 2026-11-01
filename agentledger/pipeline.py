"""Wrapper pipeline: every proposed action passes the policy check first.

Order per action:

1. policy check (agentledger.policy, the only policy check)
   - ``block`` -> ``policy_blocked`` + skip, outcome reported to the agent
   - ``require_approval`` -> ``approval_requested``, consult the
     approval gate, log ``approval_decision`` either way, execute only
     on approve
   - ``allow`` -> execute the tool
2. tool execution -> ``tool_executed`` or ``tool_failed``

The pipeline logs ``model_call`` + ``action_proposed`` for every proposal
and finishes with ``run_finished``. Every ``model_call`` carries the
agent's model name and ``latency_ms``. When an agent with
``empty_proposal_is_call`` set returns ``None``, that final call is
recorded as a ``model_call`` with ``finish: true``. After each action the
pipeline also reports the outcome back through an optional
``observe(action, outcome)`` method, so model agents can react to
results; agents without it (like the scripted one) are unaffected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from agentledger.agent import BaseAgent, ProposedAction
from agentledger.config import Settings
from agentledger.ledger import EventType, LedgerStore, Run
from agentledger.policy import evaluate
from agentledger.tools import ToolContext, ToolError, run_tool


class GuardianVerdict(BaseModel):
    verdict: Literal["allow", "flag", "block"]
    reason: str = ""


class ApprovalDecision(BaseModel):
    approved: bool
    decider: str = "auto-approve-stub"


class GuardianReview(Protocol):
    def __call__(self, task: str, action: ProposedAction, step: int) -> GuardianVerdict: ...


class ApprovalGate(Protocol):
    def __call__(self, task: str, action: ProposedAction, step: int) -> ApprovalDecision: ...


class AllowAllGuardian:
    """Step 3 stub: allow everything. Step 6 calls the guardian model."""

    def __call__(self, task: str, action: ProposedAction, step: int) -> GuardianVerdict:
        return GuardianVerdict(verdict="allow", reason="stub: guardian not yet implemented")


class AutoApprove:
    """Approve everything. Explicit opt-in for scripts and tests only."""

    def __call__(self, task: str, action: ProposedAction, step: int) -> ApprovalDecision:
        return ApprovalDecision(approved=True)


class AutoDeny:
    """Default gate: deny everything when no human is wired up (fail closed)."""

    def __call__(self, task: str, action: ProposedAction, step: int) -> ApprovalDecision:
        return ApprovalDecision(approved=False, decider="auto-deny")


@dataclass
class RunResult:
    run: Run
    steps: int
    tool_calls: int
    status: str


def _notify(agent: BaseAgent, action: ProposedAction, outcome: dict[str, Any]) -> None:
    """Report an action's outcome back to agents that accept observations."""
    observe = getattr(agent, "observe", None)
    if callable(observe):
        observe(action, outcome)


def run_agent(
    task: str,
    agent_role: str,
    agent: BaseAgent,
    store: LedgerStore,
    ctx: ToolContext,
    settings: Settings,
    guardian: GuardianReview | None = None,
    approval: ApprovalGate | None = None,
    max_steps: int | None = None,
) -> RunResult:
    """Run the agent loop until the agent finishes or the step cap hits.

    ``settings`` is required: the policy check needs the email
    allowlist and the loop cap. ``max_steps`` defaults to
    ``settings.max_tool_calls`` as a loop guard; full loop detection
    (rule 5) is a stretch goal.

    ``approval`` decides ``require_approval`` outcomes and defaults to
    denying everything; pass a human-backed gate to approve.
    ``guardian`` is still a seam for step 6 and is not consulted yet.
    """
    approval = approval or AutoDeny()
    if max_steps is not None:
        cap = max_steps
    else:
        cap = settings.max_tool_calls

    run = store.create_run(task=task, agent_role=agent_role)
    steps = 0
    tool_calls = 0
    status = "completed"

    while True:
        if steps >= cap:
            status = "stopped_max_steps"
            break
        started = time.perf_counter()
        action = agent.propose()
        latency_ms = int((time.perf_counter() - started) * 1000)
        if action is None:
            if getattr(agent, "empty_proposal_is_call", False):
                store.append_event(
                    run.id,
                    EventType.MODEL_CALL,
                    "agent",
                    {"step": steps + 1, "model": agent.name, "finish": True, "latency_ms": latency_ms},
                )
            break
        steps += 1

        store.append_event(
            run.id,
            EventType.MODEL_CALL,
            "agent",
            {"step": steps, "model": agent.name, "latency_ms": latency_ms},
        )
        store.append_event(
            run.id,
            EventType.ACTION_PROPOSED,
            "agent",
            {"step": steps, "tool": action.tool, "args": action.args},
        )

        decision = evaluate(action, agent_role, settings, tool_calls)
        if decision.outcome == "block":
            store.append_event(
                run.id,
                EventType.POLICY_BLOCKED,
                "policy",
                {
                    "step": steps,
                    "tool": action.tool,
                    "args": action.args,
                    "rule": decision.rule,
                    "reason": decision.reason,
                    "role": agent_role,
                },
            )
            _notify(agent, action, {"status": "blocked", "reason": decision.reason, "rule": decision.rule})
            continue
        if decision.outcome == "require_approval":
            store.append_event(
                run.id,
                EventType.APPROVAL_REQUESTED,
                "approval",
                {
                    "step": steps,
                    "tool": action.tool,
                    "args": action.args,
                    "rule": decision.rule,
                    "reason": decision.reason,
                },
            )
            approval_decision = approval(task, action, steps)
            store.append_event(
                run.id,
                EventType.APPROVAL_DECISION,
                approval_decision.decider,
                {"step": steps, "tool": action.tool, "approved": approval_decision.approved},
            )
            if not approval_decision.approved:
                _notify(agent, action, {"status": "denied", "decider": approval_decision.decider})
                continue

        try:
            result = run_tool(action.tool, action.args, ctx)
        except (ToolError, ValueError) as e:
            # ValueError covers Pydantic ValidationError for bad arguments.
            tool_calls += 1
            store.append_event(
                run.id, EventType.TOOL_FAILED, "tool", {"step": steps, "tool": action.tool, "error": str(e)}
            )
            _notify(agent, action, {"status": "failed", "error": str(e)})
            continue
        tool_calls += 1
        store.append_event(
            run.id, EventType.TOOL_EXECUTED, "tool", {"step": steps, "tool": action.tool, "result": result}
        )
        _notify(agent, action, {"status": "executed", "result": result})

    store.finish_run(run.id, status)
    store.append_event(
        run.id, EventType.RUN_FINISHED, "pipeline", {"steps": steps, "tool_calls": tool_calls, "status": status}
    )
    return RunResult(run=store.get_run(run.id), steps=steps, tool_calls=tool_calls, status=status)
