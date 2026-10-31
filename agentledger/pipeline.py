"""Wrapper pipeline: every proposed action passes all gates before execution.

Order per action (from PLAN.md):

1. policy check -> ``policy_blocked`` + skip when blocked
2. guardian review (high-risk tools only) -> ``guardian_verdict``; a
   ``block`` verdict also writes ``policy_blocked`` and skips
3. approval gate (high-risk tools only) -> ``approval_requested`` and
   ``approval_decision``; a denial skips
4. tool execution -> ``tool_executed`` or ``tool_failed``

Steps 5 and 6 replace the stub gates (:class:`AllowAllPolicy`,
:class:`AllowAllGuardian`, :class:`AutoApprove`) with the real policy
engine, guardian model, and human approval. The pipeline logs
``model_call`` + ``action_proposed`` for every proposal and finishes with
``run_finished``. After each action the pipeline also reports the outcome
back through an optional ``observe(action, outcome)`` method, so model
agents can react to results; agents without it (like the scripted one)
are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from agentledger.agent import BaseAgent, ProposedAction
from agentledger.config import Settings
from agentledger.ledger import EventType, LedgerStore, Run
from agentledger.tools import TOOLS, ToolContext, ToolError, run_tool


class PolicyDecision(BaseModel):
    allowed: bool
    reason: str = ""


class GuardianVerdict(BaseModel):
    verdict: Literal["allow", "flag", "block"]
    reason: str = ""


class ApprovalDecision(BaseModel):
    approved: bool
    decider: str = "auto-approve-stub"


class PolicyCheck(Protocol):
    def __call__(self, action: ProposedAction, role: str) -> PolicyDecision: ...


class GuardianReview(Protocol):
    def __call__(self, task: str, action: ProposedAction, step: int) -> GuardianVerdict: ...


class ApprovalGate(Protocol):
    def __call__(self, task: str, action: ProposedAction, step: int) -> ApprovalDecision: ...


class AllowAllPolicy:
    """Step 3 stub: allow everything. Step 5 implements the real rules."""

    def __call__(self, action: ProposedAction, role: str) -> PolicyDecision:
        return PolicyDecision(allowed=True)


class AllowAllGuardian:
    """Step 3 stub: allow everything. Step 6 calls the guardian model."""

    def __call__(self, task: str, action: ProposedAction, step: int) -> GuardianVerdict:
        return GuardianVerdict(verdict="allow", reason="stub: guardian not yet implemented")


class AutoApprove:
    """Step 3 stub: approve everything. Later this waits for a human."""

    def __call__(self, task: str, action: ProposedAction, step: int) -> ApprovalDecision:
        return ApprovalDecision(approved=True)


@dataclass
class RunResult:
    run: Run
    steps: int
    tool_calls: int
    status: str


def _risk_of(tool_name: str) -> str:
    """Risk level for gating. Unknown tools fail closed as high-risk."""
    spec = TOOLS.get(tool_name)
    return spec.risk if spec else "high"


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
    settings: Settings | None = None,
    policy: PolicyCheck | None = None,
    guardian: GuardianReview | None = None,
    approval: ApprovalGate | None = None,
    max_steps: int | None = None,
) -> RunResult:
    """Run the agent loop until the agent finishes or the step cap hits.

    ``max_steps`` defaults to ``settings.max_tool_calls`` as a loop guard;
    full loop detection (rule 5) is a stretch goal.
    """
    settings = settings or Settings()
    policy = policy or AllowAllPolicy()
    guardian = guardian or AllowAllGuardian()
    approval = approval or AutoApprove()
    cap = max_steps if max_steps is not None else settings.max_tool_calls

    run = store.create_run(task=task, agent_role=agent_role)
    steps = 0
    tool_calls = 0
    status = "completed"

    while True:
        if steps >= cap:
            status = "stopped_max_steps"
            break
        action = agent.propose()
        if action is None:
            break
        steps += 1

        store.append_event(run.id, EventType.MODEL_CALL, "agent", {"step": steps, "model": agent.name})
        store.append_event(
            run.id,
            EventType.ACTION_PROPOSED,
            "agent",
            {"step": steps, "tool": action.tool, "args": action.args},
        )

        decision = policy(action, agent_role)
        if not decision.allowed:
            store.append_event(
                run.id,
                EventType.POLICY_BLOCKED,
                "policy",
                {"step": steps, "tool": action.tool, "reason": decision.reason},
            )
            _notify(agent, action, {"status": "blocked", "reason": decision.reason})
            continue

        if _risk_of(action.tool) == "high":
            verdict = guardian(task, action, steps)
            store.append_event(
                run.id,
                EventType.GUARDIAN_VERDICT,
                "guardian",
                {"step": steps, "tool": action.tool, "verdict": verdict.verdict, "reason": verdict.reason},
            )
            if verdict.verdict == "block":
                store.append_event(
                    run.id,
                    EventType.POLICY_BLOCKED,
                    "policy",
                    {"step": steps, "tool": action.tool, "reason": f"guardian blocked: {verdict.reason}"},
                )
                _notify(agent, action, {"status": "blocked", "reason": f"guardian blocked: {verdict.reason}"})
                continue
            store.append_event(
                run.id,
                EventType.APPROVAL_REQUESTED,
                "approval",
                {"step": steps, "tool": action.tool, "args": action.args},
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
