"""Pipeline tests: gate order, event trail, and failure handling.

Custom gate doubles stand in for the step 5 policy engine and step 6
guardian; the pipeline must call them in PLAN.md order and skip execution
when any gate says no.
"""

import pytest

from agentledger import (
    ApprovalDecision,
    GuardianVerdict,
    LedgerStore,
    PolicyDecision,
    ScriptedAgent,
    default_context,
    run_agent,
)


@pytest.fixture
def store(tmp_path):
    with LedgerStore(tmp_path / "test.db") as s:
        yield s


@pytest.fixture
def ctx():
    return default_context()


def _types(store, run_id):
    return [e.event_type.value for e in store.list_events(run_id)]


class BlockPolicy:
    def __call__(self, action, role):
        return PolicyDecision(allowed=False, reason="test block")


class BlockGuardian:
    def __call__(self, task, action, step):
        return GuardianVerdict(verdict="block", reason="test guardian block")


class FlagGuardian:
    def __call__(self, task, action, step):
        return GuardianVerdict(verdict="flag", reason="test flag")


class DenyApproval:
    def __call__(self, task, action, step):
        return ApprovalDecision(approved=False, decider="test-human")


def test_low_risk_run_skips_guardian_and_approval(store, ctx):
    agent = ScriptedAgent(
        [
            {"tool": "read_document", "args": {"document_id": "doc-001"}},
            {"tool": "draft_message", "args": {"to": "a", "subject": "s", "body": "b"}},
        ]
    )
    result = run_agent("test task", "operator", agent, store, ctx)

    assert result.status == "completed"
    assert (result.steps, result.tool_calls) == (2, 2)
    assert _types(store, result.run.id) == [
        "model_call",
        "action_proposed",
        "tool_executed",
        "model_call",
        "action_proposed",
        "tool_executed",
        "run_finished",
    ]
    assert len(ctx.drafts) == 1
    assert store.verify_chain().ok


def test_high_risk_goes_through_guardian_and_approval(store, ctx):
    agent = ScriptedAgent([{"tool": "update_status", "args": {"record_id": "rec-001", "status": "closed"}}])
    result = run_agent("test task", "operator", agent, store, ctx)

    assert result.status == "completed"
    assert _types(store, result.run.id) == [
        "model_call",
        "action_proposed",
        "guardian_verdict",
        "approval_requested",
        "approval_decision",
        "tool_executed",
        "run_finished",
    ]
    assert ctx.records["rec-001"]["status"] == "closed"
    assert store.verify_chain().ok


def test_policy_block_skips_execution(store, ctx):
    agent = ScriptedAgent([{"tool": "send_email", "args": {"to": "a", "subject": "s", "body": "b"}}])
    result = run_agent("test task", "reader", agent, store, ctx, policy=BlockPolicy())

    assert result.status == "completed"
    assert (result.steps, result.tool_calls) == (1, 0)
    assert _types(store, result.run.id) == ["model_call", "action_proposed", "policy_blocked", "run_finished"]
    assert ctx.outbox == []
    assert store.verify_chain().ok


def test_guardian_block_skips_approval_and_execution(store, ctx):
    agent = ScriptedAgent(
        [{"tool": "update_status", "args": {"record_id": "rec-001", "status": "closed"}}]
    )
    result = run_agent("test task", "operator", agent, store, ctx, guardian=BlockGuardian())

    assert (result.steps, result.tool_calls) == (1, 0)
    events = _types(store, result.run.id)
    assert events == [
        "model_call",
        "action_proposed",
        "guardian_verdict",
        "policy_blocked",
        "run_finished",
    ]
    assert "approval_requested" not in events
    assert ctx.records["rec-001"]["status"] == "open"
    assert store.verify_chain().ok


def test_guardian_flag_still_requires_approval_then_executes(store, ctx):
    agent = ScriptedAgent(
        [{"tool": "update_status", "args": {"record_id": "rec-001", "status": "closed"}}]
    )
    result = run_agent("test task", "operator", agent, store, ctx, guardian=FlagGuardian())

    assert result.tool_calls == 1
    assert "approval_requested" in _types(store, result.run.id)
    assert ctx.records["rec-001"]["status"] == "closed"


def test_approval_deny_skips_execution(store, ctx):
    agent = ScriptedAgent(
        [{"tool": "update_status", "args": {"record_id": "rec-001", "status": "closed"}}]
    )
    result = run_agent("test task", "operator", agent, store, ctx, approval=DenyApproval())

    assert (result.steps, result.tool_calls) == (1, 0)
    events = store.list_events(result.run.id)
    decisions = [e for e in events if e.event_type.value == "approval_decision"]
    assert len(decisions) == 1
    assert decisions[0].payload == {"step": 1, "tool": "update_status", "approved": False}
    assert decisions[0].actor == "test-human"
    assert ctx.records["rec-001"]["status"] == "open"


def test_tool_failure_logged_and_run_continues(store, ctx):
    agent = ScriptedAgent(
        [
            {"tool": "read_document", "args": {"document_id": "nope"}},
            {"tool": "lookup_record", "args": {"record_id": "rec-001"}},
        ]
    )
    result = run_agent("test task", "operator", agent, store, ctx)

    assert result.status == "completed"
    assert (result.steps, result.tool_calls) == (2, 2)
    assert _types(store, result.run.id) == [
        "model_call",
        "action_proposed",
        "tool_failed",
        "model_call",
        "action_proposed",
        "tool_executed",
        "run_finished",
    ]
    assert store.verify_chain().ok


def test_bad_tool_args_logged_as_failure(store, ctx):
    agent = ScriptedAgent([{"tool": "read_document", "args": {}}])
    result = run_agent("test task", "operator", agent, store, ctx)

    assert (result.steps, result.tool_calls) == (1, 1)
    assert "tool_failed" in _types(store, result.run.id)
    assert store.verify_chain().ok


def test_unknown_tool_fails_safely(store, ctx):
    agent = ScriptedAgent([{"tool": "delete_everything", "args": {}}])
    result = run_agent("test task", "operator", agent, store, ctx)

    assert result.status == "completed"
    assert "tool_failed" in _types(store, result.run.id)
    assert "tool_executed" not in _types(store, result.run.id)
    assert store.verify_chain().ok


def test_empty_script_finishes_with_zero_steps(store, ctx):
    result = run_agent("test task", "operator", ScriptedAgent([]), store, ctx)

    assert result.status == "completed"
    assert (result.steps, result.tool_calls) == (0, 0)
    assert _types(store, result.run.id) == ["run_finished"]


def test_max_steps_stops_long_script(store, ctx):
    agent = ScriptedAgent(
        [
            {"tool": "lookup_record", "args": {"record_id": "rec-001"}},
            {"tool": "lookup_record", "args": {"record_id": "rec-002"}},
            {"tool": "lookup_record", "args": {"record_id": "rec-001"}},
        ]
    )
    result = run_agent("test task", "operator", agent, store, ctx, max_steps=2)

    assert result.status == "stopped_max_steps"
    assert (result.steps, result.tool_calls) == (2, 2)
    finished = [e for e in store.list_events(result.run.id) if e.event_type.value == "run_finished"]
    assert finished[0].payload["status"] == "stopped_max_steps"


def test_run_status_updated_in_db(store, ctx):
    agent = ScriptedAgent([{"tool": "lookup_record", "args": {"record_id": "rec-001"}}])
    result = run_agent("test task", "operator", agent, store, ctx)

    assert result.run.status == "completed"
    assert store.get_run(result.run.id).status == "completed"


def test_finish_run_rejects_unknown_run(store):
    with pytest.raises(ValueError, match="unknown run_id"):
        store.finish_run("missing", "completed")
    with pytest.raises(ValueError, match="unknown run_id"):
        store.get_run("missing")


class RecordingAgent:
    """Scripted agent that also records the outcomes the pipeline reports."""

    name = "recording"

    def __init__(self, actions):
        self._agent = ScriptedAgent(actions)
        self.observed = []

    def propose(self):
        return self._agent.propose()

    def observe(self, action, outcome):
        self.observed.append((action, outcome))


def test_pipeline_reports_executed_and_failed_outcomes(store, ctx):
    agent = RecordingAgent(
        [
            {"tool": "lookup_record", "args": {"record_id": "rec-001"}},
            {"tool": "read_document", "args": {"document_id": "nope"}},
        ]
    )
    run_agent("test task", "operator", agent, store, ctx)

    assert [(a.tool, o["status"]) for a, o in agent.observed] == [
        ("lookup_record", "executed"),
        ("read_document", "failed"),
    ]
    assert agent.observed[0][1]["result"]["record"]["id"] == "rec-001"
    assert "unknown document_id" in agent.observed[1][1]["error"]


def test_pipeline_reports_blocked_outcome(store, ctx):
    agent = RecordingAgent(
        [{"tool": "send_email", "args": {"to": "a", "subject": "s", "body": "b"}}]
    )
    run_agent("test task", "reader", agent, store, ctx, policy=BlockPolicy())

    assert [(a.tool, o["status"]) for a, o in agent.observed] == [("send_email", "blocked")]
    assert agent.observed[0][1]["reason"] == "test block"


def test_pipeline_reports_denied_outcome(store, ctx):
    agent = RecordingAgent(
        [{"tool": "update_status", "args": {"record_id": "rec-001", "status": "closed"}}]
    )
    run_agent("test task", "operator", agent, store, ctx, approval=DenyApproval())

    assert [(a.tool, o["status"]) for a, o in agent.observed] == [("update_status", "denied")]
    assert agent.observed[0][1]["decider"] == "test-human"
