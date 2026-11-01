"""Pipeline tests: real policy wiring, event trail, and failure handling.

The policy engine (agentledger.policy) is the only policy check: tests
use crafted roles, settings, and actions instead of policy doubles.
High-risk actions fail closed until step 5.3 wires the approval gate.
"""

import pytest

from agentledger import (
    ApprovalDecision,
    AutoApprove,
    AutoDeny,
    LedgerStore,
    ScriptedAgent,
    Settings,
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


def _settings(**overrides):
    values = {"agent_model": "a", "guardian_model": "g", "model_api_base": "https://b.test"}
    values.update(overrides)
    return Settings(**values)


def _types(store, run_id):
    return [e.event_type.value for e in store.list_events(run_id)]


def test_low_risk_run_skips_guardian_and_approval(store, ctx):
    agent = ScriptedAgent(
        [
            {"tool": "read_document", "args": {"document_id": "doc-001"}},
            {"tool": "draft_message", "args": {"to": "a", "subject": "s", "body": "b"}},
        ]
    )
    result = run_agent("test task", "operator", agent, store, ctx, _settings())

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


def test_high_risk_denied_by_default_without_gate(store, ctx):
    agent = ScriptedAgent(
        [{"tool": "update_status", "args": {"record_id": "rec-001", "status": "closed"}}]
    )
    result = run_agent("test task", "operator", agent, store, ctx, _settings())

    assert result.status == "completed"
    assert (result.steps, result.tool_calls) == (1, 0)
    assert _types(store, result.run.id) == [
        "model_call",
        "action_proposed",
        "approval_requested",
        "approval_decision",
        "run_finished",
    ]
    events = store.list_events(result.run.id)
    decision = [e for e in events if e.event_type.value == "approval_decision"][0]
    assert decision.actor == "auto-deny"
    assert decision.payload == {"step": 1, "tool": "update_status", "approved": False}
    assert ctx.records["rec-001"]["status"] == "open"
    assert store.verify_chain().ok


def test_approved_high_risk_executes(store, ctx):
    agent = ScriptedAgent(
        [{"tool": "update_status", "args": {"record_id": "rec-001", "status": "closed"}}]
    )
    result = run_agent("test task", "operator", agent, store, ctx, _settings(), approval=AutoApprove())

    assert (result.steps, result.tool_calls) == (1, 1)
    assert _types(store, result.run.id) == [
        "model_call",
        "action_proposed",
        "approval_requested",
        "approval_decision",
        "tool_executed",
        "run_finished",
    ]
    assert ctx.records["rec-001"]["status"] == "closed"
    assert store.verify_chain().ok


class DenyApproval:
    def __call__(self, task, action, step):
        return ApprovalDecision(approved=False, decider="test-human")


def test_denied_high_risk_skips_execution_but_logs_decision(store, ctx):
    agent = ScriptedAgent(
        [{"tool": "update_status", "args": {"record_id": "rec-001", "status": "closed"}}]
    )
    result = run_agent("test task", "operator", agent, store, ctx, _settings(), approval=DenyApproval())

    assert (result.steps, result.tool_calls) == (1, 0)
    events = store.list_events(result.run.id)
    requested = [e for e in events if e.event_type.value == "approval_requested"][0]
    assert requested.payload == {
        "step": 1,
        "tool": "update_status",
        "args": {"record_id": "rec-001", "status": "closed"},
        "rule": "high_risk",
        "reason": "'update_status' is high-risk and needs human approval",
    }
    decision = [e for e in events if e.event_type.value == "approval_decision"][0]
    assert decision.actor == "test-human"
    assert decision.payload["approved"] is False
    assert ctx.records["rec-001"]["status"] == "open"
    assert store.verify_chain().ok


def test_auto_deny_denies_everything():
    decision = AutoDeny()("task", None, 1)
    assert (decision.approved, decision.decider) == (False, "auto-deny")


def test_role_denied_logs_policy_blocked(store, ctx):
    agent = ScriptedAgent([{"tool": "send_email", "args": {"to": "a", "subject": "s", "body": "b"}}])
    result = run_agent("test task", "reader", agent, store, ctx, _settings())

    assert result.status == "completed"
    assert (result.steps, result.tool_calls) == (1, 0)
    assert _types(store, result.run.id) == ["model_call", "action_proposed", "policy_blocked", "run_finished"]
    blocked = [e for e in store.list_events(result.run.id) if e.event_type.value == "policy_blocked"]
    assert blocked[0].payload == {
        "step": 1,
        "tool": "send_email",
        "args": {"to": "a", "subject": "s", "body": "b"},
        "rule": "role_denied",
        "reason": "tool 'send_email' is not allowed for role 'reader'",
        "role": "reader",
    }
    assert ctx.outbox == []
    assert store.verify_chain().ok


def test_bad_domain_logs_policy_blocked(store, ctx):
    agent = ScriptedAgent(
        [{"tool": "send_email", "args": {"to": "u@evil.test", "subject": "s", "body": "b"}}]
    )
    result = run_agent("test task", "operator", agent, store, ctx, _settings())

    assert (result.steps, result.tool_calls) == (1, 0)
    blocked = [e for e in store.list_events(result.run.id) if e.event_type.value == "policy_blocked"]
    assert blocked[0].payload["rule"] == "email_domain"
    assert blocked[0].payload["role"] == "operator"
    assert ctx.outbox == []
    assert store.verify_chain().ok


def test_tool_failure_logged_and_run_continues(store, ctx):
    agent = ScriptedAgent(
        [
            {"tool": "read_document", "args": {"document_id": "nope"}},
            {"tool": "lookup_record", "args": {"record_id": "rec-001"}},
        ]
    )
    result = run_agent("test task", "operator", agent, store, ctx, _settings())

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
    result = run_agent("test task", "operator", agent, store, ctx, _settings())

    assert (result.steps, result.tool_calls) == (1, 1)
    assert "tool_failed" in _types(store, result.run.id)
    assert store.verify_chain().ok


def test_unknown_tool_blocked_by_policy(store, ctx):
    agent = ScriptedAgent([{"tool": "delete_everything", "args": {}}])
    result = run_agent("test task", "operator", agent, store, ctx, _settings())

    assert result.status == "completed"
    assert _types(store, result.run.id) == ["model_call", "action_proposed", "policy_blocked", "run_finished"]
    blocked = [e for e in store.list_events(result.run.id) if e.event_type.value == "policy_blocked"]
    assert blocked[0].payload["rule"] == "unknown_tool"
    assert "tool_executed" not in _types(store, result.run.id)
    assert store.verify_chain().ok


def test_empty_script_finishes_with_zero_steps(store, ctx):
    result = run_agent("test task", "operator", ScriptedAgent([]), store, ctx, _settings())

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
    result = run_agent("test task", "operator", agent, store, ctx, _settings(), max_steps=2)

    assert result.status == "stopped_max_steps"
    assert (result.steps, result.tool_calls) == (2, 2)
    finished = [e for e in store.list_events(result.run.id) if e.event_type.value == "run_finished"]
    assert finished[0].payload["status"] == "stopped_max_steps"


def test_run_status_updated_in_db(store, ctx):
    agent = ScriptedAgent([{"tool": "lookup_record", "args": {"record_id": "rec-001"}}])
    result = run_agent("test task", "operator", agent, store, ctx, _settings())

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
    run_agent("test task", "operator", agent, store, ctx, _settings())

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
    run_agent("test task", "reader", agent, store, ctx, _settings())

    assert [(a.tool, o["status"]) for a, o in agent.observed] == [("send_email", "blocked")]
    assert agent.observed[0][1]["rule"] == "role_denied"
    assert "reader" in agent.observed[0][1]["reason"]


def test_pipeline_reports_denied_outcome(store, ctx):
    agent = RecordingAgent(
        [{"tool": "update_status", "args": {"record_id": "rec-001", "status": "closed"}}]
    )
    run_agent("test task", "operator", agent, store, ctx, _settings(), approval=DenyApproval())

    assert [(a.tool, o["status"]) for a, o in agent.observed] == [("update_status", "denied")]
    assert agent.observed[0][1]["decider"] == "test-human"


def test_model_call_records_scripted_fake_name_and_latency(store, ctx):
    agent = ScriptedAgent([{"tool": "lookup_record", "args": {"record_id": "rec-001"}}])
    result = run_agent("test task", "operator", agent, store, ctx, _settings())

    calls = [e for e in store.list_events(result.run.id) if e.event_type.value == "model_call"]
    assert len(calls) == 1
    assert calls[0].payload["model"] == "scripted-fake"
    assert isinstance(calls[0].payload["latency_ms"], int)
    assert calls[0].payload["latency_ms"] >= 0
