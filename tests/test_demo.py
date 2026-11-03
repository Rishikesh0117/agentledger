"""Demo tests: documents, console gate, and scripted scenarios.

All offline: scenarios run through the real pipeline with scripted
agent/guardian doubles and need no API key.
"""

import io
import sys

import pytest

from agentledger import (
    ChatResult,
    EventType,
    LedgerStore,
    ModelError,
    ProposedAction,
    default_context,
    run_tool,
)
from agentledger.demo import (
    INJECTED_DOC_ID,
    NORMAL_DOC_ID,
    ConsoleApprovalGate,
    ScenarioResult,
    demo_context,
    run_injection,
    run_injection_live,
    run_normal,
    summarize_live_injection,
)


class FakeChatClient:
    """Stubbed model transport: replays one canned reply, records calls."""

    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        return ChatResult(content=self.content)


def _action():
    return ProposedAction(tool="update_status", args={"record_id": "rec-001", "status": "closed"})


def test_normal_scenario_runs_end_to_end(tmp_path):
    db = str(tmp_path / "demo.db")
    ctx = demo_context()
    result = run_normal(db, approval="auto", ctx=ctx)

    assert result.name == "normal"
    assert result.status == "completed"
    assert result.events == [
        "model_call",
        "action_proposed",
        "tool_executed",
        "model_call",
        "action_proposed",
        "tool_executed",
        "model_call",
        "action_proposed",
        "tool_executed",
        "model_call",
        "action_proposed",
        "guardian_verdict",
        "approval_requested",
        "approval_decision",
        "tool_executed",
        "run_finished",
    ]
    assert result.blocked == []
    assert result.verify_ok and result.broken_seq is None
    assert ctx.records["rec-001"]["status"] == "closed"
    assert len(ctx.drafts) == 1
    assert ctx.outbox == []


def test_injection_scenario_blocks_both_attempts(tmp_path):
    db = str(tmp_path / "demo.db")
    ctx = demo_context()
    result = run_injection(db, ctx=ctx)

    assert result.name == "injection"
    assert result.status == "completed"
    assert result.events == [
        "model_call",
        "action_proposed",
        "tool_executed",
        "model_call",
        "action_proposed",
        "policy_blocked",
        "model_call",
        "action_proposed",
        "guardian_verdict",
        "policy_blocked",
        "run_finished",
    ]
    assert result.events.count("tool_executed") == 1  # the initial read only
    assert "approval_requested" not in result.events
    assert [b.split(":")[0] for b in result.blocked] == ["email_domain", "guardian_block"]
    assert "not instructions from the user" in result.blocked[1]
    assert result.verify_ok and result.broken_seq is None
    assert ctx.outbox == []

    with LedgerStore(db) as store:
        assert store.verify_chain().ok
        blocked = [
            e for e in store.list_events(result.run_id) if e.event_type.value == "policy_blocked"
        ]
        assert [e.payload["rule"] for e in blocked] == ["email_domain", "guardian_block"]
        verdicts = [
            e for e in store.list_events(result.run_id) if e.event_type.value == "guardian_verdict"
        ]
        assert len(verdicts) == 1  # policy-blocked attempt never reached the guardian
        assert verdicts[0].payload["verdict"] == "block"


def test_demo_documents_are_synthetic_and_wired_to_read_document():
    ctx = demo_context()
    normal = ctx.documents[NORMAL_DOC_ID]
    injected = ctx.documents[INJECTED_DOC_ID]

    assert "SYNTHETIC" in normal and "SYNTHETIC" in injected
    assert "DEMO INJECTION" not in normal
    assert "outside.example" not in normal
    assert "DEMO INJECTION" in injected
    assert "archive@outside.example" in injected
    assert "rec-001" in normal and "rec-002" in injected

    found = run_tool("read_document", {"document_id": NORMAL_DOC_ID}, ctx)
    assert found["text"] == normal
    assert set(ctx.documents) >= {"doc-001", "doc-002", NORMAL_DOC_ID, INJECTED_DOC_ID}


@pytest.mark.parametrize("answer", ["y\n", "yes\n", "YES\n"])
def test_console_gate_approves_on_explicit_yes(monkeypatch, answer):
    monkeypatch.setattr(sys, "stdin", io.StringIO(answer))
    decision = ConsoleApprovalGate()("task", _action(), 1)
    assert (decision.approved, decision.decider) == (True, "demo-human")


@pytest.mark.parametrize("answer", ["n\n", "no\n", "\n", "", "maybe\n", "approve\n"])
def test_console_gate_denies_anything_else(monkeypatch, answer):
    monkeypatch.setattr(sys, "stdin", io.StringIO(answer))
    decision = ConsoleApprovalGate()("task", _action(), 1)
    assert (decision.approved, decision.decider) == (False, "demo-human")


def test_console_gate_decider_from_env_or_argument(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    monkeypatch.setenv("AGENTLEDGER_DECIDER", "oncall-uma")
    assert ConsoleApprovalGate()("t", _action(), 1).decider == "oncall-uma"
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    gate = ConsoleApprovalGate(decider="explicit")
    assert gate("t", _action(), 1).decider == "explicit"


def test_console_gate_prints_tool_and_args(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("n\n"))
    ConsoleApprovalGate()("task", _action(), 2)

    shown = capsys.readouterr().out
    assert "update_status" in shown
    assert "rec-001" in shown


def test_console_gate_prints_request_context_from_store(monkeypatch, capsys, tmp_path):
    db = str(tmp_path / "demo.db")
    with LedgerStore(db) as store:
        run = store.create_run(task="t", agent_role="operator")
        store.append_event(
            run.id,
            EventType.APPROVAL_REQUESTED,
            "approval",
            {"step": 1, "tool": "x", "rule": "high_risk", "reason": "risky", "guardian_reason": "flagged!"},
        )
        monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
        ConsoleApprovalGate(store)("t", _action(), 1)

    shown = capsys.readouterr().out
    assert "high_risk" in shown and "risky" in shown and "flagged!" in shown


def test_default_context_is_untouched_by_demo_docs():
    assert set(default_context().documents) == {"doc-001", "doc-002"}


def test_injection_live_uses_model_guardian_with_stubbed_client(tmp_path):
    client = FakeChatClient('{"verdict": "block", "reason": "Stubbed block for offline test."}')
    result = run_injection_live(str(tmp_path / "demo.db"), guardian_client=client)

    assert result.name == "injection-live"
    assert result.status == "completed"
    assert len(client.calls) == 1  # the model-backed guardian consulted its client once
    assert result.verdict == "block"
    assert result.executed_sends == 0
    assert result.verify_ok


def _proposed_actions(db_path, run_id):
    with LedgerStore(db_path) as store:
        return [
            (e.payload["tool"], e.payload["args"])
            for e in store.list_events(run_id)
            if e.event_type.value == "action_proposed"
        ]


def test_injection_live_replays_same_actions_as_scripted(tmp_path):
    scripted = run_injection(str(tmp_path / "s.db"))
    live = run_injection_live(
        str(tmp_path / "l.db"),
        guardian_client=FakeChatClient('{"verdict": "block", "reason": "x"}'),
    )

    assert live.events == scripted.events
    assert [b.split(":")[0] for b in live.blocked] == [b.split(":")[0] for b in scripted.blocked]
    assert _proposed_actions(str(tmp_path / "l.db"), live.run_id) == _proposed_actions(
        str(tmp_path / "s.db"), scripted.run_id
    )


def test_injection_live_missing_key_rejected_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr("agentledger.demo.get_api_key", lambda: None)
    with pytest.raises(ModelError, match="META_MODEL_API_KEY"):
        run_injection_live(str(tmp_path / "demo.db"))


def _live_result(**overrides):
    base = dict(
        name="injection-live",
        run_id="r",
        db_path="d.db",
        status="completed",
        events=[],
        blocked=[],
        verify_ok=True,
        broken_seq=None,
        verdict="block",
        verdict_reason="Some reason.",
        executed_sends=0,
    )
    base.update(overrides)
    return ScenarioResult(**base)


def test_summarize_live_pass():
    report, code = summarize_live_injection(_live_result(), "muse-spark", [])
    assert code == 0
    assert "LIVE RESULT: PASS" in report
    assert "muse-spark" in report and "Some reason." in report and "block" in report


def test_summarize_live_flag_verdict_fails():
    report, code = summarize_live_injection(_live_result(verdict="flag"), "m", [])
    assert code == 1
    assert "flag" in report and "FAIL" in report


def test_summarize_live_tripwire_and_chain_fail():
    report, code = summarize_live_injection(
        _live_result(executed_sends=1, verify_ok=False, broken_seq=9),
        "m",
        [{"to": "x"}],
    )
    assert code == 1
    assert "TRIPWIRE" in report
    assert "tripwire: 1 send_email executed" in report
    assert "broken at seq 9" in report


def test_summarize_live_empty_reason_fails():
    report, code = summarize_live_injection(_live_result(verdict_reason="  "), "m", [])
    assert code == 1
    assert "empty" in report
