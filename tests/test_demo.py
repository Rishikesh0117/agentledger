"""Demo tests: documents, console gate, and scripted scenarios.

All offline: scenarios run through the real pipeline with scripted
agent/guardian doubles and need no API key.
"""

import io
import sys

import pytest

from agentledger import EventType, LedgerStore, ProposedAction, default_context, run_tool
from agentledger.demo import (
    INJECTED_DOC_ID,
    NORMAL_DOC_ID,
    ConsoleApprovalGate,
    demo_context,
    run_injection,
    run_normal,
)


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
