"""Dashboard tests: pure helpers only (Streamlit itself is not needed)."""

from datetime import datetime, timezone

import dashboard
from agentledger import Event, EventType


def _event(event_type, payload, seq=1):
    return Event(
        seq=seq,
        run_id="run-1",
        ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        event_type=event_type,
        actor="agent",
        payload=payload,
        prev_hash="prev",
        hash="hash",
    )


def test_summarize_model_call():
    event = _event(EventType.MODEL_CALL, {"model": "muse-spark", "latency_ms": 12})
    assert dashboard.summarize_event(event) == "model muse-spark · 12ms"


def test_summarize_model_call_finish_without_latency():
    event = _event(EventType.MODEL_CALL, {"model": "m", "finish": True})
    assert dashboard.summarize_event(event) == "model m (finish)"


def test_summarize_action_and_tool_events():
    assert dashboard.summarize_event(_event(EventType.ACTION_PROPOSED, {"tool": "read_document"})) == (
        "read_document"
    )
    assert dashboard.summarize_event(_event(EventType.TOOL_EXECUTED, {"tool": "read_document"})) == (
        "read_document"
    )
    assert dashboard.summarize_event(_event(EventType.TOOL_FAILED, {"tool": "read_document"})) == (
        "read_document failed"
    )


def test_summarize_gate_events():
    blocked = _event(EventType.POLICY_BLOCKED, {"tool": "send_email", "rule": "role_denied"})
    assert dashboard.summarize_event(blocked) == "send_email [role_denied]"
    verdict = _event(EventType.GUARDIAN_VERDICT, {"tool": "update_status", "verdict": "block"})
    assert dashboard.summarize_event(verdict) == "update_status: block"
    requested = _event(EventType.APPROVAL_REQUESTED, {"tool": "update_status"})
    assert dashboard.summarize_event(requested) == "update_status"
    approved = _event(EventType.APPROVAL_DECISION, {"tool": "update_status", "approved": True})
    assert dashboard.summarize_event(approved) == "update_status: approved"
    denied = _event(EventType.APPROVAL_DECISION, {"tool": "update_status", "approved": False})
    assert dashboard.summarize_event(denied) == "update_status: denied"


def test_summarize_run_finished():
    event = _event(EventType.RUN_FINISHED, {"status": "completed", "steps": 2, "tool_calls": 2})
    assert dashboard.summarize_event(event) == "completed (2 steps, 2 tool calls)"


def test_blocked_events_filters_in_order():
    events = [
        _event(EventType.ACTION_PROPOSED, {"tool": "a"}, seq=1),
        _event(EventType.POLICY_BLOCKED, {"tool": "b"}, seq=2),
        _event(EventType.TOOL_EXECUTED, {"tool": "c"}, seq=3),
        _event(EventType.POLICY_BLOCKED, {"tool": "d"}, seq=4),
    ]
    assert [e.seq for e in dashboard.blocked_events(events)] == [2, 4]


def test_db_path_default_comes_from_shipped_config():
    assert dashboard.get_db_path_default() == "agentledger.db"
