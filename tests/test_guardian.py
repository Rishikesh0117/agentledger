"""Guardian tests: verdict type, protocol, and scripted fake.

Pure offline tests: FakeGuardian performs no model calls.
"""

from datetime import datetime, timezone

import pytest

from agentledger import Event, EventType, ProposedAction
from agentledger.guardian import FakeGuardian, Guardian, GuardianVerdict


def _event(n=1):
    return Event(
        seq=n,
        run_id="run-1",
        ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        event_type=EventType.ACTION_PROPOSED,
        actor="agent",
        payload={"step": n, "tool": "read_document"},
        prev_hash="prev",
        hash="hash",
    )


def _action():
    return ProposedAction(tool="send_email", args={"to": "u@evil.test"})


def test_verdict_accepts_all_three_outcomes():
    assert GuardianVerdict(verdict="allow", reason="fine").verdict == "allow"
    assert GuardianVerdict(verdict="flag", reason="hmm").verdict == "flag"
    assert GuardianVerdict(verdict="block", reason="no").verdict == "block"


def test_verdict_reason_defaults_empty():
    assert GuardianVerdict(verdict="allow").reason == ""


def test_verdict_rejects_unknown_outcome():
    with pytest.raises(ValueError):
        GuardianVerdict(verdict="maybe", reason="?")


def test_fake_replays_verdicts_in_order():
    guardian = FakeGuardian(
        [
            GuardianVerdict(verdict="allow", reason="first"),
            {"verdict": "flag", "reason": "second"},
            ("block", "third"),
        ]
    )
    assert guardian.review("task", _action(), []) == GuardianVerdict(verdict="allow", reason="first")
    assert guardian.review("task", _action(), []) == GuardianVerdict(verdict="flag", reason="second")
    assert guardian.review("task", _action(), []) == GuardianVerdict(verdict="block", reason="third")


def test_fake_exhaustion_defaults_to_allow():
    guardian = FakeGuardian([("block", "only one")])
    assert guardian.review("task", _action(), []).verdict == "block"
    extra = guardian.review("task", _action(), [])
    assert extra.verdict == "allow"
    assert "exhausted" in extra.reason


def test_fake_records_what_it_saw():
    guardian = FakeGuardian([("allow", "ok")])
    action = _action()
    trace = [_event(1), _event(2)]
    guardian.review("do the thing", action, trace)

    assert len(guardian.calls) == 1
    call = guardian.calls[0]
    assert call["task"] == "do the thing"
    assert call["action"] == action
    assert call["trace"] == trace


def test_fake_rejects_bad_verdict():
    with pytest.raises(ValueError):
        FakeGuardian([{"verdict": "maybe", "reason": "?"}])
    with pytest.raises(ValueError):
        FakeGuardian([("allow", "ok", "extra")])


def test_fake_implements_guardian_protocol():
    assert isinstance(FakeGuardian([]), Guardian)
