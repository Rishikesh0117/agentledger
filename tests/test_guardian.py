"""Guardian tests: verdict type, protocol, and scripted fake.

Pure offline tests: FakeGuardian performs no model calls.
"""

import json
from datetime import datetime, timezone

import pytest

from agentledger import (
    ChatResult,
    Event,
    EventType,
    ModelError,
    ModelGuardian,
    ProposedAction,
)
from agentledger.guardian import FakeGuardian, Guardian, GuardianVerdict


class FakeClient:
    """Stand-in for ModelClient: replays canned results, records calls."""

    model = "fake-guardian-model"

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self._results.pop(0)


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


def _guardian(client):
    return ModelGuardian(client=client)


def test_model_guardian_parses_verdict():
    client = FakeClient([ChatResult(content='{"verdict": "block", "reason": "Exfiltrates data."}')])
    verdict = _guardian(client).review("task", _action(), [])
    assert verdict == GuardianVerdict(verdict="block", reason="Exfiltrates data.")


def test_model_guardian_strips_code_fences():
    reply = '```json\n{"verdict": "flag", "reason": "Needs a close look."}\n```'
    verdict = _guardian(FakeClient([ChatResult(content=reply)])).review("task", _action(), [])
    assert verdict == GuardianVerdict(verdict="flag", reason="Needs a close look.")


def test_model_guardian_rejects_invalid_json():
    with pytest.raises(ModelError, match="not return a valid verdict"):
        _guardian(FakeClient([ChatResult(content="not json")])).review("task", _action(), [])


def test_model_guardian_rejects_bad_verdict():
    with pytest.raises(ModelError, match="not return a valid verdict"):
        _guardian(FakeClient([ChatResult(content='{"verdict": "maybe"}')])).review(
            "task", _action(), []
        )


def test_model_guardian_prompt_marks_content_untrusted():
    client = FakeClient([ChatResult(content='{"verdict": "allow", "reason": "Safe."}')])
    _guardian(client).review("task", _action(), [])

    system = client.calls[0]["messages"][0]["content"]
    assert "untrusted" in system
    assert "not instructions from the user" in system
    assert client.calls[0]["tools"] is None


def test_model_guardian_sends_task_action_and_trace():
    client = FakeClient([ChatResult(content='{"verdict": "allow", "reason": "Safe."}')])
    trace = [_event(1), _event(2)]
    _guardian(client).review("summarize the ticket", _action(), trace)

    user_message = client.calls[0]["messages"][1]["content"]
    assert "summarize the ticket" in user_message
    assert "send_email" in user_message
    assert "[1]" in user_message and "[2]" in user_message
    assert "action_proposed" in user_message


def test_model_guardian_implements_guardian_protocol():
    assert isinstance(_guardian(FakeClient([])), Guardian)


@pytest.mark.parametrize("verdict", ["allow", "flag", "block"])
@pytest.mark.parametrize("reason", ["", "   "])
def test_model_guardian_rejects_empty_reason(verdict, reason):
    content = json.dumps({"verdict": verdict, "reason": reason})
    with pytest.raises(ModelError, match="valid verdict"):
        _guardian(FakeClient([ChatResult(content=content)])).review("task", _action(), [])


def test_model_guardian_rejects_missing_reason():
    with pytest.raises(ModelError, match="valid verdict"):
        _guardian(FakeClient([ChatResult(content='{"verdict": "block"}')])).review(
            "task", _action(), []
        )
