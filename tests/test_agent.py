"""Agent tests: scripted replay and strict JSON action parsing."""

import pytest

from agentledger import ProposedAction, ScriptedAgent, parse_action_json


def test_propose_returns_actions_in_order_then_none():
    agent = ScriptedAgent(
        [
            ProposedAction(tool="read_document", args={"document_id": "doc-001"}),
            ProposedAction(tool="lookup_record", args={"record_id": "rec-001"}),
        ]
    )
    assert agent.propose() == ProposedAction(tool="read_document", args={"document_id": "doc-001"})
    assert agent.propose() == ProposedAction(tool="lookup_record", args={"record_id": "rec-001"})
    assert agent.propose() is None
    assert agent.propose() is None  # stays finished


def test_empty_script_finishes_immediately():
    assert ScriptedAgent([]).propose() is None


def test_scripted_default_name_marks_it_fake():
    assert ScriptedAgent([]).name == "scripted-fake"


def test_accepts_dicts_and_json_strings():
    agent = ScriptedAgent(
        [
            {"tool": "draft_message", "args": {"to": "a", "subject": "s", "body": "b"}},
            '{"tool": "update_status", "args": {"record_id": "rec-001", "status": "closed"}}',
        ]
    )
    assert agent.propose() == ProposedAction(
        tool="draft_message", args={"to": "a", "subject": "s", "body": "b"}
    )
    assert agent.propose() == ProposedAction(
        tool="update_status", args={"record_id": "rec-001", "status": "closed"}
    )


def test_parse_action_json_valid():
    action = parse_action_json('{"tool": "send_email", "args": {"to": "a@example.com"}}')
    assert action == ProposedAction(tool="send_email", args={"to": "a@example.com"})


def test_parse_action_json_args_defaults_to_empty():
    assert parse_action_json('{"tool": "read_document"}') == ProposedAction(
        tool="read_document", args={}
    )


def test_parse_action_json_rejects_not_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_action_json("do the thing")


def test_parse_action_json_rejects_non_object():
    with pytest.raises(ValueError, match="must be an object"):
        parse_action_json('["read_document"]')


def test_parse_action_json_rejects_bad_shape():
    with pytest.raises(ValueError):
        parse_action_json('{"args": {}}')  # missing tool
    with pytest.raises(ValueError):
        parse_action_json('{"tool": "read_document", "args": "doc-001"}')  # args not an object
    with pytest.raises(ValueError):
        parse_action_json('{"tool": "", "args": {}}')  # empty tool
    with pytest.raises(ValueError):
        parse_action_json('{"tool": "read_document", "args": {}, "extra": 1}')  # extra key
