"""Policy tests: every rule, precedence order, and role-config sync.

Pure offline tests: evaluate() takes plain values and performs no
DB, model, or tool calls.
"""

import pytest

from agentledger import (
    ROLE_TOOLS,
    PolicyDecision,
    ProposedAction,
    Settings,
    evaluate,
    load_roles,
)


def _settings(**overrides):
    values = {"agent_model": "a", "guardian_model": "g", "model_api_base": "https://b.test"}
    values.update(overrides)
    return Settings(**values)


def _action(tool, **args):
    return ProposedAction(tool=tool, args=args)


def test_allow_reader_low_risk_tools():
    settings = _settings()
    assert evaluate(_action("read_document", document_id="d"), "reader", settings) == PolicyDecision(
        outcome="allow", reason="", rule=None
    )
    assert evaluate(_action("lookup_record", record_id="r"), "reader", settings) == PolicyDecision(
        outcome="allow", reason="", rule=None
    )


def test_allow_operator_draft_message():
    decision = evaluate(_action("draft_message", to="a", subject="s", body="b"), "operator", _settings())
    assert (decision.outcome, decision.rule) == ("allow", None)


def test_block_unknown_tool():
    decision = evaluate(_action("delete_everything"), "operator", _settings())
    assert (decision.outcome, decision.rule) == ("block", "unknown_tool")
    assert "delete_everything" in decision.reason


@pytest.mark.parametrize("tool", ["draft_message", "send_email", "update_status"])
def test_block_reader_denied_tools(tool):
    decision = evaluate(_action(tool), "reader", _settings())
    assert (decision.outcome, decision.rule) == ("block", "role_denied")
    assert "reader" in decision.reason


def test_block_unknown_role():
    decision = evaluate(_action("read_document"), "admin", _settings())
    assert (decision.outcome, decision.rule) == ("block", "role_denied")
    assert "admin" in decision.reason


def test_send_email_allowed_domain_needs_approval():
    action = _action("send_email", to="user@example.com", subject="s", body="b")
    decision = evaluate(action, "operator", _settings())
    assert (decision.outcome, decision.rule) == ("require_approval", "high_risk")


def test_send_email_domain_match_is_case_insensitive():
    action = _action("send_email", to="User@Example.COM", subject="s", body="b")
    assert evaluate(action, "operator", _settings()).outcome == "require_approval"


def test_block_send_email_outside_allowlist():
    action = _action("send_email", to="user@evil.test", subject="s", body="b")
    decision = evaluate(action, "operator", _settings())
    assert (decision.outcome, decision.rule) == ("block", "email_domain")
    assert "evil.test" in decision.reason


def test_block_send_email_subdomain_not_covered():
    action = _action("send_email", to="user@mail.example.com", subject="s", body="b")
    decision = evaluate(action, "operator", _settings())
    assert (decision.outcome, decision.rule) == ("block", "email_domain")


@pytest.mark.parametrize("to", ["", "no-at-sign", "a@", "@b", None, 123])
def test_block_send_email_invalid_recipient(to):
    action = _action("send_email", to=to, subject="s", body="b")
    decision = evaluate(action, "operator", _settings())
    assert (decision.outcome, decision.rule) == ("block", "email_domain")


def test_block_send_email_missing_recipient():
    decision = evaluate(_action("send_email", subject="s", body="b"), "operator", _settings())
    assert (decision.outcome, decision.rule) == ("block", "email_domain")


def test_require_approval_high_risk_update_status():
    decision = evaluate(_action("update_status", record_id="r", status="x"), "operator", _settings())
    assert (decision.outcome, decision.rule) == ("require_approval", "high_risk")


def test_block_loop_cap_reached():
    settings = _settings(max_tool_calls=2)
    decision = evaluate(_action("lookup_record", record_id="r"), "operator", settings, tool_call_count=2)
    assert (decision.outcome, decision.rule) == ("block", "loop_cap")


def test_allow_just_below_loop_cap():
    settings = _settings(max_tool_calls=2)
    decision = evaluate(_action("lookup_record", record_id="r"), "operator", settings, tool_call_count=1)
    assert decision.outcome == "allow"


def test_first_matching_rule_wins():
    settings = _settings(max_tool_calls=1)
    assert evaluate(_action("nope"), "operator", settings, tool_call_count=9).rule == "unknown_tool"
    assert evaluate(_action("send_email"), "reader", settings, tool_call_count=9).rule == "role_denied"
    evil = _action("send_email", to="u@evil.test")
    assert evaluate(evil, "operator", settings, tool_call_count=9).rule == "email_domain"
    assert evaluate(_action("update_status"), "operator", settings, tool_call_count=9).rule == "loop_cap"


def test_policy_roles_match_shipped_config():
    roles = load_roles("config/roles.json")
    assert set(ROLE_TOOLS) == set(roles.roles)
    for role, tools in roles.roles.items():
        assert ROLE_TOOLS[role] == set(tools)
