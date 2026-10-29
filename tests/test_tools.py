"""Tool tests: registry shape, each fake tool, and error cases."""

import pytest
from pydantic import ValidationError

from agentledger import TOOLS, ToolError, default_context, run_tool


@pytest.fixture
def ctx():
    return default_context()


def test_registry_has_all_five_tools_with_risks():
    assert set(TOOLS) == {
        "read_document",
        "lookup_record",
        "draft_message",
        "send_email",
        "update_status",
    }
    assert TOOLS["read_document"].risk == "low"
    assert TOOLS["lookup_record"].risk == "low"
    assert TOOLS["draft_message"].risk == "low"
    assert TOOLS["send_email"].risk == "high"
    assert TOOLS["update_status"].risk == "high"


def test_read_document(ctx):
    result = run_tool("read_document", {"document_id": "doc-001"}, ctx)
    assert result["document_id"] == "doc-001"
    assert "refund" in result["text"]


def test_read_document_unknown_id(ctx):
    with pytest.raises(ToolError, match="unknown document_id"):
        run_tool("read_document", {"document_id": "nope"}, ctx)


def test_lookup_record_returns_a_copy(ctx):
    result = run_tool("lookup_record", {"record_id": "rec-001"}, ctx)
    assert result["record"]["status"] == "open"
    result["record"]["status"] = "mutated"
    assert ctx.records["rec-001"]["status"] == "open"


def test_lookup_record_unknown_id(ctx):
    with pytest.raises(ToolError, match="unknown record_id"):
        run_tool("lookup_record", {"record_id": "nope"}, ctx)


def test_draft_message_does_not_send(ctx):
    result = run_tool(
        "draft_message",
        {"to": "a@example.com", "subject": "hi", "body": "hello"},
        ctx,
    )
    assert result["draft_id"]
    assert len(ctx.drafts) == 1
    assert ctx.outbox == []


def test_send_email_goes_to_outbox_only(ctx):
    result = run_tool(
        "send_email",
        {"to": "a@example.com", "subject": "hi", "body": "hello"},
        ctx,
    )
    assert result["sent"] is True
    assert ctx.outbox == [{"to": "a@example.com", "subject": "hi", "body": "hello"}]


def test_update_status(ctx):
    result = run_tool("update_status", {"record_id": "rec-001", "status": "closed"}, ctx)
    assert result == {"record_id": "rec-001", "old_status": "open", "new_status": "closed"}
    assert ctx.records["rec-001"]["status"] == "closed"


def test_update_status_unknown_record(ctx):
    with pytest.raises(ToolError, match="unknown record_id"):
        run_tool("update_status", {"record_id": "nope", "status": "closed"}, ctx)


def test_update_status_rejects_empty_status(ctx):
    with pytest.raises(ValidationError):
        run_tool("update_status", {"record_id": "rec-001", "status": ""}, ctx)


def test_unknown_tool(ctx):
    with pytest.raises(ToolError, match="unknown tool"):
        run_tool("delete_everything", {}, ctx)


def test_bad_args_rejected(ctx):
    with pytest.raises(ValidationError):
        run_tool("read_document", {}, ctx)
    with pytest.raises(ValidationError):
        run_tool("send_email", {"to": "a@example.com"}, ctx)
