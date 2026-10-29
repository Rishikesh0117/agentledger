"""Fake tools with synthetic backing data.

All five tools from PLAN.md run against an in-memory :class:`ToolContext` of
clearly fake documents and records. No real side effects ever happen:
``send_email`` only appends to the context's outbox, and ``draft_message``
only records a draft.

Step 3's pipeline will call :func:`run_tool`, which validates the argument
dict against the tool's Pydantic model and dispatches to the implementation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

Risk = Literal["low", "high"]


class ToolError(Exception):
    """A tool failed: unknown tool name, missing document/record, etc."""


@dataclass
class ToolContext:
    """Synthetic backing data plus the outbox/drafts a run accumulates."""

    documents: dict[str, str] = field(default_factory=dict)
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    drafts: list[dict[str, Any]] = field(default_factory=list)
    outbox: list[dict[str, Any]] = field(default_factory=list)


def default_context() -> ToolContext:
    """Tiny synthetic dataset. Step 7 adds the demo documents; tests and
    the scripted agent build their own contexts on top of this shape."""
    return ToolContext(
        documents={
            "doc-001": (
                "Acme Corp refund policy (synthetic): refunds are approved for "
                "orders placed within 30 days. Contact support@example.com."
            ),
            "doc-002": (
                "Synthetic onboarding checklist: 1) create account, "
                "2) verify email, 3) read the refund policy."
            ),
        },
        records={
            "rec-001": {
                "id": "rec-001",
                "customer": "Test Customer A",
                "email": "customer-a@example.com",
                "status": "open",
                "summary": "Synthetic support case about a late delivery.",
            },
            "rec-002": {
                "id": "rec-002",
                "customer": "Test Customer B",
                "email": "customer-b@example.com",
                "status": "pending",
                "summary": "Synthetic support case about a damaged item.",
            },
        },
    )


class ReadDocumentArgs(BaseModel):
    document_id: str


class LookupRecordArgs(BaseModel):
    record_id: str


class DraftMessageArgs(BaseModel):
    to: str
    subject: str
    body: str


class SendEmailArgs(BaseModel):
    to: str
    subject: str
    body: str


class UpdateStatusArgs(BaseModel):
    record_id: str
    status: str = Field(min_length=1)


def read_document(args: ReadDocumentArgs, ctx: ToolContext) -> dict[str, Any]:
    try:
        text = ctx.documents[args.document_id]
    except KeyError:
        raise ToolError(f"unknown document_id: {args.document_id!r}") from None
    return {"document_id": args.document_id, "text": text}


def lookup_record(args: LookupRecordArgs, ctx: ToolContext) -> dict[str, Any]:
    try:
        record = ctx.records[args.record_id]
    except KeyError:
        raise ToolError(f"unknown record_id: {args.record_id!r}") from None
    return {"record": dict(record)}


def draft_message(args: DraftMessageArgs, ctx: ToolContext) -> dict[str, Any]:
    draft = {
        "draft_id": uuid.uuid4().hex,
        "to": args.to,
        "subject": args.subject,
        "body": args.body,
    }
    ctx.drafts.append(draft)
    return dict(draft)


def send_email(args: SendEmailArgs, ctx: ToolContext) -> dict[str, Any]:
    message = {"to": args.to, "subject": args.subject, "body": args.body}
    ctx.outbox.append(message)
    return {"sent": True, **message}


def update_status(args: UpdateStatusArgs, ctx: ToolContext) -> dict[str, Any]:
    try:
        record = ctx.records[args.record_id]
    except KeyError:
        raise ToolError(f"unknown record_id: {args.record_id!r}") from None
    old_status = record.get("status")
    record["status"] = args.status
    return {"record_id": args.record_id, "old_status": old_status, "new_status": args.status}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    risk: Risk
    description: str
    args_model: type[BaseModel]
    func: Callable[[Any, ToolContext], dict[str, Any]]


TOOLS: dict[str, ToolSpec] = {
    "read_document": ToolSpec(
        name="read_document",
        risk="low",
        description="Returns the text of a synthetic document.",
        args_model=ReadDocumentArgs,
        func=read_document,
    ),
    "lookup_record": ToolSpec(
        name="lookup_record",
        risk="low",
        description="Reads a record from a fake database.",
        args_model=LookupRecordArgs,
        func=lookup_record,
    ),
    "draft_message": ToolSpec(
        name="draft_message",
        risk="low",
        description="Drafts a message. Does not send.",
        args_model=DraftMessageArgs,
        func=draft_message,
    ),
    "send_email": ToolSpec(
        name="send_email",
        risk="high",
        description='"Sends" an email by appending it to the logged outbox.',
        args_model=SendEmailArgs,
        func=send_email,
    ),
    "update_status": ToolSpec(
        name="update_status",
        risk="high",
        description="Changes a record's status.",
        args_model=UpdateStatusArgs,
        func=update_status,
    ),
}


def get_tool(name: str) -> ToolSpec:
    try:
        return TOOLS[name]
    except KeyError:
        raise ToolError(f"unknown tool: {name!r}") from None


def run_tool(name: str, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Validate ``args`` against the tool's schema and execute it.

    Raises :class:`ToolError` for unknown tools and tool failures, and
    ``pydantic.ValidationError`` for bad arguments.
    """
    spec = get_tool(name)
    validated = spec.args_model.model_validate(args)
    return spec.func(validated, ctx)
