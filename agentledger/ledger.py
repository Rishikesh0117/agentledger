"""Hash-chained audit log for AgentLedger.

Every step of the agent pipeline appends an event to the ``events`` table.
Each event stores the hash of the previous event, forming one global chain
across all runs, so any edit, reorder, or deletion in the middle breaks
verification at the first affected entry.

Hash formula (from PLAN.md)::

    hash = SHA256(prev_hash + canonical_json(event_without_hash))

``event_without_hash`` is every stored field except ``hash`` itself:
seq, run_id, ts, event_type, actor, payload, prev_hash. The first event
uses the fixed ``GENESIS_PREV_HASH`` as its ``prev_hash``.

Known limit: deleting the *tail* of the chain (the newest events) leaves
the remaining prefix valid. Detecting tail deletion needs an external
checkpoint of the tip hash, which is out of scope for the demo.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

GENESIS_PREV_HASH = "GENESIS"


class EventType(str, Enum):
    """Every pipeline step that writes an event."""

    MODEL_CALL = "model_call"
    ACTION_PROPOSED = "action_proposed"
    POLICY_BLOCKED = "policy_blocked"
    GUARDIAN_VERDICT = "guardian_verdict"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECISION = "approval_decision"
    TOOL_EXECUTED = "tool_executed"
    TOOL_FAILED = "tool_failed"
    RUN_FINISHED = "run_finished"


class Run(BaseModel):
    id: str
    task: str = Field(min_length=1)
    agent_role: str = Field(min_length=1)
    started_at: datetime
    status: str = "running"


class Event(BaseModel):
    seq: int
    run_id: str
    ts: datetime
    event_type: EventType
    actor: str = Field(min_length=1)
    payload: dict[str, Any]
    prev_hash: str
    hash: str


@dataclass
class VerifyResult:
    ok: bool
    broken_seq: int | None = None
    reason: str = ""


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace. Encoded as UTF-8 when hashed or stored."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_hash(prev_hash: str, body: dict[str, Any]) -> str:
    """Hash one event body, chained to the previous event's hash."""
    return hashlib.sha256((prev_hash + canonical_json(body)).encode("utf-8")).hexdigest()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_ts(ts: datetime | str | None) -> str:
    """Normalize a timestamp to the exact string that is stored and hashed."""
    if ts is None:
        return _utcnow_iso()
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    datetime.fromisoformat(ts)  # validate, raises ValueError if bad
    return ts


class LedgerStore:
    """SQLite-backed audit log. Appending events is the only write operation."""

    def __init__(self, path: str | Path):
        self._db = sqlite3.connect(str(path))
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                agent_role TEXT NOT NULL,
                started_at TEXT NOT NULL,
                status TEXT NOT NULL
            )"""
        )
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(id),
                ts TEXT NOT NULL,
                event_type TEXT NOT NULL,
                actor TEXT NOT NULL,
                payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL
            )"""
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> LedgerStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def create_run(self, task: str, agent_role: str, run_id: str | None = None) -> Run:
        run = Run(
            id=run_id or uuid.uuid4().hex,
            task=task,
            agent_role=agent_role,
            started_at=datetime.now(timezone.utc),
        )
        self._db.execute(
            "INSERT INTO runs (id, task, agent_role, started_at, status) VALUES (?, ?, ?, ?, ?)",
            (run.id, run.task, run.agent_role, run.started_at.isoformat(), run.status),
        )
        self._db.commit()
        return run

    def append_event(
        self,
        run_id: str,
        event_type: EventType | str,
        actor: str,
        payload: dict[str, Any],
        ts: datetime | str | None = None,
    ) -> Event:
        event_type = EventType(event_type)  # raises ValueError on unknown type
        canonical_json(payload)  # raises TypeError if not JSON-serializable
        if self._db.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
            raise ValueError(f"unknown run_id: {run_id!r}")
        ts_str = _coerce_ts(ts)

        row = self._db.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM events").fetchone()
        seq: int = row[0]
        prev_row = self._db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        prev_hash: str = prev_row[0] if prev_row else GENESIS_PREV_HASH

        body = {
            "seq": seq,
            "run_id": run_id,
            "ts": ts_str,
            "event_type": event_type.value,
            "actor": actor,
            "payload": payload,
            "prev_hash": prev_hash,
        }
        event_hash = compute_hash(prev_hash, body)
        self._db.execute(
            """INSERT INTO events
               (seq, run_id, ts, event_type, actor, payload, prev_hash, hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (seq, run_id, ts_str, event_type.value, actor, canonical_json(payload), prev_hash, event_hash),
        )
        self._db.commit()
        return Event(
            seq=seq,
            run_id=run_id,
            ts=datetime.fromisoformat(ts_str),
            event_type=event_type,
            actor=actor,
            payload=payload,
            prev_hash=prev_hash,
            hash=event_hash,
        )

    def list_events(self, run_id: str | None = None) -> list[Event]:
        """All events in chain order, optionally filtered to one run."""
        if run_id is None:
            rows = self._db.execute(
                "SELECT seq, run_id, ts, event_type, actor, payload, prev_hash, hash"
                " FROM events ORDER BY seq"
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT seq, run_id, ts, event_type, actor, payload, prev_hash, hash"
                " FROM events WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
        return [
            Event(
                seq=seq,
                run_id=rid,
                ts=datetime.fromisoformat(ts),
                event_type=EventType(event_type),
                actor=actor,
                payload=json.loads(payload),
                prev_hash=prev_hash,
                hash=event_hash,
            )
            for seq, rid, ts, event_type, actor, payload, prev_hash, event_hash in rows
        ]

    def verify_chain(self) -> VerifyResult:
        """Walk all events in order. OK, or the first broken seq."""
        rows = self._db.execute(
            "SELECT seq, run_id, ts, event_type, actor, payload, prev_hash, hash"
            " FROM events ORDER BY seq"
        ).fetchall()
        expected_prev = GENESIS_PREV_HASH
        for seq, run_id, ts, event_type, actor, payload_text, prev_hash, event_hash in rows:
            if prev_hash != expected_prev:
                return VerifyResult(
                    ok=False, broken_seq=seq, reason="prev_hash does not match previous event's hash"
                )
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                return VerifyResult(ok=False, broken_seq=seq, reason="payload is not valid JSON")
            body = {
                "seq": seq,
                "run_id": run_id,
                "ts": ts,
                "event_type": event_type,
                "actor": actor,
                "payload": payload,
                "prev_hash": prev_hash,
            }
            if compute_hash(prev_hash, body) != event_hash:
                return VerifyResult(ok=False, broken_seq=seq, reason="hash mismatch")
            expected_prev = event_hash
        return VerifyResult(ok=True)
