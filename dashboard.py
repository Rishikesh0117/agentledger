"""AgentLedger audit dashboard (Streamlit).

Run from the repo root:

    streamlit run dashboard.py

Shows the run list, an event timeline for the selected run, blocked
actions with reasons, and a hash-chain verify button. Streamlit is
imported lazily inside main() so the pure helpers stay testable
without it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agentledger.config import load_settings
from agentledger.ledger import Event, EventType, LedgerStore


def summarize_event(event: Event) -> str:
    """One-line human summary for a timeline entry."""
    payload = event.payload
    kind = event.event_type
    if kind == EventType.MODEL_CALL:
        text = f"model {payload.get('model', '?')}"
        if payload.get("finish"):
            text += " (finish)"
        ms = payload.get("latency_ms")
        if isinstance(ms, int):
            text += f" · {ms}ms"
        return text
    if kind == EventType.ACTION_PROPOSED:
        return str(payload.get("tool", "?"))
    if kind == EventType.POLICY_BLOCKED:
        return f"{payload.get('tool', '?')} [{payload.get('rule', '?')}]"
    if kind == EventType.GUARDIAN_VERDICT:
        return f"{payload.get('tool', '?')}: {payload.get('verdict', '?')}"
    if kind == EventType.APPROVAL_REQUESTED:
        return str(payload.get("tool", "?"))
    if kind == EventType.APPROVAL_DECISION:
        decision = "approved" if payload.get("approved") else "denied"
        return f"{payload.get('tool', '?')}: {decision}"
    if kind == EventType.TOOL_EXECUTED:
        return str(payload.get("tool", "?"))
    if kind == EventType.TOOL_FAILED:
        return f"{payload.get('tool', '?')} failed"
    if kind == EventType.RUN_FINISHED:
        return (
            f"{payload.get('status', '?')} "
            f"({payload.get('steps', '?')} steps, {payload.get('tool_calls', '?')} tool calls)"
        )
    return ""


def blocked_events(events: list[Event]) -> list[Event]:
    """Policy-blocked events, in order."""
    return [e for e in events if e.event_type == EventType.POLICY_BLOCKED]


def get_db_path_default() -> str:
    """Default DB path from shipped config, or the built-in fallback."""
    try:
        return load_settings("config/settings.json", dotenv_path=None).db_path
    except (OSError, ValueError):
        return "agentledger.db"


def main() -> None:
    import streamlit as st

    st.title("AgentLedger — audit dashboard")

    db_path = st.sidebar.text_input("Audit DB path", value=get_db_path_default())
    if st.sidebar.button("Reload"):
        st.rerun()

    if not Path(db_path).exists():
        st.error(
            f"No database at {db_path}. Run a demo scenario first: "
            f"`python -m agentledger.demo --scenario both --db {db_path} --approval auto`"
        )
        return
    try:
        store = LedgerStore(db_path)
    except (sqlite3.Error, OSError) as e:
        st.error(f"Cannot open {db_path}: {e}")
        return
    try:
        if st.sidebar.button("Verify audit log"):
            st.session_state["verify_result"] = store.verify_chain()
            st.session_state["verify_events"] = len(store.list_events())
        if "verify_result" in st.session_state:
            result = st.session_state["verify_result"]
            if result.ok:
                st.sidebar.success(f"Audit log OK ({st.session_state['verify_events']} events checked)")
            else:
                st.sidebar.error(f"Broken at seq {result.broken_seq}: {result.reason}")

        runs = store.list_runs()
        if not runs:
            st.info("No runs in this database yet.")
            return
        labels = {f"{run.id[:8]} · {run.status} · {run.task[:50]}": run for run in runs}
        run = labels[st.sidebar.selectbox("Run", list(labels))]

        st.header(f"Run {run.id[:8]}")
        st.write(f"**Task:** {run.task}")
        st.write(f"**Role:** {run.agent_role} · **Status:** {run.status} · **Started:** {run.started_at}")

        events = store.list_events(run.id)
        blocked = blocked_events(events)
        st.subheader(f"Blocked actions ({len(blocked)})")
        if not blocked:
            st.write("None — every action passed the gates.")
        for event in blocked:
            payload = event.payload
            st.error(
                f"Step {payload.get('step', '?')}: {payload.get('tool', '?')} "
                f"[{payload.get('rule', '?')}] — {payload.get('reason', '')}"
            )

        st.subheader(f"Event timeline ({len(events)})")
        for event in events:
            with st.expander(f"[{event.seq}] {event.event_type.value} — {summarize_event(event)}"):
                st.json(
                    {
                        "seq": event.seq,
                        "run_id": event.run_id,
                        "ts": event.ts.isoformat(),
                        "event_type": event.event_type.value,
                        "actor": event.actor,
                        "payload": event.payload,
                    }
                )
    finally:
        store.close()


if __name__ == "__main__":
    main()
