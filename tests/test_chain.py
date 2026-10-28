"""Hash-chain tests: append, verify, and every tamper case from the demo."""

import json
import sqlite3

import pytest

from agentledger import (
    GENESIS_PREV_HASH,
    EventType,
    LedgerStore,
    canonical_json,
)


@pytest.fixture
def store(tmp_path):
    with LedgerStore(tmp_path / "test.db") as s:
        yield s


def _run(store, task="do the thing", role="operator"):
    return store.create_run(task=task, agent_role=role)


def _tamper(db_path, sql, params=()):
    db = sqlite3.connect(str(db_path))
    db.execute(sql, params)
    db.commit()
    db.close()


def test_empty_chain_verifies_ok(store):
    result = store.verify_chain()
    assert result.ok and result.broken_seq is None


def test_append_and_verify_ok(store):
    run = _run(store)
    e1 = store.append_event(run.id, EventType.MODEL_CALL, "agent", {"model": "spark"})
    e2 = store.append_event(run.id, "action_proposed", "agent", {"tool": "read_document"})
    e3 = store.append_event(run.id, EventType.TOOL_EXECUTED, "wrapper", {"ok": True})

    assert (e1.seq, e2.seq, e3.seq) == (1, 2, 3)
    assert e1.prev_hash == GENESIS_PREV_HASH
    assert e2.prev_hash == e1.hash
    assert e3.prev_hash == e2.hash
    assert store.verify_chain().ok


def test_chain_is_global_across_runs(store):
    a = _run(store, task="first")
    b = _run(store, task="second")
    store.append_event(a.id, "model_call", "agent", {})
    store.append_event(b.id, "model_call", "agent", {})
    store.append_event(a.id, "run_finished", "wrapper", {})

    assert [e.seq for e in store.list_events()] == [1, 2, 3]
    assert [e.seq for e in store.list_events(a.id)] == [1, 3]
    assert store.verify_chain().ok


def test_tampered_payload_detected(store, tmp_path):
    run = _run(store)
    for i in range(3):
        store.append_event(run.id, "model_call", "agent", {"n": i})
    _tamper(tmp_path / "test.db", "UPDATE events SET payload = ? WHERE seq = 2", ('{"n": 999}',))

    result = store.verify_chain()
    assert not result.ok and result.broken_seq == 2


def test_tampered_hash_detected(store, tmp_path):
    run = _run(store)
    for i in range(3):
        store.append_event(run.id, "model_call", "agent", {"n": i})
    _tamper(tmp_path / "test.db", "UPDATE events SET hash = ? WHERE seq = 1", ("0" * 64,))

    result = store.verify_chain()
    assert not result.ok and result.broken_seq == 1


def test_tampered_prev_hash_detected(store, tmp_path):
    run = _run(store)
    for i in range(3):
        store.append_event(run.id, "model_call", "agent", {"n": i})
    _tamper(tmp_path / "test.db", "UPDATE events SET prev_hash = ? WHERE seq = 3", ("forged",))

    result = store.verify_chain()
    assert not result.ok and result.broken_seq == 3


def test_tampered_timestamp_detected(store, tmp_path):
    run = _run(store)
    store.append_event(run.id, "model_call", "agent", {})
    _tamper(
        tmp_path / "test.db",
        "UPDATE events SET ts = ? WHERE seq = 1",
        ("2000-01-01T00:00:00+00:00",),
    )

    result = store.verify_chain()
    assert not result.ok and result.broken_seq == 1


def test_deleted_middle_event_detected(store, tmp_path):
    run = _run(store)
    for i in range(3):
        store.append_event(run.id, "model_call", "agent", {"n": i})
    _tamper(tmp_path / "test.db", "DELETE FROM events WHERE seq = 2")

    result = store.verify_chain()
    assert not result.ok and result.broken_seq == 3


def test_first_broken_seq_returned(store, tmp_path):
    run = _run(store)
    for i in range(4):
        store.append_event(run.id, "model_call", "agent", {"n": i})
    _tamper(tmp_path / "test.db", "UPDATE events SET payload = ? WHERE seq = 2", ('{"n": 999}',))
    _tamper(tmp_path / "test.db", "UPDATE events SET payload = ? WHERE seq = 4", ('{"n": 999}',))

    result = store.verify_chain()
    assert not result.ok and result.broken_seq == 2


def test_corrupt_payload_json_detected(store, tmp_path):
    run = _run(store)
    store.append_event(run.id, "model_call", "agent", {"n": 1})
    _tamper(tmp_path / "test.db", "UPDATE events SET payload = ? WHERE seq = 1", ("{broken",))

    result = store.verify_chain()
    assert not result.ok and result.broken_seq == 1


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": [3, 2]}) == '{"a":[3,2],"b":1}'
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    json.loads(canonical_json({"k": "ünïcodé ✓"}))  # UTF-8 safe


def test_payload_key_order_does_not_change_hash(store):
    run = _run(store)
    e1 = store.append_event(run.id, "model_call", "agent", {"a": 1, "b": 2}, ts="2026-01-01T00:00:00+00:00")
    e2 = store.append_event(run.id, "model_call", "agent", {"b": 2, "a": 1}, ts="2026-01-01T00:00:00+00:00")
    # Same body except seq/prev_hash; canonical payload bytes are identical.
    assert canonical_json(e1.payload) == canonical_json(e2.payload)
    assert store.verify_chain().ok


def test_unknown_event_type_rejected(store):
    run = _run(store)
    with pytest.raises(ValueError):
        store.append_event(run.id, "not_a_real_type", "agent", {})


def test_append_to_unknown_run_rejected(store):
    with pytest.raises(ValueError, match="unknown run_id"):
        store.append_event("missing", "model_call", "agent", {})


def test_non_serializable_payload_rejected(store):
    run = _run(store)
    with pytest.raises(TypeError):
        store.append_event(run.id, "model_call", "agent", {"bad": object()})
