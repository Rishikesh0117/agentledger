"""Model tests: HTTP client, strict/JSON parsing, and the model agent.

Everything here runs offline: HTTP tests monkeypatch ``urlopen``, and
agent tests use a fake client double. No test touches the network.
"""

import io
import json
import urllib.error

import pytest
from pydantic import ValidationError

from agentledger import (
    API_KEY_ENV_VAR,
    FINISH_TOOL,
    ChatResult,
    LedgerStore,
    ModelAgent,
    ModelClient,
    ModelError,
    Settings,
    ToolCall,
    default_context,
    extract_json,
    load_settings,
    run_agent,
)


class FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _chat_payload(content="", tool_calls=None):
    message = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return json.dumps({"choices": [{"message": message}]}).encode("utf-8")


def _tool_call(name, args):
    return {"function": {"name": name, "arguments": json.dumps(args)}}


class FakeClient:
    """Stand-in for ModelClient: replays canned results, records calls."""

    model = "fake-model"

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self._results.pop(0)


def _patch_urlopen(monkeypatch, payload=None, error=None):
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["request"] = request
        seen["timeout"] = timeout
        if error is not None:
            raise error
        return FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return seen


def _client():
    return ModelClient(model="muse-spark", api_key="test-key", api_base="https://api.test/v1")


def test_chat_posts_model_messages_and_auth(monkeypatch):
    seen = _patch_urlopen(monkeypatch, _chat_payload('{"tool": "finish", "args": {}}'))
    result = _client().chat([{"role": "user", "content": "hi"}])

    request = seen["request"]
    assert request.full_url == "https://api.test/v1/chat/completions"
    assert json.loads(request.data) == {
        "model": "muse-spark",
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert request.get_header("Authorization") == "Bearer test-key"
    assert request.get_header("Content-type") == "application/json"
    assert seen["timeout"] == 30
    assert result == ChatResult(content='{"tool": "finish", "args": {}}', tool_calls=[])


def test_chat_trailing_slash_base_still_works(monkeypatch):
    seen = _patch_urlopen(monkeypatch, _chat_payload("hi"))
    client = ModelClient(model="m", api_key="k", api_base="https://api.test/v1/")
    client.chat([{"role": "user", "content": "hi"}])
    assert seen["request"].full_url == "https://api.test/v1/chat/completions"


def test_chat_parses_tool_calls(monkeypatch):
    payload = _chat_payload("", [_tool_call("read_document", {"document_id": "doc-001"})])
    seen = _patch_urlopen(monkeypatch, payload)
    result = _client().chat([{"role": "user", "content": "hi"}])

    assert result.tool_calls == [ToolCall(name="read_document", args={"document_id": "doc-001"})]
    assert "tools" not in json.loads(seen["request"].data)


def test_chat_sends_tools_when_provided(monkeypatch):
    seen = _patch_urlopen(monkeypatch, _chat_payload("hi"))
    _client().chat([{"role": "user", "content": "hi"}], tools=[{"name": "read_document"}])
    assert json.loads(seen["request"].data)["tools"] == [{"name": "read_document"}]


def test_missing_api_key_raises_before_http(monkeypatch):
    def exploding_urlopen(request, timeout=None):
        raise AssertionError("must not touch the network without a key")

    monkeypatch.setattr("urllib.request.urlopen", exploding_urlopen)
    with pytest.raises(ModelError, match="META_MODEL_API_KEY"):
        ModelClient(model="m", api_key="", api_base="https://api.test/v1")


def _required_settings(**overrides):
    values = {"agent_model": "a", "guardian_model": "g", "model_api_base": "https://b.test"}
    values.update(overrides)
    return Settings(**values)


def test_from_settings_reads_key_from_env(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "env-key")
    client = ModelClient.from_settings(_required_settings())
    assert (client.model, client._api_base, client.timeout) == ("a", "https://b.test", 30)


def test_from_settings_without_key_raises(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    with pytest.raises(ModelError, match="META_MODEL_API_KEY"):
        ModelClient.from_settings(_required_settings())


def test_http_error_raises_model_error(monkeypatch):
    error = urllib.error.HTTPError(
        "https://api.test/v1/chat/completions", 500, "boom", {}, io.BytesIO(b"boom")
    )
    _patch_urlopen(monkeypatch, error=error)
    with pytest.raises(ModelError, match="HTTP 500"):
        _client().chat([{"role": "user", "content": "hi"}])


def test_unreachable_api_raises_model_error(monkeypatch):
    _patch_urlopen(monkeypatch, error=urllib.error.URLError("no route"))
    with pytest.raises(ModelError, match="unreachable"):
        _client().chat([{"role": "user", "content": "hi"}])


def test_invalid_json_response_raises(monkeypatch):
    _patch_urlopen(monkeypatch, b"{broken")
    with pytest.raises(ModelError, match="invalid JSON"):
        _client().chat([{"role": "user", "content": "hi"}])


def test_response_without_choices_raises(monkeypatch):
    _patch_urlopen(monkeypatch, json.dumps({"oops": []}).encode())
    with pytest.raises(ModelError, match="no choices"):
        _client().chat([{"role": "user", "content": "hi"}])


def test_tool_call_with_bad_arguments_raises(monkeypatch):
    payload = _chat_payload("", [{"function": {"name": "read_document", "arguments": "{bad"}}])
    _patch_urlopen(monkeypatch, payload)
    with pytest.raises(ModelError, match="invalid JSON arguments"):
        _client().chat([{"role": "user", "content": "hi"}])


def test_tool_call_with_non_object_arguments_raises(monkeypatch):
    payload = _chat_payload("", [{"function": {"name": "read_document", "arguments": "[1]"}}])
    _patch_urlopen(monkeypatch, payload)
    with pytest.raises(ModelError, match="must be an object"):
        _client().chat([{"role": "user", "content": "hi"}])


def test_extract_json_plain():
    assert extract_json('  {"tool": "finish", "args": {}}  ') == '{"tool": "finish", "args": {}}'


def test_extract_json_strips_fences():
    reply = '```json\n{"tool": "read_document", "args": {"document_id": "doc-001"}}\n```'
    assert extract_json(reply) == '{"tool": "read_document", "args": {"document_id": "doc-001"}}'


def test_extract_json_fences_without_object_raises():
    with pytest.raises(ModelError, match="no JSON object"):
        extract_json("```\njust words\n```")


def _agent(client, **kwargs):
    return ModelAgent(task="summarize doc-001", role="reader", client=client, **kwargs)


def test_propose_parses_json_action():
    client = FakeClient([ChatResult(content='{"tool": "read_document", "args": {"document_id": "doc-001"}}')])
    action = _agent(client).propose()
    assert (action.tool, action.args) == ("read_document", {"document_id": "doc-001"})


def test_propose_strips_code_fences():
    reply = '```json\n{"tool": "lookup_record", "args": {"record_id": "rec-001"}}\n```'
    action = _agent(FakeClient([ChatResult(content=reply)])).propose()
    assert (action.tool, action.args) == ("lookup_record", {"record_id": "rec-001"})


def test_propose_finish_returns_none():
    reply = json.dumps({"tool": FINISH_TOOL, "args": {}})
    assert _agent(FakeClient([ChatResult(content=reply)])).propose() is None


def test_propose_invalid_action_raises_model_error():
    with pytest.raises(ModelError, match="not return a valid action"):
        _agent(FakeClient([ChatResult(content="not json")])).propose()
    with pytest.raises(ModelError, match="not return a valid action"):
        _agent(FakeClient([ChatResult(content='{"nope": 1}')])).propose()


def test_native_tool_call_preferred_over_content():
    client = FakeClient(
        [ChatResult(content="ignored", tool_calls=[ToolCall("read_document", {"document_id": "d"})])]
    )
    action = _agent(client, native_tools=True).propose()
    assert (action.tool, action.args) == ("read_document", {"document_id": "d"})
    sent_tools = client.calls[0]["tools"]
    assert {t["name"] for t in sent_tools} >= {"read_document", FINISH_TOOL}


def test_native_finish_tool_call_returns_none():
    client = FakeClient([ChatResult(content="", tool_calls=[ToolCall(FINISH_TOOL, {})])])
    assert _agent(client, native_tools=True).propose() is None


def test_native_falls_back_to_json_content():
    reply = '{"tool": "lookup_record", "args": {"record_id": "rec-001"}}'
    action = _agent(FakeClient([ChatResult(content=reply)]), native_tools=True).propose()
    assert action.tool == "lookup_record"


def test_observe_feeds_next_proposal():
    client = FakeClient(
        [
            ChatResult(content='{"tool": "read_document", "args": {"document_id": "doc-001"}}'),
            ChatResult(content=json.dumps({"tool": FINISH_TOOL, "args": {}})),
        ]
    )
    agent = _agent(client)
    action = agent.propose()
    agent.observe(action, {"status": "executed", "result": {"text": "hello"}})
    assert agent.propose() is None

    second_call_messages = client.calls[1]["messages"]
    observation = json.loads(second_call_messages[-1]["content"])
    assert observation["outcome"] == {"status": "executed", "result": {"text": "hello"}}
    assert observation["action"]["tool"] == "read_document"


def test_system_prompt_mentions_task_role_and_tools():
    agent = _agent(FakeClient([]))
    system = agent.messages[0]["content"]
    assert "summarize doc-001" in system
    assert "reader" in system
    for tool in ("read_document", "lookup_record", "draft_message", "send_email", "update_status"):
        assert tool in system


def test_agent_name_defaults_to_model():
    assert _agent(FakeClient([])).name == "fake-model"
    assert _agent(FakeClient([]), name="custom").name == "custom"


def test_settings_requires_model_name_and_base_url():
    with pytest.raises(ValidationError) as exc_info:
        Settings()
    missing = {".".join(map(str, err["loc"])) for err in exc_info.value.errors()}
    assert {"agent_model", "guardian_model", "model_api_base"} <= missing


def test_shipped_settings_load_model_connection():
    settings = load_settings("config/settings.json", dotenv_path=None)
    assert settings.model_api_base == "https://api.meta.ai/v1"
    assert settings.model_timeout == 30


@pytest.fixture
def store(tmp_path):
    with LedgerStore(tmp_path / "test.db") as s:
        yield s


@pytest.fixture
def ctx():
    return default_context()


def test_load_settings_custom_values_used_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "env-key")
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "agent_model": "custom-agent",
                "guardian_model": "custom-guardian",
                "model_api_base": "https://custom.test/v2",
            }
        ),
        encoding="utf-8",
    )
    client = ModelClient.from_settings(load_settings(str(path), dotenv_path=None))
    assert (client.model, client._api_base) == ("custom-agent", "https://custom.test/v2")


def test_from_settings_uses_configured_values(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "env-key")
    settings = _required_settings(agent_model="custom-model", model_api_base="https://custom.test/v2")
    client = ModelClient.from_settings(settings)
    assert (client.model, client._api_base) == ("custom-model", "https://custom.test/v2")


def test_pipeline_model_call_uses_configured_model_name(store, ctx):
    client = FakeClient(
        [
            ChatResult(content='{"tool": "lookup_record", "args": {"record_id": "rec-001"}}'),
            ChatResult(content=json.dumps({"tool": FINISH_TOOL, "args": {}})),
        ]
    )
    client.model = "custom-configured-model"
    result = run_agent("t", "operator", _agent(client), store, ctx)

    calls = [e for e in store.list_events(result.run.id) if e.event_type.value == "model_call"]
    assert [c.payload["model"] for c in calls] == ["custom-configured-model"] * 2


def test_finish_call_recorded_as_model_call(store, ctx):
    client = FakeClient(
        [
            ChatResult(content='{"tool": "lookup_record", "args": {"record_id": "rec-001"}}'),
            ChatResult(content=json.dumps({"tool": FINISH_TOOL, "args": {}})),
        ]
    )
    result = run_agent("t", "operator", _agent(client), store, ctx)

    events = store.list_events(result.run.id)
    assert [e.event_type.value for e in events] == [
        "model_call",
        "action_proposed",
        "tool_executed",
        "model_call",
        "run_finished",
    ]
    finish_call = events[3]
    assert finish_call.payload["finish"] is True
    assert finish_call.payload["model"] == "fake-model"
    assert finish_call.payload["step"] == 2
    assert result.steps == 1
    assert store.verify_chain().ok


def test_model_call_includes_latency(store, ctx):
    client = FakeClient([ChatResult(content=json.dumps({"tool": FINISH_TOOL, "args": {}}))])
    result = run_agent("t", "operator", _agent(client), store, ctx)

    calls = [e for e in store.list_events(result.run.id) if e.event_type.value == "model_call"]
    assert len(calls) == 1
    assert isinstance(calls[0].payload["latency_ms"], int)
    assert calls[0].payload["latency_ms"] >= 0
