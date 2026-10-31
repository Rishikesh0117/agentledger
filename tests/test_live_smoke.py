"""Live smoke test: one real Muse Spark call. Skipped unless explicitly enabled.

Runs only when BOTH are present:

- ``AGENTLEDGER_RUN_LIVE_TESTS=1`` in the environment, and
- ``META_MODEL_API_KEY`` in the environment or local ``.env``.

Manual command (from the repo root, PowerShell)::

    $env:AGENTLEDGER_RUN_LIVE_TESTS=1; .venv\\Scripts\\python.exe -m pytest tests/test_live_smoke.py -q

The key is used for authorization only and is never printed. This file
makes exactly one model call.
"""

import os

import pytest

from agentledger import ModelAgent, ModelClient, ProposedAction, load_dotenv, load_settings

LIVE_FLAG = "AGENTLEDGER_RUN_LIVE_TESTS"

load_dotenv()  # so a key in the local .env counts; never overrides the real env.

pytestmark = pytest.mark.skipif(
    os.environ.get(LIVE_FLAG) != "1" or not os.environ.get("META_MODEL_API_KEY"),
    reason="live test disabled: set AGENTLEDGER_RUN_LIVE_TESTS=1 and META_MODEL_API_KEY",
)


def test_live_propose_returns_action_or_finish():
    settings = load_settings("config/settings.json")
    client = ModelClient.from_settings(settings)
    assert client.model == settings.agent_model
    agent = ModelAgent(task="Look up record rec-001, then finish.", role="operator", client=client)
    action = agent.propose()
    assert action is None or isinstance(action, ProposedAction)
