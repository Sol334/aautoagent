"""Tests for the A2A Agent Hub. All agent handle() calls are mocked."""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on the path so `api` package resolves correctly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies that the hub imports at startup so
# the test suite can run without installing torch, transformers, yfinance, etc.
# ---------------------------------------------------------------------------
_STUB_MODULES = [
    "torch", "transformers",
    "yfinance", "finnhub",
    "alpaca", "alpaca.trading", "alpaca.trading.client",
    "alpaca.trading.requests", "alpaca.trading.enums",
    "coinbase", "coinbase.rest",
    "fredapi",
]
for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

from fastapi.testclient import TestClient  # noqa: E402 — after path setup
from api.agent_hub import app, _AGENTS      # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _populate_agents():
    """Ensure _AGENTS is populated before each test (TestClient triggers startup)."""
    _AGENTS.clear()
    from api.agents.watchdog_agent import WatchdogAgent
    from api.agents.analyst_agent import AnalystAgent
    from api.agents.trader_agent import TraderAgent
    for cls in (WatchdogAgent, AnalystAgent, TraderAgent):
        inst = cls()
        _AGENTS[inst.name] = inst
    yield
    _AGENTS.clear()


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _rpc(text: str, rpc_id: int = 1) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "tasks/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": text}],
            }
        },
    }


# ---------------------------------------------------------------------------
# 1. Health endpoint
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "agents" in data


# ---------------------------------------------------------------------------
# 2. List agents
# ---------------------------------------------------------------------------

def test_list_agents(client):
    resp = client.get("/agents")
    assert resp.status_code == 200
    names = {a["name"] for a in resp.json()}
    assert {"watchdog", "analyst", "trader"}.issubset(names)


# ---------------------------------------------------------------------------
# 3. Call watchdog agent directly
# ---------------------------------------------------------------------------

def test_call_watchdog_agent(client):
    with patch.object(_AGENTS["watchdog"], "handle", return_value='{"new_trades": 0, "alerts_sent": 0}') as mock_handle:
        resp = client.post("/agents/watchdog", json=_rpc("dry run watchdog"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["jsonrpc"] == "2.0"
        assert body["result"]["status"] == "completed"
        assert body["result"]["result"] == '{"new_trades": 0, "alerts_sent": 0}'
        mock_handle.assert_called_once_with("dry run watchdog")


# ---------------------------------------------------------------------------
# 4. Unknown agent → 404
# ---------------------------------------------------------------------------

def test_call_unknown_agent_404(client):
    resp = client.post("/agents/nonexistent", json=_rpc("hello"))
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 5. Auto-route to watchdog
# ---------------------------------------------------------------------------

def test_auto_route_to_watchdog(client):
    with patch.object(_AGENTS["watchdog"], "handle", return_value='{"new_trades": 0}') as mock_handle:
        resp = client.post("/tasks/send", json=_rpc("check congressional trades"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["status"] == "completed"
        mock_handle.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Auto-route to analyst
# ---------------------------------------------------------------------------

def test_auto_route_to_analyst(client):
    with patch.object(_AGENTS["analyst"], "handle", return_value="Signal Table\nNVDA BUY") as mock_handle:
        resp = client.post("/tasks/send", json=_rpc("analyze NVDA"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["status"] == "completed"
        mock_handle.assert_called_once()


# ---------------------------------------------------------------------------
# 7. JSON-RPC error when agent.handle() raises
# ---------------------------------------------------------------------------

def test_jsonrpc_error_on_agent_crash(client):
    with patch.object(_AGENTS["watchdog"], "handle", side_effect=RuntimeError("boom")):
        resp = client.post("/agents/watchdog", json=_rpc("run watchdog"))
        assert resp.status_code == 200
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == -32603
        assert "boom" in body["error"]["message"]
