"""Tests for agents/base_agent.py — Ollama call wrapper."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.base_agent import BaseAgent


class TestBaseAgentInit:
    def test_http_prefix_added_when_missing(self):
        agent = BaseAgent(model="llama3", host="localhost:11434")
        assert agent.host.startswith("http://")

    def test_https_host_left_unchanged(self):
        agent = BaseAgent(model="llama3", host="https://my-server:11434")
        assert agent.host == "https://my-server:11434"

    def test_0000_replaced_with_localhost(self):
        agent = BaseAgent(model="llama3", host="http://0.0.0.0:11434")
        assert "0.0.0.0" not in agent.host
        assert "localhost" in agent.host

    def test_model_stored(self):
        agent = BaseAgent(model="deepseek-r1:7b", host="localhost")
        assert agent.model == "deepseek-r1:7b"

    def test_analyze_raises_not_implemented(self):
        agent = BaseAgent(model="x", host="localhost")
        with pytest.raises(NotImplementedError):
            agent.analyze("anything")


class TestBaseAgentGenerate:
    def _agent(self):
        return BaseAgent(model="test-model", host="http://localhost:11434")

    def _mock_ollama(self, text: str):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": text}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_returns_response_text(self):
        agent = self._agent()
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = (
                self._mock_ollama("Hello from Ollama.")
            )
            result = agent.generate("Test prompt")
        assert result == "Hello from Ollama."

    def test_strips_think_tags(self):
        agent = self._agent()
        raw = "<think>Internal reasoning here.</think>Final answer."
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = (
                self._mock_ollama(raw)
            )
            result = agent.generate("Reason about this.")
        assert result == "Final answer."
        assert "<think>" not in result

    def test_strips_multiline_think_tags(self):
        agent = self._agent()
        raw = "<think>\nstep 1\nstep 2\n</think>Result text."
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = (
                self._mock_ollama(raw)
            )
            result = agent.generate("prompt")
        assert result == "Result text."

    def test_empty_response_returns_empty_string(self):
        agent = self._agent()
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = (
                self._mock_ollama("")
            )
            result = agent.generate("prompt")
        assert result == ""

    def test_raises_on_http_status_error(self):
        agent = self._agent()
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.side_effect = (
                httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
            )
            with pytest.raises(httpx.HTTPStatusError):
                agent.generate("prompt")

    def test_raises_on_connect_error(self):
        agent = self._agent()
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.side_effect = (
                httpx.ConnectError("connection refused")
            )
            with pytest.raises(httpx.ConnectError):
                agent.generate("prompt")

    def test_text_after_multiple_think_blocks_returned(self):
        agent = self._agent()
        raw = "<think>block1</think>middle<think>block2</think>end"
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = (
                self._mock_ollama(raw)
            )
            result = agent.generate("prompt")
        assert "block1" not in result
        assert "block2" not in result
        assert "middle" in result
        assert "end" in result
