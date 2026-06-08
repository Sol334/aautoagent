"""Tests for scripts/model_benchmark.py — fully mocked."""
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.model_benchmark as mb


# ── _get_available_models ─────────────────────────────────────────────────────

class TestGetAvailableModels:
    def test_returns_empty_list_when_ollama_is_down(self):
        import httpx
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("connection refused")
            mock_client_cls.return_value = mock_client
            result = mb._get_available_models()
        assert result == []

    def test_returns_empty_list_on_exception(self):
        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.side_effect = RuntimeError("unexpected error")
            result = mb._get_available_models()
        assert result == []

    def test_parses_model_list_correctly(self):
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "models": [
                {"name": "deepseek-r1:7b"},
                {"name": "qwen3.5:9b"},
                {"name": "llama3.2:3b"},
            ]
        }
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = fake_response
            mock_client_cls.return_value = mock_client
            result = mb._get_available_models()
        assert result == ["deepseek-r1:7b", "qwen3.5:9b", "llama3.2:3b"]

    def test_returns_empty_list_when_status_not_200(self):
        fake_response = MagicMock()
        fake_response.status_code = 503
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = fake_response
            mock_client_cls.return_value = mock_client
            result = mb._get_available_models()
        assert result == []


# ── _run_test ─────────────────────────────────────────────────────────────────

class TestRunTest:
    def _make_case(self, case_id="test_case", expected="BULLISH"):
        return {
            "id": case_id,
            "description": "Test case description",
            "trades": {"AAPL": [{"representative": "Rep A", "type": "Purchase",
                                  "amount_range": "$50k", "amount_midpoint_usd": 50000,
                                  "date": "2024-01-01", "owner": "Self"}]},
            "expected_signal": expected,
        }

    def _make_mock_client(self, response_body: str, raise_exc=None):
        fake_response = MagicMock()
        fake_response.json.return_value = {"response": response_body}
        fake_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        if raise_exc:
            mock_client.post.side_effect = raise_exc
        else:
            mock_client.post.return_value = fake_response
        return mock_client

    def test_result_has_required_keys(self):
        raw = json.dumps([{"ticker": "AAPL", "signal": "BULLISH", "confidence": 0.9,
                           "committee_edge": True, "reasoning": "test"}])
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value = self._make_mock_client(raw)
            result = mb._run_test("deepseek-r1:7b", self._make_case())
        for key in ("model", "case_id", "predicted", "expected", "elapsed_s"):
            assert key in result, f"Missing key: {key}"

    def test_captures_bullish_signal_correctly(self):
        raw = json.dumps([{"ticker": "AAPL", "signal": "BULLISH", "confidence": 0.85,
                           "committee_edge": True, "reasoning": "clear buy"}])
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value = self._make_mock_client(raw)
            result = mb._run_test("qwen3.5:9b", self._make_case(expected="BULLISH"))
        assert result["predicted"] == "BULLISH"
        assert result["correct"] is True
        assert result["json_parsed"] is True

    def test_captures_bearish_signal_correctly(self):
        raw = json.dumps([{"ticker": "QQQ", "signal": "BEARISH", "confidence": 0.7,
                           "committee_edge": False, "reasoning": "mass sell"}])
        case = self._make_case(expected="BEARISH")
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value = self._make_mock_client(raw)
            result = mb._run_test("qwen3.5:9b", case)
        assert result["predicted"] == "BEARISH"
        assert result["correct"] is True

    def test_handles_ollama_timeout_gracefully(self):
        import httpx
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value = self._make_mock_client(
                "", raise_exc=httpx.TimeoutException("timed out")
            )
            result = mb._run_test("some-model", self._make_case())
        # Must not raise — must return a dict
        assert isinstance(result, dict)
        assert result["model"] == "some-model"
        assert "error" in result or result["predicted"] == "ERROR"
        assert result["correct"] is False

    def test_handles_non_json_response_gracefully(self):
        """Model returns prose instead of JSON array."""
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value = self._make_mock_client("I think it is bullish overall.")
            result = mb._run_test("bad-model", self._make_case())
        assert result["json_parsed"] is False
        assert result["predicted"] == "ERROR"

    def test_marks_incorrect_when_signal_mismatched(self):
        raw = json.dumps([{"ticker": "AAPL", "signal": "BEARISH", "confidence": 0.5,
                           "committee_edge": False, "reasoning": "selling pressure"}])
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value = self._make_mock_client(raw)
            result = mb._run_test("model-x", self._make_case(expected="BULLISH"))
        assert result["correct"] is False


# ── run ───────────────────────────────────────────────────────────────────────

class TestRun:
    def _good_result(self, model, case):
        return {
            "case_id": case["id"],
            "model": model,
            "predicted": case["expected_signal"],
            "expected": case["expected_signal"],
            "correct": True,
            "elapsed_s": 1.0,
            "json_parsed": True,
            "confidence": 0.9,
        }

    def test_calls_run_test_for_each_model_times_case(self):
        models = ["model-a", "model-b"]
        call_log = []

        def fake_run_test(model, case):
            call_log.append((model, case["id"]))
            return self._good_result(model, case)

        with patch.object(mb, "_run_test", side_effect=fake_run_test), \
             patch.object(mb, "RESULTS_PATH", Path("/tmp/bench_test.json")):
            mb.run(models)

        expected_calls = len(models) * len(mb.TEST_CASES)
        assert len(call_log) == expected_calls
        for model in models:
            for case in mb.TEST_CASES:
                assert (model, case["id"]) in call_log

    def test_prints_summary_table(self, capsys):
        models = ["test-model"]

        def fake_run_test(model, case):
            return self._good_result(model, case)

        with patch.object(mb, "_run_test", side_effect=fake_run_test), \
             patch.object(mb, "RESULTS_PATH", Path("/tmp/bench_test2.json")):
            mb.run(models)

        captured = capsys.readouterr()
        assert "MODEL BENCHMARK" in captured.out
        assert "test-model" in captured.out
        assert "Accuracy" in captured.out or "accuracy" in captured.out.lower()

    def test_empty_models_list_does_not_crash(self, capsys):
        """run([]) with no Ollama available should call sys.exit(1) — catch it."""
        with patch.object(mb, "_get_available_models", return_value=[]):
            with pytest.raises(SystemExit) as exc_info:
                mb.run([])
        assert exc_info.value.code == 1

    def test_writes_results_json(self, tmp_path):
        models = ["model-x"]
        results_file = tmp_path / "benchmark_results.json"

        def fake_run_test(model, case):
            return self._good_result(model, case)

        with patch.object(mb, "_run_test", side_effect=fake_run_test), \
             patch.object(mb, "RESULTS_PATH", results_file):
            mb.run(models)

        assert results_file.exists()
        data = json.loads(results_file.read_text())
        assert "models" in data
        assert "results" in data
        assert data["models"] == models
        assert len(data["results"]) == len(mb.TEST_CASES)
