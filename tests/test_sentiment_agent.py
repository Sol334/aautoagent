"""Tests for agents/sentiment_agent.py — finBERT news sentiment scoring."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.sentiment_agent import SentimentAgent


class TestAnalyzeHeadlines:
    def setup_method(self):
        self.agent = SentimentAgent()

    def test_empty_list_returns_zero(self):
        assert self.agent.analyze_headlines([]) == 0.0

    def test_returns_zero_when_pipeline_unavailable(self):
        with patch.object(self.agent, "_get_pipeline", return_value=None):
            result = self.agent.analyze_headlines(["NVDA beats earnings estimates."])
        assert result == 0.0

    def test_positive_headlines_return_positive_score(self):
        mock_pipe = MagicMock(return_value=[
            {"label": "positive", "score": 0.9},
            {"label": "negative", "score": 0.05},
            {"label": "neutral", "score": 0.05},
        ])
        with patch.object(self.agent, "_get_pipeline", return_value=mock_pipe):
            result = self.agent.analyze_headlines(["Record earnings growth reported."])
        assert result > 0.0

    def test_negative_headlines_return_negative_score(self):
        mock_pipe = MagicMock(return_value=[
            {"label": "positive", "score": 0.05},
            {"label": "negative", "score": 0.9},
            {"label": "neutral", "score": 0.05},
        ])
        with patch.object(self.agent, "_get_pipeline", return_value=mock_pipe):
            result = self.agent.analyze_headlines(["Massive revenue miss and guidance cut."])
        assert result < 0.0

    def test_neutral_headlines_near_zero(self):
        mock_pipe = MagicMock(return_value=[
            {"label": "positive", "score": 0.33},
            {"label": "negative", "score": 0.33},
            {"label": "neutral", "score": 0.34},
        ])
        with patch.object(self.agent, "_get_pipeline", return_value=mock_pipe):
            result = self.agent.analyze_headlines(["Company holds annual meeting."])
        assert abs(result) < 0.1

    def test_caps_at_10_headlines(self):
        calls = []

        def counting_pipe(text):
            calls.append(text)
            return [{"label": "neutral", "score": 1.0}]

        with patch.object(self.agent, "_get_pipeline", return_value=counting_pipe):
            self.agent.analyze_headlines([f"Headline {i}" for i in range(20)])
        assert len(calls) == 10

    def test_pipeline_exception_returns_zero(self):
        def bad_pipe(text):
            raise RuntimeError("Model OOM")

        with patch.object(self.agent, "_get_pipeline", return_value=bad_pipe):
            result = self.agent.analyze_headlines(["Any headline."])
        assert result == 0.0

    def test_score_in_valid_range(self):
        mock_pipe = MagicMock(return_value=[
            {"label": "positive", "score": 0.7},
            {"label": "negative", "score": 0.2},
            {"label": "neutral", "score": 0.1},
        ])
        with patch.object(self.agent, "_get_pipeline", return_value=mock_pipe):
            result = self.agent.analyze_headlines(["Big rally in tech stocks."])
        assert -1.0 <= result <= 1.0


class TestFetchHeadlines:
    def setup_method(self):
        self.agent = SentimentAgent()

    def test_empty_key_returns_empty(self):
        assert self.agent.fetch_headlines("AAPL", "") == []

    def test_change_me_key_returns_empty(self):
        assert self.agent.fetch_headlines("NVDA", "CHANGE_ME") == []

    def test_mocked_finnhub_returns_headlines(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"headline": "Apple beats Q4 estimates", "summary": "Revenue up 8%."},
            {"headline": "AAPL stock hits record", "summary": ""},
        ]
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            result = self.agent.fetch_headlines("AAPL", "testkey")
        assert len(result) == 2
        assert "Apple beats Q4 estimates" in result[0]

    def test_headline_with_empty_summary_not_appended(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"headline": "Brief headline", "summary": ""},
        ]
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            result = self.agent.fetch_headlines("TSLA", "testkey")
        assert result == ["Brief headline"]

    def test_missing_headline_key_skipped(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"headline": "", "summary": "no headline here"},
            {"headline": "Good headline", "summary": ""},
        ]
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            result = self.agent.fetch_headlines("SPY", "testkey")
        assert len(result) == 1
        assert result[0] == "Good headline"

    def test_network_error_returns_empty(self):
        with patch("httpx.get", side_effect=Exception("timeout")):
            result = self.agent.fetch_headlines("MSFT", "testkey")
        assert result == []

    def test_caps_at_10_results(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"headline": f"Headline {i}", "summary": ""} for i in range(20)
        ]
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            result = self.agent.fetch_headlines("AMZN", "testkey")
        assert len(result) <= 10
