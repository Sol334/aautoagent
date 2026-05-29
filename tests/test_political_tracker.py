"""Tests for political_tracker.py — Ollama mocked throughout."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_ROOT = CAPITAL_ROOT  # repo root (was galactic-system parent before extraction)
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(CAPITAL_ROOT))


def _make_trade(ticker="AAPL", name="Nancy Pelosi", amount=750000,
                tx_type="Purchase", date="2026-01-15"):
    return {
        "symbol": ticker,
        "name": name,
        "transactionDate": date,
        "transactionType": tx_type,
        "amount": "$500,001 - $1,000,000",
        "_amount_mid": float(amount),
        "_id": f"{ticker}:{name}:{date}:{tx_type}",
        "owner": "Spouse",
    }


_VALID_LLM_RESPONSE = json.dumps([{
    "ticker": "AAPL",
    "signal": "BULLISH",
    "confidence": 0.82,
    "committee_edge": True,
    "reasoning": "Two buys by House Commerce Committee members within 14 days."
}])

_BEARISH_LLM_RESPONSE = json.dumps([{
    "ticker": "XOM",
    "signal": "BEARISH",
    "confidence": 0.71,
    "committee_edge": False,
    "reasoning": "Energy reps selling ahead of regulatory news.",
}])


class TestPoliticalTrackerAnalyze(unittest.TestCase):
    @patch("agents.base_agent.httpx.Client")
    def test_bullish_signal_returned(self, MockClient):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": _VALID_LLM_RESPONSE}
        MockClient.return_value.__enter__.return_value.post.return_value = mock_resp

        from agents.political_tracker import PoliticalTracker
        tracker = PoliticalTracker()
        signals = tracker.analyze([_make_trade("AAPL")])

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].ticker, "AAPL")
        self.assertEqual(signals[0].signal, "BULLISH")
        self.assertAlmostEqual(signals[0].confidence, 0.82)
        self.assertTrue(signals[0].committee_edge)

    @patch("agents.base_agent.httpx.Client")
    def test_bearish_signal_returned(self, MockClient):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": _BEARISH_LLM_RESPONSE}
        MockClient.return_value.__enter__.return_value.post.return_value = mock_resp

        from agents.political_tracker import PoliticalTracker
        tracker = PoliticalTracker()
        signals = tracker.analyze([_make_trade("XOM", tx_type="Sale")])

        self.assertEqual(signals[0].signal, "BEARISH")
        self.assertFalse(signals[0].committee_edge)

    @patch("agents.base_agent.httpx.Client")
    def test_ollama_timeout_returns_neutral(self, MockClient):
        import httpx
        MockClient.return_value.__enter__.return_value.post.side_effect = (
            httpx.TimeoutException("timeout")
        )
        from agents.political_tracker import PoliticalTracker
        tracker = PoliticalTracker()
        signals = tracker.analyze([_make_trade()])

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal, "NEUTRAL")
        self.assertEqual(signals[0].confidence, 0.0)

    @patch("agents.base_agent.httpx.Client")
    def test_malformed_json_returns_neutral(self, MockClient):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "This is not JSON at all."}
        MockClient.return_value.__enter__.return_value.post.return_value = mock_resp

        from agents.political_tracker import PoliticalTracker
        tracker = PoliticalTracker()
        signals = tracker.analyze([_make_trade()])
        self.assertEqual(signals[0].signal, "NEUTRAL")

    @patch("agents.base_agent.httpx.Client")
    def test_deepseek_think_tags_stripped(self, MockClient):
        response_with_think = "<think>Reasoning here...</think>" + _VALID_LLM_RESPONSE
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": response_with_think}
        MockClient.return_value.__enter__.return_value.post.return_value = mock_resp

        from agents.base_agent import BaseAgent
        agent = BaseAgent(model="deepseek-r1:7b", host="localhost:11434")
        result = agent.generate("test")
        self.assertNotIn("<think>", result)
        self.assertNotIn("Reasoning here", result)

    def test_empty_trades_returns_empty(self):
        from agents.political_tracker import PoliticalTracker
        tracker = PoliticalTracker()
        result = tracker.analyze([])
        self.assertEqual(result, [])

    @patch("agents.base_agent.httpx.Client")
    def test_multi_ticker_grouping(self, MockClient):
        multi_response = json.dumps([
            {"ticker": "AAPL", "signal": "BULLISH", "confidence": 0.75,
             "committee_edge": False, "reasoning": "Rep A buys."},
            {"ticker": "NVDA", "signal": "BEARISH", "confidence": 0.60,
             "committee_edge": True, "reasoning": "Rep B sells."},
        ])
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": multi_response}
        MockClient.return_value.__enter__.return_value.post.return_value = mock_resp

        from agents.political_tracker import PoliticalTracker
        tracker = PoliticalTracker()
        trades = [_make_trade("AAPL"), _make_trade("NVDA", tx_type="Sale")]
        signals = tracker.analyze(trades)
        tickers = {s.ticker for s in signals}
        self.assertIn("AAPL", tickers)
        self.assertIn("NVDA", tickers)


class TestFinnhubConnector(unittest.TestCase):
    def test_no_key_raises(self):
        from data_pipelines.finnhub_connector import FinnhubConnector
        with self.assertRaises(ValueError):
            FinnhubConnector("")

    def test_change_me_raises(self):
        from data_pipelines.finnhub_connector import FinnhubConnector
        with self.assertRaises(ValueError):
            FinnhubConnector("CHANGE_ME")

    @patch("data_pipelines.finnhub_connector.httpx.Client")
    def test_congressional_trades_normalized(self, MockClient):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{
            "name": "Nancy Pelosi",
            "transactionDate": "2026-01-15",
            "transactionType": "Purchase",
            "amount": "$500,001 - $1,000,000",
            "owner": "Spouse",
        }]}
        MockClient.return_value.__enter__.return_value.get.return_value = mock_resp

        from data_pipelines.finnhub_connector import FinnhubConnector
        conn = FinnhubConnector("real_key_here")
        trades = conn.congressional_trades("AAPL", "2026-01-01", "2026-01-31")

        self.assertEqual(len(trades), 1)
        self.assertIn("_amount_mid", trades[0])
        self.assertIn("_id", trades[0])
        self.assertAlmostEqual(trades[0]["_amount_mid"], 750000.5)

    @patch("data_pipelines.finnhub_connector.httpx.Client")
    def test_retry_on_429(self, MockClient):
        rate_resp = MagicMock()
        rate_resp.status_code = 429
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"data": []}

        MockClient.return_value.__enter__.return_value.get.side_effect = [
            rate_resp, ok_resp
        ]
        from data_pipelines.finnhub_connector import FinnhubConnector
        with patch("data_pipelines.finnhub_connector.time.sleep"):
            conn = FinnhubConnector("real_key")
            trades = conn.congressional_trades("AAPL", "2026-01-01", "2026-01-31")
        self.assertEqual(trades, [])


if __name__ == "__main__":
    unittest.main()
