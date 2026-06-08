"""Tests for political_watchdog.py — all external calls mocked."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAPITAL_ROOT))


class TestAmountParser(unittest.TestCase):
    def test_range_string_midpoint(self):
        from data_pipelines.finnhub_connector import _parse_amount
        self.assertEqual(_parse_amount("$50,001 - $100,000"), 75000.5)

    def test_single_value(self):
        from data_pipelines.finnhub_connector import _parse_amount
        self.assertEqual(_parse_amount("$250,000"), 250000.0)

    def test_empty(self):
        from data_pipelines.finnhub_connector import _parse_amount
        self.assertEqual(_parse_amount(""), 0.0)

    def test_none_like(self):
        from data_pipelines.finnhub_connector import _parse_amount
        self.assertEqual(_parse_amount(None), 0.0)


class TestDetectionLogic(unittest.TestCase):
    def _make_trade(self, ticker, name, amount_mid, date=None):
        from datetime import datetime
        return {
            "symbol": ticker,
            "name": name,
            "transactionDate": date or datetime.now().strftime("%Y-%m-%d"),
            "transactionType": "Purchase",
            "amount": "$250,001 - $500,000",
            "_amount_mid": amount_mid,
            "_id": f"{ticker}:{name}:test:Purchase",
        }

    def test_high_value_detection(self):
        from scripts.political_watchdog import _is_high_value
        trade = self._make_trade("AAPL", "Pelosi", 750000.0)
        self.assertTrue(_is_high_value(trade))

    def test_below_high_value_threshold(self):
        from scripts.political_watchdog import _is_high_value
        trade = self._make_trade("AAPL", "Someone", 10000.0)
        self.assertFalse(_is_high_value(trade))

    def test_cluster_detection(self):
        from scripts.political_watchdog import _detect_clusters
        trades = [
            self._make_trade("NVDA", "Rep A", 50000),
            self._make_trade("NVDA", "Rep B", 75000),
        ]
        self.assertTrue(_detect_clusters(trades, "NVDA"))

    def test_single_rep_no_cluster(self):
        from scripts.political_watchdog import _detect_clusters
        trades = [self._make_trade("NVDA", "Rep A", 50000)]
        self.assertFalse(_detect_clusters(trades, "NVDA"))

    def test_new_trades_filter(self):
        from scripts.political_watchdog import _detect_new_trades
        all_trades = [
            self._make_trade("AAPL", "Rep A", 50000),
            self._make_trade("AAPL", "Rep B", 75000),
        ]
        seen = {"AAPL:Rep A:test:Purchase"}
        new = _detect_new_trades(all_trades, seen)
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0]["name"], "Rep B")


class TestStateManagement(unittest.TestCase):
    def test_save_and_load_state(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp = Path(f.name)

        try:
            with patch("scripts.political_watchdog.STATE_PATH", tmp):
                from scripts.political_watchdog import _save_state, _load_state
                _save_state({"id1", "id2"})
                loaded = _load_state()
                self.assertEqual(loaded, {"id1", "id2"})
        finally:
            tmp.unlink(missing_ok=True)


class TestTelegramFormat(unittest.TestCase):
    def test_format_alert_bullish(self):
        from scripts.political_watchdog import _format_alert
        from agents.political_tracker import TradeSignal
        signal = TradeSignal(
            ticker="AAPL",
            signal="BULLISH",
            confidence=0.82,
            reasoning="Two buys by committee members.",
            committee_edge=True,
            raw_trades=[{
                "representative": "Nancy Pelosi",
                "type": "Purchase",
                "amount_range": "$500,001 - $1,000,000",
            }],
        )
        msg = _format_alert(signal)
        self.assertIn("AAPL", msg)
        self.assertIn("BULLISH", msg)
        self.assertIn("82%", msg)
        self.assertIn("Pelosi", msg)
        self.assertIn("Committee edge", msg)

    def test_format_alert_bearish(self):
        from scripts.political_watchdog import _format_alert
        from agents.political_tracker import TradeSignal
        signal = TradeSignal(
            ticker="XOM",
            signal="BEARISH",
            confidence=0.65,
            reasoning="Energy committee chair selling.",
            raw_trades=[{"representative": "Rep X", "type": "Sale", "amount_range": "$100k+"}],
        )
        msg = _format_alert(signal)
        self.assertIn("📉", msg)
        self.assertIn("BEARISH", msg)


class TestWatchdogRunTestMode(unittest.TestCase):
    @patch("scripts.political_watchdog._send_telegram")
    @patch("scripts.political_watchdog.PoliticalTracker")
    @patch("scripts.political_watchdog._save_state")
    @patch("scripts.political_watchdog._append_log")
    @patch("scripts.political_watchdog._load_state", return_value=set())
    def test_test_mode_runs_without_finnhub(self, mock_load, mock_log, mock_save,
                                             MockTracker, mock_telegram):
        from agents.political_tracker import TradeSignal
        mock_instance = MockTracker.return_value
        mock_instance.analyze.return_value = [
            TradeSignal(
                ticker="AAPL",
                signal="BULLISH",
                confidence=0.82,
                reasoning="Pelosi buy is historically bullish.",
                raw_trades=[],
            )
        ]
        from scripts.political_watchdog import run
        result = run(dry_run=True, test_mode=True)
        self.assertEqual(result["new_trades"], 2)
        self.assertEqual(result["alerts_sent"], 1)
        mock_telegram.assert_called_once()
        mock_save.assert_called_once()

    @patch("scripts.political_watchdog._send_telegram")
    @patch("scripts.political_watchdog.PoliticalTracker")
    @patch("scripts.political_watchdog._save_state")
    @patch("scripts.political_watchdog._append_log")
    @patch("scripts.political_watchdog._load_state", return_value=set())
    def test_low_confidence_no_alert(self, mock_load, mock_log, mock_save,
                                      MockTracker, mock_telegram):
        from agents.political_tracker import TradeSignal
        mock_instance = MockTracker.return_value
        mock_instance.analyze.return_value = [
            TradeSignal(
                ticker="AAPL",
                signal="BULLISH",
                confidence=0.3,  # below threshold
                reasoning="Weak signal.",
                raw_trades=[],
            )
        ]
        from scripts.political_watchdog import run
        result = run(dry_run=True, test_mode=True)
        self.assertEqual(result["alerts_sent"], 0)
        mock_telegram.assert_not_called()

    def test_missing_finnhub_key_returns_error(self):
        with patch.dict("os.environ", {"FINNHUB_API_KEY": ""}, clear=False):
            # Re-import to pick up patched env
            import importlib
            import scripts.political_watchdog as wd
            importlib.reload(wd)
            result = wd.run(dry_run=False, test_mode=False)
            self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
