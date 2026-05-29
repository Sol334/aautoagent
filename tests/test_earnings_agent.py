"""Tests for galactic-capital/agents/earnings_agent.py"""

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_ROOT = CAPITAL_ROOT  # repo root (was galactic-system parent before extraction)
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(CAPITAL_ROOT))
sys.path.insert(0, str(CAPITAL_ROOT / "agents"))


def _make_mock_yf(calendar=None, raises=False):
    mock_yf = MagicMock()
    if raises:
        mock_yf.Ticker.side_effect = Exception("network")
    else:
        mock_yf.Ticker.return_value.calendar = calendar
    return mock_yf


class TestEarningsAgentEmpty(unittest.TestCase):
    def setUp(self):
        import importlib
        import earnings_agent as ea_mod
        importlib.reload(ea_mod)
        from earnings_agent import EarningsAgent
        self.agent = EarningsAgent()

    def test_returns_empty_on_none_calendar(self):
        mock_yf = _make_mock_yf(calendar=None)
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = self.agent.get_earnings_context("AAPL")
        self.assertIsNone(result["days_to_earnings"])
        self.assertFalse(result["is_within_window"])

    def test_returns_empty_when_yfinance_missing(self):
        # Remove yfinance from modules to simulate ImportError path
        saved = sys.modules.pop("yfinance", None)
        try:
            import importlib
            import earnings_agent as ea_mod
            importlib.reload(ea_mod)
            from earnings_agent import EarningsAgent
            agent = EarningsAgent()
            result = agent.get_earnings_context("AAPL")
            self.assertIsNone(result["days_to_earnings"])
        finally:
            if saved is not None:
                sys.modules["yfinance"] = saved

    def test_returns_empty_on_yfinance_exception(self):
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = Exception("network error")
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = self.agent.get_earnings_context("AAPL")
        self.assertIsNone(result["days_to_earnings"])


class TestEarningsAgentWithData(unittest.TestCase):
    def setUp(self):
        import importlib
        import earnings_agent as ea_mod
        importlib.reload(ea_mod)
        from earnings_agent import EarningsAgent
        self.agent = EarningsAgent()

    def _make_calendar_dict(self, days_out: int, eps: float = 1.5, rev: float = 5e9):
        earnings_dt = date.today() + timedelta(days=days_out)
        return {
            "Earnings Date": [earnings_dt],
            "EPS Estimate": [eps],
            "Revenue Estimate": [rev],
        }

    def test_within_window_when_earnings_in_5_days(self):
        cal = self._make_calendar_dict(5)
        mock_yf = _make_mock_yf(calendar=cal)
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = self.agent.get_earnings_context("NVDA")
        self.assertTrue(result["is_within_window"])
        self.assertEqual(result["days_to_earnings"], 5)

    def test_not_within_window_when_earnings_in_30_days(self):
        cal = self._make_calendar_dict(30)
        mock_yf = _make_mock_yf(calendar=cal)
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = self.agent.get_earnings_context("NVDA")
        self.assertFalse(result["is_within_window"])

    def test_not_within_window_when_earnings_in_1_day(self):
        cal = self._make_calendar_dict(1)
        mock_yf = _make_mock_yf(calendar=cal)
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = self.agent.get_earnings_context("NVDA")
        self.assertFalse(result["is_within_window"])

    def test_eps_and_revenue_extracted(self):
        cal = self._make_calendar_dict(7, eps=2.35, rev=12.5e9)
        mock_yf = _make_mock_yf(calendar=cal)
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = self.agent.get_earnings_context("MSFT")
        self.assertAlmostEqual(result["eps_estimate"], 2.35)
        self.assertAlmostEqual(result["revenue_estimate"], 12.5e9)

    def test_summary_string_contains_ticker_and_days(self):
        cal = self._make_calendar_dict(5)
        mock_yf = _make_mock_yf(calendar=cal)
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = self.agent.get_earnings_context("AAPL")
        self.assertIn("AAPL", result["summary"])
        self.assertIn("5", result["summary"])


if __name__ == "__main__":
    unittest.main()
