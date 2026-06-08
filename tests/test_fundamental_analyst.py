"""Tests for galactic-capital/agents/fundamental_analyst.py"""

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAPITAL_ROOT))
sys.path.insert(0, str(CAPITAL_ROOT / "agents"))


def _make_mock_yf(info: dict):
    mock_yf = MagicMock()
    mock_yf.Ticker.return_value.info = info
    return mock_yf


def _reload_agent():
    import fundamental_analyst as fa_mod
    importlib.reload(fa_mod)
    from fundamental_analyst import FundamentalAnalyst
    return FundamentalAnalyst()


class TestFundamentalAnalyst(unittest.TestCase):
    def test_bullish_signal(self):
        """PE=15, rev_growth=0.20, debtToEquity=50 (÷100 = 0.5) → BULLISH."""
        mock_yf = _make_mock_yf({
            "trailingPE": 15.0,
            "priceToBook": 2.0,
            "trailingEps": 3.5,
            "revenueGrowth": 0.20,
            "debtToEquity": 50.0,
        })
        agent = _reload_agent()
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            sig = agent.analyze("AAPL")

        self.assertEqual(sig.signal, "BULLISH")
        self.assertGreaterEqual(sig.confidence, 0.6)
        self.assertLessEqual(sig.confidence, 0.9)
        self.assertAlmostEqual(sig.pe_ratio, 15.0)
        self.assertAlmostEqual(sig.revenue_growth_yoy, 0.20)
        self.assertAlmostEqual(sig.debt_to_equity, 0.5)

    def test_bearish_signal(self):
        """PE=80, rev_growth=-0.10, debtToEquity=400 (÷100 = 4.0) → BEARISH."""
        mock_yf = _make_mock_yf({
            "trailingPE": 80.0,
            "priceToBook": 8.0,
            "trailingEps": -1.2,
            "revenueGrowth": -0.10,
            "debtToEquity": 400.0,
        })
        agent = _reload_agent()
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            sig = agent.analyze("XYZ")

        self.assertEqual(sig.signal, "BEARISH")
        self.assertGreaterEqual(sig.confidence, 0.5)
        self.assertLessEqual(sig.confidence, 0.8)
        self.assertAlmostEqual(sig.debt_to_equity, 4.0)

    def test_neutral_signal(self):
        """PE=30, rev_growth=0.05, debtToEquity=100 (÷100 = 1.0) → NEUTRAL."""
        mock_yf = _make_mock_yf({
            "trailingPE": 30.0,
            "priceToBook": 3.0,
            "trailingEps": 2.0,
            "revenueGrowth": 0.05,
            "debtToEquity": 100.0,
        })
        agent = _reload_agent()
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            sig = agent.analyze("MSFT")

        self.assertEqual(sig.signal, "NEUTRAL")
        self.assertAlmostEqual(sig.confidence, 0.4)

    def test_missing_data_neutral(self):
        """Empty info dict → NEUTRAL, confidence 0.0."""
        mock_yf = _make_mock_yf({})
        agent = _reload_agent()
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            sig = agent.analyze("EMPTY")

        self.assertEqual(sig.signal, "NEUTRAL")
        self.assertAlmostEqual(sig.confidence, 0.0)
        self.assertIsNone(sig.pe_ratio)
        self.assertIsNone(sig.pb_ratio)
        self.assertIsNone(sig.eps_ttm)
        self.assertIsNone(sig.revenue_growth_yoy)
        self.assertIsNone(sig.debt_to_equity)

    def test_yfinance_unavailable(self):
        """sys.modules['yfinance'] = None → NEUTRAL, confidence 0.0."""
        with patch.dict(sys.modules, {"yfinance": None}):
            import fundamental_analyst as fa_mod
            importlib.reload(fa_mod)
            from fundamental_analyst import FundamentalAnalyst
            agent = FundamentalAnalyst()
            sig = agent.analyze("NVDA")

        self.assertEqual(sig.signal, "NEUTRAL")
        self.assertAlmostEqual(sig.confidence, 0.0)
        self.assertIsNone(sig.pe_ratio)


if __name__ == "__main__":
    unittest.main()
