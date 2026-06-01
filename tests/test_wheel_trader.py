"""Tests for galactic-capital/scripts/wheel_trader.py"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAPITAL_ROOT))
sys.path.insert(0, str(CAPITAL_ROOT / "scripts"))


class TestBlackScholesPut(unittest.TestCase):
    def test_intrinsic_value_at_zero_time(self):
        import importlib
        import wheel_trader as wt
        importlib.reload(wt)
        # At expiry ITM put: should return intrinsic value K - S
        result = wt.black_scholes_put(S=100, K=105, T=0, r=0.05, sigma=0.20)
        self.assertAlmostEqual(result, 5.0, places=1)

    def test_otm_put_near_zero_at_expiry(self):
        import importlib
        import wheel_trader as wt
        importlib.reload(wt)
        result = wt.black_scholes_put(S=100, K=95, T=0, r=0.05, sigma=0.20)
        self.assertEqual(result, 0.0)

    def test_positive_premium_with_valid_inputs(self):
        import importlib
        import wheel_trader as wt
        importlib.reload(wt)
        # 1-week, 3% OTM put on a $500 stock with 20% IV should have positive premium
        result = wt.black_scholes_put(S=500, K=485, T=1/52, r=0.053, sigma=0.20)
        self.assertGreater(result, 0.0)

    def test_premium_increases_with_iv(self):
        import importlib
        import wheel_trader as wt
        importlib.reload(wt)
        low_iv = wt.black_scholes_put(S=400, K=388, T=1/52, r=0.053, sigma=0.15)
        high_iv = wt.black_scholes_put(S=400, K=388, T=1/52, r=0.053, sigma=0.30)
        self.assertGreater(high_iv, low_iv)


class TestWheelSimulation(unittest.TestCase):
    def test_returns_empty_when_yfinance_missing(self):
        import importlib, builtins
        import wheel_trader as wt
        importlib.reload(wt)
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yfinance":
                raise ImportError
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = wt.get_wheel_simulation("SPY")
        self.assertEqual(result, {})

    def test_returns_simulation_result_with_mocked_price(self):
        import importlib
        import wheel_trader as wt
        importlib.reload(wt)

        mock_yf = MagicMock()
        mock_info = MagicMock()
        mock_info.last_price = 520.0
        mock_info.regular_market_price = None
        mock_yf.Ticker.return_value.fast_info = mock_info
        mock_yf.Ticker.return_value.options = []

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = wt.get_wheel_simulation("SPY")

        self.assertEqual(result["ticker"], "SPY")
        self.assertEqual(result["price"], 520.0)
        self.assertIn("premium_per_share", result)
        self.assertIn("expiry", result)
        self.assertEqual(result["mode"], "simulation")

    def test_zero_contracts_when_capital_too_low(self):
        import importlib, os
        with patch.dict(os.environ, {"CAPITAL_STARTING_USD": "10"}):
            import wheel_trader as wt
            importlib.reload(wt)

            mock_yf = MagicMock()
            mock_info = MagicMock()
            mock_info.last_price = 520.0
            mock_info.regular_market_price = None
            mock_yf.Ticker.return_value.fast_info = mock_info
            mock_yf.Ticker.return_value.options = []

            with patch.dict(sys.modules, {"yfinance": mock_yf}):
                result = wt.get_wheel_simulation("SPY")

        self.assertEqual(result["contracts_possible"], 0)


if __name__ == "__main__":
    unittest.main()
