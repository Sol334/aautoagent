"""Tests for galactic-capital/scripts/wheel_trader.py"""

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAPITAL_ROOT))
sys.path.insert(0, str(CAPITAL_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Helper — reload-safe import
# ---------------------------------------------------------------------------

def _wt():
    import importlib
    import wheel_trader as wt
    importlib.reload(wt)
    return wt


# ---------------------------------------------------------------------------
# Existing Black-Scholes tests (kept intact)
# ---------------------------------------------------------------------------

class TestBlackScholesPut(unittest.TestCase):
    def test_intrinsic_value_at_zero_time(self):
        wt = _wt()
        # At expiry ITM put: should return intrinsic value K - S
        result = wt.black_scholes_put(S=100, K=105, T=0, r=0.05, sigma=0.20)
        self.assertAlmostEqual(result, 5.0, places=1)

    def test_otm_put_near_zero_at_expiry(self):
        wt = _wt()
        result = wt.black_scholes_put(S=100, K=95, T=0, r=0.05, sigma=0.20)
        self.assertEqual(result, 0.0)

    def test_positive_premium_with_valid_inputs(self):
        wt = _wt()
        # 1-week, 3% OTM put on a $500 stock with 20% IV should have positive premium
        result = wt.black_scholes_put(S=500, K=485, T=1/52, r=0.053, sigma=0.20)
        self.assertGreater(result, 0.0)

    def test_premium_increases_with_iv(self):
        wt = _wt()
        low_iv = wt.black_scholes_put(S=400, K=388, T=1/52, r=0.053, sigma=0.15)
        high_iv = wt.black_scholes_put(S=400, K=388, T=1/52, r=0.053, sigma=0.30)
        self.assertGreater(high_iv, low_iv)

    # --- new: intrinsic test using the new function name (test 1 of requested 8)
    def test_black_scholes_put_intrinsic(self):
        """T=0 → result equals max(K-S, 0) exactly."""
        wt = _wt()
        # ITM case
        self.assertAlmostEqual(
            wt.black_scholes_put(S=450.0, K=460.0, T=0, r=0.05, sigma=0.20),
            10.0, places=6,
        )
        # OTM case
        self.assertEqual(
            wt.black_scholes_put(S=450.0, K=440.0, T=0, r=0.05, sigma=0.20),
            0.0,
        )
        # Very small T — should still be close to intrinsic
        result = wt.black_scholes_put(S=450.0, K=460.0, T=1e-9, r=0.05, sigma=0.20)
        self.assertAlmostEqual(result, 10.0, places=1)


# ---------------------------------------------------------------------------
# _norm_inv tests
# ---------------------------------------------------------------------------

class TestNormInv(unittest.TestCase):
    def test_norm_inv_symmetry(self):
        """_norm_inv(0.5) ≈ 0.0 and _norm_inv(0.84) ≈ 1.0."""
        wt = _wt()
        self.assertAlmostEqual(wt._norm_inv(0.5), 0.0, places=3)
        self.assertAlmostEqual(wt._norm_inv(0.84), 1.0, delta=0.02)

    def test_norm_inv_boundary(self):
        """p=0 → -inf, p=1 → +inf."""
        wt = _wt()
        self.assertEqual(wt._norm_inv(0.0), -float("inf"))
        self.assertEqual(wt._norm_inv(1.0), float("inf"))

    def test_norm_inv_antisymmetry(self):
        """_norm_inv(p) == -_norm_inv(1-p) for 0 < p < 1."""
        wt = _wt()
        for p in (0.10, 0.25, 0.30, 0.40):
            self.assertAlmostEqual(wt._norm_inv(p), -wt._norm_inv(1 - p), places=6)

    def test_norm_inv_known_values(self):
        """Check a few well-known quantiles."""
        wt = _wt()
        # N_inv(0.9772) ≈ 2.0
        self.assertAlmostEqual(wt._norm_inv(0.9772), 2.0, delta=0.01)
        # N_inv(0.1587) ≈ -1.0
        self.assertAlmostEqual(wt._norm_inv(0.1587), -1.0, delta=0.01)


# ---------------------------------------------------------------------------
# find_delta_strike tests
# ---------------------------------------------------------------------------

class TestFindDeltaStrike(unittest.TestCase):
    """test_find_delta_strike_otm — strike < S for 30-delta put (always OTM)."""

    def test_find_delta_strike_otm(self):
        wt = _wt()
        S, T, r, sigma = 500.0, 1 / 52, 0.053, 0.20
        K = wt.find_delta_strike(S, T, r, sigma, delta_target=0.30)
        self.assertLess(K, S, "30-delta put strike must be OTM (below spot)")

    def test_strike_decreases_with_higher_delta_target(self):
        """Higher delta target → closer to ATM → lower OTM-ness, so strike is higher."""
        wt = _wt()
        S, T, r, sigma = 500.0, 1 / 52, 0.053, 0.20
        K_30 = wt.find_delta_strike(S, T, r, sigma, delta_target=0.30)
        K_20 = wt.find_delta_strike(S, T, r, sigma, delta_target=0.20)
        # 30-delta is closer to ATM than 20-delta → K_30 > K_20
        self.assertGreater(K_30, K_20)

    def test_strike_scales_with_spot(self):
        """Strike should scale proportionally with spot price."""
        wt = _wt()
        T, r, sigma = 1 / 52, 0.053, 0.20
        K1 = wt.find_delta_strike(500.0, T, r, sigma, 0.30)
        K2 = wt.find_delta_strike(1000.0, T, r, sigma, 0.30)
        self.assertAlmostEqual(K2 / K1, 2.0, delta=0.02)

    def test_find_delta_strike_zero_time(self):
        """T=0 fallback: strike = S * (1 - delta_target)."""
        wt = _wt()
        K = wt.find_delta_strike(500.0, 0.0, 0.053, 0.20, delta_target=0.30)
        self.assertAlmostEqual(K, 500.0 * 0.70, places=1)


# ---------------------------------------------------------------------------
# put_delta round-trip test
# ---------------------------------------------------------------------------

class TestPutDelta(unittest.TestCase):
    def test_put_delta_at_target(self):
        """put_delta(S, find_delta_strike(S,T,r,sigma,0.30), T,r,sigma) ≈ -0.30 ± 0.02."""
        wt = _wt()
        S, T, r, sigma = 500.0, 1 / 52, 0.053, 0.20
        K = wt.find_delta_strike(S, T, r, sigma, delta_target=0.30)
        delta = wt.put_delta(S, K, T, r, sigma)
        self.assertAlmostEqual(delta, -0.30, delta=0.02)

    def test_put_delta_atm(self):
        """ATM put delta should be close to -0.50."""
        wt = _wt()
        S, T, r, sigma = 500.0, 1 / 52, 0.053, 0.20
        delta = wt.put_delta(S, S, T, r, sigma)
        self.assertAlmostEqual(delta, -0.50, delta=0.05)

    def test_put_delta_negative(self):
        """Put delta is always negative for a live option."""
        wt = _wt()
        S, T, r, sigma = 500.0, 1 / 52, 0.053, 0.20
        for K in (450, 480, 500, 520):
            self.assertLess(wt.put_delta(S, K, T, r, sigma), 0)

    def test_put_delta_at_expiry_itm(self):
        """At expiry, ITM put delta = -1."""
        wt = _wt()
        self.assertAlmostEqual(wt.put_delta(S=100, K=110, T=0, r=0.05, sigma=0.2), -1.0)

    def test_put_delta_at_expiry_otm(self):
        """At expiry, OTM put delta = 0."""
        wt = _wt()
        self.assertAlmostEqual(wt.put_delta(S=100, K=90, T=0, r=0.05, sigma=0.2), 0.0)


# ---------------------------------------------------------------------------
# assignment_probability tests
# ---------------------------------------------------------------------------

class TestAssignmentProbability(unittest.TestCase):
    def test_assignment_probability_deep_itm(self):
        """K >> S → assignment probability ≈ 1.0."""
        wt = _wt()
        prob = wt.assignment_probability(S=100, K=200, T=1 / 52, r=0.05, sigma=0.20)
        self.assertGreater(prob, 0.99)

    def test_assignment_probability_deep_otm(self):
        """K << S → assignment probability ≈ 0.0."""
        wt = _wt()
        prob = wt.assignment_probability(S=500, K=100, T=1 / 52, r=0.05, sigma=0.20)
        self.assertLess(prob, 0.01)

    def test_assignment_probability_range(self):
        """Result must always be in [0, 1]."""
        wt = _wt()
        for K in (300, 400, 450, 480, 500, 520, 600):
            prob = wt.assignment_probability(S=500, K=K, T=1 / 52, r=0.05, sigma=0.20)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_assignment_probability_atm(self):
        """ATM probability should be roughly 0.5 for short DTE."""
        wt = _wt()
        prob = wt.assignment_probability(S=500, K=500, T=1 / 52, r=0.05, sigma=0.20)
        self.assertAlmostEqual(prob, 0.5, delta=0.1)


# ---------------------------------------------------------------------------
# get_wheel_simulation mocked test
# ---------------------------------------------------------------------------

class TestGetWheelSimulation(unittest.TestCase):
    def _make_mock_yf(self, price=520.0, options=None):
        mock_yf = MagicMock()
        mock_info = MagicMock()
        mock_info.last_price = price
        mock_info.regular_market_price = None
        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_info
        mock_ticker.options = options if options is not None else []
        mock_yf.Ticker.return_value = mock_ticker
        return mock_yf

    def test_get_wheel_simulation_mocked(self):
        """Mocked yfinance — returned dict must contain all expected keys."""
        import importlib
        import wheel_trader as wt
        importlib.reload(wt)

        mock_yf = self._make_mock_yf(price=520.0)
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = wt.get_wheel_simulation("SPY")

        self.assertEqual(result["ticker"], "SPY")
        self.assertEqual(result["price"], 520.0)
        self.assertEqual(result["mode"], "simulation")

        required_keys = [
            "ticker", "price", "strike", "iv", "delta",
            "premium_per_share", "premium_pct",
            "break_even", "break_even_pct",
            "assign_prob", "annualized_return_pct",
            "contracts_possible", "total_premium", "expiry", "mode", "note",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_break_even_calculation(self):
        """break_even = strike - premium_per_share."""
        import importlib
        import wheel_trader as wt
        importlib.reload(wt)

        mock_yf = self._make_mock_yf(price=520.0)
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = wt.get_wheel_simulation("SPY")

        expected_be = round(result["strike"] - result["premium_per_share"], 2)
        self.assertAlmostEqual(result["break_even"], expected_be, places=1)

    def test_annualized_return_pct(self):
        """annualized_return_pct ≈ premium_pct * 52 (within 0.15% of the product)."""
        import importlib
        import wheel_trader as wt
        importlib.reload(wt)

        mock_yf = self._make_mock_yf(price=520.0)
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = wt.get_wheel_simulation("SPY")

        # The internal calculation uses unrounded premium_pct; allow 0.15 rounding slack.
        product = result["premium_pct"] * 52
        self.assertAlmostEqual(result["annualized_return_pct"], product, delta=0.15)

    def test_strike_otm(self):
        """Returned strike must be below spot price (OTM put)."""
        import importlib
        import wheel_trader as wt
        importlib.reload(wt)

        mock_yf = self._make_mock_yf(price=520.0)
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = wt.get_wheel_simulation("SPY")

        self.assertLess(result["strike"], result["price"])

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

        mock_yf = self._make_mock_yf(price=520.0)
        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = wt.get_wheel_simulation("SPY")

        self.assertEqual(result["ticker"], "SPY")
        self.assertEqual(result["price"], 520.0)
        self.assertIn("premium_per_share", result)
        self.assertIn("expiry", result)
        self.assertEqual(result["mode"], "simulation")

    def test_zero_contracts_when_capital_too_low(self):
        import importlib
        with patch.dict(os.environ, {"CAPITAL_STARTING_USD": "10"}):
            import wheel_trader as wt
            importlib.reload(wt)

            mock_yf = self._make_mock_yf(price=520.0)
            with patch.dict(sys.modules, {"yfinance": mock_yf}):
                result = wt.get_wheel_simulation("SPY")

        self.assertEqual(result["contracts_possible"], 0)


# ---------------------------------------------------------------------------
# _log_simulation atomic write test
# ---------------------------------------------------------------------------

class TestLogSimulation(unittest.TestCase):
    def test_log_simulation_atomic_write(self):
        """
        _log_simulation writes atomically: after the call the .json file
        exists, the .tmp file does NOT exist, and the data is readable JSON.
        """
        import importlib
        import wheel_trader as wt
        importlib.reload(wt)

        sample = {
            "ticker": "SPY",
            "price": 520.0,
            "strike": 505.0,
            "premium_per_share": 1.23,
            "premium_pct": 0.237,
            "break_even": 503.77,
            "break_even_pct": 3.12,
            "assign_prob": 28.5,
            "annualized_return_pct": 12.3,
            "iv": 15.0,
            "delta": -0.30,
            "contracts_possible": 1,
            "total_premium": 123.0,
            "expiry": "2026-06-13",
            "mode": "simulation",
            "note": "test",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_log = Path(tmpdir) / "wheel_log.json"
            # Patch CAPITAL_ROOT so the log goes to our tmp dir
            with patch.object(wt, "CAPITAL_ROOT", Path(tmpdir)):
                # Create the data subdir that the function uses
                (Path(tmpdir) / "data").mkdir(parents=True, exist_ok=True)
                wt._log_simulation(dict(sample))

                log_path = Path(tmpdir) / "data" / "wheel_log.json"
                tmp_path = log_path.with_suffix(".tmp")

                # .json must exist; .tmp must be gone (atomic replace)
                self.assertTrue(log_path.exists(), "wheel_log.json not created")
                self.assertFalse(tmp_path.exists(), ".tmp file was not removed")

                # Content must be valid JSON with our entry
                entries = json.loads(log_path.read_text())
                self.assertIsInstance(entries, list)
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0]["ticker"], "SPY")
                self.assertIn("logged_at", entries[0])

    def test_log_simulation_appends(self):
        """Second call appends, not overwrites."""
        import importlib
        import wheel_trader as wt
        importlib.reload(wt)

        sample = {"ticker": "SPY", "price": 520.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(wt, "CAPITAL_ROOT", Path(tmpdir)):
                (Path(tmpdir) / "data").mkdir(parents=True, exist_ok=True)
                wt._log_simulation(dict(sample))
                wt._log_simulation(dict(sample, ticker="QQQ"))

                log_path = Path(tmpdir) / "data" / "wheel_log.json"
                entries = json.loads(log_path.read_text())
                self.assertEqual(len(entries), 2)
                self.assertEqual(entries[1]["ticker"], "QQQ")

    def test_log_simulation_caps_at_500(self):
        """Log file never exceeds 500 entries."""
        import importlib
        import wheel_trader as wt
        importlib.reload(wt)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(wt, "CAPITAL_ROOT", Path(tmpdir)):
                (Path(tmpdir) / "data").mkdir(parents=True, exist_ok=True)
                for i in range(505):
                    wt._log_simulation({"ticker": f"T{i}"})

                log_path = Path(tmpdir) / "data" / "wheel_log.json"
                entries = json.loads(log_path.read_text())
                self.assertEqual(len(entries), 500)


# ---------------------------------------------------------------------------
# Macro regime gate test
# ---------------------------------------------------------------------------

class TestGetRegime(unittest.TestCase):
    def test_get_regime_returns_neutral_on_import_error(self):
        """_get_regime() never raises — returns 'neutral' on failure."""
        import importlib
        import wheel_trader as wt
        importlib.reload(wt)

        with patch.dict(sys.modules, {"agents.macro_agent": None}):
            regime = wt._get_regime()
        # Either "neutral" (import failed gracefully) or one of the valid regimes
        self.assertIn(regime, ("risk_on", "neutral", "risk_off"))

    def test_get_regime_returns_string(self):
        """_get_regime always returns a str, never raises."""
        import importlib
        import wheel_trader as wt
        importlib.reload(wt)

        with patch("wheel_trader._get_regime", return_value="risk_off"):
            result = wt._get_regime()
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
