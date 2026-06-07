"""Tests for scripts/portfolio_dashboard.py"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAPITAL_ROOT))


def _import_dashboard():
    import importlib
    import scripts.portfolio_dashboard as pd
    importlib.reload(pd)
    return pd


class TestEquityMetricsEmpty(unittest.TestCase):
    """Missing log file → section shows gracefully (no crash)."""

    def test_none_input_returns_unavailable(self):
        pd = _import_dashboard()
        result = pd.compute_equity_metrics(None)
        self.assertFalse(result["available"])

    def test_empty_list_returns_unavailable(self):
        pd = _import_dashboard()
        result = pd.compute_equity_metrics([])
        self.assertFalse(result["available"])

    def test_missing_file_does_not_crash(self):
        """Full build_dashboard() with no files present must not raise."""
        pd = _import_dashboard()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            with patch.object(pd, "PAPER_TRADES_PATH",  data_dir / "paper_trades_dryrun.json"), \
                 patch.object(pd, "WHEEL_LOG_PATH",      data_dir / "wheel_log.json"), \
                 patch.object(pd, "CRYPTO_TRADES_PATH",  data_dir / "crypto_paper_trades.json"), \
                 patch.object(pd, "PORTFOLIO_STATE_PATH", data_dir / "portfolio_state.json"):
                dashboard = pd.build_dashboard()
        self.assertIn("equity", dashboard)
        self.assertFalse(dashboard["equity"]["available"])
        self.assertFalse(dashboard["wheel"]["available"])


class TestEquityMetricsCounts(unittest.TestCase):
    """Known BUY/SELL/HOLD entries → assert counts correct."""

    def _trades(self):
        return [
            {"ts": "2026-01-01", "ticker": "NVDA", "price": 800, "action": "BUY",  "shares": 1, "dry_run": True},
            {"ts": "2026-01-02", "ticker": "NVDA", "price": 810, "action": "BUY",  "shares": 1, "dry_run": True},
            {"ts": "2026-01-03", "ticker": "AAPL", "price": 180, "action": "SELL", "shares": 1, "dry_run": True},
            {"ts": "2026-01-04", "ticker": "NVDA", "price": 820, "action": "HOLD", "shares": 0, "dry_run": True},
            {"ts": "2026-01-05", "ticker": "MSFT", "price": 400, "action": "HOLD", "shares": 0, "dry_run": True},
            {"ts": "2026-01-06", "ticker": "NVDA", "price": 830, "action": "BUY",  "shares": 1, "dry_run": True},
        ]

    def test_buy_sell_hold_counts(self):
        pd = _import_dashboard()
        metrics = pd.compute_equity_metrics(self._trades())
        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["buy"],  3)
        self.assertEqual(metrics["sell"], 1)
        self.assertEqual(metrics["hold"], 2)
        self.assertEqual(metrics["total_signals"], 6)

    def test_unique_tickers(self):
        pd = _import_dashboard()
        metrics = pd.compute_equity_metrics(self._trades())
        self.assertEqual(metrics["unique_tickers"], 3)  # NVDA, AAPL, MSFT

    def test_most_active_ticker(self):
        pd = _import_dashboard()
        metrics = pd.compute_equity_metrics(self._trades())
        self.assertEqual(metrics["most_active_ticker"], "NVDA")
        self.assertEqual(metrics["most_active_count"], 4)


class TestWheelMetricsPremiumSum(unittest.TestCase):
    """Three known wheel_log entries → assert total_premium correct."""

    def _log(self):
        return [
            {"ticker": "SPY", "price": 500, "strike": 485, "premium_per_share": 1.20,
             "total_premium": 12.00, "expiry": "2026-01-10", "logged_at": "2026-01-03T10:00:00"},
            {"ticker": "QQQ", "price": 420, "strike": 405, "premium_per_share": 1.50,
             "total_premium": 15.00, "expiry": "2026-01-17", "logged_at": "2026-01-10T10:00:00"},
            {"ticker": "SPY", "price": 502, "strike": 488, "premium_per_share": 0.95,
             "total_premium": 9.50,  "expiry": "2026-01-24", "logged_at": "2026-01-17T10:00:00"},
        ]

    def test_total_premium(self):
        pd = _import_dashboard()
        metrics = pd.compute_wheel_metrics(self._log())
        self.assertTrue(metrics["available"])
        self.assertAlmostEqual(metrics["total_premium"], 36.50, places=2)

    def test_avg_weekly_premium(self):
        pd = _import_dashboard()
        metrics = pd.compute_wheel_metrics(self._log())
        self.assertAlmostEqual(metrics["avg_weekly_premium"], 12.17, places=1)

    def test_simulations_count(self):
        pd = _import_dashboard()
        metrics = pd.compute_wheel_metrics(self._log())
        self.assertEqual(metrics["simulations"], 3)

    def test_best_week(self):
        pd = _import_dashboard()
        metrics = pd.compute_wheel_metrics(self._log())
        self.assertAlmostEqual(metrics["best_week_premium"], 15.00, places=2)
        self.assertEqual(metrics["best_week_ticker"], "QQQ")

    def test_none_returns_unavailable(self):
        pd = _import_dashboard()
        metrics = pd.compute_wheel_metrics(None)
        self.assertFalse(metrics["available"])


class TestPortfolioStateDrawdown(unittest.TestCase):
    """peak=200, current=160 → drawdown=20%."""

    def test_drawdown_20_pct(self):
        pd = _import_dashboard()
        state = pd.compute_portfolio_state({"peak": 200.0, "current": 160.0})
        self.assertAlmostEqual(state["drawdown_pct"], 20.0, places=2)

    def test_no_drawdown(self):
        pd = _import_dashboard()
        state = pd.compute_portfolio_state({"peak": 200.0, "current": 200.0})
        self.assertAlmostEqual(state["drawdown_pct"], 0.0, places=2)

    def test_partial_drawdown(self):
        pd = _import_dashboard()
        state = pd.compute_portfolio_state({"peak": 1000.0, "current": 800.0})
        self.assertAlmostEqual(state["drawdown_pct"], 20.0, places=2)

    def test_none_state_uses_defaults(self):
        pd = _import_dashboard()
        state = pd.compute_portfolio_state(None)
        self.assertAlmostEqual(state["drawdown_pct"], 0.0, places=2)
        self.assertFalse(state["kill_switch_active"])


class TestKillSwitchActive(unittest.TestCase):
    """drawdown > 20% → kill_switch_active=True."""

    def test_kill_switch_triggered_above_threshold(self):
        pd = _import_dashboard()
        # 21% drawdown: peak=200, current=158
        state = pd.compute_portfolio_state({"peak": 200.0, "current": 158.0})
        self.assertGreater(state["drawdown_pct"], 20.0)
        self.assertTrue(state["kill_switch_active"])

    def test_kill_switch_safe_at_threshold(self):
        pd = _import_dashboard()
        # Exactly 20% → not active (threshold is strictly greater-than)
        state = pd.compute_portfolio_state({"peak": 200.0, "current": 160.0})
        self.assertFalse(state["kill_switch_active"])

    def test_kill_switch_safe_below_threshold(self):
        pd = _import_dashboard()
        state = pd.compute_portfolio_state({"peak": 200.0, "current": 190.0})
        self.assertFalse(state["kill_switch_active"])

    def test_render_shows_active_label(self):
        pd = _import_dashboard()
        dashboard = {
            "as_of": "2026-06-06 21:00 CT",
            "equity": {"available": False},
            "wheel": {"available": False},
            "portfolio": pd.compute_portfolio_state({"peak": 200.0, "current": 150.0}),
            "crypto_signals_logged": 0,
        }
        rendered = pd.render_dashboard(dashboard)
        self.assertIn("ACTIVE", rendered)


class TestJsonFlag(unittest.TestCase):
    """--json flag → output is valid JSON."""

    def test_json_output_via_subprocess(self):
        script = CAPITAL_ROOT / "scripts" / "portfolio_dashboard.py"
        result = subprocess.run(
            [sys.executable, str(script), "--json"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=f"stderr: {result.stderr}")
        parsed = json.loads(result.stdout)
        self.assertIn("equity", parsed)
        self.assertIn("wheel", parsed)
        self.assertIn("portfolio", parsed)

    def test_json_output_via_function(self):
        pd = _import_dashboard()
        data = pd.build_dashboard()
        # Verify it round-trips through JSON without error
        raw = json.dumps(data)
        parsed = json.loads(raw)
        self.assertIn("equity", parsed)
        self.assertIn("portfolio", parsed)


if __name__ == "__main__":
    unittest.main()
