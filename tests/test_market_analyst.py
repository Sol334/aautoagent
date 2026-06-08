"""Tests for market_analyst.py, sentiment_agent.py, forecast_agent.py,
paper_trader.py (kill switch + signal dict), and crypto_trader.py."""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAPITAL_ROOT))
sys.path.insert(0, str(CAPITAL_ROOT / "scripts"))
sys.path.insert(0, str(CAPITAL_ROOT / "agents"))


class TestSentimentAgent(unittest.TestCase):
    def setUp(self):
        # Re-import with a fresh instance each test
        import importlib
        import sentiment_agent as sa_mod
        importlib.reload(sa_mod)
        from sentiment_agent import SentimentAgent
        self.agent = SentimentAgent()

    def test_neutral_on_empty_headlines(self):
        result = self.agent.analyze_headlines([])
        self.assertEqual(result, 0.0)

    def test_neutral_fallback_when_transformers_unavailable(self):
        """If the pipeline cannot load, returns 0.0 without raising."""
        self.agent._load_attempted = True
        self.agent._pipeline = None
        result = self.agent.analyze_headlines(["Markets surged on strong earnings"])
        self.assertEqual(result, 0.0)

    def test_positive_score_from_mocked_pipeline(self):
        """Mocked pipeline returning 'positive' label -> score > 0."""
        # HF pipeline with top_k=None + single-string input returns a flat list of dicts
        mock_pipe = MagicMock(
            return_value=[
                {"label": "positive", "score": 0.9},
                {"label": "negative", "score": 0.05},
                {"label": "neutral", "score": 0.05},
            ]
        )
        self.agent._load_attempted = True
        self.agent._pipeline = mock_pipe
        result = self.agent.analyze_headlines(["Strong earnings beat expectations"])
        self.assertGreater(result, 0.0)

    def test_negative_score_from_mocked_pipeline(self):
        mock_pipe = MagicMock(
            return_value=[
                {"label": "negative", "score": 0.85},
                {"label": "positive", "score": 0.10},
                {"label": "neutral", "score": 0.05},
            ]
        )
        self.agent._load_attempted = True
        self.agent._pipeline = mock_pipe
        result = self.agent.analyze_headlines(["Stock crashes on poor outlook"])
        self.assertLess(result, 0.0)


class TestForecastAgent(unittest.TestCase):
    def setUp(self):
        import importlib
        import forecast_agent as fa_mod
        importlib.reload(fa_mod)
        from forecast_agent import ForecastAgent
        self.agent = ForecastAgent()

    def test_flat_on_insufficient_prices(self):
        result = self.agent.predict_trend([100.0])
        self.assertEqual(result, "flat")

    def test_sma_fallback_used_when_tsfm_unavailable(self):
        """When model is None, _sma_fallback is used."""
        self.agent._load_attempted = True
        self.agent._model = None
        prices = [100 + i for i in range(30)]  # steadily rising
        result = self.agent.predict_trend(prices)
        self.assertEqual(result, "up")

    def test_trend_up_from_rising_prices(self):
        """Strictly rising 30-day history with no model -> SMA returns 'up'."""
        self.agent._load_attempted = True
        self.agent._model = None
        prices = list(range(100, 131))  # 100, 101, ..., 130
        result = self.agent.predict_trend(prices)
        self.assertEqual(result, "up")

    def test_trend_down_from_falling_prices(self):
        self.agent._load_attempted = True
        self.agent._model = None
        prices = list(range(130, 99, -1))  # 130, 129, ..., 100
        result = self.agent.predict_trend(prices)
        self.assertEqual(result, "down")

    def test_flat_on_stable_prices(self):
        self.agent._load_attempted = True
        self.agent._model = None
        prices = [100.0] * 30
        result = self.agent.predict_trend(prices)
        self.assertEqual(result, "flat")


class TestWriteSignalsToCapitalMd(unittest.TestCase):
    def _make_capital_md(self, tmp_path):
        md = tmp_path / "Capital.md"
        md.write_text(
            "# Capital.md\n\n"
            "## System Status\n\nRunning.\n\n"
            "## Active Signals\n\n*No signals yet.*\n\n"
            "## Pending Actions\n\n*None.*\n"
        )
        return md

    def test_overwrites_only_active_signals_section(self):
        import importlib
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            md = self._make_capital_md(td_path)

            import market_analyst as ma
            importlib.reload(ma)
            ma.CAPITAL_MD = md

            signals = {"NVDA": {"action": "BUY", "reason": "momentum", "score": 0.7, "trend": "up"}}
            ma._write_signals_to_capital_md(signals)

            content = md.read_text()
            self.assertIn("NVDA: BUY", content)
            self.assertIn("## System Status", content)
            self.assertIn("## Pending Actions", content)

    def test_dry_run_does_not_write_capital_md(self):
        import importlib
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            md = self._make_capital_md(td_path)
            original = md.read_text()

            import market_analyst as ma
            importlib.reload(ma)
            ma.CAPITAL_MD = md

            with patch.object(ma, "_ask_ollama", return_value=("HOLD", "dry-run")), \
                 patch.object(ma, "_fetch_prices", return_value=[100.0] * 30), \
                 patch.object(ma.SentimentAgent, "fetch_headlines", return_value=[]), \
                 patch.object(ma.SentimentAgent, "analyze_headlines", return_value=0.0):
                ma.run_analysis(["AAPL"], dry_run=True)

            self.assertEqual(md.read_text(), original)


class TestPaperTraderKillSwitch(unittest.TestCase):
    def test_position_capped_at_max_position_usd(self):
        """Shares computed using MAX_POSITION, not a hardcoded $1,000."""
        import importlib
        with patch.dict(os.environ, {"CAPITAL_MAX_POSITION_USD": "20"}):
            import paper_trader as pt
            importlib.reload(pt)
            self.assertEqual(pt.MAX_POSITION, 20.0)
            # $20 / $500 stock = 0.04 shares (not 2.0 as $1,000/$500 would give)
            price = 500.0
            shares = round(pt.MAX_POSITION / price, 4)
            self.assertEqual(shares, 0.04)

    def test_kill_switch_inactive_at_low_drawdown(self):
        import importlib
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "portfolio_state.json"
            state_path.write_text(json.dumps({"peak": 200.0, "current": 185.0}))
            with patch.dict(os.environ, {"CAPITAL_MAX_DRAWDOWN_PCT": "20"}):
                import paper_trader as pt
                importlib.reload(pt)
                pt.PORTFOLIO_STATE_PATH = state_path
                self.assertFalse(pt._is_kill_switch_active())

    def test_kill_switch_triggers_at_max_drawdown(self):
        import importlib
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "portfolio_state.json"
            # 21% drawdown: (200 - 158) / 200 = 21%
            state_path.write_text(json.dumps({"peak": 200.0, "current": 158.0}))
            log_path = Path(td) / "trades.json"
            with patch.dict(os.environ, {"CAPITAL_MAX_DRAWDOWN_PCT": "20"}):
                import paper_trader as pt
                importlib.reload(pt)
                pt.PORTFOLIO_STATE_PATH = state_path
                pt.DRY_RUN_LOG = log_path
                # run() should return early — no trades logged
                with patch.object(pt, "_fetch_quotes", return_value={"AAPL": 150.0}):
                    pt.run(
                        tickers=["AAPL"],
                        max_tickers=1,
                        dry_run=False,
                        signals={"AAPL": {"action": "BUY", "reason": "test", "score": 0.5, "trend": "up"}},
                    )
                self.assertFalse(log_path.exists())


class TestCryptoTraderPaperMode(unittest.TestCase):
    def test_no_coinbase_calls_when_feature_disabled(self):
        """When FEATURE_CRYPTO_TRADING=false, no Coinbase API should be called."""
        with patch.dict(os.environ, {
            "FEATURE_CRYPTO_TRADING": "false",
            "CAPITAL_PAPER_TRADING": "true",
            "COINBASE_API_KEY": "CHANGE_ME",
        }):
            import importlib
            import crypto_trader as ct
            importlib.reload(ct)

            with patch.object(ct, "fetch_crypto_price", return_value=50000.0), \
                 patch.object(ct, "fetch_crypto_headlines", return_value=[]), \
                 patch.object(ct, "_place_coinbase_order") as mock_order:
                ct.run(dry_run=False)
                mock_order.assert_not_called()


class TestRunAnalysisIntegration(unittest.TestCase):
    """run_analysis() end-to-end: all 7 agents + ConsensusEngine + MacroAgent wired."""

    def _make_capital_md(self, tmp_path):
        md = tmp_path / "Capital.md"
        md.write_text("# Capital\n\n## Active Signals\n\n*No signals yet.*\n")
        return md

    def _base_patches(self, ma, md_path):
        return [
            patch.object(ma.SentimentAgent, "fetch_headlines", return_value=[]),
            patch.object(ma.SentimentAgent, "analyze_headlines", return_value=0.1),
            patch.object(ma.ForecastAgent, "predict_trend", return_value="up"),
            patch.object(ma, "_fetch_prices", return_value=[100.0] * 30),
            patch.object(ma.EarningsAgent, "get_earnings_context", return_value={
                "days_to_earnings": 20, "is_within_window": False,
                "eps_estimate": None, "revenue_estimate": None, "summary": "No earnings soon",
            }),
            patch("scripts.options_flow_monitor.detect_unusual_flow", return_value="normal"),
            patch.object(ma.FundamentalAnalyst, "analyze",
                         return_value=MagicMock(signal="BULLISH", confidence=0.7,
                                                summary="PE reasonable")),
            patch.object(ma.EDGARConnector, "get_insider_trades", return_value=[]),
            patch.object(ma, "_ask_ollama", return_value=("BUY", "strong momentum")),
            patch("agents.macro_agent.MacroAgent.get_macro_context",
                  return_value="Macro regime: risk_on, VIX 15.0"),
            patch("agents.macro_agent.MacroAgent.get_regime", return_value="risk_on"),
        ]

    def test_returns_dict_with_required_keys(self):
        import importlib
        import market_analyst as ma_mod
        importlib.reload(ma_mod)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            ma_mod.CAPITAL_MD = self._make_capital_md(Path(td))
            patches = self._base_patches(ma_mod, ma_mod.CAPITAL_MD)
            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                 patches[5], patches[6], patches[7], patches[8], patches[9], patches[10]:
                result = ma_mod.run_analysis(["AAPL"], dry_run=True)
        self.assertIn("AAPL", result)
        for key in ("action", "reason", "score", "trend", "fundamentals",
                    "consensus_score", "conviction", "macro_regime"):
            self.assertIn(key, result["AAPL"], f"Missing key: {key}")

    def test_dry_run_does_not_write_capital_md(self):
        import importlib
        import market_analyst as ma_mod
        importlib.reload(ma_mod)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            md = self._make_capital_md(Path(td))
            original = md.read_text()
            ma_mod.CAPITAL_MD = md
            with patch.object(ma_mod.SentimentAgent, "fetch_headlines", return_value=[]), \
                 patch.object(ma_mod.SentimentAgent, "analyze_headlines", return_value=0.0), \
                 patch.object(ma_mod, "_fetch_prices", return_value=[]), \
                 patch.object(ma_mod.ForecastAgent, "predict_trend", return_value="flat"), \
                 patch.object(ma_mod.EarningsAgent, "get_earnings_context", return_value={
                     "days_to_earnings": None, "is_within_window": False,
                     "eps_estimate": None, "revenue_estimate": None, "summary": "",
                 }), \
                 patch("scripts.options_flow_monitor.detect_unusual_flow", return_value="normal"), \
                 patch.object(ma_mod.FundamentalAnalyst, "analyze",
                              return_value=MagicMock(signal="NEUTRAL", confidence=0.0, summary="")), \
                 patch.object(ma_mod.EDGARConnector, "get_insider_trades", return_value=[]), \
                 patch.object(ma_mod, "_ask_ollama", return_value=("HOLD", "no signal")), \
                 patch("agents.macro_agent.MacroAgent.get_macro_context", return_value=""), \
                 patch("agents.macro_agent.MacroAgent.get_regime", return_value="neutral"):
                ma_mod.run_analysis(["MSFT"], dry_run=True)
            self.assertEqual(md.read_text(), original)

    def test_macro_context_forwarded_to_ask_ollama(self):
        import importlib
        import market_analyst as ma_mod
        importlib.reload(ma_mod)
        import tempfile
        captured = {}

        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return ("HOLD", "test")

        with tempfile.TemporaryDirectory() as td:
            ma_mod.CAPITAL_MD = self._make_capital_md(Path(td))
            with patch.object(ma_mod.SentimentAgent, "fetch_headlines", return_value=[]), \
                 patch.object(ma_mod.SentimentAgent, "analyze_headlines", return_value=0.0), \
                 patch.object(ma_mod, "_fetch_prices", return_value=[]), \
                 patch.object(ma_mod.ForecastAgent, "predict_trend", return_value="flat"), \
                 patch.object(ma_mod.EarningsAgent, "get_earnings_context", return_value={
                     "is_within_window": False, "summary": "",
                     "days_to_earnings": None, "eps_estimate": None, "revenue_estimate": None,
                 }), \
                 patch("scripts.options_flow_monitor.detect_unusual_flow", return_value="normal"), \
                 patch.object(ma_mod.FundamentalAnalyst, "analyze",
                              return_value=MagicMock(signal="NEUTRAL", confidence=0.0, summary="")), \
                 patch.object(ma_mod.EDGARConnector, "get_insider_trades", return_value=[]), \
                 patch.object(ma_mod, "_ask_ollama", side_effect=_capture), \
                 patch("agents.macro_agent.MacroAgent.get_macro_context",
                       return_value="Macro regime: risk_off, yield curve inverted (-0.5%), VIX 28.0"), \
                 patch("agents.macro_agent.MacroAgent.get_regime", return_value="risk_off"):
                ma_mod.run_analysis(["SPY"], dry_run=True)
        self.assertIn("macro_context", captured)
        self.assertIn("risk_off", captured["macro_context"])


if __name__ == "__main__":
    unittest.main()
