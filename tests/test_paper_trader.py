"""Tests for scripts/paper_trader.py — fully mocked."""
import sys
import types
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Stub heavy deps
for _mod in ("torch", "transformers", "alpaca", "alpaca.trading",
             "alpaca.trading.client", "alpaca.trading.requests",
             "alpaca.trading.enums"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# yfinance stub
if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = types.ModuleType("yfinance")
if not hasattr(sys.modules["yfinance"], "Ticker"):
    sys.modules["yfinance"].Ticker = MagicMock()
if not hasattr(sys.modules["yfinance"], "download"):
    sys.modules["yfinance"].download = MagicMock(return_value=MagicMock())

import scripts.paper_trader as pt


# ── _load_portfolio_state ──────────────────────────────────────────────────────

class TestLoadPortfolioState:
    def test_returns_defaults_when_file_missing(self, tmp_path):
        missing = tmp_path / "portfolio_state.json"
        with patch.object(pt, "PORTFOLIO_STATE_PATH", missing):
            state = pt._load_portfolio_state()
        assert "peak" in state
        assert "current" in state
        assert state["peak"] == pt.STARTING_CAPITAL
        assert state["current"] == pt.STARTING_CAPITAL

    def test_returns_stored_values_when_file_exists(self, tmp_path):
        stored = {"peak": 250.0, "current": 210.0}
        state_file = tmp_path / "portfolio_state.json"
        state_file.write_text(json.dumps(stored))
        with patch.object(pt, "PORTFOLIO_STATE_PATH", state_file):
            state = pt._load_portfolio_state()
        assert state["peak"] == 250.0
        assert state["current"] == 210.0

    def test_returns_defaults_when_file_is_invalid_json(self, tmp_path):
        state_file = tmp_path / "portfolio_state.json"
        state_file.write_text("not valid json {{{")
        with patch.object(pt, "PORTFOLIO_STATE_PATH", state_file):
            state = pt._load_portfolio_state()
        assert state["peak"] == pt.STARTING_CAPITAL
        assert state["current"] == pt.STARTING_CAPITAL


# ── _is_kill_switch_active ────────────────────────────────────────────────────

class TestIsKillSwitchActive:
    def test_returns_true_when_drawdown_exceeds_limit(self):
        big_loss_state = {"peak": 200.0, "current": 100.0}  # 50% drawdown
        with patch.object(pt, "_load_portfolio_state", return_value=big_loss_state), \
             patch.object(pt, "_get_effective_drawdown_limit", return_value=20.0):
            assert pt._is_kill_switch_active() is True

    def test_returns_false_when_drawdown_below_limit(self):
        healthy_state = {"peak": 200.0, "current": 195.0}  # 2.5% drawdown
        with patch.object(pt, "_load_portfolio_state", return_value=healthy_state), \
             patch.object(pt, "_get_effective_drawdown_limit", return_value=20.0):
            assert pt._is_kill_switch_active() is False

    def test_returns_false_when_portfolio_state_empty(self):
        # Missing keys → both peak and current default to STARTING_CAPITAL → 0% drawdown
        empty_state = {}
        with patch.object(pt, "_load_portfolio_state", return_value=empty_state), \
             patch.object(pt, "_get_effective_drawdown_limit", return_value=20.0):
            assert pt._is_kill_switch_active() is False

    def test_returns_false_when_peak_is_zero(self):
        zero_peak = {"peak": 0.0, "current": 0.0}
        with patch.object(pt, "_load_portfolio_state", return_value=zero_peak), \
             patch.object(pt, "_get_effective_drawdown_limit", return_value=20.0):
            assert pt._is_kill_switch_active() is False


# ── _fetch_quotes ─────────────────────────────────────────────────────────────

class TestFetchQuotes:
    def test_returns_empty_dict_when_yfinance_raises(self):
        # Simulate Ticker() itself raising (e.g., network/auth failure)
        sys.modules["yfinance"].Ticker = MagicMock(side_effect=RuntimeError("network error"))
        result = pt._fetch_quotes(["AAPL"])
        # The function catches per-ticker exceptions and stores None
        assert isinstance(result, dict)
        if "AAPL" in result:
            assert result["AAPL"] is None

    def test_returns_price_on_success(self):
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 150.0
        mock_fast_info.regular_market_price = 150.0
        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info
        sys.modules["yfinance"].Ticker = MagicMock(return_value=mock_ticker)
        result = pt._fetch_quotes(["AAPL"])
        assert "AAPL" in result
        assert result["AAPL"] == 150.0

    def test_returns_none_price_when_price_attribute_missing(self):
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = None
        mock_fast_info.regular_market_price = None
        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info
        sys.modules["yfinance"].Ticker = MagicMock(return_value=mock_ticker)
        result = pt._fetch_quotes(["TSLA"])
        assert result.get("TSLA") is None

    def test_returns_prices_for_multiple_tickers(self):
        def make_ticker(price):
            fi = MagicMock()
            fi.last_price = price
            fi.regular_market_price = price
            t = MagicMock()
            t.fast_info = fi
            return t

        calls = {"AAPL": make_ticker(150.0), "MSFT": make_ticker(300.0)}
        sys.modules["yfinance"].Ticker = MagicMock(side_effect=lambda t: calls[t])
        result = pt._fetch_quotes(["AAPL", "MSFT"])
        assert result["AAPL"] == 150.0
        assert result["MSFT"] == 300.0


# ── _read_pending_signals ─────────────────────────────────────────────────────

class TestReadPendingSignals:
    def test_returns_empty_list_when_capital_md_missing(self, tmp_path):
        # tmp_path/brain/Capital.md does not exist — no file created intentionally
        with patch.object(pt, "CAPITAL_ROOT", tmp_path):
            result = pt._read_pending_signals()
        assert result == []

    def test_parses_buy_and_sell_signals_correctly(self, tmp_path):
        brain_dir = tmp_path / "brain"
        brain_dir.mkdir(parents=True)
        capital_md = brain_dir / "Capital.md"
        capital_md.write_text(
            "# Capital\n\n"
            "## Active Signals\n"
            "- NVDA BULLISH — congress purchase\n"
            "- TSLA BEARISH — mass selling\n"
            "## Other Section\n"
            "ignored content\n"
        )
        with patch.object(pt, "CAPITAL_ROOT", tmp_path):
            result = pt._read_pending_signals()
        assert len(result) == 2
        assert any("NVDA" in s for s in result)
        assert any("TSLA" in s for s in result)

    def test_stops_parsing_at_next_section_header(self, tmp_path):
        brain_dir = tmp_path / "brain"
        brain_dir.mkdir(parents=True)
        capital_md = brain_dir / "Capital.md"
        capital_md.write_text(
            "## Active Signals\n"
            "- AAPL BULLISH\n"
            "## Positions\n"
            "- MSFT BEARISH\n"
        )
        with patch.object(pt, "CAPITAL_ROOT", tmp_path):
            result = pt._read_pending_signals()
        # Only lines in Active Signals, not those in Positions
        assert len(result) == 1
        assert any("AAPL" in s for s in result)
        assert not any("MSFT" in s for s in result)

    def test_returns_empty_list_when_no_active_signals_section(self, tmp_path):
        brain_dir = tmp_path / "brain"
        brain_dir.mkdir(parents=True)
        capital_md = brain_dir / "Capital.md"
        capital_md.write_text("# Capital\n\n## Positions\n- AAPL hold\n")
        with patch.object(pt, "CAPITAL_ROOT", tmp_path):
            result = pt._read_pending_signals()
        assert result == []


# ── _log_dryrun ───────────────────────────────────────────────────────────────

class TestLogDryrun:
    def test_creates_file_with_entries(self, tmp_path):
        log_file = tmp_path / "data" / "paper_trades_dryrun.json"
        entries = [{"ticker": "AAPL", "action": "BUY", "price": 150.0}]
        with patch.object(pt, "DRY_RUN_LOG", log_file):
            pt._log_dryrun(entries)
        assert log_file.exists()
        data = json.loads(log_file.read_text())
        assert len(data) == 1
        assert data[0]["ticker"] == "AAPL"

    def test_appends_to_existing_entries(self, tmp_path):
        log_file = tmp_path / "data" / "paper_trades_dryrun.json"
        log_file.parent.mkdir(parents=True)
        log_file.write_text(json.dumps([{"ticker": "MSFT", "action": "HOLD", "price": 300.0}]))
        new_entry = [{"ticker": "NVDA", "action": "SELL", "price": 400.0}]
        with patch.object(pt, "DRY_RUN_LOG", log_file):
            pt._log_dryrun(new_entry)
        data = json.loads(log_file.read_text())
        assert len(data) == 2
        tickers = {e["ticker"] for e in data}
        assert "MSFT" in tickers
        assert "NVDA" in tickers


# ── _get_effective_drawdown_limit ─────────────────────────────────────────────

class TestGetEffectiveDrawdownLimit:
    def test_returns_base_limit_when_macro_agent_unavailable(self):
        with patch.dict("sys.modules", {"agents.macro_agent": None}):
            limit = pt._get_effective_drawdown_limit()
        assert limit == pt.MAX_DRAWDOWN_PCT

    def test_tightens_limit_in_risk_off_regime(self):
        mock_macro = MagicMock()
        mock_macro.get_regime.return_value = "risk_off"
        mock_module = MagicMock()
        mock_module.MacroAgent = MagicMock(return_value=mock_macro)
        with patch.dict("sys.modules", {"agents.macro_agent": mock_module}):
            limit = pt._get_effective_drawdown_limit()
        expected = pt.MAX_DRAWDOWN_PCT * pt._RISK_OFF_DRAWDOWN_MULTIPLIER
        assert limit == pytest.approx(expected)

    def test_returns_base_limit_in_risk_on_regime(self):
        mock_macro = MagicMock()
        mock_macro.get_regime.return_value = "risk_on"
        mock_module = MagicMock()
        mock_module.MacroAgent = MagicMock(return_value=mock_macro)
        with patch.dict("sys.modules", {"agents.macro_agent": mock_module}):
            limit = pt._get_effective_drawdown_limit()
        assert limit == pt.MAX_DRAWDOWN_PCT


# ── run ───────────────────────────────────────────────────────────────────────

class TestRun:
    def _default_patches(self):
        """Return context managers that mock external I/O for run()."""
        return [
            patch.object(pt, "_get_effective_drawdown_limit", return_value=20.0),
            patch.object(pt, "_load_portfolio_state",
                         return_value={"peak": 200.0, "current": 200.0}),
            patch.object(pt, "_read_pending_signals", return_value=[]),
            patch.object(pt, "_fetch_quotes", return_value={"AAPL": 150.0}),
            patch.object(pt, "_log_dryrun"),
            patch.object(pt, "PAPER_TRADING", True),
        ]

    def test_dry_run_never_places_order(self):
        """With dry_run=True, no Alpaca order should be placed."""
        patches = self._default_patches()
        mock_alpaca = MagicMock()
        # patch alpaca client at the module level (it's imported lazily in run())
        with patch.dict("sys.modules", {"alpaca.trading.client": mock_alpaca}):
            for p in patches:
                p.start()
            try:
                pt.run(tickers=["AAPL"], max_tickers=5, dry_run=True)
                # The Alpaca TradingClient should never have been instantiated
                mock_alpaca.TradingClient.assert_not_called()
            finally:
                for p in patches:
                    p.stop()

    def test_kill_switch_skips_all_tickers(self, capsys):
        """When kill switch is active, run() logs a warning and returns early."""
        kill_state = {"peak": 200.0, "current": 100.0}  # 50% drawdown > 20% limit
        with patch.object(pt, "_get_effective_drawdown_limit", return_value=20.0), \
             patch.object(pt, "_load_portfolio_state", return_value=kill_state), \
             patch.object(pt, "_fetch_quotes") as mock_fetch, \
             patch.object(pt, "_log_dryrun") as mock_log, \
             patch.object(pt, "PAPER_TRADING", True):
            pt.run(tickers=["AAPL", "MSFT"], max_tickers=5, dry_run=True)
            # Prices should never be fetched because we returned early
            mock_fetch.assert_not_called()
            # Nothing logged to dryrun log either
            mock_log.assert_not_called()

    def test_run_processes_tickers_up_to_max(self, capsys):
        """run() respects max_tickers by truncating the ticker list."""
        tickers = ["AAPL", "MSFT", "NVDA", "TSLA"]
        captured_tickers = []

        def fake_fetch(tkrs):
            captured_tickers.extend(tkrs)
            return {t: 100.0 for t in tkrs}

        with patch.object(pt, "_get_effective_drawdown_limit", return_value=20.0), \
             patch.object(pt, "_load_portfolio_state",
                          return_value={"peak": 200.0, "current": 200.0}), \
             patch.object(pt, "_read_pending_signals", return_value=[]), \
             patch.object(pt, "_fetch_quotes", side_effect=fake_fetch), \
             patch.object(pt, "_log_dryrun"), \
             patch.object(pt, "PAPER_TRADING", True):
            pt.run(tickers=tickers, max_tickers=2, dry_run=True)
        assert len(captured_tickers) == 2

    def test_run_uses_passed_in_signals(self, capsys):
        """run() prefers the signals dict argument over Capital.md signals."""
        signals = {"AAPL": {"action": "BUY"}}
        with patch.object(pt, "_get_effective_drawdown_limit", return_value=20.0), \
             patch.object(pt, "_load_portfolio_state",
                          return_value={"peak": 200.0, "current": 200.0}), \
             patch.object(pt, "_read_pending_signals", return_value=[]), \
             patch.object(pt, "_fetch_quotes", return_value={"AAPL": 150.0}), \
             patch.object(pt, "_log_dryrun") as mock_log, \
             patch.object(pt, "PAPER_TRADING", True):
            pt.run(tickers=["AAPL"], max_tickers=5, dry_run=True, signals=signals)
            # _log_dryrun should have been called with an entry that has action=BUY
            mock_log.assert_called_once()
            entries = mock_log.call_args[0][0]
            assert entries[0]["action"] == "BUY"
