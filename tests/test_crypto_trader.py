"""Tests for scripts/crypto_trader.py — fully mocked."""
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Stub heavy deps before import
for _mod in ("torch", "transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# yfinance stub — must have Ticker AND fast_info for crypto_trader
if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = types.ModuleType("yfinance")
if not hasattr(sys.modules["yfinance"], "Ticker"):
    sys.modules["yfinance"].Ticker = MagicMock()

# coinbase stub
for _mod in ("coinbase", "coinbase.rest"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

import scripts.crypto_trader as ct


# ---------------------------------------------------------------------------
# fetch_crypto_price
# ---------------------------------------------------------------------------

class TestFetchCryptoPrice:
    def test_returns_none_when_yfinance_raises(self):
        """fetch_crypto_price returns None when yfinance.Ticker raises."""
        mock_ticker = MagicMock()
        mock_ticker.side_effect = Exception("connection error")
        with patch("yfinance.Ticker", mock_ticker):
            result = ct.fetch_crypto_price("BTC-USD")
        assert result is None

    def test_returns_float_on_success(self):
        """fetch_crypto_price returns a float when last_price is available."""
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 65000.0
        mock_fast_info.regular_market_price = None
        mock_ticker_obj = MagicMock()
        mock_ticker_obj.fast_info = mock_fast_info
        with patch("yfinance.Ticker", return_value=mock_ticker_obj):
            result = ct.fetch_crypto_price("BTC-USD")
        assert isinstance(result, float)
        assert result == 65000.0

    def test_uses_regular_market_price_as_fallback(self):
        """fetch_crypto_price falls back to regular_market_price when last_price is None."""
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = None
        mock_fast_info.regular_market_price = 3500.0
        mock_ticker_obj = MagicMock()
        mock_ticker_obj.fast_info = mock_fast_info
        with patch("yfinance.Ticker", return_value=mock_ticker_obj):
            result = ct.fetch_crypto_price("ETH-USD")
        assert result == 3500.0

    def test_returns_none_when_both_prices_are_none(self):
        """fetch_crypto_price returns None when both price attrs are None/falsy."""
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = None
        mock_fast_info.regular_market_price = None
        mock_ticker_obj = MagicMock()
        mock_ticker_obj.fast_info = mock_fast_info
        with patch("yfinance.Ticker", return_value=mock_ticker_obj):
            result = ct.fetch_crypto_price("SOL-USD")
        assert result is None


# ---------------------------------------------------------------------------
# fetch_crypto_headlines
# ---------------------------------------------------------------------------

class TestFetchCryptoHeadlines:
    def test_returns_empty_when_api_key_is_empty(self):
        """fetch_crypto_headlines returns [] when api_key is empty string."""
        result = ct.fetch_crypto_headlines("")
        assert result == []

    def test_returns_empty_when_api_key_is_change_me(self):
        """fetch_crypto_headlines returns [] when api_key is 'CHANGE_ME'."""
        result = ct.fetch_crypto_headlines("CHANGE_ME")
        assert result == []

    def test_returns_parsed_list_on_http_success(self):
        """fetch_crypto_headlines returns list of headlines on HTTP 200."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = [
            {"headline": "Bitcoin surges to new highs"},
            {"headline": "Ethereum upgrade scheduled"},
            {"headline": "Crypto market rallies"},
        ]
        with patch("httpx.get", return_value=mock_resp):
            result = ct.fetch_crypto_headlines("valid_api_key")
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == "Bitcoin surges to new highs"

    def test_returns_empty_on_http_error(self):
        """fetch_crypto_headlines returns [] when httpx raises an exception."""
        with patch("httpx.get", side_effect=Exception("HTTP error")):
            result = ct.fetch_crypto_headlines("valid_api_key")
        assert result == []

    def test_skips_items_without_headline(self):
        """fetch_crypto_headlines skips items that have no headline key."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = [
            {"headline": "Valid headline"},
            {"summary": "No headline here"},
            {"headline": "Another valid headline"},
        ]
        with patch("httpx.get", return_value=mock_resp):
            result = ct.fetch_crypto_headlines("valid_api_key")
        assert len(result) == 2
        assert "Valid headline" in result

    def test_caps_at_10_headlines(self):
        """fetch_crypto_headlines returns at most 10 headlines."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = [
            {"headline": f"Headline {i}"} for i in range(20)
        ]
        with patch("httpx.get", return_value=mock_resp):
            result = ct.fetch_crypto_headlines("valid_api_key")
        assert len(result) == 10


# ---------------------------------------------------------------------------
# _is_live_trading_enabled
# ---------------------------------------------------------------------------

class TestIsLiveTradingEnabled:
    def test_returns_false_when_feature_crypto_not_set(self):
        """_is_live_trading_enabled returns False when FEATURE_CRYPTO is False."""
        with patch.object(ct, "FEATURE_CRYPTO", False):
            with patch.object(ct, "PAPER_TRADING", False):
                with patch.object(ct, "COINBASE_KEY", "real_key"):
                    result = ct._is_live_trading_enabled()
        assert result is False

    def test_returns_false_when_paper_trading_is_true(self):
        """_is_live_trading_enabled returns False when CAPITAL_PAPER_TRADING=true."""
        with patch.object(ct, "FEATURE_CRYPTO", True):
            with patch.object(ct, "PAPER_TRADING", True):
                with patch.object(ct, "COINBASE_KEY", "real_key"):
                    result = ct._is_live_trading_enabled()
        assert result is False

    def test_returns_true_when_all_conditions_met(self):
        """_is_live_trading_enabled returns True when all 3 safety gates pass."""
        with patch.object(ct, "FEATURE_CRYPTO", True):
            with patch.object(ct, "PAPER_TRADING", False):
                with patch.object(ct, "COINBASE_KEY", "real_coinbase_api_key"):
                    result = ct._is_live_trading_enabled()
        assert result is True

    def test_returns_false_when_coinbase_key_is_change_me(self):
        """_is_live_trading_enabled returns False when COINBASE_KEY is 'CHANGE_ME'."""
        with patch.object(ct, "FEATURE_CRYPTO", True):
            with patch.object(ct, "PAPER_TRADING", False):
                with patch.object(ct, "COINBASE_KEY", "CHANGE_ME"):
                    result = ct._is_live_trading_enabled()
        assert result is False

    def test_returns_false_when_coinbase_key_is_empty(self):
        """_is_live_trading_enabled returns False when COINBASE_KEY is empty."""
        with patch.object(ct, "FEATURE_CRYPTO", True):
            with patch.object(ct, "PAPER_TRADING", False):
                with patch.object(ct, "COINBASE_KEY", ""):
                    result = ct._is_live_trading_enabled()
        assert result is False


# ---------------------------------------------------------------------------
# _log_trade
# ---------------------------------------------------------------------------

class TestLogTrade:
    def test_log_trade_writes_entry_to_file(self, tmp_path):
        """_log_trade writes the trade entry to the JSON log file."""
        fake_log_path = tmp_path / "data" / "crypto_paper_trades.json"
        entry = {
            "ts": "2026-06-07T10:00:00",
            "symbol": "BTC-USD",
            "price": 65000.0,
            "sentiment": 0.5,
            "action": "BUY",
            "usd_amount": 20.0,
            "executed": False,
            "paper": True,
        }
        with patch.object(ct, "CRYPTO_LOG", fake_log_path):
            ct._log_trade(entry)

        assert fake_log_path.exists()
        data = json.loads(fake_log_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["symbol"] == "BTC-USD"
        assert data[0]["action"] == "BUY"

    def test_log_trade_appends_to_existing_file(self, tmp_path):
        """_log_trade appends a second entry to the existing log."""
        fake_log_path = tmp_path / "data" / "crypto_paper_trades.json"
        fake_log_path.parent.mkdir(parents=True, exist_ok=True)
        existing = [{"symbol": "ETH-USD", "action": "HOLD"}]
        fake_log_path.write_text(json.dumps(existing))

        new_entry = {"symbol": "BTC-USD", "action": "BUY"}
        with patch.object(ct, "CRYPTO_LOG", fake_log_path):
            ct._log_trade(new_entry)

        data = json.loads(fake_log_path.read_text())
        assert len(data) == 2
        assert data[1]["symbol"] == "BTC-USD"


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class TestRun:
    def _make_mock_sentiment_agent(self, score: float = 0.5):
        mock_agent = MagicMock()
        mock_agent.analyze_headlines.return_value = score
        return mock_agent

    def test_dry_run_never_calls_place_coinbase_order(self):
        """run(dry_run=True) must never call _place_coinbase_order."""
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 65000.0
        mock_fast_info.regular_market_price = None
        mock_ticker_obj = MagicMock()
        mock_ticker_obj.fast_info = mock_fast_info

        with patch("yfinance.Ticker", return_value=mock_ticker_obj):
            with patch.object(ct, "_place_coinbase_order") as mock_order:
                with patch("agents.sentiment_agent.SentimentAgent",
                           return_value=self._make_mock_sentiment_agent(0.5)):
                    with patch.object(ct, "fetch_crypto_headlines", return_value=[]):
                        with patch.object(ct, "_log_trade"):
                            ct.run(dry_run=True)
        mock_order.assert_not_called()

    def test_run_blocks_live_trades_when_live_disabled(self):
        """run does not call _place_coinbase_order when _is_live_trading_enabled is False."""
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 65000.0
        mock_fast_info.regular_market_price = None
        mock_ticker_obj = MagicMock()
        mock_ticker_obj.fast_info = mock_fast_info

        with patch("yfinance.Ticker", return_value=mock_ticker_obj):
            with patch.object(ct, "_place_coinbase_order") as mock_order:
                with patch.object(ct, "_is_live_trading_enabled", return_value=False):
                    with patch("agents.sentiment_agent.SentimentAgent",
                               return_value=self._make_mock_sentiment_agent(0.5)):
                        with patch.object(ct, "fetch_crypto_headlines", return_value=[]):
                            with patch.object(ct, "_log_trade"):
                                ct.run(dry_run=False)
        mock_order.assert_not_called()

    def test_run_calls_log_trade_for_each_symbol_when_not_dry_run(self):
        """run(dry_run=False) calls _log_trade for each symbol with available price."""
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 65000.0
        mock_fast_info.regular_market_price = None
        mock_ticker_obj = MagicMock()
        mock_ticker_obj.fast_info = mock_fast_info

        with patch("yfinance.Ticker", return_value=mock_ticker_obj):
            with patch.object(ct, "_is_live_trading_enabled", return_value=False):
                with patch("agents.sentiment_agent.SentimentAgent",
                           return_value=self._make_mock_sentiment_agent(0.0)):
                    with patch.object(ct, "fetch_crypto_headlines", return_value=[]):
                        with patch.object(ct, "_log_trade") as mock_log:
                            ct.run(dry_run=False)
        # There are 3 symbols in CRYPTO_SYMBOLS
        assert mock_log.call_count == 3

    def test_run_skips_symbol_when_price_unavailable(self):
        """run does not log a trade for symbols where price is None."""
        with patch.object(ct, "fetch_crypto_price", return_value=None):
            with patch.object(ct, "_is_live_trading_enabled", return_value=False):
                with patch("agents.sentiment_agent.SentimentAgent",
                           return_value=self._make_mock_sentiment_agent(0.0)):
                    with patch.object(ct, "fetch_crypto_headlines", return_value=[]):
                        with patch.object(ct, "_log_trade") as mock_log:
                            ct.run(dry_run=False)
        mock_log.assert_not_called()
