"""Tests for scripts/insider_tracker.py — fully mocked."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.insider_tracker as it


# ---------------------------------------------------------------------------
# Helper — build a mock InsiderTrade with required attributes
# ---------------------------------------------------------------------------

def _make_trade(
    ticker="NVDA",
    insider_name="John Doe",
    title="CFO",
    is_buy=True,
    transaction_date="2026-06-01",
    shares=1000.0,
    price_per_share=500.0,
    value_usd=500000.0,
):
    trade = MagicMock()
    trade.ticker = ticker
    trade.insider_name = insider_name
    trade.title = title
    trade.is_buy = is_buy
    trade.transaction_date = transaction_date
    trade.shares = shares
    trade.price_per_share = price_per_share
    trade.value_usd = value_usd
    return trade


# ---------------------------------------------------------------------------
# _fmt_num
# ---------------------------------------------------------------------------

class TestFmtNum:
    def test_none_returns_dash(self):
        """_fmt_num(None) returns the em-dash placeholder."""
        assert it._fmt_num(None) == "—"

    def test_millions_format(self):
        """_fmt_num(1_500_000) returns '$1.5M'."""
        assert it._fmt_num(1_500_000) == "$1.5M"

    def test_thousands_format(self):
        """_fmt_num(50_000) returns a comma-formatted dollar string."""
        result = it._fmt_num(50_000)
        assert result.startswith("$")
        assert "50,000" in result

    def test_small_value_format(self):
        """_fmt_num(500) returns '$500.00' (default 2 decimal places)."""
        assert it._fmt_num(500) == "$500.00"

    def test_large_millions(self):
        """_fmt_num(2_000_000) returns '$2.0M'."""
        assert it._fmt_num(2_000_000) == "$2.0M"

    def test_exactly_one_million(self):
        """_fmt_num(1_000_000) returns '$1.0M'."""
        assert it._fmt_num(1_000_000) == "$1.0M"

    def test_thousands_with_decimals_param(self):
        """_fmt_num respects the decimals parameter."""
        result = it._fmt_num(5_000, decimals=0)
        assert "$5,000" in result

    def test_zero_returns_formatted(self):
        """_fmt_num(0) returns '$0.00'."""
        assert it._fmt_num(0) == "$0.00"


# ---------------------------------------------------------------------------
# _print_trades
# ---------------------------------------------------------------------------

class TestPrintTrades:
    def test_empty_list_prints_no_trades_found(self, capsys):
        """_print_trades([]) prints '(no trades found)'."""
        it._print_trades([])
        captured = capsys.readouterr()
        assert "(no trades found)" in captured.out

    def test_single_trade_prints_header_and_row(self, capsys):
        """_print_trades with one trade prints a header and one data row."""
        trade = _make_trade(ticker="NVDA", insider_name="Jane Smith", is_buy=True)
        it._print_trades([trade])
        captured = capsys.readouterr()
        assert "NVDA" in captured.out
        assert "Jane Smith" in captured.out
        assert "BUY" in captured.out

    def test_sell_trade_shows_sell_label(self, capsys):
        """_print_trades correctly labels a SELL transaction."""
        trade = _make_trade(ticker="AAPL", is_buy=False)
        it._print_trades([trade])
        captured = capsys.readouterr()
        assert "SELL" in captured.out


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_calls_get_insider_trades_for_each_ticker(self):
        """run calls EDGARConnector.get_insider_trades for each ticker in the list."""
        mock_connector = MagicMock()
        mock_connector.get_insider_trades.return_value = []

        with patch("scripts.insider_tracker.EDGARConnector",
                   return_value=mock_connector):
            it.run(["NVDA", "AAPL"], days=30)

        assert mock_connector.get_insider_trades.call_count == 2
        calls = [c.args[0] for c in mock_connector.get_insider_trades.call_args_list]
        assert "NVDA" in calls
        assert "AAPL" in calls

    def test_run_returns_dict_with_ticker_keys(self):
        """run returns a dict keyed by the requested tickers."""
        mock_connector = MagicMock()
        mock_connector.get_insider_trades.return_value = []

        with patch("scripts.insider_tracker.EDGARConnector",
                   return_value=mock_connector):
            result = it.run(["MSFT", "NVDA"], days=14)

        assert isinstance(result, dict)
        assert "MSFT" in result
        assert "NVDA" in result

    def test_run_handles_empty_trade_list_without_crashing(self):
        """run does not raise when a ticker has no insider trades."""
        mock_connector = MagicMock()
        mock_connector.get_insider_trades.return_value = []

        with patch("scripts.insider_tracker.EDGARConnector",
                   return_value=mock_connector):
            result = it.run(["SPY"], days=30)

        assert result["SPY"] == []

    def test_run_stores_trades_in_result(self):
        """run stores the returned InsiderTrade objects in the result dict."""
        trade1 = _make_trade(ticker="NVDA", is_buy=True)
        trade2 = _make_trade(ticker="NVDA", is_buy=False, transaction_date="2026-05-28")
        mock_connector = MagicMock()
        mock_connector.get_insider_trades.return_value = [trade1, trade2]

        with patch("scripts.insider_tracker.EDGARConnector",
                   return_value=mock_connector):
            result = it.run(["NVDA"], days=30)

        assert len(result["NVDA"]) == 2

    def test_run_passes_days_to_connector(self):
        """run passes the days argument as days_back to get_insider_trades."""
        mock_connector = MagicMock()
        mock_connector.get_insider_trades.return_value = []

        with patch("scripts.insider_tracker.EDGARConnector",
                   return_value=mock_connector):
            it.run(["AAPL"], days=7)

        mock_connector.get_insider_trades.assert_called_once_with("AAPL", days_back=7)

    def test_run_single_ticker_returns_single_key(self):
        """run with one ticker returns a dict with exactly one key."""
        mock_connector = MagicMock()
        mock_connector.get_insider_trades.return_value = []

        with patch("scripts.insider_tracker.EDGARConnector",
                   return_value=mock_connector):
            result = it.run(["TSLA"], days=30)

        assert list(result.keys()) == ["TSLA"]
