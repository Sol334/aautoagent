"""Tests for congressional_tracker.py."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import congressional_tracker as ct


def _make_trade(ticker, tx_type="purchase", days_ago=5, member="John Doe"):
    d = (date.today() - timedelta(days=days_ago)).isoformat()
    return {"ticker": ticker, "type": tx_type, "date": d, "member": member, "source": "test"}


class TestAnalyzeSignals:
    def test_no_trades_empty(self):
        assert ct.analyze_signals([]) == []

    def test_single_buyer_no_signal(self):
        trades = [_make_trade("NVDA", member="Alice")]
        signals = ct.analyze_signals(trades, min_cluster=2)
        assert signals == []

    def test_two_buyers_produces_signal(self):
        trades = [_make_trade("NVDA", member="Alice"), _make_trade("NVDA", member="Bob")]
        signals = ct.analyze_signals(trades, min_cluster=2)
        assert len(signals) == 1
        assert signals[0]["ticker"] == "NVDA"
        assert signals[0]["member_count"] == 2

    def test_sells_excluded(self):
        trades = [_make_trade("NVDA", "sale", member="Alice"), _make_trade("NVDA", "sale", member="Bob")]
        signals = ct.analyze_signals(trades, min_cluster=2)
        assert signals == []

    def test_sorted_by_member_count(self):
        trades = (
            [_make_trade("AAPL", member=f"M{i}") for i in range(5)]
            + [_make_trade("TSLA", member="X"), _make_trade("TSLA", member="Y")]
        )
        signals = ct.analyze_signals(trades, min_cluster=2)
        assert signals[0]["ticker"] == "AAPL"

    def test_old_trades_excluded(self):
        trades = [_make_trade("MSFT", days_ago=35, member="Alice"), _make_trade("MSFT", days_ago=35, member="Bob")]
        signals = ct.analyze_signals(trades, min_cluster=2, cluster_days=30)
        assert signals == []

    def test_confidence_scales_with_members(self):
        trades = [_make_trade("SPY", member=f"M{i}") for i in range(5)]
        signals = ct.analyze_signals(trades, min_cluster=2)
        assert signals[0]["confidence"] == 1.0

    def test_duplicate_member_deduplicated(self):
        trades = [_make_trade("AAPL", member="Alice")] * 3
        signals = ct.analyze_signals(trades, min_cluster=2)
        assert signals == []  # only 1 unique member


class TestSectorPressure:
    def test_buy_counts(self):
        trades = [_make_trade("NVDA", "purchase") for _ in range(3)]
        result = ct.sector_pressure(trades)
        assert result["TECH"]["buys"] == 3

    def test_sell_counts(self):
        trades = [_make_trade("XOM", "sale")]
        result = ct.sector_pressure(trades)
        assert result["ENERGY"]["sells"] == 1

    def test_unknown_ticker_goes_to_other(self):
        trades = [_make_trade("ZYXW", "purchase")]
        result = ct.sector_pressure(trades)
        assert result["OTHER"]["buys"] == 1


class TestWatchlistActivity:
    def test_filters_to_watchlist(self):
        with patch.object(ct, "WATCHLIST", ["AAPL", "MSFT"]):
            trades = [_make_trade("AAPL"), _make_trade("NVDA"), _make_trade("MSFT")]
            filtered = ct.watchlist_activity(trades)
            tickers = [t["ticker"] for t in filtered]
            assert "AAPL" in tickers
            assert "MSFT" in tickers
            assert "NVDA" not in tickers


class TestTickerSector:
    def test_known_ticker(self):
        assert ct.ticker_sector("NVDA") == "TECH"
        assert ct.ticker_sector("XOM") == "ENERGY"
        assert ct.ticker_sector("LMT") == "DEFENSE"

    def test_unknown_ticker(self):
        assert ct.ticker_sector("ZYXW") == "OTHER"

    def test_case_insensitive(self):
        assert ct.ticker_sector("nvda") == "TECH"


class TestFetchHouseWatcher:
    def test_network_error_returns_empty(self):
        import httpx
        with patch("httpx.get", side_effect=httpx.RequestError("err", request=MagicMock())):
            result = ct.fetch_house_watcher()
        assert result == []

    def test_parses_response(self):
        from datetime import date, timedelta
        recent = (date.today() - timedelta(days=5)).isoformat()
        mock_data = [
            {"representative": "Jane Smith", "ticker": "AAPL", "type": "purchase",
             "transaction_date": recent, "amount": "$15,001 - $50,000", "district": "AL-01"}
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_data
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=mock_resp):
            result = ct.fetch_house_watcher(days_back=30)
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"


class TestStateIO:
    def test_load_missing_returns_default(self, tmp_path):
        with patch.object(ct, "STATE_PATH", tmp_path / "state.json"):
            state = ct.load_state()
        assert "last_run" in state
        assert "seen_signals" in state

    def test_save_and_load(self, tmp_path):
        state_file = tmp_path / "state.json"
        data = {"last_run": "2026-05-01", "seen_signals": ["AAPL"]}
        with patch.object(ct, "STATE_PATH", state_file):
            ct.save_state(data)
            loaded = ct.load_state()
        assert loaded == data
