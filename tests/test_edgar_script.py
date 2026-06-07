"""Tests for scripts/edgar_connector.py — functional module, fully mocked."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import edgar_connector as ec


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_response(json_data=None, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def _make_client_mock(response):
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=response)
    return client


_TICKER_JSON_DATA = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp."},
}

_HITS_PURCHASE = [
    {
        "_source": {
            "period_of_report": "2026-05-15",
            "display_names": ["Tim Cook"],
            "transaction_type": "purchase",
            "shares": 5000,
            "price_per_share": 180.0,
        },
        "highlights": "",
    }
]

_HITS_SALE = [
    {
        "_source": {
            "period_of_report": "2026-05-15",
            "display_names": ["Jane Smith"],
            "transaction_type": "sale",
            "shares": 10000,
            "price_per_share": 200.0,
        },
        "highlights": "",
    }
]


# ── CIK lookup ─────────────────────────────────────────────────────────────────

class TestGetCikForTicker(unittest.TestCase):
    def setUp(self):
        ec._CIK_CACHE.clear()

    @patch("time.sleep")
    def test_returns_zero_padded_cik(self, _sleep):
        resp = _mock_response(json_data=_TICKER_JSON_DATA)
        with patch("httpx.Client", return_value=_make_client_mock(resp)):
            cik = ec.get_cik_for_ticker("AAPL")
        self.assertEqual(cik, "0000320193")

    @patch("time.sleep")
    def test_case_insensitive(self, _sleep):
        resp = _mock_response(json_data=_TICKER_JSON_DATA)
        with patch("httpx.Client", return_value=_make_client_mock(resp)):
            cik = ec.get_cik_for_ticker("aapl")
        self.assertEqual(cik, "0000320193")

    @patch("time.sleep")
    def test_unknown_ticker_returns_none(self, _sleep):
        resp = _mock_response(json_data=_TICKER_JSON_DATA)
        with patch("httpx.Client", return_value=_make_client_mock(resp)):
            cik = ec.get_cik_for_ticker("UNKNOWN")
        self.assertIsNone(cik)

    def test_empty_ticker_returns_none(self):
        result = ec.get_cik_for_ticker("")
        self.assertIsNone(result)

    @patch("time.sleep")
    def test_network_error_returns_none(self, _sleep):
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.get = MagicMock(side_effect=Exception("network error"))
        with patch("httpx.Client", return_value=client):
            cik = ec.get_cik_for_ticker("AAPL")
        self.assertIsNone(cik)


# ── Form 4 fetch ───────────────────────────────────────────────────────────────

class TestFetchForm4Filings(unittest.TestCase):
    @patch("time.sleep")
    def test_returns_hits_on_success(self, _sleep):
        payload = {"hits": {"hits": [{"_source": {}}]}}
        resp = _mock_response(json_data=payload)
        with patch("httpx.Client", return_value=_make_client_mock(resp)):
            result = ec.fetch_form4_filings("AAPL")
        self.assertEqual(len(result), 1)

    @patch("time.sleep")
    def test_non_200_returns_empty(self, _sleep):
        resp = _mock_response(status_code=429)
        with patch("httpx.Client", return_value=_make_client_mock(resp)):
            result = ec.fetch_form4_filings("AAPL")
        self.assertEqual(result, [])

    def test_empty_ticker_returns_empty(self):
        result = ec.fetch_form4_filings("")
        self.assertEqual(result, [])

    @patch("time.sleep")
    def test_exception_returns_empty(self, _sleep):
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.get = MagicMock(side_effect=Exception("timeout"))
        with patch("httpx.Client", return_value=client):
            result = ec.fetch_form4_filings("NVDA")
        self.assertEqual(result, [])


# ── Parse insider trades ───────────────────────────────────────────────────────

class TestParseInsiderTrades(unittest.TestCase):
    def test_purchase_classified_correctly(self):
        records = ec.parse_insider_trades(_HITS_PURCHASE, "AAPL")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["trade_type"], "purchase")
        self.assertEqual(records[0]["ticker"], "AAPL")

    def test_sale_classified_correctly(self):
        records = ec.parse_insider_trades(_HITS_SALE, "AAPL")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["trade_type"], "sale")

    def test_total_value_computed(self):
        records = ec.parse_insider_trades(_HITS_PURCHASE, "AAPL")
        self.assertAlmostEqual(records[0]["total_value"], 5000 * 180.0)

    def test_record_missing_shares_is_skipped(self):
        hit_no_shares = [
            {
                "_source": {
                    "period_of_report": "2026-05-15",
                    "display_names": ["Someone"],
                    "shares": None,
                    "price_per_share": 100.0,
                },
                "highlights": "",
            }
        ]
        records = ec.parse_insider_trades(hit_no_shares, "AAPL")
        self.assertEqual(records, [])

    def test_record_missing_date_is_skipped(self):
        hit_no_date = [
            {
                "_source": {
                    "period_of_report": "",
                    "display_names": ["Someone"],
                    "shares": 1000,
                    "price_per_share": 100.0,
                },
                "highlights": "",
            }
        ]
        records = ec.parse_insider_trades(hit_no_date, "AAPL")
        self.assertEqual(records, [])

    def test_empty_inputs_return_empty(self):
        self.assertEqual(ec.parse_insider_trades([], "AAPL"), [])
        self.assertEqual(ec.parse_insider_trades(_HITS_PURCHASE, ""), [])


# ── Signal aggregation ─────────────────────────────────────────────────────────

class TestGetInsiderSignal(unittest.TestCase):
    def test_empty_ticker_returns_neutral(self):
        result = ec.get_insider_signal("")
        self.assertEqual(result["signal"], "NEUTRAL")
        self.assertEqual(result["trade_count"], 0)

    def test_bullish_on_net_purchases(self):
        many_purchases = _HITS_PURCHASE * 3  # 3 purchases → count≥2, net>0
        search_resp = _mock_response(json_data={"hits": {"hits": many_purchases}})
        with patch("time.sleep"), patch("httpx.Client", return_value=_make_client_mock(search_resp)):
            result = ec.get_insider_signal("AAPL")
        self.assertEqual(result["signal"], "BULLISH")
        self.assertGreater(result["confidence"], 0.0)
        self.assertGreater(result["net_shares"], 0)

    def test_bearish_on_net_sales(self):
        many_sales = _HITS_SALE * 3
        search_resp = _mock_response(json_data={"hits": {"hits": many_sales}})
        with patch("time.sleep"), patch("httpx.Client", return_value=_make_client_mock(search_resp)):
            result = ec.get_insider_signal("AAPL")
        self.assertEqual(result["signal"], "BEARISH")
        self.assertLess(result["net_shares"], 0)

    def test_neutral_when_single_trade(self):
        # Only 1 trade → count < 2 → NEUTRAL regardless of direction
        search_resp = _mock_response(json_data={"hits": {"hits": _HITS_PURCHASE}})
        with patch("time.sleep"), patch("httpx.Client", return_value=_make_client_mock(search_resp)):
            result = ec.get_insider_signal("AAPL")
        self.assertEqual(result["signal"], "NEUTRAL")

    def test_exception_returns_neutral_never_raises(self):
        with patch("edgar_connector.fetch_form4_filings", side_effect=Exception("boom")):
            result = ec.get_insider_signal("AAPL")
        self.assertEqual(result["signal"], "NEUTRAL")
        self.assertEqual(result["source"], "edgar_form4")

    def test_confidence_capped_at_one(self):
        many = _HITS_PURCHASE * 10  # 10 trades → min(1.0, 10/5) = 1.0
        search_resp = _mock_response(json_data={"hits": {"hits": many}})
        with patch("time.sleep"), patch("httpx.Client", return_value=_make_client_mock(search_resp)):
            result = ec.get_insider_signal("AAPL")
        self.assertLessEqual(result["confidence"], 1.0)

    def test_recent_trades_capped_at_five(self):
        many = _HITS_PURCHASE * 10
        search_resp = _mock_response(json_data={"hits": {"hits": many}})
        with patch("time.sleep"), patch("httpx.Client", return_value=_make_client_mock(search_resp)):
            result = ec.get_insider_signal("AAPL")
        self.assertLessEqual(len(result["recent_trades"]), 5)


if __name__ == "__main__":
    unittest.main()
