"""Tests for data_pipelines/edgar_connector.py — fully mocked, no real HTTP."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipelines.edgar_connector import EDGARConnector, InsiderTrade

# ── Shared test fixtures ───────────────────────────────────────────────────────

_TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
}

_SUBMISSIONS_JSON = {
    "filings": {
        "recent": {
            "form":            ["4", "10-K", "4"],
            "filingDate":      ["2026-05-20", "2026-04-01", "2026-05-15"],
            "accessionNumber": [
                "0000320193-26-000001",
                "0000320193-26-000002",
                "0000320193-26-000003",
            ],
            "primaryDocument": [
                "xslF345X05/wf-form4_001.xml",
                "aapl10k.htm",
                "xslF345X05/wf-form4_003.xml",
            ],
        }
    }
}

_FORM4_PURCHASE_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>COOK TIMOTHY D</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <officerTitle>CEO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-20</value></transactionDate>
      <transactionCoding>
        <transactionCode>P</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare><value>185.50</value></transactionPricePerShare>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>3500000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

_FORM4_SALE_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>COOK TIMOTHY D</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <officerTitle>CEO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-20</value></transactionDate>
      <transactionCoding>
        <transactionCode>S</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>50000</value></transactionShares>
        <transactionPricePerShare><value>185.92</value></transactionPricePerShare>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>3450000</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def _make_resp(json_data=None, text_data=None, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json.return_value = json_data
    if text_data is not None:
        resp.text = text_data
    return resp


def _httpx_client_get_factory(*responses):
    """Return a context-manager mock whose .get() cycles through responses."""
    idx = {"i": 0}

    def _get(url, **kwargs):
        resp = responses[idx["i"]] if idx["i"] < len(responses) else responses[-1]
        idx["i"] += 1
        return resp

    client_mock = MagicMock()
    client_mock.__enter__ = MagicMock(return_value=client_mock)
    client_mock.__exit__ = MagicMock(return_value=False)
    client_mock.get = MagicMock(side_effect=_get)
    return client_mock


# ── Test cases ─────────────────────────────────────────────────────────────────

class TestGetCikFound(unittest.TestCase):
    def test_get_cik_found(self):
        resp = _make_resp(json_data=_TICKERS_JSON)
        client_mock = _httpx_client_get_factory(resp)
        with patch("httpx.Client", return_value=client_mock):
            connector = EDGARConnector()
            cik = connector.get_cik("AAPL")
        self.assertEqual(cik, "0000320193")

    def test_get_cik_case_insensitive(self):
        resp = _make_resp(json_data=_TICKERS_JSON)
        client_mock = _httpx_client_get_factory(resp)
        with patch("httpx.Client", return_value=client_mock):
            connector = EDGARConnector()
            cik = connector.get_cik("aapl")
        self.assertEqual(cik, "0000320193")


class TestGetCikNotFound(unittest.TestCase):
    def test_get_cik_not_found(self):
        # Response with no TSLA entry
        resp = _make_resp(json_data=_TICKERS_JSON)
        client_mock = _httpx_client_get_factory(resp)
        with patch("httpx.Client", return_value=client_mock):
            connector = EDGARConnector()
            cik = connector.get_cik("TSLA")
        self.assertIsNone(cik)

    def test_get_cik_network_error_returns_none(self):
        client_mock = MagicMock()
        client_mock.__enter__ = MagicMock(return_value=client_mock)
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.get = MagicMock(side_effect=Exception("network down"))
        with patch("httpx.Client", return_value=client_mock):
            connector = EDGARConnector()
            cik = connector.get_cik("AAPL")
        self.assertIsNone(cik)


class TestGetInsiderTradesPurchase(unittest.TestCase):
    @patch("time.sleep")
    def test_purchase_is_buy_true(self, _sleep):
        tickers_resp     = _make_resp(json_data=_TICKERS_JSON)
        submissions_resp = _make_resp(json_data=_SUBMISSIONS_JSON)
        form4_resp_1     = _make_resp(text_data=_FORM4_PURCHASE_XML)
        form4_resp_2     = _make_resp(text_data=_FORM4_PURCHASE_XML)

        client_mock = _httpx_client_get_factory(
            tickers_resp, submissions_resp, form4_resp_1, form4_resp_2
        )
        with patch("httpx.Client", return_value=client_mock):
            connector = EDGARConnector()
            trades = connector.get_insider_trades("AAPL", days_back=60)

        self.assertTrue(len(trades) > 0)
        for t in trades:
            self.assertIsInstance(t, InsiderTrade)
            self.assertTrue(t.is_buy)
            self.assertEqual(t.transaction_type, "P")
            self.assertEqual(t.ticker, "AAPL")


class TestGetInsiderTradesSale(unittest.TestCase):
    @patch("time.sleep")
    def test_sale_is_buy_false(self, _sleep):
        tickers_resp     = _make_resp(json_data=_TICKERS_JSON)
        submissions_resp = _make_resp(json_data=_SUBMISSIONS_JSON)
        form4_resp_1     = _make_resp(text_data=_FORM4_SALE_XML)
        form4_resp_2     = _make_resp(text_data=_FORM4_SALE_XML)

        client_mock = _httpx_client_get_factory(
            tickers_resp, submissions_resp, form4_resp_1, form4_resp_2
        )
        with patch("httpx.Client", return_value=client_mock):
            connector = EDGARConnector()
            trades = connector.get_insider_trades("AAPL", days_back=60)

        self.assertTrue(len(trades) > 0)
        for t in trades:
            self.assertFalse(t.is_buy)
            self.assertEqual(t.transaction_type, "S")


class TestGetInsiderTradesEmptyOnNetworkError(unittest.TestCase):
    def test_empty_on_network_error(self):
        # CIK fetch raises; should return []
        client_mock = MagicMock()
        client_mock.__enter__ = MagicMock(return_value=client_mock)
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.get = MagicMock(side_effect=Exception("connection refused"))

        with patch("httpx.Client", return_value=client_mock):
            connector = EDGARConnector()
            result = connector.get_insider_trades("AAPL", days_back=30)

        self.assertEqual(result, [])

    @patch("time.sleep")
    def test_empty_when_submissions_fetch_fails(self, _sleep):
        tickers_resp = _make_resp(json_data=_TICKERS_JSON)

        call_count = {"n": 0}

        def _get(url, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return tickers_resp
            raise Exception("EDGAR down")

        client_mock = MagicMock()
        client_mock.__enter__ = MagicMock(return_value=client_mock)
        client_mock.__exit__ = MagicMock(return_value=False)
        client_mock.get = MagicMock(side_effect=_get)

        with patch("httpx.Client", return_value=client_mock):
            connector = EDGARConnector()
            result = connector.get_insider_trades("AAPL", days_back=30)

        self.assertEqual(result, [])


class TestValueUsdComputed(unittest.TestCase):
    @patch("time.sleep")
    def test_value_usd_is_shares_times_price(self, _sleep):
        tickers_resp     = _make_resp(json_data=_TICKERS_JSON)
        submissions_resp = _make_resp(json_data=_SUBMISSIONS_JSON)
        form4_resp_1     = _make_resp(text_data=_FORM4_PURCHASE_XML)
        form4_resp_2     = _make_resp(text_data=_FORM4_PURCHASE_XML)

        client_mock = _httpx_client_get_factory(
            tickers_resp, submissions_resp, form4_resp_1, form4_resp_2
        )
        with patch("httpx.Client", return_value=client_mock):
            connector = EDGARConnector()
            trades = connector.get_insider_trades("AAPL", days_back=60)

        self.assertTrue(len(trades) > 0)
        for t in trades:
            if t.price_per_share is not None:
                expected = round(t.shares * t.price_per_share, 2)
                self.assertAlmostEqual(t.value_usd, expected, places=2)

    def test_value_usd_none_when_no_price(self):
        """value_usd stays None when price_per_share is missing from XML."""
        xml_no_price = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>JANE DOE</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><officerTitle>CFO</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-01</value></transactionDate>
      <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

        connector = EDGARConnector()
        trades = connector._parse_form4(xml_no_price, "AAPL")
        self.assertEqual(len(trades), 1)
        self.assertIsNone(trades[0].price_per_share)
        self.assertIsNone(trades[0].value_usd)


class TestTitleDerivation(unittest.TestCase):
    def test_director_flag_used_when_no_officer_title(self):
        xml_director = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>BOARD MEMBER</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>100.00</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""
        connector = EDGARConnector()
        trades = connector._parse_form4(xml_director, "AAPL")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].title, "Director")

    def test_ten_percent_owner_flag(self):
        xml_owner = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>BIG FUND LP</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isTenPercentOwner>1</isTenPercentOwner>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>200000</value></transactionShares>
        <transactionPricePerShare><value>50.00</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""
        connector = EDGARConnector()
        trades = connector._parse_form4(xml_owner, "AAPL")
        self.assertEqual(trades[0].title, "10% Owner")


class TestIdDeduplication(unittest.TestCase):
    @patch("time.sleep")
    def test_id_format(self, _sleep):
        tickers_resp     = _make_resp(json_data=_TICKERS_JSON)
        submissions_resp = _make_resp(json_data=_SUBMISSIONS_JSON)
        form4_resp_1     = _make_resp(text_data=_FORM4_PURCHASE_XML)
        form4_resp_2     = _make_resp(text_data=_FORM4_PURCHASE_XML)

        client_mock = _httpx_client_get_factory(
            tickers_resp, submissions_resp, form4_resp_1, form4_resp_2
        )
        with patch("httpx.Client", return_value=client_mock):
            connector = EDGARConnector()
            trades = connector.get_insider_trades("AAPL", days_back=60)

        for t in trades:
            parts = t._id.split(":")
            self.assertEqual(parts[0], "AAPL")
            self.assertIn(t.transaction_type, parts)


if __name__ == "__main__":
    unittest.main()
