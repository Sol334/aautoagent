"""SEC EDGAR Form 4 insider trade connector for Galactic Capital.

Fetches insider transactions with a ~2-day disclosure lag — much faster than
the 45-day congressional trade window, giving earlier entry signals.
"""

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, timedelta

import httpx

log = logging.getLogger(__name__)

_TICKERS_URL   = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS   = "https://data.sec.gov/submissions/CIK{}.json"
_FILING_BASE   = "https://www.sec.gov/Archives/edgar/data/{}/{}/{}"
_USER_AGENT    = {"User-Agent": "GalacticCapital research@galacticroof.com"}
_MAX_FILINGS   = 10


@dataclass
class InsiderTrade:
    ticker: str
    insider_name: str
    title: str
    transaction_type: str
    transaction_date: str
    shares: float
    price_per_share: float | None
    value_usd: float | None
    shares_owned_after: float | None
    is_buy: bool
    _id: str


class EDGARConnector:
    def __init__(self, timeout: int = 15):
        self._timeout = timeout
        self._cik_cache: dict[str, str] = {}

    # ── CIK lookup ────────────────────────────────────────────────────────────

    def _load_cik_map(self) -> dict[str, str]:
        """Fetch SEC company_tickers.json once and build ticker→CIK10 map."""
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(_TICKERS_URL, headers=_USER_AGENT)
            resp.raise_for_status()
            data = resp.json()
            return {
                rec["ticker"].upper(): str(rec["cik_str"]).zfill(10)
                for rec in data.values()
                if "ticker" in rec and "cik_str" in rec
            }
        except Exception as exc:
            log.warning("EDGAR CIK map fetch failed: %s", exc)
            return {}

    def get_cik(self, ticker: str) -> str | None:
        """Return zero-padded 10-digit CIK for ticker, or None if unknown."""
        if not self._cik_cache:
            self._cik_cache = self._load_cik_map()
        return self._cik_cache.get(ticker.upper())

    # ── Form 4 fetch + parse ──────────────────────────────────────────────────

    def _fetch_json(self, url: str) -> dict | None:
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url, headers=_USER_AGENT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("EDGAR JSON fetch failed (%s): %s", url, exc)
            return None

    def _fetch_xml(self, url: str) -> str | None:
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url, headers=_USER_AGENT)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            log.warning("EDGAR XML fetch failed (%s): %s", url, exc)
            return None

    @staticmethod
    def _text(el: ET.Element | None, path: str) -> str:
        """Return stripped text at XPath path, or empty string."""
        if el is None:
            return ""
        node = el.find(path)
        return (node.text or "").strip() if node is not None else ""

    def _parse_form4(self, xml_text: str, ticker: str) -> list[InsiderTrade]:
        """Parse a Form 4 XML document into InsiderTrade records."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            log.warning("Form 4 XML parse error for %s: %s", ticker, exc)
            return []

        owner = root.find(".//reportingOwner")
        if owner is None:
            return []

        insider_name = self._text(owner, ".//rptOwnerName")

        # Derive title from officerTitle, or fall back to relationship flags
        title = self._text(owner, ".//reportingOwnerRelationship/officerTitle")
        if not title:
            if self._text(owner, ".//reportingOwnerRelationship/isDirector") == "1":
                title = "Director"
            elif self._text(owner, ".//reportingOwnerRelationship/isTenPercentOwner") == "1":
                title = "10% Owner"

        trades: list[InsiderTrade] = []
        for txn in root.findall(".//nonDerivativeTransaction"):
            try:
                tx_date  = self._text(txn, "transactionDate/value")
                tx_code  = self._text(txn, "transactionCoding/transactionCode")
                shares_s = self._text(txn, "transactionAmounts/transactionShares/value")
                price_s  = self._text(txn, "transactionAmounts/transactionPricePerShare/value")
                after_s  = self._text(txn, "postTransactionAmounts/sharesOwnedFollowingTransaction/value")

                if not shares_s or not tx_date or not tx_code:
                    continue

                shares = float(shares_s)
                price  = float(price_s) if price_s else None
                after  = float(after_s) if after_s else None
                value  = round(shares * price, 2) if price is not None else None
                is_buy = tx_code == "P"

                _id = f"{ticker}:{insider_name}:{tx_date}:{tx_code}:{shares}"

                trades.append(InsiderTrade(
                    ticker=ticker,
                    insider_name=insider_name,
                    title=title,
                    transaction_type=tx_code,
                    transaction_date=tx_date,
                    shares=shares,
                    price_per_share=price,
                    value_usd=value,
                    shares_owned_after=after,
                    is_buy=is_buy,
                    _id=_id,
                ))
            except (ValueError, TypeError) as exc:
                log.debug("Skipping malformed Form 4 transaction for %s: %s", ticker, exc)
                continue

        return trades

    # ── Public interface ──────────────────────────────────────────────────────

    def get_insider_trades(self, ticker: str, days_back: int = 30) -> list[InsiderTrade]:
        """Return Form 4 insider trades for ticker filed within days_back days.

        Never raises — returns [] on any error.
        """
        try:
            return self._get_insider_trades(ticker, days_back)
        except Exception as exc:
            log.warning("get_insider_trades failed for %s: %s", ticker, exc)
            return []

    def _get_insider_trades(self, ticker: str, days_back: int) -> list[InsiderTrade]:
        cik = self.get_cik(ticker)
        if cik is None:
            log.info("No CIK found for ticker %s", ticker)
            return []

        submissions = self._fetch_json(_SUBMISSIONS.format(cik))
        if submissions is None:
            return []

        recent = submissions.get("filings", {}).get("recent", {})
        forms    = recent.get("form", [])
        dates    = recent.get("filingDate", [])
        accnos   = recent.get("accessionNumber", [])
        docs     = recent.get("primaryDocument", [])

        cutoff = (date.today() - timedelta(days=days_back)).isoformat()

        # Collect at most _MAX_FILINGS Form 4s within the date window
        targets: list[tuple[str, str, str]] = []
        for form, filing_date, accno, doc in zip(forms, dates, accnos, docs):
            if form == "4" and filing_date >= cutoff:
                targets.append((accno, doc, filing_date))
            if len(targets) >= _MAX_FILINGS:
                break

        all_trades: list[InsiderTrade] = []
        cik_nodash = cik.lstrip("0")  # EDGAR path uses the numeric CIK without leading zeros
        for accno, doc, _ in targets:
            accno_nodash = accno.replace("-", "")
            url = _FILING_BASE.format(cik_nodash, accno_nodash, doc)
            xml_text = self._fetch_xml(url)
            if xml_text:
                all_trades.extend(self._parse_form4(xml_text, ticker))
            # Respect SEC 10 req/sec rate limit
            time.sleep(0.15)

        return all_trades
