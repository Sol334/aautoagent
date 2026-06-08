#!/usr/bin/env python3
"""SEC EDGAR Form 4 data pipeline — Galactic Capital

Fetches insider trade disclosures (officer/director buy/sell) from the SEC's
free public API. Form 4 filings are the highest-weighted signal (25%) in the
ConsensusEngine.

No API key required. SEC requires a descriptive User-Agent header.

Usage:
    python scripts/edgar_connector.py --ticker AAPL
    python scripts/edgar_connector.py --ticker NVDA --days 60

Env vars:
    CAPITAL_EDGAR_DAYS — default lookback window (default: 30)
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

log = logging.getLogger("edgar_connector")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

# SEC requires a descriptive User-Agent — plain "python-requests" is blocked.
_SEC_HEADERS = {
    "User-Agent": "GalacticCapital/1.0 galacticroof@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_TICKER_JSON  = "https://data.sec.gov/files/company_tickers.json"

# Module-level CIK cache — populated on first lookup per ticker
_CIK_CACHE: dict[str, str] = {}


# ── CIK lookup ────────────────────────────────────────────────────────────────

def get_cik_for_ticker(ticker: str) -> str | None:
    """Return zero-padded 10-digit CIK for ``ticker``, or None on failure.

    Uses the SEC's bulk company_tickers.json (~2 MB) for O(1) lookup after
    the first call. Result is cached in ``_CIK_CACHE``.
    """
    if not ticker:
        return None
    upper = ticker.upper().strip()
    if upper in _CIK_CACHE:
        return _CIK_CACHE[upper]
    try:
        time.sleep(0.1)
        with httpx.Client(timeout=10, headers=_SEC_HEADERS) as client:
            r = client.get(_TICKER_JSON)
            if r.status_code == 200:
                for item in r.json().values():
                    if str(item.get("ticker", "")).upper() == upper:
                        cik = str(item.get("cik_str", "")).zfill(10)
                        _CIK_CACHE[upper] = cik
                        return cik
    except Exception as exc:
        log.warning("CIK lookup failed for %s: %s", ticker, exc)
    return None


# ── Form 4 fetching ───────────────────────────────────────────────────────────

def fetch_form4_filings(ticker: str, days_back: int = 30) -> list[dict]:
    """Return raw EDGAR search-index hits for Form 4 filings on ``ticker``.

    Queries the EDGAR full-text search API (free, no auth). Each element in
    the returned list is a raw ``hits.hits`` dict from the API.
    """
    if not ticker:
        return []
    upper = ticker.upper().strip()
    end   = datetime.utcnow()
    start = end - timedelta(days=days_back)
    params = {
        "q":         f'"{upper}"',
        "forms":     "4",
        "dateRange": "custom",
        "startdt":   start.strftime("%Y-%m-%d"),
        "enddt":     end.strftime("%Y-%m-%d"),
    }
    try:
        time.sleep(0.1)
        with httpx.Client(timeout=10, headers=_SEC_HEADERS) as client:
            r = client.get(_EDGAR_SEARCH, params=params)
            if r.status_code != 200:
                log.warning("EDGAR search returned %s for %s", r.status_code, ticker)
                return []
            hits = r.json().get("hits", {}).get("hits", [])
            return hits if isinstance(hits, list) else []
    except Exception as exc:
        log.warning("EDGAR fetch failed for %s: %s", ticker, exc)
        return []


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_insider_trades(filings: list[dict], ticker: str) -> list[dict]:
    """Normalize raw EDGAR hits into insider-trade dicts.

    Each output record has:
        ticker, insider_name, role, trade_date, trade_type,
        shares, price_per_share, total_value, source

    Records missing ``period_of_report`` are skipped (no date = no signal).
    On any per-record parse error: log warning and continue.
    """
    if not filings or not ticker:
        return []
    upper = ticker.upper().strip()
    records: list[dict] = []
    for hit in filings:
        try:
            if not isinstance(hit, dict):
                continue
            src = hit.get("_source", {})
            trade_date = src.get("period_of_report", "")
            if not trade_date:
                continue

            # Insider name — EDGAR uses "display_names" list
            display = src.get("display_names", [])
            insider_name = display[0].strip() if display else "Unknown"

            # Role — best-effort from file_date_form_type or fallback
            role = "officer"
            biz = src.get("biz_location", "") or ""
            if "director" in str(src).lower() and "officer" not in str(biz).lower():
                role = "director"

            # Trade type — look in transaction_type or fall back to text signals
            tx_type_raw = str(src.get("transaction_type", "") or "")
            highlights  = str(hit.get("highlights", "") or "")
            combined    = (tx_type_raw + " " + highlights).lower()
            if any(kw in combined for kw in ("sale", " s ", "disposition")):
                trade_type = "sale"
            else:
                trade_type = "purchase"

            # Shares and price — present in structured filings, None if absent
            raw_shares = src.get("shares")
            raw_price  = src.get("price_per_share")
            if raw_shares is None:
                continue  # can't compute net without share count
            shares = float(raw_shares)
            price  = float(raw_price) if raw_price is not None else 0.0

            records.append({
                "ticker":          upper,
                "insider_name":    insider_name,
                "role":            role,
                "trade_date":      trade_date,
                "trade_type":      trade_type,
                "shares":          shares,
                "price_per_share": price,
                "total_value":     shares * price,
                "source":          "edgar_form4",
            })
        except Exception as exc:
            log.warning("Skipping malformed Form 4 record: %s", exc)
    return records


# ── Signal aggregation ────────────────────────────────────────────────────────

def get_insider_signal(ticker: str, days_back: int = 30) -> dict:
    """Aggregate Form 4 filings into a ConsensusEngine-compatible signal dict.

    Signal rules:
        net_shares > 0  AND trade_count ≥ 2 → BULLISH
        net_shares < 0  AND trade_count ≥ 2 → BEARISH
        otherwise                            → NEUTRAL
    Confidence = min(1.0, trade_count / 5).
    Never raises — returns NEUTRAL on any exception.
    """
    neutral: dict = {
        "ticker":       (ticker.upper().strip() if ticker else ""),
        "signal":       "NEUTRAL",
        "confidence":   0.0,
        "trade_count":  0,
        "net_shares":   0.0,
        "recent_trades": [],
        "summary":      "No recent insider trades found.",
        "source":       "edgar_form4",
    }
    if not ticker:
        return neutral
    try:
        upper    = ticker.upper().strip()
        filings  = fetch_form4_filings(upper, days_back)
        trades   = parse_insider_trades(filings, upper)
        if not trades:
            return neutral

        net = sum(
            t["shares"] if t["trade_type"] == "purchase" else -t["shares"]
            for t in trades
        )
        count  = len(trades)
        recent = sorted(trades, key=lambda t: t["trade_date"], reverse=True)[:5]

        signal     = "NEUTRAL"
        confidence = 0.0
        if count >= 2:
            if net > 0:
                signal, confidence = "BULLISH", min(1.0, count / 5)
            elif net < 0:
                signal, confidence = "BEARISH", min(1.0, count / 5)

        summary = (
            f"{count} insider {'trade' if count == 1 else 'trades'} for {upper} "
            f"over {days_back}d. Net: {net:+,.0f} shares ({signal})."
        )
        return {
            "ticker":       upper,
            "signal":       signal,
            "confidence":   round(confidence, 2),
            "trade_count":  count,
            "net_shares":   round(net, 2),
            "recent_trades": recent,
            "summary":      summary,
            "source":       "edgar_form4",
        }
    except Exception as exc:
        log.warning("EDGAR signal failed for %s: %s", ticker, exc)
        return neutral


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SEC EDGAR Form 4 insider signal")
    p.add_argument("--ticker", required=True)
    p.add_argument("--days",   type=int, default=int(os.getenv("CAPITAL_EDGAR_DAYS", "30")))
    args = p.parse_args()

    result = get_insider_signal(args.ticker, args.days)
    print(f"\n=== EDGAR Insider Signal: {result['ticker']} ===")
    print(f"  Signal:     {result['signal']} (confidence {result['confidence']:.0%})")
    print(f"  Trades:     {result['trade_count']}  Net shares: {result['net_shares']:+,.0f}")
    print(f"  Summary:    {result['summary']}")
    if result["recent_trades"]:
        print("  Recent:")
        for t in result["recent_trades"]:
            print(f"    {t['trade_date']}  {t['insider_name'][:20]}  {t['trade_type'].upper()}  {t['shares']:,.0f} @ ${t['price_per_share']:.2f}")
