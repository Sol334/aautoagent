"""Finnhub API wrapper for Galactic Capital.

Free tier: 60 API calls/minute.
Congressional trading endpoint requires Finnhub API key.
"""

import re
import time
import logging

import httpx

log = logging.getLogger(__name__)

_BASE = "https://finnhub.io/api/v1"
_RETRY_DELAYS = (2, 4, 8)


def _parse_amount(amount_str: str) -> float:
    """Convert '$50,001 - $100,000' range string to numeric midpoint."""
    if not amount_str:
        return 0.0
    digits = re.findall(r"[\d,]+", amount_str.replace(",", ""))
    nums = [int(d) for d in digits if d.isdigit()]
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


class FinnhubConnector:
    def __init__(self, api_key: str):
        if not api_key or api_key == "CHANGE_ME":
            raise ValueError("FINNHUB_API_KEY not set — add it to .env")
        self._key = api_key
        self._last_call = 0.0

    def _get(self, path: str, params: dict | None = None, timeout: int = 15) -> dict:
        """GET with rate limiting (1 req/sec) and retry on 429."""
        params = params or {}
        params["token"] = self._key

        # Respect free-tier rate limit (60/min ≈ 1/sec)
        elapsed = time.monotonic() - self._last_call
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
            self._last_call = time.monotonic()
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.get(f"{_BASE}{path}", params=params)
                if resp.status_code == 429 and delay is not None:
                    log.warning("Finnhub rate limit hit — retrying in %ds", delay)
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                if attempt == len(_RETRY_DELAYS):
                    raise
                log.warning("Finnhub HTTP %s — retrying", exc.response.status_code)
                time.sleep(delay)
        return {}

    def congressional_trades(self, symbol: str, from_date: str, to_date: str) -> list:
        """Return list of congressional trade records for a symbol."""
        data = self._get(
            "/stock/congressional-trading",
            {"symbol": symbol, "from": from_date, "to": to_date},
        )
        raw = data.get("data") or []
        # Normalize — attach parsed amount and generate stable ID
        trades = []
        for rec in raw:
            rec = dict(rec)
            rec["_amount_mid"] = _parse_amount(rec.get("amount", ""))
            rec["_id"] = f"{symbol}:{rec.get('name','')}:{rec.get('transactionDate','')}:{rec.get('transactionType','')}"
            trades.append(rec)
        return trades

    def insider_transactions(self, symbol: str) -> list:
        """Return recent insider buy/sell transactions."""
        data = self._get("/stock/insider-transactions", {"symbol": symbol})
        return data.get("data") or []

    def quote(self, symbol: str) -> dict:
        """Return current quote dict: {c: close, h: high, l: low, o: open, pc: prev_close}."""
        return self._get("/quote", {"symbol": symbol})
