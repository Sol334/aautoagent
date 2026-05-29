"""
galactic-capital/agents/earnings_agent.py

Earnings calendar context agent. Uses yfinance (free, no key) to fetch upcoming
earnings dates and analyst estimates for watchlist tickers.

Provides "IV expansion window" detection: earnings 2-10 days out is the sweet
spot for pre-earnings drift plays (buy 2-3 days before, sell day before event).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

log = logging.getLogger(__name__)

_EARNINGS_WINDOW_MIN_DAYS = 2
_EARNINGS_WINDOW_MAX_DAYS = 10


class EarningsAgent:
    def get_earnings_context(self, ticker: str) -> dict[str, Any]:
        """
        Returns earnings context for the given ticker.

        Result keys:
          days_to_earnings  int | None   — calendar days until next earnings
          eps_estimate      float | None — consensus EPS estimate
          revenue_estimate  float | None — consensus revenue estimate (dollars)
          is_within_window  bool         — True if earnings in 2–10 days
          summary           str          — human-readable one-liner for Ollama prompt
        """
        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance not installed — earnings context unavailable")
            return self._empty()

        try:
            cal = yf.Ticker(ticker).calendar
        except Exception as exc:
            log.debug("yfinance calendar error for %s: %s", ticker, exc)
            return self._empty()

        if cal is None or (hasattr(cal, "empty") and cal.empty):
            return self._empty()

        earnings_date = self._parse_earnings_date(cal)
        if earnings_date is None:
            return self._empty()

        today = date.today()
        days = (earnings_date - today).days

        eps_est = self._extract(cal, "EPS Estimate")
        rev_est = self._extract(cal, "Revenue Estimate")

        in_window = _EARNINGS_WINDOW_MIN_DAYS <= days <= _EARNINGS_WINDOW_MAX_DAYS
        summary = self._build_summary(ticker, days, eps_est, rev_est)

        return {
            "days_to_earnings": days,
            "eps_estimate": eps_est,
            "revenue_estimate": rev_est,
            "is_within_window": in_window,
            "summary": summary,
        }

    def _parse_earnings_date(self, cal: Any) -> date | None:
        """Extract earnings date from yfinance calendar (dict or DataFrame)."""
        try:
            if isinstance(cal, dict):
                raw = cal.get("Earnings Date")
                if raw is None:
                    return None
                if hasattr(raw, "__iter__") and not isinstance(raw, str):
                    raw = list(raw)[0]
                if hasattr(raw, "date"):
                    return raw.date()
                return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
            # DataFrame layout
            for col in ("Earnings Date", "Earnings Dates"):
                if col in cal.columns:
                    val = cal[col].iloc[0]
                    if hasattr(val, "date"):
                        return val.date()
                    return datetime.strptime(str(val)[:10], "%Y-%m-%d").date()
        except Exception as exc:
            log.debug("earnings date parse error: %s", exc)
        return None

    def _extract(self, cal: Any, key: str) -> float | None:
        try:
            if isinstance(cal, dict):
                val = cal.get(key)
            else:
                val = cal[key].iloc[0] if key in cal.columns else None
            if val is None:
                return None
            # Unwrap list/Series values (yfinance sometimes returns iterables)
            if hasattr(val, "__iter__") and not isinstance(val, (str, float, int)):
                items = list(val)
                val = items[0] if items else None
            if val is None:
                return None
            return float(val)
        except Exception:
            return None

    def _build_summary(
        self,
        ticker: str,
        days: int,
        eps: float | None,
        rev: float | None,
    ) -> str:
        parts = [f"Earnings in {days} day{'s' if days != 1 else ''}"]
        if eps is not None:
            parts.append(f"EPS est. ${eps:.2f}")
        if rev is not None:
            rev_b = rev / 1e9
            parts.append(f"Rev est. ${rev_b:.1f}B")
        return f"{ticker}: " + ", ".join(parts)

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "days_to_earnings": None,
            "eps_estimate": None,
            "revenue_estimate": None,
            "is_within_window": False,
            "summary": "",
        }
