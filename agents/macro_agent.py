"""
galactic-capital/agents/macro_agent.py

Macro regime filter using FRED yield curve data and VIX.

Regime logic:
  risk_off  — 10Y-2Y spread inverted (< 0) AND VIX > 25
  neutral   — either condition met but not both
  risk_on   — neither condition met

In risk_off regime, paper_trader.py tightens the max drawdown floor by 50%
to protect the $200 starting capital.

FRED API key: free at https://fred.stlouisfed.org/docs/api/api_key.html
Set: FRED_API_KEY=<your_key> in .env
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

_VIX_RISK_OFF_THRESHOLD = 25.0
_YIELD_CURVE_SERIES = {
    "10y": "DGS10",
    "2y": "DGS2",
}


class MacroAgent:
    def __init__(self) -> None:
        self._fred_key = os.getenv("FRED_API_KEY", "")
        self._cache: dict[str, Any] = {}

    def get_regime(self) -> str:
        """
        Returns "risk_on", "neutral", or "risk_off".

        risk_off = yield curve inverted (spread < 0) AND VIX > threshold.
        Falls back to "neutral" if any data is unavailable.
        """
        spread = self._get_yield_curve_spread()
        vix = self._get_vix()

        if spread is None or vix is None:
            log.debug("macro data unavailable — defaulting to neutral")
            return "neutral"

        inverted = spread < 0
        high_vix = vix > _VIX_RISK_OFF_THRESHOLD

        if inverted and high_vix:
            return "risk_off"
        if inverted or high_vix:
            return "neutral"
        return "risk_on"

    def get_macro_context(self) -> str:
        """One-sentence summary for Ollama prompt injection."""
        spread = self._get_yield_curve_spread()
        vix = self._get_vix()
        regime = self.get_regime()

        parts = [f"Macro regime: {regime}"]
        if spread is not None:
            direction = "inverted" if spread < 0 else "normal"
            parts.append(f"yield curve {direction} ({spread:+.2f}%)")
        if vix is not None:
            parts.append(f"VIX {vix:.1f}")
        return ", ".join(parts)

    def _get_yield_curve_spread(self) -> float | None:
        if "spread" in self._cache:
            return self._cache["spread"]

        rate_10y = self._fred_latest(_YIELD_CURVE_SERIES["10y"])
        rate_2y = self._fred_latest(_YIELD_CURVE_SERIES["2y"])
        if rate_10y is None or rate_2y is None:
            return None
        spread = rate_10y - rate_2y
        self._cache["spread"] = spread
        return spread

    def _get_vix(self) -> float | None:
        if "vix" in self._cache:
            return self._cache["vix"]

        try:
            import yfinance as yf
            ticker = yf.Ticker("^VIX")
            info = ticker.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
            if price is not None:
                self._cache["vix"] = float(price)
                return float(price)
        except Exception as exc:
            log.debug("VIX fetch error: %s", exc)
        return None

    def _fred_latest(self, series_id: str) -> float | None:
        if not self._fred_key or self._fred_key == "CHANGE_ME":
            return self._fred_latest_no_key(series_id)
        try:
            from fredapi import Fred
            fred = Fred(api_key=self._fred_key)
            series = fred.get_series(series_id, observation_start="2020-01-01")
            val = series.dropna().iloc[-1]
            return float(val)
        except Exception as exc:
            log.debug("FRED API error for %s: %s", series_id, exc)
            return None

    def _fred_latest_no_key(self, series_id: str) -> float | None:
        """Fallback: fetch FRED via public JSON endpoint (no key required for recent data)."""
        try:
            import urllib.request, json as _json
            url = (
                f"https://fred.stlouisfed.org/graph/fredgraph.json"
                f"?id={series_id}&vintage_date="
            )
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = _json.loads(resp.read())
            observations = data.get("observations", [])
            if not observations:
                return None
            for obs in reversed(observations):
                if obs.get("value") not in (".", "", None):
                    return float(obs["value"])
        except Exception as exc:
            log.debug("FRED public JSON error for %s: %s", series_id, exc)
        return None
