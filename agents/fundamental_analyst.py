"""
galactic-capital/agents/fundamental_analyst.py

Fundamental analysis agent. Pulls key metrics from yfinance and returns a
structured signal for integration into the 5-signal market analyst pipeline.
No Ollama call — logic is deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

_PE_BULLISH_MAX = 25.0
_PE_BEARISH_MIN = 50.0
_REV_GROWTH_BULLISH_MIN = 0.10
_REV_GROWTH_BEARISH_MAX = -0.05
_DE_BULLISH_MAX = 1.0
_DE_BEARISH_MIN = 3.0


@dataclass
class FundamentalSignal:
    ticker: str
    signal: str
    confidence: float
    pe_ratio: float | None
    pb_ratio: float | None
    eps_ttm: float | None
    revenue_growth_yoy: float | None
    debt_to_equity: float | None
    summary: str


class FundamentalAnalyst:
    def analyze(self, ticker: str) -> FundamentalSignal:
        """Return a FundamentalSignal for ticker. Never raises."""
        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance not installed — fundamental analysis unavailable")
            return self._neutral(ticker, all_none=True)

        try:
            info = yf.Ticker(ticker).info
        except Exception as exc:
            log.debug("yfinance info error for %s: %s", ticker, exc)
            return self._neutral(ticker, all_none=True)

        pe = self._float(info.get("trailingPE"))
        pb = self._float(info.get("priceToBook"))
        eps = self._float(info.get("trailingEps"))
        rev_growth = self._float(info.get("revenueGrowth"))

        raw_de = self._float(info.get("debtToEquity"))
        # yfinance reports debtToEquity as ratio × 100
        de = raw_de / 100.0 if raw_de is not None else None

        signal, confidence = self._classify(pe, rev_growth, de)
        summary = self._build_summary(ticker, signal, pe, rev_growth, de)

        return FundamentalSignal(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            pe_ratio=pe,
            pb_ratio=pb,
            eps_ttm=eps,
            revenue_growth_yoy=rev_growth,
            debt_to_equity=de,
            summary=summary,
        )

    def _classify(
        self,
        pe: float | None,
        rev_growth: float | None,
        de: float | None,
    ) -> tuple[str, float]:
        bullish_hits = 0
        bearish_hits = 0

        if pe is not None:
            if pe < _PE_BULLISH_MAX:
                bullish_hits += 1
            elif pe > _PE_BEARISH_MIN:
                bearish_hits += 1

        if rev_growth is not None:
            if rev_growth > _REV_GROWTH_BULLISH_MIN:
                bullish_hits += 1
            elif rev_growth < _REV_GROWTH_BEARISH_MAX:
                bearish_hits += 1

        if de is not None:
            if de < _DE_BULLISH_MAX:
                bullish_hits += 1
            elif de > _DE_BEARISH_MIN:
                bearish_hits += 1

        total_available = sum(x is not None for x in (pe, rev_growth, de))
        if total_available == 0:
            return "NEUTRAL", 0.0

        if bullish_hits == total_available and bullish_hits > 0:
            # All available metrics are bullish
            confidence = min(0.6 + 0.1 * (bullish_hits - 1), 0.9)
            return "BULLISH", confidence

        if bearish_hits > 0:
            # Any bearish trigger fires BEARISH
            confidence = min(0.5 + 0.1 * (bearish_hits - 1), 0.8)
            return "BEARISH", confidence

        return "NEUTRAL", 0.4

    def _build_summary(
        self,
        ticker: str,
        signal: str,
        pe: float | None,
        rev_growth: float | None,
        de: float | None,
    ) -> str:
        parts: list[str] = []
        if pe is not None:
            parts.append(f"PE={pe:.1f}")
        if rev_growth is not None:
            parts.append(f"RevGrowth={rev_growth*100:.1f}%")
        if de is not None:
            parts.append(f"D/E={de:.2f}")
        metrics = ", ".join(parts) if parts else "insufficient data"
        return f"{ticker} fundamentals {signal.lower()}: {metrics}."

    @staticmethod
    def _float(val: object) -> float | None:
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _neutral(ticker: str, *, all_none: bool = False) -> FundamentalSignal:
        return FundamentalSignal(
            ticker=ticker,
            signal="NEUTRAL",
            confidence=0.0 if all_none else 0.4,
            pe_ratio=None,
            pb_ratio=None,
            eps_ttm=None,
            revenue_growth_yoy=None,
            debt_to_equity=None,
            summary=f"{ticker} fundamentals neutral: insufficient data.",
        )
