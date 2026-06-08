"""Weighted multi-signal consensus aggregator for Galactic Capital."""

import logging
from dataclasses import dataclass, field
from datetime import date

log = logging.getLogger(__name__)

_SENIOR_TITLES = {"ceo", "cfo", "president", "coo"}


@dataclass
class ConsensusSignal:
    ticker: str
    weighted_score: float
    conviction: str
    bullish_factors: list[str] = field(default_factory=list)
    bearish_factors: list[str] = field(default_factory=list)
    summary: str = ""


class ConsensusEngine:
    _WEIGHTS = {
        "insider":      0.25,
        "political":    0.20,
        "options_flow": 0.20,
        "fundamentals": 0.15,
        "sentiment":    0.10,
        "forecast":     0.05,
        "earnings":     0.05,
    }

    def score(
        self,
        ticker: str,
        *,
        sentiment: float = 0.0,
        trend: str = "flat",
        earnings_ctx: dict | None = None,
        options_flow: str = "normal",
        fundamental: object | None = None,
        insider_trades: list | None = None,
        political_signal: str = "",
    ) -> ConsensusSignal:
        try:
            return self._score(
                ticker,
                sentiment=sentiment,
                trend=trend,
                earnings_ctx=earnings_ctx,
                options_flow=options_flow,
                fundamental=fundamental,
                insider_trades=insider_trades,
                political_signal=political_signal,
            )
        except Exception as exc:
            log.warning("ConsensusEngine.score failed for %s: %s", ticker, exc)
            return ConsensusSignal(
                ticker=ticker,
                weighted_score=0.0,
                conviction="low",
                bullish_factors=[],
                bearish_factors=[],
                summary=f"{ticker}: low conviction neutral — error during scoring (score +0.00)",
            )

    def _score(
        self,
        ticker: str,
        *,
        sentiment: float,
        trend: str,
        earnings_ctx: dict | None,
        options_flow: str,
        fundamental: object | None,
        insider_trades: list | None,
        political_signal: str,
    ) -> ConsensusSignal:
        components: dict[str, float] = {}
        bullish_factors: list[str] = []
        bearish_factors: list[str] = []

        # ── insider ───────────────────────────────────────────────────────────
        insider_score = self._score_insider(insider_trades, bullish_factors, bearish_factors)
        components["insider"] = insider_score

        # ── political ─────────────────────────────────────────────────────────
        if "bullish" in political_signal.lower():
            pol_score = 1.0
            bullish_factors.append("congressional_bullish")
        elif "bearish" in political_signal.lower():
            pol_score = -1.0
            bearish_factors.append("congressional_bearish")
        else:
            pol_score = 0.0
        components["political"] = pol_score

        # ── options_flow ──────────────────────────────────────────────────────
        if options_flow == "unusual_calls":
            flow_score = 1.0
            bullish_factors.append("unusual_calls")
        elif options_flow == "unusual_puts":
            flow_score = -1.0
            bearish_factors.append("unusual_puts")
        else:
            flow_score = 0.0
        components["options_flow"] = flow_score

        # ── fundamentals ──────────────────────────────────────────────────────
        if fundamental is None:
            fund_score = 0.0
        else:
            sig = getattr(fundamental, "signal", "NEUTRAL").upper()
            conf = float(getattr(fundamental, "confidence", 0.0))
            if sig == "BULLISH":
                fund_score = 1.0 * conf
                label = getattr(fundamental, "summary", f"fundamentals_bullish (conf={conf:.2f})")
                bullish_factors.append(f"fundamentals_bullish ({label})")
            elif sig == "BEARISH":
                fund_score = -1.0 * conf
                label = getattr(fundamental, "summary", f"fundamentals_bearish (conf={conf:.2f})")
                bearish_factors.append(f"fundamentals_bearish ({label})")
            else:
                fund_score = 0.0
        components["fundamentals"] = fund_score

        # ── sentiment ─────────────────────────────────────────────────────────
        components["sentiment"] = float(sentiment)
        if sentiment > 0:
            bullish_factors.append(f"sentiment {sentiment:+.2f}")
        elif sentiment < 0:
            bearish_factors.append(f"sentiment {sentiment:+.2f}")

        # ── forecast ──────────────────────────────────────────────────────────
        trend_lower = trend.lower()
        if "bullish" in trend_lower or trend_lower == "up":
            forecast_score = 1.0
            bullish_factors.append("forecast_bullish")
        elif "bearish" in trend_lower or trend_lower == "down":
            forecast_score = -1.0
            bearish_factors.append("forecast_bearish")
        else:
            forecast_score = 0.0
        components["forecast"] = forecast_score

        # ── earnings ─────────────────────────────────────────────────────────
        # Earnings window is a risk flag, not a directional signal.
        components["earnings"] = 0.0

        # ── weighted sum ──────────────────────────────────────────────────────
        raw = sum(components[k] * self._WEIGHTS[k] for k in self._WEIGHTS)
        weighted_score = max(-1.0, min(1.0, raw))

        # ── conviction ────────────────────────────────────────────────────────
        abs_score = abs(weighted_score)
        if abs_score > 0.5:
            conviction = "high"
        elif abs_score > 0.25:
            conviction = "medium"
        else:
            conviction = "low"

        if weighted_score > 0:
            direction = "bullish"
        elif weighted_score < 0:
            direction = "bearish"
        else:
            direction = "neutral"

        summary = (
            f"{ticker}: {conviction} conviction {direction} — "
            f"{len(bullish_factors)} bullish, {len(bearish_factors)} bearish signals "
            f"(score {weighted_score:+.2f})"
        )

        return ConsensusSignal(
            ticker=ticker,
            weighted_score=weighted_score,
            conviction=conviction,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            summary=summary,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _score_insider(
        self,
        insider_trades: list | None,
        bullish_factors: list[str],
        bearish_factors: list[str],
    ) -> float:
        if not insider_trades:
            return 0.0

        today = date.today()
        scores: list[float] = []

        for trade in insider_trades:
            direction = 1.0 if getattr(trade, "is_buy", False) else -1.0

            # Recency weight from transaction_date
            tx_date_str = getattr(trade, "transaction_date", "") or ""
            recency_weight = self._recency_weight(tx_date_str, today)

            # Title weight
            title = getattr(trade, "title", "") or ""
            title_weight = self._title_weight(title)

            scores.append(direction * recency_weight * title_weight)

            # Build human-readable factor label
            value_usd = getattr(trade, "value_usd", None)
            insider_name = getattr(trade, "insider_name", "")
            age_days = self._days_ago(tx_date_str, today)
            age_str = f"{age_days} day{'s' if age_days != 1 else ''} ago" if age_days is not None else "unknown date"
            value_str = f" ${value_usd:,.0f}" if value_usd is not None else ""
            label_title = title if title else "Unknown"

            if direction > 0:
                bullish_factors.append(
                    f"insider_buy: {label_title}{value_str} ({age_str})"
                )
            else:
                bearish_factors.append(
                    f"insider_sell: {label_title}"
                )

        if not scores:
            return 0.0

        raw = sum(scores) / len(scores)
        return max(-1.0, min(1.0, raw))

    @staticmethod
    def _recency_weight(tx_date_str: str, today: date) -> float:
        days = ConsensusEngine._days_ago(tx_date_str, today)
        if days is None:
            return 0.5
        if days <= 7:
            return 1.0
        if days <= 14:
            return 0.75
        if days <= 30:
            return 0.5
        return 0.25

    @staticmethod
    def _title_weight(title: str) -> float:
        normalized = title.lower()
        if any(t in normalized for t in _SENIOR_TITLES):
            return 1.0
        if "director" in normalized:
            return 0.75
        if "10%" in normalized or "10 %" in normalized:
            return 0.5
        return 0.25

    @staticmethod
    def _days_ago(tx_date_str: str, today: date) -> int | None:
        if not tx_date_str:
            return None
        try:
            tx = date.fromisoformat(tx_date_str[:10])
            return max(0, (today - tx).days)
        except (ValueError, TypeError):
            return None
