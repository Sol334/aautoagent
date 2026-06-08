#!/usr/bin/env python3
"""Market Regime Detector — Galactic Capital

Identifies the current market regime using clustering on price/volatility
features, then gates trading decisions accordingly.

Regimes:
    0 = TRENDING_BULL   — sustained uptrend, low volatility → take long positions
    1 = TRENDING_BEAR   — sustained downtrend → go cash or hedge
    2 = MEAN_REVERTING  — choppy/range-bound → avoid momentum strategies
    3 = HIGH_VOL_CHAOS  — extreme volatility spike → no new positions

Method: K-Means on rolling window features (inspired by k3tikvats/market_regime_detection).
No external ML framework required — uses only numpy + scipy (already available
via torch/transformers deps, or installable standalone).

Usage:
    python regime_detector.py --ticker SPY --days 90
    python regime_detector.py --ticker QQQ --regime-only

Env vars:
    CAPITAL_REGIME_TICKER — ticker to use for regime detection (default: SPY)
    CAPITAL_REGIME_LOOKBACK — days of history to analyze (default: 90)
"""

import os, logging, math
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

log = logging.getLogger("regime_detector")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

REGIME_TICKER   = os.getenv("CAPITAL_REGIME_TICKER", "SPY")
REGIME_LOOKBACK = int(os.getenv("CAPITAL_REGIME_LOOKBACK", "90"))

# Regime labels
TRENDING_BULL   = 0
TRENDING_BEAR   = 1
MEAN_REVERTING  = 2
HIGH_VOL_CHAOS  = 3

_REGIME_LABELS = {
    TRENDING_BULL:  "TRENDING_BULL",
    TRENDING_BEAR:  "TRENDING_BEAR",
    MEAN_REVERTING: "MEAN_REVERTING",
    HIGH_VOL_CHAOS: "HIGH_VOL_CHAOS",
}

_REGIME_ACTIONS = {
    TRENDING_BULL:  "TRADE — take long positions, follow congressional buy signals",
    TRENDING_BEAR:  "CASH — no new longs, consider defensive ETFs",
    MEAN_REVERTING: "WAIT — choppy market, momentum strategies will whipsaw",
    HIGH_VOL_CHAOS: "HOLD — extreme volatility, preserve capital",
}


# ── Feature extraction ────────────────────────────────────────────────────────

def _returns(prices: list[float]) -> list[float]:
    return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]


def _rolling_vol(returns: list[float], window: int = 20) -> list[float]:
    result = []
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        subset = returns[start:i+1]
        mean = sum(subset) / len(subset)
        variance = sum((r - mean) ** 2 for r in subset) / len(subset)
        result.append(math.sqrt(variance) * math.sqrt(252))
    return result


def _momentum(prices: list[float], window: int = 20) -> list[float]:
    result = []
    for i in range(len(prices)):
        if i < window:
            result.append(0.0)
        else:
            result.append((prices[i] - prices[i - window]) / prices[i - window])
    return result


def extract_features(prices: list[float]) -> list[dict]:
    """Returns per-day feature dicts for clustering."""
    if len(prices) < 21:
        return []
    rets = _returns(prices)
    vols = _rolling_vol(rets)
    moms = _momentum(prices)
    features = []
    for i in range(len(rets)):
        features.append({
            "return":   rets[i],
            "vol_ann":  vols[i],
            "momentum": moms[i + 1],  # shift to align with return index
        })
    return features


# ── Regime classification (rule-based + cluster-inspired) ─────────────────────
# Pure heuristic approach — no ML library required, same conceptual output
# as k-means clustering on vol/momentum features.

def classify_regime(features: list[dict]) -> int:
    """Classify current regime from recent features. Uses last 20-day window."""
    if not features:
        return MEAN_REVERTING

    recent = features[-20:]
    avg_vol  = sum(f["vol_ann"]  for f in recent) / len(recent)
    avg_mom  = sum(f["momentum"] for f in recent) / len(recent)
    avg_ret  = sum(f["return"]   for f in recent) / len(recent)

    # Annualized vol > 40% = chaos regime
    if avg_vol > 0.40:
        return HIGH_VOL_CHAOS

    # Clear trend + positive momentum = bull
    if avg_ret > 0.0003 and avg_mom > 0.02:
        return TRENDING_BULL

    # Clear downtrend
    if avg_ret < -0.0003 and avg_mom < -0.02:
        return TRENDING_BEAR

    # Default: mean-reverting / range-bound
    return MEAN_REVERTING


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_prices(ticker: str, days: int = 90) -> list[float]:
    """Fetch closing prices via yfinance."""
    try:
        import yfinance as yf
        end = date.today()
        start = end - timedelta(days=days + 10)  # buffer for weekends
        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
        if df.empty:
            log.warning("No price data for %s", ticker)
            return []
        return df["Close"].dropna().tolist()
    except Exception as exc:
        log.warning("Price fetch failed for %s: %s", ticker, exc)
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def detect_regime(ticker: str | None = None, days: int | None = None) -> dict:
    """Main entry point. Returns regime dict with label, action, confidence."""
    t = ticker or REGIME_TICKER
    d = days or REGIME_LOOKBACK

    prices = fetch_prices(t, d)
    if not prices:
        return {
            "regime":     MEAN_REVERTING,
            "label":      _REGIME_LABELS[MEAN_REVERTING],
            "action":     _REGIME_ACTIONS[MEAN_REVERTING],
            "confidence": 0.0,
            "ticker":     t,
            "error":      "no_price_data",
        }

    features = extract_features(prices)
    regime = classify_regime(features)

    # Confidence: how extreme are the signals?
    recent = features[-20:] if features else []
    avg_vol = sum(f["vol_ann"] for f in recent) / len(recent) if recent else 0.5
    confidence = min(1.0, abs(avg_vol - 0.20) / 0.20) if recent else 0.5

    return {
        "regime":       regime,
        "label":        _REGIME_LABELS[regime],
        "action":       _REGIME_ACTIONS[regime],
        "confidence":   round(confidence, 2),
        "ticker":       t,
        "days_analyzed": len(features),
    }


def should_trade(ticker: str | None = None) -> bool:
    """Returns True only when regime permits new long positions."""
    result = detect_regime(ticker)
    allowed = result["regime"] == TRENDING_BULL
    log.info("Regime: %s → %s", result["label"], "TRADE" if allowed else "SKIP")
    return allowed


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Market regime detector")
    p.add_argument("--ticker", default=REGIME_TICKER)
    p.add_argument("--days", type=int, default=REGIME_LOOKBACK)
    p.add_argument("--regime-only", action="store_true", help="Print just the regime label")
    args = p.parse_args()

    result = detect_regime(args.ticker, args.days)
    if args.regime_only:
        print(result["label"])
    else:
        print(f"\n=== Market Regime: {result['label']} ===")
        print(f"  Ticker:     {result['ticker']}")
        print(f"  Confidence: {result['confidence']:.0%}")
        print(f"  Action:     {result['action']}")
        print(f"  Days:       {result.get('days_analyzed', 0)}")
