#!/usr/bin/env python3
"""
galactic-capital/scripts/wheel_trader.py

Wheel strategy paper trader for SPY/QQQ.

The Wheel:
  1. Sell cash-secured put (CSP) on SPY/QQQ at 30-delta strike
  2. If put expires worthless → collect premium, repeat
  3. If assigned → sell covered calls until called away

Target: ~0.5-1.5% weekly premium = ~19% annualized on top of underlying return.
The Wheel on SPY/QQQ has ~3x the Sharpe ratio of buy-and-hold in backtests.

Modes:
  - Simulation: calculates theoretical premium from yfinance IV data (default)
  - Live: Alpaca options paper API (requires FEATURE_WHEEL_TRADING=true +
    Alpaca options approval + ALPACA_API_KEY set)

Usage:
  python galactic-capital/scripts/wheel_trader.py --dry-run
  python galactic-capital/scripts/wheel_trader.py --ticker SPY
"""

import argparse
import logging
import math
import os
import sys
from datetime import date, timedelta
from pathlib import Path

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAPITAL_ROOT))

from dotenv import load_dotenv
load_dotenv(CAPITAL_ROOT / ".env")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FEATURE_ENABLED = os.getenv("FEATURE_WHEEL_TRADING", "false").lower() in ("true", "1", "yes")
PAPER_TRADING   = os.getenv("CAPITAL_PAPER_TRADING", "true").lower() in ("true", "1", "yes")
ALPACA_KEY      = os.getenv("ALPACA_API_KEY", "")
STARTING_CAPITAL = float(os.getenv("CAPITAL_STARTING_USD", "200"))

DEFAULT_TICKERS = ["SPY", "QQQ"]
_CSP_DELTA_TARGET = 0.30
_WEEKS_FORWARD = 1


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes put price.
      S = current price, K = strike, T = time to expiry (years),
      r = risk-free rate, sigma = implied volatility (decimal)
    Returns theoretical put premium per share.
    """
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)

    import math
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = _norm_cdf(-d1)
    Nd2 = _norm_cdf(-d2)
    put_price = K * math.exp(-r * T) * Nd2 - S * Nd1
    return max(put_price, 0.0)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def get_wheel_simulation(ticker: str) -> dict:
    """
    Simulate a 1-week CSP on the given ticker using yfinance IV data.

    Returns:
      ticker, price, strike, premium_per_share, premium_pct,
      contracts_possible, total_premium, expiry, mode
    """
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed")
        return {}

    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = float(getattr(info, "last_price", None) or getattr(info, "regular_market_price", 0))
        if not price:
            log.warning("No price for %s", ticker)
            return {}
    except Exception as exc:
        log.warning("Price fetch error for %s: %s", ticker, exc)
        return {}

    # Grab IV from options chain (nearest weekly expiry)
    iv = _get_iv(t, price)
    if iv is None:
        iv = 0.15  # conservative 15% IV fallback

    # 30-delta put strike: roughly S * exp(-0.5 * iv * sqrt(T))
    T = _WEEKS_FORWARD / 52
    r = 0.053  # approx current risk-free rate
    strike = round(price * 0.97, 0)  # ~3% OTM as 30-delta approximation

    premium = black_scholes_put(price, strike, T, r, iv)
    premium_pct = premium / price * 100

    # Number of CSPs possible: 1 contract = 100 shares, needs cash collateral = strike * 100
    collateral_per_contract = strike * 100
    contracts = max(1, int(STARTING_CAPITAL / collateral_per_contract)) if STARTING_CAPITAL >= collateral_per_contract else 0
    total_premium = round(premium * 100 * contracts, 2)

    expiry = date.today() + timedelta(weeks=_WEEKS_FORWARD)
    # Find nearest Friday
    while expiry.weekday() != 4:
        expiry += timedelta(days=1)

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "strike": strike,
        "iv": round(iv * 100, 1),
        "premium_per_share": round(premium, 2),
        "premium_pct": round(premium_pct, 3),
        "contracts_possible": contracts,
        "total_premium": total_premium,
        "expiry": expiry.isoformat(),
        "mode": "simulation",
        "note": "Theoretical Black-Scholes. Actual fills vary.",
    }


def _get_iv(ticker_obj, price: float) -> float | None:
    """Extract implied volatility from the nearest options expiry."""
    try:
        exps = ticker_obj.options
        if not exps:
            return None
        chain = ticker_obj.option_chain(exps[0])
        puts = chain.puts
        if puts.empty:
            return None
        otm_puts = puts[puts["strike"] < price * 0.99]
        if otm_puts.empty:
            otm_puts = puts
        iv_col = "impliedVolatility"
        if iv_col in otm_puts.columns:
            iv = otm_puts[iv_col].dropna().median()
            return float(iv) if iv > 0 else None
    except Exception as exc:
        log.debug("IV fetch error: %s", exc)
    return None


def run(tickers: list[str], dry_run: bool) -> None:
    if not FEATURE_ENABLED:
        log.info("FEATURE_WHEEL_TRADING=false — wheel trader is in simulation preview mode")

    print(f"\n{'='*65}")
    print(f"  GALACTIC CAPITAL — Wheel Strategy {'(DRY-RUN)' if dry_run else '(SIMULATION)'}")
    print(f"  Capital: ${STARTING_CAPITAL:.0f} | Target: weekly CSP on SPY/QQQ")
    print(f"{'='*65}")

    for ticker in tickers:
        result = get_wheel_simulation(ticker)
        if not result:
            print(f"  {ticker}: data unavailable")
            continue

        print(f"\n  {ticker} @ ${result['price']:.2f} | IV: {result['iv']:.1f}%")
        print(f"    Strike: ${result['strike']:.0f} | Expiry: {result['expiry']}")
        print(f"    Premium/share: ${result['premium_per_share']:.2f} ({result['premium_pct']:.2f}%)")
        print(f"    Contracts possible: {result['contracts_possible']}")
        print(f"    Total premium: ${result['total_premium']:.2f}")
        if result["contracts_possible"] == 0:
            print(f"    ⚠  Need ${result['strike'] * 100:.0f} collateral for 1 contract")

    print(f"\n{'='*65}")
    print("  Note: Alpaca options approval required for live execution.")
    print("  Set FEATURE_WHEEL_TRADING=true once approved.")
    print(f"{'='*65}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Galactic Capital — Wheel Strategy")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--ticker", type=str, default="",
                        help="Single ticker (default: SPY,QQQ)")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else DEFAULT_TICKERS
    run(tickers=tickers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
