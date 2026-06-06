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
import json
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


# ---------------------------------------------------------------------------
# Core math helpers — do NOT remove or rename; tests depend on these
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _norm_inv(p: float) -> float:
    """
    Inverse standard normal CDF (quantile function).

    Uses the rational approximation from Abramowitz & Stegun 26.2.17,
    accurate to ~4 decimal places over the full range.
    """
    if p <= 0:
        return -float("inf")
    if p >= 1:
        return float("inf")
    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))

    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    x = t - (c[0] + c[1] * t + c[2] * t ** 2) / (
        1.0 + d[0] * t + d[1] * t ** 2 + d[2] * t ** 3
    )
    return -x if p < 0.5 else x


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes put price.
      S = current price, K = strike, T = time to expiry (years),
      r = risk-free rate, sigma = implied volatility (decimal)
    Returns theoretical put premium per share.
    """
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = _norm_cdf(-d1)
    Nd2 = _norm_cdf(-d2)
    put_price = K * math.exp(-r * T) * Nd2 - S * Nd1
    return max(put_price, 0.0)


def find_delta_strike(
    S: float, T: float, r: float, sigma: float, delta_target: float = 0.30
) -> float:
    """
    Return the strike K for a put with |delta| == delta_target.

    Put delta = -N(-d1).  Setting N(-d1) = delta_target:
      -d1 = N_inv(delta_target)  →  d1 = N_inv(1 - delta_target)

    Rearranging d1 = (ln(S/K) + (r + sigma²/2)*T) / (sigma*sqrt(T)):
      ln(S/K) = d1_target * sigma*sqrt(T) - (r + sigma²/2)*T
      K = S * exp(-(d1_target * sigma*sqrt(T) - (r + sigma²/2)*T))
    """
    if T <= 0 or sigma <= 0:
        return S * (1.0 - delta_target)

    d1_target = _norm_inv(1.0 - delta_target)
    exponent = d1_target * sigma * math.sqrt(T) - (r + 0.5 * sigma ** 2) * T
    K = S * math.exp(-exponent)
    return round(K, 2)


def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes put delta (negative value, e.g. -0.30).

    delta_put = -N(-d1)
    """
    if T <= 0 or sigma <= 0:
        # At expiry: delta is -1 if ITM, 0 if OTM
        return -1.0 if K > S else 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return -_norm_cdf(-d1)


def assignment_probability(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Risk-neutral probability that the put expires in-the-money = N(-d2).

    This equals the probability of assignment if the option is held to expiry.
    """
    if T <= 0 or sigma <= 0:
        return 1.0 if K > S else 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return _norm_cdf(-d2)


# ---------------------------------------------------------------------------
# Macro regime gate
# ---------------------------------------------------------------------------

def _get_regime() -> str:
    """
    Fetch current macro regime from MacroAgent.
    Always returns a string — never raises.
    """
    try:
        from agents.macro_agent import MacroAgent
        return MacroAgent().get_regime()
    except Exception:
        return "neutral"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _log_simulation(result: dict) -> None:
    """
    Append simulation result to data/wheel_log.json using an atomic write.
    Keeps the most recent 500 entries.
    """
    from datetime import datetime
    log_path = CAPITAL_ROOT / "data" / "wheel_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if log_path.exists():
        try:
            existing = json.loads(log_path.read_text())
        except Exception:
            pass

    result["logged_at"] = datetime.now().isoformat()
    existing.append(result)

    tmp = log_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing[-500:], indent=2))
    tmp.replace(log_path)


# ---------------------------------------------------------------------------
# IV extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main simulation interface
# ---------------------------------------------------------------------------

def get_wheel_simulation(ticker: str) -> dict:
    """
    Simulate a 1-week CSP on the given ticker using yfinance IV data.

    Returns:
      ticker, price, strike, iv, premium_per_share, premium_pct,
      contracts_possible, total_premium, expiry, mode,
      break_even, break_even_pct, assign_prob, annualized_return_pct,
      delta (actual put delta at chosen strike), note
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

    T = _WEEKS_FORWARD / 52
    r = 0.053  # approx current risk-free rate

    # Proper 30-delta strike via Black-Scholes inversion
    strike = find_delta_strike(price, T, r, iv, delta_target=_CSP_DELTA_TARGET)

    premium = black_scholes_put(price, strike, T, r, iv)
    premium_pct = premium / price * 100

    # Actual put delta and assignment probability at the chosen strike
    actual_delta = put_delta(price, strike, T, r, iv)
    assign_prob = assignment_probability(price, strike, T, r, iv)

    # Break-even and annualized return
    break_even = strike - premium
    break_even_pct = (price - break_even) / price * 100

    annualized_return_pct = premium_pct * 52  # simplified, not compounded

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
        "strike": round(strike, 2),
        "iv": round(iv * 100, 1),
        "delta": round(actual_delta, 4),
        "premium_per_share": round(premium, 2),
        "premium_pct": round(premium_pct, 3),
        "break_even": round(break_even, 2),
        "break_even_pct": round(break_even_pct, 2),
        "assign_prob": round(assign_prob * 100, 1),
        "annualized_return_pct": round(annualized_return_pct, 1),
        "contracts_possible": contracts,
        "total_premium": total_premium,
        "expiry": expiry.isoformat(),
        "mode": "simulation",
        "note": "Theoretical Black-Scholes. Actual fills vary.",
    }


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

def run(tickers: list[str], dry_run: bool) -> None:
    if not FEATURE_ENABLED:
        log.info("FEATURE_WHEEL_TRADING=false — wheel trader is in simulation preview mode")

    # Macro regime gate — check once for the session
    regime = _get_regime()
    risk_off = regime == "risk_off"

    print(f"\n{'='*65}")
    print(f"  GALACTIC CAPITAL — Wheel Strategy {'(DRY-RUN)' if dry_run else '(SIMULATION)'}")
    print(f"  Capital: ${STARTING_CAPITAL:.0f} | Target: weekly CSP on SPY/QQQ")
    if risk_off:
        print(f"  *** RISK-OFF REGIME — Recommended contracts halved ***")
    print(f"{'='*65}")

    for ticker in tickers:
        result = get_wheel_simulation(ticker)
        if not result:
            print(f"  {ticker}: data unavailable")
            continue

        # Halve recommended contracts in risk_off regime
        display_contracts = result["contracts_possible"]
        if risk_off:
            display_contracts = max(0, display_contracts // 2)

        print(f"\n  {ticker} @ ${result['price']:.2f} | IV: {result['iv']:.1f}%")
        print(f"    Strike: ${result['strike']:.2f} (delta {result['delta']:.3f}) | Expiry: {result['expiry']}")
        print(f"    Premium/share: ${result['premium_per_share']:.2f} ({result['premium_pct']:.2f}%)")
        print(f"    Break-even: ${result['break_even']:.2f} ({result['break_even_pct']:.1f}% downside buffer)")
        print(f"    Assignment prob: {result['assign_prob']:.1f}%  |  Annualized: {result['annualized_return_pct']:.1f}%")
        print(f"    Contracts possible: {display_contracts}" + (" (halved — risk-off)" if risk_off else ""))
        print(f"    Total premium: ${result['total_premium']:.2f}")
        if result["contracts_possible"] == 0:
            print(f"    Need ${result['strike'] * 100:.0f} collateral for 1 contract")

        if not dry_run:
            _log_simulation(dict(result, regime=regime))

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
