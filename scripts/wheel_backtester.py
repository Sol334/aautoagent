#!/usr/bin/env python3
"""
galactic-capital/scripts/wheel_backtester.py

Backtest the cash-secured put wheel strategy over historical OHLCV data.
Uses Black-Scholes + 20-period rolling historical volatility as IV proxy.

Usage:
  python scripts/wheel_backtester.py --ticker SPY --years 2
  python scripts/wheel_backtester.py --ticker QQQ --start 2023-01-01 --end 2024-12-31
  python scripts/wheel_backtester.py --ticker SPY --iv 0.18
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAPITAL_ROOT))

try:
    import numpy as np
except ImportError:
    print("numpy required: pip install numpy")
    sys.exit(1)

from scripts.wheel_trader import black_scholes_put, find_delta_strike


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def backtest_wheel(
    ticker: str,
    start_date: str,
    end_date: str,
    capital: float = 200.0,
    delta_target: float = 0.30,
    weeks_forward: int = 1,
    risk_free_rate: float = 0.053,
    iv_override: float | None = None,
) -> dict:
    """
    Simulate a rolling weekly CSP strategy over historical weekly data.

    Returns a results dict with all metrics, or {} on data-fetch failure.
    Never raises.
    """
    try:
        return _run_backtest(
            ticker, start_date, end_date, capital,
            delta_target, weeks_forward, risk_free_rate, iv_override,
        )
    except Exception as exc:
        return {"error": str(exc), "ticker": ticker}


def _run_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    capital: float,
    delta_target: float,
    weeks_forward: int,
    risk_free_rate: float,
    iv_override: float | None,
) -> dict:
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "yfinance not installed", "ticker": ticker}

    hist = yf.Ticker(ticker).history(
        start=start_date, end=end_date, interval="1wk"
    )

    if hist is None or len(hist) < 2:
        return {"error": "insufficient data", "ticker": ticker}

    closes = hist["Close"]

    # 20-period rolling historical volatility (annualised with √52 for weekly data)
    log_returns = np.log(closes / closes.shift(1)).dropna()
    hv_series   = log_returns.rolling(20).std() * np.sqrt(52)

    T = weeks_forward / 52.0

    weekly_results = []
    # Iterate week i, with next week i+1 as the assignment check
    close_vals = closes.values
    close_idx  = closes.index

    for i in range(len(close_vals) - 1):
        S          = float(close_vals[i])
        next_close = float(close_vals[i + 1])
        week_label = close_idx[i].strftime("%Y-%m-%d") if hasattr(close_idx[i], "strftime") else str(close_idx[i])[:10]

        # Resolve IV: override > rolling HV > fallback 0.15
        if iv_override is not None:
            iv = float(iv_override)
        else:
            hv_val = hv_series.iloc[i] if i < len(hv_series) else float("nan")
            iv = float(hv_val) if (not np.isnan(hv_val) and hv_val > 0) else 0.15

        strike  = find_delta_strike(S, T, risk_free_rate, iv, delta_target=delta_target)
        premium = black_scholes_put(S, strike, T, risk_free_rate, iv)

        # Assignment: next week's close fell below the strike
        assigned = next_close < strike
        if assigned:
            # Loss = short-put loss minus premium collected (per share)
            pnl_per_share = premium - (strike - next_close)
        else:
            pnl_per_share = premium

        # Scale to 1 contract (100 shares)
        pnl = round(pnl_per_share * 100, 4)

        weekly_results.append({
            "week":     week_label,
            "price":    round(S, 2),
            "strike":   round(strike, 2),
            "iv":       round(iv, 4),
            "premium":  round(premium, 4),
            "assigned": assigned,
            "pnl":      pnl,
        })

    if not weekly_results:
        return {"error": "no tradeable weeks", "ticker": ticker}

    # ── Aggregate metrics ────────────────────────────────────────────────────

    weeks_simulated        = len(weekly_results)
    total_premium_collected = round(sum(r["premium"] * 100 for r in weekly_results), 2)
    assignments            = sum(1 for r in weekly_results if r["assigned"])
    assignment_rate        = round(assignments / weeks_simulated, 4) if weeks_simulated else 0.0
    total_pnl              = round(sum(r["pnl"] for r in weekly_results), 2)

    # Years spanned
    years = weeks_simulated / 52.0
    annualized_return_pct = round(total_pnl / capital / years * 100, 2) if (capital > 0 and years > 0) else 0.0

    # Sharpe ratio (weekly basis → annualise with √52)
    pnl_arr = np.array([r["pnl"] for r in weekly_results], dtype=float)
    pnl_std = float(np.std(pnl_arr, ddof=1)) if len(pnl_arr) > 1 else 0.0
    pnl_mean = float(np.mean(pnl_arr))
    sharpe_ratio = round((pnl_mean / pnl_std) * np.sqrt(52), 4) if pnl_std > 0 else 0.0

    # Max drawdown on cumulative PnL curve
    cum_pnl    = np.cumsum(pnl_arr)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown_series = running_max - cum_pnl
    max_dd_pct = 0.0
    if capital > 0 and len(running_max) > 0 and running_max.max() > 0:
        max_dd_pct = round(float(drawdown_series.max()) / capital * 100, 2)

    # Buy-and-hold return over same period
    initial_price = float(close_vals[0])
    final_price   = float(close_vals[-1])
    buy_hold_return_pct = round((final_price - initial_price) / initial_price * 100, 2) if initial_price > 0 else 0.0

    return {
        "ticker":                   ticker,
        "start_date":               start_date,
        "end_date":                 end_date,
        "weeks_simulated":          weeks_simulated,
        "total_premium_collected":  total_premium_collected,
        "assignments":              assignments,
        "assignment_rate":          assignment_rate,
        "total_pnl":                total_pnl,
        "annualized_return_pct":    annualized_return_pct,
        "sharpe_ratio":             sharpe_ratio,
        "max_drawdown_pct":         max_dd_pct,
        "buy_hold_return_pct":      buy_hold_return_pct,
        "weekly_results":           weekly_results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _render_report(result: dict) -> str:
    if "error" in result:
        return f"\n  ERROR: {result['error']}\n"

    lines = []
    sep  = "=" * 60
    thin = "─" * 41

    lines.append(sep)
    lines.append(f"  GALACTIC CAPITAL — Wheel Backtest: {result['ticker']}")
    lines.append(f"  {result['start_date']}  →  {result['end_date']}")
    lines.append(sep)
    lines.append("")
    lines.append("  SUMMARY")
    lines.append(f"  {thin}")

    def row(label, val, width=30):
        return f"  {label:<{width}} {val}"

    lines.append(row("Weeks simulated:", str(result["weeks_simulated"])))
    lines.append(row("Total premium collected:", f"${result['total_premium_collected']:.2f}"))
    lines.append(row("Assignments:", f"{result['assignments']} ({result['assignment_rate']*100:.1f}% rate)"))
    lines.append(row("Total P&L:", f"${result['total_pnl']:.2f}"))
    lines.append(row("Annualized return:", f"{result['annualized_return_pct']:.2f}%"))
    lines.append(row("Sharpe ratio:", f"{result['sharpe_ratio']:.3f}"))
    lines.append(row("Max drawdown:", f"{result['max_drawdown_pct']:.2f}%"))
    lines.append(row("Buy-and-hold return:", f"{result['buy_hold_return_pct']:.2f}%"))
    lines.append("")
    lines.append("  LAST 8 WEEKS")
    lines.append(f"  {thin}")
    lines.append(f"  {'Week':<12} {'Price':>8} {'Strike':>8} {'IV':>6} {'Prem':>7} {'Asgn':>5} {'PnL':>8}")
    lines.append(f"  {thin}")

    for w in result["weekly_results"][-8:]:
        asgn = "YES" if w["assigned"] else "no"
        lines.append(
            f"  {w['week']:<12} {w['price']:>8.2f} {w['strike']:>8.2f}"
            f" {w['iv']:>6.3f} {w['premium']:>7.4f} {asgn:>5} {w['pnl']:>8.2f}"
        )

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Galactic Capital — Wheel Strategy Backtester")
    parser.add_argument("--ticker",  default="SPY",  help="Ticker to backtest (default: SPY)")
    parser.add_argument("--years",   type=float,     help="Years of history from today (overrides --start/--end)")
    parser.add_argument("--start",   default="",     help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default="",     help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=200.0, help="Starting capital (default: 200)")
    parser.add_argument("--delta",   type=float, default=0.30,  help="Delta target (default: 0.30)")
    parser.add_argument("--iv",      type=float, default=None,  help="Fixed IV override (e.g. 0.18)")
    args = parser.parse_args()

    end_date   = args.end   or date.today().isoformat()
    if args.years:
        start_date = (date.today() - timedelta(days=int(args.years * 365))).isoformat()
    else:
        start_date = args.start or (date.today() - timedelta(days=365)).isoformat()

    result = backtest_wheel(
        ticker        = args.ticker.upper(),
        start_date    = start_date,
        end_date      = end_date,
        capital       = args.capital,
        delta_target  = args.delta,
        iv_override   = args.iv,
    )

    print(_render_report(result))


if __name__ == "__main__":
    main()
