#!/usr/bin/env python3
"""
galactic-capital/scripts/portfolio_dashboard.py

CLI dashboard that reads existing trade logs and renders a P&L summary.

Usage:
  python scripts/portfolio_dashboard.py          # full dashboard
  python scripts/portfolio_dashboard.py --json   # JSON output for n8n
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAPITAL_ROOT))

PAPER_TRADES_PATH    = CAPITAL_ROOT / "data" / "paper_trades_dryrun.json"
WHEEL_LOG_PATH       = CAPITAL_ROOT / "data" / "wheel_log.json"
CRYPTO_TRADES_PATH   = CAPITAL_ROOT / "data" / "crypto_paper_trades.json"
PORTFOLIO_STATE_PATH = CAPITAL_ROOT / "data" / "portfolio_state.json"

_MAX_DRAWDOWN_PCT = float(os.getenv("CAPITAL_MAX_DRAWDOWN_PCT", "20"))


# ---------------------------------------------------------------------------
# Data loaders — never raise
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> list | dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metric computers
# ---------------------------------------------------------------------------

def compute_equity_metrics(trades: list | None) -> dict:
    if not trades:
        return {"available": False}

    counts = Counter(str(t.get("action", "")).upper() for t in trades)
    tickers = [str(t.get("ticker", "")) for t in trades if t.get("ticker")]
    ticker_counts = Counter(tickers)
    most_active = ticker_counts.most_common(1)[0] if ticker_counts else ("—", 0)

    return {
        "available": True,
        "total_signals": len(trades),
        "buy": counts.get("BUY", 0),
        "sell": counts.get("SELL", 0),
        "hold": counts.get("HOLD", 0),
        "unique_tickers": len(set(tickers)),
        "most_active_ticker": most_active[0],
        "most_active_count": most_active[1],
    }


def compute_wheel_metrics(wheel_log: list | None) -> dict:
    if not wheel_log:
        return {"available": False}

    total_premium = round(sum(float(e.get("total_premium", 0)) for e in wheel_log), 2)
    n = len(wheel_log)
    avg_weekly = round(total_premium / n, 2) if n else 0.0
    annualized = round(avg_weekly * 52, 2)

    best_entry = max(wheel_log, key=lambda e: float(e.get("total_premium", 0)), default=None)
    best_week_premium = round(float(best_entry.get("total_premium", 0)), 2) if best_entry else 0.0
    best_week_ticker  = best_entry.get("ticker", "—") if best_entry else "—"
    # Use logged_at date if expiry not present
    best_week_date = (
        best_entry.get("expiry") or best_entry.get("logged_at", "")[:10]
        if best_entry else "—"
    )

    # Annualized return on STARTING_CAPITAL for context
    starting = float(os.getenv("CAPITAL_STARTING_USD", "200"))
    ann_pct   = round(annualized / starting * 100, 1) if starting else 0.0

    return {
        "available": True,
        "simulations": n,
        "total_premium": total_premium,
        "avg_weekly_premium": avg_weekly,
        "best_week_premium": best_week_premium,
        "best_week_ticker": best_week_ticker,
        "best_week_date": best_week_date,
        "annualized_projection": annualized,
        "annualized_pct_on_capital": ann_pct,
        "capital": starting,
    }


def compute_portfolio_state(state: dict | None) -> dict:
    starting = float(os.getenv("CAPITAL_STARTING_USD", "200"))
    if not state:
        state = {"peak": starting, "current": starting}

    peak    = float(state.get("peak", starting))
    current = float(state.get("current", starting))
    drawdown = round((peak - current) / peak * 100, 2) if peak > 0 else 0.0
    kill_switch = drawdown > _MAX_DRAWDOWN_PCT

    return {
        "peak": peak,
        "current": current,
        "drawdown_pct": drawdown,
        "kill_switch_active": kill_switch,
    }


# ---------------------------------------------------------------------------
# Dashboard builder — returns structured dict
# ---------------------------------------------------------------------------

def build_dashboard() -> dict:
    equity_data  = _load_json(PAPER_TRADES_PATH)
    wheel_data   = _load_json(WHEEL_LOG_PATH)
    crypto_data  = _load_json(CRYPTO_TRADES_PATH)
    portfolio    = _load_json(PORTFOLIO_STATE_PATH)

    equity_metrics  = compute_equity_metrics(equity_data if isinstance(equity_data, list) else None)
    wheel_metrics   = compute_wheel_metrics(wheel_data if isinstance(wheel_data, list) else None)
    portfolio_state = compute_portfolio_state(portfolio if isinstance(portfolio, dict) else None)

    crypto_count = len(crypto_data) if isinstance(crypto_data, list) else 0

    return {
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M CT"),
        "equity": equity_metrics,
        "wheel": wheel_metrics,
        "portfolio": portfolio_state,
        "crypto_signals_logged": crypto_count,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _eq_line(label: str, value: str, width: int = 32) -> str:
    return f"  {label:<{width}} {value}"


def render_dashboard(data: dict) -> str:
    lines = []
    sep = "=" * 60
    thin = "─" * 41

    lines.append(sep)
    lines.append("  GALACTIC CAPITAL — Portfolio Dashboard")
    lines.append(f"  As of: {data['as_of']}")
    lines.append(sep)

    # Equity section
    eq = data["equity"]
    lines.append("")
    lines.append("  EQUITY SIGNALS (paper_trades_dryrun.json)")
    lines.append(f"  {thin}")
    if not eq.get("available"):
        lines.append("  No data yet")
    else:
        lines.append(_eq_line("Total signals logged:", str(eq["total_signals"])))
        bsh = f"{eq['buy']} / {eq['sell']} / {eq['hold']}"
        lines.append(_eq_line("BUY / SELL / HOLD:", bsh))
        lines.append(_eq_line("Unique tickers:", str(eq["unique_tickers"])))
        lines.append(_eq_line("Most active:", f"{eq['most_active_ticker']} ({eq['most_active_count']} signals)"))

    # Wheel section
    wh = data["wheel"]
    lines.append("")
    lines.append("  WHEEL STRATEGY (wheel_log.json)")
    lines.append(f"  {thin}")
    if not wh.get("available"):
        lines.append("  No data yet")
    else:
        lines.append(_eq_line("Simulations logged:", str(wh["simulations"])))
        lines.append(_eq_line("Total premium (sim):", f"${wh['total_premium']:.2f}"))
        lines.append(_eq_line("Avg weekly premium:", f"${wh['avg_weekly_premium']:.2f}"))
        best = f"${wh['best_week_premium']:.2f} ({wh['best_week_ticker']}, {wh['best_week_date']})"
        lines.append(_eq_line("Best week:", best))
        ann = f"${wh['annualized_projection']:.2f} (~{wh['annualized_pct_on_capital']:.1f}% on ${wh['capital']:.0f} capital)"
        lines.append(_eq_line("Annualized projection:", ann))

    # Portfolio state
    ps = data["portfolio"]
    lines.append("")
    lines.append("  PORTFOLIO STATE")
    lines.append(f"  {thin}")
    lines.append(_eq_line("Peak:", f"${ps['peak']:.2f}"))
    lines.append(_eq_line("Current:", f"${ps['current']:.2f}"))
    lines.append(_eq_line("Drawdown:", f"{ps['drawdown_pct']:.1f}%"))
    ks_label = "ACTIVE ⚠" if ps["kill_switch_active"] else "SAFE ✓"
    lines.append(_eq_line("Kill switch:", ks_label))

    # Crypto footer
    lines.append("")
    lines.append(f"  Crypto signals logged (crypto_paper_trades.json): {data['crypto_signals_logged']}")

    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Galactic Capital — Portfolio Dashboard")
    parser.add_argument("--json", dest="json_out", action="store_true",
                        help="Emit JSON instead of formatted text (for n8n)")
    args = parser.parse_args()

    data = build_dashboard()

    if args.json_out:
        print(json.dumps(data, indent=2))
    else:
        print(render_dashboard(data))


if __name__ == "__main__":
    main()
