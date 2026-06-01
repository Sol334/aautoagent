#!/usr/bin/env python3
"""
galactic-capital/scripts/paper_trader.py

Dry-run paper trading engine. Fetches real prices via yfinance (no API key),
reads pending signals from brain/Capital.md, and prints what trades would be made.

Live execution is disabled until:
  1. CAPITAL_PAPER_TRADING=false in .env
  2. ALPACA_API_KEY and ALPACA_SECRET_KEY are set
  3. Ryan explicitly confirms in writing

Usage:
  python galactic-capital/scripts/paper_trader.py --dry-run
  python galactic-capital/scripts/paper_trader.py --max-tickers 3
  python galactic-capital/scripts/paper_trader.py --watchlist AAPL,MSFT,NVDA
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
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

PAPER_TRADING = os.getenv("CAPITAL_PAPER_TRADING", "true").lower() in ("true", "1", "yes")
ALPACA_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
DEFAULT_WATCHLIST = os.getenv(
    "CAPITAL_WATCHLIST", "AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL,META,JPM,GS,XOM"
)
DRY_RUN_LOG = CAPITAL_ROOT / "data" / "paper_trades_dryrun.json"
PORTFOLIO_STATE_PATH = CAPITAL_ROOT / "data" / "portfolio_state.json"

MAX_POSITION      = float(os.getenv("CAPITAL_MAX_POSITION_USD", "20"))
STARTING_CAPITAL  = float(os.getenv("CAPITAL_STARTING_USD", "200"))
MAX_DRAWDOWN_PCT  = float(os.getenv("CAPITAL_MAX_DRAWDOWN_PCT", "20"))
# In risk_off macro regime the drawdown floor tightens by 50% to protect capital
_RISK_OFF_DRAWDOWN_MULTIPLIER = 0.5


def _load_portfolio_state() -> dict:
    if PORTFOLIO_STATE_PATH.exists():
        try:
            return json.loads(PORTFOLIO_STATE_PATH.read_text())
        except Exception:
            pass
    return {"peak": STARTING_CAPITAL, "current": STARTING_CAPITAL}


def _is_kill_switch_active() -> bool:
    state = _load_portfolio_state()
    peak = state.get("peak", STARTING_CAPITAL)
    current = state.get("current", STARTING_CAPITAL)
    if peak <= 0:
        return False
    drawdown_pct = (peak - current) / peak * 100
    return drawdown_pct > MAX_DRAWDOWN_PCT


def _fetch_quotes(tickers: list) -> dict:
    """Fetch current prices via yfinance. No API key needed."""
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed — run: pip install yfinance")
        return {}

    results = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
            results[ticker] = float(price) if price else None
        except Exception as exc:
            log.warning("yfinance error for %s: %s", ticker, exc)
            results[ticker] = None
    return results


def _read_pending_signals() -> list:
    """Parse brain/Capital.md for lines under '## Active Signals'."""
    brain = CAPITAL_ROOT / "brain" / "Capital.md"
    if not brain.exists():
        return []
    in_section = False
    signals = []
    for line in brain.read_text().splitlines():
        if line.strip() == "## Active Signals":
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line.strip() and not line.startswith("*"):
            signals.append(line.strip("- ").strip())
    return signals


def _log_dryrun(entries: list) -> None:
    DRY_RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if DRY_RUN_LOG.exists():
        try:
            existing = json.loads(DRY_RUN_LOG.read_text())
        except Exception:
            pass
    existing.extend(entries)
    DRY_RUN_LOG.write_text(json.dumps(existing[-500:], indent=2))


def _get_effective_drawdown_limit() -> float:
    """Return the active drawdown limit, tightened 50% in risk_off macro regime."""
    try:
        from agents.macro_agent import MacroAgent
        regime = MacroAgent().get_regime()
        if regime == "risk_off":
            limit = MAX_DRAWDOWN_PCT * _RISK_OFF_DRAWDOWN_MULTIPLIER
            log.warning("Macro regime: RISK_OFF — tightening drawdown limit to %.0f%%", limit)
            return limit
    except Exception as exc:
        log.debug("macro_agent unavailable: %s", exc)
    return MAX_DRAWDOWN_PCT


def run(tickers: list, max_tickers: int, dry_run: bool, signals: dict | None = None) -> None:
    # Kill switch check: halt if portfolio drawdown exceeds threshold
    effective_limit = _get_effective_drawdown_limit()
    state = _load_portfolio_state()
    peak = state.get("peak", STARTING_CAPITAL)
    current = state.get("current", STARTING_CAPITAL)
    if peak > 0 and (peak - current) / peak * 100 > effective_limit:
        log.warning(
            "KILL SWITCH ACTIVE: portfolio drawdown exceeded %.0f%% — skipping all trades",
            effective_limit,
        )
        return

    # Safety checks before anything else
    if not PAPER_TRADING:
        log.warning("CAPITAL_PAPER_TRADING=false — live trading not yet implemented")
        sys.exit(0)

    if not dry_run and (not ALPACA_KEY or ALPACA_KEY == "CHANGE_ME"):
        log.info("ALPACA_API_KEY not configured — forcing dry-run mode")
        dry_run = True

    tickers = tickers[:max_tickers]
    log.info("Paper trader starting — %d tickers, dry_run=%s", len(tickers), dry_run)

    # Pending signals: prefer passed-in dict from market_analyst, else parse Capital.md
    brain_signals = _read_pending_signals()
    if brain_signals:
        log.info("Pending signals from Capital.md: %s", brain_signals)
    else:
        log.info("No pending signals in brain/Capital.md")

    # Fetch prices
    quotes = _fetch_quotes(tickers)

    log_entries = []
    print(f"\n{'='*60}")
    print(f"  GALACTIC CAPITAL — Paper Trader ({datetime.now():%Y-%m-%d %H:%M CT})")
    print(f"  Mode: {'DRY-RUN' if dry_run else 'PAPER (Alpaca)'}")
    print(f"  Max position: ${MAX_POSITION:.0f} | Drawdown limit: {MAX_DRAWDOWN_PCT:.0f}%")
    print(f"{'='*60}")

    for ticker in tickers:
        price = quotes.get(ticker)
        if price is None:
            print(f"  {ticker:6s}  — price unavailable")
            continue

        # Resolve action: passed-in signals dict takes priority over Capital.md text
        action = "HOLD"
        if signals and ticker in signals:
            action = signals[ticker].get("action", "HOLD")
        else:
            for sig in brain_signals:
                if ticker.upper() in sig.upper():
                    action = "BUY" if "BULLISH" in sig.upper() else "SELL" if "BEARISH" in sig.upper() else "HOLD"
                    break

        shares = round(MAX_POSITION / price, 4) if action in ("BUY", "SELL") else 0

        if action == "HOLD":
            print(f"  {ticker:6s}  ${price:>9.2f}  HOLD")
        else:
            print(f"  {ticker:6s}  ${price:>9.2f}  Would {action} {shares} shares (~${MAX_POSITION:.0f})")

        log_entries.append({
            "ts": datetime.now().isoformat(),
            "ticker": ticker,
            "price": price,
            "action": action,
            "shares": shares,
            "dry_run": dry_run,
        })

    print(f"{'='*60}\n")
    _log_dryrun(log_entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Galactic Capital — Paper Trader")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Print trades only, no execution (default: True)")
    parser.add_argument("--max-tickers", type=int, default=12,
                        help="Maximum number of tickers to process")
    parser.add_argument("--watchlist", type=str, default=DEFAULT_WATCHLIST,
                        help="Comma-separated ticker list")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.watchlist.split(",") if t.strip()]
    run(tickers=tickers, max_tickers=args.max_tickers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
