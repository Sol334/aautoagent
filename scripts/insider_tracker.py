#!/usr/bin/env python3
"""SEC EDGAR insider trade tracker — Galactic Capital

Fetches Form 4 disclosures (2-day lag) as a fast-signal complement to the
congressional trade tracker (45-day lag).

Usage:
    python scripts/insider_tracker.py --ticker NVDA
    python scripts/insider_tracker.py --watchlist        # full CAPITAL_WATCHLIST
    python scripts/insider_tracker.py --days 14
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipelines.edgar_connector import EDGARConnector, InsiderTrade

log = logging.getLogger("insider_tracker")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

WATCHLIST = [
    t.strip().upper()
    for t in os.getenv("CAPITAL_WATCHLIST", "AAPL,MSFT,NVDA,SPY,QQQ").split(",")
    if t.strip()
]

_COL_W = {"ticker": 6, "name": 30, "title": 18, "type": 5, "date": 12, "shares": 12, "price": 10, "value": 14}


def _fmt_num(n: float | None, decimals: int = 2) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"${n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"${n:,.{decimals}f}"
    return f"${n:.{decimals}f}"


def _print_trades(trades: list[InsiderTrade]) -> None:
    if not trades:
        print("  (no trades found)")
        return

    header = (
        f"{'Ticker':<{_COL_W['ticker']}} "
        f"{'Insider':<{_COL_W['name']}} "
        f"{'Title':<{_COL_W['title']}} "
        f"{'Type':<{_COL_W['type']}} "
        f"{'Date':<{_COL_W['date']}} "
        f"{'Shares':>{_COL_W['shares']}} "
        f"{'Price':>{_COL_W['price']}} "
        f"{'Value':>{_COL_W['value']}}"
    )
    print(header)
    print("-" * len(header))

    for t in sorted(trades, key=lambda x: x.transaction_date, reverse=True):
        buy_sell = "BUY " if t.is_buy else "SELL"
        name = t.insider_name[:_COL_W["name"] - 1] if len(t.insider_name) > _COL_W["name"] else t.insider_name
        title = t.title[:_COL_W["title"] - 1] if len(t.title) > _COL_W["title"] else t.title
        print(
            f"{t.ticker:<{_COL_W['ticker']}} "
            f"{name:<{_COL_W['name']}} "
            f"{title:<{_COL_W['title']}} "
            f"{buy_sell} "
            f"{t.transaction_date:<{_COL_W['date']}} "
            f"{t.shares:>{_COL_W['shares']},.0f} "
            f"{_fmt_num(t.price_per_share):>{_COL_W['price']}} "
            f"{_fmt_num(t.value_usd):>{_COL_W['value']}}"
        )


def run(tickers: list[str], days: int) -> dict[str, list[InsiderTrade]]:
    connector = EDGARConnector()
    results: dict[str, list[InsiderTrade]] = {}
    for ticker in tickers:
        log.info("Fetching Form 4 filings for %s (last %d days)…", ticker, days)
        trades = connector.get_insider_trades(ticker, days_back=days)
        results[ticker] = trades
        buys  = sum(1 for t in trades if t.is_buy)
        sells = sum(1 for t in trades if not t.is_buy)
        print(f"\n=== {ticker} — {len(trades)} insider transactions (buys={buys}, sells={sells}) ===")
        _print_trades(trades)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEC EDGAR Form 4 insider trade tracker")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ticker", metavar="TICKER", help="Single ticker to look up")
    group.add_argument("--watchlist", action="store_true", help="Run for full CAPITAL_WATCHLIST")
    parser.add_argument("--days", type=int, default=30, help="Look-back window in days (default 30)")
    args = parser.parse_args()

    if args.watchlist:
        tickers = WATCHLIST
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        parser.print_help()
        sys.exit(0)

    run(tickers, args.days)
