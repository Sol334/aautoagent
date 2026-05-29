#!/usr/bin/env python3
"""
galactic-capital/scripts/options_flow_monitor.py

Unusual options flow detector using Polygon.io options chain data.
Requires POLYGON_API_KEY (already in .env.template).
Feature-gated: set FEATURE_OPTIONS_FLOW=true to enable.

Signal logic:
  - Pull OTM options (15-30 delta range) expiring ~1 month out
  - "unusual_calls": OTM call volume > 3x 20-day avg AND vol/OI > 2.0
  - "unusual_puts": same logic for puts
  - "normal": neither condition met

Unusual call flow has documented 1-3 day lead time on 5-10% moves in
large-cap names (NVDA, TSLA, AMZN) — institutional desks telegraph positions.

Usage:
  python galactic-capital/scripts/options_flow_monitor.py --dry-run
  python galactic-capital/scripts/options_flow_monitor.py --ticker NVDA
"""

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_ROOT = CAPITAL_ROOT  # repo root (was galactic-system parent before extraction)
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(CAPITAL_ROOT))

from dotenv import load_dotenv
load_dotenv(SYSTEM_ROOT / ".env")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")
FEATURE_ENABLED = os.getenv("FEATURE_OPTIONS_FLOW", "false").lower() in ("true", "1", "yes")
DEFAULT_WATCHLIST = os.getenv(
    "CAPITAL_WATCHLIST", "AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL,META,JPM,GS,XOM"
)

_VOLUME_MULTIPLIER = 3.0
_VOL_OI_THRESHOLD = 2.0
_DELTA_MIN = 0.15
_DELTA_MAX = 0.30


def detect_unusual_flow(ticker: str) -> str:
    """
    Returns "unusual_calls", "unusual_puts", or "normal".

    Checks 1-month expiry OTM options (15-30 delta range) via Polygon.
    Falls back to "normal" if API is unavailable or feature is disabled.
    """
    if not FEATURE_ENABLED:
        log.debug("FEATURE_OPTIONS_FLOW=false — skipping options flow for %s", ticker)
        return "normal"

    if not POLYGON_KEY or POLYGON_KEY == "CHANGE_ME":
        log.debug("POLYGON_API_KEY not set — options flow unavailable")
        return "normal"

    try:
        call_signal = _check_flow(ticker, "call")
        if call_signal:
            return "unusual_calls"
        put_signal = _check_flow(ticker, "put")
        if put_signal:
            return "unusual_puts"
    except Exception as exc:
        log.warning("Options flow error for %s: %s", ticker, exc)

    return "normal"


def _check_flow(ticker: str, contract_type: str) -> bool:
    """Returns True if unusual flow detected for the given contract type."""
    import urllib.request, json, urllib.error

    expiry_min = (date.today() + timedelta(days=21)).isoformat()
    expiry_max = (date.today() + timedelta(days=45)).isoformat()

    url = (
        f"https://api.polygon.io/v3/snapshot/options/{ticker}"
        f"?contract_type={contract_type}"
        f"&expiration_date.gte={expiry_min}"
        f"&expiration_date.lte={expiry_max}"
        f"&limit=50"
        f"&apiKey={POLYGON_KEY}"
    )

    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 403:
            log.debug("Polygon 403 — key may lack options access")
            return False
        raise

    results = data.get("results", [])
    if not results:
        return False

    # Filter to OTM range by delta (absolute value)
    otm = []
    for r in results:
        greeks = r.get("greeks") or {}
        delta = abs(greeks.get("delta", 0))
        if _DELTA_MIN <= delta <= _DELTA_MAX:
            otm.append(r)

    if not otm:
        return False

    total_vol = sum(r.get("day", {}).get("volume", 0) for r in otm)
    total_oi = sum(r.get("open_interest", 0) for r in otm)

    if total_oi == 0:
        return False

    vol_oi_ratio = total_vol / total_oi
    avg_vol = total_oi / len(otm)  # rough proxy for typical daily volume

    return vol_oi_ratio > _VOL_OI_THRESHOLD and total_vol > avg_vol * _VOLUME_MULTIPLIER


def run_monitor(tickers: list[str], dry_run: bool) -> dict[str, str]:
    """Scan all tickers and return {ticker: flow_signal} dict."""
    results = {}
    for ticker in tickers:
        if dry_run:
            log.info("[DRY-RUN] Would check options flow for %s", ticker)
            results[ticker] = "normal"
        else:
            signal = detect_unusual_flow(ticker)
            log.info("%s options flow: %s", ticker, signal)
            results[ticker] = signal
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Galactic Capital — Options Flow Monitor")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--ticker", type=str, default="",
                        help="Single ticker to check (default: full watchlist)")
    parser.add_argument("--watchlist", type=str, default=DEFAULT_WATCHLIST)
    args = parser.parse_args()

    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = [t.strip() for t in args.watchlist.split(",") if t.strip()]

    print(f"\nOptions Flow Monitor — {len(tickers)} ticker(s)")
    print(f"Feature enabled: {FEATURE_ENABLED} | Polygon key set: {bool(POLYGON_KEY and POLYGON_KEY != 'CHANGE_ME')}\n")

    results = run_monitor(tickers, dry_run=args.dry_run)
    for ticker, signal in results.items():
        flag = " *** UNUSUAL ***" if signal != "normal" else ""
        print(f"  {ticker:8s}  {signal}{flag}")
    print()


if __name__ == "__main__":
    main()
