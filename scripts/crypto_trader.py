#!/usr/bin/env python3
"""
galactic-capital/scripts/crypto_trader.py

Crypto market analyzer and paper trader. Tracks BTC-USD, ETH-USD, SOL-USD.

Price data: yfinance (no key required).
News sentiment: Finnhub /news?category=crypto (requires FINNHUB_API_KEY).
Live execution: Coinbase Advanced Trade API — only enabled when ALL three are true:
  1. FEATURE_CRYPTO_TRADING=true
  2. COINBASE_API_KEY is set (not CHANGE_ME)
  3. CAPITAL_PAPER_TRADING=false

Usage:
  python galactic-capital/scripts/crypto_trader.py --dry-run        # no logging
  python galactic-capital/scripts/crypto_trader.py --since 24       # last 24h window
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

from agents.sentiment_agent import SentimentAgent

CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
COINBASE_KEY = os.getenv("COINBASE_API_KEY", "")
COINBASE_SECRET = os.getenv("COINBASE_API_SECRET", "")
PAPER_TRADING = os.getenv("CAPITAL_PAPER_TRADING", "true").lower() in ("true", "1", "yes")
FEATURE_CRYPTO = os.getenv("FEATURE_CRYPTO_TRADING", "false").lower() in ("true", "1", "yes")
MAX_POSITION = float(os.getenv("CAPITAL_MAX_POSITION_USD", "20"))

CRYPTO_LOG = CAPITAL_ROOT / "data" / "crypto_paper_trades.json"


def fetch_crypto_price(symbol: str) -> float | None:
    """Fetch latest crypto price via yfinance (no API key required)."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).fast_info
        price = getattr(info, "last_price", None) or getattr(
            info, "regular_market_price", None
        )
        return float(price) if price else None
    except Exception as exc:
        log.warning("fetch_crypto_price(%s): %s", symbol, exc)
        return None


def fetch_crypto_headlines(api_key: str) -> list:
    """Fetch recent crypto news headlines from Finnhub."""
    if not api_key or api_key == "CHANGE_ME":
        return []
    try:
        import httpx
        resp = httpx.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "crypto", "token": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json()
        return [item.get("headline", "") for item in items[:10] if item.get("headline")]
    except Exception as exc:
        log.warning("fetch_crypto_headlines error: %s", exc)
        return []


def _is_live_trading_enabled() -> bool:
    return (
        FEATURE_CRYPTO
        and not PAPER_TRADING
        and bool(COINBASE_KEY)
        and COINBASE_KEY != "CHANGE_ME"
    )


def _place_coinbase_order(symbol: str, action: str, usd_amount: float) -> bool:
    """Place a real Coinbase order. Only reached when all safety gates pass."""
    try:
        from coinbase.rest import RESTClient
        client = RESTClient(api_key=COINBASE_KEY, api_secret=COINBASE_SECRET)
        order_id = f"galactic-{datetime.now():%Y%m%d%H%M%S}"
        # Coinbase product IDs use USDC as quote: BTC-USD -> BTC-USDC
        product_id = symbol.replace("-USD", "-USDC")

        if action == "BUY":
            client.market_order_buy(
                client_order_id=order_id,
                product_id=product_id,
                quote_size=str(usd_amount),
            )
        else:
            client.market_order_sell(
                client_order_id=order_id,
                product_id=product_id,
                base_size=str(usd_amount),
            )
        log.info("Coinbase order placed: %s %s $%.2f", action, product_id, usd_amount)
        return True
    except Exception as exc:
        log.error("Coinbase order failed: %s", exc)
        return False


def _log_trade(entry: dict) -> None:
    CRYPTO_LOG.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if CRYPTO_LOG.exists():
        try:
            existing = json.loads(CRYPTO_LOG.read_text())
        except Exception:
            pass
    existing.append(entry)
    CRYPTO_LOG.write_text(json.dumps(existing[-500:], indent=2))


def run(since_hours: int = 24, dry_run: bool = False) -> None:
    sentiment_agent = SentimentAgent()
    headlines = fetch_crypto_headlines(FINNHUB_KEY)
    crypto_sentiment = sentiment_agent.analyze_headlines(headlines)

    live = _is_live_trading_enabled()
    mode_label = "LIVE (Coinbase)" if live else "PAPER (log only)"

    print(f"\n{'='*60}")
    print(
        f"  GALACTIC CAPITAL — Crypto Trader ({datetime.now():%Y-%m-%d %H:%M CT})"
    )
    print(f"  Mode: {'DRY-RUN' if dry_run else mode_label}")
    print(f"  Crypto market sentiment: {crypto_sentiment:+.2f}")
    print(f"{'='*60}")

    for symbol in CRYPTO_SYMBOLS:
        price = fetch_crypto_price(symbol)
        if price is None:
            print(f"  {symbol:10s}  — price unavailable")
            continue

        if crypto_sentiment > 0.3:
            action = "BUY"
        elif crypto_sentiment < -0.3:
            action = "SELL"
        else:
            action = "HOLD"

        usd_amount = MAX_POSITION if action in ("BUY", "SELL") else 0.0
        print(
            f"  {symbol:10s}  ${price:>12,.2f}  {action}"
            + (f"  ~${usd_amount:.0f}" if usd_amount else "")
        )

        entry = {
            "ts": datetime.now().isoformat(),
            "symbol": symbol,
            "price": price,
            "sentiment": crypto_sentiment,
            "action": action,
            "usd_amount": usd_amount,
            "executed": False,
            "paper": not live,
        }

        if not dry_run:
            if live and action in ("BUY", "SELL"):
                entry["executed"] = _place_coinbase_order(symbol, action, usd_amount)
            _log_trade(entry)

    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Galactic Capital — Crypto Trader")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print only — no trade logging or order placement",
    )
    parser.add_argument(
        "--since", type=int, default=24,
        help="Hours of history to consider for analysis window",
    )
    args = parser.parse_args()
    run(since_hours=args.since, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
