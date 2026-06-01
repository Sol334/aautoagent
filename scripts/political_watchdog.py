#!/usr/bin/env python3
"""
galactic-capital/scripts/political_watchdog.py

Monitors congressional stock trades via Finnhub, analyzes with local LLM,
and alerts Ryan via Telegram when significant activity is detected.

Schedule: every 6 hours via cron or n8n HTTP node
Manual:
  python galactic-capital/scripts/political_watchdog.py           # live run
  python galactic-capital/scripts/political_watchdog.py --dry-run # no Telegram
  python galactic-capital/scripts/political_watchdog.py --test    # synthetic trade
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

CAPITAL_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(CAPITAL_ROOT))

from dotenv import load_dotenv
load_dotenv(CAPITAL_ROOT / ".env")

import httpx

from agents.political_tracker import PoliticalTracker, TradeSignal
from data_pipelines.finnhub_connector import FinnhubConnector

# ── config ───────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FINNHUB_KEY     = os.getenv("FINNHUB_API_KEY", "")
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT   = os.getenv("RYAN_TELEGRAM_CHAT_ID", "")
WATCHLIST       = [t.strip() for t in os.getenv("CAPITAL_WATCHLIST",
    "AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL,META,JPM,GS,XOM,SPY,QQQ").split(",") if t.strip()]
STATE_PATH      = Path(os.getenv("WATCHDOG_STATE_PATH",
    str(CAPITAL_ROOT / "data" / "watchdog_state.json")))
LOG_PATH        = Path(os.getenv("WATCHDOG_LOG_PATH",
    str(CAPITAL_ROOT / "data" / "watchdog_log.json")))
LOOKBACK_DAYS   = int(os.getenv("WATCHDOG_LOOKBACK_DAYS", "7"))
HIGH_VALUE_MIN  = float(os.getenv("WATCHDOG_HIGH_VALUE_MIN", "250000"))
CLUSTER_MIN     = int(os.getenv("WATCHDOG_CLUSTER_MIN", "2"))
CLUSTER_DAYS    = int(os.getenv("WATCHDOG_CLUSTER_DAYS", "30"))
SIGNAL_MIN_CONF = float(os.getenv("WATCHDOG_SIGNAL_MIN_CONF", "0.5"))


# ── state management ─────────────────────────────────────────────────────────────────────────────

def _load_state() -> set:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.exists():
        try:
            return set(json.loads(STATE_PATH.read_text()))
        except Exception:
            pass
    return set()


def _save_state(seen: set) -> None:
    STATE_PATH.write_text(json.dumps(sorted(seen), indent=2))


def _append_log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_entries = []
    if LOG_PATH.exists():
        try:
            log_entries = json.loads(LOG_PATH.read_text())
        except Exception:
            pass
    log_entries.append(entry)
    # Keep last 1000 entries
    LOG_PATH.write_text(json.dumps(log_entries[-1000:], indent=2))


# ── Telegram ──────────────────────────────────────────────────────────────────────────────────

def _send_telegram(message: str, dry_run: bool) -> None:
    if dry_run:
        print(f"\n[DRY-RUN TELEGRAM]\n{message}\n")
        return
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        log.warning("Telegram not configured — skipping alert")
        return
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT, "text": message, "parse_mode": "HTML"},
            )
            if resp.status_code != 200:
                log.warning("Telegram send failed: %s", resp.text[:200])
    except Exception as exc:
        log.error("Telegram error: %s", exc)


def _format_alert(signal: TradeSignal) -> str:
    direction = {"BULLISH": "📈", "BEARISH": "📉", "AVOID": "⚠️", "NEUTRAL": "➡️"}.get(
        signal.signal, "❓"
    )
    trades_summary = "\n".join(
        f"  • {t['representative']} — {t['type']} {t['amount_range']}"
        for t in signal.raw_trades[:5]
    )
    edge = " 🏗️ Committee edge" if signal.committee_edge else ""
    return (
        f"{direction} <b>GALACTIC CAPITAL — Political Signal</b>\n"
        f"<b>Ticker:</b> {signal.ticker} | <b>Signal:</b> {signal.signal} "
        f"({signal.confidence:.0%} conf){edge}\n\n"
        f"<b>Congressional Trades:</b>\n{trades_summary}\n\n"
        f"<b>Analysis:</b> {signal.reasoning[:400]}"
    )


# ── detection logic ─────────────────────────────────────────────────────────────────────────────

def _detect_new_trades(all_trades: list, seen_ids: set) -> list:
    """Filter to trades not yet seen."""
    return [t for t in all_trades if t.get("_id") not in seen_ids]


def _detect_clusters(trades: list, ticker: str) -> bool:
    """True if 2+ reps traded same ticker within CLUSTER_DAYS."""
    from collections import Counter
    window = datetime.now() - timedelta(days=CLUSTER_DAYS)
    recent = [
        t for t in trades
        if t.get("transactionDate", "") >= window.strftime("%Y-%m-%d")
    ]
    reps = Counter(t.get("name", "") for t in recent)
    unique_reps = sum(1 for v in reps.values() if v > 0)
    return unique_reps >= CLUSTER_MIN


def _is_high_value(trade: dict) -> bool:
    return trade.get("_amount_mid", 0) >= HIGH_VALUE_MIN


# ── synthetic test data ────────────────────────────────────────────────────────────────────────────

_TEST_TRADES = [
    {
        "symbol": "AAPL",
        "name": "Nancy Pelosi",
        "transactionDate": datetime.now().strftime("%Y-%m-%d"),
        "transactionType": "Purchase",
        "amount": "$500,001 - $1,000,000",
        "_amount_mid": 750000.0,
        "_id": "AAPL:Nancy Pelosi:test:Purchase",
        "owner": "Spouse",
    },
    {
        "symbol": "AAPL",
        "name": "Dan Crenshaw",
        "transactionDate": datetime.now().strftime("%Y-%m-%d"),
        "transactionType": "Purchase",
        "amount": "$15,001 - $50,000",
        "_amount_mid": 32500.0,
        "_id": "AAPL:Dan Crenshaw:test:Purchase",
        "owner": "Self",
    },
]


# ── main ──────────────────────────────────────────────────────────────────────────────────────

def run(dry_run: bool = False, test_mode: bool = False) -> dict:
    log.info("Political Watchdog starting (%s)", "TEST" if test_mode else "LIVE" if not dry_run else "DRY-RUN")

    seen_ids = _load_state()
    new_trades_by_ticker: dict = {}

    if test_mode:
        log.info("Test mode — injecting synthetic Pelosi/Crenshaw AAPL trades")
        # Clear test IDs from state so they always fire in test mode
        seen_ids -= {t["_id"] for t in _TEST_TRADES}
        new_trades_by_ticker["AAPL"] = _TEST_TRADES
    else:
        if not FINNHUB_KEY or FINNHUB_KEY == "CHANGE_ME":
            log.error("FINNHUB_API_KEY not set in .env — add it and retry")
            return {"error": "FINNHUB_API_KEY missing"}

        connector = FinnhubConnector(FINNHUB_KEY)
        from_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")

        for ticker in WATCHLIST:
            log.info("Fetching congressional trades for %s", ticker)
            try:
                trades = connector.congressional_trades(ticker, from_date, to_date)
            except Exception as exc:
                log.warning("Finnhub error for %s: %s", ticker, exc)
                continue

            # Attach symbol for tracker
            for t in trades:
                t.setdefault("symbol", ticker)

            new = _detect_new_trades(trades, seen_ids)
            if not new:
                continue

            # Alert on cluster or high-value
            cluster = _detect_clusters(trades, ticker)
            hv = any(_is_high_value(t) for t in new)
            if cluster or hv:
                new_trades_by_ticker[ticker] = new

    if not new_trades_by_ticker:
        log.info("No significant new congressional trades found")
        return {"new_trades": 0, "alerts_sent": 0}

    # Feed to LLM
    flat_trades = [t for ts in new_trades_by_ticker.values() for t in ts]
    log.info("Analyzing %d new trades across %d tickers", len(flat_trades), len(new_trades_by_ticker))

    tracker = PoliticalTracker()
    signals = tracker.analyze(flat_trades)

    alerts_sent = 0
    for signal in signals:
        if signal.signal in ("BULLISH", "BEARISH") and signal.confidence >= SIGNAL_MIN_CONF:
            msg = _format_alert(signal)
            _send_telegram(msg, dry_run)
            alerts_sent += 1
            log.info("Alert sent: %s %s (%.0f%%)", signal.ticker, signal.signal, signal.confidence * 100)
        else:
            log.info("No alert for %s (%s, conf=%.2f)", signal.ticker, signal.signal, signal.confidence)

    # Update state with newly seen trade IDs
    new_ids = {t["_id"] for ts in new_trades_by_ticker.values() for t in ts}
    seen_ids |= new_ids
    _save_state(seen_ids)

    # Append to log
    _append_log({
        "ts": datetime.now().isoformat(),
        "mode": "test" if test_mode else ("dry-run" if dry_run else "live"),
        "tickers_checked": WATCHLIST if not test_mode else ["AAPL"],
        "new_trades": len(flat_trades),
        "alerts_sent": alerts_sent,
        "signals": [{"ticker": s.ticker, "signal": s.signal, "confidence": s.confidence} for s in signals],
    })

    log.info("Done — %d alert(s) sent", alerts_sent)
    return {"new_trades": len(flat_trades), "alerts_sent": alerts_sent}


def main() -> None:
    parser = argparse.ArgumentParser(description="Galactic Capital — Political Watchdog")
    parser.add_argument("--dry-run", action="store_true", help="Fetch real data but skip Telegram")
    parser.add_argument("--test", action="store_true", help="Inject synthetic trade, verify pipeline end-to-end")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run, test_mode=args.test)
    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
