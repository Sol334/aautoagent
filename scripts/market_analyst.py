#!/usr/bin/env python3
"""
galactic-capital/scripts/market_analyst.py

Orchestrates SentimentAgent + ForecastAgent + Ollama reasoning to produce
trade signals for each ticker in CAPITAL_WATCHLIST, then writes the ## Active
Signals section of brain/Capital.md.

Run modes:
  python galactic-capital/scripts/market_analyst.py            # full run
  python galactic-capital/scripts/market_analyst.py --dry-run  # print, no write
  python galactic-capital/scripts/market_analyst.py --ticker NVDA  # single ticker
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime
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

from agents.base_agent import BaseAgent
from agents.sentiment_agent import SentimentAgent
from agents.forecast_agent import ForecastAgent
from agents.earnings_agent import EarningsAgent
from scripts.options_flow_monitor import detect_unusual_flow

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv(
    "OLLAMA_REASONING_MODEL", os.getenv("OLLAMA_FAST_MODEL", "deepseek-r1:7b")
)
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
DEFAULT_WATCHLIST = os.getenv(
    "CAPITAL_WATCHLIST",
    "AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL,META,JPM,GS,XOM,SPY,QQQ",
)
CAPITAL_MD = CAPITAL_ROOT / "brain" / "Capital.md"


def _fetch_prices(ticker: str, period: str = "90d") -> list:
    """Fetch closing prices via yfinance. Returns empty list on failure."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period)
        return hist["Close"].tolist() if not hist.empty else []
    except Exception as exc:
        log.warning("_fetch_prices(%s): %s", ticker, exc)
        return []


def _read_political_signals() -> dict:
    """Read recent political signals from Capital.md ## Signal Log section."""
    if not CAPITAL_MD.exists():
        return {}
    content = CAPITAL_MD.read_text()
    signals = {}
    in_log = False
    for line in content.splitlines():
        if "## Signal Log" in line:
            in_log = True
            continue
        if in_log and line.startswith("## "):
            break
        if in_log and "|" in line:
            for ticker in re.findall(r'\b[A-Z]{2,5}\b', line):
                if "BUY" in line.upper() or "BULLISH" in line.upper():
                    signals[ticker] = "bullish (congressional)"
                elif "SELL" in line.upper() or "BEARISH" in line.upper():
                    signals[ticker] = "bearish (congressional)"
    return signals


def _ask_ollama(
    ticker: str,
    sentiment: float,
    trend: str,
    political: str,
    earnings_summary: str = "",
    options_flow: str = "normal",
) -> tuple:
    """Query Ollama for a BUY/SELL/HOLD recommendation. Returns (action, reason)."""
    extra_lines = ""
    if earnings_summary:
        extra_lines += f"- Earnings context: {earnings_summary}\n"
    if options_flow != "normal":
        extra_lines += f"- Options flow: {options_flow} (institutional positioning signal)\n"

    prompt = (
        f"You are a financial analyst. For ticker {ticker}:\n"
        f"- Sentiment score: {sentiment:+.2f} (range -1 to +1, positive=bullish)\n"
        f"- Price forecast: {trend}\n"
        f"- Political signal: {political or 'none'}\n"
        f"{extra_lines}"
        f"\nWhat is your trade recommendation? Reply with exactly one of: BUY, SELL, or HOLD\n"
        f"Then on the same line add a colon and a single short reason (max 10 words).\n"
        f"Example: BUY: strong sentiment and upward price momentum\n"
        f"Your answer:"
    )
    try:
        agent = BaseAgent(model=OLLAMA_MODEL, host=OLLAMA_HOST)
        raw = agent.generate(prompt, timeout=60)
        match = re.search(r'\b(BUY|SELL|HOLD)\b[:\s]*(.*)', raw, re.IGNORECASE)
        if match:
            action = match.group(1).upper()
            reason = match.group(2).strip()[:80]
            return action, reason
        for kw in ("BUY", "SELL", "HOLD"):
            if kw in raw.upper():
                return kw, raw.strip()[:80]
        return "HOLD", "no clear signal"
    except Exception as exc:
        log.warning("Ollama unavailable for %s (%s) — defaulting HOLD", ticker, exc)
        return "HOLD", "Ollama unavailable"


def _write_signals_to_capital_md(signals: dict) -> None:
    """Overwrite only the ## Active Signals section in brain/Capital.md."""
    if not CAPITAL_MD.exists():
        log.warning("Capital.md not found at %s", CAPITAL_MD)
        return

    content = CAPITAL_MD.read_text()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CT")

    lines = []
    for ticker, info in signals.items():
        action = info["action"]
        reason = info["reason"]
        score = info["score"]
        trend = info["trend"]
        lines.append(
            f"- {ticker}: {action} (sentiment {score:+.2f}, forecast {trend}, {reason})"
        )

    new_section = "## Active Signals\n\n"
    if lines:
        new_section += f"*Last updated: {timestamp}*\n\n" + "\n".join(lines) + "\n"
    else:
        new_section += f"*No actionable signals as of {timestamp}.*\n"

    # Replace the ## Active Signals section (up to the next ## heading or end of file)
    updated = re.sub(
        r'## Active Signals\n.*?(?=\n## |\Z)',
        new_section,
        content,
        flags=re.DOTALL,
    )
    if "## Active Signals" not in updated:
        updated = content.rstrip() + "\n\n" + new_section

    CAPITAL_MD.write_text(updated)
    log.info("Capital.md updated with %d signal(s)", len(signals))


def run_analysis(tickers: list, dry_run: bool = False) -> dict:
    """Run full analysis pipeline.

    Returns {ticker: {action: str, reason: str, score: float, trend: str}}.
    """
    sentiment_agent = SentimentAgent()
    forecast_agent = ForecastAgent()
    earnings_agent = EarningsAgent()
    political = _read_political_signals()

    results = {}
    for ticker in tickers:
        log.info("Analyzing %s ...", ticker)

        headlines = sentiment_agent.fetch_headlines(ticker, FINNHUB_KEY)
        score = sentiment_agent.analyze_headlines(headlines)

        prices = _fetch_prices(ticker)
        trend = forecast_agent.predict_trend(prices) if prices else "flat"

        earnings_ctx = earnings_agent.get_earnings_context(ticker)
        earnings_summary = earnings_ctx.get("summary", "")
        if earnings_ctx.get("is_within_window"):
            log.info("%s: EARNINGS WINDOW — %s", ticker, earnings_summary)

        flow = detect_unusual_flow(ticker)
        if flow != "normal":
            log.info("%s: UNUSUAL OPTIONS FLOW — %s", ticker, flow)

        pol_signal = political.get(ticker, "")
        action, reason = _ask_ollama(
            ticker, score, trend, pol_signal, earnings_summary, flow
        )

        results[ticker] = {
            "action": action,
            "reason": reason,
            "score": score,
            "trend": trend,
        }
        log.info(
            "%s: %s  sentiment %.2f  forecast %s", ticker, action, score, trend
        )

    if not dry_run:
        _write_signals_to_capital_md(results)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Galactic Capital — Market Analyst")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print signals only, do not write to Capital.md",
    )
    parser.add_argument(
        "--ticker", type=str, default="",
        help="Analyze a single ticker (overrides CAPITAL_WATCHLIST)",
    )
    parser.add_argument(
        "--watchlist", type=str, default=DEFAULT_WATCHLIST,
        help="Comma-separated ticker list",
    )
    args = parser.parse_args()

    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = [t.strip().upper() for t in args.watchlist.split(",") if t.strip()]

    log.info(
        "Market Analyst starting — %d tickers, dry_run=%s", len(tickers), args.dry_run
    )
    signals = run_analysis(tickers, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print(
        f"  GALACTIC CAPITAL — Market Analyst ({datetime.now():%Y-%m-%d %H:%M CT})"
    )
    print(f"  Mode: {'DRY-RUN' if args.dry_run else 'LIVE (wrote Capital.md)'}")
    print(f"{'='*60}")
    for ticker, info in signals.items():
        print(
            f"  {ticker:6s}  {info['action']:4s}  sentiment {info['score']:+.2f}"
            f"  forecast {info['trend']:4s}  — {info['reason']}"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
