import json
import os
import re


class AnalystAgent:
    name = "analyst"
    description = "Market signal analyst — runs 6-factor pipeline (sentiment, forecast, earnings, options, political, fundamentals) per ticker"
    skills = ["market_analysis", "trade_signals", "sentiment", "forecast", "fundamentals", "analyze"]

    def handle(self, text: str) -> str:
        try:
            # Extract tickers from text (e.g. "analyze NVDA AAPL" → ["NVDA", "AAPL"])
            found = re.findall(r'\b[A-Z]{1,5}\b', text)
            # Filter to plausible tickers: 1–5 uppercase letters, excluding common English words
            _STOP = {"A", "I", "OR", "AND", "FOR", "THE", "IN", "ON", "AT", "TO", "DO"}
            tickers = [t for t in found if t not in _STOP] or None

            if not tickers:
                default_watchlist = os.getenv(
                    "CAPITAL_WATCHLIST",
                    "AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL,META,JPM,GS,XOM,SPY,QQQ",
                )
                tickers = [t.strip() for t in default_watchlist.split(",") if t.strip()]

            from scripts.market_analyst import run_analysis
            result = run_analysis(tickers=tickers, dry_run=True)

            lines = ["Signal Table"]
            lines.append(f"{'Ticker':<8} {'Action':<6} {'Score':>6}  {'Trend':<8}  Reason")
            lines.append("-" * 70)
            for ticker, data in result.items():
                lines.append(
                    f"{ticker:<8} {data.get('action','?'):<6} {data.get('score', 0):>6.2f}"
                    f"  {data.get('trend','?'):<8}  {data.get('reason','')}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"error: {exc}"
