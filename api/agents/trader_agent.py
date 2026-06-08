import io
import os
import re
import sys


class TraderAgent:
    name = "trader"
    description = "Paper trader — evaluates pending signals and prints what trades would be made (dry-run only, no live execution)"
    skills = ["paper_trading", "position_sizing", "kill_switch", "trade_execution"]

    def handle(self, text: str) -> str:
        # dry_run is ALWAYS forced True — the hub must never execute live trades
        try:
            # Parse max_tickers from text if provided (e.g. "run trader max 3")
            max_tickers = 5
            m = re.search(r'max[_\s]*(\d+)', text, re.IGNORECASE)
            if m:
                max_tickers = int(m.group(1))

            # Parse optional explicit ticker list
            found = re.findall(r'\b[A-Z]{1,5}\b', text)
            _STOP = {"A", "I", "OR", "AND", "FOR", "THE", "IN", "ON", "AT", "TO", "DO"}
            tickers = [t for t in found if t not in _STOP] or None

            if not tickers:
                default_watchlist = os.getenv(
                    "CAPITAL_WATCHLIST",
                    "AAPL,MSFT,NVDA,TSLA,AMZN,GOOGL,META,JPM,GS,XOM",
                )
                tickers = [t.strip() for t in default_watchlist.split(",") if t.strip()]

            # Capture stdout since paper_trader.run() prints its output
            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                from scripts.paper_trader import run
                run(tickers=tickers, max_tickers=max_tickers, dry_run=True)
            finally:
                sys.stdout = old_stdout

            output = captured.getvalue().strip()
            return output if output else "Paper trader completed — no output produced"
        except Exception as exc:
            return f"error: {exc}"
