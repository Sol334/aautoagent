# Capital.md — Living Investment Context

> Updated automatically by galactic-capital scripts after each significant event.
> Read by paper_trader.py and galactic_orchestrator.py for context.

## System Status

- **Mode:** Paper trading (CAPITAL_PAPER_TRADING=true)
- **Hardware:** RTX 3060 Profile A — upgrade to 3090 Profile B pending
- **Watchdog:** Runs every 6 hours monitoring CAPITAL_WATCHLIST
- **Live trading:** Disabled until ALPACA_API_KEY is set + Ryan confirms

## Watchlist

Default: AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, JPM, GS, XOM, SPY, QQQ
Override via CAPITAL_WATCHLIST in .env

## Active Signals

*No signals yet — watchdog has not run with a live Finnhub key.*

## Pending Actions

*None.*

## Model Routing (Capital)

| Task | Model |
|---|---|
| Congressional trade reasoning | OLLAMA_REASONING_MODEL (deepseek-r1:7b) |
| Sentiment analysis | OLLAMA_FAST_MODEL (llama3.2:3b) |
| Benchmarking | All available models |

## Risk Rules

- Paper trading only until CAPITAL_PAPER_TRADING=false + Alpaca keys confirmed
- No single position > 10% of paper portfolio
- Stop-loss: 7% below entry for individual stocks
- No leveraged ETFs in paper phase

---
## Signal Log
