# Galactic Capital

A fully local, paper-first **financial intelligence system**. It watches what the big players
do — congressional disclosures, macro regime shifts, options flow, news sentiment — turns those
into trade signals via local Ollama models, and paper-trades them with a hard drawdown kill
switch. No cloud AI. No real money moves until you explicitly flip the safety gates.

> Extracted from the `galactic-capital/` arm of [`Sol334/galactic-system`](https://github.com/Sol334/galactic-system)
> into its own repo so it can grow and deploy independently.

## What it does

| Signal source | Module | What it reads |
|---|---|---|
| Congressional trades | `agents/political_tracker.py` + `scripts/political_watchdog.py` | Finnhub congressional disclosures → LLM → BUY/SELL/HOLD signals, alerts via Telegram |
| Macro regime | `agents/macro_agent.py` | FRED 10Y–2Y yield curve + VIX → risk_on / neutral / risk_off |
| News sentiment | `agents/sentiment_agent.py` | ProsusAI/finBERT scores last 10 headlines (−1 to +1) |
| Price forecast | `agents/forecast_agent.py` | IBM Granite TTM-R2 trend on 90-day history |
| Earnings windows | `agents/earnings_agent.py` | yfinance calendar → IV expansion windows |
| Options flow | `scripts/options_flow_monitor.py` | Polygon OTM vol/OI → unusual call/put activity |

`scripts/market_analyst.py` fuses all five into one Ollama decision per ticker.
`scripts/paper_trader.py` sizes positions and enforces the kill switch.

## Market focus (recommended for a small starting stake)

1. **US equities + ETFs** driven by the congressional-trade + macro + sentiment signals (paper).
   Best signal-to-noise, free data, the core edge.
2. **Options *flow* as a confirmation signal, not as trades** — real options need far more
   collateral than a starter stake; keep `wheel_trader` / `options_flow_monitor` as indicators.
3. **Small BTC/ETH crypto sleeve (paper)** — 24/7, Coinbase wired, keep size tiny.
4. **Skip forex** — thin retail edge, leverage risk.

## Quick start

```bash
pip install -r requirements.txt
cp .env.template .env     # add FINNHUB / ALPACA / POLYGON / FRED keys
make test                 # 69 tests, fully mocked — no keys required
make watchdog             # dry-run the congressional-trade monitor
```

You also need a local [Ollama](https://ollama.com) instance (`OLLAMA_HOST`, default
`localhost:11434`). See `CLAUDE.md` for the full architecture and command reference.

## Safety

Everything ships in **paper mode** (`CAPITAL_PAPER_TRADING=true`). Live execution requires
setting broker keys AND flipping the gate to `false`. `FEATURE_OPTIONS_FLOW`,
`FEATURE_CRYPTO_TRADING`, and `FEATURE_WHEEL_TRADING` are all `false` by default.

**This is not financial advice.** All signals are for personal review only.
