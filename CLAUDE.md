# CLAUDE.md — Galactic Capital

Financial intelligence + paper-trading system. Runs fully local on Ollama — no cloud AI.
Originally the `galactic-capital/` arm of `Sol334/galactic-system`; extracted into its own
repo (`Sol334/aautoagent`) so it can version and deploy independently. It still expects an
Ollama instance and (optionally) a Telegram bot, which it shares with the roofing stack when
run on the same machine.

## Setup

```bash
pip install -r requirements.txt
cp .env.template .env        # fill in FINNHUB / ALPACA / POLYGON / FRED keys
pytest tests/ -q             # 69 tests, fully mocked — no keys needed
```

## Common Commands

```bash
# Congressional trade tracker (new — multi-source STOCK Act signals)
python scripts/congressional_tracker.py --recent          # last 30 days
python scripts/congressional_tracker.py --sector TECH     # filter by sector
# Set QUIVER_API_TOKEN for Quiver Quantitative data (best, $30/mo)
# Falls back to free House Stock Watcher API automatically

# Market regime detector (new — gates all trading decisions)
python scripts/regime_detector.py --ticker SPY            # current regime
python scripts/regime_detector.py --regime-only           # just the label

# Political watchdog (congressional trade monitor)
python scripts/political_watchdog.py --test       # Synthetic trade → Telegram alert
python scripts/political_watchdog.py --dry-run    # Real Finnhub, no Telegram
python scripts/political_watchdog.py              # Live run

# Market analyst + paper trader
python scripts/market_analyst.py --dry-run --ticker NVDA
python scripts/paper_trader.py --dry-run
python scripts/paper_trader.py --max-tickers 5

# Options flow monitor (requires FEATURE_OPTIONS_FLOW=true)
python scripts/options_flow_monitor.py --dry-run --ticker NVDA

# Wheel strategy simulator
python scripts/wheel_trader.py --ticker SPY

# Crypto paper trading
python scripts/crypto_trader.py --dry-run

# Model benchmark
python scripts/model_benchmark.py

# Or via the Makefile
make test        # pytest
make watchdog    # political_watchdog --dry-run
make paper       # paper_trader --dry-run
make analyst     # market_analyst --dry-run --ticker NVDA
```

## Architecture

```
.
├── agents/
│   ├── base_agent.py           Ollama blocking call wrapper (strips <think> tags)
│   ├── political_tracker.py    Congressional trade → LLM → TradeSignal list
│   ├── sentiment_agent.py      ProsusAI/finBERT news sentiment scoring (-1.0 to +1.0)
│   ├── forecast_agent.py       IBM Granite TTM-R2 price trend prediction
│   ├── earnings_agent.py       yfinance earnings calendar — IV expansion window detection
│   └── macro_agent.py          FRED yield curve + VIX → risk_on / neutral / risk_off regime
├── data_pipelines/
│   └── finnhub_connector.py    Finnhub REST client (rate limiting, retry, amount parser)
├── scripts/
│   ├── congressional_tracker.py  Multi-source STOCK Act signal aggregator (free + Quiver tier)
│   ├── regime_detector.py        K-Means-inspired regime classification — gates all trades
│   ├── political_watchdog.py     Main pipeline: fetch → detect → analyze → alert
│   ├── paper_trader.py           Position sizing with kill switch + macro regime integration
│   ├── market_analyst.py         5-signal aggregator → Ollama BUY/SELL/HOLD → Capital.md
│   ├── options_flow_monitor.py   Polygon OTM options vol/OI unusual activity detection
│   ├── wheel_trader.py           Black-Scholes CSP simulator for SPY/QQQ weekly premium
│   ├── crypto_trader.py          Coinbase paper trading (FEATURE_CRYPTO_TRADING gate)
│   └── model_benchmark.py        Score Ollama models on historical congressional trades
├── brain/
│   └── Capital.md              Living context: active signals, positions, risk rules
├── config/
│   └── docker-compose.capital.yml  Postgres:5433, Redis:6380
└── tests/                      69 tests, fully mocked
```

## Signal Pipeline (market_analyst.py)

For each ticker in `CAPITAL_WATCHLIST`, the pipeline runs 5 data sources:

1. **finBERT sentiment** — last 10 Finnhub headlines → score -1 to +1
2. **Price forecast** — 90-day yfinance history → IBM Granite trend prediction
3. **Earnings calendar** — yfinance → days to earnings, EPS estimate, IV window flag
4. **Options flow** — Polygon OTM vol/OI ratio → unusual_calls / unusual_puts / normal
5. **Political signal** — cross-reference Capital.md signal log for congressional trades

All 5 signals feed into a single Ollama prompt → BUY / SELL / HOLD + reason.

**Macro overlay**: If `FRED_API_KEY` set, `MacroAgent` checks 10Y-2Y yield curve spread + VIX.
In `risk_off` regime (spread < 0 AND VIX > 25): kill switch tightens 50% (20% → 10% drawdown limit).

## Kill Switch

`paper_trader.py` tracks portfolio state in `data/portfolio_state.json`. When drawdown exceeds
`CAPITAL_MAX_DRAWDOWN_PCT` (default 20%), all trades are blocked until manually reset. In
`risk_off` macro regime, the limit tightens to 10%.

## Configuration

`.env` is loaded from the repo root (`load_dotenv(CAPITAL_ROOT / ".env")`). See `.env.template`
for every variable. Ollama is reached at `OLLAMA_HOST`; Telegram alerts use `TELEGRAM_BOT_TOKEN`
+ `RYAN_TELEGRAM_CHAT_ID`.

## Safety Rules

- `CAPITAL_PAPER_TRADING=true` is the default — no real trades ever executed in this state
- `paper_trader.py` requires `ALPACA_API_KEY` set AND `CAPITAL_PAPER_TRADING=false` for live execution
- `political_watchdog.py --dry-run` never sends Telegram messages
- `FEATURE_OPTIONS_FLOW=false` by default — enable once `POLYGON_API_KEY` is confirmed
- `FEATURE_CRYPTO_TRADING=false` by default — enable once `COINBASE_API_KEY` is confirmed
- No financial advice is generated — all signals are for personal review only
