#!/usr/bin/env python3
"""Congressional Trade Tracker — Galactic Capital

Monitors STOCK Act disclosures for congressional trades that signal sector
momentum. Three data tiers, use whichever you have access to:

  Tier 1 (free): House Stock Watcher API (housestockwatcher.com)
  Tier 2 ($30/mo): Quiver Quantitative API (quiverquant.com) — most complete
  Tier 3 (fallback): Capitol Trades API (capitoltrades.com)

Strategy: when multiple congress members buy the same sector within 30 days,
this is a statistically significant signal. We gate this with market regime
(see regime_detector.py) before acting on paper positions.

Usage:
    python congressional_tracker.py --recent          # last 30 days
    python congressional_tracker.py --sector TECH     # filter by sector
    python congressional_tracker.py --dry-run         # print signals, no trade

Env vars:
    QUIVER_API_TOKEN — Quiver Quantitative API key ($30/mo, best data)
    CAPITAL_WATCHLIST — comma-separated tickers to monitor
"""

import os, re, json, logging, argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

log = logging.getLogger("congressional_tracker")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

QUIVER_TOKEN  = os.getenv("QUIVER_API_TOKEN", "")
WATCHLIST     = [t.strip().upper() for t in os.getenv("CAPITAL_WATCHLIST", "AAPL,MSFT,NVDA,SPY,QQQ").split(",") if t.strip()]
STATE_PATH    = Path(os.getenv("WATCHDOG_STATE_PATH", "/tmp/galactic/capital/congressional_state.json"))

_HOUSE_WATCHER_URL = "https://housestockwatcher.com/api"
_QUIVER_BASE       = "https://api.quiverquant.com/beta"


# ── Public dataclass (spec-compliant API surface) ─────────────────────────────

@dataclass
class CongressTrade:
    ticker: str
    rep_name: str
    chamber: str
    trade_date: str
    trade_type: str
    amount_range: str
    amount_est: float
    sector: str
    source: str


def parse_amount_est(amount_str: str) -> float:
    """Parse '$50,001 - $100,000' → 75000.0  |  '$1,000,001 +' → 1000001.0."""
    if not amount_str:
        return 0.0
    try:
        cleaned = re.sub(r"[^\d\-]", "", amount_str)
        if "-" in cleaned:
            parts = cleaned.split("-")
            if len(parts) == 2 and parts[0] and parts[1]:
                return (float(parts[0]) + float(parts[1])) / 2.0
        if cleaned:
            return float(cleaned)
    except Exception:
        pass
    return 0.0


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_house_watcher(days_back: int = 30) -> list[dict]:
    """Free tier: House Stock Watcher — covers House members only."""
    try:
        r = httpx.get(f"{_HOUSE_WATCHER_URL}/transactions", timeout=30)
        r.raise_for_status()
        cutoff = (date.today() - timedelta(days=days_back)).isoformat()
        trades = []
        for t in r.json():
            tx_date = (t.get("transaction_date") or t.get("disclosure_date") or "")[:10]
            if tx_date >= cutoff:
                trades.append({
                    "source":   "house_watcher",
                    "member":   t.get("representative", ""),
                    "ticker":   t.get("ticker", "").upper(),
                    "type":     t.get("type", ""),
                    "amount":   t.get("amount", ""),
                    "date":     tx_date,
                    "district": t.get("district", ""),
                })
        return trades
    except Exception as exc:
        log.warning("House Watcher fetch failed: %s", exc)
        return []


def fetch_quiver(days_back: int = 30) -> list[dict]:
    """Tier 2: Quiver Quantitative — Senate + House, structured data."""
    if not QUIVER_TOKEN:
        log.debug("QUIVER_API_TOKEN not set — skipping Quiver fetch")
        return []
    try:
        headers = {"Authorization": f"Token {QUIVER_TOKEN}"}
        r = httpx.get(f"{_QUIVER_BASE}/live/congresstrading", headers=headers, timeout=30)
        r.raise_for_status()
        cutoff = (date.today() - timedelta(days=days_back)).isoformat()
        trades = []
        for t in r.json():
            tx_date = (t.get("Date") or "")[:10]
            if tx_date >= cutoff:
                trades.append({
                    "source":   "quiver",
                    "member":   t.get("Representative", ""),
                    "ticker":   (t.get("Ticker") or "").upper(),
                    "type":     t.get("Transaction", ""),
                    "amount":   t.get("Range", ""),
                    "date":     tx_date,
                    "party":    t.get("Party", ""),
                    "chamber":  t.get("Chamber", ""),
                })
        return trades
    except Exception as exc:
        log.warning("Quiver fetch failed: %s", exc)
        return []


def fetch_trades(days_back: int = 30) -> list[dict]:
    """Returns best available trade data, Quiver preferred over House Watcher."""
    quiver = fetch_quiver(days_back)
    if quiver:
        return quiver
    return fetch_house_watcher(days_back)


# ── Analysis ──────────────────────────────────────────────────────────────────

# Rough sector map for common tickers (extend as needed)
_SECTOR_MAP = {
    "AAPL": "TECH", "MSFT": "TECH", "NVDA": "TECH", "META": "TECH",
    "GOOGL": "TECH", "AMZN": "TECH", "TSLA": "TECH",
    "JPM": "FINANCE", "GS": "FINANCE", "BAC": "FINANCE", "WFC": "FINANCE",
    "XOM": "ENERGY", "CVX": "ENERGY", "COP": "ENERGY",
    "LMT": "DEFENSE", "RTX": "DEFENSE", "NOC": "DEFENSE",
    "JNJ": "HEALTH", "UNH": "HEALTH", "PFE": "HEALTH", "ABBV": "HEALTH",
    "XHB": "CONSTRUCTION", "HD": "CONSTRUCTION", "LOW": "CONSTRUCTION",
    "SPY": "INDEX", "QQQ": "INDEX", "IWM": "INDEX",
}


def ticker_sector(ticker: str) -> str:
    return _SECTOR_MAP.get(ticker.upper(), "OTHER")


def analyze_signals(
    trades: list[dict],
    min_cluster: int = 2,
    cluster_days: int = 30,
    buy_types: tuple = ("purchase", "buy", "Purchase", "Buy"),
) -> list[dict]:
    """Identify tickers bought by multiple congress members in the window.

    Returns signals sorted by member count (highest conviction first).
    """
    cutoff = (date.today() - timedelta(days=cluster_days)).isoformat()
    buys: dict[str, set] = defaultdict(set)
    for t in trades:
        if t.get("type", "").lower() in {b.lower() for b in buy_types}:
            if t.get("date", "") >= cutoff and t.get("ticker"):
                buys[t["ticker"]].add(t.get("member", "unknown"))

    signals = []
    for ticker, members in buys.items():
        if len(members) >= min_cluster:
            signals.append({
                "ticker":       ticker,
                "sector":       ticker_sector(ticker),
                "member_count": len(members),
                "members":      sorted(members),
                "signal":       "BUY_CLUSTER",
                "confidence":   min(1.0, len(members) / 5),
            })
    return sorted(signals, key=lambda x: x["member_count"], reverse=True)


def watchlist_activity(trades: list[dict]) -> list[dict]:
    """Filter trades to only tickers in CAPITAL_WATCHLIST."""
    return [t for t in trades if t.get("ticker") in WATCHLIST]


def sector_pressure(trades: list[dict]) -> dict[str, dict]:
    """Returns buy/sell counts per sector."""
    pressure: dict[str, dict] = defaultdict(lambda: {"buys": 0, "sells": 0, "members": set()})
    for t in trades:
        sector = ticker_sector(t.get("ticker", ""))
        tx = t.get("type", "").lower()
        if "purchase" in tx or "buy" in tx:
            pressure[sector]["buys"] += 1
            pressure[sector]["members"].add(t.get("member", ""))
        elif "sale" in tx or "sell" in tx:
            pressure[sector]["sells"] += 1
    # Convert sets to counts for serialization
    return {k: {**v, "members": len(v["members"])} for k, v in pressure.items()}


# ── Spec-compliant aliases (return CongressTrade dataclasses) ─────────────────

def _dict_to_trade(t: dict, source: str) -> CongressTrade:
    raw_type = t.get("type", "")
    trade_type = "Purchase" if "purchase" in raw_type.lower() or "buy" in raw_type.lower() else (
        "Sale" if "sale" in raw_type.lower() or "sell" in raw_type.lower() else raw_type.capitalize()
    )
    amt_range = t.get("amount", "")
    return CongressTrade(
        ticker=t.get("ticker", "").upper(),
        rep_name=t.get("member", ""),
        chamber=t.get("chamber", "House"),
        trade_date=t.get("date", ""),
        trade_type=trade_type,
        amount_range=amt_range,
        amount_est=parse_amount_est(amt_range),
        sector=ticker_sector(t.get("ticker", "")),
        source=source,
    )


def fetch_house_stock_watcher(days_back: int = 30) -> list[CongressTrade]:
    return [_dict_to_trade(t, "housestockwatcher") for t in fetch_house_watcher(days_back)]


def fetch_quiver_quantitative(days_back: int = 30) -> list[CongressTrade]:
    return [_dict_to_trade(t, "quiver") for t in fetch_quiver(days_back)]


def fetch_all_trades(days_back: int = 30) -> list[CongressTrade]:
    raw = fetch_trades(days_back)
    seen: set = set()
    result: list[CongressTrade] = []
    for t in raw:
        trade = _dict_to_trade(t, t.get("source", ""))
        sig = (trade.ticker, trade.rep_name, trade.trade_date, trade.trade_type)
        if sig not in seen:
            seen.add(sig)
            result.append(trade)
    return result


def filter_by_sector(trades: list[CongressTrade], sector: str) -> list[CongressTrade]:
    if not sector:
        return trades
    return [t for t in trades if t.sector.upper() == sector.upper()]


# ── Persistence ───────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"last_run": "", "seen_signals": []}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(STATE_PATH)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Congressional trade signal tracker")
    p.add_argument("--recent", action="store_true", help="Show last 30 days of trades")
    p.add_argument("--sector", help="Filter signals by sector (e.g. TECH, DEFENSE)")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    trades = fetch_trades(args.days)
    log.info("Fetched %d trades", len(trades))

    signals = analyze_signals(trades)
    if args.sector:
        signals = [s for s in signals if s["sector"].upper() == args.sector.upper()]

    pressure = sector_pressure(trades)
    print(f"\n=== Congressional Trade Signals (last {args.days}d) ===")
    for sig in signals:
        print(f"  {sig['ticker']:6s} [{sig['sector']:12s}] {sig['member_count']} members — confidence {sig['confidence']:.0%}")
        print(f"    Members: {', '.join(sig['members'][:5])}")

    print("\n=== Sector Pressure ===")
    for sector, data in sorted(pressure.items(), key=lambda x: -x[1]["buys"]):
        print(f"  {sector:12s}  buys={data['buys']:3d}  sells={data['sells']:3d}  unique_members={data['members']}")

    state = load_state()
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    if not args.dry_run:
        save_state(state)
