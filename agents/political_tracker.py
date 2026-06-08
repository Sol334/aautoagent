"""Congressional trade analysis agent.

Receives a list of normalized Finnhub congressional trade records,
groups by ticker, feeds to Ollama reasoning model, returns TradeSignals.
"""

import json
import logging
import os
from dataclasses import dataclass, field

from .base_agent import BaseAgent

log = logging.getLogger(__name__)

_PROMPT = """\
You are a financial intelligence analyst reviewing recent congressional stock trades.
Analyze the following trades and determine the investment signal for each ticker.

Rules:
- Congressional trades are disclosed under the STOCK Act (45-day lag allowed)
- Weight these factors: committee membership relevance, trade size, direction (buy/sell), cluster size
- BULLISH: multiple buys by well-positioned reps, or high-confidence single large buy
- BEARISH: significant sells by reps with sector oversight
- AVOID: reps with poor accuracy history or conflicted committee assignments
- NEUTRAL: mixed signals or insufficient data

Trades (JSON):
{trades_json}

Respond in this exact JSON format, one entry per ticker:
[
  {{
    "ticker": "AAPL",
    "signal": "BULLISH",
    "confidence": 0.72,
    "committee_edge": true,
    "reasoning": "Two buys by reps on the House Commerce Committee within 14 days..."
  }}
]

Only output the JSON array, nothing else."""


@dataclass
class TradeSignal:
    ticker: str
    signal: str        # BULLISH / BEARISH / NEUTRAL / AVOID
    confidence: float  # 0.0 – 1.0
    reasoning: str
    committee_edge: bool = False
    raw_trades: list = field(default_factory=list)


class PoliticalTracker(BaseAgent):
    def __init__(self):
        super().__init__(
            model=os.getenv("OLLAMA_REASONING_MODEL", "deepseek-r1:7b"),
            host=os.getenv("OLLAMA_HOST", "localhost:11434"),
        )

    def analyze(self, trades: list) -> list:
        """Analyze a list of trade records grouped by ticker. Returns list[TradeSignal]."""
        if not trades:
            return []

        # Group by ticker for prompt
        by_ticker: dict = {}
        for t in trades:
            sym = t.get("symbol") or t.get("ticker", "UNKNOWN")
            by_ticker.setdefault(sym, []).append({
                "representative": t.get("name", ""),
                "date": t.get("transactionDate", ""),
                "type": t.get("transactionType", ""),
                "amount_range": t.get("amount", ""),
                "amount_midpoint_usd": t.get("_amount_mid", 0),
                "owner": t.get("owner", ""),
            })

        prompt = _PROMPT.format(trades_json=json.dumps(by_ticker, indent=2))

        try:
            raw = self.generate(prompt, timeout=120)
        except Exception as exc:
            log.error("PoliticalTracker LLM call failed: %s", exc)
            return [
                TradeSignal(
                    ticker=sym,
                    signal="NEUTRAL",
                    confidence=0.0,
                    reasoning=f"LLM unavailable: {exc}",
                    raw_trades=ts,
                )
                for sym, ts in by_ticker.items()
            ]

        # Parse JSON from model response
        try:
            # Find JSON array in response (model may prefix/suffix text)
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON array found in response")
            parsed = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError) as exc:
            log.warning("Could not parse LLM JSON: %s — raw: %.200s", exc, raw)
            return [
                TradeSignal(
                    ticker=sym,
                    signal="NEUTRAL",
                    confidence=0.0,
                    reasoning="Could not parse LLM response",
                    raw_trades=ts,
                )
                for sym, ts in by_ticker.items()
            ]

        signals = []
        for item in parsed:
            ticker = item.get("ticker", "UNKNOWN")
            signals.append(TradeSignal(
                ticker=ticker,
                signal=item.get("signal", "NEUTRAL").upper(),
                confidence=float(item.get("confidence", 0.0)),
                reasoning=item.get("reasoning", ""),
                committee_edge=bool(item.get("committee_edge", False)),
                raw_trades=by_ticker.get(ticker, []),
            ))

        return signals
