#!/usr/bin/env python3
"""
galactic-capital/agents/sentiment_agent.py

News headline sentiment scoring using ProsusAI/finBERT — trained on Financial
PhraseBank, earnings releases, and 8-K filings (~440 MB, CPU-friendly).

Returns a float in [-1.0, 1.0]: positive = bullish, negative = bearish, 0 = neutral.
Falls back gracefully to 0.0 (neutral) if transformers is not installed or the model
fails to load — the rest of the pipeline keeps running.
"""

import logging
import sys
from pathlib import Path

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_ROOT = CAPITAL_ROOT  # repo root (was galactic-system parent before extraction)
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(CAPITAL_ROOT))

log = logging.getLogger(__name__)

_SENTIMENT_MODEL = "ProsusAI/finbert"


class SentimentAgent:
    def __init__(self):
        self._pipeline = None
        self._load_attempted = False

    def _get_pipeline(self):
        if self._load_attempted:
            return self._pipeline
        self._load_attempted = True
        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "text-classification",
                model=_SENTIMENT_MODEL,
                top_k=None,
            )
            log.info("SentimentAgent: loaded %s", _SENTIMENT_MODEL)
        except Exception as exc:
            log.warning(
                "SentimentAgent: transformers unavailable, using neutral fallback (%s)", exc
            )
            self._pipeline = None
        return self._pipeline

    def analyze_headlines(self, headlines: list) -> float:
        """Score a list of headlines. Returns -1.0 to +1.0. Empty list -> 0.0."""
        if not headlines:
            return 0.0

        pipe = self._get_pipeline()
        if pipe is None:
            return 0.0

        try:
            scores = []
            for headline in headlines[:10]:
                results = pipe(str(headline)[:512])
                # results: list of {label, score} dicts
                label_scores = {r["label"].lower(): r["score"] for r in results}
                weighted = (
                    label_scores.get("positive", 0.0)
                    - label_scores.get("negative", 0.0)
                )
                scores.append(weighted)
            return sum(scores) / len(scores) if scores else 0.0
        except Exception as exc:
            log.warning("SentimentAgent.analyze_headlines error: %s", exc)
            return 0.0

    def fetch_headlines(self, ticker: str, api_key: str) -> list:
        """Fetch last 10 news headlines for ticker from Finnhub."""
        if not api_key or api_key == "CHANGE_ME":
            log.debug("FINNHUB_API_KEY not set — skipping headlines for %s", ticker)
            return []
        try:
            import httpx
            resp = httpx.get(
                "https://finnhub.io/api/v1/news",
                params={"symbol": ticker, "token": api_key},
                timeout=10,
            )
            resp.raise_for_status()
            items = resp.json()
            headlines = []
            for item in items[:10]:
                title = item.get("headline", "")
                summary = item.get("summary", "")
                if title:
                    headlines.append(f"{title}. {summary}" if summary else title)
            return headlines
        except Exception as exc:
            log.warning("fetch_headlines(%s) error: %s", ticker, exc)
            return []
