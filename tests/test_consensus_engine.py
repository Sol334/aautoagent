"""Tests for decision/consensus_engine.py — pure Python, no external deps."""

import sys
import types
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decision.consensus_engine import ConsensusEngine, ConsensusSignal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trade(
    *,
    is_buy: bool,
    title: str = "CEO",
    transaction_date: str | None = None,
    value_usd: float | None = 45000.0,
    insider_name: str = "TEST INSIDER",
) -> types.SimpleNamespace:
    if transaction_date is None:
        transaction_date = date.today().isoformat()
    return types.SimpleNamespace(
        is_buy=is_buy,
        title=title,
        transaction_date=transaction_date,
        value_usd=value_usd,
        insider_name=insider_name,
    )


def _make_fundamental(signal: str, confidence: float, summary: str = "") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        signal=signal,
        confidence=confidence,
        summary=summary or f"PE=18, RevGrowth=22%",
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestAllBullish(unittest.TestCase):
    def test_all_bullish(self):
        engine = ConsensusEngine()
        result = engine.score(
            "AAPL",
            sentiment=0.8,
            trend="bullish",
            options_flow="unusual_calls",
            fundamental=_make_fundamental("BULLISH", 0.9),
            insider_trades=[_make_trade(is_buy=True, title="CEO")],
            political_signal="bullish (congressional)",
        )
        self.assertIsInstance(result, ConsensusSignal)
        self.assertGreater(result.weighted_score, 0)
        self.assertIn(result.conviction, ("high", "medium"))
        self.assertTrue(len(result.bullish_factors) > 0)


class TestAllBearish(unittest.TestCase):
    def test_all_bearish(self):
        engine = ConsensusEngine()
        result = engine.score(
            "TSLA",
            sentiment=-0.8,
            trend="bearish",
            options_flow="unusual_puts",
            fundamental=_make_fundamental("BEARISH", 0.9),
            insider_trades=[_make_trade(is_buy=False, title="CEO")],
            political_signal="bearish (congressional)",
        )
        self.assertLess(result.weighted_score, 0)


class TestNeutralNoSignals(unittest.TestCase):
    def test_neutral_no_signals(self):
        engine = ConsensusEngine()
        result = engine.score("XYZ")
        self.assertAlmostEqual(result.weighted_score, 0.0, places=6)
        self.assertEqual(result.conviction, "low")


class TestInsiderBuyCeoWeight(unittest.TestCase):
    def test_insider_buy_ceo_contributes_positively(self):
        engine = ConsensusEngine()
        trade = _make_trade(is_buy=True, title="CEO", value_usd=180000.0)
        result = engine.score("NVDA", insider_trades=[trade])
        # insider weight 0.25, direction +1, recency 1.0, title 1.0 → +0.25
        self.assertGreater(result.weighted_score, 0)
        self.assertTrue(
            any("insider_buy" in f for f in result.bullish_factors)
        )


class TestInsiderRecencyDecay(unittest.TestCase):
    def test_older_trade_has_lower_absolute_contribution(self):
        engine = ConsensusEngine()

        recent_date = date.today().isoformat()
        old_date = (date.today() - timedelta(days=20)).isoformat()

        trade_recent = _make_trade(is_buy=True, title="CFO", transaction_date=recent_date)
        trade_old = _make_trade(is_buy=True, title="CFO", transaction_date=old_date)

        result_recent = engine.score("AAPL", insider_trades=[trade_recent])
        result_old = engine.score("AAPL", insider_trades=[trade_old])

        self.assertGreater(result_recent.weighted_score, result_old.weighted_score)


class TestUnusualCallsContribution(unittest.TestCase):
    def test_unusual_calls_positive(self):
        engine = ConsensusEngine()
        result_calls = engine.score("SPY", options_flow="unusual_calls")
        result_normal = engine.score("SPY", options_flow="normal")
        self.assertGreater(result_calls.weighted_score, result_normal.weighted_score)
        self.assertIn("unusual_calls", result_calls.bullish_factors)


class TestFundamentalConfidenceScaling(unittest.TestCase):
    def test_high_confidence_greater_than_low_confidence(self):
        engine = ConsensusEngine()
        result_high = engine.score(
            "MSFT",
            fundamental=_make_fundamental("BULLISH", 0.9),
        )
        result_low = engine.score(
            "MSFT",
            fundamental=_make_fundamental("BULLISH", 0.3),
        )
        self.assertGreater(result_high.weighted_score, result_low.weighted_score)


class TestConvictionThresholds(unittest.TestCase):
    """Drive conviction labels by controlling all inputs to land at known scores."""

    def _score_from_sentiment_only(self, engine: ConsensusEngine, sentiment: float) -> ConsensusSignal:
        # sentiment weight = 0.10; with all other inputs at zero this is the only signal.
        return engine.score("TST", sentiment=sentiment)

    def test_high_conviction(self):
        # Need |score| > 0.5. options_flow=unusual_calls (0.20) + political bullish (0.20)
        # + sentiment 1.0 (0.10) + forecast bullish (0.05) = 0.55 > 0.5
        engine = ConsensusEngine()
        result = engine.score(
            "TST",
            sentiment=1.0,
            trend="bullish",
            options_flow="unusual_calls",
            political_signal="bullish (congressional)",
        )
        self.assertEqual(result.conviction, "high")

    def test_medium_conviction(self):
        # options_flow unusual_calls (0.20) + sentiment 0.6 (0.06) = 0.26 > 0.25
        engine = ConsensusEngine()
        result = engine.score("TST", sentiment=0.6, options_flow="unusual_calls")
        self.assertEqual(result.conviction, "medium")

    def test_low_conviction(self):
        # Only sentiment = 0.1 → score = 0.01 < 0.25
        engine = ConsensusEngine()
        result = engine.score("TST", sentiment=0.1)
        self.assertEqual(result.conviction, "low")


class TestSummaryString(unittest.TestCase):
    def test_summary_contains_ticker_and_score(self):
        engine = ConsensusEngine()
        result = engine.score("GOOG", sentiment=0.5, options_flow="unusual_calls")
        self.assertIn("GOOG", result.summary)
        self.assertIn("score", result.summary)

    def test_summary_direction_neutral_when_zero(self):
        engine = ConsensusEngine()
        result = engine.score("ZZZ")
        self.assertIn("neutral", result.summary)


class TestNeverRaises(unittest.TestCase):
    def test_bad_insider_trade_does_not_raise(self):
        """An insider trade object missing expected attrs should not raise."""
        engine = ConsensusEngine()
        bad_trade = types.SimpleNamespace()  # no attributes at all
        result = engine.score("ERR", insider_trades=[bad_trade])
        self.assertIsInstance(result, ConsensusSignal)

    def test_bad_fundamental_does_not_raise(self):
        engine = ConsensusEngine()
        bad_fund = types.SimpleNamespace()  # no .signal or .confidence
        result = engine.score("ERR", fundamental=bad_fund)
        self.assertIsInstance(result, ConsensusSignal)


if __name__ == "__main__":
    unittest.main()
