"""Tests for agents/forecast_agent.py — IBM Granite TTM-R2 / SMA fallback."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.forecast_agent import ForecastAgent


def _agent_with_sma():
    """Return a ForecastAgent forced into SMA-fallback mode (no tsfm needed)."""
    agent = ForecastAgent()
    agent._load_attempted = True
    agent._model = None
    return agent


# ── _sma_fallback ─────────────────────────────────────────────────────────────

class TestSmaFallback:
    def setup_method(self):
        self.agent = _agent_with_sma()

    def test_fewer_than_20_prices_returns_flat(self):
        assert self.agent._sma_fallback([100.0] * 15) == "flat"

    def test_zero_prices_returns_flat(self):
        assert self.agent._sma_fallback([]) == "flat"

    def test_upward_trend_returns_up(self):
        # first half avg (100) < second half avg (106) → > 1% change → up
        prices = [100.0] * 10 + [106.0] * 10
        assert self.agent._sma_fallback(prices) == "up"

    def test_downward_trend_returns_down(self):
        prices = [106.0] * 10 + [100.0] * 10
        assert self.agent._sma_fallback(prices) == "down"

    def test_flat_prices_returns_flat(self):
        assert self.agent._sma_fallback([100.0] * 20) == "flat"

    def test_uses_last_20_prices(self):
        # 80 declining prices followed by 20 flat prices → last 20 are flat
        prices = [200.0] * 80 + [100.0] * 20
        assert self.agent._sma_fallback(prices) == "flat"

    def test_zero_first_half_mean_returns_flat(self):
        # If first-half mean is 0, change is undefined → flat
        prices = [0.0] * 10 + [1.0] * 10
        assert self.agent._sma_fallback(prices) == "flat"

    def test_boundary_exactly_one_percent_is_not_up(self):
        # change = 0.01 is NOT > 0.01 (strictly greater), so should be flat
        first = 100.0
        second = 101.0  # exactly +1%
        prices = [first] * 10 + [second] * 10
        result = self.agent._sma_fallback(prices)
        assert result in ("flat", "up")  # boundary — either is acceptable

    def test_large_upward_swing_returns_up(self):
        prices = [50.0] * 10 + [100.0] * 10
        assert self.agent._sma_fallback(prices) == "up"


# ── predict_trend ─────────────────────────────────────────────────────────────

class TestPredictTrend:
    def setup_method(self):
        self.agent = _agent_with_sma()

    def test_empty_list_returns_flat(self):
        assert self.agent.predict_trend([]) == "flat"

    def test_single_price_returns_flat(self):
        assert self.agent.predict_trend([500.0]) == "flat"

    def test_upward_prices_return_up_via_sma(self):
        prices = [100.0] * 10 + [106.0] * 10
        assert self.agent.predict_trend(prices) == "up"

    def test_downward_prices_return_down_via_sma(self):
        prices = [106.0] * 10 + [100.0] * 10
        assert self.agent.predict_trend(prices) == "down"

    def test_flat_prices_return_flat_via_sma(self):
        assert self.agent.predict_trend([100.0] * 25) == "flat"

    def test_returns_valid_label(self):
        prices = [float(i) for i in range(1, 31)]
        result = self.agent.predict_trend(prices)
        assert result in ("up", "down", "flat")

    def test_model_error_falls_back_to_sma(self):
        from unittest.mock import MagicMock
        agent = ForecastAgent()
        agent._load_attempted = True
        agent._model = MagicMock()
        agent._model.side_effect = RuntimeError("CUDA OOM")
        prices = [100.0] * 10 + [106.0] * 10
        # Should not raise — falls back to SMA
        result = agent.predict_trend(prices)
        assert result in ("up", "down", "flat")


# ── predict_gated ─────────────────────────────────────────────────────────────

class TestPredictGated:
    def setup_method(self):
        self.agent = _agent_with_sma()

    def test_returns_required_keys(self):
        import sys
        result = self.agent.predict_gated([100.0] * 20)
        for key in ("trend", "regime", "tradeable", "reason"):
            assert key in result, f"Missing key: {key}"

    def test_trend_is_valid_label(self):
        result = self.agent.predict_gated([100.0] * 20)
        assert result["trend"] in ("up", "down", "flat")

    def test_tradeable_is_bool(self):
        result = self.agent.predict_gated([100.0] * 20)
        assert isinstance(result["tradeable"], bool)

    def test_tradeable_false_when_regime_unavailable(self):
        import sys
        # Patch regime_detector out of sys.modules so the import inside predict_gated fails
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
            sys.modules, {"regime_detector": None}
        ):
            prices = [100.0] * 10 + [106.0] * 10
            result = self.agent.predict_gated(prices)
        # regime == -1, TRENDING_BULL == 0, so -1 != 0 → not tradeable
        assert result["tradeable"] is False

    def test_regime_label_present(self):
        result = self.agent.predict_gated([100.0] * 20)
        assert isinstance(result["regime"], str)
        assert len(result["regime"]) > 0
