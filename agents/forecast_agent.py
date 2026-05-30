#!/usr/bin/env python3
"""
galactic-capital/agents/forecast_agent.py

Price trend forecasting using IBM Granite TTM-R2 (ibm-granite/granite-timeseries-ttm-r2,
~18 MB, CPU-compatible). Returns "up", "down", or "flat".

Falls back to a 20-day SMA slope calculation if tsfm-public is not installed or the
model fails — the SMA fallback has zero additional dependencies.
"""

import logging
import sys
from pathlib import Path

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_ROOT = CAPITAL_ROOT  # repo root (was galactic-system parent before extraction)
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(CAPITAL_ROOT))

log = logging.getLogger(__name__)

_GRANITE_MODEL = "ibm-granite/granite-timeseries-ttm-r2"
_TREND_THRESHOLD = 0.01  # >1% predicted change counts as directional


class ForecastAgent:
    def __init__(self):
        self._model = None
        self._load_attempted = False

    def _get_model(self):
        if self._load_attempted:
            return self._model
        self._load_attempted = True
        try:
            from tsfm_public import TinyTimeMixer
            self._model = TinyTimeMixer.from_pretrained(_GRANITE_MODEL)
            self._model.eval()
            log.info("ForecastAgent: loaded %s", _GRANITE_MODEL)
        except Exception as exc:
            log.warning(
                "ForecastAgent: tsfm-public unavailable, using SMA fallback (%s)", exc
            )
            self._model = None
        return self._model

    def _sma_fallback(self, prices: list) -> str:
        """20-day SMA slope as a zero-dependency fallback trend indicator."""
        if len(prices) < 20:
            return "flat"
        recent = list(prices[-20:])
        first_half = sum(recent[:10]) / 10
        second_half = sum(recent[10:]) / 10
        if first_half == 0:
            return "flat"
        change = (second_half - first_half) / first_half
        if change > _TREND_THRESHOLD:
            return "up"
        if change < -_TREND_THRESHOLD:
            return "down"
        return "flat"

    def predict_trend(self, prices: list) -> str:
        """Predict price trend from closing prices. Returns 'up', 'down', or 'flat'."""
        if len(prices) < 2:
            return "flat"

        model = self._get_model()
        if model is None:
            return self._sma_fallback(prices)

        try:
            import torch

            target_len = 512
            seq = list(prices[-target_len:])
            if len(seq) < target_len:
                seq = [seq[0]] * (target_len - len(seq)) + seq

            # Shape: [batch=1, seq_len=512, n_vars=1]
            arr = torch.tensor([[v] for v in seq], dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(arr)

            if hasattr(output, "prediction_outputs"):
                forecast = output.prediction_outputs[0, 0, 0].item()
            elif isinstance(output, (list, tuple)):
                forecast = float(output[0].flatten()[0])
            else:
                forecast = float(output.flatten()[0])

            last = prices[-1]
            if last == 0:
                return "flat"
            change = (forecast - last) / last
            if change > _TREND_THRESHOLD:
                return "up"
            if change < -_TREND_THRESHOLD:
                return "down"
            return "flat"
        except Exception as exc:
            log.warning("ForecastAgent.predict_trend error, falling back to SMA: %s", exc)
            return self._sma_fallback(prices)

    def predict_gated(self, prices: list, ticker: str = "SPY") -> dict:
        """Regime-gated prediction — returns trend + regime + actionable bool.

        Only recommends action in TRENDING_BULL regime to avoid whipsaw losses
        in mean-reverting or high-volatility market conditions.
        """
        try:
            from regime_detector import detect_regime, TRENDING_BULL
            regime_result = detect_regime(ticker)
        except Exception:
            regime_result = {"regime": -1, "label": "UNKNOWN", "action": "unknown", "confidence": 0.0}
            TRENDING_BULL = 0

        trend = self.predict_trend(prices)
        tradeable = regime_result.get("regime") == TRENDING_BULL and trend == "up"
        return {
            "trend":     trend,
            "regime":    regime_result.get("label", "UNKNOWN"),
            "tradeable": tradeable,
            "reason":    regime_result.get("action", ""),
        }
