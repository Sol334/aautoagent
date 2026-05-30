"""Tests for regime_detector.py — market regime classification."""
import sys
import math
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import regime_detector as rd


class TestExtractFeatures:
    def test_requires_21_prices(self):
        assert rd.extract_features([100.0] * 20) == []

    def test_returns_correct_length(self):
        prices = [100.0 + i * 0.5 for i in range(30)]
        features = rd.extract_features(prices)
        assert len(features) == 29  # len(prices) - 1 (returns)

    def test_features_have_required_keys(self):
        prices = [100.0 + i for i in range(30)]
        features = rd.extract_features(prices)
        assert "return" in features[0]
        assert "vol_ann" in features[0]
        assert "momentum" in features[0]

    def test_uptrend_positive_return(self):
        prices = [100.0 + i for i in range(30)]
        features = rd.extract_features(prices)
        assert features[-1]["return"] > 0

    def test_downtrend_negative_return(self):
        prices = [200.0 - i for i in range(30)]
        features = rd.extract_features(prices)
        assert features[-1]["return"] < 0


class TestClassifyRegime:
    def test_empty_features_returns_mean_reverting(self):
        assert rd.classify_regime([]) == rd.MEAN_REVERTING

    def _make_features(self, ret, vol, mom, n=20):
        return [{"return": ret, "vol_ann": vol, "momentum": mom}] * n

    def test_high_vol_chaos(self):
        features = self._make_features(0.001, 0.50, 0.05)
        assert rd.classify_regime(features) == rd.HIGH_VOL_CHAOS

    def test_trending_bull(self):
        features = self._make_features(0.001, 0.15, 0.05)
        assert rd.classify_regime(features) == rd.TRENDING_BULL

    def test_trending_bear(self):
        features = self._make_features(-0.001, 0.15, -0.05)
        assert rd.classify_regime(features) == rd.TRENDING_BEAR

    def test_mean_reverting(self):
        features = self._make_features(0.0, 0.15, 0.0)
        assert rd.classify_regime(features) == rd.MEAN_REVERTING

    def test_vol_40_exactly_is_chaos(self):
        # 0.41 should be chaos
        features = self._make_features(0.001, 0.41, 0.05)
        assert rd.classify_regime(features) == rd.HIGH_VOL_CHAOS


class TestDetectRegime:
    def test_no_prices_returns_mean_reverting_with_error(self):
        with patch.object(rd, "fetch_prices", return_value=[]):
            result = rd.detect_regime("SPY")
        assert result["regime"] == rd.MEAN_REVERTING
        assert result["error"] == "no_price_data"

    def test_bull_market_returns_bull(self):
        prices = [100.0 + i * 0.8 for i in range(100)]  # steady uptrend
        with patch.object(rd, "fetch_prices", return_value=prices):
            result = rd.detect_regime("SPY")
        assert result["regime"] == rd.TRENDING_BULL
        assert result["label"] == "TRENDING_BULL"

    def test_crash_returns_chaos(self):
        # Alternating ±5% = extreme volatility
        prices = []
        p = 100.0
        for i in range(100):
            p *= 1.05 if i % 2 == 0 else 0.95
            prices.append(p)
        with patch.object(rd, "fetch_prices", return_value=prices):
            result = rd.detect_regime("SPY")
        assert result["regime"] in (rd.HIGH_VOL_CHAOS, rd.MEAN_REVERTING)

    def test_result_has_all_keys(self):
        prices = [100.0 + i for i in range(60)]
        with patch.object(rd, "fetch_prices", return_value=prices):
            result = rd.detect_regime("SPY")
        for key in ("regime", "label", "action", "confidence", "ticker"):
            assert key in result


class TestShouldTrade:
    def test_bull_regime_allows_trading(self):
        with patch.object(rd, "detect_regime", return_value={"regime": rd.TRENDING_BULL, "label": "TRENDING_BULL"}):
            assert rd.should_trade() is True

    def test_chaos_blocks_trading(self):
        with patch.object(rd, "detect_regime", return_value={"regime": rd.HIGH_VOL_CHAOS, "label": "HIGH_VOL_CHAOS"}):
            assert rd.should_trade() is False

    def test_bear_blocks_trading(self):
        with patch.object(rd, "detect_regime", return_value={"regime": rd.TRENDING_BEAR, "label": "TRENDING_BEAR"}):
            assert rd.should_trade() is False


class TestHelpers:
    def test_returns_calculation(self):
        prices = [100.0, 110.0, 99.0]
        rets = rd._returns(prices)
        assert len(rets) == 2
        assert abs(rets[0] - 0.10) < 0.001

    def test_rolling_vol_length(self):
        rets = [0.01 * (i % 3 - 1) for i in range(30)]
        vols = rd._rolling_vol(rets)
        assert len(vols) == 30

    def test_momentum_first_window_is_zero(self):
        prices = [100.0 + i for i in range(30)]
        moms = rd._momentum(prices, window=20)
        assert moms[0] == 0.0
        assert moms[19] == 0.0
        assert moms[20] != 0.0

    def test_regime_labels_all_defined(self):
        for regime in (rd.TRENDING_BULL, rd.TRENDING_BEAR, rd.MEAN_REVERTING, rd.HIGH_VOL_CHAOS):
            assert regime in rd._REGIME_LABELS
            assert regime in rd._REGIME_ACTIONS
