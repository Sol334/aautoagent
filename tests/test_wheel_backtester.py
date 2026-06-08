"""Tests for scripts/wheel_backtester.py — fully mocked, no yfinance/network calls."""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Stub heavy dependencies before import
# ---------------------------------------------------------------------------
for _mod in ("torch", "transformers"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Ensure yfinance has a Ticker attribute for patch() to find.
# test_agent_hub.py may have registered an empty stub first, so we always
# add Ticker if it's missing — don't rely on the "not in sys.modules" guard alone.
if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = types.ModuleType("yfinance")
if not hasattr(sys.modules["yfinance"], "Ticker"):
    sys.modules["yfinance"].Ticker = MagicMock()


from scripts.wheel_backtester import backtest_wheel, _render_report  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hist(prices: list[float]):
    """Return a fake history object where hist['Close'] returns a real pandas Series."""
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    idx = pd.date_range("2023-01-01", periods=len(prices), freq="W")
    closes = pd.Series(prices, index=idx)

    class _FakeHist:
        def __len__(self):
            return len(prices)
        def __bool__(self):
            return bool(prices)
        def __getitem__(self, key):
            if key == "Close":
                return closes
            raise KeyError(key)

    return _FakeHist()


def _mock_ticker(prices: list[float]):
    """Return a yf.Ticker mock whose .history() returns _make_hist(prices)."""
    m = MagicMock()
    m.history.return_value = _make_hist(prices)
    return m


# ---------------------------------------------------------------------------
# 1. Returns error dict on insufficient data
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_single_row_returns_error(self):
        with patch("yfinance.Ticker", return_value=_mock_ticker([500.0])):
            result = backtest_wheel("SPY", "2024-01-01", "2024-01-07")
        assert "error" in result

    def test_empty_history_returns_error(self):
        with patch("yfinance.Ticker", return_value=_mock_ticker([])):
            result = backtest_wheel("SPY", "2024-01-01", "2024-06-01")
        assert "error" in result


# ---------------------------------------------------------------------------
# 2. Happy path — deterministic metrics
# ---------------------------------------------------------------------------

class TestHappyPath:
    @pytest.fixture()
    def result(self):
        prices = [500.0 + i * 2 for i in range(30)]  # 30 weeks, steadily rising
        with patch("yfinance.Ticker", return_value=_mock_ticker(prices)):
            return backtest_wheel(
                "SPY",
                "2023-01-01",
                "2023-08-01",
                capital=50_000.0,
                delta_target=0.30,
                iv_override=0.18,
            )

    def test_no_error_key(self, result):
        assert "error" not in result

    def test_ticker_preserved(self, result):
        assert result["ticker"] == "SPY"

    def test_weeks_simulated_correct(self, result):
        # 30 prices → 29 tradeable weeks (last week has no "next" close)
        assert result["weeks_simulated"] == 29

    def test_premium_collected_positive(self, result):
        assert result["total_premium_collected"] > 0

    def test_assignment_rate_in_range(self, result):
        assert 0.0 <= result["assignment_rate"] <= 1.0

    def test_sharpe_ratio_finite(self, result):
        assert isinstance(result["sharpe_ratio"], float)
        assert not (result["sharpe_ratio"] != result["sharpe_ratio"])  # not NaN

    def test_buy_hold_return_correct(self, result):
        # 30 prices: 500 → 558, so buy-hold = (558-500)/500 * 100 = 11.6%
        assert abs(result["buy_hold_return_pct"] - 11.6) < 0.5

    def test_weekly_results_length_matches_weeks_simulated(self, result):
        assert len(result["weekly_results"]) == result["weeks_simulated"]

    def test_each_week_has_required_fields(self, result):
        required = {"week", "price", "strike", "iv", "premium", "assigned", "pnl"}
        for week in result["weekly_results"]:
            assert required.issubset(week.keys()), f"Missing keys in week: {week}"


# ---------------------------------------------------------------------------
# 3. IV override is applied
# ---------------------------------------------------------------------------

class TestIVOverride:
    def test_fixed_iv_used_in_all_weeks(self):
        prices = [400.0] * 25  # flat prices
        with patch("yfinance.Ticker", return_value=_mock_ticker(prices)):
            result = backtest_wheel("QQQ", "2023-01-01", "2023-07-01",
                                    iv_override=0.20)
        assert "error" not in result
        for week in result["weekly_results"]:
            assert abs(week["iv"] - 0.20) < 1e-9


# ---------------------------------------------------------------------------
# 4. Renderer returns string with key labels
# ---------------------------------------------------------------------------

class TestRenderReport:
    def test_error_result_shows_error(self):
        out = _render_report({"error": "insufficient data", "ticker": "SPY"})
        assert "ERROR" in out
        assert "insufficient data" in out

    def test_normal_result_contains_summary_fields(self):
        result = {
            "ticker": "SPY",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "weeks_simulated": 52,
            "total_premium_collected": 1234.56,
            "assignments": 8,
            "assignment_rate": 0.154,
            "total_pnl": 980.0,
            "annualized_return_pct": 12.5,
            "sharpe_ratio": 1.234,
            "max_drawdown_pct": 3.2,
            "buy_hold_return_pct": 24.0,
            "weekly_results": [],
        }
        out = _render_report(result)
        assert "SPY" in out
        assert "1234.56" in out
        assert "12.50" in out
        assert "Sharpe" in out


# ---------------------------------------------------------------------------
# 5. Never raises — returns error dict on exception
# ---------------------------------------------------------------------------

class TestNeverRaises:
    def test_yfinance_exception_returns_error_dict(self):
        with patch("yfinance.Ticker", side_effect=RuntimeError("network error"), create=True):
            result = backtest_wheel("SPY", "2024-01-01", "2024-12-31")
        assert isinstance(result, dict)
        assert "error" in result
        assert "SPY" in str(result)
