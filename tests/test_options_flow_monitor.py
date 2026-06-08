"""Tests for scripts/options_flow_monitor.py — no Polygon API key needed."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestFeatureGate(unittest.TestCase):
    """detect_unusual_flow returns 'normal' when feature or key is absent."""

    def _detect(self, ticker: str = "AAPL", env: dict | None = None) -> str:
        import importlib
        with patch.dict(os.environ, env or {}, clear=False):
            import options_flow_monitor as ofm
            importlib.reload(ofm)
            return ofm.detect_unusual_flow(ticker)

    def test_returns_normal_when_feature_disabled(self):
        result = self._detect(env={"FEATURE_OPTIONS_FLOW": "false", "POLYGON_API_KEY": "somekey"})
        self.assertEqual(result, "normal")

    def test_returns_normal_when_feature_flag_zero(self):
        result = self._detect(env={"FEATURE_OPTIONS_FLOW": "0", "POLYGON_API_KEY": "somekey"})
        self.assertEqual(result, "normal")

    def test_returns_normal_when_polygon_key_missing(self):
        result = self._detect(env={"FEATURE_OPTIONS_FLOW": "true", "POLYGON_API_KEY": ""})
        self.assertEqual(result, "normal")

    def test_returns_normal_when_polygon_key_is_change_me(self):
        result = self._detect(env={"FEATURE_OPTIONS_FLOW": "true", "POLYGON_API_KEY": "CHANGE_ME"})
        self.assertEqual(result, "normal")

    def test_returns_normal_when_api_raises(self):
        import importlib
        with patch.dict(os.environ, {"FEATURE_OPTIONS_FLOW": "true", "POLYGON_API_KEY": "key"}):
            import options_flow_monitor as ofm
            importlib.reload(ofm)
            with patch.object(ofm, "_check_flow", side_effect=Exception("network down")):
                result = ofm.detect_unusual_flow("NVDA")
        self.assertEqual(result, "normal")


class TestCheckFlow(unittest.TestCase):
    """_check_flow logic with mocked Polygon responses."""

    def _run_check(self, ticker: str, contract_type: str, polygon_json: dict) -> bool:
        import importlib
        import httpx
        with patch.dict(os.environ, {"POLYGON_API_KEY": "testkey"}):
            import options_flow_monitor as ofm
            importlib.reload(ofm)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = polygon_json
            mock_resp.raise_for_status = MagicMock()
            with patch("httpx.get", return_value=mock_resp):
                return ofm._check_flow(ticker, contract_type)

    def _otm_contract(self, delta: float, volume: int, open_interest: int) -> dict:
        return {
            "greeks": {"delta": delta},
            "day": {"volume": volume},
            "open_interest": open_interest,
        }

    def test_unusual_calls_detected(self):
        # vol/OI = 600/100 = 6.0 > 2.0, total_vol=600 > typical_daily*(3.0) = 10*3=30
        contract = self._otm_contract(delta=0.20, volume=600, open_interest=100)
        result = self._run_check("NVDA", "call", {"results": [contract]})
        self.assertTrue(result)

    def test_normal_when_vol_oi_ratio_below_threshold(self):
        # vol/OI = 1.5 < 2.0 threshold
        contract = self._otm_contract(delta=0.20, volume=150, open_interest=100)
        result = self._run_check("AAPL", "call", {"results": [contract]})
        self.assertFalse(result)

    def test_returns_false_on_empty_results(self):
        result = self._run_check("TSLA", "put", {"results": []})
        self.assertFalse(result)

    def test_returns_false_when_no_otm_contracts_in_delta_range(self):
        # delta=0.60 is ITM, outside the 0.15–0.30 OTM window
        contract = self._otm_contract(delta=0.60, volume=999, open_interest=10)
        result = self._run_check("AMZN", "call", {"results": [contract]})
        self.assertFalse(result)

    def test_returns_false_when_open_interest_zero(self):
        contract = self._otm_contract(delta=0.20, volume=500, open_interest=0)
        result = self._run_check("MSFT", "call", {"results": [contract]})
        self.assertFalse(result)

    def test_polygon_403_returns_false(self):
        import importlib
        import httpx
        with patch.dict(os.environ, {"POLYGON_API_KEY": "testkey"}):
            import options_flow_monitor as ofm
            importlib.reload(ofm)
            mock_resp = MagicMock()
            mock_resp.status_code = 403
            mock_resp.raise_for_status = MagicMock()
            with patch("httpx.get", return_value=mock_resp):
                result = ofm._check_flow("SPY", "call")
        self.assertFalse(result)


class TestDetectUnusualFlowEndToEnd(unittest.TestCase):
    """detect_unusual_flow integration: feature enabled, Polygon mocked."""

    def _detect_mocked(self, call_result: bool, put_result: bool) -> str:
        import importlib
        with patch.dict(os.environ, {"FEATURE_OPTIONS_FLOW": "true", "POLYGON_API_KEY": "testkey"}):
            import options_flow_monitor as ofm
            importlib.reload(ofm)
            side_effects = {"call": call_result, "put": put_result}
            with patch.object(ofm, "_check_flow", side_effect=lambda t, ct: side_effects[ct]):
                return ofm.detect_unusual_flow("AAPL")

    def test_returns_unusual_calls_when_call_signal_fires(self):
        self.assertEqual(self._detect_mocked(call_result=True, put_result=False), "unusual_calls")

    def test_returns_unusual_puts_when_only_put_fires(self):
        self.assertEqual(self._detect_mocked(call_result=False, put_result=True), "unusual_puts")

    def test_returns_normal_when_no_signal(self):
        self.assertEqual(self._detect_mocked(call_result=False, put_result=False), "normal")

    def test_calls_take_priority_over_puts(self):
        self.assertEqual(self._detect_mocked(call_result=True, put_result=True), "unusual_calls")


class TestRunMonitor(unittest.TestCase):
    def test_dry_run_returns_normal_for_all(self):
        import importlib
        import options_flow_monitor as ofm
        importlib.reload(ofm)
        results = ofm.run_monitor(["AAPL", "NVDA"], dry_run=True)
        self.assertEqual(results["AAPL"], "normal")
        self.assertEqual(results["NVDA"], "normal")

    def test_live_run_calls_detect_for_each_ticker(self):
        import importlib
        with patch.dict(os.environ, {"FEATURE_OPTIONS_FLOW": "false"}):
            import options_flow_monitor as ofm
            importlib.reload(ofm)
            with patch.object(ofm, "detect_unusual_flow", return_value="normal") as mock_detect:
                results = ofm.run_monitor(["TSLA", "SPY"], dry_run=False)
            self.assertEqual(mock_detect.call_count, 2)
            self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
