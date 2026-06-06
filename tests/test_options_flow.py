"""Tests for galactic-capital/scripts/options_flow_monitor.py"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAPITAL_ROOT))
sys.path.insert(0, str(CAPITAL_ROOT / "scripts"))


class TestOptionsFlowMonitorFeatureGate(unittest.TestCase):
    def test_returns_normal_when_feature_disabled(self):
        import importlib, os
        with patch.dict(os.environ, {
            "FEATURE_OPTIONS_FLOW": "false",
            "POLYGON_API_KEY": "real_key",
        }):
            import options_flow_monitor as ofm
            importlib.reload(ofm)
            result = ofm.detect_unusual_flow("NVDA")
        self.assertEqual(result, "normal")

    def test_returns_normal_when_no_polygon_key(self):
        import importlib, os
        with patch.dict(os.environ, {
            "FEATURE_OPTIONS_FLOW": "true",
            "POLYGON_API_KEY": "CHANGE_ME",
        }):
            import options_flow_monitor as ofm
            importlib.reload(ofm)
            result = ofm.detect_unusual_flow("NVDA")
        self.assertEqual(result, "normal")


class TestOptionsFlowMonitorSignals(unittest.TestCase):
    def _make_chain_result(self, volume: int, oi: int, delta: float = 0.22):
        return [
            {
                "greeks": {"delta": delta},
                "day": {"volume": volume},
                "open_interest": oi,
            }
        ]

    def test_unusual_calls_detected(self):
        import importlib, os, json
        payload = {"results": self._make_chain_result(volume=5000, oi=500)}

        class FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return payload

        with patch.dict(os.environ, {
            "FEATURE_OPTIONS_FLOW": "true",
            "POLYGON_API_KEY": "test_key",
        }):
            import options_flow_monitor as ofm
            importlib.reload(ofm)
            with patch("httpx.get", return_value=FakeResp()):
                result = ofm.detect_unusual_flow("NVDA")
        self.assertEqual(result, "unusual_calls")

    def test_normal_on_low_volume(self):
        import importlib, os, json
        payload = {"results": self._make_chain_result(volume=100, oi=10000)}

        class FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return payload

        with patch.dict(os.environ, {
            "FEATURE_OPTIONS_FLOW": "true",
            "POLYGON_API_KEY": "test_key",
        }):
            import options_flow_monitor as ofm
            importlib.reload(ofm)
            with patch("httpx.get", return_value=FakeResp()):
                result = ofm.detect_unusual_flow("NVDA")
        self.assertEqual(result, "normal")

    def test_dry_run_returns_normal_without_api_call(self):
        import importlib, os
        with patch.dict(os.environ, {
            "FEATURE_OPTIONS_FLOW": "true",
            "POLYGON_API_KEY": "test_key",
            "CAPITAL_WATCHLIST": "AAPL",
        }):
            import options_flow_monitor as ofm
            importlib.reload(ofm)
            with patch("httpx.get") as mock_get:
                results = ofm.run_monitor(["AAPL"], dry_run=True)
                mock_get.assert_not_called()
        self.assertEqual(results["AAPL"], "normal")


if __name__ == "__main__":
    unittest.main()
