"""Tests for galactic-capital/agents/macro_agent.py"""

import sys
import unittest
from pathlib import Path

CAPITAL_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_ROOT = CAPITAL_ROOT  # repo root (was galactic-system parent before extraction)
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(CAPITAL_ROOT))
sys.path.insert(0, str(CAPITAL_ROOT / "agents"))


class TestMacroAgentRegime(unittest.TestCase):
    def _make_agent(self, spread: float | None, vix: float | None):
        import importlib
        import macro_agent as ma_mod
        importlib.reload(ma_mod)
        from macro_agent import MacroAgent
        agent = MacroAgent()
        agent._get_yield_curve_spread = lambda: spread
        agent._get_vix = lambda: vix
        return agent

    def test_risk_off_when_inverted_and_high_vix(self):
        agent = self._make_agent(spread=-0.5, vix=28.0)
        self.assertEqual(agent.get_regime(), "risk_off")

    def test_neutral_when_inverted_but_low_vix(self):
        agent = self._make_agent(spread=-0.3, vix=18.0)
        self.assertEqual(agent.get_regime(), "neutral")

    def test_neutral_when_high_vix_but_not_inverted(self):
        agent = self._make_agent(spread=0.5, vix=30.0)
        self.assertEqual(agent.get_regime(), "neutral")

    def test_risk_on_when_normal_spread_and_low_vix(self):
        agent = self._make_agent(spread=1.2, vix=15.0)
        self.assertEqual(agent.get_regime(), "risk_on")

    def test_neutral_when_data_unavailable(self):
        agent = self._make_agent(spread=None, vix=None)
        self.assertEqual(agent.get_regime(), "neutral")

    def test_neutral_when_only_spread_unavailable(self):
        agent = self._make_agent(spread=None, vix=30.0)
        self.assertEqual(agent.get_regime(), "neutral")


class TestMacroAgentContext(unittest.TestCase):
    def test_context_includes_regime(self):
        import importlib
        import macro_agent as ma_mod
        importlib.reload(ma_mod)
        from macro_agent import MacroAgent
        agent = MacroAgent()
        agent._get_yield_curve_spread = lambda: 1.0
        agent._get_vix = lambda: 14.0
        ctx = agent.get_macro_context()
        self.assertIn("risk_on", ctx)

    def test_context_includes_vix_value(self):
        import importlib
        import macro_agent as ma_mod
        importlib.reload(ma_mod)
        from macro_agent import MacroAgent
        agent = MacroAgent()
        agent._get_yield_curve_spread = lambda: -0.2
        agent._get_vix = lambda: 27.5
        ctx = agent.get_macro_context()
        self.assertIn("27.5", ctx)


if __name__ == "__main__":
    unittest.main()
