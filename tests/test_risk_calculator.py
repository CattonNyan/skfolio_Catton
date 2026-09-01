"""Tests for volatility-based risk budget calculator module."""

import unittest
import tempfile
import json
from pathlib import Path
import pandas as pd

from scripts.crypto_portfolio_optimizer import generate_synthetic_crypto_data
from scripts.crypto_risk_budget_calculator import (
    calculate_volatility_metrics,
    compute_risk_guidelines,
    export_risk_json,
)


class RiskCalculatorTests(unittest.TestCase):
    def test_calculate_volatility_metrics(self):
        prices = generate_synthetic_crypto_data(periods=50)
        df = calculate_volatility_metrics(prices)
        self.assertIn("periodic_vol", df.columns)
        self.assertIn("semi_dev", df.columns)
        self.assertTrue((df["semi_dev"] > 0).all())

    def test_compute_risk_guidelines(self):
        prices = generate_synthetic_crypto_data(periods=50)
        weights = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        guidelines = compute_risk_guidelines(prices, weights=weights, risk_reward_ratio=2.0)
        self.assertIn("BTC/USDT", guidelines)
        g = guidelines["BTC/USDT"]
        self.assertLess(g["recommended_stoploss"], 0.0)
        self.assertGreater(g["recommended_take_profit"], 0.0)
        # Take-profit should be roughly 2x stoploss
        self.assertAlmostEqual(g["recommended_take_profit"], abs(g["recommended_stoploss"]) * 2.0, places=3)

    def test_export_risk_json(self):
        sample = {
            "BTC/USDT": {"recommended_stoploss": -0.04, "recommended_take_profit": 0.08, "weight": 0.5, "semi_dev": 1.2, "risk_reward_ratio": 2.0}
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "risk_test.json"
            export_risk_json(sample, out_file)
            self.assertTrue(out_file.is_file())
            data = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertIn("freqtrade_stoploss_config", data)
            self.assertEqual(data["freqtrade_stoploss_config"]["BTC/USDT"], -0.04)


if __name__ == "__main__":
    unittest.main()
