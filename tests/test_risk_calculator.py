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
    update_freqtrade_risk_config,
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

    def test_non_positive_risk_parameters_are_rejected(self):
        prices = generate_synthetic_crypto_data(periods=50)
        for bad_val in (0, -1, True, False, float("nan"), float("inf")):
            with self.subTest(bad_multiplier=bad_val), self.assertRaises(ValueError):
                compute_risk_guidelines(prices, risk_multiplier=bad_val)
            with self.subTest(bad_ratio=bad_val), self.assertRaises(ValueError):
                compute_risk_guidelines(prices, risk_reward_ratio=bad_val)

        for bad_weights in ("not_a_dict", [1, 2], {"BTC": -0.5}, {"BTC": True}, {"BTC": float("nan")}):
            with self.subTest(bad_weights=bad_weights), self.assertRaises(ValueError):
                compute_risk_guidelines(prices, weights=bad_weights)

        for bad_prices in ("not_a_df", pd.DataFrame(), pd.DataFrame({"A": [-1.0, 2.0]})):
            with self.subTest(bad_prices=bad_prices), self.assertRaises(ValueError):
                calculate_volatility_metrics(bad_prices)

    def test_export_risk_json(self):
        sample = {
            "BTC/USDT": {"recommended_stoploss": -0.04, "recommended_take_profit": 0.08, "weight": 0.5, "semi_dev": 1.2, "risk_reward_ratio": 2.0}
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "risk_test.json"
            export_risk_json(sample, out_file)
            self.assertTrue(out_file.is_file())
            data = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(data["assets"]["BTC/USDT"]["recommended_stoploss"], -0.04)

    def test_update_freqtrade_risk_config(self):
        sample = {
            "BTC/USDT": {
                "recommended_stoploss": -0.04,
                "recommended_take_profit": 0.08,
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"exchange": {"name": "binance"}}), encoding="utf-8")
            self.assertTrue(update_freqtrade_risk_config(sample, config_path, "real-test-data"))
            data = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(data["pair_risk_limits"]["BTC/USDT"]["recommended_take_profit"], 0.08)
            self.assertEqual(data["skfolio_risk"]["data_source"], "real-test-data")

    def test_synthetic_risk_config_update_is_rejected(self):
        sample = {
            "BTC/USDT": {
                "recommended_stoploss": -0.04,
                "recommended_take_profit": 0.08,
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            original = json.dumps({"exchange": {"name": "binance"}})
            config_path.write_text(original, encoding="utf-8")
            self.assertFalse(update_freqtrade_risk_config(sample, config_path, "synthetic"))
            self.assertEqual(config_path.read_text(encoding="utf-8"), original)

    def test_effective_number_of_constituents(self):
        from scripts.crypto_risk_budget_calculator import calculate_effective_number_of_constituents
        # 4 equal assets: ENC should be 4.0
        enc_eq = calculate_effective_number_of_constituents([0.25, 0.25, 0.25, 0.25])
        self.assertAlmostEqual(enc_eq, 4.0, places=2)

        # Concentrated asset: ENC should be 1.0
        enc_conc = calculate_effective_number_of_constituents([1.0, 0.0, 0.0])
        self.assertAlmostEqual(enc_conc, 1.0, places=2)

    def test_effective_number_of_bets(self):
        from scripts.crypto_risk_budget_calculator import calculate_effective_number_of_bets
        import numpy as np
        # Diagonal uncorrelated covariance
        cov = np.diag([0.04, 0.04, 0.04])
        # Equal weights in equal risk uncorrelated assets -> ENB should be 3.0
        res = calculate_effective_number_of_bets([1/3, 1/3, 1/3], cov)
        self.assertAlmostEqual(res["enb_entropy"], 3.0, places=1)
        self.assertAlmostEqual(res["enb_herfindahl"], 3.0, places=1)

        # Unequal risk: one hyper-volatile asset
        cov_skew = np.diag([0.01, 0.01, 1.0])
        res_skew = calculate_effective_number_of_bets([1/3, 1/3, 1/3], cov_skew)
        # ENB should be significantly less than 3 because asset 3 dominates risk
        self.assertLess(res_skew["enb_entropy"], 2.0)

    def test_to_dict_risk_budget_result(self):
        from scripts.crypto_risk_budget_calculator import to_dict_risk_budget_result
        sample = {
            "BTC/USDT": {"recommended_stoploss": -0.04, "recommended_take_profit": 0.08, "weight": 0.5, "semi_dev": 1.2, "risk_reward_ratio": 2.0}
        }
        res = to_dict_risk_budget_result(sample, data_source="unit-test")
        self.assertEqual(res["data_source"], "unit-test")
        self.assertIn("assets", res)
        self.assertEqual(res["assets"]["BTC/USDT"]["recommended_stoploss"], -0.04)
        self.assertIn("generated_at", res)


if __name__ == "__main__":
    unittest.main()

