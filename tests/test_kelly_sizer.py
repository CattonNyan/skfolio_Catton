import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from scripts.crypto_kelly_sizer import (
    KellyResult,
    MultiAssetKellyResult,
    calculate_risk_of_ruin,
    calculate_discrete_kelly,
    calculate_continuous_kelly,
    calculate_portfolio_kelly,
    calculate_multi_asset_kelly_allocation,
    export_portfolio_kelly_csv,
    export_multi_asset_kelly_json,
    export_multi_asset_kelly_csv,
    main as kelly_main,
)


class KellySizerTests(unittest.TestCase):
    def test_discrete_kelly_positive_edge(self):
        # 60% win rate, 1.5 payoff ratio
        # f* = (0.6 * 1.5 - 0.4) / 1.5 = (0.9 - 0.4) / 1.5 = 0.5 / 1.5 = 0.3333...
        res = calculate_discrete_kelly(0.60, 1.5, fraction=0.5)
        self.assertTrue(res.is_positive_edge)
        self.assertAlmostEqual(res.full_kelly, 1.0 / 3.0, places=4)
        self.assertAlmostEqual(res.half_kelly, 1.0 / 6.0, places=4)
        self.assertAlmostEqual(res.fractional_kelly, 1.0 / 6.0, places=4)
        self.assertGreater(res.expected_growth_rate, 0.0)

    def test_discrete_kelly_negative_edge(self):
        # 40% win rate, 1.0 payoff ratio (negative expectation)
        res = calculate_discrete_kelly(0.40, 1.0)
        self.assertFalse(res.is_positive_edge)
        self.assertEqual(res.full_kelly, 0.0)
        self.assertEqual(res.fractional_kelly, 0.0)

    def test_continuous_kelly(self):
        # Mean return 0.20 (20%), volatility 0.40 (40%)
        # f* = 0.20 / (0.40^2) = 0.20 / 0.16 = 1.25 -> clamped to max_allocation=1.0
        res = calculate_continuous_kelly(0.20, 0.40, max_allocation=1.0)
        self.assertTrue(res.is_positive_edge)
        self.assertEqual(res.full_kelly, 1.0)
        self.assertAlmostEqual(res.half_kelly, 0.625, places=5)

    def test_portfolio_kelly(self):
        rng = np.random.default_rng(42)
        returns_df = pd.DataFrame({
            "BTC/USDT": rng.normal(0.002, 0.02, 100),
            "ETH/USDT": rng.normal(0.003, 0.03, 100),
        })
        weights = calculate_portfolio_kelly(returns_df, fraction=0.5, max_total_weight=1.0)
        self.assertEqual(len(weights), 2)
        self.assertTrue(all(w >= 0.0 for w in weights))
        self.assertLessEqual(weights.sum(), 1.0)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            calculate_discrete_kelly(1.5, 2.0)
        with self.assertRaises(ValueError):
            calculate_discrete_kelly(0.5, -1.0)
        with self.assertRaises(ValueError):
            calculate_discrete_kelly(0.5, 1.5, max_allocation=0.0)
        with self.assertRaises(ValueError):
            calculate_continuous_kelly(0.1, -0.05)
        with self.assertRaises(ValueError):
            calculate_continuous_kelly(0.1, 0.2, fraction=-0.1)
        with self.assertRaises(ValueError):
            calculate_continuous_kelly(0.1, 0.2, max_allocation=0.0)
        with self.assertRaises(ValueError):
            calculate_portfolio_kelly(pd.DataFrame({"A": [0.01, 0.02]}), fraction=-0.5)
        with self.assertRaises(ValueError):
            calculate_portfolio_kelly(pd.DataFrame({"A": [0.01, 0.02]}), max_total_weight=0.0)

    def test_kelly_result_to_dict(self):
        res = KellyResult(
            full_kelly=0.3333,
            fractional_kelly=0.1667,
            fraction=0.5,
            expected_growth_rate=0.061,
            half_kelly=0.1667,
            is_positive_edge=True,
        )
        d = res.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["full_kelly"], 0.3333)
        self.assertEqual(d["half_kelly"], 0.1667)
        self.assertEqual(d["is_positive_edge"], True)

    def test_cli_and_json_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sub" / "kelly_res.json"
            test_args = [
                "crypto_kelly_sizer.py",
                "--win-rate", "0.60",
                "--payoff", "1.5",
                "--capital", "10000.0",
                "--export-json", str(out_file),
            ]
            with patch("sys.argv", test_args):
                kelly_main()

            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["capital"], 10000.0)
            self.assertIn("full_kelly_dollars", data)
            self.assertIn("half_kelly_dollars", data)
            self.assertEqual(data["is_positive_edge"], True)

    def test_direct_export_kelly_json(self):
        from scripts.crypto_kelly_sizer import export_kelly_json, to_dict_kelly_result
        res = calculate_discrete_kelly(0.60, 1.5, fraction=0.5)
        d = to_dict_kelly_result(res, win_rate=0.60, payoff_ratio=1.5, fraction=0.5, capital=5000.0)
        self.assertEqual(d["capital"], 5000.0)
        self.assertEqual(d["win_rate"], 0.60)
        self.assertIn("fractional_kelly_dollars", d)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "direct_kelly.json"
            export_kelly_json(res, out_file, win_rate=0.60, payoff_ratio=1.5, fraction=0.5, capital=5000.0)
            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["capital"], 5000.0)
            self.assertEqual(loaded["payoff_ratio"], 1.5)

    def test_export_kelly_csv(self):
        from scripts.crypto_kelly_sizer import export_kelly_csv
        res = calculate_discrete_kelly(0.60, 1.5, fraction=0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "kelly.csv"
            export_kelly_csv(res, out_file, win_rate=0.60, payoff_ratio=1.5, fraction=0.5, capital=5000.0)
            self.assertTrue(out_file.exists())
            df = pd.read_csv(out_file)
            self.assertIn("metric", df.columns)
            self.assertIn("value", df.columns)
            metrics = dict(zip(df["metric"], df["value"]))
            self.assertAlmostEqual(float(metrics["capital"]), 5000.0)
            self.assertAlmostEqual(float(metrics["full_kelly"]), 0.3333, places=2)


    def test_calculate_risk_of_ruin(self):
        # Positive edge scenario: 60% win rate, 1.5 payoff ratio, 2% risk, 50% drawdown ruin
        ror_positive = calculate_risk_of_ruin(0.60, 1.5, bet_fraction=0.02, loss_limit=0.5)
        self.assertGreaterEqual(ror_positive, 0.0)
        self.assertLess(ror_positive, 0.05)

        # Negative edge scenario: 40% win rate, 1.0 payoff ratio -> guaranteed ruin 1.0
        ror_negative = calculate_risk_of_ruin(0.40, 1.0, bet_fraction=0.02, loss_limit=0.5)
        self.assertEqual(ror_negative, 1.0)

        # Zero edge scenario: 50% win rate, 1.0 payoff ratio -> ruin 1.0
        ror_zero = calculate_risk_of_ruin(0.50, 1.0, bet_fraction=0.02, loss_limit=0.5)
        self.assertEqual(ror_zero, 1.0)

        # High risk bet fraction should increase probability of ruin
        ror_small_bet = calculate_risk_of_ruin(0.55, 1.2, bet_fraction=0.02, loss_limit=0.5)
        ror_large_bet = calculate_risk_of_ruin(0.55, 1.2, bet_fraction=0.10, loss_limit=0.5)
        self.assertLess(ror_small_bet, ror_large_bet)

        # Boundary & validation checks
        with self.assertRaises(ValueError):
            calculate_risk_of_ruin(0.0, 1.5)
        with self.assertRaises(ValueError):
            calculate_risk_of_ruin(1.0, 1.5)
        with self.assertRaises(ValueError):
            calculate_risk_of_ruin(0.5, 0.0)
        with self.assertRaises(ValueError):
            calculate_risk_of_ruin(0.5, 1.5, bet_fraction=0.0)
        with self.assertRaises(ValueError):
            calculate_risk_of_ruin(0.5, 1.5, loss_limit=0.0)

    def test_discrete_kelly_includes_risk_of_ruin(self):
        res = calculate_discrete_kelly(0.60, 1.5, fraction=0.5, ruin_limit=0.5)
        self.assertTrue(hasattr(res, "risk_of_ruin"))
        self.assertGreaterEqual(res.risk_of_ruin, 0.0)
        self.assertLessEqual(res.risk_of_ruin, 1.0)
        d = res.to_dict()
        self.assertIn("risk_of_ruin", d)
        self.assertEqual(d["risk_of_ruin"], round(res.risk_of_ruin, 4))

    def test_export_portfolio_kelly_csv(self):
        weights = pd.Series([0.45, 0.35, 0.20], index=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "portfolio_kelly.csv"
            saved_path = export_portfolio_kelly_csv(weights, out_file)
            self.assertTrue(saved_path.exists())
            df = pd.read_csv(saved_path)
            self.assertListEqual(list(df.columns), ["asset", "weight", "allocation_pct"])
            self.assertEqual(len(df), 3)
            self.assertEqual(df.iloc[0]["asset"], "BTC/USDT")
            self.assertAlmostEqual(df.iloc[0]["weight"], 0.45)
    def test_calculate_multi_asset_kelly_allocation(self):
        rng = np.random.default_rng(42)
        n_obs = 100
        returns_df = pd.DataFrame({
            "BTC/USDT": rng.normal(0.003, 0.02, n_obs),
            "ETH/USDT": rng.normal(0.0025, 0.03, n_obs),
            "SOL/USDT": rng.normal(0.004, 0.04, n_obs),
        })
        res = calculate_multi_asset_kelly_allocation(
            returns_df=returns_df,
            fraction=0.5,
            shrinkage=0.2,
            max_weight_per_asset=0.5,
            capital=100000.0,
        )
        self.assertIsInstance(res, MultiAssetKellyResult)
        self.assertEqual(res.assets, ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        self.assertEqual(res.fraction, 0.5)
        self.assertEqual(res.capital, 100000.0)
        self.assertIsNotNone(res.position_dollars)

        # Ensure all weights are non-negative and capped
        for a in res.assets:
            self.assertGreaterEqual(res.full_kelly_weights[a], 0.0)
            self.assertLessEqual(res.full_kelly_weights[a], 0.500001)
            # Fractional should be 0.5 * full
            self.assertAlmostEqual(res.fractional_kelly_weights[a], res.full_kelly_weights[a] * 0.5, places=5)
            self.assertAlmostEqual(res.half_kelly_weights[a], res.full_kelly_weights[a] * 0.5, places=5)
            self.assertAlmostEqual(res.quarter_kelly_weights[a], res.full_kelly_weights[a] * 0.25, places=5)
            self.assertAlmostEqual(res.position_dollars[a], res.fractional_kelly_weights[a] * 100000.0, places=2)

        self.assertGreaterEqual(res.portfolio_volatility, 0.0)
        self.assertGreaterEqual(res.drawdown_50_prob, 0.0)
        self.assertLessEqual(res.drawdown_50_prob, 1.0)

        # Test to_dict serialization
        d = res.to_dict()
        self.assertIn("full_kelly_weights", d)
        self.assertIn("fractional_kelly_weights", d)
        self.assertIn("drawdown_50_prob", d)
        self.assertEqual(d["capital"], 100000.0)

    def test_multi_asset_kelly_validation(self):
        rng = np.random.default_rng(42)
        valid_df = pd.DataFrame({
            "BTC/USDT": rng.normal(0.002, 0.02, 50),
            "ETH/USDT": rng.normal(0.002, 0.02, 50),
        })
        # Empty DataFrame
        with self.assertRaises(ValueError):
            calculate_multi_asset_kelly_allocation(pd.DataFrame())
        # Single column
        with self.assertRaises(ValueError):
            calculate_multi_asset_kelly_allocation(pd.DataFrame({"BTC": [0.01, 0.02]}))
        # Invalid shrinkage
        with self.assertRaises(ValueError):
            calculate_multi_asset_kelly_allocation(valid_df, shrinkage=-0.1)
        with self.assertRaises(ValueError):
            calculate_multi_asset_kelly_allocation(valid_df, shrinkage=1.5)
        # Invalid fraction
        with self.assertRaises(ValueError):
            calculate_multi_asset_kelly_allocation(valid_df, fraction=0.0)
        with self.assertRaises(ValueError):
            calculate_multi_asset_kelly_allocation(valid_df, fraction=2.5)
        # Invalid max_weight_per_asset
        with self.assertRaises(ValueError):
            calculate_multi_asset_kelly_allocation(valid_df, max_weight_per_asset=0.0)

    def test_multi_asset_kelly_exports(self):
        rng = np.random.default_rng(42)
        returns_df = pd.DataFrame({
            "BTC/USDT": rng.normal(0.003, 0.02, 60),
            "ETH/USDT": rng.normal(0.002, 0.03, 60),
        })
        res = calculate_multi_asset_kelly_allocation(returns_df, capital=50000.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "multi_kelly.json"
            csv_path = Path(tmpdir) / "multi_kelly.csv"

            export_multi_asset_kelly_json(res, json_path)
            export_multi_asset_kelly_csv(res, csv_path)

            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())

            with open(json_path, "r", encoding="utf-8") as f:
                jdata = json.load(f)
            self.assertIn("full_kelly_weights", jdata)
            self.assertEqual(jdata["capital"], 50000.0)

            df = pd.read_csv(csv_path)
            self.assertIn("asset", df.columns)
            self.assertIn("fractional_kelly_weight", df.columns)
            self.assertIn("position_dollars", df.columns)
            self.assertEqual(len(df), 2)

    def test_cli_multi_asset_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "cli_multi.json"
            csv_path = Path(tmpdir) / "cli_multi.csv"
            test_args = [
                "crypto_kelly_sizer.py",
                "--multi-asset",
                "--fraction", "0.5",
                "--capital", "50000",
                "--export-multi-json", str(json_path),
                "--export-multi-csv", str(csv_path),
            ]
            with patch("sys.argv", test_args):
                kelly_main()

            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            with open(json_path, "r", encoding="utf-8") as f:
                jdata = json.load(f)
            self.assertEqual(jdata["capital"], 50000.0)
            self.assertEqual(jdata["fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()

