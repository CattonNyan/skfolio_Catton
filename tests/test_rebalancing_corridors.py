import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from scripts.crypto_rebalancing_corridors import (
    CorridorBand,
    PolicyResult,
    calculate_proportional_bands,
    calculate_volatility_bands,
    calculate_leland_cost_optimal_bands,
    simulate_corridor_rebalancing,
    generate_synthetic_crypto_prices,
    export_corridor_analysis_json,
    export_corridor_analysis_csv,
    main as corridors_main,
)


class CryptoRebalancingCorridorsTests(unittest.TestCase):
    def setUp(self):
        self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        self.weights = {"BTC/USDT": 0.50, "ETH/USDT": 0.30, "SOL/USDT": 0.20}
        self.vols = {"BTC/USDT": 0.45, "ETH/USDT": 0.65, "SOL/USDT": 0.85}
        self.prices_df = generate_synthetic_crypto_prices(symbols=self.symbols, n_days=90, seed=42)

    def test_calculate_proportional_bands(self):
        bands = calculate_proportional_bands(self.weights, relative_tolerance=0.20)
        self.assertEqual(len(bands), 3)
        for sym, band in bands.items():
            self.assertIsInstance(band, CorridorBand)
            self.assertEqual(band.asset, sym)
            self.assertAlmostEqual(band.target_weight, self.weights[sym], places=4)
            # Width should be 2 * relative_tolerance * target_weight
            expected_width = self.weights[sym] * 0.40
            self.assertAlmostEqual(band.band_width, expected_width, places=4)
            self.assertLess(band.lower_band, band.target_weight)
            self.assertGreater(band.upper_band, band.target_weight)

    def test_calculate_volatility_bands(self):
        bands = calculate_volatility_bands(
            self.weights, asset_volatilities=self.vols, base_cost=0.001, kappa=0.15
        )
        self.assertEqual(len(bands), 3)
        # Higher volatility should produce wider band relative to target weight
        sol_rel_width = bands["SOL/USDT"].band_width / self.weights["SOL/USDT"]
        btc_rel_width = bands["BTC/USDT"].band_width / self.weights["BTC/USDT"]
        self.assertGreater(sol_rel_width, btc_rel_width)

    def test_calculate_leland_cost_optimal_bands(self):
        bands = calculate_leland_cost_optimal_bands(
            self.weights,
            asset_volatilities=self.vols,
            transaction_costs=0.001,
            risk_aversion=3.0,
        )
        self.assertEqual(len(bands), 3)
        for sym, band in bands.items():
            self.assertIsInstance(band, CorridorBand)
            self.assertGreater(band.band_width, 0.0)
            self.assertLessEqual(band.upper_band, 1.0)
            self.assertGreaterEqual(band.lower_band, 0.0)
            d = band.to_dict()
            self.assertIn("lower_band", d)
            self.assertIn("upper_band", d)

    def test_validation_errors(self):
        # Empty weights
        with self.assertRaises(ValueError):
            calculate_proportional_bands({})

        # Weights not summing to 1
        with self.assertRaises(ValueError):
            calculate_proportional_bands({"BTC/USDT": 0.2, "ETH/USDT": 0.2})

        # Negative relative tolerance
        with self.assertRaises(ValueError):
            calculate_proportional_bands(self.weights, relative_tolerance=-0.1)

        # Non-positive kappa
        with self.assertRaises(ValueError):
            calculate_volatility_bands(self.weights, self.vols, kappa=0.0)

        # Non-positive risk aversion
        with self.assertRaises(ValueError):
            calculate_leland_cost_optimal_bands(self.weights, self.vols, risk_aversion=0.0)

    def test_simulate_corridor_rebalancing_all_policies(self):
        bands = calculate_leland_cost_optimal_bands(self.weights, self.vols)
        res = simulate_corridor_rebalancing(
            prices_df=self.prices_df,
            target_weights=self.weights,
            bands=bands,
            fee_rate=0.001,
            slippage_rate=0.0005,
            initial_capital=10000.0,
        )
        self.assertIn("policies", res)
        policies = res["policies"]

        expected_policies = [
            "BuyAndHold",
            "DailyCalendar",
            "WeeklyCalendar",
            "MonthlyCalendar",
            "CorridorToTarget",
            "CorridorToBoundary",
        ]
        for p in expected_policies:
            self.assertIn(p, policies)
            pol = policies[p]
            self.assertIn("final_nav", pol)
            self.assertIn("sharpe_ratio", pol)
            self.assertIn("total_turnover", pol)
            self.assertIn("total_costs_paid", pol)
            self.assertIn("rebalance_count", pol)

        # Buy & Hold should have 0 rebalances
        self.assertEqual(policies["BuyAndHold"]["rebalance_count"], 0)
        self.assertEqual(policies["BuyAndHold"]["total_turnover"], 0.0)

        # Daily rebalancing should have the most rebalances
        self.assertGreaterEqual(
            policies["DailyCalendar"]["rebalance_count"],
            policies["CorridorToTarget"]["rebalance_count"],
        )
        self.assertGreaterEqual(
            policies["DailyCalendar"]["total_turnover"],
            policies["CorridorToTarget"]["total_turnover"],
        )

        # Corridor policy should have positive avoided rebalances %
        self.assertGreater(policies["CorridorToTarget"]["avoided_rebalances_pct"], 0.0)

    def test_export_json_and_csv(self):
        bands = calculate_leland_cost_optimal_bands(self.weights, self.vols)
        res = simulate_corridor_rebalancing(
            prices_df=self.prices_df,
            target_weights=self.weights,
            bands=bands,
            fee_rate=0.001,
            slippage_rate=0.0005,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "corridor_res.json"
            csv_path = Path(tmpdir) / "corridor_res.csv"

            export_corridor_analysis_json(res, json_path)
            self.assertTrue(json_path.exists())
            with open(json_path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIn("policies", loaded)

            export_corridor_analysis_csv(res, csv_path)
            self.assertTrue(csv_path.exists())
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 6)
            self.assertIn("policy_name", df.columns)
            self.assertIn("total_turnover", df.columns)

    def test_synthetic_data_generation(self):
        df = generate_synthetic_crypto_prices(["BTC/USDT", "ETH/USDT"], n_days=50, seed=123)
        self.assertEqual(len(df), 50)
        self.assertEqual(list(df.columns), ["BTC/USDT", "ETH/USDT"])
        self.assertTrue(np.all(df.to_numpy() > 0))

    def test_cli_execution_example(self):
        test_args = ["crypto_rebalancing_corridors.py", "--example", "--days", "30"]
        with patch.object(sys, "argv", test_args):
            corridors_main()


if __name__ == "__main__":
    unittest.main()
