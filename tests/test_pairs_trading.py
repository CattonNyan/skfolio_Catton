import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from scripts.crypto_pairs_trading import (
    CointegrationResult,
    PairsTradingSimulationResult,
    calculate_cointegration,
    calculate_rolling_zscore,
    simulate_pairs_trading,
    scan_crypto_pairs,
    export_pairs_trading_json,
    export_pairs_trading_csv,
    generate_synthetic_pairs_data,
    main as pairs_main,
)


class CryptoPairsTradingTests(unittest.TestCase):
    def setUp(self):
        self.y, self.x = generate_synthetic_pairs_data(n_bars=400, seed=42)

    def test_calculate_cointegration_synthetic(self):
        coint = calculate_cointegration(self.y, self.x, asset_y_name="ETH/USDT", asset_x_name="BTC/USDT")
        self.assertIsInstance(coint, CointegrationResult)
        self.assertEqual(coint.asset_y, "ETH/USDT")
        self.assertEqual(coint.asset_x, "BTC/USDT")
        self.assertGreater(coint.hedge_ratio_beta, 0.0)
        self.assertGreater(coint.r_squared, 0.50)
        self.assertLess(coint.adf_statistic, coint.critical_values["5%"])
        self.assertTrue(coint.is_cointegrated)
        self.assertGreater(coint.half_life_bars, 0.0)
        self.assertLess(coint.half_life_bars, 100.0)

        # Dataclass serialization
        d = coint.to_dict()
        self.assertIn("hedge_ratio_beta", d)
        self.assertIn("adf_statistic", d)
        self.assertIn("half_life_bars", d)

    def test_calculate_cointegration_uncorrelated(self):
        rng = np.random.default_rng(999)
        # Two completely independent random walks
        rw1 = np.cumprod(1.0 + rng.normal(0.0001, 0.02, 300)) * 100.0
        rw2 = np.cumprod(1.0 + rng.normal(0.0001, 0.02, 300)) * 50.0

        coint = calculate_cointegration(rw1, rw2)
        # Should not be strongly cointegrated (or have much weaker ADF stat)
        self.assertIsInstance(coint.adf_statistic, float)

    def test_calculate_rolling_zscore(self):
        spread = pd.Series(self.y - 0.06 * self.x)
        z = calculate_rolling_zscore(spread, window=20)
        self.assertEqual(len(z), len(spread))
        self.assertTrue(np.all(np.isfinite(z)))
        # Mean of Z-score should be around 0
        self.assertAlmostEqual(float(z.iloc[30:].mean()), 0.0, delta=0.5)

    def test_simulate_pairs_trading_execution(self):
        sim = simulate_pairs_trading(
            prices_y=self.y,
            prices_x=self.x,
            entry_z=1.5,
            exit_z=0.5,
            stop_z=3.5,
            rolling_window=25,
            fee_rate=0.0005,
            initial_capital=10000.0,
        )

        self.assertIn("metrics", sim)
        self.assertIn("trades", sim)
        self.assertIn("nav_history", sim)
        self.assertIn("cointegration", sim)

        m = sim["metrics"]
        self.assertGreater(m["total_trades"], 0)
        self.assertGreater(m["capital_final"], 0.0)
        self.assertGreaterEqual(m["win_rate_pct"], 0.0)
        self.assertLessEqual(m["win_rate_pct"], 100.0)
        self.assertGreaterEqual(m["max_drawdown_pct"], 0.0)
        self.assertEqual(m["capital_initial"], 10000.0)

    def test_scan_crypto_pairs(self):
        rng = np.random.default_rng(123)
        base = np.cumprod(1.0 + rng.normal(0.0005, 0.02, 100))
        df = pd.DataFrame({
            "BTC": base * 60000.0,
            "ETH": base * 3000.0 * (1.0 + rng.normal(0, 0.01, 100)),
            "SOL": base * 150.0 * (1.0 + rng.normal(0, 0.02, 100)),
        })
        scan = scan_crypto_pairs(df, min_r2=0.10)
        self.assertGreater(len(scan), 0)
        for item in scan:
            self.assertIn("asset_y", item)
            self.assertIn("asset_x", item)
            self.assertIn("adf_statistic", item)
            self.assertIn("is_cointegrated", item)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            # Mismatched length
            calculate_cointegration([1.0, 2.0], [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            # Less than 15 observations
            calculate_cointegration(np.ones(10), np.ones(10))
        with self.assertRaises(ValueError):
            # Negative prices
            calculate_cointegration([-1.0] * 20, [1.0] * 20)
        with self.assertRaises(ValueError):
            # Zero variance
            calculate_cointegration([5.0] * 20, [10.0] * 20)
        with self.assertRaises(ValueError):
            # Invalid thresholds: exit_z >= entry_z
            simulate_pairs_trading(self.y, self.x, entry_z=1.0, exit_z=2.0)
        with self.assertRaises(ValueError):
            # Negative fees
            simulate_pairs_trading(self.y, self.x, fee_rate=-0.01)

    def test_export_pairs_trading_json_and_csv(self):
        sim = simulate_pairs_trading(self.y, self.x, entry_z=1.5, exit_z=0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "pairs_report.json"
            csv_file = Path(tmpdir) / "pairs_report.csv"

            export_pairs_trading_json(sim, json_file)
            self.assertTrue(json_file.exists())
            with open(json_file, "r", encoding="utf-8") as f:
                loaded_json = json.load(f)
            self.assertIn("metrics", loaded_json)

            export_pairs_trading_csv(sim, csv_file)
            self.assertTrue(csv_file.exists())
            df = pd.read_csv(csv_file)
            self.assertEqual(len(df), 1)
            self.assertIn("total_return_pct", df.columns)
            self.assertIn("hedge_ratio_beta", df.columns)

    def test_main_cli_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = str(Path(tmpdir) / "cli_pairs.json")
            csv_path = str(Path(tmpdir) / "cli_pairs.csv")
            test_args = [
                "crypto_pairs_trading.py",
                "--symbol-a", "ETH/USDT",
                "--symbol-b", "BTC/USDT",
                "--entry-z", "1.5",
                "--exit-z", "0.5",
                "--export-json", json_path,
                "--export-csv", csv_path,
            ]
            with patch.object(sys, "argv", test_args):
                pairs_main()

            self.assertTrue(Path(json_path).exists())
            self.assertTrue(Path(csv_path).exists())

    def test_main_cli_scan_example(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = str(Path(tmpdir) / "scan_pairs.json")
            csv_path = str(Path(tmpdir) / "scan_pairs.csv")
            test_args = [
                "crypto_pairs_trading.py",
                "--scan-example",
                "--export-json", json_path,
                "--export-csv", csv_path,
            ]
            with patch.object(sys, "argv", test_args):
                pairs_main()

            self.assertTrue(Path(json_path).exists())
            self.assertTrue(Path(csv_path).exists())


if __name__ == "__main__":
    unittest.main()
