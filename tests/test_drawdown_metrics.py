import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from scripts.crypto_drawdown_metrics import (
    compute_annualized_cagr,
    compute_burke_ratio,
    compute_drawdown_duration_stats,
    compute_drawdown_metrics_summary,
    compute_drawdown_series,
    compute_martin_ratio,
    compute_max_drawdown,
    compute_nav_series,
    compute_pain_index,
    compute_pain_ratio,
    compute_sterling_ratio,
    compute_tail_ratio,
    compute_common_sense_ratio,
    compute_k_ratio,
    compute_ulcer_index,
    export_drawdown_metrics_json,
    main as drawdown_main,
)


class DrawdownMetricsTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.random_returns = np.random.normal(0.0005, 0.02, 100)

    def test_monotonic_positive_returns(self):
        # Strictly positive returns -> no drawdowns
        pos_returns = np.array([0.01, 0.02, 0.015, 0.03, 0.005])
        mdd = compute_max_drawdown(pos_returns, is_returns=True)
        ui = compute_ulcer_index(pos_returns, is_returns=True)
        pi = compute_pain_index(pos_returns, is_returns=True)

        self.assertEqual(mdd, 0.0)
        self.assertEqual(ui, 0.0)
        self.assertEqual(pi, 0.0)

        martin = compute_martin_ratio(pos_returns, risk_free_rate=0.0, is_returns=True)
        self.assertEqual(martin, 999.0)

    def test_known_drawdown_path(self):
        # Known price trajectory: 100 -> 120 -> 90 -> 100
        # Peak: 100 (DD=0), 120 (DD=0), 90 (DD=(90-120)/120 = -25%), 100 (DD=(100-120)/120 = -16.6667%)
        prices = np.array([100.0, 120.0, 90.0, 100.0])
        dd = compute_drawdown_series(prices, is_returns=False)

        np.testing.assert_allclose(dd[0], 0.0, atol=1e-4)
        np.testing.assert_allclose(dd[1], 0.0, atol=1e-4)
        np.testing.assert_allclose(dd[2], -25.0, atol=1e-4)
        np.testing.assert_allclose(dd[3], -16.6667, atol=1e-3)

        mdd = compute_max_drawdown(prices, is_returns=False)
        self.assertAlmostEqual(mdd, 25.0, places=3)

        # UI = sqrt((0^2 + 0^2 + 25^2 + 16.6667^2) / 4)
        expected_ui = np.sqrt((0.0 + 0.0 + 25.0**2 + (100.0/6.0)**2) / 4.0)
        ui = compute_ulcer_index(prices, is_returns=False)
        self.assertAlmostEqual(ui, expected_ui, places=3)

        # PI = (0 + 0 + 25 + 16.6667) / 4
        expected_pi = (0.0 + 0.0 + 25.0 + (100.0/6.0)) / 4.0
        pi = compute_pain_index(prices, is_returns=False)
        self.assertAlmostEqual(pi, expected_pi, places=3)

    def test_price_and_returns_equivalence(self):
        prices = pd.Series([100.0, 105.0, 95.0, 98.0, 110.0, 102.0])
        returns = prices.pct_change().dropna()

        mdd_price = compute_max_drawdown(prices, is_returns=False)
        mdd_ret = compute_max_drawdown(returns, is_returns=True)
        self.assertAlmostEqual(mdd_price, mdd_ret, places=3)

    def test_summary_dictionary_integrity(self):
        summary = compute_drawdown_metrics_summary(self.random_returns, risk_free_rate=1.0, is_returns=True)
        required_keys = [
            "cagr_pct",
            "max_drawdown_pct",
            "ulcer_index",
            "pain_index",
            "martin_ratio",
            "pain_ratio",
            "burke_ratio",
            "sterling_ratio",
            "calmar_ratio",
            "omega_ratio",
            "gain_to_pain_ratio",
            "tail_ratio",
            "common_sense_ratio",
            "k_ratio",
        ]
        for k in required_keys:
            self.assertIn(k, summary)
            self.assertIsInstance(summary[k], float)

    def test_burke_ratio_modified(self):
        burke_mod = compute_burke_ratio(self.random_returns, risk_free_rate=0.0, is_returns=True, modified=True)
        martin = compute_martin_ratio(self.random_returns, risk_free_rate=0.0, is_returns=True)
        self.assertAlmostEqual(burke_mod, martin, places=4)

    def test_invalid_and_edge_inputs(self):
        with self.assertRaises(ValueError):
            compute_drawdown_series(np.array([]))

        with self.assertRaises(ValueError):
            compute_drawdown_series(np.array([np.nan, np.inf]))

        with self.assertRaises(ValueError):
            compute_nav_series(np.array([-10.0, 20.0]), is_returns=False)

    def test_drawdown_duration_stats(self):
        # Known prices: 100 -> 90 -> 95 -> 105 (underwater for 2 periods: 90, 95) -> 100 -> 110
        prices = [100.0, 90.0, 95.0, 105.0, 100.0, 110.0]
        stats = compute_drawdown_duration_stats(prices, is_returns=False)
        self.assertEqual(stats["max_drawdown_duration"], 2.0)
        self.assertEqual(stats["current_drawdown_duration"], 0.0)
        self.assertEqual(stats["drawdown_episodes_count"], 2.0)
        self.assertIn("time_underwater_pct", stats)
        self.assertGreater(stats["time_underwater_pct"], 0.0)

        # Monotonic positive series has 0 duration
        mono_stats = compute_drawdown_duration_stats([100.0, 101.0, 102.0], is_returns=False)
        self.assertEqual(mono_stats["max_drawdown_duration"], 0.0)
        self.assertEqual(mono_stats["drawdown_episodes_count"], 0.0)
        self.assertEqual(mono_stats["time_underwater_pct"], 0.0)

    def test_cli_and_json_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sub" / "metrics_test.json"
            test_args = ["crypto_drawdown_metrics.py", "--rf", "2.5", "--export-json", str(out_file)]
            with patch("sys.argv", test_args):
                drawdown_main()

            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("cagr_pct", data)
            self.assertIn("max_drawdown_pct", data)
            self.assertIn("ulcer_index", data)
            self.assertIn("pain_index", data)
            self.assertIn("martin_ratio", data)
            self.assertIn("sterling_ratio", data)
            self.assertIn("omega_ratio", data)
            self.assertIn("gain_to_pain_ratio", data)
            self.assertIn("time_underwater_pct", data)

    def test_sterling_ratio(self):
        # Monotonic positive -> mdd is 0 -> ratio clamped to 999.0
        pos_returns = np.array([0.01, 0.02, 0.015, 0.03, 0.005])
        self.assertEqual(compute_sterling_ratio(pos_returns, risk_free_rate=0.0, is_returns=True), 999.0)

        # Price trajectory: 100 -> 120 -> 90 -> 100 (net return 0%, max dd = 25%)
        prices = np.array([100.0, 120.0, 90.0, 100.0])
        sr = compute_sterling_ratio(prices, risk_free_rate=0.0, is_returns=False)
        self.assertIsInstance(sr, float)
        self.assertAlmostEqual(sr, 0.0, places=4)

    def test_omega_ratio(self):
        from scripts.crypto_drawdown_metrics import compute_omega_ratio
        # Positive returns only -> zero downside -> 999.0
        pos = np.array([0.02, 0.03, 0.01])
        self.assertEqual(compute_omega_ratio(pos, is_returns=True), 999.0)

        # Mixed returns: upside sum 0.05, downside abs sum 0.02 -> 2.5
        mixed = np.array([0.03, -0.01, 0.02, -0.01])
        self.assertAlmostEqual(compute_omega_ratio(mixed, is_returns=True), 2.5, places=2)

        # Price series mode
        prices = np.array([100.0, 105.0, 100.0])
        omega = compute_omega_ratio(prices, is_returns=False)
        self.assertGreater(omega, 0.0)

    def test_gain_to_pain_ratio(self):
        from scripts.crypto_drawdown_metrics import compute_gain_to_pain_ratio
        # Positive returns only -> zero loss -> 999.0
        pos = np.array([0.02, 0.03, 0.01])
        self.assertEqual(compute_gain_to_pain_ratio(pos, is_returns=True), 999.0)

        # Mixed returns: sum(all) = 0.03, sum(losses) = 0.02 -> 1.5
        mixed = np.array([0.03, -0.01, 0.02, -0.01])
        self.assertAlmostEqual(compute_gain_to_pain_ratio(mixed, is_returns=True), 1.5, places=2)

        # Price series mode
        prices = np.array([100.0, 110.0, 105.0])
        gpr = compute_gain_to_pain_ratio(prices, is_returns=False)
        self.assertGreater(gpr, 0.0)

    def test_export_drawdown_metrics_json_direct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "exported_metrics.json"
            sample = {"cagr_pct": 14.5, "sterling_ratio": 1.25, "pain_index": 5.2}
            export_drawdown_metrics_json(sample, out_file)
            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["sterling_ratio"], 1.25)
            self.assertEqual(loaded["cagr_pct"], 14.5)

    def test_export_drawdown_metrics_csv_direct(self):
        from scripts.crypto_drawdown_metrics import export_drawdown_metrics_csv
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "exported_metrics.csv"
            sample = {"cagr_pct": 14.5, "sterling_ratio": 1.25, "pain_index": 5.2}
            export_drawdown_metrics_csv(sample, out_file)
            self.assertTrue(out_file.exists())
            df = pd.read_csv(out_file)
            self.assertIn("metric", df.columns)
            self.assertIn("value", df.columns)
            metrics_dict = dict(zip(df["metric"], df["value"]))
            self.assertAlmostEqual(metrics_dict["sterling_ratio"], 1.25)
            self.assertAlmostEqual(metrics_dict["cagr_pct"], 14.5)

    def test_tail_ratio(self):
        # Sample with positive tail asymmetry
        returns = np.array([-0.01, -0.01, -0.01, 0.01, 0.05, 0.10])
        tail = compute_tail_ratio(returns, percentile=95.0, is_returns=True)
        self.assertGreater(tail, 1.0)

        # Zero lower tail (no downside)
        zero_tail = np.array([0.0, 0.0, 0.02, 0.05])
        self.assertEqual(compute_tail_ratio(zero_tail, is_returns=True), 999.0)

        # Price series mode
        prices = np.array([100.0, 105.0, 102.0, 115.0])
        tail_p = compute_tail_ratio(prices, is_returns=False)
        self.assertGreater(tail_p, 0.0)

    def test_common_sense_ratio(self):
        returns = np.array([-0.01, -0.02, 0.03, 0.04, 0.05])
        csr = compute_common_sense_ratio(returns, is_returns=True)
        self.assertGreater(csr, 0.0)

        # Zero or negative returns produce 0.0
        neg = np.array([-0.01, -0.02, -0.03])
        self.assertEqual(compute_common_sense_ratio(neg, is_returns=True), 0.0)

    def test_k_ratio(self):
        # Strongly upward linear trajectory -> high K-ratio
        upward_returns = np.array([0.01] * 20)
        k_up = compute_k_ratio(upward_returns, is_returns=True)
        self.assertGreater(k_up, 1.0)

        # Flat / zero returns
        flat_returns = np.array([0.0] * 10)
        self.assertEqual(compute_k_ratio(flat_returns, is_returns=True), 0.0)

        # Price series mode
        upward_prices = np.linspace(100, 200, 30)
        k_price = compute_k_ratio(upward_prices, is_returns=False)
        self.assertGreater(k_price, 1.0)


if __name__ == "__main__":
    unittest.main()
