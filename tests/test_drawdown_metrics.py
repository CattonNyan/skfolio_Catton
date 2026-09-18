import unittest
import numpy as np
import pandas as pd

from scripts.crypto_drawdown_metrics import (
    compute_annualized_cagr,
    compute_burke_ratio,
    compute_drawdown_metrics_summary,
    compute_drawdown_series,
    compute_martin_ratio,
    compute_max_drawdown,
    compute_nav_series,
    compute_pain_index,
    compute_pain_ratio,
    compute_ulcer_index,
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
            "calmar_ratio",
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


if __name__ == "__main__":
    unittest.main()
