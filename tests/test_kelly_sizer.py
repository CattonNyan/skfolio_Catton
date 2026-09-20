import unittest
import numpy as np
import pandas as pd

from scripts.crypto_kelly_sizer import (
    calculate_discrete_kelly,
    calculate_continuous_kelly,
    calculate_portfolio_kelly,
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


if __name__ == "__main__":
    unittest.main()
