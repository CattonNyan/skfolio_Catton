"""Tests for Quantitative Multi-Factor Analyzer module."""

import unittest
import pandas as pd
import numpy as np

from scripts.crypto_portfolio_optimizer import generate_synthetic_crypto_data
from scripts.crypto_factor_analyzer import (
    compute_crypto_factors,
    select_smart_beta_universe,
)


class FactorAnalyzerTests(unittest.TestCase):
    def test_compute_crypto_factors_basic(self):
        prices = generate_synthetic_crypto_data(periods=100)
        df = compute_crypto_factors(prices, lookback_bars=50)

        self.assertFalse(df.empty)
        self.assertIn("momentum", df.columns)
        self.assertIn("volatility", df.columns)
        self.assertIn("sortino_ratio", df.columns)
        self.assertIn("composite_score", df.columns)
        # Should contain all assets
        self.assertEqual(len(df), len(prices.columns))

    def test_insufficient_history_rejected(self):
        prices = generate_synthetic_crypto_data(periods=30)
        with self.assertRaises(ValueError):
            compute_crypto_factors(prices, lookback_bars=60)

    def test_invalid_lookback_rejected(self):
        prices = generate_synthetic_crypto_data(periods=30)

        for lookback in (0, 1, -1, 2.5, True):
            with self.subTest(lookback=lookback), self.assertRaises(ValueError):
                compute_crypto_factors(prices, lookback_bars=lookback)

    def test_select_smart_beta_universe(self):
        prices = generate_synthetic_crypto_data(periods=100)
        top_assets, filtered_df = select_smart_beta_universe(prices, top_n=2, lookback_bars=50)

        self.assertEqual(len(top_assets), 2)
        self.assertEqual(list(filtered_df.columns), top_assets)
        self.assertEqual(len(filtered_df), len(prices))

    def test_invalid_top_n_rejected(self):
        prices = generate_synthetic_crypto_data(periods=30)

        for top_n in (0, -1, 1.5, True, len(prices.columns) + 1):
            with self.subTest(top_n=top_n), self.assertRaises(ValueError):
                select_smart_beta_universe(
                    prices, top_n=top_n, lookback_bars=20
                )

    def test_invalid_prices_rejected(self):
        valid = generate_synthetic_crypto_data(periods=50)
        invalid_cases = (
            "not_a_df",
            pd.DataFrame(),
            valid.replace(valid.iloc[0, 0], -10.0),
            pd.DataFrame({"A": ["bad", "str"], "B": [1.0, 2.0]}),
        )
        for bad in invalid_cases:
            with self.subTest(bad=type(bad)), self.assertRaises(ValueError):
                compute_crypto_factors(bad, lookback_bars=20)


    def test_custom_factor_weights(self):
        from scripts.crypto_factor_analyzer import compute_crypto_factors
        prices = generate_synthetic_crypto_data(periods=100)
        custom_w = {"momentum": 0.70, "low_volatility": 0.30, "trend_strength": 0.0, "sortino_ratio": 0.0}
        df = compute_crypto_factors(prices, lookback_bars=50, factor_weights=custom_w)
        self.assertIn("composite_score", df.columns)
        self.assertEqual(len(df), len(prices.columns))

    def test_invalid_factor_weights_rejected(self):
        from scripts.crypto_factor_analyzer import compute_crypto_factors
        prices = generate_synthetic_crypto_data(periods=50)
        bad_cases = (
            {},
            {"unknown_factor": 1.0},
            {"momentum": -0.5, "low_volatility": 1.5},
            {"momentum": 0.0, "low_volatility": 0.0, "trend_strength": 0.0, "sortino_ratio": 0.0},
            {"momentum": "high"},
            "not_a_dict",
        )
        for bad in bad_cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                compute_crypto_factors(prices, lookback_bars=20, factor_weights=bad)

    def test_generate_factor_tilted_weights(self):
        from scripts.crypto_factor_analyzer import compute_crypto_factors, generate_factor_tilted_weights
        prices = generate_synthetic_crypto_data(periods=100)
        factors = compute_crypto_factors(prices, lookback_bars=50)
        
        # Equal weights for top 2
        w_eq = generate_factor_tilted_weights(factors, top_n=2, weighting="equal")
        self.assertAlmostEqual(sum(w_eq.values()), 1.0, places=3)
        non_zero = [k for k, v in w_eq.items() if v > 0]
        self.assertEqual(len(non_zero), 2)
        
        # Score weighted
        w_score = generate_factor_tilted_weights(factors, top_n=3, weighting="score_weighted")
        self.assertAlmostEqual(sum(w_score.values()), 1.0, places=3)
        for v in w_score.values():
            self.assertGreaterEqual(v, 0.0)

    def test_omega_ratio(self):
        from scripts.crypto_factor_analyzer import compute_omega_ratio
        # Balanced returns: positive and negative
        rets = np.array([0.02, 0.04, -0.01, -0.02, 0.03])
        # Gains: 0.02 + 0.04 + 0.03 = 0.09
        # Losses: 0.01 + 0.02 = 0.03 -> Omega = 0.09 / 0.03 = 3.0
        omega = compute_omega_ratio(rets, threshold=0.0)
        self.assertAlmostEqual(omega, 3.0, places=4)

        # All positive returns
        self.assertEqual(compute_omega_ratio([0.01, 0.02]), float("inf"))

        with self.assertRaises(ValueError):
            compute_omega_ratio([])

    def test_gain_to_pain_ratio(self):
        from scripts.crypto_factor_analyzer import compute_gain_to_pain_ratio
        rets = np.array([0.05, 0.03, -0.02, -0.01])
        # Total return = 0.05 + 0.03 - 0.02 - 0.01 = 0.05
        # Pain = 0.02 + 0.01 = 0.03 -> GPR = 0.05 / 0.03 = 1.6667
        gpr = compute_gain_to_pain_ratio(rets)
        self.assertAlmostEqual(gpr, 5.0 / 3.0, places=4)

        with self.assertRaises(ValueError):
            compute_gain_to_pain_ratio([])


if __name__ == "__main__":
    unittest.main()

