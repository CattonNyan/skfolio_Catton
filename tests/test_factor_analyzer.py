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
        self.assertIn("composite_score", df.columns)
        # Should contain all assets
        self.assertEqual(len(df), len(prices.columns))

    def test_insufficient_history_rejected(self):
        prices = generate_synthetic_crypto_data(periods=30)
        with self.assertRaises(ValueError):
            compute_crypto_factors(prices, lookback_bars=60)

    def test_select_smart_beta_universe(self):
        prices = generate_synthetic_crypto_data(periods=100)
        top_assets, filtered_df = select_smart_beta_universe(prices, top_n=2, lookback_bars=50)

        self.assertEqual(len(top_assets), 2)
        self.assertEqual(list(filtered_df.columns), top_assets)
        self.assertEqual(len(filtered_df), len(prices))


if __name__ == "__main__":
    unittest.main()
