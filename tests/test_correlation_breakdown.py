"""Tests for Correlation Breakdown & Decoupling detector module."""

import unittest
import pandas as pd
import numpy as np

from scripts.crypto_portfolio_optimizer import generate_synthetic_crypto_data
from scripts.crypto_correlation_breakdown import detect_correlation_breakdown


class CorrelationBreakdownTests(unittest.TestCase):
    def test_detect_correlation_breakdown_basic(self):
        prices = generate_synthetic_crypto_data(periods=100)
        res = detect_correlation_breakdown(prices, rolling_window=20)

        self.assertFalse(len(res) == 0)
        # Should contain other assets except benchmark
        for asset, data in res.items():
            self.assertIn("current_correlation", data)
            self.assertIn("historical_mean_corr", data)
            self.assertIn("status", data)
            self.assertIn("diversification_score", data)
            self.assertGreaterEqual(data["diversification_score"], 0.0)
            self.assertLessEqual(data["diversification_score"], 100.0)

    def test_insufficient_data_rejected(self):
        prices = generate_synthetic_crypto_data(periods=20)
        with self.assertRaises(ValueError):
            detect_correlation_breakdown(prices, rolling_window=30)

    def test_inverse_correlation_detection(self):
        dates = pd.date_range("2026-01-01", periods=60, freq="1D")
        np.random.seed(42)
        bench = np.cumsum(np.random.normal(0.01, 0.02, 60)) + 100
        # Exactly inverse asset
        inv_asset = 200 - bench

        prices = pd.DataFrame({"BTC": bench, "INVERSE": inv_asset}, index=dates)
        res = detect_correlation_breakdown(prices, benchmark="BTC", rolling_window=15)

        self.assertIn("INVERSE", res)
        self.assertLess(res["INVERSE"]["current_correlation"], 0.0)
        self.assertIn("INVERSE_DECOUPLING", res["INVERSE"]["status"])
        self.assertTrue(res["INVERSE"]["is_anomaly"])
        # High diversification score because of negative correlation
        self.assertGreater(res["INVERSE"]["diversification_score"], 50.0)


if __name__ == "__main__":
    unittest.main()
