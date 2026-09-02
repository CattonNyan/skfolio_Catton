"""Tests for Monte Carlo simulation module."""

import unittest
import pandas as pd
from scripts.crypto_portfolio_optimizer import generate_synthetic_crypto_data
from scripts.crypto_monte_carlo import simulate_monte_carlo_paths


class MonteCarloTests(unittest.TestCase):
    def test_simulate_monte_carlo_basic(self):
        prices = generate_synthetic_crypto_data(periods=100)
        weights = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        res = simulate_monte_carlo_paths(
            prices=prices,
            weights=weights,
            days=30,
            num_simulations=200,
            initial_capital=10000.0,
            seed=42,
        )

        self.assertEqual(res["days"], 30)
        self.assertEqual(res["num_simulations"], 200)
        self.assertEqual(res["initial_capital"], 10000.0)
        self.assertGreater(res["median_final_wealth"], 0)
        self.assertLessEqual(res["worst_case_5pct"], res["best_case_95pct"])

        # Check path progression
        self.assertEqual(len(res["path_p50"]), 31)  # Day 0 to Day 30
        self.assertEqual(res["path_p50"][0], 10000.0)

    def test_invalid_capital_rejected(self):
        prices = generate_synthetic_crypto_data(periods=50)
        weights = {"BTC/USDT": 1.0}
        with self.assertRaises(ValueError):
            simulate_monte_carlo_paths(prices, weights, initial_capital=-500.0)

    def test_no_common_assets_rejected(self):
        prices = generate_synthetic_crypto_data(periods=50)
        weights = {"NON_EXISTENT/COIN": 1.0}
        with self.assertRaises(ValueError):
            simulate_monte_carlo_paths(prices, weights)


if __name__ == "__main__":
    unittest.main()
