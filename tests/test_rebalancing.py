"""Tests for crypto portfolio rebalancing backtest module."""

import unittest
import pandas as pd
import numpy as np

from scripts.crypto_portfolio_optimizer import generate_synthetic_crypto_data
from scripts.crypto_rebalancing_backtest import calculate_drawdown, simulate_rebalancing


class RebalancingTests(unittest.TestCase):
    def test_calculate_drawdown(self):
        nav = pd.Series([1.0, 1.1, 1.05, 0.88, 0.95, 1.2])
        mdd, dd_series = calculate_drawdown(nav)
        self.assertIsInstance(mdd, float)
        self.assertLessEqual(mdd, 0.0)
        # Minimum was 0.88 from peak 1.1 -> (0.88 - 1.1) / 1.1 = -0.2
        self.assertAlmostEqual(mdd, -0.2, places=2)

    def test_simulate_rebalancing_execution(self):
        prices = generate_synthetic_crypto_data(periods=200)
        res = simulate_rebalancing(
            prices=prices,
            train_bars=80,
            rebalance_freq_bars=20,
            fee_rate=0.001,
            model_choice="Equal Weight",
        )
        self.assertIn("summary", res)
        self.assertIn("Total Return (%)", res["summary"])
        self.assertIn("Max Drawdown (%)", res["summary"])
        self.assertIsInstance(res["nav_port"], pd.Series)
        self.assertEqual(len(res["nav_port"]), 119)


if __name__ == "__main__":
    unittest.main()
