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
        self.assertIn("Sortino Ratio (Ann.)", res["summary"])
        self.assertIn("Calmar Ratio", res["summary"])
        self.assertIsInstance(res["nav_port"], pd.Series)
        self.assertEqual(len(res["nav_port"]), 119)

    def test_invalid_rebalancing_parameters_rejected(self):
        prices = generate_synthetic_crypto_data(periods=50)
        invalid_parameters = (
            {"train_bars": 1},
            {"train_bars": True},
            {"rebalance_freq_bars": 0},
            {"rebalance_freq_bars": 2.5},
            {"fee_rate": -0.01},
            {"fee_rate": float("nan")},
            {"fee_rate": 1.0},
            {"model_choice": "Unknown"},
        )
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                simulate_rebalancing(prices, **parameters)

    def test_hourly_rebalancing_annualization(self):
        dates = pd.date_range("2026-01-01", periods=150, freq="1h")
        prices = pd.DataFrame(
            {
                "BTC/USDT": np.linspace(50000, 60000, 150),
                "ETH/USDT": np.linspace(3000, 3500, 150),
            },
            index=dates,
        )
        res = simulate_rebalancing(
            prices=prices,
            train_bars=50,
            rebalance_freq_bars=20,
            fee_rate=0.001,
            model_choice="Equal Weight",
        )
        s = res["summary"]
        self.assertIn("Sortino Ratio (Ann.)", s)
        self.assertIn("Calmar Ratio", s)
        self.assertGreater(s["Total Return (%)"], 0)

    def test_invalid_prices_rejected(self):
        valid = generate_synthetic_crypto_data(periods=50)
        invalid_cases = (
            "not_a_df",
            valid.iloc[:, :1],  # 1 asset only
            pd.DataFrame(),     # empty df
            valid.replace(valid.iloc[0, 0], -10.0),  # negative price
            valid.replace(valid.iloc[0, 0], float("nan")),  # nan price
            pd.DataFrame({"A": ["bad", "str"], "B": [1.0, 2.0]}),  # non-numeric
        )
        for bad_prices in invalid_cases:
            with self.subTest(bad_prices=type(bad_prices)), self.assertRaises(ValueError):
                simulate_rebalancing(bad_prices)


if __name__ == "__main__":
    unittest.main()
