"""Tests for crypto portfolio rebalancing backtest module."""

import unittest
import pandas as pd
import numpy as np

from scripts.crypto_portfolio_optimizer import generate_synthetic_crypto_data
from scripts.crypto_rebalancing_backtest import (
    calculate_drawdown,
    calculate_weight_drift,
    simulate_drift_band_rebalancing,
    simulate_rebalancing,
)


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

    def test_min_semi_variance_rebalancing(self):
        prices = generate_synthetic_crypto_data(periods=150)
        res = simulate_rebalancing(
            prices=prices,
            train_bars=60,
            rebalance_freq_bars=15,
            fee_rate=0.001,
            model_choice="Min Semi-Variance",
        )
        self.assertIn("summary", res)
        self.assertEqual(res["summary"]["Model"], "Min Semi-Variance")
        self.assertGreater(len(res["nav_port"]), 0)

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


    def test_min_cvar_and_schur_rebalancing(self):
        prices = generate_synthetic_crypto_data(periods=150)
        for model in ("Min CVaR", "Schur"):
            with self.subTest(model=model):
                res = simulate_rebalancing(
                    prices=prices,
                    train_bars=60,
                    rebalance_freq_bars=15,
                    fee_rate=0.001,
                    model_choice=model,
                )
                self.assertIn("summary", res)
                self.assertEqual(res["summary"]["Model"], model)
                self.assertGreater(len(res["nav_port"]), 0)


    def test_tolerance_band_rebalancing(self):
        prices = generate_synthetic_crypto_data(periods=200)
        # 1. Without tolerance band (every scheduled rebalance executes)
        res_no_band = simulate_rebalancing(
            prices=prices,
            train_bars=80,
            rebalance_freq_bars=15,
            fee_rate=0.001,
            model_choice="Risk Parity",
            tolerance_band=None,
        )
        # 2. With high tolerance band (e.g. 0.15 = 15% drift required)
        res_with_band = simulate_rebalancing(
            prices=prices,
            train_bars=80,
            rebalance_freq_bars=15,
            fee_rate=0.001,
            model_choice="Risk Parity",
            tolerance_band=0.15,
        )
        self.assertIn("Skipped Rebalances", res_with_band["summary"])
        self.assertGreater(res_with_band["skipped_rebalances"], 0)
        # Turnover with tolerance band should be strictly less than or equal to without
        self.assertLessEqual(
            res_with_band["summary"]["Average Turnover (%)"],
            res_no_band["summary"]["Average Turnover (%)"],
        )

    def test_tolerance_band_invalid_parameters_rejected(self):
        prices = generate_synthetic_crypto_data(periods=50)
        for bad_band in (-0.01, 1.5, "0.05", True, float("nan")):
            with self.subTest(bad_band=bad_band), self.assertRaises(ValueError):
                simulate_rebalancing(prices, tolerance_band=bad_band)

    def test_drawdown_guard(self):
        # Generate data with a steep downward shock
        prices = generate_synthetic_crypto_data(periods=120)
        # Force a steep crash in second half
        prices.iloc[60:] = prices.iloc[60:] * np.linspace(1.0, 0.5, len(prices) - 60)[:, None]

        res = simulate_rebalancing(
            prices=prices,
            train_bars=40,
            rebalance_freq_bars=10,
            drawdown_guard=0.10,  # 10% drawdown threshold
        )
        self.assertIn("Guard Triggers", res["summary"])
        self.assertGreater(res["guard_triggers"], 0)
        self.assertEqual(res["summary"]["Drawdown Guard (%)"], 10.0)

    def test_drawdown_guard_invalid_parameters_rejected(self):
        prices = generate_synthetic_crypto_data(periods=50)
        for bad_guard in (-0.05, 0.0, 1.0, 1.2, "bad", True, float("nan")):
            with self.subTest(bad_guard=bad_guard), self.assertRaises(ValueError):
                simulate_rebalancing(prices, drawdown_guard=bad_guard)

    def test_calculate_weight_drift(self):
        w1 = np.array([0.5, 0.3, 0.2])
        w2 = np.array([0.45, 0.35, 0.20])
        drift = calculate_weight_drift(w1, w2)
        self.assertAlmostEqual(drift, 0.05, places=4)

        with self.assertRaises(ValueError):
            calculate_weight_drift(w1, np.array([0.5, 0.5]))

    def test_simulate_drift_band_rebalancing(self):
        prices = generate_synthetic_crypto_data(periods=150)
        res = simulate_drift_band_rebalancing(
            prices=prices,
            band=0.05,
            train_bars=50,
            max_holding_bars=30,
            fee_rate=0.001,
        )
        self.assertIn("summary", res)
        s = res["summary"]
        self.assertEqual(s["Band (%)"], 5.0)
        self.assertIn("Ulcer Index (%)", s)
        self.assertIn("Rebalance Triggers", s)
        self.assertIn("Max Drift Observed (%)", s)
        self.assertIsInstance(res["nav_port"], pd.Series)
        self.assertGreater(len(res["nav_port"]), 0)

    def test_rebalancing_summary_includes_ulcer_and_martin(self):
        prices = generate_synthetic_crypto_data(periods=120)
        res = simulate_rebalancing(
            prices=prices,
            train_bars=50,
            rebalance_freq_bars=15,
            fee_rate=0.001,
            model_choice="Equal Weight",
        )
        s = res["summary"]
        self.assertIn("Ulcer Index (%)", s)
        self.assertIn("Martin Ratio", s)
        self.assertIsInstance(s["Ulcer Index (%)"], float)
        self.assertIsInstance(s["Martin Ratio"], float)


if __name__ == "__main__":
    unittest.main()

