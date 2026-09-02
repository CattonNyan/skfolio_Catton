"""Tests for Freqtrade multi-strategy allocation optimizer."""

import unittest
import pandas as pd
import numpy as np

from scripts.freqtrade_strategy_optimizer import (
    optimize_strategy_allocation,
    parse_freqtrade_backtest_trades,
)


class StrategyOptimizerTests(unittest.TestCase):
    def test_parse_backtest_trades(self):
        sample_json = {
            "strategy": {
                "StratA": {
                    "trades": [
                        {"close_date": "2026-01-01 12:00:00", "profit_abs": 50.0},
                        {"close_date": "2026-01-01 18:00:00", "profit_abs": 30.0},
                    ]
                },
                "StratB": {
                    "trades": [
                        {"close_date": "2026-01-01 15:00:00", "profit_abs": -20.0},
                    ]
                },
            }
        }
        df = parse_freqtrade_backtest_trades(sample_json)
        self.assertFalse(df.empty)
        self.assertIn("StratA", df.columns)
        self.assertIn("StratB", df.columns)
        self.assertEqual(df.loc["2026-01-01", "StratA"], 80.0)
        self.assertEqual(df.loc["2026-01-01", "StratB"], -20.0)

    def test_optimize_strategy_allocation(self):
        dates = pd.date_range("2026-01-01", periods=30, freq="1D")
        np.random.seed(42)
        # Strat Low Vol (std 10) vs Strat High Vol (std 50)
        daily = pd.DataFrame(
            {
                "LowVolStrat": np.random.normal(10, 10, size=30),
                "HighVolStrat": np.random.normal(10, 50, size=30),
            },
            index=dates,
        )
        res = optimize_strategy_allocation(daily, total_capital=10000.0, model="Risk Parity")
        weights = res["weights"]
        self.assertIn("LowVolStrat", weights)
        self.assertIn("HighVolStrat", weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)
        # Low volatility strategy should receive higher allocation in Risk Parity
        self.assertGreater(weights["LowVolStrat"], weights["HighVolStrat"])

    def test_empty_profits_fallback(self):
        res = optimize_strategy_allocation(pd.DataFrame(), total_capital=1000.0)
        self.assertIn("Strategy_A", res["weights"])
        self.assertEqual(res["weights"]["Strategy_A"], 0.5)


if __name__ == "__main__":
    unittest.main()
