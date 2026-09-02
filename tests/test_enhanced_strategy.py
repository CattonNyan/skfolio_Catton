"""Tests for SkfolioEnhancedAtrStrategy module."""

import unittest
from datetime import datetime
import pandas as pd
import numpy as np

from strategies.SkfolioEnhancedAtrStrategy import SkfolioEnhancedAtrStrategy


class EnhancedStrategyTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "pair_risk_limits": {
                "BTC/USDT": {"recommended_stoploss": -0.045, "recommended_takeprofit": 0.09},
                "SOL/USDT": {"recommended_stoploss": -0.105, "recommended_takeprofit": 0.21},
            }
        }
        self.strategy = SkfolioEnhancedAtrStrategy(config=self.config)

    def test_custom_stoploss_tailored_per_asset(self):
        # BTC stoploss should match its risk profile (-4.5%)
        sl_btc = self.strategy.custom_stoploss(
            pair="BTC/USDT",
            trade=None,
            current_time=datetime.now(),
            current_rate=50000.0,
            current_profit=-0.01,
            after_fill=False,
        )
        self.assertEqual(sl_btc, -0.045)

        # SOL stoploss should match its wider volatility (-10.5%)
        sl_sol = self.strategy.custom_stoploss(
            pair="SOL/USDT",
            trade=None,
            current_time=datetime.now(),
            current_rate=200.0,
            current_profit=-0.02,
            after_fill=False,
        )
        self.assertEqual(sl_sol, -0.105)

    def test_break_even_stoploss_on_profit(self):
        # When trade is up +4%, stoploss moves into positive (+0.5%)
        sl = self.strategy.custom_stoploss(
            pair="BTC/USDT",
            trade=None,
            current_time=datetime.now(),
            current_rate=55000.0,
            current_profit=0.04,
            after_fill=False,
        )
        self.assertEqual(sl, 0.005)

    def test_populate_indicators_and_signals(self):
        dates = pd.date_range("2026-01-01", periods=60, freq="15min")
        df = pd.DataFrame(
            {
                "date": dates,
                "open": np.linspace(100, 150, 60),
                "high": np.linspace(101, 152, 60),
                "low": np.linspace(99, 149, 60),
                "close": np.linspace(100, 150, 60),
                "volume": [1000.0] * 60,
            }
        )
        df = self.strategy.populate_indicators(df, metadata={"pair": "BTC/USDT"})
        self.assertIn("ema20", df.columns)
        self.assertIn("rsi", df.columns)

        df = self.strategy.populate_entry_trend(df, metadata={"pair": "BTC/USDT"})
        self.assertIn("enter_long", df.columns)

        df = self.strategy.populate_exit_trend(df, metadata={"pair": "BTC/USDT"})
        self.assertIn("exit_long", df.columns)


if __name__ == "__main__":
    unittest.main()
