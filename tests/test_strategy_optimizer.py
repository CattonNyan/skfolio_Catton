"""Tests for Freqtrade multi-strategy allocation optimizer."""

import unittest
import pandas as pd
import numpy as np

from scripts.freqtrade_strategy_optimizer import (
    export_strategy_allocation_csv,
    load_freqtrade_backtest_file,
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
        self.assertIn("portfolio_metrics", res)
        self.assertIn("annualized_sharpe_ratio", res["portfolio_metrics"])
        self.assertIn("annualized_sortino_ratio", res["portfolio_metrics"])
        self.assertIn("diversification_ratio", res["portfolio_metrics"])

    def test_empty_profits_fallback(self):
        res = optimize_strategy_allocation(pd.DataFrame(), total_capital=1000.0)
        self.assertIn("Strategy_A", res["weights"])
        self.assertEqual(res["weights"]["Strategy_A"], 0.5)

    def test_invalid_total_capital_rejected(self):
        for capital in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(capital=capital), self.assertRaises(ValueError):
                optimize_strategy_allocation(
                    pd.DataFrame(), total_capital=capital
                )

    def test_unknown_model_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported allocation model"):
            optimize_strategy_allocation(
                pd.DataFrame(), model="Maximum Return"
            )

    def test_min_variance_and_invalid_inputs(self):
        dates = pd.date_range("2026-01-01", periods=10, freq="1D")
        daily = pd.DataFrame({"A": [1.0] * 10, "B": [2.0] * 10}, index=dates)
        res = optimize_strategy_allocation(daily, total_capital=5000.0, model="Min Variance")
        self.assertIn("A", res["weights"])
        self.assertAlmostEqual(sum(res["weights"].values()), 1.0, places=3)

        for bad_cap in (True, False):
            with self.subTest(bad_cap=bad_cap), self.assertRaises(ValueError):
                optimize_strategy_allocation(daily, total_capital=bad_cap)

        for bad_df in ("not_a_df", [1, 2], None):
            with self.subTest(bad_df=bad_df), self.assertRaises(ValueError):
                optimize_strategy_allocation(bad_df)

    def test_export_strategy_allocation_csv(self):
        import tempfile
        from pathlib import Path

        dates = pd.date_range("2026-01-01", periods=20, freq="1D")
        daily = pd.DataFrame(
            {"Strat1": [10.0] * 20, "Strat2": [20.0] * 20},
            index=dates,
        )
        res = optimize_strategy_allocation(daily, total_capital=10000.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "alloc.csv"
            export_strategy_allocation_csv(res, out_csv)
            self.assertTrue(out_csv.exists())
            df = pd.read_csv(out_csv)
            self.assertIn("type", df.columns)
            self.assertIn("name", df.columns)
            self.assertIn("weight", df.columns)
            self.assertIn("capital", df.columns)
            names = set(df["name"])
            self.assertIn("Strat1", names)
            self.assertIn("Strat2", names)
            self.assertIn("annualized_sharpe_ratio", names)

    def test_load_backtest_file_json_and_zip(self):
        import tempfile
        import zipfile
        import json
        from pathlib import Path

        mock_data = {
            "strategy": {
                "StratA": {"trades": [{"profit_abs": 10.0, "close_date": "2026-01-01"}]}
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            json_file = tmp / "backtest-result.json"
            json_file.write_text(json.dumps(mock_data), encoding="utf-8")

            # Load JSON directly
            d1, p1 = load_freqtrade_backtest_file(json_file)
            self.assertEqual(p1, json_file)
            self.assertIn("StratA", d1["strategy"])

            # Create Zip
            zip_file = tmp / "backtest-result.zip"
            with zipfile.ZipFile(zip_file, "w") as zf:
                zf.writestr("backtest-result.json", json.dumps(mock_data))
                zf.writestr("backtest-result_config.json", "{}")

            # Load Zip directly
            d2, p2 = load_freqtrade_backtest_file(zip_file)
            self.assertEqual(p2, zip_file)
            self.assertIn("StratA", d2["strategy"])

            # Load Directory
            d3, p3 = load_freqtrade_backtest_file(tmp)
            self.assertIn("StratA", d3["strategy"])


if __name__ == "__main__":
    unittest.main()

