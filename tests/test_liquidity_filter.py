import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from scripts.crypto_liquidity_filter import (
    calculate_liquidity_tier_and_score,
    compute_amihud_illiquidity,
    compute_liquidity_capacity_cap,
    estimate_corwin_schultz_spread,
    estimate_kyle_lambda,
    estimate_roll_spread,
    export_liquidity_report_csv,
    filter_crypto_universe,
    main as liquidity_main,
)


class LiquidityFilterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 50
        # Highly liquid asset
        p1 = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, n))
        v1 = rng.uniform(10000, 20000, n)
        h1 = p1 * (1.0 + rng.uniform(0.001, 0.005, n))
        l1 = p1 * (1.0 - rng.uniform(0.001, 0.005, n))
        self.liquid_df = pd.DataFrame({"close": p1, "volume": v1, "high": h1, "low": l1})

        # Illiquid penny asset
        p2 = 0.01 * np.cumprod(1.0 + rng.normal(0.0, 0.08, n))
        v2 = rng.uniform(10, 50, n)  # very low volume
        h2 = p2 * (1.0 + rng.uniform(0.05, 0.15, n))
        l2 = p2 * (1.0 - rng.uniform(0.05, 0.15, n))
        self.illiquid_df = pd.DataFrame({"close": p2, "volume": v2, "high": h2, "low": l2})

    def test_compute_amihud_illiquidity(self):
        ret = np.array([0.01, -0.02, 0.015])
        vol = np.array([100000.0, 200000.0, 150000.0])
        amihud = compute_amihud_illiquidity(ret, vol)
        self.assertGreater(amihud, 0.0)

        with self.assertRaises(ValueError):
            compute_amihud_illiquidity([], [])

    def test_estimate_corwin_schultz_spread(self):
        spread = estimate_corwin_schultz_spread(self.liquid_df["high"], self.liquid_df["low"])
        self.assertGreaterEqual(spread, 0.0)
        self.assertLess(spread, 20.0)

    def test_estimate_roll_spread(self):
        # Bounce series with negative serial covariance
        prices = np.array([100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0])
        roll_spread = estimate_roll_spread(prices)
        self.assertGreater(roll_spread, 0.0)

        # Insufficient data
        self.assertEqual(estimate_roll_spread([100.0, 101.0]), 0.0)

    def test_estimate_kyle_lambda(self):
        ret = np.array([0.01, 0.02, 0.015, 0.03, 0.025, 0.04])
        vol = np.array([10000, 40000, 20000, 90000, 60000, 160000])
        lam = estimate_kyle_lambda(ret, vol)
        self.assertGreaterEqual(lam, 0.0)
        self.assertEqual(estimate_kyle_lambda([0.01], [100.0]), 0.0)

    def test_liquidity_tier_and_score(self):
        score_high, tier_high = calculate_liquidity_tier_and_score(
            mean_vol_usd=50000000.0, amihud=0.01, spread_pct=0.05, roll_pct=0.02
        )
        self.assertGreaterEqual(score_high, 75.0)
        self.assertIn("Tier 1", tier_high)

        score_low, tier_low = calculate_liquidity_tier_and_score(
            mean_vol_usd=500.0, amihud=5.0, spread_pct=5.0, roll_pct=3.0
        )
        self.assertLess(score_low, 35.0)
        self.assertIn("Tier 4", tier_low)

    def test_compute_liquidity_capacity_cap(self):
        cap_usd, max_wt = compute_liquidity_capacity_cap(
            mean_volume_usd=1000000.0, portfolio_nav_usd=500000.0, max_adv_pct=5.0
        )
        # 5% of 1,000,000 = 50,000 USD
        self.assertEqual(cap_usd, 50000.0)
        # 50,000 / 500,000 = 10%
        self.assertEqual(max_wt, 10.0)

    def test_filter_crypto_universe(self):
        data = {
            "LIQUID/USDT": self.liquid_df,
            "ILLIQUID/USDT": self.illiquid_df,
        }
        liquid_syms, report = filter_crypto_universe(
            data,
            min_mean_volume_usd=50000.0,
            max_amihud=2.0,
            max_spread_pct=3.0,
            portfolio_nav_usd=200000.0,
            max_adv_pct=5.0,
        )
        self.assertIn("LIQUID/USDT", liquid_syms)
        self.assertNotIn("ILLIQUID/USDT", liquid_syms)
        self.assertTrue(report["LIQUID/USDT"].is_liquid)
        self.assertFalse(report["ILLIQUID/USDT"].is_liquid)
        self.assertIsNotNone(report["ILLIQUID/USDT"].rejection_reason)
        self.assertIn("roll_spread_pct", report["LIQUID/USDT"].to_dict())
        self.assertIn("liquidity_tier", report["LIQUID/USDT"].to_dict())
        self.assertIn("capacity_cap_usd", report["LIQUID/USDT"].to_dict())

    def test_filter_with_composite_score_threshold(self):
        data = {
            "LIQUID/USDT": self.liquid_df,
            "ILLIQUID/USDT": self.illiquid_df,
        }
        strict_syms, strict_report = filter_crypto_universe(
            data,
            min_composite_score=100.0,
        )
        self.assertEqual(len(strict_syms), 0)
        self.assertIn("Composite score", strict_report["LIQUID/USDT"].rejection_reason)

    def test_filter_with_slippage_threshold(self):
        data = {
            "LIQUID/USDT": self.liquid_df,
            "ILLIQUID/USDT": self.illiquid_df,
        }
        liquid_syms, report = filter_crypto_universe(
            data,
            min_mean_volume_usd=1000.0,
            max_amihud=10.0,
            trade_size_usd=10000.0,
            max_slippage_pct=5.0,
        )
        self.assertIn("LIQUID/USDT", liquid_syms)
        self.assertIn("estimated_slippage_pct", report["LIQUID/USDT"].to_dict())

        strict_syms, strict_report = filter_crypto_universe(
            data,
            max_slippage_pct=0.000001,
        )
        self.assertEqual(len(strict_syms), 0)
        self.assertIn("Est. slippage", strict_report["LIQUID/USDT"].rejection_reason)

    def test_liquidity_metrics_to_dict(self):
        data = {"LIQUID/USDT": self.liquid_df}
        _, report = filter_crypto_universe(data)
        d = report["LIQUID/USDT"].to_dict()
        self.assertEqual(d["symbol"], "LIQUID/USDT")
        self.assertIn("mean_volume_usd", d)
        self.assertIn("median_volume_usd", d)
        self.assertIn("amihud_illiquidity", d)
        self.assertIn("estimated_spread_pct", d)
        self.assertIn("roll_spread_pct", d)
        self.assertIn("kyle_lambda", d)
        self.assertIn("composite_score", d)
        self.assertIn("liquidity_tier", d)
        self.assertIn("capacity_cap_usd", d)
        self.assertIn("max_weight_pct", d)
        self.assertIn("is_liquid", d)

    def test_export_liquidity_report_csv(self):
        data = {
            "LIQUID/USDT": self.liquid_df,
            "ILLIQUID/USDT": self.illiquid_df,
        }
        _, report = filter_crypto_universe(data)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "liquidity_test.csv"
            export_liquidity_report_csv(report, out_file)
            self.assertTrue(out_file.exists())
            df = pd.read_csv(out_file)
            self.assertIn("symbol", df.columns)
            self.assertIn("is_liquid", df.columns)
            self.assertIn("mean_volume_usd", df.columns)
            self.assertIn("roll_spread_pct", df.columns)
            self.assertIn("liquidity_tier", df.columns)
            self.assertEqual(len(df), 2)

    def test_cli_execution_with_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "test.json"
            out_csv = Path(tmpdir) / "test.csv"
            test_args = [
                "crypto_liquidity_filter.py",
                "--portfolio-nav", "150000",
                "--max-adv", "4.0",
                "--export-json", str(out_json),
                "--export-csv", str(out_csv),
            ]
            with patch("sys.argv", test_args):
                liquidity_main()

            self.assertTrue(out_json.exists())
            self.assertTrue(out_csv.exists())
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("liquid_symbols", data)
            self.assertIn("portfolio_nav_usd", data)
            self.assertIn("metrics", data)


if __name__ == "__main__":
    unittest.main()
