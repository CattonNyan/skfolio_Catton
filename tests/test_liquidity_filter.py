import unittest
import numpy as np
import pandas as pd

from scripts.crypto_liquidity_filter import (
    compute_amihud_illiquidity,
    estimate_corwin_schultz_spread,
    filter_crypto_universe,
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

        # Invalid input checks
        with self.assertRaises(ValueError):
            compute_amihud_illiquidity([], [])

    def test_estimate_corwin_schultz_spread(self):
        spread = estimate_corwin_schultz_spread(self.liquid_df["high"], self.liquid_df["low"])
        self.assertGreaterEqual(spread, 0.0)
        self.assertLess(spread, 20.0)

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
        )
        self.assertIn("LIQUID/USDT", liquid_syms)
        self.assertNotIn("ILLIQUID/USDT", liquid_syms)
        self.assertTrue(report["LIQUID/USDT"].is_liquid)
        self.assertFalse(report["ILLIQUID/USDT"].is_liquid)
        self.assertIsNotNone(report["ILLIQUID/USDT"].rejection_reason)


if __name__ == "__main__":
    unittest.main()
