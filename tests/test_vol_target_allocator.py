import unittest
import numpy as np
import pandas as pd

from scripts.crypto_vol_target_allocator import (
    calculate_portfolio_realized_volatility,
    apply_volatility_targeting,
    simulate_vol_targeted_backtest,
)
from scripts.crypto_portfolio_optimizer import generate_synthetic_crypto_data


class VolTargetAllocatorTests(unittest.TestCase):
    def test_apply_volatility_targeting_de_risking(self):
        # High realized vol (60%) vs target (30%) -> scalar should be ~0.50
        base_w = {"BTC/USDT": 0.60, "ETH/USDT": 0.40}
        res = apply_volatility_targeting(base_w, realized_vol_ann=0.60, target_vol_ann=0.30)
        self.assertAlmostEqual(res.vol_scalar, 0.50, places=2)
        self.assertAlmostEqual(res.cash_weight, 0.50, places=2)
        self.assertAlmostEqual(res.scaled_weights["BTC/USDT"], 0.30, places=2)
        self.assertFalse(res.is_leveraged)

    def test_apply_volatility_targeting_calm_market(self):
        # Low realized vol (15%) vs target (30%), max_leverage 1.0 -> scalar clamped to 1.0
        base_w = {"BTC/USDT": 0.50, "ETH/USDT": 0.50}
        res = apply_volatility_targeting(base_w, realized_vol_ann=0.15, target_vol_ann=0.30, max_leverage=1.0)
        self.assertEqual(res.vol_scalar, 1.0)
        self.assertEqual(res.cash_weight, 0.0)

    def test_calculate_portfolio_realized_volatility(self):
        rng = np.random.default_rng(42)
        rets = pd.DataFrame({
            "BTC/USDT": rng.normal(0, 0.02, 100),
            "ETH/USDT": rng.normal(0, 0.03, 100),
        })
        vol = calculate_portfolio_realized_volatility(rets, {"BTC/USDT": 0.5, "ETH/USDT": 0.5})
        self.assertGreater(vol, 0.0)

    def test_simulate_vol_targeted_backtest(self):
        prices = generate_synthetic_crypto_data(periods=100)
        base_w = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        res = simulate_vol_targeted_backtest(
            prices[["BTC/USDT", "ETH/USDT"]],
            base_weights=base_w,
            target_vol_ann=0.25,
            lookback_bars=20,
        )
        self.assertIn("nav_targeted", res)
        self.assertIn("mdd_targeted_pct", res)
        self.assertIn("sharpe_targeted", res)
        self.assertIn("calmar_targeted", res)
        self.assertIn("sharpe_static", res)
        self.assertIn("calmar_static", res)
        self.assertIn("return_targeted_pct", res)
        self.assertIn("return_static_pct", res)
        self.assertGreater(len(res["nav_targeted"]), 0)

    def test_vol_target_result_to_dict(self):
        base_w = {"BTC/USDT": 0.60, "ETH/USDT": 0.40}
        res = apply_volatility_targeting(base_w, realized_vol_ann=0.50, target_vol_ann=0.25)
        d = res.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["vol_scalar"], 0.5)
        self.assertEqual(d["target_vol_ann"], 0.25)
        self.assertEqual(d["realized_vol_ann"], 0.50)
        self.assertEqual(d["cash_weight"], 0.50)
        self.assertEqual(d["scaled_weights"]["BTC/USDT"], 0.30)
        self.assertEqual(d["scaled_weights"]["ETH/USDT"], 0.20)
        self.assertFalse(d["is_leveraged"])

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            apply_volatility_targeting({"A": 1.0}, realized_vol_ann=0.2, target_vol_ann=-0.1)
        with self.assertRaises(ValueError):
            apply_volatility_targeting({"A": 1.0}, realized_vol_ann=0.2, max_leverage=-1.0)


if __name__ == "__main__":
    unittest.main()
