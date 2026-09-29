import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from scripts.crypto_vol_target_allocator import (
    calculate_portfolio_realized_volatility,
    apply_volatility_targeting,
    export_vol_target_json,
    export_vol_target_csv,
    simulate_vol_targeted_backtest,
    to_dict_vol_target_backtest,
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

    def test_to_dict_vol_target_backtest_and_export_json(self):
        prices = generate_synthetic_crypto_data(periods=60)
        base_w = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        res = simulate_vol_targeted_backtest(
            prices[["BTC/USDT", "ETH/USDT"]],
            base_weights=base_w,
            target_vol_ann=0.25,
            lookback_bars=15,
        )
        data = to_dict_vol_target_backtest(res, include_nav=False)
        self.assertIn("mdd_targeted_pct", data)
        self.assertIn("sharpe_targeted", data)
        self.assertIn("calmar_targeted", data)
        self.assertIn("mean_scalar", data)
        self.assertNotIn("nav_targeted", data)

        data_with_nav = to_dict_vol_target_backtest(res, include_nav=True)
        self.assertIn("nav_targeted", data_with_nav)
        self.assertIn("nav_static", data_with_nav)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test exporting backtest dict
            out_file = Path(tmpdir) / "sub" / "vol_target.json"
            export_vol_target_json(res, out_file)
            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIn("mdd_targeted_pct", loaded)
            self.assertIn("sharpe_targeted", loaded)

            # Test exporting VolTargetResult object
            vt_obj = apply_volatility_targeting(base_w, realized_vol_ann=0.50, target_vol_ann=0.25)
            obj_file = Path(tmpdir) / "sub" / "vt_obj.json"
            export_vol_target_json(vt_obj, obj_file)
            self.assertTrue(obj_file.exists())
            with open(obj_file, "r", encoding="utf-8") as f:
                obj_data = json.load(f)
            self.assertEqual(obj_data["target_vol_ann"], 0.25)
            self.assertEqual(obj_data["vol_scalar"], 0.5)

    def test_export_vol_target_csv(self):
        prices = generate_synthetic_crypto_data(periods=60)
        base_w = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        res = simulate_vol_targeted_backtest(
            prices[["BTC/USDT", "ETH/USDT"]],
            base_weights=base_w,
            target_vol_ann=0.25,
            lookback_bars=15,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sub" / "vol_target.csv"
            export_vol_target_csv(res, out_file)
            self.assertTrue(out_file.exists())
            df = pd.read_csv(out_file)
            self.assertIn("metric", df.columns)
            self.assertIn("value", df.columns)
            metrics = dict(zip(df["metric"], df["value"]))
            self.assertIn("mdd_targeted_pct", metrics)
            self.assertIn("sharpe_targeted", metrics)

            vt_obj = apply_volatility_targeting(base_w, realized_vol_ann=0.50, target_vol_ann=0.25)
            obj_file = Path(tmpdir) / "sub" / "vt_obj.csv"
            export_vol_target_csv(vt_obj, obj_file)
            self.assertTrue(obj_file.exists())
            df_obj = pd.read_csv(obj_file)
            metrics_obj = dict(zip(df_obj["metric"], df_obj["value"]))
            self.assertAlmostEqual(float(metrics_obj["target_vol_ann"]), 0.25)
            self.assertAlmostEqual(float(metrics_obj["vol_scalar"]), 0.5)


if __name__ == "__main__":
    unittest.main()

