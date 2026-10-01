"""Unit tests for Kimchi Premium regime-based tactical allocation module."""

from __future__ import annotations

import unittest
from scripts.crypto_kimchi_regime import (
    KimchiRegime,
    adjust_portfolio_weights_by_kimchi,
    classify_kimchi_regime,
)


class KimchiRegimeTests(unittest.TestCase):
    def test_classify_kimchi_regime_thresholds(self):
        # Extreme Overheated
        res_over = classify_kimchi_regime(7.5)
        self.assertEqual(res_over["regime"], KimchiRegime.EXTREME_OVERHEATED)
        self.assertEqual(res_over["target_cash_ratio"], 0.60)

        # Moderate Overheated
        res_mod = classify_kimchi_regime(4.0)
        self.assertEqual(res_mod["regime"], KimchiRegime.MODERATE_OVERHEATED)
        self.assertEqual(res_mod["target_cash_ratio"], 0.30)

        # Fair Equilibrium
        res_fair = classify_kimchi_regime(1.5)
        self.assertEqual(res_fair["regime"], KimchiRegime.FAIR_EQUILIBRIUM)
        self.assertEqual(res_fair["target_cash_ratio"], 0.15)

        # Negative Discount
        res_disc = classify_kimchi_regime(-2.0)
        self.assertEqual(res_disc["regime"], KimchiRegime.NEGATIVE_DISCOUNT)
        self.assertEqual(res_disc["target_crypto_ratio"], 0.95)

    def test_custom_moderate_threshold(self):
        # Custom moderate threshold at 2.0 with overheated at 4.0
        res = classify_kimchi_regime(2.5, overheated_threshold=4.0, discount_threshold=0.0, moderate_threshold=2.0)
        self.assertEqual(res["regime"], KimchiRegime.MODERATE_OVERHEATED)
        self.assertEqual(res["moderate_threshold"], 2.0)

        # Invalid moderate threshold (out of bounds)
        with self.assertRaises(ValueError):
            classify_kimchi_regime(2.5, overheated_threshold=4.0, discount_threshold=0.0, moderate_threshold=4.5)
        with self.assertRaises(ValueError):
            classify_kimchi_regime(2.5, overheated_threshold=4.0, discount_threshold=0.0, moderate_threshold=-1.0)

    def test_adjust_portfolio_weights_by_kimchi_sums_to_one(self):
        base_weights = {
            "KRW-BTC": 0.50,
            "KRW-ETH": 0.30,
            "KRW-SOL": 0.20,
        }
        for prem in [-3.0, 1.0, 4.5, 8.0]:
            adj = adjust_portfolio_weights_by_kimchi(base_weights, premium_pct=prem)
            self.assertIn("KRW", adj)
            total_weight = sum(adj.values())
            self.assertAlmostEqual(total_weight, 1.0, places=2)

    def test_extreme_overheated_scales_down_crypto(self):
        base_weights = {"KRW-BTC": 0.60, "KRW-ETH": 0.40}
        adj = adjust_portfolio_weights_by_kimchi(base_weights, premium_pct=9.0)
        # Cash should be ~60%
        self.assertAlmostEqual(adj["KRW"], 0.60, places=2)
        self.assertAlmostEqual(adj["KRW-BTC"], 0.24, places=2)
        self.assertAlmostEqual(adj["KRW-ETH"], 0.16, places=2)

    def test_negative_discount_maximizes_crypto(self):
        base_weights = {"KRW-BTC": 0.60, "KRW-ETH": 0.40}
        adj = adjust_portfolio_weights_by_kimchi(base_weights, premium_pct=-2.5)
        # Cash should be ~5%
        self.assertAlmostEqual(adj["KRW"], 0.05, places=2)
        self.assertAlmostEqual(adj["KRW-BTC"] + adj["KRW-ETH"], 0.95, places=2)

    def test_invalid_parameters_rejected(self):
        # Invalid premium
        with self.assertRaises(ValueError):
            classify_kimchi_regime(True)  # type: ignore
        with self.assertRaises(ValueError):
            classify_kimchi_regime(float("nan"))
        with self.assertRaises(ValueError):
            classify_kimchi_regime(float("inf"))

        # Contradictory thresholds
        with self.assertRaises(ValueError):
            classify_kimchi_regime(3.0, overheated_threshold=0.0, discount_threshold=5.0)

        # Invalid base weights
        with self.assertRaises(ValueError):
            adjust_portfolio_weights_by_kimchi({}, premium_pct=2.0)
        with self.assertRaises(ValueError):
            adjust_portfolio_weights_by_kimchi({"BTC": -0.5}, premium_pct=2.0)
        with self.assertRaises(ValueError):
            adjust_portfolio_weights_by_kimchi({"BTC": True}, premium_pct=2.0)  # type: ignore
        with self.assertRaises(ValueError):
            adjust_portfolio_weights_by_kimchi({"BTC": 0.0, "ETH": 0.0}, premium_pct=2.0)

    def test_export_kimchi_regime_csv(self):
        import tempfile
        from pathlib import Path
        import pandas as pd
        from scripts.crypto_kimchi_regime import export_kimchi_regime_csv

        regime_info = {
            "premium_pct": 4.5,
            "regime": "MODERATE_OVERHEATED",
            "target_crypto_ratio": 0.70,
            "target_cash_ratio": 0.30,
            "tactical_action": "Moderate De-risking",
        }
        adjusted_weights = {
            "KRW-BTC": 0.35,
            "KRW-ETH": 0.35,
            "KRW": 0.30,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sub" / "kimchi_regime.csv"
            res_path = export_kimchi_regime_csv(regime_info, adjusted_weights, out_file)
            self.assertEqual(res_path, out_file)
            self.assertTrue(out_file.exists())

            df = pd.read_csv(out_file)
            self.assertIn("section", df.columns)
            self.assertIn("key", df.columns)
            self.assertIn("value", df.columns)
            self.assertIn("regime", df.columns)
            self.assertTrue(any(df["key"] == "KRW-BTC"))
            self.assertTrue(any(df["key"] == "regime"))


if __name__ == "__main__":
    unittest.main()

