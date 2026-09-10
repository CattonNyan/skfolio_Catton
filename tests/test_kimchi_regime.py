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


if __name__ == "__main__":
    unittest.main()
