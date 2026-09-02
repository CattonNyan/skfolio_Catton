"""Tests for Crypto Macro Regime cash allocator module."""

import unittest
from scripts.crypto_macro_regime import (
    adjust_cash_allocation_by_regime,
    fetch_fear_and_greed_index,
)


class MacroRegimeTests(unittest.TestCase):
    def test_extreme_greed_allocates_high_cash(self):
        base_w = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        res = adjust_cash_allocation_by_regime(base_w, fng_value=85, total_wallet=10000.0)

        self.assertEqual(res["market_regime"], "Extreme Greed")
        self.assertEqual(res["cash_ratio"], 0.40)
        self.assertIn("USDT (Cash)", res["adjusted_weights"])
        self.assertEqual(res["adjusted_weights"]["USDT (Cash)"], 0.40)
        self.assertAlmostEqual(sum(res["adjusted_weights"].values()), 1.0, places=3)

    def test_extreme_fear_full_deployment(self):
        base_w = {"BTC/USDT": 0.7, "ETH/USDT": 0.3}
        res = adjust_cash_allocation_by_regime(base_w, fng_value=15, total_wallet=10000.0)

        self.assertEqual(res["market_regime"], "Extreme Fear")
        self.assertEqual(res["cash_ratio"], 0.0)
        self.assertNotIn("USDT (Cash)", res["adjusted_weights"])
        self.assertAlmostEqual(sum(res["adjusted_weights"].values()), 1.0, places=3)

    def test_out_of_range_fng_rejected(self):
        with self.assertRaises(ValueError):
            adjust_cash_allocation_by_regime({"BTC/USDT": 1.0}, fng_value=120)

    def test_fetch_fear_and_greed_returns_valid_tuple(self):
        val, label = fetch_fear_and_greed_index()
        self.assertIsInstance(val, int)
        self.assertIsInstance(label, str)
        self.assertGreaterEqual(val, 0)
        self.assertLessEqual(val, 100)

    def test_filter_crypto_weights_for_freqtrade(self):
        from scripts.crypto_macro_regime import filter_crypto_weights_for_freqtrade

        raw_weights = {"BTC/USDT": 0.35, "ETH/USDT": 0.25, "USDT (Cash)": 0.40}
        clean_pairs, cash = filter_crypto_weights_for_freqtrade(raw_weights)

        self.assertEqual(cash, 0.40)
        self.assertNotIn("USDT (Cash)", clean_pairs)
        self.assertIn("BTC/USDT", clean_pairs)
        self.assertIn("ETH/USDT", clean_pairs)


if __name__ == "__main__":
    unittest.main()
