"""Unit tests for Korea Travel Rule compliance and safe transfer advisor."""

from __future__ import annotations

import unittest
from scripts.crypto_travel_rule_advisor import (
    COMMON_REMITTANCE_FEE_PRESETS,
    calculate_travel_rule_plan,
    get_coin_transfer_preset,
)


class TravelRuleAdvisorTests(unittest.TestCase):
    def test_get_coin_transfer_preset(self):
        self.assertEqual(get_coin_transfer_preset("XRP"), 1.0)
        self.assertEqual(get_coin_transfer_preset("KRW-SOL"), 0.01)
        self.assertEqual(get_coin_transfer_preset("TRX/USDT"), 1.0)
        self.assertEqual(get_coin_transfer_preset("UNKNOWN"), 0.0)
        with self.assertRaises(ValueError):
            get_coin_transfer_preset(123)  # type: ignore

    def test_auto_preset_fee_application(self):
        res = calculate_travel_rule_plan(
            coin_symbol="XRP",
            target_amount=400.0,
            coin_price_krw=2000.0,
            auto_preset_fee=True,
        )
        self.assertEqual(res["total_network_fee_coins"], 1.0)

    def test_under_threshold_single_transfer(self):
        # 400 XRP * 2,000 KRW = 800,000 KRW (< 1,000,000 KRW)
        res = calculate_travel_rule_plan(
            coin_symbol="XRP",
            target_amount=400.0,
            coin_price_krw=2000.0,
            network_fee_coins=0.5,
        )
        self.assertFalse(res["requires_travel_rule"])
        self.assertEqual(res["recommended_batches"], 1)
        self.assertEqual(res["per_batch_coins"], 400.0)
        self.assertEqual(res["total_value_krw"], 800000.0)
        self.assertEqual(res["total_network_fee_coins"], 0.5)

    def test_over_threshold_batch_splitting(self):
        # 3,000 XRP * 1,900 KRW = 5,700,000 KRW (> 1,000,000 KRW)
        # Safe buffer = 950,000 KRW -> 5,700,000 / 950,000 = 6 batches exactly
        res = calculate_travel_rule_plan(
            coin_symbol="XRP",
            target_amount=3000.0,
            coin_price_krw=1900.0,
            safe_buffer_krw=950000.0,
            network_fee_coins=1.0,
        )
        self.assertTrue(res["requires_travel_rule"])
        self.assertEqual(res["recommended_batches"], 6)
        self.assertAlmostEqual(res["per_batch_coins"], 500.0, places=2)
        self.assertLessEqual(res["per_batch_krw"], 950000.0)
        self.assertEqual(res["total_network_fee_coins"], 6.0)

    def test_network_fees_compounding(self):
        # 10,000 TRX * 200 KRW = 2,000,000 KRW -> 3 batches (2,000,000 / 950,000 = 2.1 -> 3)
        res = calculate_travel_rule_plan(
            coin_symbol="TRX",
            target_amount=10000.0,
            coin_price_krw=200.0,
            network_fee_coins=2.0,
        )
        self.assertEqual(res["recommended_batches"], 3)
        self.assertEqual(res["total_network_fee_coins"], 6.0)
        self.assertEqual(res["total_network_fee_krw"], 1200.0)

    def test_invalid_parameters_rejected(self):
        # Invalid coin symbol
        with self.assertRaises(ValueError):
            calculate_travel_rule_plan(coin_symbol="", target_amount=10.0, coin_price_krw=100.0)
        with self.assertRaises(ValueError):
            calculate_travel_rule_plan(coin_symbol=True, target_amount=10.0, coin_price_krw=100.0)  # type: ignore

        # Invalid target amount
        with self.assertRaises(ValueError):
            calculate_travel_rule_plan("BTC", target_amount=0.0, coin_price_krw=1000.0)
        with self.assertRaises(ValueError):
            calculate_travel_rule_plan("BTC", target_amount=-5.0, coin_price_krw=1000.0)
        with self.assertRaises(ValueError):
            calculate_travel_rule_plan("BTC", target_amount=True, coin_price_krw=1000.0)  # type: ignore

        # Invalid coin price
        with self.assertRaises(ValueError):
            calculate_travel_rule_plan("BTC", target_amount=1.0, coin_price_krw=0.0)
        with self.assertRaises(ValueError):
            calculate_travel_rule_plan("BTC", target_amount=1.0, coin_price_krw=False)  # type: ignore

        # Safe buffer >= threshold
        with self.assertRaises(ValueError):
            calculate_travel_rule_plan(
                "BTC",
                target_amount=1.0,
                coin_price_krw=1000.0,
                threshold_krw=1000000.0,
                safe_buffer_krw=1000000.0,
            )

        # Invalid network fee
        with self.assertRaises(ValueError):
            calculate_travel_rule_plan("BTC", target_amount=1.0, coin_price_krw=1000.0, network_fee_coins=-0.1)

    def test_interval_scheduling_and_anti_structuring(self):
        # 3 batches with 15 min interval -> 2 gaps = 30 minutes total
        res = calculate_travel_rule_plan(
            coin_symbol="XRP",
            target_amount=1200.0,
            coin_price_krw=2000.0,  # 2.4M KRW -> 3 batches
            interval_minutes=15,
        )
        self.assertEqual(res["recommended_batches"], 3)
        self.assertEqual(res["batch_interval_minutes"], 15)
        self.assertEqual(res["total_duration_minutes"], 30)
        self.assertFalse(res["anti_structuring_alert"])

        # High amount (6M KRW) should trigger anti-structuring alert
        res_high = calculate_travel_rule_plan(
            coin_symbol="BTC",
            target_amount=0.06,
            coin_price_krw=100000000.0,  # 6M KRW
            daily_warning_threshold_krw=5000000.0,
        )
        self.assertTrue(res_high["anti_structuring_alert"])
        self.assertIn("FDS/STR", res_high["anti_structuring_note"])


if __name__ == "__main__":
    unittest.main()
