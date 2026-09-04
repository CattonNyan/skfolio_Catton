"""Tests for Kimchi Premium analyzer module."""

import unittest
from scripts.crypto_kimchi_premium import compute_kimchi_premium


class KimchiPremiumTests(unittest.TestCase):
    def test_compute_kimchi_premium_exact(self):
        # 100 USDT * 1,000 KRW/USDT = 100,000 Fair KRW
        # Upbit price = 105,000 KRW -> Premium = +5.0%
        upbit = {"BTC": 105000.0}
        binance = {"BTC": 100.0}
        res = compute_kimchi_premium(upbit, binance, usdt_krw_rate=1000.0)

        self.assertIn("BTC", res)
        self.assertEqual(res["BTC"]["premium_pct"], 5.0)
        self.assertEqual(res["BTC"]["krw_difference"], 5000.0)
        self.assertEqual(res["BTC"]["fair_krw"], 100000.0)

    def test_negative_exchange_rate_rejected(self):
        with self.assertRaises(ValueError):
            compute_kimchi_premium({"BTC": 1000.0}, {"BTC": 1.0}, usdt_krw_rate=-100.0)

    def test_discount_status_labeling(self):
        # Upbit 95,000 KRW vs Fair 100,000 KRW -> -5% Discount
        upbit = {"ETH": 95000.0}
        binance = {"ETH": 100.0}
        res = compute_kimchi_premium(upbit, binance, usdt_krw_rate=1000.0)
        self.assertIn("Discount", res["ETH"]["status"])

    def test_fetch_live_usd_krw_rate(self):
        from scripts.crypto_kimchi_premium import fetch_live_usd_krw_rate
        rate, source = fetch_live_usd_krw_rate(timeout=2.0)
        self.assertIsInstance(rate, float)
        self.assertGreater(rate, 500.0)
        self.assertIsInstance(source, str)

    def test_multi_coins_arbitrage_computation(self):
        upbit = {"BTC": 105000.0, "DOGE": 210.0}
        binance = {"BTC": 100.0, "DOGE": 0.20}
        res = compute_kimchi_premium(upbit, binance, usdt_krw_rate=1000.0)
        self.assertIn("BTC", res)
        self.assertIn("DOGE", res)
        self.assertEqual(res["DOGE"]["premium_pct"], 5.0)


if __name__ == "__main__":
    unittest.main()
