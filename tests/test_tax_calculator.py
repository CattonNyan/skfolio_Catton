"""Tests for Crypto Capital Gains Tax Simulator module."""

import unittest
from scripts.crypto_tax_calculator import compute_crypto_tax_impact


class TaxCalculatorTests(unittest.TestCase):
    def test_compute_crypto_tax_basic(self):
        # Gains: 10M, Losses: 2M -> Net: 8M
        # Allowance: 2.5M -> Taxable Base: 5.5M
        # Tax: 5.5M * 22% = 1.21M
        trades = [6000000.0, 4000000.0, -2000000.0]
        res = compute_crypto_tax_impact(
            realized_profits=trades,
            annual_allowance_krw=2500000.0,
            tax_rate=0.22,
            initial_capital_krw=50000000.0,
        )

        self.assertEqual(res["net_realized_profit"], 8000000.0)
        self.assertEqual(res["gross_realized_gains"], 10000000.0)
        self.assertEqual(res["gross_realized_losses"], 2000000.0)
        self.assertEqual(res["taxable_base"], 5500000.0)
        self.assertEqual(res["estimated_tax_krw"], 1210000.0)
        self.assertEqual(res["after_tax_profit_krw"], 8000000.0 - 1210000.0)
        self.assertTrue(res["is_taxable"])

    def test_under_allowance_zero_tax(self):
        # Profit 2,000,000 KRW is under 2,500,000 KRW allowance -> Tax = 0
        trades = [2000000.0]
        res = compute_crypto_tax_impact(
            realized_profits=trades,
            annual_allowance_krw=2500000.0,
            tax_rate=0.22,
        )
        self.assertEqual(res["taxable_base"], 0.0)
        self.assertEqual(res["estimated_tax_krw"], 0.0)
        self.assertFalse(res["is_taxable"])

    def test_invalid_tax_rate_rejected(self):
        with self.assertRaises(ValueError):
            compute_crypto_tax_impact([1000000.0], tax_rate=1.5)
        with self.assertRaises(ValueError):
            compute_crypto_tax_impact([1000000.0], tax_rate=-0.1)

    def test_invalid_capital_rejected(self):
        with self.assertRaises(ValueError):
            compute_crypto_tax_impact([1000000.0], initial_capital_krw=-500.0)


if __name__ == "__main__":
    unittest.main()
