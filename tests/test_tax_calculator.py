"""Tests for Crypto Capital Gains Tax Simulator module."""

import unittest
from scripts.crypto_tax_calculator import (
    calculate_tax_loss_harvesting_target,
    compute_crypto_tax_impact,
)


class TaxCalculatorTests(unittest.TestCase):
    def test_tax_loss_harvesting_target_calculation(self):
        # 10M net profit > 2.5M allowance -> need 7.5M loss harvest
        res = calculate_tax_loss_harvesting_target(10000000.0, annual_allowance_krw=2500000.0, tax_rate=0.22)
        self.assertTrue(res["needs_harvesting"])
        self.assertEqual(res["taxable_excess_krw"], 7500000.0)
        self.assertEqual(res["recommended_loss_harvest_krw"], 7500000.0)
        self.assertEqual(res["potential_tax_savings_krw"], 7500000.0 * 0.22)

        # Under allowance -> 0 loss harvest needed
        res_under = calculate_tax_loss_harvesting_target(2000000.0, annual_allowance_krw=2500000.0)
        self.assertFalse(res_under["needs_harvesting"])
        self.assertEqual(res_under["recommended_loss_harvest_krw"], 0.0)

        # Invalid arguments
        with self.assertRaises(ValueError):
            calculate_tax_loss_harvesting_target(True)  # type: ignore
        with self.assertRaises(ValueError):
            calculate_tax_loss_harvesting_target(5000000.0, tax_rate=1.5)

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
        for tax_rate in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(tax_rate=tax_rate), self.assertRaises(ValueError):
                compute_crypto_tax_impact([1000000.0], tax_rate=tax_rate)

    def test_invalid_allowance_rejected(self):
        for allowance in (-1.0, float("nan"), float("inf")):
            with self.subTest(allowance=allowance), self.assertRaises(ValueError):
                compute_crypto_tax_impact(
                    [1000000.0], annual_allowance_krw=allowance
                )

    def test_invalid_capital_rejected(self):
        with self.assertRaises(ValueError):
            compute_crypto_tax_impact([1000000.0], initial_capital_krw=-500.0)

    def test_non_finite_profit_rejected(self):
        for profit in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(profit=profit), self.assertRaises(ValueError):
                compute_crypto_tax_impact([1000000.0, profit])

    def test_bool_parameters_and_fx_rate_rejected(self):
        for bad_bool in (True, False):
            with self.subTest(bad_bool=bad_bool):
                with self.assertRaises(ValueError):
                    compute_crypto_tax_impact([1000000.0], annual_allowance_krw=bad_bool)
                with self.assertRaises(ValueError):
                    compute_crypto_tax_impact([1000000.0], tax_rate=bad_bool)
                with self.assertRaises(ValueError):
                    compute_crypto_tax_impact([1000000.0], initial_capital_krw=bad_bool)
                with self.assertRaises(ValueError):
                    compute_crypto_tax_impact([1000000.0], usdt_krw_rate=bad_bool)
                with self.assertRaises(ValueError):
                    compute_crypto_tax_impact([bad_bool])

        for bad_rate in (-1.0, 0, float("nan"), float("inf")):
            with self.subTest(bad_rate=bad_rate), self.assertRaises(ValueError):
                compute_crypto_tax_impact([1000000.0], usdt_krw_rate=bad_rate)

    def test_carried_forward_loss_netting(self):
        # 10M profit, 4M carried loss -> adjusted profit: 6M
        # 2.5M basic allowance -> taxable base: 3.5M
        # tax: 3.5M * 0.22 = 770,000 KRW
        res = compute_crypto_tax_impact(
            realized_profits=[10000000.0],
            annual_allowance_krw=2500000.0,
            tax_rate=0.22,
            carried_forward_loss_krw=4000000.0,
        )
        self.assertEqual(res["carried_forward_loss_krw"], 4000000.0)
        self.assertEqual(res["carried_loss_applied_krw"], 4000000.0)
        self.assertEqual(res["remaining_carried_loss_krw"], 0.0)
        self.assertEqual(res["taxable_base"], 3500000.0)
        self.assertEqual(res["estimated_tax_krw"], 770000.0)

        # Invalid carried loss
        with self.assertRaises(ValueError):
            compute_crypto_tax_impact([1000000.0], carried_forward_loss_krw=-500.0)
        with self.assertRaises(ValueError):
            compute_crypto_tax_impact([1000000.0], carried_forward_loss_krw=True)


if __name__ == "__main__":
    unittest.main()
