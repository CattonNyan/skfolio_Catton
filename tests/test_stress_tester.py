"""Tests for historical crypto stress tester module."""

import unittest
from scripts.crypto_stress_tester import evaluate_stress_test


class StressTesterTests(unittest.TestCase):
    def test_evaluate_stress_test_basic(self):
        weights = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        wallet = 10000.0
        results = evaluate_stress_test(weights, total_wallet=wallet)

        self.assertIn("2020 March Covid Crash", results)
        self.assertIn("2022 May Luna/UST Collapse", results)
        self.assertIn("2022 Nov FTX Insolvency", results)
        self.assertIn("2021 May China Mining Ban", results)

        for name, metrics in results.items():
            self.assertLess(metrics["portfolio_loss_pct"], 0)
            self.assertAlmostEqual(
                metrics["dollar_loss"] + metrics["remaining_balance"],
                wallet,
                places=1,
            )

    def test_solana_heavy_ftx_shock(self):
        # Heavy SOL allocation should suffer worst in FTX collapse
        sol_heavy = {"SOL/USDT": 0.9, "BTC/USDT": 0.1}
        results = evaluate_stress_test(sol_heavy, total_wallet=10000.0)
        ftx_loss = results["2022 Nov FTX Insolvency"]["portfolio_loss_pct"]
        luna_loss = results["2022 May Luna/UST Collapse"]["portfolio_loss_pct"]
        # FTX shock (-58% for SOL) should be deeper than Luna shock (-38% for SOL)
        self.assertLess(ftx_loss, luna_loss)

    def test_custom_shock_injection(self):
        weights = {"BTC/USDT": 1.0}
        custom = {"BTC": -0.70}
        results = evaluate_stress_test(
            weights,
            total_wallet=1000.0,
            custom_shock=custom,
        )
        self.assertIn("Custom User Shock", results)
        self.assertEqual(results["Custom User Shock"]["portfolio_loss_pct"], -70.0)
        self.assertEqual(results["Custom User Shock"]["dollar_loss"], 700.0)
        self.assertEqual(results["Custom User Shock"]["remaining_balance"], 300.0)


if __name__ == "__main__":
    unittest.main()
