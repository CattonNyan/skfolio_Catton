import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd

from scripts.crypto_stress_tester import (
    evaluate_stress_test,
    export_stress_test_json,
    export_stress_test_csv,
    summarize_stress_test_results,
    to_dict_stress_test_result,
)


class StressTesterTests(unittest.TestCase):
    def test_evaluate_stress_test_basic(self):
        weights = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        wallet = 10000.0
        results = evaluate_stress_test(weights, total_wallet=wallet)

        self.assertIn("2020 March Covid Crash", results)
        self.assertIn("2022 May Luna/UST Collapse", results)
        self.assertIn("2022 Nov FTX Insolvency", results)
        self.assertIn("2021 May China Mining Ban", results)
        self.assertIn("2024 August Yen Carry Trade Unwind", results)
        self.assertIn("2023 March SVB USDC Depeg Crisis", results)

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

    def test_symbol_format_normalization(self):
        # Test "BTC_USDT", "BTC:USDT", and plain "BTC" should match historical "BTC" shock (-48% Covid)
        w_slash = {"BTC/USDT": 1.0}
        w_underscore = {"BTC_USDT": 1.0}
        w_colon = {"BTC:USDT": 1.0}
        w_plain = {"BTC": 1.0}

        r1 = evaluate_stress_test(w_slash)["2020 March Covid Crash"]["portfolio_loss_pct"]
        r2 = evaluate_stress_test(w_underscore)["2020 March Covid Crash"]["portfolio_loss_pct"]
        r3 = evaluate_stress_test(w_colon)["2020 March Covid Crash"]["portfolio_loss_pct"]
        r4 = evaluate_stress_test(w_plain)["2020 March Covid Crash"]["portfolio_loss_pct"]

        self.assertEqual(r1, r2)
        self.assertEqual(r1, r3)
        self.assertEqual(r1, r4)
        self.assertEqual(r1, -38.0)

    def test_unnormalized_weights_auto_scaled(self):
        # Weights given in percentage sum = 100 instead of 1.0
        w_pct = {"BTC/USDT": 50.0, "ETH/USDT": 50.0}
        w_norm = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        r_pct = evaluate_stress_test(w_pct, total_wallet=10000.0)
        r_norm = evaluate_stress_test(w_norm, total_wallet=10000.0)
        self.assertEqual(
            r_pct["2020 March Covid Crash"]["portfolio_loss_pct"],
            r_norm["2020 March Covid Crash"]["portfolio_loss_pct"],
        )

    def test_invalid_wallet_rejected(self):
        for wallet in (-1000.0, 0.0, float("nan"), float("inf")):
            with self.subTest(wallet=wallet), self.assertRaises(ValueError):
                evaluate_stress_test({"BTC/USDT": 1.0}, total_wallet=wallet)

    def test_non_finite_weight_rejected(self):
        for weight in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                evaluate_stress_test(
                    {"BTC/USDT": 1.0, "ETH/USDT": weight}
                )

    def test_invalid_custom_shock_and_types(self):
        for bad_wallet in (True, False):
            with self.subTest(bad_wallet=bad_wallet), self.assertRaises(ValueError):
                evaluate_stress_test({"BTC/USDT": 1.0}, total_wallet=bad_wallet)

        for bad_weights in ("not_a_dict", [1, 2], {}):
            with self.subTest(bad_weights=bad_weights), self.assertRaises(ValueError):
                evaluate_stress_test(bad_weights)

        for bad_shock in ("not_a_dict", {}, {"": -0.2}, {"BTC": True}, {"BTC": "bad"}, {"BTC": -1.5}):
            with self.subTest(bad_shock=bad_shock), self.assertRaises(ValueError):
                evaluate_stress_test({"BTC/USDT": 1.0}, custom_shock=bad_shock)


    def test_recovery_required_and_new_scenarios(self):
        weights = {"BTC/USDT": 0.5, "ETH/USDT": 0.3, "SOL/USDT": 0.2}
        results = evaluate_stress_test(weights, total_wallet=10000.0)

        # Check new shock scenarios present
        self.assertIn("2024 October Geopolitical De-escalation Shock", results)
        self.assertIn("DeFi Liquidity Cascade & Flash Crash", results)

        # Verify recovery_required_pct
        for name, metrics in results.items():
            self.assertIn("recovery_required_pct", metrics)
            loss_pct = abs(metrics["portfolio_loss_pct"]) / 100.0
            expected_rec = (loss_pct / (1.0 - loss_pct)) * 100
            self.assertAlmostEqual(metrics["recovery_required_pct"], expected_rec, places=1)

    def test_summarize_stress_test_results(self):
        # Empty results handling
        empty_summary = summarize_stress_test_results({})
        self.assertEqual(empty_summary["scenarios_tested"], 0)
        self.assertEqual(empty_summary["worst_scenario"], "")

        # Standard stress test evaluation summary
        weights = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        wallet = 10000.0
        results = evaluate_stress_test(weights, total_wallet=wallet)
        summary = summarize_stress_test_results(results)

        self.assertGreater(summary["scenarios_tested"], 0)
        self.assertIn("worst_scenario", summary)
        self.assertLess(summary["worst_loss_pct"], 0.0)
        self.assertGreater(summary["worst_dollar_loss"], 0.0)
        self.assertLess(summary["avg_loss_pct"], 0.0)
        self.assertLessEqual(summary["worst_loss_pct"], summary["avg_loss_pct"])
        self.assertIn(summary["overall_resilience"], ["A (High Resilience)", "B (Moderate Resilience)", "C (Significant Impact)", "D (Severe Vulnerability)"])

    def test_to_dict_and_export_json(self):
        weights = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        wallet = 10000.0
        results = evaluate_stress_test(weights, total_wallet=wallet)
        payload = to_dict_stress_test_result(results, total_wallet=wallet, weights=weights)
        self.assertIn("scenarios", payload)
        self.assertIn("summary", payload)
        self.assertEqual(payload["total_wallet"], 10000.0)
        self.assertEqual(payload["weights"]["BTC/USDT"], 0.6)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sub" / "stress.json"
            export_stress_test_json(results, out_file, total_wallet=wallet, weights=weights)
            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIn("scenarios", loaded)
            self.assertIn("summary", loaded)
            self.assertEqual(loaded["total_wallet"], 10000.0)

    def test_export_stress_test_csv(self):
        weights = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        wallet = 10000.0
        results = evaluate_stress_test(weights, total_wallet=wallet)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sub" / "stress.csv"
            export_stress_test_csv(results, out_file)
            self.assertTrue(out_file.exists())
            df = pd.read_csv(out_file)
            self.assertIn("scenario", df.columns)
            self.assertIn("portfolio_loss_pct", df.columns)
            self.assertIn("dollar_loss", df.columns)
            self.assertIn("remaining_balance", df.columns)
            self.assertIn("recovery_required_pct", df.columns)
            self.assertIn("resilience_grade", df.columns)
            self.assertEqual(len(df), len(results))

    def test_new_crash_scenarios_and_severity(self):
        weights = {"BTC/USDT": 0.5, "ETH/USDT": 0.3, "SOL/USDT": 0.2}
        results = evaluate_stress_test(weights, total_wallet=10000.0)

        # 3 new historical & exploit scenarios
        self.assertIn("2018 Jan Crypto Winter Freeze", results)
        self.assertIn("2021 Nov ATH Liquidation Cascade", results)
        self.assertIn("DeFi Staking & Bridge Exploit Shock", results)

        # Winter freeze should be deeper than COVID crash
        freeze_loss = results["2018 Jan Crypto Winter Freeze"]["portfolio_loss_pct"]
        covid_loss = results["2020 March Covid Crash"]["portfolio_loss_pct"]
        self.assertLess(freeze_loss, covid_loss)

    def test_asset_level_vulnerability_breakdown(self):
        weights = {"BTC/USDT": 0.5, "SOL/USDT": 0.5}
        wallet = 10000.0
        results = evaluate_stress_test(weights, total_wallet=wallet)

        ftx = results["2022 Nov FTX Insolvency"]
        self.assertIn("asset_breakdown", ftx)
        self.assertEqual(ftx["primary_risk_driver"], "SOL/USDT")

        breakdown = ftx["asset_breakdown"]
        self.assertIn("BTC/USDT", breakdown)
        self.assertIn("SOL/USDT", breakdown)

        # Contribution sum must equal total portfolio loss
        total_contrib = sum(b["contribution_pct"] for b in breakdown.values())
        self.assertAlmostEqual(total_contrib, ftx["portfolio_loss_pct"], places=2)

        # SOL contribution (-29.0%) is greater in magnitude than BTC (-11.5%)
        self.assertLess(breakdown["SOL/USDT"]["contribution_pct"], breakdown["BTC/USDT"]["contribution_pct"])
        self.assertGreater(breakdown["SOL/USDT"]["loss_share_pct"], breakdown["BTC/USDT"]["loss_share_pct"])

    def test_cash_buffer_hedging(self):
        weights = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        wallet = 10000.0

        unhedged = evaluate_stress_test(weights, total_wallet=wallet, cash_buffer=0.0)
        hedged = evaluate_stress_test(weights, total_wallet=wallet, cash_buffer=0.25)

        for sc_name in unhedged:
            unh_loss = unhedged[sc_name]["portfolio_loss_pct"]
            hdg_loss = hedged[sc_name]["portfolio_loss_pct"]
            # 25% cash buffer should reduce loss by exactly 25%
            self.assertAlmostEqual(hdg_loss, unh_loss * 0.75, places=2)
            self.assertGreater(hedged[sc_name]["capital_saved_dollars"], 0.0)
            self.assertEqual(hedged[sc_name]["cash_buffer_pct"], 25.0)

    def test_invalid_cash_buffer_rejected(self):
        for bad_buffer in (-0.1, 1.0, 1.5, float("nan"), float("inf"), True, False):
            if bad_buffer is False:
                continue  # 0.0 is valid
            with self.subTest(bad_buffer=bad_buffer), self.assertRaises(ValueError):
                evaluate_stress_test({"BTC/USDT": 1.0}, cash_buffer=bad_buffer)

    def test_svar_and_ses_summary_metrics(self):
        weights = {"BTC/USDT": 0.7, "ETH/USDT": 0.3}
        results = evaluate_stress_test(weights, total_wallet=10000.0)
        summary = summarize_stress_test_results(results)

        self.assertIn("svar_95_pct", summary)
        self.assertIn("ses_95_pct", summary)
        self.assertIn("median_loss_pct", summary)
        self.assertIn("primary_vulnerability", summary)

        # Expected shortfall in the tail must be as severe or worse than VaR
        self.assertLessEqual(summary["ses_95_pct"], summary["svar_95_pct"])
        self.assertLessEqual(summary["svar_95_pct"], summary["avg_loss_pct"])
        self.assertLess(summary["svar_95_pct"], 0.0)

    def test_export_asset_level_csv(self):
        weights = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        wallet = 10000.0
        results = evaluate_stress_test(weights, total_wallet=wallet)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "asset_stress.csv"
            export_stress_test_csv(results, out_file, include_asset_breakdown=True)
            self.assertTrue(out_file.exists())
            df = pd.read_csv(out_file)

            self.assertIn("scenario", df.columns)
            self.assertIn("asset", df.columns)
            self.assertIn("effective_weight", df.columns)
            self.assertIn("asset_drop_pct", df.columns)
            self.assertIn("contribution_pct", df.columns)
            self.assertIn("dollar_impact", df.columns)
            self.assertIn("loss_share_pct", df.columns)
            # 2 assets per scenario
            self.assertEqual(len(df), len(results) * 2)


if __name__ == "__main__":
    unittest.main()
