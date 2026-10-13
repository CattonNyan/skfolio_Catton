import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.crypto_tail_dependence import (
    CopulaConcordanceResult,
    TailDependenceResult,
    compute_bivariate_tail_dependence,
    compute_copula_rank_concordance,
    compute_tail_dependence_matrix,
    compute_tail_diversification_index,
    export_tail_dependence_csv,
    export_tail_dependence_json,
    main as tail_main,
    simulate_tail_shock_portfolio_loss,
    to_dict_tail_dependence_matrix,
)


class TailDependenceTests(unittest.TestCase):
    def test_strongly_co_dependent_assets(self):
        # Two assets with common crash shock
        rng = np.random.default_rng(42)
        n = 500
        shock = rng.normal(0, 0.05, n)
        r1 = shock + rng.normal(0, 0.005, n)
        r2 = shock + rng.normal(0, 0.005, n)

        res = compute_bivariate_tail_dependence(r1, r2, quantile=0.05)
        self.assertGreater(res.lower_tail, 0.50)
        self.assertGreater(res.upper_tail, 0.50)
        self.assertEqual(res.quantile, 0.05)

    def test_tail_dependence_matrix(self):
        rng = np.random.default_rng(42)
        rets = pd.DataFrame({
            "BTC": rng.normal(0, 0.02, 200),
            "ETH": rng.normal(0, 0.03, 200),
            "SOL": rng.normal(0, 0.04, 200),
        })
        df_l, df_u, scores = compute_tail_dependence_matrix(rets, quantile=0.10)
        self.assertEqual(df_l.shape, (3, 3))
        self.assertEqual(df_u.shape, (3, 3))
        self.assertEqual(len(scores), 3)
        # Diagonals must be 1.0
        for col in rets.columns:
            self.assertEqual(df_l.loc[col, col], 1.0)
            self.assertEqual(df_u.loc[col, col], 1.0)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            compute_bivariate_tail_dependence([1, 2], [1, 2])
        with self.assertRaises(ValueError):
            compute_bivariate_tail_dependence([1] * 20, [1] * 20, quantile=-0.1)
        with self.assertRaises(ValueError):
            compute_bivariate_tail_dependence([1] * 20, [1] * 20, quantile=0.6)
        with self.assertRaises(ValueError):
            compute_tail_dependence_matrix(pd.DataFrame({"A": [1, 2, 3]}))
        with self.assertRaises(ValueError):
            compute_copula_rank_concordance(pd.DataFrame({"A": [1, 2, 3]}))
        with self.assertRaises(ValueError):
            simulate_tail_shock_portfolio_loss({"A": 1.0}, pd.DataFrame({"A": [1, 2, 3]}), quantile=0.05)

    def test_tail_dependence_result_to_dict(self):
        result = TailDependenceResult(
            lower_tail=0.75,
            upper_tail=0.82,
            tail_asymmetry=-0.07,
            quantile=0.05,
        )
        d = result.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["lower_tail"], 0.75)
        self.assertEqual(d["upper_tail"], 0.82)
        self.assertEqual(d["tail_asymmetry"], -0.07)
        self.assertEqual(d["quantile"], 0.05)

    def test_copula_rank_concordance(self):
        rng = np.random.default_rng(42)
        rets = pd.DataFrame({
            "BTC": rng.normal(0, 0.02, 100),
            "ETH": rng.normal(0, 0.03, 100),
            "SOL": rng.normal(0, 0.04, 100),
        })
        results = compute_copula_rank_concordance(rets, quantile=0.05)
        self.assertEqual(len(results), 3)  # 3 pairs: BTC-ETH, BTC-SOL, ETH-SOL
        first = results[0]
        self.assertIsInstance(first, CopulaConcordanceResult)
        self.assertEqual(first.asset_a, "BTC")
        self.assertEqual(first.asset_b, "ETH")
        self.assertIn("pearson_corr", first.to_dict())
        self.assertIn("spearman_rho", first.to_dict())
        self.assertIn("kendall_tau", first.to_dict())
        self.assertIn("tail_dep_gap", first.to_dict())

    def test_tail_diversification_index(self):
        lower_df = pd.DataFrame(
            [[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        # Concentrated in A and B (high tail dependence 0.8)
        w_conc = {"A": 0.5, "B": 0.5, "C": 0.0}
        tdi_conc = compute_tail_diversification_index(w_conc, lower_df)
        self.assertAlmostEqual(tdi_conc, 0.2, places=2)

        # Diversified with C (low tail dependence 0.2)
        w_div = {"A": 0.5, "B": 0.0, "C": 0.5}
        tdi_div = compute_tail_diversification_index(w_div, lower_df)
        self.assertAlmostEqual(tdi_div, 0.8, places=2)
        self.assertGreater(tdi_div, tdi_conc)

        # Single asset returns 0.0
        self.assertEqual(compute_tail_diversification_index({"A": 1.0}, lower_df), 0.0)

    def test_simulate_tail_shock_portfolio_loss(self):
        rng = np.random.default_rng(42)
        n = 300
        common_shock = rng.standard_t(df=3, size=n) * 0.03
        rets = pd.DataFrame({
            "BTC": common_shock + rng.normal(0, 0.01, n),
            "ETH": common_shock * 1.2 + rng.normal(0, 0.015, n),
            "SOL": common_shock * 1.5 + rng.normal(0, 0.02, n),
        })
        weights = {"BTC": 0.4, "ETH": 0.4, "SOL": 0.2}
        sim = simulate_tail_shock_portfolio_loss(weights, rets, quantile=0.05)
        self.assertIn("conditional_tail_loss_pct", sim)
        self.assertIn("conditional_tail_max_loss_pct", sim)
        self.assertIn("empirical_var_pct", sim)
        self.assertIn("empirical_es_pct", sim)
        self.assertIn("gaussian_var_pct", sim)
        self.assertIn("gaussian_es_pct", sim)
        self.assertIn("diversification_delta_pct", sim)
        self.assertIn("tail_diversification_index", sim)
        self.assertGreater(sim["conditional_tail_loss_pct"], 0.0)

    def test_json_export_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sub" / "tail_test.json"
            from unittest.mock import patch
            test_args = ["crypto_tail_dependence.py", "--quantile", "0.05", "--simulate-tail-shock", "--export-json", str(out_file)]
            with patch("sys.argv", test_args):
                tail_main()

            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["quantile"], 0.05)
            self.assertIn("lower_tail_matrix", data)
            self.assertIn("upper_tail_matrix", data)
            self.assertIn("systemic_crash_vulnerability", data)
            self.assertIn("copula_concordance", data)
            self.assertIn("tail_stress_simulation", data)

    def test_to_dict_and_direct_export_json(self):
        rng = np.random.default_rng(42)
        rets = pd.DataFrame({
            "BTC": rng.normal(0, 0.02, 100),
            "ETH": rng.normal(0, 0.03, 100),
        })
        df_l, df_u, scores = compute_tail_dependence_matrix(rets, quantile=0.05)
        conc = compute_copula_rank_concordance(rets, quantile=0.05)
        sim = simulate_tail_shock_portfolio_loss({"BTC": 0.5, "ETH": 0.5}, rets, quantile=0.05)
        d = to_dict_tail_dependence_matrix(df_l, df_u, scores, quantile=0.05, concordance=conc, tail_stress=sim)
        self.assertEqual(d["quantile"], 0.05)
        self.assertIn("lower_tail_matrix", d)
        self.assertIn("upper_tail_matrix", d)
        self.assertIn("systemic_crash_vulnerability", d)
        self.assertIn("copula_concordance", d)
        self.assertIn("tail_stress_simulation", d)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sub" / "direct_tail.json"
            export_tail_dependence_json(df_l, df_u, scores, out_file, quantile=0.05, concordance=conc, tail_stress=sim)
            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["quantile"], 0.05)
            self.assertIn("copula_concordance", loaded)
            self.assertIn("tail_stress_simulation", loaded)

    def test_export_tail_dependence_csv(self):
        rng = np.random.default_rng(42)
        rets = pd.DataFrame({
            "BTC": rng.normal(0, 0.02, 100),
            "ETH": rng.normal(0, 0.03, 100),
        })
        df_l, df_u, scores = compute_tail_dependence_matrix(rets, quantile=0.05)
        conc = compute_copula_rank_concordance(rets, quantile=0.05)
        sim = simulate_tail_shock_portfolio_loss({"BTC": 0.5, "ETH": 0.5}, rets, quantile=0.05)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sub" / "tail.csv"
            export_tail_dependence_csv(df_l, df_u, scores, out_file, quantile=0.05, concordance=conc, tail_stress=sim)
            self.assertTrue(out_file.exists())
            df = pd.read_csv(out_file)
            self.assertIn("record_type", df.columns)
            self.assertIn("asset_a", df.columns)
            self.assertIn("lower_tail", df.columns)
            self.assertIn("pearson_corr", df.columns)
            self.assertIn("kendall_tau", df.columns)
            self.assertIn("tail_dep_gap", df.columns)
            record_types = set(df["record_type"].unique())
            self.assertIn("pair_tail_dependence", record_types)
            self.assertIn("systemic_vulnerability", record_types)
            self.assertIn("copula_concordance", record_types)
            self.assertIn("tail_stress_simulation", record_types)


if __name__ == "__main__":
    unittest.main()
