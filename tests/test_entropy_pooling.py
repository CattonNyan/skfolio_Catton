import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from scripts.crypto_entropy_pooling import (
    EntropyView,
    EntropyPoolingResult,
    build_prior_probabilities,
    calculate_effective_number_of_scenarios,
    calculate_kl_divergence,
    build_view_linear_constraints,
    compute_entropy_pooling,
    compute_scenario_moments,
    optimize_portfolio_weights,
    run_entropy_pooling_analysis,
    simulate_entropy_pooling_walk_forward,
    generate_synthetic_crypto_scenarios,
    export_entropy_pooling_json,
    export_entropy_pooling_csv,
    main as entropy_main,
)


class CryptoEntropyPoolingTests(unittest.TestCase):
    def setUp(self):
        self.prices_df = generate_synthetic_crypto_scenarios(n_bars=300, seed=42)
        self.returns_df = self.prices_df.pct_change().dropna()
        self.symbols = list(self.returns_df.columns)

    def test_build_prior_probabilities_uniform(self):
        p = build_prior_probabilities(100, decay_lambda=1.0)
        self.assertEqual(len(p), 100)
        self.assertAlmostEqual(float(np.sum(p)), 1.0, places=9)
        self.assertTrue(np.allclose(p, 0.01))

    def test_build_prior_probabilities_decay(self):
        p = build_prior_probabilities(50, decay_lambda=0.98)
        self.assertEqual(len(p), 50)
        self.assertAlmostEqual(float(np.sum(p)), 1.0, places=9)
        # Recent scenario should have strictly higher probability than older
        self.assertGreater(p[-1], p[0])

    def test_build_prior_probabilities_invalid(self):
        with self.assertRaises(ValueError):
            build_prior_probabilities(0)
        with self.assertRaises(ValueError):
            build_prior_probabilities(10, decay_lambda=-0.5)
        with self.assertRaises(ValueError):
            build_prior_probabilities(10, decay_lambda=1.5)

    def test_effective_number_of_scenarios(self):
        # Uniform: ENS = S
        p_uniform = np.ones(50) / 50.0
        ens_uni = calculate_effective_number_of_scenarios(p_uniform)
        self.assertAlmostEqual(ens_uni, 50.0, places=3)

        # Concentrated: ENS < S
        p_conc = np.zeros(50)
        p_conc[0] = 0.90
        p_conc[1:] = 0.10 / 49.0
        ens_conc = calculate_effective_number_of_scenarios(p_conc)
        self.assertLess(ens_conc, 10.0)

    def test_calculate_kl_divergence(self):
        p = np.ones(20) / 20.0
        # Identical
        kl_zero = calculate_kl_divergence(p, p)
        self.assertAlmostEqual(kl_zero, 0.0, places=5)

        # Perturbed
        p_mod = p.copy()
        p_mod[0] *= 2.0
        p_mod /= np.sum(p_mod)
        kl_pos = calculate_kl_divergence(p_mod, p)
        self.assertGreater(kl_pos, 0.0)

    def test_build_view_linear_constraints(self):
        p_prior = build_prior_probabilities(len(self.returns_df))
        views = [
            EntropyView("absolute_return", ["BTC"], ">=", 0.35),
            EntropyView("relative_return", ["SOL", "ETH"], ">=", 0.10),
            EntropyView("volatility", ["AVAX"], "<=", 0.60),
            EntropyView("tail_risk", ["BTC"], "<=", 0.05),
        ]
        A_eq, b_eq, A_ineq, b_ineq = build_view_linear_constraints(self.returns_df, views, p_prior)
        self.assertIsNone(A_eq)
        self.assertIsNone(b_eq)
        self.assertIsNotNone(A_ineq)
        self.assertIsNotNone(b_ineq)
        self.assertEqual(len(b_ineq), 4)

    def test_build_view_linear_constraints_errors(self):
        p_prior = build_prior_probabilities(len(self.returns_df))
        # Unknown asset
        with self.assertRaises(ValueError):
            build_view_linear_constraints(
                self.returns_df,
                [EntropyView("absolute_return", ["UNKNOWN_COIN"], ">=", 0.10)],
                p_prior,
            )
        # Invalid relative return (needs 2 assets)
        with self.assertRaises(ValueError):
            build_view_linear_constraints(
                self.returns_df,
                [EntropyView("relative_return", ["BTC"], ">=", 0.10)],
                p_prior,
            )

    def test_compute_entropy_pooling_confidence_zero(self):
        p_prior = build_prior_probabilities(len(self.returns_df))
        views = [EntropyView("absolute_return", ["BTC"], ">=", 0.35)]
        A_eq, b_eq, A_ineq, b_ineq = build_view_linear_constraints(self.returns_df, views, p_prior)
        p_post, kl = compute_entropy_pooling(
            p_prior, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, confidence=0.0
        )
        self.assertTrue(np.allclose(p_post, p_prior))
        self.assertEqual(kl, 0.0)

    def test_compute_scenario_moments_positive_definite(self):
        p = build_prior_probabilities(len(self.returns_df))
        mu, cov = compute_scenario_moments(self.returns_df, p, shrinkage=0.05)
        self.assertEqual(len(mu), len(self.symbols))
        self.assertEqual(cov.shape, (len(self.symbols), len(self.symbols)))
        # Eigenvalues must all be strictly positive
        eigs = np.linalg.eigvals(cov)
        self.assertTrue(np.all(eigs > 0))

    def test_optimize_portfolio_weights(self):
        p = build_prior_probabilities(len(self.returns_df))
        mu, cov = compute_scenario_moments(self.returns_df, p)

        w_sharpe = optimize_portfolio_weights(mu, cov, objective="max_sharpe")
        self.assertAlmostEqual(float(np.sum(w_sharpe)), 1.0, places=4)
        self.assertTrue(np.all(w_sharpe >= -1e-6))

        w_minvar = optimize_portfolio_weights(mu, cov, objective="min_var")
        self.assertAlmostEqual(float(np.sum(w_minvar)), 1.0, places=4)
        self.assertTrue(np.all(w_minvar >= -1e-6))

        w_ew = optimize_portfolio_weights(mu, cov, objective="equal_weight")
        self.assertAlmostEqual(float(np.sum(w_ew)), 1.0, places=4)
        self.assertEqual(len(set(np.round(w_ew, 4))), 1)

    def test_run_entropy_pooling_analysis(self):
        views = [
            EntropyView("relative_return", ["SOL", "ETH"], ">=", 0.15),
            EntropyView("absolute_return", ["BTC"], ">=", 0.25),
        ]
        res = run_entropy_pooling_analysis(self.returns_df, views=views, confidence=0.85)
        self.assertIsInstance(res, EntropyPoolingResult)
        self.assertGreater(res.kl_divergence, 0.0)
        self.assertGreater(res.effective_scenarios_prior, res.effective_scenarios_posterior)
        self.assertIn("BTC", res.posterior_expected_returns_ann)
        self.assertIn("SOL", res.weights_posterior_max_sharpe)

    def test_simulate_entropy_pooling_walk_forward(self):
        views = [EntropyView("relative_return", ["SOL", "ETH"], ">=", 0.10)]
        bt_res = simulate_entropy_pooling_walk_forward(
            self.returns_df,
            views=views,
            lookback=90,
            rebalance_freq=15,
            confidence=0.8,
            initial_capital=10000.0,
        )
        self.assertIn("Entropy_Pooled_Sharpe", bt_res)
        self.assertIn("Prior_Max_Sharpe", bt_res)
        self.assertIn("Equal_Weight", bt_res)
        for pol, m in bt_res.items():
            self.assertIn("final_equity", m)
            self.assertIn("sharpe_ratio", m)
            self.assertIn("max_drawdown_pct", m)
            self.assertIn("total_turnover", m)

    def test_export_json_and_csv(self):
        views = [EntropyView("absolute_return", ["BTC"], ">=", 0.20)]
        res = run_entropy_pooling_analysis(self.returns_df, views=views)
        json_str = export_entropy_pooling_json(res)
        parsed = json.loads(json_str)
        self.assertIn("entropy_pooling", parsed)
        self.assertIn("kl_divergence", parsed["entropy_pooling"])

        csv_str = export_entropy_pooling_csv(res)
        self.assertIn("symbol,prior_return_ann_pct", csv_str)
        self.assertIn("BTC", csv_str)

    def test_cli_execution(self):
        test_args = ["crypto_entropy_pooling.py", "--example", "--confidence", "0.75"]
        with patch.object(sys, "argv", test_args):
            entropy_main()


if __name__ == "__main__":
    unittest.main()
