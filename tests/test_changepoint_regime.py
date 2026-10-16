import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from scripts.crypto_changepoint_regime import (
    ChangepointEvent,
    RegimeState,
    classify_market_regime,
    detect_multivariate_changepoints,
    calculate_regime_covariance_matrix,
    optimize_min_variance_weights,
    analyze_changepoint_regimes,
    simulate_changepoint_rebalancing,
    generate_synthetic_regime_prices,
    export_changepoint_analysis_json,
    export_changepoint_analysis_csv,
    main as changepoint_main,
)


class CryptoChangepointRegimeTests(unittest.TestCase):
    def setUp(self):
        self.prices_df = generate_synthetic_regime_prices(n_bars=250, seed=42)
        self.returns_df = self.prices_df.pct_change().dropna()

    def test_classify_market_regime(self):
        calm = classify_market_regime(ann_return=0.15, ann_vol=0.25, mean_corr=0.40)
        self.assertEqual(calm, "Low-Vol Calm Bull")

        shock = classify_market_regime(ann_return=-0.35, ann_vol=0.65, mean_corr=0.85)
        self.assertEqual(shock, "Systemic Liquidity Shock")

        momentum = classify_market_regime(ann_return=0.40, ann_vol=0.55, mean_corr=0.60)
        self.assertEqual(momentum, "High-Vol Momentum Bull")

    def test_detect_multivariate_changepoints_synthetic(self):
        cps = detect_multivariate_changepoints(
            self.returns_df, window=30, threshold=1.8, min_distance=20
        )
        self.assertIsInstance(cps, list)
        self.assertGreater(len(cps), 0)
        for cp in cps:
            self.assertIsInstance(cp, (int, np.integer))
            self.assertGreater(cp, 30)
            self.assertLess(cp, len(self.returns_df))

    def test_detect_multivariate_changepoints_short_data(self):
        short_df = self.returns_df.iloc[:20]
        cps = detect_multivariate_changepoints(short_df, window=30)
        self.assertEqual(cps, [])

    def test_calculate_regime_covariance_matrix(self):
        cps = [95]
        cov_mat, cov_dict = calculate_regime_covariance_matrix(
            self.returns_df, changepoints=cps, decay_halflife=30, shrinkage=0.15
        )
        self.assertEqual(cov_mat.shape, (4, 4))
        # Positive definiteness check: eigenvalues > 0
        eigenvals = np.linalg.eigvalsh(cov_mat)
        self.assertTrue(np.all(eigenvals > 0))

        # Dictionary format
        self.assertIn("BTC/USDT", cov_dict)
        self.assertIn("ETH/USDT", cov_dict["BTC/USDT"])

    def test_optimize_min_variance_weights(self):
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
        cov = np.cov(self.returns_df.to_numpy(), rowvar=False)
        w = optimize_min_variance_weights(cov, symbols)

        self.assertEqual(len(w), 4)
        for s in symbols:
            self.assertGreaterEqual(w[s], 0.0)
            self.assertLessEqual(w[s], 1.0)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=2)

    def test_analyze_changepoint_regimes(self):
        res = analyze_changepoint_regimes(
            self.prices_df, window=30, threshold=1.8, min_distance=20
        )
        self.assertIn("current_regime", res)
        self.assertIn("regimes", res)
        self.assertIn("changepoint_events", res)
        self.assertIn("adaptive_weights", res)
        self.assertIn("static_weights", res)
        self.assertIn("current_covariance_matrix", res)

        self.assertGreaterEqual(len(res["regimes"]), 1)
        r0 = res["regimes"][0]
        self.assertIn("regime_id", r0)
        self.assertIn("name", r0)
        self.assertIn("optimal_weights", r0)

    def test_analyze_changepoint_regimes_too_short(self):
        short_prices = self.prices_df.iloc[:30]
        with self.assertRaises(ValueError):
            analyze_changepoint_regimes(short_prices)

    def test_simulate_changepoint_rebalancing(self):
        sim = simulate_changepoint_rebalancing(
            self.prices_df, fee_rate=0.001, initial_capital=10000.0, window=30
        )
        self.assertIn("policies", sim)
        policies = sim["policies"]

        for pol_name in ["BuyAndHold", "StaticRolling30", "AdaptiveChangepoint"]:
            self.assertIn(pol_name, policies)
            p = policies[pol_name]
            self.assertIn("final_nav", p)
            self.assertIn("cagr_pct", p)
            self.assertIn("sharpe_ratio", p)
            self.assertIn("max_drawdown_pct", p)
            self.assertIn("rebalance_count", p)

        self.assertEqual(policies["BuyAndHold"]["rebalance_count"], 0)

    def test_export_json_and_csv(self):
        sim = simulate_changepoint_rebalancing(
            self.prices_df, fee_rate=0.001, initial_capital=10000.0, window=30
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            json_p = Path(tmpdir) / "cp_res.json"
            csv_p = Path(tmpdir) / "cp_res.csv"

            export_changepoint_analysis_json(sim, json_p)
            self.assertTrue(json_p.exists())
            with open(json_p, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIn("policies", loaded)

            export_changepoint_analysis_csv(sim, csv_p)
            self.assertTrue(csv_p.exists())
            df = pd.read_csv(csv_p)
            self.assertGreater(len(df), 0)
            self.assertIn("regime_id", df.columns)
            self.assertIn("name", df.columns)
            self.assertIn("weight_BTC/USDT", df.columns)
            self.assertNotIn("optimal_weights", df.columns)

    def test_custom_symbols_and_threshold(self):
        five_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "DOT/USDT"]
        df5 = generate_synthetic_regime_prices(n_bars=180, seed=7, symbols=five_symbols)
        self.assertEqual(list(df5.columns), five_symbols)
        sim = simulate_changepoint_rebalancing(
            df5, fee_rate=0.001, initial_capital=5000.0, window=25, threshold=1.7
        )
        self.assertEqual(sim["analysis"]["metadata"]["symbols"], five_symbols)
        self.assertIn("AdaptiveChangepoint", sim["policies"])

    def test_cli_execution_example(self):
        test_args = ["crypto_changepoint_regime.py", "--example", "--threshold", "1.9"]
        with patch.object(sys, "argv", test_args):
            changepoint_main()


if __name__ == "__main__":
    unittest.main()
