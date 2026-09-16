import unittest
import numpy as np
import pandas as pd

from scripts.crypto_synthetic_data import generate_correlated_crypto_paths


class SyntheticDataTests(unittest.TestCase):
    def test_shape_and_columns(self):
        assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        df = generate_correlated_crypto_paths(n_bars=100, assets=assets, seed=42)
        self.assertEqual(df.shape, (100, 3))
        self.assertEqual(list(df.columns), assets)
        self.assertTrue((df.values > 0).all())
        self.assertFalse(df.isna().any().any())

    def test_correlation_structure(self):
        # Generate with high target correlation (0.80)
        corr_target = np.array([[1.0, 0.8], [0.8, 1.0]])
        df = generate_correlated_crypto_paths(
            n_bars=1000,
            assets=["A", "B"],
            correlation_matrix=corr_target,
            jump_intensity=0.0,
            seed=42,
        )
        emp_corr = df.pct_change().dropna().corr().iloc[0, 1]
        self.assertAlmostEqual(emp_corr, 0.8, delta=0.10)

    def test_jump_diffusion_flash_crash(self):
        # 100% jump probability with negative shock
        df = generate_correlated_crypto_paths(
            n_bars=10,
            assets=["BTC/USDT"],
            jump_intensity=1.0,
            jump_mean=-0.10,
            seed=42,
        )
        rets = df.pct_change().dropna().values
        self.assertTrue(np.mean(rets) < 0)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            generate_correlated_crypto_paths(n_bars=1)
        with self.assertRaises(ValueError):
            generate_correlated_crypto_paths(assets=[])
        with self.assertRaises(ValueError):
            generate_correlated_crypto_paths(jump_intensity=-0.1)
        with self.assertRaises(ValueError):
            generate_correlated_crypto_paths(jump_intensity=1.5)


if __name__ == "__main__":
    unittest.main()
