import unittest
import numpy as np
import pandas as pd

from scripts.crypto_tail_dependence import (
    compute_bivariate_tail_dependence,
    compute_tail_dependence_matrix,
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


if __name__ == "__main__":
    unittest.main()
