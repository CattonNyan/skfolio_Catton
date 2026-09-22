import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.crypto_tail_dependence import (
    TailDependenceResult,
    compute_bivariate_tail_dependence,
    compute_tail_dependence_matrix,
    main as tail_main,
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

    def test_json_export_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sub" / "tail_test.json"
            from unittest.mock import patch
            test_args = ["crypto_tail_dependence.py", "--quantile", "0.05", "--export-json", str(out_file)]
            with patch("sys.argv", test_args):
                tail_main()

            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["quantile"], 0.05)
            self.assertIn("lower_tail_matrix", data)
            self.assertIn("upper_tail_matrix", data)
            self.assertIn("systemic_crash_vulnerability", data)


if __name__ == "__main__":
    unittest.main()
