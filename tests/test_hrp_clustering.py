"""Tests for HRP clustering module."""

import unittest
import pandas as pd
import numpy as np

from scripts.crypto_portfolio_optimizer import generate_synthetic_crypto_data
from scripts.crypto_hrp_clustering import compute_correlation_matrix


class HrpClusteringTests(unittest.TestCase):
    def test_correlation_matrix_computation(self):
        prices = generate_synthetic_crypto_data(periods=50)
        returns = prices.pct_change().dropna()
        corr = compute_correlation_matrix(returns)
        self.assertIsInstance(corr, pd.DataFrame)
        self.assertEqual(corr.shape[0], corr.shape[1])
        # Diagonal elements must be approximately 1.0
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
