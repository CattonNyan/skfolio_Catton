"""Tests for crypto portfolio optimizer script."""

import unittest
from pathlib import Path
import pandas as pd

from scripts.crypto_portfolio_optimizer import (
    generate_synthetic_crypto_data,
    find_freqtrade_data_dirs,
)


class CryptoOptimizerTests(unittest.TestCase):
    def test_synthetic_data_generation(self):
        df = generate_synthetic_crypto_data(periods=100)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        self.assertIn("BTC/USDT", df.columns)
        self.assertIn("ETH/USDT", df.columns)
        self.assertTrue((df["BTC/USDT"] > 0).all())

    def test_find_freqtrade_data_dirs(self):
        dirs = find_freqtrade_data_dirs()
        self.assertIsInstance(dirs, list)
        for d in dirs:
            self.assertTrue(d.is_dir())


if __name__ == "__main__":
    unittest.main()
