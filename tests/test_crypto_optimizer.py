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

    def test_export_freqtrade_allocation(self):
        import json
        import tempfile
        from scripts.crypto_portfolio_optimizer import export_freqtrade_allocation

        sample_results = {
            "Risk Parity (ERC)": {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "test_allocation.json"
            success = export_freqtrade_allocation(
                results=sample_results,
                target_path=target_path,
                model_name="Risk Parity (ERC)",
                total_wallet=1000.0,
            )
            self.assertTrue(success)
            self.assertTrue(target_path.is_file())
            data = json.loads(target_path.read_text(encoding="utf-8"))
            self.assertEqual(data["pair_whitelist"], ["BTC/USDT", "ETH/USDT"])
            self.assertEqual(data["pair_weights"]["BTC/USDT"], 0.6)
            self.assertEqual(data["stake_amounts"]["BTC/USDT"], 600.0)


if __name__ == "__main__":
    unittest.main()
