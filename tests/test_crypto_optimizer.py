"""Tests for crypto portfolio optimizer script."""

import unittest
from pathlib import Path
import pandas as pd

from scripts.crypto_portfolio_optimizer import (
    MarketDataUnavailableError,
    generate_synthetic_crypto_data,
    find_freqtrade_data_dirs,
    load_market_data,
    load_from_feather_dir,
    positive_float,
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

    def test_real_data_load_fails_closed(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(MarketDataUnavailableError):
                load_market_data(data_dir=tmpdir, timeframe="15m")

    def test_loader_never_falls_back_to_another_timeframe(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            pd.DataFrame(
                {
                    "date": pd.date_range("2026-01-01", periods=3, freq="1h"),
                    "close": [100.0, 101.0, 102.0],
                }
            ).to_feather(data_dir / "BTC_USDT-1h.feather")

            self.assertTrue(load_from_feather_dir(data_dir, timeframe="15m").empty)
            loaded = load_from_feather_dir(data_dir, timeframe="1h")
            self.assertEqual(list(loaded.columns), ["BTC/USDT"])
            self.assertEqual(len(loaded), 3)

    def test_synthetic_data_requires_explicit_opt_in(self):
        prices, source = load_market_data(use_synthetic=True, synthetic_periods=25)
        self.assertEqual(source, "synthetic")
        self.assertEqual(len(prices), 25)

    def test_positive_float_rejects_invalid_wallet_values(self):
        import argparse

        self.assertEqual(positive_float("1000"), 1000.0)
        for value in ("0", "-1", "nan", "inf"):
            with self.assertRaises(argparse.ArgumentTypeError):
                positive_float(value)

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

    def test_export_normalizes_weights_and_updates_freqtrade_config(self):
        import json
        import tempfile
        from scripts.crypto_portfolio_optimizer import export_freqtrade_allocation

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "config.json"
            target_path.write_text(
                json.dumps({"exchange": {"name": "binance", "pair_whitelist": []}}),
                encoding="utf-8",
            )
            success = export_freqtrade_allocation(
                results={"model": {"BTC/USDT": 3.0, "ETH/USDT": 1.0}},
                target_path=target_path,
                model_name="model",
                data_source="real-test-data",
            )
            self.assertTrue(success)
            data = json.loads(target_path.read_text(encoding="utf-8"))
            self.assertEqual(data["exchange"]["pair_whitelist"], ["BTC/USDT", "ETH/USDT"])
            self.assertEqual(data["pair_weights"], {"BTC/USDT": 0.75, "ETH/USDT": 0.25})
            self.assertEqual(data["skfolio_allocation"]["data_source"], "real-test-data")

    def test_export_preserves_malformed_existing_json(self):
        import tempfile
        from scripts.crypto_portfolio_optimizer import export_freqtrade_allocation

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "config.json"
            original = "{ malformed user config"
            target_path.write_text(original, encoding="utf-8")
            success = export_freqtrade_allocation(
                results={"model": {"BTC/USDT": 1.0}},
                target_path=target_path,
                model_name="model",
            )
            self.assertFalse(success)
            self.assertEqual(target_path.read_text(encoding="utf-8"), original)

    def test_synthetic_allocation_export_is_rejected(self):
        import tempfile
        from scripts.crypto_portfolio_optimizer import export_freqtrade_allocation

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "allocation.json"
            success = export_freqtrade_allocation(
                results={"model": {"BTC/USDT": 1.0}},
                target_path=target_path,
                model_name="model",
                data_source="synthetic",
            )
            self.assertFalse(success)
            self.assertFalse(target_path.exists())


    def test_export_csv_allocation(self):
        import tempfile
        from scripts.crypto_portfolio_optimizer import export_csv_allocation

        sample_results = {
            "Risk Parity (ERC)": {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "test_allocation.csv"
            success = export_csv_allocation(
                results=sample_results,
                target_path=target_path,
                model_name="Risk Parity (ERC)",
                total_wallet=1000.0,
            )
            self.assertTrue(success)
            self.assertTrue(target_path.is_file())
            content = target_path.read_text(encoding="utf-8-sig")
            self.assertIn("Asset,Weight_Percent,Weight_Fraction,Allocated_Amount", content)
            self.assertIn("BTC/USDT,60.00%,0.6,600.0", content)


if __name__ == "__main__":
    unittest.main()
