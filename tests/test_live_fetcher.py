"""Tests for live crypto data fetcher module."""

import unittest
import tempfile
from pathlib import Path
import pandas as pd

from scripts.fetch_live_crypto import data_dict_to_prices, save_market_data


class LiveFetcherTests(unittest.TestCase):
    def test_save_market_data(self):
        sample_df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=10, freq="1h"),
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [95.0] * 10,
            "close": [102.0] * 10,
            "volume": [1000.0] * 10,
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            saved = save_market_data(
                data_dict={"BTC/USDT": sample_df},
                output_dir=out_dir,
                timeframe="1h",
            )
            self.assertGreaterEqual(len(saved), 1)
            csv_path = out_dir / "BTC_USDT-1h.csv"
            self.assertTrue(csv_path.is_file())

    def test_data_dict_to_prices(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="1h")
        df1 = pd.DataFrame({"date": dates, "close": [100.0, 101.0, 102.0, 103.0, 104.0]})
        df2 = pd.DataFrame({"date": dates, "close": [50.0, 51.0, 52.0, 53.0, 54.0]})
        prices = data_dict_to_prices({"BTC/USDT": df1, "ETH/USDT": df2})
        self.assertFalse(prices.empty)
        self.assertEqual(list(prices.columns), ["BTC/USDT", "ETH/USDT"])
        self.assertEqual(len(prices), 5)

    def test_invalid_parameters_rejected(self):
        from scripts.fetch_live_crypto import fetch_ohlcv_ccxt
        for bad_tf in ("invalid", "15x", "m15", "", None):
            with self.subTest(bad_tf=bad_tf), self.assertRaises(ValueError):
                save_market_data({}, Path("."), timeframe=bad_tf)
            with self.subTest(bad_tf=bad_tf), self.assertRaises(ValueError):
                fetch_ohlcv_ccxt(timeframe=bad_tf)

        for bad_limit in (0, -1, True, False, 1.5):
            with self.subTest(bad_limit=bad_limit), self.assertRaises(ValueError):
                fetch_ohlcv_ccxt(limit=bad_limit)

        for bad_dict in ("not_a_dict", [1, 2], None):
            with self.subTest(bad_dict=bad_dict), self.assertRaises(ValueError):
                data_dict_to_prices(bad_dict)
            with self.subTest(bad_dict=bad_dict), self.assertRaises(ValueError):
                save_market_data(bad_dict, Path("."))


if __name__ == "__main__":
    unittest.main()
