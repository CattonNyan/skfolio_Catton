import json
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from scripts.fetch_coinbase_crypto import (
    normalize_coinbase_symbol,
    fetch_coinbase_spot_price,
    fetch_coinbase_candles,
    fetch_coinbase_multi_assets,
    GRANULARITY_MAP,
)


class CoinbaseFetcherTests(unittest.TestCase):
    def test_normalize_coinbase_symbol(self):
        self.assertEqual(normalize_coinbase_symbol("BTC"), "BTC-USD")
        self.assertEqual(normalize_coinbase_symbol("eth/usd"), "ETH-USD")
        self.assertEqual(normalize_coinbase_symbol("SOL-USD"), "SOL-USD")
        self.assertEqual(normalize_coinbase_symbol("BTC_EUR"), "BTC-EUR")

        with self.assertRaises(ValueError):
            normalize_coinbase_symbol("")
        with self.assertRaises(ValueError):
            normalize_coinbase_symbol(123)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            fetch_coinbase_candles("BTC", timeframe="48h")
        with self.assertRaises(ValueError):
            fetch_coinbase_candles("BTC", limit=0)
        with self.assertRaises(ValueError):
            fetch_coinbase_candles("BTC", limit=500)

    @patch("urllib.request.urlopen")
    def test_fetch_spot_price_mock(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"data": {"amount": "65432.10", "currency": "USD"}}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        price = fetch_coinbase_spot_price("BTC-USD")
        self.assertAlmostEqual(price, 65432.10, places=2)

    @patch("urllib.request.urlopen")
    def test_fetch_candles_mock(self, mock_urlopen):
        # Coinbase candle: [ time, low, high, open, close, volume ]
        mock_bars = [
            [1700000000, 60000.0, 62000.0, 60500.0, 61500.0, 150.5],
            [1700086400, 61000.0, 63000.0, 61500.0, 62800.0, 180.2],
        ]
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_bars).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        df = fetch_coinbase_candles("BTC-USD", timeframe="1d", limit=2)
        self.assertEqual(len(df), 2)
        self.assertIn("close", df.columns)
        self.assertEqual(df["close"].iloc[-1], 62800.0)

    @patch("scripts.fetch_coinbase_crypto.fetch_coinbase_candles")
    def test_fetch_multi_assets(self, mock_candles):
        t = pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True)
        mock_candles.side_effect = [
            pd.DataFrame({"date": t, "close": [60000.0, 61000.0]}),
            pd.DataFrame({"date": t, "close": [3000.0, 3100.0]}),
        ]
        res = fetch_coinbase_multi_assets(["BTC", "ETH"])
        self.assertEqual(len(res.columns), 2)
        self.assertIn("BTC-USD", res.columns)
        self.assertIn("ETH-USD", res.columns)


if __name__ == "__main__":
    unittest.main()
