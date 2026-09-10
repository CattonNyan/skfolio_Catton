"""Unit tests for Upbit KRW market price fetcher."""

from __future__ import annotations

import unittest
import pandas as pd
from scripts.fetch_upbit_crypto import (
    fetch_upbit_candles,
    fetch_upbit_historical_prices,
    fetch_upbit_ticker,
    validate_upbit_market_code,
)


class UpbitFetcherTests(unittest.TestCase):
    def test_validate_upbit_market_code(self):
        self.assertEqual(validate_upbit_market_code("krw-btc"), "KRW-BTC")
        self.assertEqual(validate_upbit_market_code("KRW-ETH"), "KRW-ETH")
        self.assertEqual(validate_upbit_market_code("  KRW-SOL  "), "KRW-SOL")

        # Invalid formats
        with self.assertRaises(ValueError):
            validate_upbit_market_code(None)  # type: ignore
        with self.assertRaises(ValueError):
            validate_upbit_market_code("")
        with self.assertRaises(ValueError):
            validate_upbit_market_code("BTCUSDT")
        with self.assertRaises(ValueError):
            validate_upbit_market_code("KRW-BTC-EXTRA")
        with self.assertRaises(ValueError):
            validate_upbit_market_code(True)  # type: ignore

    def test_fetch_upbit_ticker_basic(self):
        markets = ["KRW-BTC", "KRW-ETH"]
        ticker = fetch_upbit_ticker(markets, timeout=2.0)
        self.assertIsInstance(ticker, dict)
        for m in markets:
            self.assertIn(m, ticker)
            self.assertGreater(ticker[m], 0)

    def test_fetch_upbit_candles_basic(self):
        df = fetch_upbit_candles("KRW-BTC", count=10, timeframe="days", timeout=2.0)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)
        required_cols = {"date", "open", "high", "low", "close", "volume"}
        self.assertTrue(required_cols.issubset(df.columns))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["date"]))
        self.assertTrue((df["close"] > 0).all())

    def test_fetch_upbit_historical_prices_skfolio_ready(self):
        markets = ["KRW-BTC", "KRW-ETH"]
        prices = fetch_upbit_historical_prices(markets, count=15, timeframe="days", timeout=2.0)
        self.assertIsInstance(prices, pd.DataFrame)
        self.assertEqual(len(prices), 15)
        self.assertEqual(list(prices.columns), markets)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(prices.index))
        self.assertFalse(prices.isna().any().any())

    def test_invalid_parameters_rejected(self):
        # Invalid count
        with self.assertRaises(ValueError):
            fetch_upbit_candles("KRW-BTC", count=0)
        with self.assertRaises(ValueError):
            fetch_upbit_candles("KRW-BTC", count=250)
        with self.assertRaises(ValueError):
            fetch_upbit_candles("KRW-BTC", count=True)  # type: ignore

        # Invalid timeframe
        with self.assertRaises(ValueError):
            fetch_upbit_candles("KRW-BTC", count=10, timeframe="seconds/10")

        # Invalid timeout
        with self.assertRaises(ValueError):
            fetch_upbit_candles("KRW-BTC", count=10, timeout=-1.0)
        with self.assertRaises(ValueError):
            fetch_upbit_candles("KRW-BTC", count=10, timeout=False)  # type: ignore

        # Invalid markets
        with self.assertRaises(ValueError):
            fetch_upbit_ticker([])
        with self.assertRaises(ValueError):
            fetch_upbit_historical_prices([])


if __name__ == "__main__":
    unittest.main()
