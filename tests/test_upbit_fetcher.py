"""Unit tests for Upbit KRW market price fetcher."""

from __future__ import annotations

import unittest
import pandas as pd
from scripts.fetch_upbit_crypto import (
    compute_upbit_spread,
    export_upbit_prices_csv,
    fetch_upbit_candles,
    fetch_upbit_historical_prices,
    fetch_upbit_market_list,
    fetch_upbit_orderbook,
    fetch_upbit_spot_price,
    fetch_upbit_ticker,
    normalize_upbit_symbol,
    validate_upbit_market_code,
)


class UpbitFetcherTests(unittest.TestCase):
    def test_normalize_upbit_symbol(self):
        self.assertEqual(normalize_upbit_symbol("btc"), "KRW-BTC")
        self.assertEqual(normalize_upbit_symbol("KRW-ETH"), "KRW-ETH")
        self.assertEqual(normalize_upbit_symbol("sol/krw"), "KRW-SOL")
        self.assertEqual(normalize_upbit_symbol("xrp_krw"), "KRW-XRP")
        with self.assertRaises(ValueError):
            normalize_upbit_symbol("")
        with self.assertRaises(ValueError):
            normalize_upbit_symbol(None)  # type: ignore

    def test_fetch_upbit_market_list(self):
        markets = fetch_upbit_market_list(quote_currency="KRW", timeout=2.0)
        self.assertIsInstance(markets, list)
        self.assertGreater(len(markets), 0)
        self.assertTrue(all(m.startswith("KRW-") for m in markets))
        with self.assertRaises(ValueError):
            fetch_upbit_market_list("")
        with self.assertRaises(ValueError):
            fetch_upbit_market_list(timeout=-1.0)

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

    def test_fetch_upbit_spot_price(self):
        price = fetch_upbit_spot_price("BTC", timeout=2.0)
        self.assertIsInstance(price, float)
        self.assertGreater(price, 0)
        price_eth = fetch_upbit_spot_price("KRW-ETH", timeout=2.0)
        self.assertIsInstance(price_eth, float)
        self.assertGreater(price_eth, 0)

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

        # Invalid orderbook arguments
        with self.assertRaises(ValueError):
            fetch_upbit_orderbook(None)  # type: ignore
        with self.assertRaises(ValueError):
            fetch_upbit_orderbook("KRW-BTC", timeout=-1.0)
        with self.assertRaises(ValueError):
            compute_upbit_spread("invalid")  # type: ignore
        with self.assertRaises(ValueError):
            compute_upbit_spread({"orderbook_units": []})

    def test_fetch_upbit_orderbook_and_spread(self):
        ob = fetch_upbit_orderbook("KRW-BTC", timeout=2.0)
        self.assertIsInstance(ob, dict)
        self.assertIn("market", ob)
        self.assertIn("orderbook_units", ob)
        self.assertGreater(len(ob["orderbook_units"]), 0)

        spread = compute_upbit_spread(ob)
        self.assertIsInstance(spread, dict)
        self.assertGreater(spread["best_bid"], 0)
        self.assertGreater(spread["best_ask"], 0)
        self.assertGreaterEqual(spread["best_ask"], spread["best_bid"])
        self.assertGreater(spread["mid_price"], 0)
        self.assertGreaterEqual(spread["spread_krw"], 0)
        self.assertGreaterEqual(spread["spread_bps"], 0)
        self.assertGreater(spread["bid_depth_krw"], 0)
        self.assertGreater(spread["ask_depth_krw"], 0)

    def test_compute_upbit_spread_deterministic(self):
        mock_ob = {
            "market": "KRW-BTC",
            "total_ask_size": 1.5,
            "total_bid_size": 2.5,
            "orderbook_units": [
                {
                    "ask_price": 100500.0,
                    "bid_price": 100000.0,
                    "ask_size": 1.5,
                    "bid_size": 2.5,
                }
            ],
        }
        spread = compute_upbit_spread(mock_ob)
        self.assertEqual(spread["best_bid"], 100000.0)
        self.assertEqual(spread["best_ask"], 100500.0)
        self.assertEqual(spread["mid_price"], 100250.0)
        self.assertEqual(spread["spread_krw"], 500.0)
        self.assertAlmostEqual(spread["spread_bps"], (500.0 / 100250.0) * 10000.0, places=2)
        self.assertEqual(spread["bid_depth_krw"], 250000.0)
        self.assertEqual(spread["ask_depth_krw"], 150750.0)
        self.assertEqual(spread["total_bid_size"], 2.5)
        self.assertEqual(spread["total_ask_size"], 1.5)

    def test_export_upbit_prices_csv(self):
        import tempfile
        from pathlib import Path

        df = pd.DataFrame(
            {"KRW-BTC": [100000000.0, 101000000.0], "KRW-ETH": [4000000.0, 4050000.0]},
            index=pd.date_range("2026-01-01", periods=2, freq="1D"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "upbit_prices.csv"
            saved_path = export_upbit_prices_csv(df, out_csv)
            self.assertEqual(saved_path, out_csv)
            self.assertTrue(out_csv.exists())
            loaded = pd.read_csv(out_csv, index_col=0)
            self.assertEqual(len(loaded), 2)
            self.assertIn("KRW-BTC", loaded.columns)
            self.assertIn("KRW-ETH", loaded.columns)


if __name__ == "__main__":
    unittest.main()
