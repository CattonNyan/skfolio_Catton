"""Unit tests for Upbit KRW market price fetcher."""

from __future__ import annotations

import unittest
import pandas as pd
from scripts.fetch_upbit_crypto import (
    compute_upbit_market_impact,
    compute_upbit_spread,
    export_upbit_orderbook_csv,
    export_upbit_prices_csv,
    fetch_upbit_candles,
    fetch_upbit_historical_prices,
    fetch_upbit_market_list,
    fetch_upbit_orderbook,
    fetch_upbit_spot_price,
    fetch_upbit_ticker,
    main,
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

    def test_compute_upbit_market_impact(self):
        mock_ob = {
            "market": "KRW-BTC",
            "orderbook_units": [
                {
                    "ask_price": 100100000.0,
                    "bid_price": 100000000.0,
                    "ask_size": 1.0,
                    "bid_size": 1.0,
                },
                {
                    "ask_price": 100200000.0,
                    "bid_price": 99900000.0,
                    "ask_size": 2.0,
                    "bid_size": 2.0,
                },
            ],
        }

        # 1. Buy small order (50,050,000 KRW = 0.5 BTC at 100,100,000 KRW)
        res_buy_small = compute_upbit_market_impact(mock_ob, order_size_krw=50050000.0, side="buy")
        self.assertTrue(res_buy_small["fully_filled"])
        self.assertAlmostEqual(res_buy_small["vwap_price"], 100100000.0, places=1)
        self.assertAlmostEqual(res_buy_small["mid_price"], 100050000.0, places=1)
        self.assertAlmostEqual(res_buy_small["price_impact_bps"], 0.0, places=2)
        self.assertGreater(res_buy_small["slippage_bps"], 0.0)

        # 2. Buy large order sweeping into level 2 ask (150,000,000 KRW)
        res_buy_large = compute_upbit_market_impact(mock_ob, order_size_krw=150000000.0, side="buy")
        self.assertTrue(res_buy_large["fully_filled"])
        self.assertGreater(res_buy_large["vwap_price"], 100100000.0)
        self.assertGreater(res_buy_large["price_impact_bps"], 0.0)

        # 3. Sell small order (50,000,000 KRW = 0.5 BTC at 100,000,000 KRW)
        res_sell_small = compute_upbit_market_impact(mock_ob, order_size_krw=50000000.0, side="sell")
        self.assertTrue(res_sell_small["fully_filled"])
        self.assertAlmostEqual(res_sell_small["vwap_price"], 100000000.0, places=1)
        self.assertAlmostEqual(res_sell_small["price_impact_bps"], 0.0, places=2)

        # 4. Sell large order sweeping into level 2 bid
        res_sell_large = compute_upbit_market_impact(mock_ob, order_size_krw=150000000.0, side="sell")
        self.assertTrue(res_sell_large["fully_filled"])
        self.assertLess(res_sell_large["vwap_price"], 100000000.0)
        self.assertGreater(res_sell_large["price_impact_bps"], 0.0)

        # 5. Huge order exceeding book depth
        res_huge = compute_upbit_market_impact(mock_ob, order_size_krw=1000000000.0, side="buy")
        self.assertFalse(res_huge["fully_filled"])
        self.assertGreater(res_huge["unfilled_krw"], 0.0)
        self.assertLess(res_huge["fill_pct"], 100.0)

        # 6. Error handling
        with self.assertRaises(ValueError):
            compute_upbit_market_impact(mock_ob, order_size_krw=-100.0)
        with self.assertRaises(ValueError):
            compute_upbit_market_impact(mock_ob, order_size_krw=1000.0, side="invalid")
        with self.assertRaises(ValueError):
            compute_upbit_market_impact({"orderbook_units": []}, order_size_krw=1000.0)

    def test_export_upbit_orderbook_csv(self):
        import tempfile
        from pathlib import Path

        mock_ob = {
            "market": "KRW-BTC",
            "orderbook_units": [
                {
                    "ask_price": 100100000.0,
                    "bid_price": 100000000.0,
                    "ask_size": 1.0,
                    "bid_size": 1.0,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Export ladder
            ladder_csv = Path(tmpdir) / "ladder.csv"
            export_upbit_orderbook_csv(mock_ob, ladder_csv)
            self.assertTrue(ladder_csv.exists())
            df_ladder = pd.read_csv(ladder_csv)
            self.assertIn("ask_price", df_ladder.columns)
            self.assertIn("bid_price", df_ladder.columns)

            # 2. Export impact record
            impact_res = compute_upbit_market_impact(mock_ob, order_size_krw=50000000.0, side="buy")
            impact_csv = Path(tmpdir) / "impact.csv"
            export_upbit_orderbook_csv(impact_res, impact_csv)
            self.assertTrue(impact_csv.exists())
            df_impact = pd.read_csv(impact_csv)
            self.assertEqual(df_impact["market"].iloc[0], "KRW-BTC")
            self.assertEqual(df_impact["side"].iloc[0], "buy")

    def test_upbit_orderbook_cli(self):
        import io
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        mock_ob = {
            "market": "KRW-BTC",
            "orderbook_units": [
                {
                    "ask_price": 100100000.0,
                    "bid_price": 100000000.0,
                    "ask_size": 2.0,
                    "bid_size": 2.0,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = str(Path(tmpdir) / "upbit_ob.csv")
            test_args = [
                "fetch_upbit_crypto.py",
                "--orderbook", "BTC",
                "--order-size", "50000000",
                "--side", "buy",
                "--depth-levels", "5",
                "--export-orderbook-csv", out_csv,
            ]
            with patch("scripts.fetch_upbit_crypto.fetch_upbit_orderbook", return_value=mock_ob):
                with patch("sys.argv", test_args):
                    captured = io.StringIO()
                    with patch("sys.stdout", captured):
                        main()
                    output = captured.getvalue()
                    self.assertIn("Upbit Order Book & Liquidity Metrics", output)
                    self.assertIn("Order Book Depth Ladder", output)
                    self.assertIn("Institutional Market Impact Simulation", output)
                    self.assertIn("Saved Upbit order book metrics", output)
                    self.assertTrue(Path(out_csv).exists())


if __name__ == "__main__":
    unittest.main()
