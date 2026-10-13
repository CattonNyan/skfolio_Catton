"""Tests for live crypto data fetcher module."""

import unittest
import tempfile
from pathlib import Path
import pandas as pd

from scripts.fetch_live_crypto import (
    data_dict_to_prices,
    export_live_prices_csv,
    save_market_data,
)


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

    def test_export_live_prices_csv(self):
        dates = pd.date_range("2026-01-01", periods=3, freq="1h")
        prices_df = pd.DataFrame(
            {"BTC/USDT": [100.0, 101.0, 102.0], "ETH/USDT": [50.0, 51.0, 52.0]},
            index=dates,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "live_prices.csv"
            out = export_live_prices_csv(prices_df, csv_path)
            self.assertEqual(out, csv_path)
            self.assertTrue(csv_path.is_file())

            loaded = pd.read_csv(csv_path, index_col=0)
            self.assertEqual(list(loaded.columns), ["BTC/USDT", "ETH/USDT"])
            self.assertEqual(len(loaded), 3)

    def test_export_live_prices_csv_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            with self.assertRaises(TypeError):
                export_live_prices_csv("not_a_dataframe", csv_path)

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

    def test_compute_live_market_impact_buy_and_sell(self):
        from scripts.fetch_live_crypto import compute_live_market_impact

        bids = [[100.0, 2.0], [99.0, 5.0], [98.0, 10.0]]
        asks = [[101.0, 2.0], [102.0, 5.0], [103.0, 10.0]]

        # Buy order: $202 should fill level 1 (2.0 @ 101.0 = $202)
        res_buy = compute_live_market_impact(202.0, bids, asks, side="buy", quote_currency="USDT")
        self.assertEqual(res_buy["side"], "buy")
        self.assertAlmostEqual(res_buy["vwap_price"], 101.0, places=2)
        self.assertAlmostEqual(res_buy["mid_price"], 100.5, places=2)
        self.assertAlmostEqual(res_buy["executed_order_size"], 202.0, places=2)
        self.assertAlmostEqual(res_buy["executed_base_qty"], 2.0, places=4)
        self.assertTrue(res_buy["fully_filled"])
        self.assertAlmostEqual(res_buy["price_impact_bps"], 0.0, places=1)
        self.assertGreater(res_buy["slippage_bps"], 0.0)

        # Larger Buy order walking into level 2: $202 + $102 = $304
        res_buy2 = compute_live_market_impact(304.0, bids, asks, side="buy", quote_currency="USDT")
        self.assertTrue(res_buy2["fully_filled"])
        self.assertGreater(res_buy2["vwap_price"], 101.0)
        self.assertGreater(res_buy2["price_impact_bps"], 0.0)

        # Sell order: $200 should fill level 1 (2.0 @ 100.0 = $200)
        res_sell = compute_live_market_impact(200.0, bids, asks, side="sell", quote_currency="USDT")
        self.assertEqual(res_sell["side"], "sell")
        self.assertAlmostEqual(res_sell["vwap_price"], 100.0, places=2)
        self.assertTrue(res_sell["fully_filled"])
        self.assertAlmostEqual(res_sell["price_impact_bps"], 0.0, places=1)
        self.assertGreater(res_sell["slippage_bps"], 0.0)

    def test_compute_live_market_impact_partial_fill(self):
        from scripts.fetch_live_crypto import compute_live_market_impact

        bids = [[100.0, 1.0]]  # Depth = $100
        asks = [[101.0, 1.0]]  # Depth = $101

        # Request $500 buy against $101 ask book
        res = compute_live_market_impact(500.0, bids, asks, side="buy")
        self.assertFalse(res["fully_filled"])
        self.assertAlmostEqual(res["executed_order_size"], 101.0, places=2)
        self.assertAlmostEqual(res["unfilled_size"], 399.0, places=2)
        self.assertAlmostEqual(res["fill_percentage"], 20.2, places=1)

    def test_compute_live_market_impact_validation(self):
        from scripts.fetch_live_crypto import compute_live_market_impact

        bids = [[100.0, 1.0]]
        asks = [[101.0, 1.0]]

        for bad_size in (0.0, -100.0, True, False, "bad"):
            with self.subTest(bad_size=bad_size), self.assertRaises(ValueError):
                compute_live_market_impact(bad_size, bids, asks)

        with self.assertRaises(ValueError):
            compute_live_market_impact(100.0, bids, asks, side="invalid")

        with self.assertRaises(ValueError):
            compute_live_market_impact(100.0, [], asks)

        with self.assertRaises(ValueError):
            compute_live_market_impact(100.0, bids, [])

    def test_print_live_orderbook_ladder(self):
        from scripts.fetch_live_crypto import print_live_orderbook_ladder
        import io
        from contextlib import redirect_stdout

        bids = [[100.0, 1.5], [99.0, 2.5]]
        asks = [[101.0, 1.2], [102.0, 3.0]]

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_live_orderbook_ladder("BTC/USDT", bids, asks, depth_levels=2)
        out = buf.getvalue()
        self.assertIn("Depth Ladder for BTC/USDT", out)
        self.assertIn("#1", out)
        self.assertIn("$100.00", out)
        self.assertIn("$101.00", out)

        # Non-positive depth levels shouldn't fail or output anything
        buf_empty = io.StringIO()
        with redirect_stdout(buf_empty):
            print_live_orderbook_ladder("BTC/USDT", bids, asks, depth_levels=0)
        self.assertEqual(buf_empty.getvalue(), "")

    def test_export_live_orderbook_csv(self):
        from scripts.fetch_live_crypto import export_live_orderbook_csv

        bids = [[100.0, 1.5], [99.0, 2.5]]
        asks = [[101.0, 1.2], [102.0, 3.0]]
        ob_dict = {"bids": bids, "asks": asks}

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "orderbook_ladder.csv"
            out = export_live_orderbook_csv(ob_dict, csv_path)
            self.assertEqual(out, csv_path)
            self.assertTrue(csv_path.is_file())

            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 2)
            self.assertIn("level", df.columns)
            self.assertIn("bid_price", df.columns)
            self.assertIn("ask_price", df.columns)
            self.assertIn("bid_depth", df.columns)
            self.assertIn("ask_depth", df.columns)

        # Test list of simulation records export
        sim_records = [
            {"symbol": "BTC/USDT", "side": "buy", "order_size": 1000.0, "vwap_price": 101.0, "slippage_bps": 5.0},
            {"symbol": "ETH/USDT", "side": "buy", "order_size": 500.0, "vwap_price": 50.5, "slippage_bps": 3.0},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sim_records.csv"
            out = export_live_orderbook_csv(sim_records, csv_path)
            self.assertTrue(csv_path.is_file())
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 2)
            self.assertIn("vwap_price", df.columns)
            self.assertIn("slippage_bps", df.columns)

    def test_export_live_orderbook_csv_invalid(self):
        from scripts.fetch_live_crypto import export_live_orderbook_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "invalid.csv"
            with self.assertRaises(ValueError):
                export_live_orderbook_csv("not_a_valid_record", csv_path)

    def test_fetch_orderbook_ccxt_validation(self):
        from scripts.fetch_live_crypto import fetch_orderbook_ccxt

        for bad_symbol in ("", "   ", None):
            with self.subTest(bad_symbol=bad_symbol), self.assertRaises(ValueError):
                fetch_orderbook_ccxt(symbol=bad_symbol)

        for bad_limit in (0, -5, True, False, "bad"):
            with self.subTest(bad_limit=bad_limit), self.assertRaises(ValueError):
                fetch_orderbook_ccxt(limit=bad_limit)

        with self.assertRaises(ValueError):
            fetch_orderbook_ccxt(exchange_id="invalid_exchange_xyz")


if __name__ == "__main__":
    unittest.main()
