import json
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from scripts.fetch_coinbase_crypto import (
    calculate_market_impact_slippage,
    export_coinbase_prices_csv,
    export_coinbase_orderbook_csv,
    fetch_coinbase_candles,
    fetch_coinbase_multi_assets,
    fetch_coinbase_orderbook,
    fetch_coinbase_spot_price,
    normalize_coinbase_symbol,
    print_coinbase_orderbook_ladder,
    main,
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

    def test_orderbook_invalid_params(self):
        with self.assertRaises(ValueError):
            fetch_coinbase_orderbook("BTC-USD", level=4)
        with self.assertRaises(ValueError):
            calculate_market_impact_slippage(-100, [["10", "1"]], [["11", "1"]])
        with self.assertRaises(ValueError):
            calculate_market_impact_slippage(100, [["10", "1"]], [["11", "1"]], side="invalid")
        with self.assertRaises(ValueError):
            calculate_market_impact_slippage(100, [], [])

    @patch("urllib.request.urlopen")
    def test_fetch_orderbook_mock(self, mock_urlopen):
        mock_data = {
            "sequence": 12345,
            "bids": [["60000.0", "1.5", 3], ["59900.0", "2.0", 5]],
            "asks": [["60100.0", "1.2", 2], ["60200.0", "3.0", 4]],
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        ob = fetch_coinbase_orderbook("BTC-USD", level=2)
        self.assertIn("bids", ob)
        self.assertIn("asks", ob)
        self.assertEqual(len(ob["bids"]), 2)
        self.assertEqual(float(ob["bids"][0][0]), 60000.0)

    def test_calculate_market_impact_slippage(self):
        bids = [["60000.0", "1.0"], ["59900.0", "2.0"]]
        asks = [["60100.0", "1.0"], ["60200.0", "2.0"]]

        # 1. Buy order small enough to be filled entirely by level 1 ask (30,050 USD = 0.5 BTC at 60100)
        res_buy_small = calculate_market_impact_slippage(30050.0, bids, asks, side="buy")
        self.assertTrue(res_buy_small["fully_filled"])
        self.assertAlmostEqual(res_buy_small["vwap_price"], 60100.0, places=4)
        self.assertAlmostEqual(res_buy_small["mid_price"], 60050.0, places=4)
        self.assertAlmostEqual(res_buy_small["spread_usd"], 100.0, places=4)
        self.assertAlmostEqual(res_buy_small["price_impact_bps"], 0.0, places=4)
        self.assertGreater(res_buy_small["slippage_bps"], 0.0)

        # Test with corrupted zero-price level in orderbook
        corrupt_asks = [["0.0", "1.0"], ["60100.0", "1.0"]]
        res_corrupt = calculate_market_impact_slippage(30050.0, bids, corrupt_asks, side="buy")
        self.assertTrue(res_corrupt["fully_filled"])
        self.assertAlmostEqual(res_corrupt["vwap_price"], 60100.0, places=4)

        # 2. Buy order larger than level 1 ask (needs 60100 USD + 60200 USD = 120,300 USD)
        res_buy_large = calculate_market_impact_slippage(120300.0, bids, asks, side="buy")
        self.assertTrue(res_buy_large["fully_filled"])
        # VWAP must be higher than 60100 because it absorbed part of level 2 (60200)
        self.assertGreater(res_buy_large["vwap_price"], 60100.0)
        self.assertGreater(res_buy_large["price_impact_bps"], 0.0)

        # 3. Sell order small enough for level 1 bid (30,000 USD = 0.5 BTC at 60000)
        res_sell_small = calculate_market_impact_slippage(30000.0, bids, asks, side="sell")
        self.assertTrue(res_sell_small["fully_filled"])
        self.assertAlmostEqual(res_sell_small["vwap_price"], 60000.0, places=4)
        self.assertAlmostEqual(res_sell_small["price_impact_bps"], 0.0, places=4)

        # 4. Order exceeding total book depth
        res_huge = calculate_market_impact_slippage(1_000_000.0, bids, asks, side="buy")
        self.assertFalse(res_huge["fully_filled"])
        self.assertGreater(res_huge["unfilled_usd"], 0.0)
        self.assertEqual(res_huge["executed_usd"], 60100.0 * 1.0 + 60200.0 * 2.0)

    def test_export_coinbase_prices_csv(self):
        import tempfile
        from pathlib import Path

        df = pd.DataFrame(
            {"BTC-USD": [60000.0, 61000.0], "ETH-USD": [3000.0, 3100.0]},
            index=pd.date_range("2026-01-01", periods=2, freq="1D"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "coinbase.csv"
            export_coinbase_prices_csv(df, out_csv)
            self.assertTrue(out_csv.exists())
            loaded = pd.read_csv(out_csv, index_col=0)
            self.assertEqual(list(loaded.columns), ["BTC-USD", "ETH-USD"])
            self.assertEqual(len(loaded), 2)

    def test_export_coinbase_orderbook_csv(self):
        import tempfile
        from pathlib import Path

        records = [
            {
                "symbol": "BTC-USD",
                "side": "buy",
                "order_size_usd": 50000.0,
                "mid_price": 60050.0,
                "vwap_price": 60100.0,
                "spread_usd": 100.0,
                "spread_bps": 16.65,
                "slippage_bps": 8.33,
                "price_impact_bps": 0.0,
                "fully_filled": True,
                "unfilled_usd": 0.0,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "orderbook.csv"
            export_coinbase_orderbook_csv(records, out_csv)
            self.assertTrue(out_csv.exists())
            loaded = pd.read_csv(out_csv)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded["symbol"].iloc[0], "BTC-USD")
            self.assertEqual(loaded["side"].iloc[0], "buy")
            self.assertEqual(loaded["slippage_bps"].iloc[0], 8.33)

    @patch("sys.stdout")
    @patch("scripts.fetch_coinbase_crypto.fetch_coinbase_orderbook")
    def test_coinbase_orderbook_cli(self, mock_fetch_ob, mock_stdout):
        import io
        import tempfile
        from pathlib import Path

        mock_fetch_ob.return_value = {
            "bids": [["60000.0", "2.0"]],
            "asks": [["60100.0", "2.0"]],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = str(Path(tmpdir) / "cb_ob.csv")
            test_args = [
                "fetch_coinbase_crypto.py",
                "--symbols", "BTC",
                "--orderbook",
                "--order-size", "25000",
                "--side", "buy",
                "--book-level", "2",
                "--export-csv", out_csv,
            ]
            with patch("sys.argv", test_args):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    main()
                output = captured.getvalue()
                self.assertIn("Coinbase order book", output)
                self.assertIn("BTC-USD", output)
                self.assertIn("Saved Coinbase orderbook impact metrics", output)
                self.assertTrue(Path(out_csv).exists())

    def test_print_coinbase_orderbook_ladder(self):
        import io
        bids = [["60000.0", "1.5"], ["59900.0", "2.0"]]
        asks = [["60100.0", "1.2"], ["60200.0", "2.5"]]

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            print_coinbase_orderbook_ladder("BTC-USD", bids, asks, depth_levels=2)
        out = captured.getvalue()
        self.assertIn("Depth Ladder for BTC-USD", out)
        self.assertIn("#1", out)
        self.assertIn("$60,000.00", out)
        self.assertIn("$60,100.00", out)

        # depth_levels <= 0 does nothing
        captured_zero = io.StringIO()
        with patch("sys.stdout", captured_zero):
            print_coinbase_orderbook_ladder("BTC-USD", bids, asks, depth_levels=0)
        self.assertEqual(captured_zero.getvalue(), "")

    def test_export_coinbase_orderbook_csv_ladder(self):
        import tempfile
        from pathlib import Path

        ob_data = {
            "bids": [["60000.0", "1.5"]],
            "asks": [["60100.0", "1.2"]],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_ladder = Path(tmpdir) / "ladder.csv"
            export_coinbase_orderbook_csv(ob_data, out_ladder)
            self.assertTrue(out_ladder.exists())
            loaded = pd.read_csv(out_ladder)
            self.assertEqual(len(loaded), 1)
            self.assertIn("level", loaded.columns)
            self.assertIn("bid_price", loaded.columns)
            self.assertIn("ask_price", loaded.columns)
            self.assertIn("bid_depth_usd", loaded.columns)
            self.assertEqual(loaded["bid_price"].iloc[0], 60000.0)

            # Single dict record
            out_single = Path(tmpdir) / "single.csv"
            single_rec = {"symbol": "BTC-USD", "vwap_price": 60050.0}
            export_coinbase_orderbook_csv(single_rec, out_single)
            self.assertTrue(out_single.exists())
            loaded_single = pd.read_csv(out_single)
            self.assertEqual(len(loaded_single), 1)
            self.assertEqual(loaded_single["vwap_price"].iloc[0], 60050.0)

            with self.assertRaises(ValueError):
                export_coinbase_orderbook_csv("invalid_type", Path(tmpdir) / "err.csv")

    @patch("sys.stdout")
    @patch("scripts.fetch_coinbase_crypto.fetch_coinbase_orderbook")
    def test_coinbase_orderbook_cli_with_depth_ladder(self, mock_fetch_ob, mock_stdout):
        import io
        import tempfile
        from pathlib import Path

        mock_fetch_ob.return_value = {
            "bids": [["60000.0", "2.0"], ["59900.0", "3.0"]],
            "asks": [["60100.0", "2.0"], ["60200.0", "4.0"]],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = str(Path(tmpdir) / "cb_ladder.csv")
            test_args = [
                "fetch_coinbase_crypto.py",
                "--symbols", "BTC",
                "--orderbook",
                "--order-size", "25000",
                "--side", "buy",
                "--depth-levels", "2",
                "--export-csv", out_csv,
            ]
            with patch("sys.argv", test_args):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    main()
                output = captured.getvalue()
                self.assertIn("Coinbase order book", output)
                self.assertIn("Depth Ladder for BTC-USD", output)
                self.assertIn("#1", output)
                self.assertIn("Saved Coinbase orderbook impact metrics", output)
                self.assertTrue(Path(out_csv).exists())


if __name__ == "__main__":
    unittest.main()

