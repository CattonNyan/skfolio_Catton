import json
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from scripts.fetch_bithumb_crypto import (
    normalize_bithumb_symbol,
    fetch_bithumb_candlestick,
    fetch_bithumb_spot_price,
    fetch_bithumb_multi_assets,
    fetch_bithumb_orderbook,
    compute_bithumb_spread,
    compute_bithumb_market_impact,
    export_bithumb_prices_csv,
    export_bithumb_orderbook_csv,
    main,
    VALID_BITHUMB_INTERVALS,
)



class BithumbFetcherTests(unittest.TestCase):
    def test_normalize_bithumb_symbol(self):
        self.assertEqual(normalize_bithumb_symbol("BTC"), ("BTC", "KRW"))
        self.assertEqual(normalize_bithumb_symbol("eth/krw"), ("ETH", "KRW"))
        self.assertEqual(normalize_bithumb_symbol("KRW-SOL"), ("SOL", "KRW"))
        self.assertEqual(normalize_bithumb_symbol("XRP_KRW"), ("XRP", "KRW"))
        self.assertEqual(normalize_bithumb_symbol("BTC_USDT", payment_currency="USDT"), ("BTC", "USDT"))

        with self.assertRaises(ValueError):
            normalize_bithumb_symbol("")
        with self.assertRaises(ValueError):
            normalize_bithumb_symbol(123)

    def test_invalid_interval(self):
        with self.assertRaises(ValueError):
            fetch_bithumb_candlestick("BTC", chart_interval="48h")

    @patch("urllib.request.urlopen")
    def test_fetch_bithumb_candlestick_mock(self, mock_urlopen):
        # Sample Bithumb mock response
        mock_data = {
            "status": "0000",
            "data": [
                [1700000000000, "50000000", "51000000", "52000000", "49500000", "120.5"],
                [1700086400000, "51000000", "51500000", "53000000", "50500000", "95.2"],
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        df = fetch_bithumb_candlestick("BTC", chart_interval="24h", count=2)
        self.assertEqual(len(df), 2)
        self.assertIn("close", df.columns)
        self.assertEqual(df["close"].iloc[-1], 51500000.0)

    @patch("scripts.fetch_bithumb_crypto.fetch_bithumb_candlestick")
    def test_fetch_bithumb_multi_assets(self, mock_fetch_candle):
        t1 = pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True)
        mock_fetch_candle.side_effect = [
            pd.DataFrame({"date": t1, "close": [100.0, 105.0]}),
            pd.DataFrame({"date": t1, "close": [50.0, 52.0]}),
        ]
        res = fetch_bithumb_multi_assets(["BTC", "ETH"], count=2)
        self.assertEqual(len(res.columns), 2)
        self.assertIn("BTC/KRW", res.columns)
        self.assertIn("ETH/KRW", res.columns)

    def test_orderbook_invalid_parameters(self):
        with self.assertRaises(ValueError):
            fetch_bithumb_orderbook("BTC", count=0)
        with self.assertRaises(ValueError):
            fetch_bithumb_orderbook("BTC", count=100)
        with self.assertRaises(ValueError):
            compute_bithumb_spread({})
        with self.assertRaises(ValueError):
            compute_bithumb_spread({"bids": [], "asks": []})

    @patch("urllib.request.urlopen")
    def test_fetch_bithumb_orderbook_mock(self, mock_urlopen):
        mock_data = {
            "status": "0000",
            "data": {
                "timestamp": "1700000000000",
                "order_currency": "BTC",
                "payment_currency": "KRW",
                "bids": [{"price": "90000000", "quantity": "0.5"}, {"price": "89900000", "quantity": "1.0"}],
                "asks": [{"price": "90050000", "quantity": "0.8"}, {"price": "90100000", "quantity": "1.2"}],
            }
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        ob = fetch_bithumb_orderbook("BTC-KRW", count=2)
        self.assertIn("bids", ob)
        self.assertIn("asks", ob)
        self.assertEqual(len(ob["bids"]), 2)

        spread_info = compute_bithumb_spread(ob)
        self.assertEqual(spread_info["best_bid"], 90000000.0)
        self.assertEqual(spread_info["best_ask"], 90050000.0)
        self.assertEqual(spread_info["spread_krw"], 50000.0)
        self.assertAlmostEqual(spread_info["mid_price"], 90025000.0, places=1)
        self.assertGreater(spread_info["spread_bps"], 0.0)
        self.assertGreater(spread_info["bid_depth_krw"], 0.0)
        self.assertGreater(spread_info["ask_depth_krw"], 0.0)

    @patch("urllib.request.urlopen")
    def test_fetch_bithumb_spot_price_mock(self, mock_urlopen):
        mock_data = {
            "status": "0000",
            "data": {
                "closing_price": "95000000",
            },
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        price = fetch_bithumb_spot_price("BTC")
        self.assertEqual(price, 95000000.0)

        with self.assertRaises(ValueError):
            fetch_bithumb_candlestick("BTC", count=0)

    def test_export_bithumb_prices_csv(self):
        import tempfile
        from pathlib import Path

        df = pd.DataFrame(
            {"BTC/KRW": [100.0, 105.0], "ETH/KRW": [50.0, 52.0]},
            index=pd.date_range("2026-01-01", periods=2, freq="1D"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "bithumb.csv"
            export_bithumb_prices_csv(df, out_csv)
            self.assertTrue(out_csv.exists())
            loaded = pd.read_csv(out_csv, index_col=0)
            self.assertEqual(list(loaded.columns), ["BTC/KRW", "ETH/KRW"])
            self.assertEqual(len(loaded), 2)

    def test_export_bithumb_orderbook_csv(self):
        import tempfile
        from pathlib import Path

        records = [
            {
                "symbol": "BTC/KRW",
                "best_bid": 100000000.0,
                "best_ask": 100050000.0,
                "mid_price": 100025000.0,
                "spread_krw": 50000.0,
                "spread_bps": 5.0,
                "bid_depth_krw": 500000000.0,
                "ask_depth_krw": 600000000.0,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "orderbook.csv"
            export_bithumb_orderbook_csv(records, out_csv)
            self.assertTrue(out_csv.exists())
            loaded = pd.read_csv(out_csv)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded["symbol"].iloc[0], "BTC/KRW")
            self.assertEqual(loaded["spread_bps"].iloc[0], 5.0)

    @patch("sys.stdout")
    @patch("scripts.fetch_bithumb_crypto.fetch_bithumb_orderbook")
    def test_orderbook_cli_execution(self, mock_fetch_ob, mock_stdout):
        import io
        import tempfile
        from pathlib import Path

        mock_fetch_ob.return_value = {
            "bids": [{"price": "90000000", "quantity": "1.0"}],
            "asks": [{"price": "90050000", "quantity": "1.0"}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = str(Path(tmpdir) / "bithumb_ob.csv")
            test_args = ["fetch_bithumb_crypto.py", "--symbols", "BTC", "--orderbook", "--orderbook-count", "10", "--export-csv", out_csv]
            with patch("sys.argv", test_args):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    main()
                output = captured.getvalue()
                self.assertIn("Bithumb order book & market depth", output)
                self.assertIn("BTC/KRW", output)
                self.assertIn("Saved Bithumb orderbook metrics", output)
                self.assertTrue(Path(out_csv).exists())

    def test_compute_bithumb_market_impact(self):
        mock_ob = {
            "order_currency": "BTC",
            "payment_currency": "KRW",
            "bids": [
                {"price": "100000000", "quantity": "0.5"},
                {"price": "99900000", "quantity": "1.0"},
            ],
            "asks": [
                {"price": "100100000", "quantity": "0.5"},
                {"price": "100200000", "quantity": "1.0"},
            ],
        }

        # 1. Buy side partial fill on level 1
        buy_res = compute_bithumb_market_impact(mock_ob, order_size_krw=25000000.0, side="buy")
        self.assertEqual(buy_res["symbol"], "BTC/KRW")
        self.assertEqual(buy_res["side"], "buy")
        self.assertEqual(buy_res["vwap_price"], 100100000.0)
        self.assertEqual(buy_res["executed_krw"], 25000000.0)
        self.assertTrue(buy_res["fully_filled"])
        self.assertAlmostEqual(buy_res["fill_percentage"], 100.0)
        self.assertGreater(buy_res["slippage_bps"], 0.0)

        # 2. Buy side multi-level sweep
        buy_deep = compute_bithumb_market_impact(mock_ob, order_size_krw=100000000.0, side="buy")
        self.assertGreater(buy_deep["vwap_price"], 100100000.0)
        self.assertTrue(buy_deep["fully_filled"])
        self.assertGreater(buy_deep["price_impact_bps"], 0.0)

        # 3. Sell side execution
        sell_res = compute_bithumb_market_impact(mock_ob, order_size_krw=30000000.0, side="sell")
        self.assertEqual(sell_res["side"], "sell")
        self.assertEqual(sell_res["vwap_price"], 100000000.0)
        self.assertTrue(sell_res["fully_filled"])
        self.assertGreater(sell_res["slippage_bps"], 0.0)

        # 4. Oversized order resulting in partial fill
        oversized = compute_bithumb_market_impact(mock_ob, order_size_krw=500000000.0, side="buy")
        self.assertFalse(oversized["fully_filled"])
        self.assertLess(oversized["fill_percentage"], 100.0)
        self.assertGreater(oversized["unfilled_krw"], 0.0)

    def test_compute_bithumb_market_impact_invalid(self):
        valid_ob = {
            "bids": [{"price": "100", "quantity": "1"}],
            "asks": [{"price": "101", "quantity": "1"}],
        }
        with self.assertRaises(ValueError):
            compute_bithumb_market_impact(valid_ob, order_size_krw=-500)
        with self.assertRaises(ValueError):
            compute_bithumb_market_impact(valid_ob, order_size_krw=1000, side="invalid")
        with self.assertRaises(ValueError):
            compute_bithumb_market_impact("not_dict", order_size_krw=1000)
        with self.assertRaises(ValueError):
            compute_bithumb_market_impact({"bids": []}, order_size_krw=1000)

    def test_export_bithumb_orderbook_csv_ladder_and_impact(self):
        import tempfile
        from pathlib import Path

        mock_ob = {
            "bids": [{"price": "100000000", "quantity": "0.5"}],
            "asks": [{"price": "100100000", "quantity": "0.5"}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            ladder_csv = Path(tmpdir) / "ladder.csv"
            export_bithumb_orderbook_csv(mock_ob, ladder_csv)
            self.assertTrue(ladder_csv.exists())
            df_ladder = pd.read_csv(ladder_csv)
            self.assertIn("bid_price", df_ladder.columns)
            self.assertIn("ask_price", df_ladder.columns)

            impact_data = compute_bithumb_market_impact(mock_ob, order_size_krw=10000000.0, side="buy")
            impact_csv = Path(tmpdir) / "impact.csv"
            export_bithumb_orderbook_csv(impact_data, impact_csv)
            self.assertTrue(impact_csv.exists())
            df_impact = pd.read_csv(impact_csv)
            self.assertIn("vwap_price", df_impact.columns)
            self.assertIn("slippage_bps", df_impact.columns)

    @patch("sys.stdout")
    @patch("scripts.fetch_bithumb_crypto.fetch_bithumb_orderbook")
    def test_orderbook_cli_with_order_size_and_depth(self, mock_fetch_ob, mock_stdout):
        import io
        import tempfile
        from pathlib import Path

        mock_fetch_ob.return_value = {
            "order_currency": "BTC",
            "payment_currency": "KRW",
            "bids": [{"price": "90000000", "quantity": "1.0"}],
            "asks": [{"price": "90050000", "quantity": "1.0"}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = str(Path(tmpdir) / "bithumb_impact.csv")
            test_args = [
                "fetch_bithumb_crypto.py",
                "--symbols", "BTC",
                "--orderbook",
                "--order-size", "50000000",
                "--side", "buy",
                "--depth-levels", "3",
                "--export-csv", out_csv,
            ]
            with patch("sys.argv", test_args):
                captured = io.StringIO()
                with patch("sys.stdout", captured):
                    main()
                output = captured.getvalue()
                self.assertIn("Institutional Order Execution Simulation", output)
                self.assertIn("Depth Ladder for BTC/KRW", output)
                self.assertIn("Saved Bithumb orderbook metrics", output)
                self.assertTrue(Path(out_csv).exists())


if __name__ == "__main__":
    unittest.main()

