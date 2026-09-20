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


if __name__ == "__main__":
    unittest.main()
