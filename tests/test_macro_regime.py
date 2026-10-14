"""Tests for Crypto Macro Regime cash allocator module."""

import unittest
from scripts.crypto_macro_regime import (
    adjust_cash_allocation_by_regime,
    fetch_fear_and_greed_history,
    fetch_fear_and_greed_index,
)


class MacroRegimeTests(unittest.TestCase):
    def test_fetch_fear_and_greed_history(self):
        df = fetch_fear_and_greed_history(limit=5, timeout=2.0)
        self.assertEqual(len(df), 5)
        self.assertIn("date", df.columns)
        self.assertIn("value", df.columns)
        self.assertIn("classification", df.columns)
        with self.assertRaises(ValueError):
            fetch_fear_and_greed_history(limit=0)
        with self.assertRaises(ValueError):
            fetch_fear_and_greed_history(limit=5, timeout=-1.0)

    def test_extreme_greed_allocates_high_cash(self):
        base_w = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        res = adjust_cash_allocation_by_regime(base_w, fng_value=85, total_wallet=10000.0)

        self.assertEqual(res["market_regime"], "Extreme Greed")
        self.assertEqual(res["cash_ratio"], 0.40)
        self.assertIn("USDT (Cash)", res["adjusted_weights"])
        self.assertEqual(res["adjusted_weights"]["USDT (Cash)"], 0.40)
        self.assertAlmostEqual(sum(res["adjusted_weights"].values()), 1.0, places=3)

    def test_extreme_fear_full_deployment(self):
        base_w = {"BTC/USDT": 0.7, "ETH/USDT": 0.3}
        res = adjust_cash_allocation_by_regime(base_w, fng_value=15, total_wallet=10000.0)

        self.assertEqual(res["market_regime"], "Extreme Fear")
        self.assertEqual(res["cash_ratio"], 0.0)
        self.assertNotIn("USDT (Cash)", res["adjusted_weights"])
        self.assertAlmostEqual(sum(res["adjusted_weights"].values()), 1.0, places=3)

    def test_out_of_range_fng_rejected(self):
        with self.assertRaises(ValueError):
            adjust_cash_allocation_by_regime({"BTC/USDT": 1.0}, fng_value=120)

    def test_invalid_wallet_rejected(self):
        for wallet in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(wallet=wallet), self.assertRaises(ValueError):
                adjust_cash_allocation_by_regime(
                    {"BTC/USDT": 1.0}, fng_value=50, total_wallet=wallet
                )

    def test_fetch_fear_and_greed_returns_valid_tuple(self):
        val, label = fetch_fear_and_greed_index()
        self.assertIsInstance(val, int)
        self.assertIsInstance(label, str)
        self.assertGreaterEqual(val, 0)
        self.assertLessEqual(val, 100)

    def test_filter_crypto_weights_for_freqtrade(self):
        from scripts.crypto_macro_regime import filter_crypto_weights_for_freqtrade

        raw_weights = {"BTC/USDT": 0.35, "ETH/USDT": 0.25, "USDT (Cash)": 0.40}
        clean_pairs, cash = filter_crypto_weights_for_freqtrade(raw_weights)

        self.assertEqual(cash, 0.40)
        self.assertNotIn("USDT (Cash)", clean_pairs)
        self.assertIn("BTC/USDT", clean_pairs)
        self.assertIn("ETH/USDT", clean_pairs)

    def test_calculate_macro_regime_weights_helper(self):
        from scripts.crypto_macro_regime import calculate_macro_regime_weights

        base_w = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        adj = calculate_macro_regime_weights(base_w, fear_and_greed_value=85)
        self.assertIn("USDT (Cash)", adj)
        self.assertEqual(adj["USDT (Cash)"], 0.40)
        self.assertAlmostEqual(sum(adj.values()), 1.0, places=3)

    def test_invalid_parameters_and_weights_rejected(self):
        for bad_fng in (True, False, -1, 101, 50.5):
            with self.subTest(bad_fng=bad_fng), self.assertRaises(ValueError):
                adjust_cash_allocation_by_regime({"BTC/USDT": 1.0}, fng_value=bad_fng)

        for bad_limit in (0, -1, True, False, 1.5):
            with self.subTest(bad_limit=bad_limit), self.assertRaises(ValueError):
                fetch_fear_and_greed_index(limit=bad_limit)

        for bad_weights in ("not_a_dict", [1, 2], {"": 1.0}, {"BTC": -0.5}, {"BTC": True}, {"BTC": float("nan")}):
            with self.subTest(bad_weights=bad_weights), self.assertRaises(ValueError):
                adjust_cash_allocation_by_regime(bad_weights, fng_value=50)

    def test_export_macro_regime_csv(self):
        import tempfile
        from pathlib import Path
        import pandas as pd
        from scripts.crypto_macro_regime import export_macro_regime_csv

        sample_res = {
            "fng_value": 75,
            "market_regime": "Extreme Greed",
            "cash_ratio": 0.40,
            "crypto_ratio": 0.60,
            "total_wallet": 10000.0,
            "adjusted_weights": {
                "BTC/USDT": 0.35,
                "ETH/USDT": 0.25,
                "USDT (Cash)": 0.40,
            },
            "capital_allocation": {
                "BTC/USDT": 3500.0,
                "ETH/USDT": 2500.0,
                "USDT (Cash)": 4000.0,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sub" / "macro_regime.csv"
            res_path = export_macro_regime_csv(sample_res, out_file)
            self.assertEqual(res_path, out_file)
            self.assertTrue(out_file.exists())

            df = pd.read_csv(out_file)
            self.assertIn("section", df.columns)
            self.assertIn("asset", df.columns)
            self.assertIn("weight", df.columns)
            self.assertIn("allocated_usd", df.columns)
            self.assertTrue(any(df["asset"] == "BTC/USDT"))
            self.assertTrue(any(df["asset"] == "market_regime"))

    def test_sentiment_trend_regimes(self):
        from scripts.crypto_macro_regime import (
            calculate_sentiment_trend_regime,
            adjust_weights_by_sentiment_trend,
        )

        # 1. Bull Overheating: Price > SMA200 and F&G >= 75
        res_overheating = calculate_sentiment_trend_regime(
            fng_value=82,
            btc_price=72000.0,
            btc_sma200=55000.0,
        )
        self.assertEqual(res_overheating["composite_regime"], "Bull Market Overheating")
        self.assertEqual(res_overheating["cash_ratio"], 0.35)
        self.assertTrue(res_overheating["is_bullish_trend"])
        self.assertGreater(res_overheating["trend_distance_pct"], 0.0)

        # 2. Bull Dip: Price > SMA200 and F&G < 45
        res_dip = calculate_sentiment_trend_regime(
            fng_value=32,
            btc_price=60000.0,
            btc_sma200=55000.0,
        )
        self.assertEqual(res_dip["composite_regime"], "Bull Market Dip")
        self.assertEqual(res_dip["cash_ratio"], 0.0)
        self.assertEqual(res_dip["crypto_ratio"], 1.0)

        # 3. Bear Market Rally: Price < SMA200 and F&G >= 60
        res_bear_rally = calculate_sentiment_trend_regime(
            fng_value=68,
            btc_price=45000.0,
            btc_sma200=55000.0,
        )
        self.assertEqual(res_bear_rally["composite_regime"], "Bear Market Rally (Dead-Cat Bounce)")
        self.assertEqual(res_bear_rally["cash_ratio"], 0.50)
        self.assertFalse(res_bear_rally["is_bullish_trend"])

        # 4. Bear Market Capitulation: Price < SMA200 and F&G < 25
        res_capitulation = calculate_sentiment_trend_regime(
            fng_value=18,
            btc_price=38000.0,
            btc_sma200=55000.0,
        )
        self.assertEqual(res_capitulation["composite_regime"], "Bear Market Capitulation")
        self.assertEqual(res_capitulation["cash_ratio"], 0.15)

        # 5. Weight adjustment with capital allocation
        base_w = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        adj_res = adjust_weights_by_sentiment_trend(
            base_weights=base_w,
            fng_value=82,
            btc_price=72000.0,
            btc_sma200=55000.0,
            total_wallet=20000.0,
        )
        self.assertEqual(adj_res["composite_regime"], "Bull Market Overheating")
        self.assertEqual(adj_res["adjusted_weights"]["USDT (Cash)"], 0.35)
        self.assertEqual(adj_res["capital_allocation"]["USDT (Cash)"], 7000.0)
        self.assertAlmostEqual(sum(adj_res["adjusted_weights"].values()), 1.0, places=3)
        self.assertNotIn("USDT (Cash)", adj_res["freqtrade_pair_weights"])

    def test_sentiment_trend_validation(self):
        from scripts.crypto_macro_regime import calculate_sentiment_trend_regime

        with self.assertRaises(ValueError):
            calculate_sentiment_trend_regime(fng_value=-5, btc_price=60000.0, btc_sma200=55000.0)
        with self.assertRaises(ValueError):
            calculate_sentiment_trend_regime(fng_value=50, btc_price=-100.0, btc_sma200=55000.0)
        with self.assertRaises(ValueError):
            calculate_sentiment_trend_regime(fng_value=50, btc_price=60000.0, btc_sma200=0.0)
        with self.assertRaises(ValueError):
            calculate_sentiment_trend_regime(fng_value=50, btc_price=float("nan"), btc_sma200=55000.0)

    def test_main_cli_trend_regime(self):
        import tempfile
        from pathlib import Path
        import json
        from unittest.mock import patch
        from scripts.crypto_macro_regime import main as macro_main

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "trend_out.json"
            csv_file = Path(tmpdir) / "trend_out.csv"
            cli_args = [
                "crypto_macro_regime.py",
                "--fng", "85",
                "--btc-price", "72000",
                "--btc-sma200", "58000",
                "--use-trend",
                "--wallet-size", "50000",
                "--export-json", str(json_file),
                "--export-csv", str(csv_file),
            ]
            with patch("sys.argv", cli_args):
                macro_main()

            self.assertTrue(json_file.exists())
            self.assertTrue(csv_file.exists())
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["composite_regime"], "Bull Market Overheating")
            self.assertEqual(data["cash_ratio"], 0.35)
            self.assertEqual(data["total_wallet"], 50000.0)


if __name__ == "__main__":
    unittest.main()

