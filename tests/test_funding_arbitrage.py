import json
import tempfile
import unittest
from pathlib import Path

from scripts.freqtrade_funding_arbitrage import (
    filter_funding_rate_pairs,
    generate_freqtrade_funding_config,
    export_funding_config,
)


class FundingArbitrageTests(unittest.TestCase):
    def test_filter_funding_rate_pairs(self):
        rates = {
            "HIGH/USDT": 0.0004,  # ~43.8% APR
            "MID/USDT": 0.00015,  # ~16.4% APR
            "LOW/USDT": 0.00003,  # ~3.3% APR
        }
        res = filter_funding_rate_pairs(rates, min_apr_pct=15.0)
        self.assertIn("HIGH/USDT", res)
        self.assertIn("MID/USDT", res)
        self.assertNotIn("LOW/USDT", res)
        # Verify sorted descending
        keys = list(res.keys())
        self.assertEqual(keys[0], "HIGH/USDT")
        self.assertEqual(keys[1], "MID/USDT")

    def test_generate_freqtrade_funding_config(self):
        pairs = ["BTC/USDT", "ETH/USDT"]
        cfg = generate_freqtrade_funding_config(pairs, stake_per_pair=300.0)
        self.assertEqual(cfg["stake_amount"], 300.0)
        self.assertEqual(cfg["trading_mode"], "futures")
        self.assertEqual(cfg["margin_mode"], "isolated")
        self.assertEqual(cfg["exchange"]["pair_whitelist"], pairs)

    def test_export_funding_config(self):
        pairs = ["SOL/USDT"]
        cfg = generate_freqtrade_funding_config(pairs, stake_per_pair=100.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "config_funding.json"
            export_funding_config(cfg, out_path)
            self.assertTrue(out_path.exists())
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["stake_amount"], 100.0)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            filter_funding_rate_pairs([], min_apr_pct=-1.0)
        with self.assertRaises(ValueError):
            generate_freqtrade_funding_config([], stake_per_pair=100.0)
        with self.assertRaises(ValueError):
            generate_freqtrade_funding_config(["BTC/USDT"], stake_per_pair=-50.0)


if __name__ == "__main__":
    unittest.main()
