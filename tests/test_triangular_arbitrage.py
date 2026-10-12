import unittest
from scripts.crypto_triangular_arbitrage import (
    calculate_triangular_arbitrage,
    filter_profitable_opportunities,
    scan_triangular_pairs,
)


class TriangularArbitrageTests(unittest.TestCase):
    def test_arbitrage_profitable_cycle(self):
        # Discrepancy case:
        # BTC/USDT = 50,000
        # ETH/USDT = 3,000
        # Theoretical ETH/BTC = 3000 / 50000 = 0.060
        # If market ETH/BTC = 0.055 (ETH is cheap relative to BTC on cross-market)
        # Buying ETH with USDT (Leg 1), Selling ETH for BTC (Leg 2), Selling BTC for USDT (Leg 3)
        # Or reverse:
        opps = calculate_triangular_arbitrage(
            p_a_quote=50000.0,
            p_b_quote=3000.0,
            p_b_a=0.055,
            fee_rate=0.0005,
            slippage=0.0002,
        )
        self.assertEqual(len(opps), 2)
        # One cycle should have positive gross return
        profitable_opps = [o for o in opps if o.is_profitable]
        self.assertGreaterEqual(len(profitable_opps), 1)

    def test_fee_drag_eliminates_small_gap(self):
        # 0.05% theoretical gap but 0.2% fee per leg -> should not be net profitable
        opps = calculate_triangular_arbitrage(
            p_a_quote=50000.0,
            p_b_quote=3000.0,
            p_b_a=0.05995,
            fee_rate=0.002,  # 0.2% * 3 = 0.6% drag
            slippage=0.001,
        )
        for o in opps:
            self.assertFalse(o.is_profitable)

    def test_arbitrage_with_capital(self):
        opps = calculate_triangular_arbitrage(
            p_a_quote=50000.0,
            p_b_quote=3000.0,
            p_b_a=0.055,
            fee_rate=0.0005,
            slippage=0.0002,
            capital=10000.0,
        )
        self.assertEqual(len(opps), 2)
        for o in opps:
            self.assertEqual(o.capital, 10000.0)
            expected_profit = round(10000.0 * (o.net_return_pct / 100.0), 2)
            self.assertEqual(o.profit_dollars, expected_profit)
            d = o.to_dict()
            self.assertEqual(d["capital"], 10000.0)
            self.assertEqual(d["profit_dollars"], expected_profit)

    def test_filter_profitable_opportunities(self):
        opps = calculate_triangular_arbitrage(
            p_a_quote=50000.0,
            p_b_quote=3000.0,
            p_b_a=0.055,
            fee_rate=0.0005,
            slippage=0.0002,
        )
        profitable = filter_profitable_opportunities(opps, min_net_pct=0.0)
        self.assertTrue(all(o.net_return_pct >= 0.0 for o in profitable))
        strict = filter_profitable_opportunities(opps, min_net_pct=50.0)
        self.assertEqual(len(strict), 0)

    def test_scan_triangular_pairs_with_capital_and_hurdle(self):
        prices = {
            "BTC/USDT": 60000.0,
            "ETH/USDT": 3000.0,
            "ETH/BTC": 0.045,
        }
        opps = scan_triangular_pairs(prices, fee_rate=0.0005, capital=5000.0, min_profit_pct=0.0)
        self.assertGreater(len(opps), 0)
        self.assertTrue(all(o.capital == 5000.0 for o in opps))
        self.assertTrue(all(o.net_return_pct >= 0.0 for o in opps))

    def test_scan_triangular_pairs(self):
        prices = {
            "BTC/USDT": 60000.0,
            "ETH/USDT": 3000.0,
            "ETH/BTC": 0.045,  # Mispriced cross
        }
        opps = scan_triangular_pairs(prices, fee_rate=0.0005)
        self.assertGreater(len(opps), 0)
        self.assertTrue(any(o.is_profitable for o in opps))

    def test_scan_triangular_pairs_with_custom_delimiters(self):
        prices = {
            "BTC-USDT": 60000.0,
            "ETH_USDT": 3000.0,
            "ETH-BTC": 0.045,
        }
        opps = scan_triangular_pairs(prices, fee_rate=0.0005)
        self.assertGreater(len(opps), 0)
        self.assertTrue(any(o.is_profitable for o in opps))

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            calculate_triangular_arbitrage(0, 3000, 0.05)
        with self.assertRaises(ValueError):
            calculate_triangular_arbitrage(50000, 3000, 0.05, fee_rate=-0.01)
        with self.assertRaises(ValueError):
            calculate_triangular_arbitrage(50000, 3000, 0.05, capital=-100)

    def test_arbitrage_to_dict_serialization(self):
        opps = calculate_triangular_arbitrage(50000.0, 3000.0, 0.055)
        d = opps[0].to_dict()
        self.assertIn("cycle", d)
        self.assertIn("gross_return_pct", d)
        self.assertIn("fee_drag_pct", d)
        self.assertIn("net_return_pct", d)
        self.assertIn("is_profitable", d)
        self.assertIn("legs", d)
        self.assertEqual(len(d["legs"]), 3)

    def test_export_triangular_arbitrage_csv(self):
        import tempfile
        from pathlib import Path
        import pandas as pd
        from scripts.crypto_triangular_arbitrage import export_triangular_arbitrage_csv

        opps = calculate_triangular_arbitrage(50000.0, 3000.0, 0.055)
        csv_str = export_triangular_arbitrage_csv(opps)
        self.assertIn("cycle", csv_str)
        self.assertIn("gross_return_pct", csv_str)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "arbitrage.csv"
            export_triangular_arbitrage_csv(opps, out_file)
            self.assertTrue(out_file.exists())
            df = pd.read_csv(out_file)
            self.assertEqual(len(df), len(opps))
            self.assertIn("cycle", df.columns)
            self.assertIn("legs", df.columns)

    def test_export_triangular_arbitrage_csv_with_capital(self):
        import tempfile
        from pathlib import Path
        import pandas as pd
        from scripts.crypto_triangular_arbitrage import export_triangular_arbitrage_csv

        opps = calculate_triangular_arbitrage(50000.0, 3000.0, 0.055, capital=10000.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "arbitrage_capital.csv"
            export_triangular_arbitrage_csv(opps, out_file)
            self.assertTrue(out_file.exists())
            df = pd.read_csv(out_file)
            self.assertIn("capital", df.columns)
            self.assertIn("profit_dollars", df.columns)
            self.assertEqual(float(df["capital"].iloc[0]), 10000.0)


if __name__ == "__main__":
    unittest.main()
