import unittest
from scripts.crypto_triangular_arbitrage import (
    calculate_triangular_arbitrage,
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


if __name__ == "__main__":
    unittest.main()
