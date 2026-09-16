import unittest
from scripts.crypto_kimchi_hedging import simulate_kimchi_hedging


class KimchiHedgingTests(unittest.TestCase):
    def test_profitable_hedging_simulation(self):
        # Entry at 1.0% KP, Exit at 4.0% KP over 30 days
        res = simulate_kimchi_hedging(
            capital_krw=100_000_000.0,
            entry_kp_pct=1.0,
            exit_kp_pct=4.0,
            holding_days=30,
            daily_funding_rate=0.0003,
        )
        self.assertTrue(res.is_profitable)
        self.assertGreater(res.gross_spread_profit_krw, 2_000_000)
        self.assertGreater(res.funding_income_krw, 0)
        self.assertGreater(res.annualized_apr_pct, 10.0)
        self.assertLess(res.break_even_spread_pct, 1.0)

    def test_unfavorable_negative_spread(self):
        # Premium compressed from 5.0% to 1.0% with negative funding rate
        res = simulate_kimchi_hedging(
            capital_krw=10_000_000.0,
            entry_kp_pct=5.0,
            exit_kp_pct=1.0,
            holding_days=10,
            daily_funding_rate=-0.001,
        )
        self.assertFalse(res.is_profitable)
        self.assertLess(res.net_profit_krw, 0)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            simulate_kimchi_hedging(capital_krw=-1000)
        with self.assertRaises(ValueError):
            simulate_kimchi_hedging(holding_days=0)
        with self.assertRaises(ValueError):
            simulate_kimchi_hedging(binance_fee=-0.01)


if __name__ == "__main__":
    unittest.main()
