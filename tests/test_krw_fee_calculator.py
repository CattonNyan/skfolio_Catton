"""Unit tests for Korean crypto exchange fee and portfolio drag simulator."""

from __future__ import annotations

import unittest
from scripts.crypto_krw_fee_calculator import (
    KOREAN_EXCHANGE_PRESETS,
    compute_krw_fee_drag,
    get_korean_exchange_preset,
)


class KrwFeeCalculatorTests(unittest.TestCase):
    def test_get_korean_exchange_preset(self):
        for name in ["upbit", "bithumb_coupon", "bithumb_standard", "coinone", "korbit"]:
            preset = get_korean_exchange_preset(name)
            self.assertIn("maker_fee", preset)
            self.assertIn("taker_fee", preset)
            self.assertIn("withdrawal_fee_krw", preset)
            self.assertGreater(preset["maker_fee"], 0)

        # Invalid exchange name
        with self.assertRaises(ValueError):
            get_korean_exchange_preset("unknown_exchange")
        with self.assertRaises(ValueError):
            get_korean_exchange_preset("")

    def test_compute_krw_fee_drag_basic(self):
        capital = 10_000_000.0  # 1천만원
        turnover = 4.0          # 400%
        maker_fee = 0.0005      # 0.05%
        taker_fee = 0.0005      # 0.05%
        withdrawals = 10        # 10회
        w_fee = 1000.0          # 1천원

        res = compute_krw_fee_drag(
            portfolio_value_krw=capital,
            annual_turnover=turnover,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            annual_withdrawals=withdrawals,
            withdrawal_fee_krw=w_fee,
        )

        expected_volume = capital * turnover  # 40,000,000
        expected_trade_fee = expected_volume * 0.0005  # 20,000
        expected_cash_fee = withdrawals * w_fee  # 10,000
        expected_total_fee = expected_trade_fee + expected_cash_fee  # 30,000
        expected_drag = (expected_total_fee / capital) * 100.0  # 0.3%

        self.assertEqual(res["portfolio_value_krw"], capital)
        self.assertEqual(res["annual_trade_volume_krw"], expected_volume)
        self.assertEqual(res["annual_trading_fees_krw"], expected_trade_fee)
        self.assertEqual(res["annual_withdrawal_fees_krw"], expected_cash_fee)
        self.assertEqual(res["total_annual_fees_krw"], expected_total_fee)
        self.assertAlmostEqual(res["fee_drag_pct"], expected_drag, places=2)

    def test_zero_turnover_only_withdrawal_fees(self):
        capital = 50_000_000.0
        res = compute_krw_fee_drag(
            portfolio_value_krw=capital,
            annual_turnover=0.0,
            annual_withdrawals=5,
            withdrawal_fee_krw=1000.0,
        )
        self.assertEqual(res["annual_trading_fees_krw"], 0.0)
        self.assertEqual(res["annual_withdrawal_fees_krw"], 5000.0)
        self.assertEqual(res["total_annual_fees_krw"], 5000.0)
        self.assertEqual(res["effective_cost_bps"], 0.0)

    def test_invalid_parameters_rejected(self):
        # Invalid capital
        with self.assertRaises(ValueError):
            compute_krw_fee_drag(portfolio_value_krw=0.0)
        with self.assertRaises(ValueError):
            compute_krw_fee_drag(portfolio_value_krw=-1000.0)
        with self.assertRaises(ValueError):
            compute_krw_fee_drag(portfolio_value_krw=True)  # type: ignore

        # Invalid turnover
        with self.assertRaises(ValueError):
            compute_krw_fee_drag(portfolio_value_krw=1000.0, annual_turnover=-1.0)
        with self.assertRaises(ValueError):
            compute_krw_fee_drag(portfolio_value_krw=1000.0, annual_turnover=False)  # type: ignore

        # Invalid fee rates
        with self.assertRaises(ValueError):
            compute_krw_fee_drag(portfolio_value_krw=1000.0, maker_fee=-0.01)
        with self.assertRaises(ValueError):
            compute_krw_fee_drag(portfolio_value_krw=1000.0, maker_fee=1.5)
        with self.assertRaises(ValueError):
            compute_krw_fee_drag(portfolio_value_krw=1000.0, taker_fee=True)  # type: ignore

        # Invalid withdrawals
        with self.assertRaises(ValueError):
            compute_krw_fee_drag(portfolio_value_krw=1000.0, annual_withdrawals=-1)
        with self.assertRaises(ValueError):
            compute_krw_fee_drag(portfolio_value_krw=1000.0, annual_withdrawals=True)  # type: ignore


if __name__ == "__main__":
    unittest.main()
