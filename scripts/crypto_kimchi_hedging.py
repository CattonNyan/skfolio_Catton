"""Kimchi Premium Arbitrage and Delta-Neutral Hedging Simulator.

Simulates institutional basis trading and carry arbitrage between global and Korean exchanges:
- Entry when Kimchi Premium is narrow (or negative)
- Delta-neutral perpetual futures hedge on Binance/Bybit
- Accumulates 8-hour funding rates during holding window
- Unwinds position when Kimchi Premium widens to target
- Accurately deducts exchange maker/taker fees, transfer network fees, and calculates annualized APR.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass
class HedgingSimulationResult:
    """Kimchi premium hedging simulation results."""
    capital_krw: float
    entry_kp_pct: float
    exit_kp_pct: float
    holding_days: int
    gross_spread_profit_krw: float
    funding_income_krw: float
    total_fees_krw: float
    net_profit_krw: float
    net_return_pct: float
    annualized_apr_pct: float
    break_even_spread_pct: float
    is_profitable: bool


def simulate_kimchi_hedging(
    capital_krw: float = 50_000_000.0,
    entry_kp_pct: float = 1.0,
    exit_kp_pct: float = 4.0,
    holding_days: int = 30,
    daily_funding_rate: float = 0.0003,  # ~0.01% per 8h = 0.03% daily
    binance_fee: float = 0.0004,         # Binance VIP/maker/taker ~0.04%
    upbit_fee: float = 0.0005,           # Upbit 0.05%
    network_fee_krw: float = 5000.0,     # XRP or TRX transfer fee
) -> HedgingSimulationResult:
    """Simulate delta-neutral Kimchi Premium arbitrage.

    Parameters
    ----------
    capital_krw : float
        Total capital committed to the arbitrage trade in KRW.
    entry_kp_pct : float
        Kimchi premium % at entry (e.g. 1.0%).
    exit_kp_pct : float
        Kimchi premium % at exit (e.g. 4.0%).
    holding_days : int
        Holding duration in days.
    daily_funding_rate : float
        Net daily funding rate earned (positive = long pays short, so short receives).
    binance_fee : float
        Fee rate per trade leg on Binance.
    upbit_fee : float
        Fee rate per trade leg on Upbit.
    network_fee_krw : float
        Network withdrawal fee in KRW.
    """
    if capital_krw <= 0:
        raise ValueError("Capital must be strictly positive.")
    if holding_days <= 0:
        raise ValueError("Holding days must be a strictly positive integer.")
    if binance_fee < 0 or upbit_fee < 0:
        raise ValueError("Fee rates must be non-negative.")

    # 1. Spread profit: delta in Kimchi Premium on spot capital
    spread_delta = (exit_kp_pct - entry_kp_pct) / 100.0
    gross_spread_krw = capital_krw * spread_delta

    # 2. Funding fee income from 1x short hedge
    funding_yield = daily_funding_rate * holding_days
    funding_income_krw = capital_krw * funding_yield

    # 3. Transaction fees:
    # Binance spot buy + Binance futures short entry + Binance futures exit + Upbit spot sell
    total_fee_rate = (binance_fee * 3.0) + upbit_fee
    exchange_fees_krw = capital_krw * total_fee_rate
    total_fees_krw = exchange_fees_krw + network_fee_krw

    # 4. Net Profit and APR
    net_profit_krw = gross_spread_krw + funding_income_krw - total_fees_krw
    net_return_pct = (net_profit_krw / capital_krw) * 100.0
    annualized_apr = net_return_pct * (365.0 / holding_days)

    # Break-even spread required: fees / capital
    break_even_spread_pct = ((total_fees_krw - funding_income_krw) / capital_krw) * 100.0

    return HedgingSimulationResult(
        capital_krw=capital_krw,
        entry_kp_pct=entry_kp_pct,
        exit_kp_pct=exit_kp_pct,
        holding_days=holding_days,
        gross_spread_profit_krw=round(gross_spread_krw, 0),
        funding_income_krw=round(funding_income_krw, 0),
        total_fees_krw=round(total_fees_krw, 0),
        net_profit_krw=round(net_profit_krw, 0),
        net_return_pct=round(net_return_pct, 3),
        annualized_apr_pct=round(annualized_apr, 2),
        break_even_spread_pct=round(break_even_spread_pct, 3),
        is_profitable=net_profit_krw > 0,
    )


def main():
    parser = argparse.ArgumentParser(description="Kimchi Premium Arbitrage & Hedging Simulator.")
    parser.add_argument("--capital", type=float, default=50_000_000.0, help="Capital in KRW.")
    parser.add_argument("--entry-kp", type=float, default=1.0, help="Entry KP %.")
    parser.add_argument("--exit-kp", type=float, default=4.0, help="Exit KP %.")
    parser.add_argument("--days", type=int, default=30, help="Holding duration (days).")
    parser.add_argument("--daily-funding", type=float, default=0.0003, help="Daily funding rate (0.0003 = 0.03%).")
    args = parser.parse_args()

    res = simulate_kimchi_hedging(
        capital_krw=args.capital,
        entry_kp_pct=args.entry_kp,
        exit_kp_pct=args.exit_kp,
        holding_days=args.days,
        daily_funding_rate=args.daily_funding,
    )

    print("================ Kimchi Premium Hedging Simulation ================")
    print(f"  Capital Committed   : ₩{res.capital_krw:,.0f}")
    print(f"  KP Entry / Exit     : {res.entry_kp_pct:.1f}% -> {res.exit_kp_pct:.1f}% (Duration: {res.holding_days} days)")
    print(f"  Gross Spread Profit : ₩{res.gross_spread_profit_krw:,.0f}")
    print(f"  Funding Fee Income  : ₩{res.funding_income_krw:,.0f}")
    print(f"  Total Fees & Drag   : -₩{res.total_fees_krw:,.0f}")
    print(f"  Net Profit          : ₩{res.net_profit_krw:,.0f}")
    print(f"  Net Return          : {res.net_return_pct:+.2f}%")
    print(f"  Annualized APR      : {res.annualized_apr_pct:+.2f}%")
    print(f"  Break-Even Spread   : {res.break_even_spread_pct:.3f}%")
    print("====================================================================")


if __name__ == "__main__":
    main()
