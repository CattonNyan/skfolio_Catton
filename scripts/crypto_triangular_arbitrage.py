"""Crypto Triangular Arbitrage Scanner and Profitability Analyzer.

Identifies triangular cross-currency pricing discrepancies across 3-legged trading loops:
e.g. Quote (USDT/KRW) -> Asset A -> Asset B -> Quote
Calculates:
- Implied theoretical cross-rate vs market cross-rate
- Fee drag across 3 execution legs
- Net profit percentage after maker/taker fees and slippage buffer
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
class ArbitrageOpportunity:
    """Represents an identified triangular arbitrage opportunity."""
    cycle: str  # e.g. "USDT -> BTC -> ETH -> USDT"
    gross_return_pct: float
    fee_drag_pct: float
    net_return_pct: float
    is_profitable: bool
    legs: list[str]


def calculate_triangular_arbitrage(
    p_a_quote: float,   # Price of A in Quote (e.g. BTC/USDT)
    p_b_quote: float,   # Price of B in Quote (e.g. ETH/USDT)
    p_b_a: float,       # Price of B in A (e.g. ETH/BTC)
    fee_rate: float = 0.001,  # 0.1% fee per leg
    slippage: float = 0.0005, # 0.05% slippage buffer
    base_quote: str = "USDT",
    asset_a: str = "BTC",
    asset_b: str = "ETH",
) -> list[ArbitrageOpportunity]:
    """Calculate forward and reverse triangular arbitrage between Quote, A, and B.

    Forward cycle:
      1. Buy A with Quote: 1 Quote -> (1 / p_a_quote) of A
      2. Sell A for B (or buy B with A): (1 / p_a_quote) / p_b_a of B
      3. Sell B for Quote: ((1 / p_a_quote) / p_b_a) * p_b_quote of Quote
      Product = p_b_quote / (p_a_quote * p_b_a)

    Reverse cycle:
      1. Buy B with Quote: 1 / p_b_quote of B
      2. Buy A with B: (1 / p_b_quote) * p_b_a of A
      3. Sell A for Quote: ((1 / p_b_quote) * p_b_a) * p_a_quote of Quote
      Product = (p_a_quote * p_b_a) / p_b_quote
    """
    if p_a_quote <= 0 or p_b_quote <= 0 or p_b_a <= 0:
        raise ValueError("Prices must be strictly positive.")
    if fee_rate < 0 or slippage < 0:
        raise ValueError("Fees and slippage must be non-negative.")

    total_cost_factor = ((1.0 - fee_rate) ** 3) * ((1.0 - slippage) ** 3)
    fee_drag = (1.0 - total_cost_factor) * 100.0

    # 1. Forward Cycle
    fwd_gross_mult = p_b_quote / (p_a_quote * p_b_a)
    fwd_gross_pct = (fwd_gross_mult - 1.0) * 100.0
    fwd_net_mult = fwd_gross_mult * total_cost_factor
    fwd_net_pct = (fwd_net_mult - 1.0) * 100.0

    fwd_opp = ArbitrageOpportunity(
        cycle=f"{base_quote} -> {asset_a} -> {asset_b} -> {base_quote}",
        gross_return_pct=round(fwd_gross_pct, 4),
        fee_drag_pct=round(fee_drag, 4),
        net_return_pct=round(fwd_net_pct, 4),
        is_profitable=fwd_net_pct > 0.0,
        legs=[
            f"Buy {asset_a} with {base_quote} @ {p_a_quote}",
            f"Convert {asset_a} to {asset_b} @ {p_b_a}",
            f"Sell {asset_b} for {base_quote} @ {p_b_quote}",
        ],
    )

    # 2. Reverse Cycle
    rev_gross_mult = (p_a_quote * p_b_a) / p_b_quote
    rev_gross_pct = (rev_gross_mult - 1.0) * 100.0
    rev_net_mult = rev_gross_mult * total_cost_factor
    rev_net_pct = (rev_net_mult - 1.0) * 100.0

    rev_opp = ArbitrageOpportunity(
        cycle=f"{base_quote} -> {asset_b} -> {asset_a} -> {base_quote}",
        gross_return_pct=round(rev_gross_pct, 4),
        fee_drag_pct=round(fee_drag, 4),
        net_return_pct=round(rev_net_pct, 4),
        is_profitable=rev_net_pct > 0.0,
        legs=[
            f"Buy {asset_b} with {base_quote} @ {p_b_quote}",
            f"Convert {asset_b} to {asset_a} @ {p_b_a}",
            f"Sell {asset_a} for {base_quote} @ {p_a_quote}",
        ],
    )

    return [fwd_opp, rev_opp]


def scan_triangular_pairs(
    prices_dict: dict[str, float],
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
) -> list[ArbitrageOpportunity]:
    """Scan a dictionary of current prices for triangular arbitrage opportunities.

    Dict format expected:
    - 'BTC/USDT': 60000.0
    - 'ETH/USDT': 3000.0
    - 'ETH/BTC': 0.051
    """
    opportunities: list[ArbitrageOpportunity] = []
    # Identify cross pairs with '/'
    cross_pairs = [k for k in prices_dict if "/" in k and not k.endswith("/USDT") and not k.endswith("/KRW")]

    for cross in cross_pairs:
        base, quote = cross.split("/")
        p_cross = prices_dict[cross]

        for settlement in ["USDT", "KRW"]:
            pair_base = f"{base}/{settlement}"
            pair_quote = f"{quote}/{settlement}"
            if pair_base in prices_dict and pair_quote in prices_dict:
                # p_b_quote is pair_base (base/settlement), p_a_quote is pair_quote (quote/settlement)
                p_base = prices_dict[pair_base]
                p_quote = prices_dict[pair_quote]
                opps = calculate_triangular_arbitrage(
                    p_a_quote=p_quote,
                    p_b_quote=p_base,
                    p_b_a=p_cross,
                    fee_rate=fee_rate,
                    slippage=slippage,
                    base_quote=settlement,
                    asset_a=quote,
                    asset_b=base,
                )
                opportunities.extend(opps)

    return opportunities


def main():
    parser = argparse.ArgumentParser(description="Triangular Arbitrage Scanner.")
    parser.add_argument("--p-btc-usdt", type=float, default=65000.0, help="BTC/USDT price")
    parser.add_argument("--p-eth-usdt", type=float, default=3500.0, help="ETH/USDT price")
    parser.add_argument("--p-eth-btc", type=float, default=0.0545, help="ETH/BTC price")
    parser.add_argument("--fee", type=float, default=0.00075, help="Fee per leg (default 0.075%)")
    args = parser.parse_args()

    opps = calculate_triangular_arbitrage(
        p_a_quote=args.p_btc_usdt,
        p_b_quote=args.p_eth_usdt,
        p_b_a=args.p_eth_btc,
        fee_rate=args.fee,
        base_quote="USDT",
        asset_a="BTC",
        asset_b="ETH",
    )

    print("================ Triangular Arbitrage Scanner ================")
    for o in opps:
        status = "[PROFITABLE]" if o.is_profitable else "[NO PROFIT]"
        print(f"{status} Cycle: {o.cycle}")
        print(f"  Gross Return: {o.gross_return_pct:+.3f}% | Fee Drag: -{o.fee_drag_pct:.3f}% | Net: {o.net_return_pct:+.3f}%")
        for idx, leg in enumerate(o.legs, 1):
            print(f"    Leg {idx}: {leg}")


if __name__ == "__main__":
    main()
