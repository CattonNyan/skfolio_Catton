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
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

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
    capital: float | None = None
    profit_dollars: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert arbitrage opportunity to serializable dictionary."""
        d: dict[str, Any] = {
            "cycle": self.cycle,
            "gross_return_pct": self.gross_return_pct,
            "fee_drag_pct": self.fee_drag_pct,
            "net_return_pct": self.net_return_pct,
            "is_profitable": self.is_profitable,
            "legs": list(self.legs),
        }
        if self.capital is not None:
            d["capital"] = round(float(self.capital), 2)
            d["profit_dollars"] = round(float(self.profit_dollars), 2)
        return d


def calculate_triangular_arbitrage(
    p_a_quote: float,   # Price of A in Quote (e.g. BTC/USDT)
    p_b_quote: float,   # Price of B in Quote (e.g. ETH/USDT)
    p_b_a: float,       # Price of B in A (e.g. ETH/BTC)
    fee_rate: float = 0.001,  # 0.1% fee per leg
    slippage: float = 0.0005, # 0.05% slippage buffer
    base_quote: str = "USDT",
    asset_a: str = "BTC",
    asset_b: str = "ETH",
    capital: float | None = None,
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
    if capital is not None and capital <= 0:
        raise ValueError("Capital must be strictly positive.")

    total_cost_factor = ((1.0 - fee_rate) ** 3) * ((1.0 - slippage) ** 3)
    fee_drag = (1.0 - total_cost_factor) * 100.0

    # 1. Forward Cycle
    fwd_gross_mult = p_b_quote / (p_a_quote * p_b_a)
    fwd_gross_pct = (fwd_gross_mult - 1.0) * 100.0
    fwd_net_mult = fwd_gross_mult * total_cost_factor
    fwd_net_pct = (fwd_net_mult - 1.0) * 100.0
    fwd_profit = round(float(capital) * (fwd_net_pct / 100.0), 2) if capital is not None else 0.0

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
        capital=capital,
        profit_dollars=fwd_profit,
    )

    # 2. Reverse Cycle
    rev_gross_mult = (p_a_quote * p_b_a) / p_b_quote
    rev_gross_pct = (rev_gross_mult - 1.0) * 100.0
    rev_net_mult = rev_gross_mult * total_cost_factor
    rev_net_pct = (rev_net_mult - 1.0) * 100.0
    rev_profit = round(float(capital) * (rev_net_pct / 100.0), 2) if capital is not None else 0.0

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
        capital=capital,
        profit_dollars=rev_profit,
    )

    return [fwd_opp, rev_opp]


def filter_profitable_opportunities(
    opportunities: list[ArbitrageOpportunity],
    min_net_pct: float = 0.0,
) -> list[ArbitrageOpportunity]:
    """Filter arbitrage opportunities satisfying minimum net profitability hurdle."""
    return [o for o in opportunities if o.net_return_pct >= min_net_pct]


def scan_triangular_pairs(
    prices_dict: dict[str, float],
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
    capital: float | None = None,
    min_profit_pct: float | None = None,
) -> list[ArbitrageOpportunity]:
    """Scan a dictionary of current prices for triangular arbitrage opportunities.

    Dict format expected:
    - 'BTC/USDT': 60000.0
    - 'ETH/USDT': 3000.0
    - 'ETH/BTC': 0.051
    """
    opportunities: list[ArbitrageOpportunity] = []
    # Normalize pair delimiters to standard BASE/QUOTE format
    norm_prices: dict[str, float] = {}
    for k, v in prices_dict.items():
        clean_k = str(k).strip().upper().replace("-", "/").replace("_", "/")
        norm_prices[clean_k] = float(v)

    # Identify cross pairs with '/'
    cross_pairs = [k for k in norm_prices if "/" in k and not k.endswith("/USDT") and not k.endswith("/KRW")]

    for cross in cross_pairs:
        parts = cross.split("/")
        if len(parts) != 2:
            continue
        base, quote = parts
        p_cross = norm_prices[cross]

        for settlement in ["USDT", "KRW"]:
            pair_base = f"{base}/{settlement}"
            pair_quote = f"{quote}/{settlement}"
            if pair_base in norm_prices and pair_quote in norm_prices:
                # p_b_quote is pair_base (base/settlement), p_a_quote is pair_quote (quote/settlement)
                p_base = norm_prices[pair_base]
                p_quote = norm_prices[pair_quote]
                opps = calculate_triangular_arbitrage(
                    p_a_quote=p_quote,
                    p_b_quote=p_base,
                    p_b_a=p_cross,
                    fee_rate=fee_rate,
                    slippage=slippage,
                    base_quote=settlement,
                    asset_a=quote,
                    asset_b=base,
                    capital=capital,
                )
                opportunities.extend(opps)

    if min_profit_pct is not None:
        opportunities = filter_profitable_opportunities(opportunities, min_net_pct=min_profit_pct)

    return opportunities


def export_triangular_arbitrage_csv(
    opportunities: list[ArbitrageOpportunity | dict[str, Any]],
    filepath: str | Path | None = None,
) -> str:
    """Export identified triangular arbitrage opportunities to CSV string or file."""
    rows = []
    has_capital = any(
        (o.capital is not None if isinstance(o, ArbitrageOpportunity) else "capital" in o)
        for o in opportunities
    )
    for o in opportunities:
        d = o.to_dict() if isinstance(o, ArbitrageOpportunity) else dict(o)
        legs = d.get("legs", [])
        row = {
            "cycle": d.get("cycle", ""),
            "gross_return_pct": d.get("gross_return_pct", 0.0),
            "fee_drag_pct": d.get("fee_drag_pct", 0.0),
            "net_return_pct": d.get("net_return_pct", 0.0),
            "is_profitable": d.get("is_profitable", False),
            "legs": " -> ".join(legs) if isinstance(legs, list) else str(legs),
        }
        if has_capital:
            row["capital"] = d.get("capital", "")
            row["profit_dollars"] = d.get("profit_dollars", "")
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_str = df.to_csv(index=False)
    if filepath:
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(csv_str, encoding="utf-8")
    return csv_str


def main():
    parser = argparse.ArgumentParser(description="Triangular Arbitrage Scanner.")
    parser.add_argument("--p-btc-usdt", type=float, default=65000.0, help="BTC/USDT price")
    parser.add_argument("--p-eth-usdt", type=float, default=3500.0, help="ETH/USDT price")
    parser.add_argument("--p-eth-btc", type=float, default=0.0545, help="ETH/BTC price")
    parser.add_argument("--fee", type=float, default=0.00075, help="Fee per leg (default 0.075%%)")
    parser.add_argument("--capital", type=float, default=None, help="Capital allocated in quote currency (USD/USDT)")
    parser.add_argument("--min-profit-pct", type=float, default=None, help="Minimum net profit percentage hurdle rate")
    parser.add_argument("--profitable-only", action="store_true", help="Display only profitable arbitrage opportunities")
    parser.add_argument("--export-json", type=str, default=None, help="Export identified opportunities to JSON file")
    parser.add_argument("--export-csv", type=str, default=None, help="Export identified opportunities to CSV file")
    args = parser.parse_args()

    opps = calculate_triangular_arbitrage(
        p_a_quote=args.p_btc_usdt,
        p_b_quote=args.p_eth_usdt,
        p_b_a=args.p_eth_btc,
        fee_rate=args.fee,
        base_quote="USDT",
        asset_a="BTC",
        asset_b="ETH",
        capital=args.capital,
    )

    if args.min_profit_pct is not None:
        opps = filter_profitable_opportunities(opps, min_net_pct=args.min_profit_pct)
    elif args.profitable_only:
        opps = filter_profitable_opportunities(opps, min_net_pct=0.0)

    print("================ Triangular Arbitrage Scanner ================")
    if args.capital:
        print(f"  Allocated Capital  : ${args.capital:,.2f}")
    for o in opps:
        status = "[PROFITABLE]" if o.is_profitable else "[NO PROFIT]"
        dollar_str = f" | Profit: ${o.profit_dollars:+,.2f}" if o.capital is not None else ""
        print(f"{status} Cycle: {o.cycle}")
        print(f"  Gross Return: {o.gross_return_pct:+.3f}% | Fee Drag: -{o.fee_drag_pct:.3f}% | Net: {o.net_return_pct:+.3f}%{dollar_str}")
        for idx, leg in enumerate(o.legs, 1):
            print(f"    Leg {idx}: {leg}")

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([o.to_dict() for o in opps], f, indent=2, ensure_ascii=False)
        print(f"[+] Opportunities exported to {out_path}")

    if args.export_csv:
        export_triangular_arbitrage_csv(opps, args.export_csv)
        print(f"[+] Opportunities CSV exported to {args.export_csv}")


if __name__ == "__main__":
    main()
