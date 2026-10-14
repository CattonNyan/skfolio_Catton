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


@dataclass
class TriangularCycleCapacityResult:
    """Represents triangular arbitrage order book capacity and optimal sizing analytics."""
    cycle: str
    bottleneck_capacity_usd: float
    optimal_trade_size_usd: float
    max_dollar_profit: float
    optimal_net_return_pct: float
    base_net_return_pct: float
    leg_capacities_usd: dict[str, float]
    breakeven_capacity_usd: float
    is_actionable: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert capacity analysis result to serializable dictionary."""
        return {
            "cycle": self.cycle,
            "bottleneck_capacity_usd": round(float(self.bottleneck_capacity_usd), 2),
            "optimal_trade_size_usd": round(float(self.optimal_trade_size_usd), 2),
            "max_dollar_profit": round(float(self.max_dollar_profit), 2),
            "optimal_net_return_pct": round(float(self.optimal_net_return_pct), 4),
            "base_net_return_pct": round(float(self.base_net_return_pct), 4),
            "leg_capacities_usd": {k: round(float(v), 2) for k, v in self.leg_capacities_usd.items()},
            "breakeven_capacity_usd": round(float(self.breakeven_capacity_usd), 2),
            "is_actionable": self.is_actionable,
        }


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


def estimate_triangular_cycle_capacity(
    p_a_quote: float,
    p_b_quote: float,
    p_b_a: float,
    depth_a_quote_usd: float,
    depth_b_quote_usd: float,
    depth_b_a_usd: float,
    fee_rate: float = 0.00075,
    base_slippage: float = 0.0002,
    slippage_impact_factor: float = 0.005,
    base_quote: str = "USDT",
    asset_a: str = "BTC",
    asset_b: str = "ETH",
) -> list[TriangularCycleCapacityResult]:
    """Estimate liquidity capacity, bottleneck leg, and optimal trade sizing for triangular arbitrage.

    Models dynamic market impact across the 3 execution legs as trade size increases:
      effective_slippage(S) = base_slippage + slippage_impact_factor * (S / bottleneck_capacity)
    and determines optimal trade size S* that maximizes net dollar profit.
    """
    if p_a_quote <= 0 or p_b_quote <= 0 or p_b_a <= 0:
        raise ValueError("Prices must be strictly positive.")
    if depth_a_quote_usd <= 0 or depth_b_quote_usd <= 0 or depth_b_a_usd <= 0:
        raise ValueError("Order book depths must be strictly positive.")
    if fee_rate < 0 or base_slippage < 0 or slippage_impact_factor < 0:
        raise ValueError("Fees and slippage parameters must be non-negative.")

    bottleneck = min(depth_a_quote_usd, depth_b_quote_usd, depth_b_a_usd)

    # 1. Forward Cycle: Quote -> A -> B -> Quote
    # Gross multiplier: p_b_quote / (p_a_quote * p_b_a)
    fwd_gross_mult = p_b_quote / (p_a_quote * p_b_a)
    # 2. Reverse Cycle: Quote -> B -> A -> Quote
    # Gross multiplier: (p_a_quote * p_b_a) / p_b_quote
    rev_gross_mult = (p_a_quote * p_b_a) / p_b_quote

    results: list[TriangularCycleCapacityResult] = []
    cycles = [
        (
            f"{base_quote} -> {asset_a} -> {asset_b} -> {base_quote}",
            fwd_gross_mult,
            {
                f"Leg 1 ({asset_a}/{base_quote})": depth_a_quote_usd,
                f"Leg 2 ({asset_b}/{asset_a})": depth_b_a_usd,
                f"Leg 3 ({asset_b}/{base_quote})": depth_b_quote_usd,
            },
        ),
        (
            f"{base_quote} -> {asset_b} -> {asset_a} -> {base_quote}",
            rev_gross_mult,
            {
                f"Leg 1 ({asset_b}/{base_quote})": depth_b_quote_usd,
                f"Leg 2 ({asset_b}/{asset_a})": depth_b_a_usd,
                f"Leg 3 ({asset_a}/{base_quote})": depth_a_quote_usd,
            },
        ),
    ]

    for cycle_name, gross_mult, leg_caps in cycles:
        base_cost_factor = ((1.0 - fee_rate) ** 3) * ((1.0 - base_slippage) ** 3)
        base_net_pct = (gross_mult * base_cost_factor - 1.0) * 100.0

        n_steps = 100
        best_size = 0.0
        max_profit = 0.0
        best_net_pct = 0.0
        be_capacity = 0.0

        for i in range(1, n_steps + 1):
            s = (i / n_steps) * bottleneck
            slip_s = base_slippage + slippage_impact_factor * (s / bottleneck)
            cost_s = ((1.0 - fee_rate) ** 3) * ((1.0 - slip_s) ** 3)
            net_pct_s = (gross_mult * cost_s - 1.0) * 100.0
            profit_s = s * (net_pct_s / 100.0)

            if net_pct_s > 0:
                be_capacity = s
                if profit_s > max_profit:
                    max_profit = profit_s
                    best_size = s
                    best_net_pct = net_pct_s

        res = TriangularCycleCapacityResult(
            cycle=cycle_name,
            bottleneck_capacity_usd=round(bottleneck, 2),
            optimal_trade_size_usd=round(best_size, 2),
            max_dollar_profit=round(max_profit, 2),
            optimal_net_return_pct=round(best_net_pct, 4),
            base_net_return_pct=round(base_net_pct, 4),
            leg_capacities_usd=leg_caps,
            breakeven_capacity_usd=round(be_capacity, 2),
            is_actionable=max_profit > 0.0,
        )
        results.append(res)

    return results


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


def export_triangular_capacity_json(
    results: list[TriangularCycleCapacityResult],
    filepath: str | Path,
) -> Path:
    """Export triangular arbitrage capacity analysis to a JSON file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
    return out_path


def export_triangular_capacity_csv(
    results: list[TriangularCycleCapacityResult],
    filepath: str | Path,
) -> Path:
    """Export triangular arbitrage capacity analysis to a CSV file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        d = r.to_dict()
        row = {
            "cycle": d["cycle"],
            "bottleneck_capacity_usd": d["bottleneck_capacity_usd"],
            "optimal_trade_size_usd": d["optimal_trade_size_usd"],
            "max_dollar_profit": d["max_dollar_profit"],
            "optimal_net_return_pct": d["optimal_net_return_pct"],
            "base_net_return_pct": d["base_net_return_pct"],
            "breakeven_capacity_usd": d["breakeven_capacity_usd"],
            "is_actionable": d["is_actionable"],
            "leg_capacities": "; ".join(f"{k}: ${v:,.2f}" for k, v in d["leg_capacities_usd"].items()),
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Triangular Arbitrage Scanner & Capacity Estimator.")
    parser.add_argument("--p-btc-usdt", type=float, default=65000.0, help="BTC/USDT price")
    parser.add_argument("--p-eth-usdt", type=float, default=3500.0, help="ETH/USDT price")
    parser.add_argument("--p-eth-btc", type=float, default=0.0545, help="ETH/BTC price")
    parser.add_argument("--fee", type=float, default=0.00075, help="Fee per leg (default 0.075%%)")
    parser.add_argument("--capital", type=float, default=None, help="Capital allocated in quote currency (USD/USDT)")
    parser.add_argument("--min-profit-pct", type=float, default=None, help="Minimum net profit percentage hurdle rate")
    parser.add_argument("--profitable-only", action="store_true", help="Display only profitable arbitrage opportunities")
    parser.add_argument("--estimate-capacity", action="store_true", help="Estimate liquidity bottleneck and optimal trade sizing")
    parser.add_argument("--depth-a", type=float, default=50000.0, help="Order book depth for Pair A/Quote in USD")
    parser.add_argument("--depth-b", type=float, default=40000.0, help="Order book depth for Pair B/Quote in USD")
    parser.add_argument("--depth-cross", type=float, default=30000.0, help="Order book depth for Cross Pair B/A in USD")
    parser.add_argument("--export-capacity-json", type=str, default=None, help="Export capacity analysis to JSON")
    parser.add_argument("--export-capacity-csv", type=str, default=None, help="Export capacity analysis to CSV")
    parser.add_argument("--export-json", type=str, default=None, help="Export identified opportunities to JSON file")
    parser.add_argument("--export-csv", type=str, default=None, help="Export identified opportunities to CSV file")
    args = parser.parse_args()

    if args.estimate_capacity:
        cap_results = estimate_triangular_cycle_capacity(
            p_a_quote=args.p_btc_usdt,
            p_b_quote=args.p_eth_usdt,
            p_b_a=args.p_eth_btc,
            depth_a_quote_usd=args.depth_a,
            depth_b_quote_usd=args.depth_b,
            depth_b_a_usd=args.depth_cross,
            fee_rate=args.fee,
        )
        print("======== Triangular Arbitrage Liquidity Capacity & Sizing ========")
        for cr in cap_results:
            status = "[ACTIONABLE]" if cr.is_actionable else "[UNPROFITABLE]"
            print(f"{status} Cycle: {cr.cycle}")
            print(f"  Bottleneck Depth   : ${cr.bottleneck_capacity_usd:,.2f}")
            print(f"  Optimal Trade Size : ${cr.optimal_trade_size_usd:,.2f}")
            print(f"  Max Dollar Profit  : ${cr.max_dollar_profit:+,.2f} ({cr.optimal_net_return_pct:+.3f}%)")
            print(f"  Breakeven Capacity : ${cr.breakeven_capacity_usd:,.2f}")
            print("  Leg Capacities:")
            for leg_k, leg_v in cr.leg_capacities_usd.items():
                print(f"    - {leg_k}: ${leg_v:,.2f}")
        print("==================================================================")

        if args.export_capacity_json:
            export_triangular_capacity_json(cap_results, args.export_capacity_json)
            print(f"[+] Capacity analysis exported to {args.export_capacity_json}")

        if args.export_capacity_csv:
            export_triangular_capacity_csv(cap_results, args.export_capacity_csv)
            print(f"[+] Capacity CSV exported to {args.export_capacity_csv}")
        return

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
