"""Cryptocurrency After-Tax Net Return & Capital Gains Tax Simulator.

Simulates the Korean Virtual Asset Income Tax (and global capital gains tax):
- Netting of realized gains and losses across all traded crypto assets (손익 통산).
- Deducts annual statutory basic tax allowance (default: KRW 2,500,000, customizable).
- Applies 22% separate income tax (20% national tax + 2% local income tax).
- Evaluates tax drag and after-tax CAGR for periodic rebalanced portfolios.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd


def compute_crypto_tax_impact(
    realized_profits: list[float],
    annual_allowance_krw: float = 2500000.0,
    tax_rate: float = 0.22,
    usdt_krw_rate: float = 1350.0,
    initial_capital_krw: float = 10000000.0,
) -> dict[str, object]:
    """
    Calculate annual crypto capital gains tax, net of loss deduction.

    Parameters:
    - realized_profits: List of realized profits/losses in KRW (or converted to KRW)
    - annual_allowance_krw: Basic deduction threshold (e.g. 2,500,000 KRW)
    - tax_rate: Effective tax rate including local tax (e.g. 0.22 for 22%)
    - usdt_krw_rate: Exchange rate applied
    - initial_capital_krw: Starting capital for return calculation
    """
    if tax_rate < 0 or tax_rate > 1:
        raise ValueError("Tax rate must be between 0.0 and 1.0 (e.g. 0.22).")
    if initial_capital_krw <= 0:
        raise ValueError("Initial capital must be strictly positive.")

    profits_arr = np.array(realized_profits, dtype=float)
    gains = float(profits_arr[profits_arr > 0].sum()) if len(profits_arr[profits_arr > 0]) > 0 else 0.0
    losses = float(abs(profits_arr[profits_arr < 0].sum())) if len(profits_arr[profits_arr < 0]) > 0 else 0.0

    net_realized_profit = float(profits_arr.sum())

    # Tax Base after Loss Offsetting & Basic Allowance
    taxable_base = max(0.0, net_realized_profit - annual_allowance_krw)
    estimated_tax = taxable_base * tax_rate
    after_tax_profit = net_realized_profit - estimated_tax

    pre_tax_return = (net_realized_profit / initial_capital_krw) * 100
    after_tax_return = (after_tax_profit / initial_capital_krw) * 100
    tax_drag_pct = pre_tax_return - after_tax_return

    return {
        "initial_capital_krw": round(initial_capital_krw, 2),
        "gross_realized_gains": round(gains, 2),
        "gross_realized_losses": round(losses, 2),
        "net_realized_profit": round(net_realized_profit, 2),
        "annual_allowance_krw": round(annual_allowance_krw, 2),
        "taxable_base": round(taxable_base, 2),
        "effective_tax_rate_pct": round(tax_rate * 100, 1),
        "estimated_tax_krw": round(estimated_tax, 2),
        "after_tax_profit_krw": round(after_tax_profit, 2),
        "pre_tax_return_pct": round(pre_tax_return, 2),
        "after_tax_return_pct": round(after_tax_return, 2),
        "tax_drag_pct": round(tax_drag_pct, 2),
        "is_taxable": taxable_base > 0,
    }


def print_tax_report(res: dict[str, object]):
    """Print terminal report of capital gains tax simulation."""
    print("================================================================================")
    print("             KOREA VIRTUAL ASSET CAPITAL GAINS TAX SIMULATION                   ")
    print("================================================================================")
    print(f"초기 원금 (Initial Capital)        : ₩{res['initial_capital_krw']:,.0f}")
    print(f"총 실현 이익 (Gross Gains)         : ₩{res['gross_realized_gains']:,.0f}")
    print(f"총 실현 손실 (Gross Losses)        : -₩{res['gross_realized_losses']:,.0f}")
    print(f"손익 통산 순수익 (Net Profit)      : ₩{res['net_realized_profit']:,.0f}")
    print("--------------------------------------------------------------------------------")
    print(f"기본 공제액 (Annual Allowance)     : ₩{res['annual_allowance_krw']:,.0f}")
    print(f"과세 표준 (Taxable Base)           : ₩{res['taxable_base']:,.0f}")
    print(f"세율 (Effective Tax Rate)          : {res['effective_tax_rate_pct']:.1f}% (국세 20% + 지방세 2%)")
    print(f"예상 납부 세액 (Estimated Tax)     : ₩{res['estimated_tax_krw']:,.0f}")
    print("--------------------------------------------------------------------------------")
    print(f"세전 순수익률 (Pre-Tax Return)     : {res['pre_tax_return_pct']:+.2f}%")
    print(f"세후 순수익률 (After-Tax Return)   : {res['after_tax_return_pct']:+.2f}%")
    print(f"세금 잠식률 (Tax Drag)             : -{res['tax_drag_pct']:.2f}%p")
    print(f"최종 세후 순이익 (After-Tax Profit): ₩{res['after_tax_profit_krw']:,.0f}")
    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Crypto Capital Gains Tax Simulator")
    parser.add_argument("--profit", type=float, default=12000000.0, help="Annual net realized profit in KRW")
    parser.add_argument("--capital", type=float, default=50000000.0, help="Initial capital in KRW")
    parser.add_argument("--allowance", type=float, default=2500000.0, help="Basic allowance in KRW (default: 2,500,000)")
    parser.add_argument("--tax-rate", type=float, default=0.22, help="Effective tax rate (default: 0.22)")
    parser.add_argument("--export-json", type=str, default="", help="Path to export JSON metrics")
    args = parser.parse_args()

    # Representative sample profits: mixed gains and losses
    sample_trades = [
        args.profit * 0.70,
        args.profit * 0.50,
        -args.profit * 0.20,
    ]

    res = compute_crypto_tax_impact(
        realized_profits=sample_trades,
        annual_allowance_krw=args.allowance,
        tax_rate=args.tax_rate,
        initial_capital_krw=args.capital,
    )

    print_tax_report(res)

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[+] Tax simulation metrics exported to: {out_path}")


if __name__ == "__main__":
    main()
