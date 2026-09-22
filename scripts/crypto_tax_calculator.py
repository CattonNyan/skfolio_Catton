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
    carried_forward_loss_krw: float = 0.0,
) -> dict[str, object]:
    """
    Calculate annual crypto capital gains tax, net of loss deduction and carried-forward losses.

    Parameters:
    - realized_profits: List of realized profits/losses in KRW (or converted to KRW)
    - annual_allowance_krw: Basic deduction threshold (e.g. 2,500,000 KRW)
    - tax_rate: Effective tax rate including local tax (e.g. 0.22 for 22%)
    - usdt_krw_rate: Exchange rate applied
    - initial_capital_krw: Starting capital for return calculation
    - carried_forward_loss_krw: Prior-year accumulated net loss carried forward to offset gains (이월결손금)
    """
    if isinstance(annual_allowance_krw, bool) or not np.isfinite(annual_allowance_krw) or annual_allowance_krw < 0:
        raise ValueError("Annual allowance must be a finite non-negative amount.")
    if isinstance(tax_rate, bool) or not np.isfinite(tax_rate) or tax_rate < 0 or tax_rate > 1:
        raise ValueError("Tax rate must be between 0.0 and 1.0 (e.g. 0.22).")
    if isinstance(usdt_krw_rate, bool) or not np.isfinite(usdt_krw_rate) or usdt_krw_rate <= 0:
        raise ValueError("Exchange rate must be a finite strictly positive number.")
    if isinstance(initial_capital_krw, bool) or not np.isfinite(initial_capital_krw) or initial_capital_krw <= 0:
        raise ValueError("Initial capital must be strictly positive.")
    if isinstance(carried_forward_loss_krw, bool) or not np.isfinite(carried_forward_loss_krw) or carried_forward_loss_krw < 0:
        raise ValueError("Carried-forward loss must be a finite non-negative amount.")

    if any(isinstance(p, bool) for p in realized_profits):
        raise ValueError("Realized profits must not contain boolean values.")
    profits_arr = np.array(realized_profits, dtype=float)
    if not np.all(np.isfinite(profits_arr)):
        raise ValueError("Realized profits must contain only finite values.")
    gains = float(profits_arr[profits_arr > 0].sum()) if len(profits_arr[profits_arr > 0]) > 0 else 0.0
    losses = float(abs(profits_arr[profits_arr < 0].sum())) if len(profits_arr[profits_arr < 0]) > 0 else 0.0

    net_realized_profit = float(profits_arr.sum())

    # Offset prior carried-forward losses before basic allowance
    carried_loss_applied = 0.0
    remaining_carried_loss = carried_forward_loss_krw
    if net_realized_profit > 0 and carried_forward_loss_krw > 0:
        carried_loss_applied = min(net_realized_profit, carried_forward_loss_krw)
        adjusted_profit = net_realized_profit - carried_loss_applied
        remaining_carried_loss = carried_forward_loss_krw - carried_loss_applied
    elif net_realized_profit < 0:
        adjusted_profit = net_realized_profit
        remaining_carried_loss = carried_forward_loss_krw + abs(net_realized_profit)
    else:
        adjusted_profit = net_realized_profit

    # Tax Base after Loss Offsetting, Carried Losses, & Basic Allowance
    taxable_base = max(0.0, adjusted_profit - annual_allowance_krw)
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
        "carried_forward_loss_krw": round(carried_forward_loss_krw, 2),
        "carried_loss_applied_krw": round(carried_loss_applied, 2),
        "remaining_carried_loss_krw": round(remaining_carried_loss, 2),
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


def calculate_tax_loss_harvesting_target(
    current_net_realized_profit_krw: float,
    annual_allowance_krw: float = 2500000.0,
    tax_rate: float = 0.22,
) -> dict[str, object]:
    """
    Calculate required loss realization to completely offset crypto tax liability (Tax-Loss Harvesting).

    Parameters:
    - current_net_realized_profit_krw: Net realized profit accumulated year-to-date in KRW
    - annual_allowance_krw: Basic statutory tax exemption (default: 2,500,000 KRW)
    - tax_rate: Effective tax rate (default: 0.22)
    """
    if isinstance(current_net_realized_profit_krw, bool) or not np.isfinite(current_net_realized_profit_krw):
        raise ValueError("Current profit must be a finite number.")
    if isinstance(annual_allowance_krw, bool) or not np.isfinite(annual_allowance_krw) or annual_allowance_krw < 0:
        raise ValueError("Annual allowance must be a non-negative finite number.")
    if isinstance(tax_rate, bool) or not np.isfinite(tax_rate) or tax_rate < 0 or tax_rate > 1:
        raise ValueError("Tax rate must be between 0.0 and 1.0.")

    excess_profit = max(0.0, current_net_realized_profit_krw - annual_allowance_krw)
    potential_tax_saved = excess_profit * tax_rate

    return {
        "current_net_profit_krw": round(current_net_realized_profit_krw, 2),
        "annual_allowance_krw": round(annual_allowance_krw, 2),
        "taxable_excess_krw": round(excess_profit, 2),
        "recommended_loss_harvest_krw": round(excess_profit, 2),
        "potential_tax_savings_krw": round(potential_tax_saved, 2),
        "needs_harvesting": excess_profit > 0,
    }


def compare_tax_allowance_tiers(
    realized_profits: list[float],
    tiers: list[float] | None = None,
    tax_rate: float = 0.22,
    initial_capital_krw: float = 50000000.0,
    carried_forward_loss_krw: float = 0.0,
) -> pd.DataFrame:
    """
    Compare tax liabilities, tax drag, and after-tax profits across different basic deduction allowance tiers.
    E.g. standard KRW 2,500,000 vs proposed KRW 50,000,000 allowance.
    """
    if tiers is None:
        tiers = [2500000.0, 5000000.0, 10000000.0, 50000000.0]

    rows = []
    for allowance in tiers:
        res = compute_crypto_tax_impact(
            realized_profits=realized_profits,
            annual_allowance_krw=float(allowance),
            tax_rate=tax_rate,
            initial_capital_krw=initial_capital_krw,
            carried_forward_loss_krw=carried_forward_loss_krw,
        )
        rows.append({
            "annual_allowance_krw": res["annual_allowance_krw"],
            "taxable_base": res["taxable_base"],
            "estimated_tax_krw": res["estimated_tax_krw"],
            "after_tax_profit_krw": res["after_tax_profit_krw"],
            "after_tax_return_pct": res["after_tax_return_pct"],
            "tax_drag_pct": res["tax_drag_pct"],
        })
    return pd.DataFrame(rows)


def format_allowance_comparison_table(df: pd.DataFrame) -> str:
    """Format multi-tier allowance comparison DataFrame into readable text table."""
    lines = [
        "=========================================================================================",
        "                 KOREA CRYPTO TAX ALLOWANCE TIER COMPARISON MATRIX                       ",
        "=========================================================================================",
        f"{'Basic Allowance':<20} | {'Taxable Base':<16} | {'Estimated Tax':<16} | {'After-Tax Return'} | {'Tax Drag'}",
        "-----------------------------------------------------------------------------------------",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"₩{r['annual_allowance_krw']:>17,.0f} | ₩{r['taxable_base']:>14,.0f} | ₩{r['estimated_tax_krw']:>14,.0f} | {r['after_tax_return_pct']:>16.2f}% | -{r['tax_drag_pct']:>6.2f}%p"
        )
    lines.append("=========================================================================================")
    return "\n".join(lines)


def print_tax_report(res: dict[str, object]):
    """Print terminal report of capital gains tax simulation."""
    print("================================================================================")
    print("             KOREA VIRTUAL ASSET CAPITAL GAINS TAX SIMULATION                   ")
    print("================================================================================")
    print(f"초기 원금 (Initial Capital)        : ₩{res['initial_capital_krw']:,.0f}")
    print(f"총 실현 이익 (Gross Gains)         : ₩{res['gross_realized_gains']:,.0f}")
    print(f"총 실현 손실 (Gross Losses)        : -₩{res['gross_realized_losses']:,.0f}")
    print(f"손익 통산 순수익 (Net Profit)      : ₩{res['net_realized_profit']:,.0f}")
    if res.get("carried_forward_loss_krw", 0) > 0:
        print(f"이월결손금 공제 (Carried Loss)     : -₩{res['carried_loss_applied_krw']:,.0f} (잔여: ₩{res['remaining_carried_loss_krw']:,.0f})")
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
    parser.add_argument("--carried-loss", type=float, default=0.0, help="Prior year carried-forward loss in KRW")
    parser.add_argument("--tax-rate", type=float, default=0.22, help="Effective tax rate (default: 0.22)")
    parser.add_argument("--compare-tiers", action="store_true", help="Compare tax burden across multiple basic allowance tiers")
    parser.add_argument("--export-json", type=str, default="", help="Path to export JSON metrics")
    args = parser.parse_args()

    # Representative sample profits: mixed gains and losses
    sample_trades = [
        args.profit * 0.70,
        args.profit * 0.50,
        -args.profit * 0.20,
    ]

    if args.compare_tiers:
        tier_df = compare_tax_allowance_tiers(
            realized_profits=sample_trades,
            tax_rate=args.tax_rate,
            initial_capital_krw=args.capital,
            carried_forward_loss_krw=args.carried_loss,
        )
        print(format_allowance_comparison_table(tier_df))
        if args.export_json:
            out_path = Path(args.export_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(tier_df.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")
            print(f"[+] Allowance tiers comparison exported to: {out_path}")
        return

    res = compute_crypto_tax_impact(
        realized_profits=sample_trades,
        annual_allowance_krw=args.allowance,
        tax_rate=args.tax_rate,
        initial_capital_krw=args.capital,
        carried_forward_loss_krw=args.carried_loss,
    )

    print_tax_report(res)

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[+] Tax simulation metrics exported to: {out_path}")


if __name__ == "__main__":
    main()
