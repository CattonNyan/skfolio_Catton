"""Kimchi Premium Regime-Based Tactical Asset Allocation Module.

Translates domestic market sentiment and arbitrage imbalance (Kimchi Premium)
into tactical crypto-to-cash risk allocation shifts:
- Extreme Overheating (> 5%): De-risk and scale up defensive cash allocation.
- Neutral / Fair (0% to 3%): Maintain baseline optimal skfolio weights.
- Negative Discount (< 0%): Rebalance aggressively into crypto risk assets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np


class KimchiRegime:
    EXTREME_OVERHEATED = "EXTREME_OVERHEATED"
    MODERATE_OVERHEATED = "MODERATE_OVERHEATED"
    FAIR_EQUILIBRIUM = "FAIR_EQUILIBRIUM"
    NEGATIVE_DISCOUNT = "NEGATIVE_DISCOUNT"


def classify_kimchi_regime(
    premium_pct: float,
    overheated_threshold: float = 5.0,
    discount_threshold: float = 0.0,
) -> dict[str, object]:
    """
    Classify current market state based on Kimchi Premium level.

    Parameters:
    - premium_pct: Current Kimchi Premium percentage (e.g. 4.2 for +4.2%)
    - overheated_threshold: Threshold above which market is considered overheated
    - discount_threshold: Threshold below which market is at a discount (negative premium)
    """
    if isinstance(premium_pct, bool) or not isinstance(premium_pct, (int, float)) or not math.isfinite(premium_pct):
        raise ValueError("Premium percentage must be a finite number.")
    if isinstance(overheated_threshold, bool) or not isinstance(overheated_threshold, (int, float)) or not math.isfinite(overheated_threshold):
        raise ValueError("Overheated threshold must be a finite number.")
    if isinstance(discount_threshold, bool) or not isinstance(discount_threshold, (int, float)) or not math.isfinite(discount_threshold):
        raise ValueError("Discount threshold must be a finite number.")
    if overheated_threshold <= discount_threshold:
        raise ValueError("Overheated threshold must be strictly greater than discount threshold.")

    if premium_pct >= overheated_threshold:
        regime = KimchiRegime.EXTREME_OVERHEATED
        target_crypto_ratio = 0.40
        target_cash_ratio = 0.60
        action = "Heavy Profit Taking / Move to Cash or Stablecoins (High Dumping Risk)"
    elif premium_pct >= 3.0:
        regime = KimchiRegime.MODERATE_OVERHEATED
        target_crypto_ratio = 0.70
        target_cash_ratio = 0.30
        action = "Moderate De-risking / Rebalance Excess Domestic Gains"
    elif premium_pct < discount_threshold:
        regime = KimchiRegime.NEGATIVE_DISCOUNT
        target_crypto_ratio = 0.95
        target_cash_ratio = 0.05
        action = "Aggressive Accumulation / Arbitrage Discount Inflow"
    else:
        regime = KimchiRegime.FAIR_EQUILIBRIUM
        target_crypto_ratio = 0.85
        target_cash_ratio = 0.15
        action = "Maintain Baseline Risk-Parity / Standard Optimal Allocation"

    return {
        "premium_pct": round(premium_pct, 2),
        "regime": regime,
        "target_crypto_ratio": target_crypto_ratio,
        "target_cash_ratio": target_cash_ratio,
        "tactical_action": action,
    }


def adjust_portfolio_weights_by_kimchi(
    base_weights: dict[str, float],
    premium_pct: float,
    cash_asset: str = "KRW",
) -> dict[str, float]:
    """
    Scale a set of asset weights to account for the Kimchi Premium regime.

    Parameters:
    - base_weights: Dictionary of asset names to weights summing to ~1.0
    - premium_pct: Current Kimchi premium percentage
    - cash_asset: Symbol for the cash/safe haven asset (default: 'KRW')

    Returns:
    - Adjusted weights dict including cash_asset, summing to 1.0
    """
    if not isinstance(base_weights, dict) or not base_weights:
        raise ValueError("Base weights must be a non-empty dictionary.")

    for k, v in base_weights.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"Weight for asset '{k}' must be a non-negative finite number.")

    total_base = sum(base_weights.values())
    if total_base <= 0:
        raise ValueError("Sum of base weights must be strictly positive.")

    # Normalize base weights
    norm_base = {k: v / total_base for k, v in base_weights.items()}

    regime_info = classify_kimchi_regime(premium_pct)
    target_crypto = float(regime_info["target_crypto_ratio"])
    target_cash = float(regime_info["target_cash_ratio"])

    # Scale crypto assets proportionally by target_crypto
    adjusted: dict[str, float] = {}
    for asset, weight in norm_base.items():
        adjusted[asset] = round(weight * target_crypto, 4)

    adjusted[cash_asset] = round(target_cash, 4)

    # Normalize to exactly 1.0
    total_adj = sum(adjusted.values())
    if total_adj > 0:
        adjusted = {k: round(v / total_adj, 4) for k, v in adjusted.items()}

    return adjusted


def print_kimchi_regime_report(regime_info: dict[str, object], adjusted_weights: dict[str, float]):
    """Print terminal report of Kimchi Premium tactical allocation."""
    print("================================================================================")
    print(f"        KIMCHI PREMIUM TACTICAL REGIME (Premium: {regime_info['premium_pct']:+.2f}%)        ")
    print("================================================================================")
    print(f"현재 시장 레짐 (Market Regime)     : {regime_info['regime']}")
    print(f"권장 크립토 위험자산 비중          : {regime_info['target_crypto_ratio'] * 100:.1f}%")
    print(f"권장 안전자산(현금) 비중           : {regime_info['target_cash_ratio'] * 100:.1f}%")
    print(f"전술적 행동 지침                   : {regime_info['tactical_action']}")
    print("--------------------------------------------------------------------------------")
    print(f"{'Asset':<15} | {'Tactical Target Weight':>22}")
    print("--------------------------------------------------------------------------------")
    for asset, w in adjusted_weights.items():
        print(f"{asset:<15} | {w * 100:>21.2f}%")
    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Kimchi Premium Tactical Asset Allocation")
    parser.add_argument("--premium", type=float, default=None, help="Kimchi Premium pct (default: fetch live)")
    parser.add_argument("--export-json", type=str, default="", help="Path to export results JSON")
    args = parser.parse_args()

    premium_val = args.premium
    if premium_val is None:
        try:
            from scripts.crypto_kimchi_premium import fetch_live_usd_krw_rate, compute_kimchi_premium
            fx_rate, _ = fetch_live_usd_krw_rate()
            # Sample live ticker
            sample_upbit = {"BTC": 136500000.0}
            sample_binance = {"BTC": 98000.0}
            res = compute_kimchi_premium(sample_upbit, sample_binance, usdt_krw_rate=fx_rate)
            premium_val = float(res["BTC"]["premium_pct"])
        except Exception:
            premium_val = 3.5

    regime_info = classify_kimchi_regime(premium_val)

    sample_base = {
        "KRW-BTC": 0.40,
        "KRW-ETH": 0.30,
        "KRW-SOL": 0.15,
        "KRW-XRP": 0.15,
    }
    adjusted = adjust_portfolio_weights_by_kimchi(sample_base, premium_pct=premium_val)
    print_kimchi_regime_report(regime_info, adjusted)

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "regime_info": regime_info,
            "adjusted_weights": adjusted,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[+] Regime results exported to: {out_path}")


if __name__ == "__main__":
    main()
