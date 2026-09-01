"""Volatility-Based Dynamic Stoploss and Take-Profit Guide Calculator.

Computes coin-specific risk management parameters:
1. Calculates Periodic Volatility and Downside Semi-Deviation for each crypto asset.
2. Scales recommended stoploss (SL) and take-profit (TP) based on asset volatility and desired Risk-Reward ratio.
3. Generates Freqtrade-compatible stoploss and minimal_roi parameter recommendations.
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

from scripts.crypto_portfolio_optimizer import (
    find_freqtrade_data_dirs,
    generate_synthetic_crypto_data,
    load_from_feather_dir,
    run_optimization,
)


def calculate_volatility_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate periodic volatility, annualized volatility, and downside semi-deviation."""
    returns = prices.pct_change().dropna()
    metrics = []

    for col in returns.columns:
        ret = returns[col]
        vol = float(ret.std())
        # Downside semi-deviation (only negative returns)
        neg_ret = ret[ret < 0]
        semi_dev = float(neg_ret.std()) if len(neg_ret) > 1 else vol
        metrics.append({
            "asset": col,
            "periodic_vol": vol,
            "semi_dev": semi_dev,
        })

    return pd.DataFrame(metrics).set_index("asset")


def compute_risk_guidelines(
    prices: pd.DataFrame,
    weights: dict[str, float] | None = None,
    risk_multiplier: float = 2.0,
    risk_reward_ratio: float = 2.0,
) -> dict[str, dict[str, float]]:
    """
    Compute recommended dynamic Stoploss (SL) and Take-Profit (TP) per asset.

    Formula:
    - Base Stoploss = -1.0 * (semi_dev * risk_multiplier) (bounded between -2.0% and -15.0%)
    - Base Take-Profit = abs(Stoploss) * risk_reward_ratio
    """
    vol_df = calculate_volatility_metrics(prices)
    guidelines: dict[str, dict[str, float]] = {}

    for asset, row in vol_df.iterrows():
        weight = weights.get(asset, 1.0 / len(vol_df)) if weights else 1.0 / len(vol_df)
        semi_dev = row["semi_dev"]

        # Calculate dynamic stoploss percentage
        sl_pct = -1.0 * max(0.025, min(0.15, semi_dev * risk_multiplier * 5))
        tp_pct = abs(sl_pct) * risk_reward_ratio

        guidelines[asset] = {
            "weight": round(weight, 4),
            "semi_dev": round(semi_dev * 100, 3),
            "recommended_stoploss": round(sl_pct, 4),
            "recommended_take_profit": round(tp_pct, 4),
            "risk_reward_ratio": round(risk_reward_ratio, 1),
        }

    return guidelines


def print_risk_report(guidelines: dict[str, dict[str, float]]):
    """Print formatted risk management guide table."""
    print("================================================================================")
    print("       VOLATILITY-BASED DYNAMIC STOPLOSS & TAKE-PROFIT GUIDELINE                ")
    print("================================================================================")
    print(f"{'Coin / Pair':<15} | {'Weight':>8} | {'Semi-Dev':>10} | {'Stoploss (SL)':>14} | {'Take-Profit (TP)':>16}")
    print("--------------------------------------------------------------------------------")

    for asset, g in guidelines.items():
        weight_str = f"{g['weight']*100:.1f}%"
        dev_str = f"{g['semi_dev']:.2f}%"
        sl_str = f"{g['recommended_stoploss']*100:.2f}%"
        tp_str = f"{g['recommended_take_profit']*100:.2f}%"
        print(f"{asset:<15} | {weight_str:>8} | {dev_str:>10} | {sl_str:>14} | {tp_str:>16}")

    print("--------------------------------------------------------------------------------")
    print(" - Stoploss is dynamically scaled to absorb routine downside noise.")
    print(" - Take-profit targets a 1:2.0 Risk-Reward ratio on average.")
    print("================================================================================\n")


def export_risk_json(guidelines: dict[str, dict[str, float]], output_path: Path):
    """Export guidelines to JSON for easy loading in Freqtrade strategies."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "assets": guidelines,
        "freqtrade_stoploss_config": {
            k: v["recommended_stoploss"] for k, v in guidelines.items()
        },
        "freqtrade_minimal_roi": {
            "0": 0.05,
            "30": 0.025,
            "60": 0.01,
        }
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"[+] Risk guidelines exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Volatility-Based Dynamic SL/TP Guide Calculator")
    parser.add_argument("--data-dir", type=str, default="", help="Directory containing Freqtrade feather files")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe")
    parser.add_argument("--risk-mult", type=float, default=2.0, help="Multiplier for downside deviation (default: 2.0)")
    parser.add_argument("--rr-ratio", type=float, default=2.0, help="Risk-Reward ratio (default: 2.0)")
    parser.add_argument("--export-json", type=str, default="", help="Path to export risk guideline JSON")
    parser.add_argument("--use-synthetic", action="store_true", help="Force synthetic data")
    args = parser.parse_args()

    prices = pd.DataFrame()
    if not args.use_synthetic:
        candidate_dirs = find_freqtrade_data_dirs()
        for d in candidate_dirs:
            if d.is_dir():
                prices = load_from_feather_dir(d, timeframe=args.timeframe)
                if not prices.empty:
                    print(f"[*] Loaded market data from: {d} ({len(prices)} bars)")
                    break

    if prices.empty or args.use_synthetic:
        prices = generate_synthetic_crypto_data(periods=500)

    # Calculate optimal weights
    results = run_optimization(prices)
    weights = results.get("Risk Parity (ERC)", {c: 1.0/len(prices.columns) for c in prices.columns}) if results else None

    guidelines = compute_risk_guidelines(
        prices=prices,
        weights=weights,
        risk_multiplier=args.risk_mult,
        risk_reward_ratio=args.rr_ratio,
    )

    print_risk_report(guidelines)

    if args.export_json:
        export_risk_json(guidelines, Path(args.export_json))


if __name__ == "__main__":
    main()
