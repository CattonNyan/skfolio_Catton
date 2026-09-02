"""Volatility-Based Dynamic Stoploss and Take-Profit Guide Calculator.

Computes coin-specific risk management parameters:
1. Calculates Periodic Volatility and Downside Semi-Deviation for each crypto asset.
2. Scales recommended stoploss (SL) and take-profit (TP) based on asset volatility and desired Risk-Reward ratio.
3. Generates Freqtrade-compatible stoploss and minimal_roi parameter recommendations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
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
    MarketDataUnavailableError,
    find_freqtrade_data_dirs,
    generate_synthetic_crypto_data,
    load_market_data,
    load_from_feather_dir,
    positive_float,
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
    if not np.isfinite(risk_multiplier) or risk_multiplier <= 0:
        raise ValueError("risk_multiplier must be a positive finite number")
    if not np.isfinite(risk_reward_ratio) or risk_reward_ratio <= 0:
        raise ValueError("risk_reward_ratio must be a positive finite number")

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


def print_risk_report(guidelines: dict[str, dict[str, float]], total_wallet: float = 10000.0):
    """Print formatted risk management guide table."""
    print("==========================================================================================")
    print("              VOLATILITY-BASED DYNAMIC STOPLOSS & TAKE-PROFIT GUIDELINE                   ")
    print("==========================================================================================")
    print(f"{'Coin / Pair':<15} | {'Weight':>8} | {'Stake ($)':>10} | {'Semi-Dev':>10} | {'Stoploss (SL)':>14} | {'Take-Profit (TP)':>16}")
    print("------------------------------------------------------------------------------------------")

    for asset, g in guidelines.items():
        weight_str = f"{g['weight']*100:.1f}%"
        stake_str = f"${g['weight'] * total_wallet:,.2f}"
        dev_str = f"{g['semi_dev']:.2f}%"
        sl_str = f"{g['recommended_stoploss']*100:.2f}%"
        tp_str = f"{g['recommended_take_profit']*100:.2f}%"
        print(f"{asset:<15} | {weight_str:>8} | {stake_str:>10} | {dev_str:>10} | {sl_str:>14} | {tp_str:>16}")

    print("------------------------------------------------------------------------------------------")
    print(" - Stoploss is dynamically scaled to absorb routine downside noise.")
    print(" - Take-profit targets a 1:2.0 Risk-Reward ratio on average.")
    print("==========================================================================================\n")


def export_risk_json(
    guidelines: dict[str, dict[str, float]],
    output_path: Path,
    data_source: str = "unspecified",
):
    """Export guidelines to JSON for easy loading in Freqtrade strategies."""
    export_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "assets": guidelines,
    }
    _atomic_write_json(output_path, export_payload)
    print(f"[+] Risk guidelines exported to: {output_path}")


def _atomic_write_json(output_path: Path, payload: dict) -> None:
    """Durably replace a JSON file without exposing a partially written file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            json.dump(payload, temp_file, indent=2)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_name = temp_file.name
        os.replace(temp_name, output_path)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()


def update_freqtrade_risk_config(
    guidelines: dict[str, dict[str, float]],
    config_path: Path,
    data_source: str,
) -> bool:
    """Inject callback-consumable pair risk limits into a Freqtrade config."""
    if data_source.lower() == "synthetic":
        print("[!] Refusing to inject synthetic-data risk limits into Freqtrade.")
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("exchange"), dict):
            raise ValueError("target is not a Freqtrade configuration")

        limits: dict[str, dict[str, float]] = {}
        for pair, values in guidelines.items():
            stoploss = float(values["recommended_stoploss"])
            take_profit = float(values["recommended_take_profit"])
            if not np.isfinite(stoploss) or not -1 < stoploss < 0:
                raise ValueError(f"invalid stoploss for {pair}")
            if not np.isfinite(take_profit) or not 0 < take_profit < 10:
                raise ValueError(f"invalid take-profit for {pair}")
            limits[pair] = {
                "recommended_stoploss": stoploss,
                "recommended_take_profit": take_profit,
            }

        config["pair_risk_limits"] = limits
        config["skfolio_risk"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_source": data_source,
        }
        _atomic_write_json(config_path, config)
        print(f"[+] Freqtrade risk callbacks configured in: {config_path}")
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"[!] Freqtrade risk config was not modified: {error}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Volatility-Based Dynamic SL/TP Guide Calculator")
    parser.add_argument("--data-dir", type=str, default="", help="Directory containing Freqtrade feather files")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe")
    parser.add_argument("--risk-mult", type=positive_float, default=2.0, help="Multiplier for downside deviation (default: 2.0)")
    parser.add_argument("--rr-ratio", type=positive_float, default=2.0, help="Risk-Reward ratio (default: 2.0)")
    parser.add_argument("--total-wallet", "--wallet", type=positive_float, default=10000.0, help="Total wallet size in USDT (default: 10000.0)")
    parser.add_argument("--export-json", type=str, default="", help="Path to export risk guideline JSON")
    parser.add_argument("--freqtrade-config", type=str, default="", help="Existing Freqtrade config.json to update")
    parser.add_argument("--use-synthetic", action="store_true", help="Force synthetic data")
    args = parser.parse_args()

    try:
        prices, data_source = load_market_data(
            data_dir=args.data_dir or None,
            timeframe=args.timeframe,
            use_synthetic=args.use_synthetic,
            synthetic_periods=500,
        )
    except MarketDataUnavailableError as error:
        parser.error(str(error))

    print(
        "[!] DATA SOURCE: SYNTHETIC (--use-synthetic was explicitly enabled)"
        if data_source == "synthetic"
        else f"[+] DATA SOURCE: REAL ({data_source}, {len(prices)} bars)"
    )

    # Calculate optimal weights
    results = run_optimization(prices)
    weights = results.get("Risk Parity (ERC)", {c: 1.0/len(prices.columns) for c in prices.columns}) if results else None

    guidelines = compute_risk_guidelines(
        prices=prices,
        weights=weights,
        risk_multiplier=args.risk_mult,
        risk_reward_ratio=args.rr_ratio,
    )

    print_risk_report(guidelines, total_wallet=args.total_wallet)

    if args.export_json:
        export_risk_json(guidelines, Path(args.export_json), data_source=data_source)

    if args.freqtrade_config:
        if not update_freqtrade_risk_config(
            guidelines,
            Path(args.freqtrade_config),
            data_source=data_source,
        ):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
