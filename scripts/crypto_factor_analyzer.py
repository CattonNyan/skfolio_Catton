"""Cryptocurrency Multi-Factor Quantitative Analyzer & Smart Beta Screener.

Ranks and filters the crypto universe based on academic and hedge fund factors:
- Momentum Factor: Historical return over lookback window
- Low Volatility Factor: Inverse return standard deviation
- Trend Strength Factor: Ratio of short-term SMA(20) to long-term SMA(60)
Combines factor z-scores into a Composite Smart Beta rank to screen top assets.
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
)


def compute_crypto_factors(
    prices: pd.DataFrame,
    lookback_bars: int = 60,
) -> pd.DataFrame:
    """
    Compute Momentum, Low Volatility, and Trend Strength factors per asset.

    Returns DataFrame containing raw factors and composite z-scores.
    """
    if len(prices) < lookback_bars:
        raise ValueError(f"Prices length ({len(prices)}) is shorter than lookback ({lookback_bars}).")

    recent_prices = prices.iloc[-lookback_bars:]
    returns = recent_prices.pct_change().dropna()

    records = []
    for asset in prices.columns:
        p_series = recent_prices[asset]
        r_series = returns[asset]

        # 1. Momentum Factor: cumulative return
        momentum = float((p_series.iloc[-1] / p_series.iloc[0]) - 1.0)

        # 2. Low Volatility Factor: inverse standard deviation
        vol = float(r_series.std() + 1e-9)
        low_vol = float(1.0 / vol)

        # 3. Trend Strength Factor: Fast SMA(20) / Slow SMA(min(60, lookback))
        fast_win = min(20, len(p_series))
        slow_win = min(60, len(p_series))
        sma_fast = float(p_series.iloc[-fast_win:].mean())
        sma_slow = float(p_series.iloc[-slow_win:].mean())
        trend_ratio = float(sma_fast / (sma_slow + 1e-9))

        records.append({
            "asset": asset,
            "momentum": momentum,
            "volatility": vol,
            "low_volatility": low_vol,
            "trend_strength": trend_ratio,
        })

    df = pd.DataFrame(records).set_index("asset")

    # Compute Z-Scores across assets
    def zscore(series: pd.Series) -> pd.Series:
        std = series.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std

    df["z_momentum"] = zscore(df["momentum"])
    df["z_low_vol"] = zscore(df["low_volatility"])
    df["z_trend"] = zscore(df["trend_strength"])

    # Composite Smart Beta Score: 40% Momentum + 30% Low Vol + 30% Trend
    df["composite_score"] = (
        0.40 * df["z_momentum"] +
        0.30 * df["z_low_vol"] +
        0.30 * df["z_trend"]
    )

    df = df.sort_values(by="composite_score", ascending=False)
    return df


def select_smart_beta_universe(
    prices: pd.DataFrame,
    top_n: int = 3,
    lookback_bars: int = 60,
) -> tuple[list[str], pd.DataFrame]:
    """
    Select top N assets using multi-factor ranking and slice price DataFrame.
    """
    factors = compute_crypto_factors(prices, lookback_bars=lookback_bars)
    selected_assets = list(factors.index[:top_n])
    filtered_prices = prices[selected_assets]
    return selected_assets, filtered_prices


def print_factor_report(df: pd.DataFrame):
    """Print terminal report of multi-factor ranking."""
    print("================================================================================")
    print("            QUANTITATIVE MULTI-FACTOR CRYPTO SCREENER REPORT                    ")
    print("================================================================================")
    print(f"{'Rank':<5} | {'Asset':<12} | {'Momentum':>10} | {'Vol(%)':>8} | {'Trend':>8} | {'Score':>8}")
    print("--------------------------------------------------------------------------------")

    for rank, (asset, row) in enumerate(df.iterrows(), start=1):
        mom_str = f"{row['momentum']*100:+.2f}%"
        vol_str = f"{row['volatility']*100:.2f}%"
        trend_str = f"{row['trend_strength']:.3f}"
        score_str = f"{row['composite_score']:+.3f}"
        print(f"{rank:<5} | {asset:<12} | {mom_str:>10} | {vol_str:>8} | {trend_str:>8} | {score_str:>8}")

    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Crypto Quantitative Multi-Factor Analyzer")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback bars for factor calculation")
    parser.add_argument("--top-n", type=int, default=3, help="Number of top assets to select")
    parser.add_argument("--export-json", type=str, default="", help="Path to export results JSON")
    parser.add_argument("--use-synthetic", action="store_true", help="Force synthetic sample data")
    args = parser.parse_args()

    prices = pd.DataFrame()
    if not args.use_synthetic:
        candidate_dirs = find_freqtrade_data_dirs()
        for d in candidate_dirs:
            if d.is_dir():
                prices = load_from_feather_dir(d, timeframe="15m")
                if not prices.empty:
                    break

    if prices.empty or args.use_synthetic:
        prices = generate_synthetic_crypto_data(periods=200)

    factors_df = compute_crypto_factors(prices, lookback_bars=args.lookback)
    print_factor_report(factors_df)

    selected, _ = select_smart_beta_universe(prices, top_n=args.top_n, lookback_bars=args.lookback)
    print(f"[+] Top {args.top_n} Smart Beta Universe Selected: {selected}\n")

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_dict = factors_df.reset_index().to_dict(orient="records")
        out_path.write_text(json.dumps(out_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[+] Factor rankings exported to: {out_path}")


if __name__ == "__main__":
    main()
