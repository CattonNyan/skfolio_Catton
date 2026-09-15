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


DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
    "momentum": 0.30,
    "low_volatility": 0.25,
    "trend_strength": 0.25,
    "sortino_ratio": 0.20,
}

_FACTOR_KEY_TO_Z: dict[str, str] = {
    "momentum": "z_momentum",
    "low_volatility": "z_low_vol",
    "trend_strength": "z_trend",
    "sortino_ratio": "z_sortino",
}


def compute_crypto_factors(
    prices: pd.DataFrame,
    lookback_bars: int = 60,
    factor_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute Momentum, Low Volatility, Trend Strength, and Downside Sortino factors per asset.

    Returns DataFrame containing raw factors, z-scores, and weighted composite score.
    """
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] < 1 or len(prices) < 2:
        raise ValueError("Prices must be a DataFrame with at least one asset and two rows.")
    try:
        price_values = prices.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Prices must contain only numeric values.") from error
    if not np.all(np.isfinite(price_values)) or np.any(price_values <= 0):
        raise ValueError("Prices must contain only finite, strictly positive values.")
    if (
        isinstance(lookback_bars, bool)
        or not isinstance(lookback_bars, int)
        or lookback_bars < 2
    ):
        raise ValueError("Lookback bars must be an integer of at least 2.")
    if len(prices) < lookback_bars:
        raise ValueError(f"Prices length ({len(prices)}) is shorter than lookback ({lookback_bars}).")

    # Validate and normalize factor weights
    active_weights = dict(DEFAULT_FACTOR_WEIGHTS)
    if factor_weights is not None:
        if not isinstance(factor_weights, dict) or not factor_weights:
            raise ValueError("Factor weights must be a non-empty dictionary.")
        unknown_keys = set(factor_weights.keys()) - set(_FACTOR_KEY_TO_Z.keys())
        if unknown_keys:
            raise ValueError(f"Unknown factor weight keys: {unknown_keys}. Allowed: {list(_FACTOR_KEY_TO_Z.keys())}")
        clean_w = {}
        for k, v in factor_weights.items():
            if isinstance(v, bool) or not isinstance(v, (int, float, np.number)) or not np.isfinite(v) or v < 0:
                raise ValueError(f"Factor weight for '{k}' must be a finite non-negative number.")
            clean_w[k] = float(v)
        total_w = sum(clean_w.values())
        if total_w <= 0:
            raise ValueError("Sum of factor weights must be strictly positive.")
        active_weights = {k: v / total_w for k, v in clean_w.items()}

    recent_prices = prices.iloc[-lookback_bars:]
    returns = recent_prices.pct_change().dropna()

    fast_win = min(20, len(recent_prices))
    slow_win = min(60, len(recent_prices))
    momentum = (recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1.0
    vol = returns.std() + 1e-9
    low_vol = 1.0 / vol
    sma_fast = recent_prices.iloc[-fast_win:].mean()
    sma_slow = recent_prices.iloc[-slow_win:].mean()
    trend_ratio = sma_fast / (sma_slow + 1e-9)

    neg_returns = returns.where(returns < 0)
    downside_dev = neg_returns.std().fillna(vol) + 1e-9
    sortino_ratio = returns.mean() / downside_dev

    df = pd.DataFrame({
        "momentum": momentum,
        "volatility": vol,
        "low_volatility": low_vol,
        "trend_strength": trend_ratio,
        "sortino_ratio": sortino_ratio,
    }, index=prices.columns).rename_axis("asset")

    # Compute Z-Scores across assets
    def zscore(series: pd.Series) -> pd.Series:
        std = series.std()
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std

    df["z_momentum"] = zscore(df["momentum"])
    df["z_low_vol"] = zscore(df["low_volatility"])
    df["z_trend"] = zscore(df["trend_strength"])
    df["z_sortino"] = zscore(df["sortino_ratio"])

    # Composite Smart Beta Score using active weights
    comp = pd.Series(0.0, index=df.index)
    for factor_key, weight in active_weights.items():
        z_col = _FACTOR_KEY_TO_Z[factor_key]
        comp += weight * df[z_col]
    df["composite_score"] = comp

    df = df.sort_values(by="composite_score", ascending=False)
    return df


def generate_factor_tilted_weights(
    factors_df: pd.DataFrame,
    top_n: int = 3,
    weighting: str = "score_weighted",
) -> dict[str, float]:
    """
    Generate portfolio allocation weights tilted towards top-ranked Smart Beta assets.

    Parameters:
    - factors_df: DataFrame output of compute_crypto_factors
    - top_n: Number of highest-ranked assets to include
    - weighting: 'equal' or 'score_weighted' (score-proportional)
    """
    if not isinstance(factors_df, pd.DataFrame) or "composite_score" not in factors_df.columns:
        raise ValueError("factors_df must be a DataFrame containing 'composite_score'.")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a strictly positive integer.")
    if top_n > len(factors_df):
        raise ValueError(f"top_n ({top_n}) cannot exceed available assets ({len(factors_df)}).")
    if weighting not in {"equal", "score_weighted"}:
        raise ValueError("weighting must be either 'equal' or 'score_weighted'.")

    top_subset = factors_df.iloc[:top_n]
    all_assets = list(factors_df.index)
    weights = {a: 0.0 for a in all_assets}

    if weighting == "equal":
        eq_w = 1.0 / top_n
        for a in top_subset.index:
            weights[a] = round(eq_w, 4)
    else:
        # Shift scores so all top_n scores are positive
        raw_scores = top_subset["composite_score"].values
        min_s = float(np.min(raw_scores))
        positive_scores = raw_scores - min_s + 1.0  # Base shift ensure strictly positive
        s_sum = float(np.sum(positive_scores))
        norm_weights = positive_scores / s_sum
        for a, w in zip(top_subset.index, norm_weights):
            weights[a] = round(float(w), 4)

    # Adjust rounding residual to exactly 1.0
    w_sum = sum(weights.values())
    if w_sum > 0:
        first_asset = list(top_subset.index)[0]
        weights[first_asset] = round(weights[first_asset] + (1.0 - w_sum), 4)

    return weights


def select_smart_beta_universe(
    prices: pd.DataFrame,
    top_n: int = 3,
    lookback_bars: int = 60,
) -> tuple[list[str], pd.DataFrame]:
    """
    Select top N assets using multi-factor ranking and slice price DataFrame.
    """
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("Top N must be a strictly positive integer.")
    if top_n > len(prices.columns):
        raise ValueError(
            f"Top N ({top_n}) cannot exceed the number of assets ({len(prices.columns)})."
        )
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
