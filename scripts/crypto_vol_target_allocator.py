"""Volatility-Targeting and Leverage Dynamic Allocator for Crypto Portfolios.

Implements institutional constant volatility targeting:
- Dynamically scales risky portfolio weights inversely proportional to rolling realized volatility
- Shifts allocation into cash/stablecoin during crypto flash crashes and high-volatility regimes
- Prevents deep drawdowns while compounding capital during low-volatility trend regimes
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass
class VolTargetResult:
    """Container for volatility targeting allocation result."""
    target_vol_ann: float
    realized_vol_ann: float
    vol_scalar: float
    scaled_weights: dict[str, float]
    cash_weight: float
    is_leveraged: bool


def calculate_portfolio_realized_volatility(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    annual_factor: float = 365.0,
) -> float:
    """Calculate annualized realized volatility for a given portfolio weight mapping."""
    if returns_df.empty or len(returns_df) < 2:
        raise ValueError("Returns DataFrame must have at least 2 rows.")

    w = np.array([weights.get(c, 0.0) for c in returns_df.columns], dtype=float)
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()

    cov = returns_df.cov().values
    var = float(w @ cov @ w)
    if var <= 0:
        return 0.0
    return float(np.sqrt(var * annual_factor))


def apply_volatility_targeting(
    base_weights: dict[str, float],
    realized_vol_ann: float,
    target_vol_ann: float = 0.30,
    max_leverage: float = 1.0,
    min_scalar: float = 0.10,
) -> VolTargetResult:
    """Scale base portfolio weights by target volatility ratio.

    Scalar = clamp(target_vol / realized_vol, min_scalar, max_leverage)
    """
    if target_vol_ann <= 0.0 or max_leverage <= 0.0:
        raise ValueError("Target volatility and max leverage must be strictly positive.")
    if realized_vol_ann <= 0.0:
        # If realized volatility is near zero, clamp to max leverage
        scalar = max_leverage
    else:
        raw_scalar = target_vol_ann / realized_vol_ann
        scalar = max(min_scalar, min(float(raw_scalar), max_leverage))

    scaled = {}
    total_risky = 0.0
    for asset, w in base_weights.items():
        scaled_w = float(w * scalar)
        scaled[asset] = round(scaled_w, 4)
        total_risky += scaled_w

    cash_weight = max(0.0, round(1.0 - total_risky, 4)) if scalar <= 1.0 else 0.0

    return VolTargetResult(
        target_vol_ann=target_vol_ann,
        realized_vol_ann=round(realized_vol_ann, 4),
        vol_scalar=round(scalar, 4),
        scaled_weights=scaled,
        cash_weight=cash_weight,
        is_leveraged=scalar > 1.0,
    )


def simulate_vol_targeted_backtest(
    prices_df: pd.DataFrame,
    base_weights: dict[str, float],
    target_vol_ann: float = 0.35,
    lookback_bars: int = 30,
    max_leverage: float = 1.0,
    annual_factor: float = 365.0,
) -> dict[str, object]:
    """Simulate walk-forward volatility targeting portfolio vs unhedged static portfolio."""
    if len(prices_df) < lookback_bars + 2:
        raise ValueError("Prices must have more rows than lookback_bars.")

    returns = prices_df.pct_change().dropna()
    dates = returns.index[lookback_bars:]

    nav_static = [1.0]
    nav_targeted = [1.0]
    scalars = []

    for t in range(lookback_bars, len(returns)):
        window = returns.iloc[t - lookback_bars : t]
        realized_vol = calculate_portfolio_realized_volatility(window, base_weights, annual_factor)
        vt_res = apply_volatility_targeting(
            base_weights=base_weights,
            realized_vol_ann=realized_vol,
            target_vol_ann=target_vol_ann,
            max_leverage=max_leverage,
        )
        scalars.append(vt_res.vol_scalar)

        bar_ret = returns.iloc[t]
        static_ret = sum(base_weights.get(c, 0.0) * bar_ret[c] for c in returns.columns)
        targeted_ret = sum(vt_res.scaled_weights.get(c, 0.0) * bar_ret[c] for c in returns.columns)

        nav_static.append(nav_static[-1] * (1.0 + static_ret))
        nav_targeted.append(nav_targeted[-1] * (1.0 + targeted_ret))

    s_static = pd.Series(nav_static[1:], index=dates, name="Static Unhedged")
    s_targeted = pd.Series(nav_targeted[1:], index=dates, name="Vol-Targeted")

    # Max Drawdown
    def mdd(series: pd.Series) -> float:
        peak = series.cummax()
        return float(((series - peak) / peak).min() * 100.0)

    return {
        "nav_static": s_static,
        "nav_targeted": s_targeted,
        "mdd_static_pct": round(mdd(s_static), 2),
        "mdd_targeted_pct": round(mdd(s_targeted), 2),
        "return_static_pct": round((s_static.iloc[-1] - 1.0) * 100.0, 2),
        "return_targeted_pct": round((s_targeted.iloc[-1] - 1.0) * 100.0, 2),
        "mean_scalar": round(float(np.mean(scalars)), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Volatility Targeting Allocator.")
    parser.add_argument("--target-vol", type=float, default=0.30, help="Target annualized volatility (e.g. 0.30 = 30%).")
    parser.add_argument("--realized-vol", type=float, default=0.60, help="Current realized volatility (e.g. 0.60 = 60%).")
    args = parser.parse_args()

    base_w = {"BTC/USDT": 0.60, "ETH/USDT": 0.40}
    res = apply_volatility_targeting(base_w, realized_vol_ann=args.realized_vol, target_vol_ann=args.target_vol)

    print("================ Constant Volatility Targeting ================")
    print(f"  Target Volatility    : {res.target_vol_ann * 100:.1f}%")
    print(f"  Realized Volatility  : {res.realized_vol_ann * 100:.1f}%")
    print(f"  Volatility Scalar    : {res.vol_scalar:.3f}x")
    print(f"  Cash / USDT Buffer   : {res.cash_weight * 100:.1f}%")
    print(f"  Scaled Asset Weights : {res.scaled_weights}")
    print("================================================================")


if __name__ == "__main__":
    main()
