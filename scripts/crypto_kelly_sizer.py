"""Kelly Criterion and Fractional-Kelly Position Sizer for Crypto Portfolios.

Provides optimal capital growth position sizing:
- Discrete Kelly formula (win rate, payoff ratio)
- Continuous Kelly formula (mean excess return / variance)
- Multi-Asset Kelly allocator (unconstrained and constrained)
- Half-Kelly / Fractional Kelly scaling to curb crypto tail risk drawdown
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
class KellyResult:
    """Kelly calculation results container."""
    full_kelly: float
    fractional_kelly: float
    fraction: float
    expected_growth_rate: float
    half_kelly: float
    is_positive_edge: bool


def calculate_discrete_kelly(
    win_rate: float,
    payoff_ratio: float,
    fraction: float = 0.5,
    max_allocation: float = 1.0,
) -> KellyResult:
    """Calculate discrete Kelly fraction: f* = (p * b - (1 - p)) / b.

    Parameters
    ----------
    win_rate : float
        Probability of winning trade (0.0 < p < 1.0).
    payoff_ratio : float
        Ratio of average win to average loss (b > 0).
    fraction : float
        Kelly fraction multiplier (0.5 for Half-Kelly).
    max_allocation : float
        Maximum upper bound cap on position size.
    """
    if not (0.0 < win_rate < 1.0):
        raise ValueError("Win rate must be strictly between 0 and 1.")
    if payoff_ratio <= 0.0:
        raise ValueError("Payoff ratio must be strictly positive.")
    if fraction <= 0.0:
        raise ValueError("Fraction must be strictly positive.")
    if max_allocation <= 0.0:
        raise ValueError("Max allocation must be strictly positive.")

    loss_rate = 1.0 - win_rate
    full_k = (win_rate * payoff_ratio - loss_rate) / payoff_ratio

    is_positive = full_k > 0.0
    full_k_clamped = max(0.0, min(float(full_k), max_allocation))
    frac_k = max(0.0, min(float(full_k * fraction), max_allocation))
    half_k = max(0.0, min(float(full_k * 0.5), max_allocation))

    # Geometric growth rate g(f) = p * ln(1 + b*f) + (1-p) * ln(1 - f)
    growth = 0.0
    if is_positive and frac_k < 1.0:
        growth = win_rate * np.log(1.0 + payoff_ratio * frac_k) + loss_rate * np.log(max(1e-9, 1.0 - frac_k))

    return KellyResult(
        full_kelly=full_k_clamped,
        fractional_kelly=frac_k,
        fraction=fraction,
        expected_growth_rate=float(growth),
        half_kelly=half_k,
        is_positive_edge=is_positive,
    )


def calculate_continuous_kelly(
    mean_return: float,
    volatility: float,
    risk_free_rate: float = 0.0,
    fraction: float = 0.5,
    max_allocation: float = 1.0,
) -> KellyResult:
    """Calculate continuous Gaussian Kelly fraction: f* = (mu - r) / sigma^2.

    Parameters
    ----------
    mean_return : float
        Expected annualized or per-period return.
    volatility : float
        Annualized or per-period return standard deviation.
    risk_free_rate : float
        Risk-free rate per period.
    fraction : float
        Kelly scaling fraction (0.5 for Half-Kelly).
    max_allocation : float
        Cap on leverage / position allocation.
    """
    if volatility <= 0.0:
        raise ValueError("Volatility must be strictly positive.")
    if fraction <= 0.0:
        raise ValueError("Fraction must be strictly positive.")
    if max_allocation <= 0.0:
        raise ValueError("Max allocation must be strictly positive.")

    excess = mean_return - risk_free_rate
    variance = volatility ** 2
    full_k = excess / variance

    is_positive = full_k > 0.0
    full_k_clamped = max(0.0, min(float(full_k), max_allocation))
    frac_k = max(0.0, min(float(full_k * fraction), max_allocation))
    half_k = max(0.0, min(float(full_k * 0.5), max_allocation))

    # Growth rate g(f) = r + f*(mu - r) - 0.5 * f^2 * sigma^2
    growth = risk_free_rate + frac_k * excess - 0.5 * (frac_k ** 2) * variance

    return KellyResult(
        full_kelly=full_k_clamped,
        fractional_kelly=frac_k,
        fraction=fraction,
        expected_growth_rate=float(growth),
        half_kelly=half_k,
        is_positive_edge=is_positive,
    )


def calculate_portfolio_kelly(
    returns_df: pd.DataFrame,
    fraction: float = 0.5,
    max_total_weight: float = 1.0,
) -> pd.Series:
    """Calculate multi-asset unconstrained continuous Kelly weights: w* = fraction * Sigma^-1 * mu.

    Normalizes weights to max_total_weight if sum exceeds boundary.
    """
    if returns_df.empty or len(returns_df) < 2:
        raise ValueError("Returns DataFrame must have at least 2 observations.")
    if fraction <= 0.0:
        raise ValueError("Fraction must be strictly positive.")
    if max_total_weight <= 0.0:
        raise ValueError("Max total weight must be strictly positive.")

    mu = returns_df.mean().to_numpy()
    sigma = returns_df.cov().to_numpy()

    # Regularize covariance to avoid inversion singularity
    reg_sigma = sigma + np.eye(len(mu)) * 1e-6
    try:
        inv_sigma = np.linalg.pinv(reg_sigma)
        raw_weights = inv_sigma @ mu
    except Exception:
        raw_weights = np.ones(len(mu)) / len(mu)

    # Apply long-only non-negative constraint
    raw_weights = np.maximum(0.0, raw_weights)
    scaled_weights = raw_weights * fraction

    total = np.sum(scaled_weights)
    if total > max_total_weight and total > 0:
        scaled_weights = (scaled_weights / total) * max_total_weight

    return pd.Series(scaled_weights, index=returns_df.columns, name="Kelly_Weight")


def main():
    parser = argparse.ArgumentParser(description="Crypto Kelly Criterion Position Sizer.")
    parser.add_argument("--win-rate", type=float, default=0.55, help="Strategy win rate (e.g. 0.55)")
    parser.add_argument("--payoff", type=float, default=1.8, help="Payoff ratio (win/loss ratio)")
    parser.add_argument("--fraction", type=float, default=0.5, help="Fractional Kelly multiplier (default 0.5)")
    args = parser.parse_args()

    res = calculate_discrete_kelly(args.win_rate, args.payoff, fraction=args.fraction)
    print("================ Crypto Kelly Criterion Sizing ================")
    print(f"  Win Rate           : {args.win_rate * 100:.1f}%")
    print(f"  Payoff Ratio       : {args.payoff:.2f}x")
    print(f"  Full Kelly (f*)    : {res.full_kelly * 100:.2f}%")
    print(f"  Half Kelly (0.5x)  : {res.half_kelly * 100:.2f}%")
    print(f"  Chosen Frac ({args.fraction}x): {res.fractional_kelly * 100:.2f}%")
    print(f"  Exp. Growth Rate   : {res.expected_growth_rate * 100:.3f}% per trade")
    print(f"  Positive Edge      : {'YES' if res.is_positive_edge else 'NO'}")
    print("================================================================")


if __name__ == "__main__":
    main()
