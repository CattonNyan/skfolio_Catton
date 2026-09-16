"""Correlated Multi-Asset Synthetic Crypto Path and Jump-Diffusion Generator.

Generates realistic crypto price paths:
- Correlated multivariate geometric Brownian motion using Cholesky decomposition
- Merton (1976) Jump-Diffusion process to simulate sudden crypto flash crashes and liquidation wicks
- Configurable asset-specific drifts, volatilities, and correlation structure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


def generate_correlated_crypto_paths(
    n_bars: int = 200,
    assets: Sequence[str] = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"),
    initial_prices: Sequence[float] | None = None,
    volatilities: Sequence[float] | None = None,
    correlation_matrix: np.ndarray | None = None,
    annual_drifts: Sequence[float] | None = None,
    jump_intensity: float = 0.02,      # Probability of jump per bar
    jump_mean: float = -0.05,           # Average jump shock (-5% flash crash)
    jump_std: float = 0.04,             # Jump size standard deviation
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate correlated multivariate crypto price series with Merton jump diffusion.

    Parameters
    ----------
    n_bars : int
        Number of time steps / candles to simulate.
    assets : Sequence[str]
        Asset symbol names.
    initial_prices : Sequence[float], optional
        Starting price per asset (defaults to 100.0).
    volatilities : Sequence[float], optional
        Standard deviation per bar for each asset.
    correlation_matrix : np.ndarray, optional
        Target correlation matrix (N x N). If None, an empirical crypto correlation is used.
    annual_drifts : Sequence[float], optional
        Drift return per bar.
    jump_intensity : float
        Poisson jump probability per bar (e.g. 0.02 = 2% chance).
    jump_mean : float
        Mean return shock when a jump occurs (typically negative in crypto flash crashes).
    jump_std : float
        Standard deviation of the jump shock.
    seed : int, optional
        RNG seed for reproducibility.
    """
    if n_bars < 2:
        raise ValueError("n_bars must be at least 2.")
    n_assets = len(assets)
    if n_assets == 0:
        raise ValueError("assets cannot be empty.")
    if jump_intensity < 0.0 or jump_intensity > 1.0:
        raise ValueError("jump_intensity must be between 0.0 and 1.0.")

    rng = np.random.default_rng(seed)

    if initial_prices is None:
        p0 = np.full(n_assets, 100.0)
    else:
        p0 = np.asarray(initial_prices, dtype=float)
        if len(p0) != n_assets or np.any(p0 <= 0):
            raise ValueError("initial_prices must match assets length and be positive.")

    if volatilities is None:
        # Realistic crypto volatilities: BTC ~0.015, ETH ~0.022, Alts ~0.035
        vols = np.linspace(0.015, 0.035, n_assets)
    else:
        vols = np.asarray(volatilities, dtype=float)
        if len(vols) != n_assets or np.any(vols <= 0):
            raise ValueError("volatilities must match assets length and be strictly positive.")

    if annual_drifts is None:
        drifts = np.full(n_assets, 0.0005)
    else:
        drifts = np.asarray(annual_drifts, dtype=float)
        if len(drifts) != n_assets:
            raise ValueError("annual_drifts length must match assets.")

    if correlation_matrix is None:
        # Typical crypto correlation: baseline ~0.50 off-diagonal
        corr = np.full((n_assets, n_assets), 0.50)
        np.fill_diagonal(corr, 1.0)
    else:
        corr = np.asarray(correlation_matrix, dtype=float)
        if corr.shape != (n_assets, n_assets):
            raise ValueError("correlation_matrix shape must match (n_assets, n_assets).")

    # Regularize correlation matrix to ensure positive definiteness
    corr_reg = corr + np.eye(n_assets) * 1e-7
    try:
        chol_l = np.linalg.cholesky(corr_reg)
    except np.linalg.LinAlgError:
        # Fallback to eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-6)
        corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        inv_diag = 1.0 / np.sqrt(np.diag(corr_psd))
        corr_psd = corr_psd * inv_diag[:, None] * inv_diag[None, :]
        chol_l = np.linalg.cholesky(corr_psd)

    # 1. Generate standard correlated Brownian innovations
    uncorrelated_z = rng.standard_normal((n_bars, n_assets))
    correlated_z = uncorrelated_z @ chol_l.T

    # 2. Add Merton Poisson Jump-Diffusion
    jump_mask = rng.uniform(0.0, 1.0, (n_bars, n_assets)) < jump_intensity
    jump_magnitudes = rng.normal(jump_mean, jump_std, (n_bars, n_assets))
    jumps = jump_mask * jump_magnitudes

    # Log returns: r_t = (drift - 0.5 * vol^2) + vol * Z + jumps
    log_returns = (drifts - 0.5 * (vols ** 2)) + (vols * correlated_z) + jumps

    # Cumulative compounding
    price_paths = np.empty((n_bars, n_assets), dtype=float)
    price_paths[0] = p0
    for t in range(1, n_bars):
        price_paths[t] = price_paths[t - 1] * np.exp(log_returns[t])

    dates = pd.date_range(start="2026-01-01", periods=n_bars, freq="1h")
    df = pd.DataFrame(price_paths, index=dates, columns=list(assets))
    return df


def main():
    parser = argparse.ArgumentParser(description="Correlated Synthetic Crypto Generator.")
    parser.add_argument("--bars", type=int, default=150, help="Number of bars.")
    parser.add_argument("--jump-prob", type=float, default=0.03, help="Flash crash jump probability.")
    args = parser.parse_args()

    df = generate_correlated_crypto_paths(n_bars=args.bars, jump_intensity=args.jump_prob)
    print(f"[+] Generated synthetic crypto data: {df.shape}")
    print(f"[*] Correlation matrix:\n{df.pct_change().dropna().corr().round(3)}")
    print(f"[*] Summary Statistics:\n{df.describe().round(2)}")


if __name__ == "__main__":
    main()
