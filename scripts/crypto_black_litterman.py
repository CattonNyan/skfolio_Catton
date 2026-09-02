"""Black-Litterman Portfolio Optimization Engine for Cryptocurrencies.

Combines market equilibrium returns (prior) with subjective investor views
using Bayesian statistics to compute robust, stabilized portfolio weights.
"""

from __future__ import annotations

import argparse
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


def compute_black_litterman_weights(
    prices: pd.DataFrame,
    views: list[str] | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> dict[str, object]:
    """
    Compute Black-Litterman posterior expected returns and optimal weights.

    Parameters:
    - prices: Historical Close prices
    - views: List of view strings, e.g. ["BTC/USDT>ETH/USDT:0.05", "SOL/USDT:0.10"]
    - tau: Scalar representing uncertainty in prior estimate (default: 0.05)
    - risk_aversion: Risk aversion parameter lambda (default: 2.5)
    """
    returns = prices.pct_change().dropna()
    assets = list(returns.columns)
    n = len(assets)

    # Historical covariance matrix
    sigma = returns.cov().values

    # 1. Market Prior (Equal weight benchmark if market-cap unavailable)
    w_prior = np.ones(n) / n
    pi = risk_aversion * np.dot(sigma, w_prior)

    if not views:
        # Without views, Black-Litterman collapses to the market equilibrium
        return {
            "prior_weights": dict(zip(assets, w_prior)),
            "posterior_weights": dict(zip(assets, w_prior)),
            "implied_returns": dict(zip(assets, pi)),
            "posterior_returns": dict(zip(assets, pi)),
        }

    # 2. Parse Views into P (Pick matrix) and Q (View vector)
    p_rows = []
    q_vals = []

    for v in views:
        v = v.strip()
        if not v:
            continue
        try:
            if ":" in v:
                expr, val_str = v.split(":")
                val = float(val_str)
            else:
                expr, val = v, 0.05

            p_row = np.zeros(n)
            if ">" in expr:
                # Relative view: AssetA > AssetB by val
                a, b = expr.split(">")
                a, b = a.strip(), b.strip()
                if a in assets and b in assets:
                    p_row[assets.index(a)] = 1.0
                    p_row[assets.index(b)] = -1.0
                    p_rows.append(p_row)
                    q_vals.append(val)
            else:
                # Absolute view: AssetA by val
                a = expr.strip()
                if a in assets:
                    p_row[assets.index(a)] = 1.0
                    p_rows.append(p_row)
                    q_vals.append(val)
        except Exception as e:
            print(f"[!] Warning: Could not parse view '{v}': {e}")

    if not p_rows:
        return {
            "prior_weights": dict(zip(assets, w_prior)),
            "posterior_weights": dict(zip(assets, w_prior)),
            "implied_returns": dict(zip(assets, pi)),
            "posterior_returns": dict(zip(assets, pi)),
        }

    P = np.array(p_rows)
    Q = np.array(q_vals)
    k = len(Q)

    # 3. View uncertainty matrix Omega (He & Litterman specification: diag(P * (tau * Sigma) * P^T))
    omega = np.diag(np.diag(P @ (tau * sigma) @ P.T))
    # Ensure positive definiteness
    omega += np.eye(k) * 1e-8

    # 4. Black-Litterman Master Formula for Posterior Returns
    # E(R) = [ (tau * Sigma)^-1 + P^T * Omega^-1 * P ]^-1 * [ (tau * Sigma)^-1 * Pi + P^T * Omega^-1 * Q ]
    tau_sigma_inv = np.linalg.pinv(tau * sigma)
    omega_inv = np.linalg.pinv(omega)

    m_inv = np.linalg.pinv(tau_sigma_inv + P.T @ omega_inv @ P)
    er_posterior = m_inv @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

    # 5. Optimal Posterior Weights (Unconstrained Mean-Variance solution, then normalized to sum to 1)
    sigma_inv = np.linalg.pinv(sigma)
    raw_weights = (1.0 / risk_aversion) * sigma_inv @ er_posterior

    # Long-only projection (non-negative clipping)
    w_post = np.clip(raw_weights, 0.0, None)
    fallback_used = False
    if w_post.sum() > 0:
        w_post = w_post / w_post.sum()
    else:
        print("[!] Warning: All posterior weights were non-positive due to extreme bearish views. Falling back to prior weights.")
        w_post = w_prior
        fallback_used = True

    return {
        "prior_weights": dict(zip(assets, w_prior)),
        "posterior_weights": dict(zip(assets, w_post)),
        "implied_returns": dict(zip(assets, pi)),
        "posterior_returns": dict(zip(assets, er_posterior)),
        "fallback_to_prior": fallback_used,
    }


def print_black_litterman_report(res: dict[str, object]):
    """Print comparative table between Prior and Black-Litterman Posterior."""
    print("================================================================================")
    print("             BLACK-LITTERMAN BAYESIAN ASSET ALLOCATION REPORT                   ")
    print("================================================================================")
    print(f"{'Asset':<15} | {'Prior Weight':>14} | {'BL Post Weight':>16} | {'Delta':>12}")
    print("--------------------------------------------------------------------------------")

    priors = res["prior_weights"]
    posts = res["posterior_weights"]

    for asset in priors:
        w_pri = priors[asset] * 100
        w_pos = posts[asset] * 100
        delta = w_pos - w_pri
        print(f"{asset:<15} | {w_pri:>13.2f}% | {w_pos:>15.2f}% | {delta:>+11.2f}%")

    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Crypto Black-Litterman Optimization Engine")
    parser.add_argument("--data-dir", type=str, default="", help="Directory containing Freqtrade feather files")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe")
    parser.add_argument("--views", nargs="+", default=["BTC/USDT>ETH/USDT:0.02"], help="Views list (e.g. BTC/USDT>ETH/USDT:0.02 SOL/USDT:0.05)")
    parser.add_argument("--tau", type=float, default=0.05, help="Scalar tau uncertainty (default: 0.05)")
    parser.add_argument("--risk-aversion", type=float, default=2.5, help="Risk aversion lambda (default: 2.5)")
    parser.add_argument("--use-synthetic", action="store_true", help="Force synthetic sample data")
    args = parser.parse_args()

    prices = pd.DataFrame()
    if not args.use_synthetic:
        candidate_dirs = find_freqtrade_data_dirs()
        for d in candidate_dirs:
            if d.is_dir():
                prices = load_from_feather_dir(d, timeframe=args.timeframe)
                if not prices.empty:
                    print(f"[*] Loaded data from {d} ({len(prices)} bars)")
                    break

    if prices.empty or args.use_synthetic:
        prices = generate_synthetic_crypto_data(periods=500)

    res = compute_black_litterman_weights(
        prices=prices,
        views=args.views,
        tau=args.tau,
        risk_aversion=args.risk_aversion,
    )

    print_black_litterman_report(res)


if __name__ == "__main__":
    main()
