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
    MarketDataUnavailableError,
    find_freqtrade_data_dirs,
    generate_synthetic_crypto_data,
    load_market_data,
    load_from_feather_dir,
    positive_float,
)


def compute_black_litterman_weights(
    prices: pd.DataFrame,
    views: list[str] | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    prior_weights: dict[str, float] | None = None,
) -> dict[str, object]:
    """
    Compute Black-Litterman posterior expected returns and optimal weights.

    Parameters:
    - prices: Historical Close prices
    - views: List of view strings, e.g. ["BTC/USDT>ETH/USDT:0.05", "SOL/USDT:0.10"]
    - tau: Scalar representing uncertainty in prior estimate (default: 0.05)
    - risk_aversion: Risk aversion parameter lambda (default: 2.5)
    - prior_weights: Optional benchmark / market equilibrium weights (default: equal weights)
    """
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] == 0 or len(prices) < 2:
        raise ValueError("Prices must contain at least one asset and two rows.")
    try:
        price_values = prices.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Prices must contain only numeric values.") from error
    if not np.all(np.isfinite(price_values)) or np.any(price_values <= 0):
        raise ValueError("Prices must contain only finite, strictly positive values.")
    if isinstance(tau, bool) or not isinstance(tau, (int, float, np.number)) or not np.isfinite(tau) or tau <= 0:
        raise ValueError("Tau must be a finite, strictly positive number.")
    if isinstance(risk_aversion, bool) or not isinstance(risk_aversion, (int, float, np.number)) or not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("Risk aversion must be a finite, strictly positive number.")

    returns = prices.pct_change().dropna()
    assets = list(returns.columns)
    n = len(assets)

    # Historical covariance matrix
    sigma = returns.cov().values

    # 1. Market Prior (Equal weight benchmark if custom prior not supplied)
    if prior_weights is not None:
        if not isinstance(prior_weights, dict) or not prior_weights:
            raise ValueError("Prior weights must be a non-empty asset-to-weight mapping.")
        try:
            parsed_prior = {asset: float(weight) for asset, weight in prior_weights.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("Prior weights must contain numeric values.") from error
        unknown_assets = set(parsed_prior).difference(assets)
        if unknown_assets:
            raise ValueError(f"Prior weights contain unknown assets: {sorted(unknown_assets)}")
        if any(not isinstance(asset, str) or not np.isfinite(weight) or weight < 0 for asset, weight in parsed_prior.items()):
            raise ValueError("Prior weights must be finite and non-negative.")
        w_prior = np.array([parsed_prior.get(asset, 0.0) for asset in assets], dtype=float)
        if w_prior.sum() <= 0:
            raise ValueError("Prior weights must have a positive total.")
        w_prior = w_prior / w_prior.sum()
    else:
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

    for view in views:
        if not isinstance(view, str) or not view.strip():
            raise ValueError("Views must be non-empty strings.")
        view = view.strip()
        try:
            if ":" in view:
                expr, val_str = view.rsplit(":", 1)
                value = float(val_str)
            else:
                expr, value = view, 0.05
        except ValueError as error:
            raise ValueError(f"Invalid view value: {view}") from error
        if not np.isfinite(value):
            raise ValueError(f"View value must be finite: {view}")

        p_row = np.zeros(n)
        if ">" in expr:
            parts = expr.split(">")
            if len(parts) != 2:
                raise ValueError(f"Invalid relative view format: {view}")
            asset_a, asset_b = (part.strip() for part in parts)
            if asset_a not in assets or asset_b not in assets or asset_a == asset_b:
                raise ValueError(f"Relative view contains invalid assets: {view}")
            p_row[assets.index(asset_a)] = 1.0
            p_row[assets.index(asset_b)] = -1.0
        else:
            asset = expr.strip()
            if asset not in assets:
                raise ValueError(f"Absolute view contains an unknown asset: {view}")
            p_row[assets.index(asset)] = 1.0
        p_rows.append(p_row)
        q_vals.append(value)

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


def parse_cli_prior_weights(raw_items: list[str]) -> dict[str, float]:
    """Parse key:value strings from CLI into an asset prior weights dictionary."""
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("prior_weights must be a non-empty list of 'SYMBOL:WEIGHT' strings.")
    weights: dict[str, float] = {}
    for item in raw_items:
        if not isinstance(item, str) or ":" not in item:
            raise ValueError(f"Prior weight must be in 'SYMBOL:WEIGHT' format, got: {item}")
        pair, w_str = item.split(":", 1)
        pair = pair.strip()
        if not pair:
            raise ValueError(f"Asset symbol cannot be empty in: {item}")
        try:
            val = float(w_str)
        except ValueError as err:
            raise ValueError(f"Invalid numeric weight in: {item}") from err
        if not np.isfinite(val) or val < 0:
            raise ValueError(f"Weight must be finite and non-negative in: {item}")
        weights[pair] = val
    return weights


def main():
    parser = argparse.ArgumentParser(description="Crypto Black-Litterman Optimization Engine")
    parser.add_argument("--data-dir", type=str, default="", help="Directory containing Freqtrade feather files")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe")
    parser.add_argument("--views", nargs="+", default=["BTC/USDT>ETH/USDT:0.02"], help="정성적 전망(Views) 리스트 (상대전망: BTC/USDT>ETH/USDT:0.02, 절대전망: SOL/USDT:0.05)")
    parser.add_argument("--tau", type=positive_float, default=0.05, help="사전 수익률 추정의 불확실성 스케일러 tau (0.01~0.1 권장, 기본: 0.05)")
    parser.add_argument("--risk-aversion", type=positive_float, default=2.5, help="투자자 위험 회피 계수 lambda (1.0 공격적 ~ 5.0 보수적, 기본: 2.5)")
    parser.add_argument("--prior-weights", nargs="+", default=None, help="커스텀 사전 비중 리스트 (예: BTC/USDT:0.6 ETH/USDT:0.4)")
    parser.add_argument("--config-file", type=str, default="", help="비중을 불러올 config.json 또는 allocation JSON 파일 경로")
    parser.add_argument("--use-synthetic", action="store_true", help="Force synthetic sample data")
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

    prior_w = None
    if args.prior_weights:
        try:
            prior_w = parse_cli_prior_weights(args.prior_weights)
        except ValueError as err:
            parser.error(str(err))
    elif args.config_file and Path(args.config_file).is_file():
        try:
            import json
            cfg = json.loads(Path(args.config_file).read_text(encoding="utf-8"))
            prior_w = cfg.get("pair_weights", cfg.get("weights"))
        except Exception as e:
            print(f"[!] Warning: Could not read prior weights from {args.config_file}: {e}")

    res = compute_black_litterman_weights(
        prices=prices,
        views=args.views,
        tau=args.tau,
        risk_aversion=args.risk_aversion,
        prior_weights=prior_w,
    )

    print_black_litterman_report(res)


if __name__ == "__main__":
    main()
