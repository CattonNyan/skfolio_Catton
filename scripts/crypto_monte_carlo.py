"""Cryptocurrency Portfolio Monte Carlo Future Path Simulator.

Simulates forward-looking asset trajectories using Geometric Brownian Motion (GBM):
- Simulates 1,000+ stochastic price paths over N days.
- Computes Value at Risk (VaR 95% & 99%) and Conditional VaR (CVaR).
- Evaluates Probability of Capital Loss and maximum expected drawdown cone.
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


def simulate_monte_carlo_paths(
    prices: pd.DataFrame,
    weights: dict[str, float],
    days: int = 90,
    num_simulations: int = 1000,
    initial_capital: float = 10000.0,
    seed: int = 42,
) -> dict[str, object]:
    """
    Simulate future portfolio paths using Geometric Brownian Motion (GBM).

    Returns summary metrics, percentile cones, and risk distributions.
    """
    if initial_capital <= 0:
        raise ValueError("Initial capital must be strictly positive.")

    returns = prices.pct_change().dropna()
    common_assets = [a for a in weights if a in returns.columns]
    if not common_assets:
        raise ValueError("No matching assets found between prices and weights.")

    # Normalize weights over common assets
    raw_weights = np.array([weights[a] for a in common_assets])
    w = raw_weights / raw_weights.sum()

    # Portfolio historical daily/bar returns
    port_returns = np.dot(returns[common_assets].values, w)
    mu = float(np.mean(port_returns))
    sigma = float(np.std(port_returns) + 1e-9)

    # Detect candle frequency if DatetimeIndex to scale parameters to daily horizon
    bars_per_day = 1.0
    if isinstance(prices.index, pd.DatetimeIndex) and len(prices.index) > 1:
        try:
            diffs = prices.index.to_series().diff().dropna()
            median_seconds = float(diffs.dt.total_seconds().median())
            if 0 < median_seconds < 86400:
                bars_per_day = 86400.0 / median_seconds
        except Exception:
            pass

    daily_mu = mu * bars_per_day
    daily_sigma = sigma * np.sqrt(bars_per_day)

    # Drift and shock components for GBM (daily step)
    dt = 1.0
    drift = (daily_mu - 0.5 * (daily_sigma**2)) * dt
    vol = daily_sigma * np.sqrt(dt)

    np.random.seed(seed)
    # Generate random shocks: shape = (num_simulations, days)
    random_shocks = np.random.normal(0, 1, size=(num_simulations, days))
    step_returns = np.exp(drift + vol * random_shocks)

    # Cumulative wealth paths starting from initial_capital
    paths = np.zeros((num_simulations, days + 1))
    paths[:, 0] = initial_capital
    for t in range(1, days + 1):
        paths[:, t] = paths[:, t - 1] * step_returns[:, t - 1]

    # Final wealth distribution at day T
    final_wealth = paths[:, -1]
    net_profits = final_wealth - initial_capital
    returns_pct = (final_wealth / initial_capital) - 1.0

    # Risk Metrics
    losses = initial_capital - final_wealth
    var_95 = float(np.percentile(losses, 95))
    var_99 = float(np.percentile(losses, 99))
    cvar_candidates = [loss for loss in losses if loss >= var_95]
    cvar_95 = float(np.mean(cvar_candidates)) if len(cvar_candidates) > 0 else max(0.0, var_95)
    if np.isnan(cvar_95):
        cvar_95 = max(0.0, var_95)

    prob_loss = float(np.mean(final_wealth < initial_capital) * 100)
    prob_severe_loss = float(np.mean(final_wealth < initial_capital * 0.70) * 100)
    prob_doubling = float(np.mean(final_wealth >= initial_capital * 2.0) * 100)

    p05_path = np.percentile(paths, 5, axis=0).tolist()
    p50_path = np.percentile(paths, 50, axis=0).tolist()
    p95_path = np.percentile(paths, 95, axis=0).tolist()

    return {
        "days": days,
        "num_simulations": num_simulations,
        "initial_capital": initial_capital,
        "expected_final_wealth": round(float(np.mean(final_wealth)), 2),
        "median_final_wealth": round(float(np.median(final_wealth)), 2),
        "worst_case_5pct": round(float(np.percentile(final_wealth, 5)), 2),
        "best_case_95pct": round(float(np.percentile(final_wealth, 95)), 2),
        "var_95_dollar": round(max(0.0, var_95), 2),
        "var_99_dollar": round(max(0.0, var_99), 2),
        "cvar_95_dollar": round(max(0.0, cvar_95), 2),
        "prob_loss_pct": round(prob_loss, 2),
        "prob_severe_loss_pct": round(prob_severe_loss, 2),
        "prob_doubling_pct": round(prob_doubling, 2),
        "path_p05": [round(x, 2) for x in p05_path],
        "path_p50": [round(x, 2) for x in p50_path],
        "path_p95": [round(x, 2) for x in p95_path],
    }


# Backwards compatibility and dashboard integration alias
simulate_monte_carlo = simulate_monte_carlo_paths


def print_monte_carlo_report(res: dict[str, object]):
    """Print terminal report of Monte Carlo simulation."""
    print("================================================================================")
    print(f"        MONTE CARLO FORWARD-LOOKING RISK SIMULATION ({res['days']} Days, {res['num_simulations']:,} Paths)      ")
    print("================================================================================")
    print(f"{'Initial Capital':<26}: ${res['initial_capital']:,.2f}")
    print(f"{'Expected Final Wealth':<26}: ${res['expected_final_wealth']:,.2f}")
    print(f"{'Median Final Wealth':<26}: ${res['median_final_wealth']:,.2f}")
    print(f"{'95% Confidence Band':<26}: ${res['worst_case_5pct']:,.2f} ~ ${res['best_case_95pct']:,.2f}")
    print("--------------------------------------------------------------------------------")
    print("                      DOWNSIDE RISK & PROBABILITY PROFILE                       ")
    print("--------------------------------------------------------------------------------")
    print(f"{'Value at Risk (VaR 95%)':<26}: ${res['var_95_dollar']:,.2f} (최대 5% 확률 손실액)")
    print(f"{'Value at Risk (VaR 99%)':<26}: ${res['var_99_dollar']:,.2f} (최대 1% 극단 손실액)")
    print(f"{'Conditional VaR (CVaR 95%)':<26}: ${res['cvar_95_dollar']:,.2f} (95% 초과 손실 시 평균 손실액)")
    print(f"{'원금 손실 확률 (Prob Loss)':<24}: {res['prob_loss_pct']:.2f}%")
    print(f"{'원금 30% 이상 폭락 확률':<24}: {res['prob_severe_loss_pct']:.2f}%")
    print(f"{'원금 2배 달성 확률':<25}: {res['prob_doubling_pct']:.2f}%")
    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Crypto Monte Carlo Portfolio Simulator")
    parser.add_argument("--days", type=int, default=90, help="Future simulation horizon in days")
    parser.add_argument("--sims", type=int, default=1000, help="Number of simulated paths")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--export-json", type=str, default="", help="Path to export JSON metrics")
    parser.add_argument("--use-synthetic", action="store_true", help="Force synthetic data")
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
        prices = generate_synthetic_crypto_data(periods=500)

    weights = {"BTC/USDT": 0.50, "ETH/USDT": 0.30, "SOL/USDT": 0.20}
    res = simulate_monte_carlo_paths(
        prices=prices,
        weights=weights,
        days=args.days,
        num_simulations=args.sims,
        initial_capital=args.capital,
    )

    print_monte_carlo_report(res)

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[+] Monte Carlo metrics exported to: {out_path}")


if __name__ == "__main__":
    main()
