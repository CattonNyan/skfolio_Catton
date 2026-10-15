"""Dynamic Rebalancing Corridor & Optimal No-Trade Zone Optimizer for Crypto Portfolios.

Implements volatility-adjusted, proportional, and Leland (1999) transaction-cost-optimal
rebalancing bands (no-trade zones). Simulates and benchmarks multi-asset portfolio execution:
1. Buy & Hold benchmark (zero turnover)
2. Daily Calendar Rebalancing
3. Weekly Calendar Rebalancing (7-day periodic)
4. Monthly Calendar Rebalancing (30-day periodic)
5. Corridor-to-Target Rebalancing (breach resets portfolio to target weights)
6. Corridor-to-Boundary Rebalancing (breach executes minimal intervention back to corridor band)

Calculates transaction cost drag, avoided rebalances, turnover savings, net CAGR, net Sharpe,
Sortino ratio, and Maximum Drawdown (MDD).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

# Ensure local repository root is discoverable
root_dir = str(Path(__file__).resolve().parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd


@dataclass
class CorridorBand:
    """Rebalancing corridor band for an individual crypto asset."""

    asset: str
    target_weight: float
    lower_band: float
    upper_band: float
    band_width: float
    volatility: float
    transaction_cost: float

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


@dataclass
class PolicyResult:
    """Performance and turnover results for a rebalancing policy."""

    policy_name: str
    rebalance_count: int
    rebalance_frequency_pct: float
    total_turnover: float
    annualized_turnover: float
    total_costs_paid: float
    cost_drag_pct: float
    avoided_rebalances_pct: float
    initial_capital: float
    final_nav: float
    total_return_pct: float
    cagr_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


def calculate_proportional_bands(
    target_weights: dict[str, float],
    relative_tolerance: float = 0.20,
    asset_volatilities: dict[str, float] | None = None,
    transaction_costs: dict[str, float] | float = 0.001,
) -> dict[str, CorridorBand]:
    """Calculate proportional corridor bands around target weights.

    Parameters
    ----------
    target_weights : dict[str, float]
        Dictionary mapping asset symbols to target weights (summing to ~1.0).
    relative_tolerance : float, default=0.20
        Relative width tolerance fraction (e.g. 0.20 means +/- 20% of target weight).
    asset_volatilities : dict[str, float], optional
        Annualized volatility per asset.
    transaction_costs : dict[str, float] or float, default=0.001
        Round-trip transaction cost fraction.

    Returns
    -------
    dict[str, CorridorBand]
        Asset-level corridor bands.
    """
    if relative_tolerance <= 0.0:
        raise ValueError(f"relative_tolerance must be positive, got {relative_tolerance}")
    if not target_weights:
        raise ValueError("target_weights cannot be empty")

    total_w = sum(target_weights.values())
    if abs(total_w - 1.0) > 0.05:
        raise ValueError(f"Target weights must sum close to 1.0, got sum = {total_w:.4f}")

    bands: dict[str, CorridorBand] = {}
    for asset, w in target_weights.items():
        if w < 0.0:
            raise ValueError(f"Target weight for {asset} cannot be negative: {w}")
        half_width = w * relative_tolerance
        lower = max(0.0, w - half_width)
        upper = min(1.0, w + half_width)

        vol = asset_volatilities.get(asset, 0.50) if asset_volatilities else 0.50
        cost = (
            transaction_costs.get(asset, 0.001)
            if isinstance(transaction_costs, dict)
            else float(transaction_costs)
        )

        bands[asset] = CorridorBand(
            asset=asset,
            target_weight=round(float(w), 6),
            lower_band=round(float(lower), 6),
            upper_band=round(float(upper), 6),
            band_width=round(float(upper - lower), 6),
            volatility=round(float(vol), 4),
            transaction_cost=round(float(cost), 6),
        )

    return bands


def calculate_volatility_bands(
    target_weights: dict[str, float],
    asset_volatilities: dict[str, float],
    base_cost: float = 0.001,
    kappa: float = 0.15,
) -> dict[str, CorridorBand]:
    """Calculate volatility-adjusted rebalancing corridors.

    Higher-volatility assets have wider price drift; widening bands prevents excessive
    turnover while preserving target exposure.

    Band half-width delta_i = max(0.01, min(0.4 * w_i, kappa * vol_i * sqrt(cost / 0.001)))
    """
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if not target_weights:
        raise ValueError("target_weights cannot be empty")

    total_w = sum(target_weights.values())
    if abs(total_w - 1.0) > 0.05:
        raise ValueError(f"Target weights must sum close to 1.0, got sum = {total_w:.4f}")

    bands: dict[str, CorridorBand] = {}
    for asset, w in target_weights.items():
        if w < 0.0:
            raise ValueError(f"Target weight for {asset} cannot be negative: {w}")
        vol = max(0.05, float(asset_volatilities.get(asset, 0.50)))
        cost_scale = math.sqrt(max(0.0001, base_cost) / 0.001)

        delta = kappa * vol * cost_scale
        # Cap delta to maximum 40% of target weight and at least 0.005 absolute
        half_width = max(0.005, min(0.40 * w, delta * w))

        lower = max(0.0, w - half_width)
        upper = min(1.0, w + half_width)

        bands[asset] = CorridorBand(
            asset=asset,
            target_weight=round(float(w), 6),
            lower_band=round(float(lower), 6),
            upper_band=round(float(upper), 6),
            band_width=round(float(upper - lower), 6),
            volatility=round(float(vol), 4),
            transaction_cost=round(float(base_cost), 6),
        )

    return bands


def calculate_leland_cost_optimal_bands(
    target_weights: dict[str, float],
    asset_volatilities: dict[str, float],
    transaction_costs: dict[str, float] | float = 0.001,
    risk_aversion: float = 3.0,
) -> dict[str, CorridorBand]:
    """Calculate Leland (1999) transaction-cost optimal no-trade bands.

    Analytically solves the optimal no-trade band around target weight under
    proportional transaction costs and constant relative risk aversion (CRRA):
        delta_i = [ (3 / 4) * (cost_i * vol_i^2 * w_i^2) / gamma ]^(1/3)
    """
    if risk_aversion <= 0.0:
        raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")
    if not target_weights:
        raise ValueError("target_weights cannot be empty")

    bands: dict[str, CorridorBand] = {}
    for asset, w in target_weights.items():
        if w < 0.0:
            raise ValueError(f"Target weight for {asset} cannot be negative: {w}")

        cost = (
            transaction_costs.get(asset, 0.001)
            if isinstance(transaction_costs, dict)
            else float(transaction_costs)
        )
        vol = max(0.05, float(asset_volatilities.get(asset, 0.50)))

        # Leland cubic-root formulation
        numerator = 0.75 * cost * (vol**2) * (w**2)
        half_width = (numerator / risk_aversion) ** (1.0 / 3.0)

        # Clamping to reasonable bounds [0.05*w, 0.45*w]
        half_width = max(0.005, min(0.45 * w, half_width))

        lower = max(0.0, w - half_width)
        upper = min(1.0, w + half_width)

        bands[asset] = CorridorBand(
            asset=asset,
            target_weight=round(float(w), 6),
            lower_band=round(float(lower), 6),
            upper_band=round(float(upper), 6),
            band_width=round(float(upper - lower), 6),
            volatility=round(float(vol), 4),
            transaction_cost=round(float(cost), 6),
        )

    return bands


def _compute_performance_metrics(
    nav_series: list[float],
    rebalance_count: int,
    total_days: int,
    total_turnover: float,
    total_costs: float,
    initial_capital: float,
    policy_name: str,
    baseline_rebalance_count: int = 0,
    annual_trading_days: int = 365,
) -> PolicyResult:
    """Compute risk, return, and turnover metrics for a portfolio NAV time series."""
    navs = np.array(nav_series, dtype=np.float64)
    final_nav = float(navs[-1])
    total_return_pct = float(((final_nav / initial_capital) - 1.0) * 100.0)

    years = max(1.0 / annual_trading_days, total_days / annual_trading_days)
    if final_nav > 0 and initial_capital > 0:
        cagr_pct = float(((final_nav / initial_capital) ** (1.0 / years) - 1.0) * 100.0)
    else:
        cagr_pct = -100.0

    # Daily returns
    daily_rets = np.diff(navs) / navs[:-1]
    if len(daily_rets) > 1 and np.std(daily_rets) > 1e-12:
        vol_pct = float(np.std(daily_rets, ddof=1) * np.sqrt(annual_trading_days) * 100.0)
        mean_annual_ret = float(np.mean(daily_rets) * annual_trading_days)
        sharpe = float(mean_annual_ret / (vol_pct / 100.0))

        # Sortino (downside deviation)
        downside = daily_rets[daily_rets < 0.0]
        if len(downside) > 1:
            downside_vol = float(np.sqrt(np.mean(downside**2)) * np.sqrt(annual_trading_days))
            sortino = float(mean_annual_ret / downside_vol) if downside_vol > 1e-8 else 0.0
        else:
            sortino = 999.0 if mean_annual_ret > 0 else 0.0
    else:
        vol_pct = 0.0
        sharpe = 0.0
        sortino = 0.0

    # Maximum Drawdown (MDD)
    running_max = np.maximum.accumulate(navs)
    drawdowns = (navs - running_max) / running_max
    max_dd_pct = float(abs(np.min(drawdowns)) * 100.0)

    calmar = (
        float(cagr_pct / max_dd_pct)
        if max_dd_pct > 1e-6
        else (999.0 if cagr_pct > 0 else 0.0)
    )

    reb_freq_pct = float((rebalance_count / max(1, total_days)) * 100.0)
    ann_turnover = float(total_turnover / max(1e-4, years))
    cost_drag_pct = float((total_costs / initial_capital) * 100.0)

    avoided_pct = (
        float(
            ((baseline_rebalance_count - rebalance_count) / baseline_rebalance_count)
            * 100.0
        )
        if baseline_rebalance_count > 0
        else 0.0
    )

    return PolicyResult(
        policy_name=policy_name,
        rebalance_count=int(rebalance_count),
        rebalance_frequency_pct=round(reb_freq_pct, 2),
        total_turnover=round(float(total_turnover), 4),
        annualized_turnover=round(ann_turnover, 4),
        total_costs_paid=round(float(total_costs), 2),
        cost_drag_pct=round(cost_drag_pct, 2),
        avoided_rebalances_pct=round(avoided_pct, 2),
        initial_capital=round(float(initial_capital), 2),
        final_nav=round(final_nav, 2),
        total_return_pct=round(total_return_pct, 2),
        cagr_pct=round(cagr_pct, 2),
        annualized_volatility_pct=round(vol_pct, 2),
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        calmar_ratio=round(calmar, 2),
    )


def simulate_corridor_rebalancing(
    prices_df: pd.DataFrame,
    target_weights: dict[str, float],
    bands: dict[str, CorridorBand],
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    initial_capital: float = 10000.0,
    annual_trading_days: int = 365,
) -> dict[str, Any]:
    """Simulate walk-forward multi-asset portfolio rebalancing across multiple policies.

    Policies compared:
    1. BuyAndHold: No rebalancing after day 0
    2. DailyCalendar: Rebalance to target every day
    3. WeeklyCalendar: Rebalance to target every 7 days
    4. MonthlyCalendar: Rebalance to target every 30 days
    5. CorridorToTarget: Rebalance to target weights only when a band is breached
    6. CorridorToBoundary: Minimal intervention; rebalance breached assets only back to boundary
    """
    if prices_df.empty or len(prices_df) < 5:
        raise ValueError("prices_df must contain at least 5 price observations")
    for asset in target_weights:
        if asset not in prices_df.columns:
            raise KeyError(f"Asset '{asset}' missing from prices_df columns: {list(prices_df.columns)}")

    assets = list(target_weights.keys())
    prices = prices_df[assets].to_numpy(dtype=np.float64)
    n_days = len(prices)

    cost_multiplier = fee_rate + slippage_rate

    # First run Daily Calendar to establish baseline rebalance count
    daily_reb_count = n_days - 1

    policies_results: dict[str, PolicyResult] = {}
    nav_histories: dict[str, list[float]] = {}

    policy_specs = [
        "BuyAndHold",
        "DailyCalendar",
        "WeeklyCalendar",
        "MonthlyCalendar",
        "CorridorToTarget",
        "CorridorToBoundary",
    ]

    for policy in policy_specs:
        # Initialize Day 0
        capital = initial_capital
        p0 = prices[0]
        # Initial purchase costs
        init_costs = capital * cost_multiplier
        capital -= init_costs
        quantities = np.array(
            [(capital * target_weights[a]) / p0[i] for i, a in enumerate(assets)],
            dtype=np.float64,
        )

        nav_history = [initial_capital]
        reb_count = 0
        total_turnover = 0.0
        total_costs = init_costs

        for t in range(1, n_days):
            current_prices = prices[t]
            asset_values = quantities * current_prices
            curr_nav = float(np.sum(asset_values))

            if curr_nav <= 0.0:
                nav_history.append(0.0)
                continue

            current_weights = asset_values / curr_nav

            should_rebalance = False
            target_w_step = np.array([target_weights[a] for a in assets], dtype=np.float64)

            if policy == "BuyAndHold":
                should_rebalance = False
            elif policy == "DailyCalendar":
                should_rebalance = True
            elif policy == "WeeklyCalendar":
                should_rebalance = (t % 7 == 0)
            elif policy == "MonthlyCalendar":
                should_rebalance = (t % 30 == 0)
            elif policy == "CorridorToTarget":
                # Check if ANY asset breached its band
                for i, a in enumerate(assets):
                    band = bands[a]
                    if (
                        current_weights[i] < band.lower_band
                        or current_weights[i] > band.upper_band
                    ):
                        should_rebalance = True
                        break
            elif policy == "CorridorToBoundary":
                # Check if ANY asset breached
                breached = False
                new_w = current_weights.copy()
                for i, a in enumerate(assets):
                    band = bands[a]
                    if current_weights[i] > band.upper_band:
                        new_w[i] = band.upper_band
                        breached = True
                    elif current_weights[i] < band.lower_band:
                        new_w[i] = band.lower_band
                        breached = True

                if breached:
                    should_rebalance = True
                    # Re-normalize new weights to sum to 1.0
                    new_w_sum = np.sum(new_w)
                    if new_w_sum > 0:
                        target_w_step = new_w / new_w_sum
                    else:
                        target_w_step = np.array(
                            [target_weights[a] for a in assets], dtype=np.float64
                        )

            if should_rebalance:
                weight_diffs = np.abs(target_w_step - current_weights)
                turnover = float(np.sum(weight_diffs))

                if turnover > 1e-5:
                    reb_count += 1
                    total_turnover += turnover

                    # Execution cost
                    traded_value = turnover * curr_nav
                    rebalance_cost = traded_value * cost_multiplier
                    total_costs += rebalance_cost

                    post_cost_nav = max(0.0, curr_nav - rebalance_cost)
                    quantities = np.array(
                        [
                            (post_cost_nav * target_w_step[i]) / current_prices[i]
                            for i in range(len(assets))
                        ],
                        dtype=np.float64,
                    )
                    curr_nav = post_cost_nav

            nav_history.append(curr_nav)

        res = _compute_performance_metrics(
            nav_series=nav_history,
            rebalance_count=reb_count,
            total_days=n_days,
            total_turnover=total_turnover,
            total_costs=total_costs,
            initial_capital=initial_capital,
            policy_name=policy,
            baseline_rebalance_count=daily_reb_count,
            annual_trading_days=annual_trading_days,
        )
        policies_results[policy] = res
        nav_histories[policy] = [round(x, 2) for x in nav_history]

    return {
        "policies": {k: v.to_dict() for k, v in policies_results.items()},
        "bands": {k: v.to_dict() for k, v in bands.items()},
        "metadata": {
            "n_days": n_days,
            "assets": assets,
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "initial_capital": initial_capital,
            "annual_trading_days": annual_trading_days,
        },
    }


def generate_synthetic_crypto_prices(
    symbols: list[str] | None = None,
    n_days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic correlated geometric Brownian motion prices for crypto assets."""
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]

    rng = np.random.default_rng(seed)
    n_assets = len(symbols)

    # Base drifts and volatilities
    drifts = np.array([0.0008, 0.0010, 0.0012, 0.0011][:n_assets])
    vols = np.array([0.025, 0.035, 0.045, 0.048][:n_assets])

    # Correlated returns
    base_corr = 0.65
    cov_matrix = np.full((n_assets, n_assets), base_corr)
    np.fill_diagonal(cov_matrix, 1.0)
    l_factor = np.linalg.cholesky(cov_matrix)

    z = rng.standard_normal((n_days, n_assets))
    correlated_z = z @ l_factor.T

    daily_returns = drifts + vols * correlated_z

    start_prices = [65000.0, 3400.0, 150.0, 35.0][:n_assets]
    price_paths = np.zeros((n_days, n_assets))

    for i in range(n_assets):
        cum_ret = np.cumprod(1.0 + daily_returns[:, i])
        price_paths[:, i] = start_prices[i] * cum_ret

    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    return pd.DataFrame(price_paths, index=dates, columns=symbols)


def export_corridor_analysis_json(result: dict[str, Any], filepath: str | Path) -> None:
    """Export corridor optimization and backtest results to JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def export_corridor_analysis_csv(result: dict[str, Any], filepath: str | Path) -> None:
    """Export policy performance comparison to CSV."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    policies = result.get("policies", {})
    records = list(policies.values())

    df = pd.DataFrame(records)
    df.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    """Command-line interface for crypto rebalancing corridor optimizer."""
    parser = argparse.ArgumentParser(
        description="Dynamic Rebalancing Corridor & No-Trade Zone Optimizer for Crypto Portfolios"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"],
        help="List of crypto asset symbols (default: BTC/USDT ETH/USDT SOL/USDT AVAX/USDT)",
    )
    parser.add_argument(
        "--corridor-type",
        choices=["proportional", "volatility", "leland"],
        default="leland",
        help="Corridor band model: proportional, volatility, or leland (default: leland)",
    )
    parser.add_argument(
        "--relative-tol",
        type=float,
        default=0.20,
        help="Relative tolerance fraction for proportional bands (default: 0.20)",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=3.0,
        help="Risk aversion parameter gamma for Leland optimal bands (default: 3.0)",
    )
    parser.add_argument(
        "--fee",
        type=float,
        default=0.001,
        help="Exchange fee rate per rebalancing leg (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0005,
        help="Execution slippage rate per leg (default: 0.0005 = 0.05%%)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial portfolio capital in USD (default: 10000.0)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of backtest simulation days (default: 365)",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default="",
        help="Path to export full results to JSON",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default="",
        help="Path to export policy comparison to CSV",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run example benchmark on synthetic crypto market data",
    )

    args = parser.parse_args()

    symbols = args.symbols
    n_assets = len(symbols)
    # Equal weight distribution as target
    equal_w = round(1.0 / n_assets, 4)
    target_weights = {s: equal_w for s in symbols}
    target_weights[symbols[-1]] = round(1.0 - sum(equal_w for _ in symbols[:-1]), 4)

    # Simulated volatilities
    sample_vols = {s: 0.50 + 0.10 * i for i, s in enumerate(symbols)}

    if args.corridor_type == "proportional":
        bands = calculate_proportional_bands(
            target_weights,
            relative_tolerance=args.relative_tol,
            asset_volatilities=sample_vols,
            transaction_costs=args.fee,
        )
    elif args.corridor_type == "volatility":
        bands = calculate_volatility_bands(
            target_weights,
            asset_volatilities=sample_vols,
            base_cost=args.fee,
        )
    else:
        bands = calculate_leland_cost_optimal_bands(
            target_weights,
            asset_volatilities=sample_vols,
            transaction_costs=args.fee,
            risk_aversion=args.risk_aversion,
        )

    print(f"[*] Rebalancing Corridor Model: {args.corridor_type.upper()}")
    print("========== Asset Target Weights & No-Trade Corridors ==========")
    for sym, b in bands.items():
        print(
            f"  {sym:12s} | Target: {b.target_weight*100:5.2f}% | "
            f"Lower: {b.lower_band*100:5.2f}% | Upper: {b.upper_band*100:5.2f}% | "
            f"Width: {b.band_width*100:5.2f}%"
        )
    print("===============================================================")

    prices_df = generate_synthetic_crypto_prices(symbols=symbols, n_days=args.days, seed=42)

    sim = simulate_corridor_rebalancing(
        prices_df=prices_df,
        target_weights=target_weights,
        bands=bands,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        initial_capital=args.capital,
    )

    print("\n========== Rebalancing Policy Comparative Benchmark ==========")
    fmt_hdr = f"{'Policy':20s} | {'Reb Cnt':8s} | {'Turnover':10s} | {'Costs ($)':10s} | {'Avoided%':9s} | {'Net CAGR':9s} | {'Sharpe':7s} | {'MDD':7s}"
    print(fmt_hdr)
    print("-" * len(fmt_hdr))
    for pol_name, pol in sim["policies"].items():
        print(
            f"{pol_name:20s} | {pol['rebalance_count']:8d} | {pol['total_turnover']:10.2f} | "
            f"${pol['total_costs_paid']:9.2f} | {pol['avoided_rebalances_pct']:8.1f}% | "
            f"{pol['cagr_pct']:8.2f}% | {pol['sharpe_ratio']:7.2f} | -{pol['max_drawdown_pct']:6.2f}%"
        )
    print("===============================================================")

    if args.export_json:
        export_corridor_analysis_json(sim, args.export_json)
        print(f"[+] Results exported to JSON: {args.export_json}")

    if args.export_csv:
        export_corridor_analysis_csv(sim, args.export_csv)
        print(f"[+] Policy benchmark exported to CSV: {args.export_csv}")


if __name__ == "__main__":
    main()
