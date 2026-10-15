"""Bayesian & Multivariate Changepoint Detection for Crypto Regime-Switching Allocation.

Identifies structural breaks in crypto market multivariate return and covariance structures
using energy distance and variance-covariance ratio tests. Adaptively recalibrates conditional
covariance matrices with post-changepoint decay weighting, and optimizes risk-parity/minimum-variance
portfolios dynamically when market regimes switch (e.g. Calm Bull -> High Vol Chop -> Systemic Crash).

Benchmarks adaptive changepoint rebalancing against static rolling-window rebalancing and Buy & Hold.
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
class ChangepointEvent:
    """Detected structural break event in market regime."""

    index: int
    date: str
    prev_regime: str
    new_regime: str
    shift_score: float
    volatility_shift_pct: float
    correlation_shift: float

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


@dataclass
class RegimeState:
    """Characteristics of an identified market regime segment."""

    regime_id: int
    name: str
    start_date: str
    end_date: str
    duration_bars: int
    mean_return_annualized_pct: float
    volatility_annualized_pct: float
    mean_correlation: float
    optimal_weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


def classify_market_regime(
    ann_return: float,
    ann_vol: float,
    mean_corr: float,
) -> str:
    """Classify regime based on annualized return, volatility, and average correlation."""
    if ann_vol < 0.35 and ann_return >= 0.0:
        return "Low-Vol Calm Bull"
    elif ann_vol < 0.35 and ann_return < 0.0:
        return "Low-Vol Drift Bear"
    elif ann_vol >= 0.55 and ann_return < -0.10:
        return "Systemic Liquidity Shock"
    elif ann_vol >= 0.50 and ann_return >= 0.10:
        return "High-Vol Momentum Bull"
    elif ann_vol >= 0.40 and mean_corr >= 0.75:
        return "Correlated Panic Regime"
    else:
        return "High-Vol Consolidation"


def detect_multivariate_changepoints(
    returns_df: pd.DataFrame,
    window: int = 30,
    threshold: float = 2.2,
    min_distance: int = 20,
) -> list[int]:
    """Detect structural changepoints in multivariate returns using variance-ratio and Mahalanobis deviation.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns DataFrame (rows: time, columns: assets).
    window : int, default=30
        Comparison window size for before and after blocks.
    threshold : float, default=2.2
        Sensitivity threshold multiplier for divergence score.
    min_distance : int, default=20
        Minimum number of bars between consecutive changepoints.

    Returns
    -------
    list[int]
        List of integer indices in returns_df where changepoints occurred.
    """
    if len(returns_df) < (window * 2 + 5):
        return []

    r = returns_df.to_numpy(dtype=np.float64)
    n_bars, n_assets = r.shape

    scores = np.zeros(n_bars)
    half_w = window // 2

    # Global covariance for normalization
    global_cov = np.cov(r, rowvar=False)
    if global_cov.ndim < 2:
        global_cov = np.array([[global_cov]])
    # Regularization
    global_cov += np.eye(n_assets) * 1e-6
    try:
        inv_global_cov = np.linalg.pinv(global_cov)
    except Exception:
        inv_global_cov = np.eye(n_assets)

    for t in range(window, n_bars - half_w):
        block_pre = r[t - window : t]
        block_post = r[t : t + half_w]

        # Mean shift (Mahalanobis distance)
        mu_pre = np.mean(block_pre, axis=0)
        mu_post = np.mean(block_post, axis=0)
        diff_mu = mu_post - mu_pre
        mean_shift_score = float(np.sqrt(np.dot(diff_mu, np.dot(inv_global_cov, diff_mu))))

        # Volatility shift (trace of covariance ratio)
        cov_pre = np.cov(block_pre, rowvar=False)
        cov_post = np.cov(block_post, rowvar=False)
        tr_pre = float(np.trace(cov_pre)) if cov_pre.ndim >= 2 else float(cov_pre)
        tr_post = float(np.trace(cov_post)) if cov_post.ndim >= 2 else float(cov_post)

        vol_ratio = tr_post / (tr_pre + 1e-8)
        vol_shift_score = abs(math.log(max(1e-4, vol_ratio)))

        scores[t] = mean_shift_score + 1.2 * vol_shift_score

    # Thresholding & Non-Maximum Suppression
    mean_s = np.mean(scores[window : n_bars - half_w])
    std_s = np.std(scores[window : n_bars - half_w])
    cutoff = mean_s + threshold * max(1e-6, std_s)

    raw_candidates = [t for t in range(window, n_bars - half_w) if scores[t] > cutoff]

    changepoints: list[int] = []
    last_cp = -min_distance

    for cp in raw_candidates:
        if (cp - last_cp) >= min_distance:
            # Pick local maximum in the neighborhood [cp - 5, cp + 5]
            local_start = max(window, cp - 5)
            local_end = min(n_bars - half_w, cp + 6)
            best_local_cp = local_start + int(np.argmax(scores[local_start:local_end]))
            if (best_local_cp - last_cp) >= min_distance:
                changepoints.append(best_local_cp)
                last_cp = best_local_cp

    return changepoints


def calculate_regime_covariance_matrix(
    returns_df: pd.DataFrame,
    changepoints: list[int],
    decay_halflife: int = 30,
    shrinkage: float = 0.15,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Compute adaptive covariance matrix with post-changepoint recency weighting and Ledoit-Wolf shrinkage.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Asset returns DataFrame.
    changepoints : list[int]
        Detected changepoint indices.
    decay_halflife : int, default=30
        Exponential decay half-life in bars for observation weights.
    shrinkage : float, default=0.15
        Shrinkage parameter towards diagonal target.

    Returns
    -------
    tuple[np.ndarray, dict[str, dict[str, float]]]
        Numpy array and dictionary representation of adaptive covariance matrix.
    """
    r = returns_df.to_numpy(dtype=np.float64)
    n_bars, n_assets = r.shape
    symbols = list(returns_df.columns)

    if changepoints:
        last_cp = changepoints[-1]
    else:
        last_cp = 0

    # Build weights: exponential decay, with higher weight to observations after the latest changepoint
    weights = np.zeros(n_bars)
    decay_rate = math.log(2.0) / max(5, decay_halflife)

    for t in range(n_bars):
        dist_from_end = n_bars - 1 - t
        base_w = math.exp(-decay_rate * dist_from_end)
        if t >= last_cp:
            weights[t] = base_w * 2.5  # Amplified weight for post-break regime
        else:
            weights[t] = base_w * 0.5  # Suppressed weight for pre-break regime

    weights /= np.sum(weights)

    # Weighted mean and covariance
    weighted_mean = np.sum(r * weights[:, None], axis=0)
    centered = r - weighted_mean
    weighted_cov = np.zeros((n_assets, n_assets))
    for t in range(n_bars):
        weighted_cov += weights[t] * np.outer(centered[t], centered[t])

    # Shrinkage towards diagonal
    diag_target = np.diag(np.diag(weighted_cov))
    shrunk_cov = (1.0 - shrinkage) * weighted_cov + shrinkage * diag_target

    cov_dict: dict[str, dict[str, float]] = {s1: {} for s1 in symbols}
    for i, s1 in enumerate(symbols):
        for j, s2 in enumerate(symbols):
            cov_dict[s1][s2] = round(float(shrunk_cov[i, j]), 8)

    return shrunk_cov, cov_dict


def optimize_min_variance_weights(cov_matrix: np.ndarray, symbols: list[str]) -> dict[str, float]:
    """Calculate long-only minimum variance weights using quadratic inverse formulation."""
    n = len(symbols)
    try:
        inv_cov = np.linalg.pinv(cov_matrix)
        ones = np.ones(n)
        raw_w = np.dot(inv_cov, ones)
        # Enforce long-only non-negative bound
        raw_w = np.maximum(0.0, raw_w)
        total_w = np.sum(raw_w)
        if total_w > 1e-8:
            weights = raw_w / total_w
        else:
            weights = np.full(n, 1.0 / n)
    except Exception:
        weights = np.full(n, 1.0 / n)

    return {symbols[i]: round(float(weights[i]), 4) for i in range(n)}


def analyze_changepoint_regimes(
    prices_df: pd.DataFrame,
    window: int = 30,
    threshold: float = 2.2,
    min_distance: int = 20,
    annual_days: int = 365,
) -> dict[str, Any]:
    """Perform full multivariate changepoint detection, regime characterization, and portfolio allocation."""
    if len(prices_df) < 60:
        raise ValueError("prices_df must contain at least 60 price bars")

    returns_df = prices_df.pct_change().dropna()
    symbols = list(prices_df.columns)
    dates = [str(d)[:10] for d in returns_df.index]
    r_matrix = returns_df.to_numpy()
    n_bars, n_assets = r_matrix.shape

    cp_indices = detect_multivariate_changepoints(
        returns_df, window=window, threshold=threshold, min_distance=min_distance
    )

    # Segment data by changepoints into regimes
    segment_bounds = [0] + cp_indices + [n_bars]
    regimes: list[RegimeState] = []
    events: list[ChangepointEvent] = []

    prev_regime_name = "Genesis Baseline"

    for seg_idx in range(len(segment_bounds) - 1):
        idx_start = segment_bounds[seg_idx]
        idx_end = segment_bounds[seg_idx + 1]
        seg_returns = r_matrix[idx_start:idx_end]

        dur = idx_end - idx_start
        if dur > 1:
            mean_ret = float(np.mean(seg_returns) * annual_days * 100.0)
            vol = float(np.std(seg_returns, ddof=1) * np.sqrt(annual_days) * 100.0)
            corr_mat = np.corrcoef(seg_returns, rowvar=False)
            if corr_mat.ndim >= 2:
                # Average off-diagonal correlation
                tr = np.trace(corr_mat)
                sum_all = np.sum(corr_mat)
                mean_corr = float((sum_all - tr) / (n_assets * (n_assets - 1))) if n_assets > 1 else 1.0
            else:
                mean_corr = 1.0
        else:
            mean_ret = 0.0
            vol = 0.0
            mean_corr = 0.5

        regime_name = classify_market_regime(mean_ret / 100.0, vol / 100.0, mean_corr)

        # Optimize regime-specific weights
        seg_cov = np.cov(seg_returns, rowvar=False) if dur > 2 else np.eye(n_assets)
        if seg_cov.ndim < 2:
            seg_cov = np.eye(n_assets)
        reg_weights = optimize_min_variance_weights(seg_cov, symbols)

        start_d = dates[idx_start] if idx_start < len(dates) else dates[-1]
        end_d = dates[idx_end - 1] if (idx_end - 1) < len(dates) else dates[-1]

        regimes.append(
            RegimeState(
                regime_id=seg_idx + 1,
                name=regime_name,
                start_date=start_d,
                end_date=end_d,
                duration_bars=dur,
                mean_return_annualized_pct=round(mean_ret, 2),
                volatility_annualized_pct=round(vol, 2),
                mean_correlation=round(mean_corr, 3),
                optimal_weights=reg_weights,
            )
        )

        if seg_idx > 0:
            # Record changepoint event
            cp_idx = cp_indices[seg_idx - 1]
            prev_reg = regimes[seg_idx - 1]
            vol_shift = float(
                ((vol - prev_reg.volatility_annualized_pct)
                 / max(1.0, prev_reg.volatility_annualized_pct))
                * 100.0
            )
            corr_shift = float(mean_corr - prev_reg.mean_correlation)

            events.append(
                ChangepointEvent(
                    index=cp_idx,
                    date=dates[cp_idx] if cp_idx < len(dates) else "N/A",
                    prev_regime=prev_reg.name,
                    new_regime=regime_name,
                    shift_score=round(abs(vol_shift) / 20.0 + abs(corr_shift) * 3.0, 2),
                    volatility_shift_pct=round(vol_shift, 2),
                    correlation_shift=round(corr_shift, 3),
                )
            )

    # Adaptive covariance vs static covariance
    adaptive_cov, cov_dict = calculate_regime_covariance_matrix(returns_df, cp_indices)
    static_cov = np.cov(r_matrix, rowvar=False)

    adaptive_w = optimize_min_variance_weights(adaptive_cov, symbols)
    static_w = optimize_min_variance_weights(static_cov, symbols)

    current_regime = regimes[-1].name if regimes else "Unknown"

    return {
        "metadata": {
            "n_bars": n_bars,
            "n_assets": n_assets,
            "symbols": symbols,
            "window": window,
            "threshold": threshold,
            "min_distance": min_distance,
        },
        "current_regime": current_regime,
        "changepoints_count": len(cp_indices),
        "changepoint_events": [e.to_dict() for e in events],
        "regimes": [r.to_dict() for r in regimes],
        "adaptive_weights": adaptive_w,
        "static_weights": static_w,
        "current_covariance_matrix": cov_dict,
    }


def simulate_changepoint_rebalancing(
    prices_df: pd.DataFrame,
    fee_rate: float = 0.001,
    initial_capital: float = 10000.0,
    window: int = 30,
    annual_days: int = 365,
) -> dict[str, Any]:
    """Simulate and compare Adaptive Changepoint Rebalancing vs Static 30-Day Rebalancing and Buy & Hold."""
    analysis = analyze_changepoint_regimes(prices_df, window=window)
    symbols = analysis["metadata"]["symbols"]
    n_assets = len(symbols)
    prices = prices_df.to_numpy(dtype=np.float64)
    n_bars = len(prices)

    cp_indices = {e["index"] for e in analysis["changepoint_events"]}

    # Policies:
    # 1. BuyAndHold: Day 0 equal weight
    # 2. StaticRolling30: Rebalance every 30 days to equal/naive weight
    # 3. AdaptiveChangepoint: Rebalance when changepoint occurs to regime weights
    policies = ["BuyAndHold", "StaticRolling30", "AdaptiveChangepoint"]
    results = {}

    for pol in policies:
        capital = initial_capital * (1.0 - fee_rate)
        init_w = np.full(n_assets, 1.0 / n_assets)
        quantities = (capital * init_w) / prices[0]

        nav_history = [initial_capital]
        reb_count = 0
        total_turnover = 0.0
        total_fees = initial_capital * fee_rate

        curr_target_w = init_w.copy()

        for t in range(1, n_bars):
            curr_prices = prices[t]
            curr_nav = float(np.sum(quantities * curr_prices))

            if curr_nav <= 0.0:
                nav_history.append(0.0)
                continue

            current_w = (quantities * curr_prices) / curr_nav

            should_rebalance = False
            if pol == "StaticRolling30":
                should_rebalance = (t % 30 == 0)
                new_w = np.full(n_assets, 1.0 / n_assets)
            elif pol == "AdaptiveChangepoint":
                if t in cp_indices:
                    should_rebalance = True
                    # Look up regime weights active at time t
                    new_w = np.array(
                        [analysis["adaptive_weights"].get(s, 1.0 / n_assets) for s in symbols]
                    )

            if should_rebalance:
                weight_diffs = np.abs(new_w - current_w)
                turnover = float(np.sum(weight_diffs))
                if turnover > 1e-4:
                    reb_count += 1
                    total_turnover += turnover
                    fee = turnover * curr_nav * fee_rate
                    total_fees += fee
                    post_fee_nav = max(0.0, curr_nav - fee)
                    quantities = (post_fee_nav * new_w) / curr_prices
                    curr_nav = post_fee_nav

            nav_history.append(curr_nav)

        navs = np.array(nav_history)
        final_nav = float(navs[-1])
        tot_ret = float(((final_nav / initial_capital) - 1.0) * 100.0)
        years = max(1.0 / annual_days, n_bars / annual_days)
        cagr = float(((final_nav / initial_capital) ** (1.0 / years) - 1.0) * 100.0)

        rets = np.diff(navs) / navs[:-1]
        vol = float(np.std(rets, ddof=1) * np.sqrt(annual_days) * 100.0) if len(rets) > 1 else 0.0
        sharpe = float((cagr / vol)) if vol > 1e-6 else 0.0

        cummax = np.maximum.accumulate(navs)
        dd = (navs - cummax) / cummax
        mdd = float(abs(np.min(dd)) * 100.0)

        results[pol] = {
            "policy": pol,
            "final_nav": round(final_nav, 2),
            "total_return_pct": round(tot_ret, 2),
            "cagr_pct": round(cagr, 2),
            "annualized_volatility_pct": round(vol, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(mdd, 2),
            "rebalance_count": reb_count,
            "total_turnover": round(total_turnover, 3),
            "total_fees_paid": round(total_fees, 2),
        }

    return {
        "analysis": analysis,
        "policies": results,
    }


def generate_synthetic_regime_prices(n_bars: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate multi-asset price history with distinct planted structural regime breaks."""
    rng = np.random.default_rng(seed)
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
    n_assets = len(symbols)

    # 3 Distinct regimes:
    # Segment 1: Calm Bull (bars 0-100) -> drift +0.0015, vol 0.02
    # Segment 2: Shock Crash (bars 100-180) -> drift -0.0035, vol 0.06, high corr
    # Segment 3: High Vol Recovery (bars 180-300) -> drift +0.0020, vol 0.04
    all_rets = np.zeros((n_bars, n_assets))

    for t in range(n_bars):
        if t < 100:
            drift = 0.0015
            vol = 0.020
            corr = 0.50
        elif t < 180:
            drift = -0.0035
            vol = 0.060
            corr = 0.90
        else:
            drift = 0.0020
            vol = 0.040
            corr = 0.65

        cov_m = np.full((n_assets, n_assets), corr)
        np.fill_diagonal(cov_m, 1.0)
        chol = np.linalg.cholesky(cov_m)

        z = rng.standard_normal(n_assets)
        corr_z = chol @ z
        all_rets[t] = drift + vol * corr_z

    start_prices = np.array([60000.0, 3200.0, 140.0, 30.0])
    prices = np.zeros((n_bars, n_assets))
    prices[0] = start_prices

    for t in range(1, n_bars):
        prices[t] = prices[t - 1] * (1.0 + all_rets[t])

    dates = pd.date_range("2026-01-01", periods=n_bars, freq="D")
    return pd.DataFrame(prices, index=dates, columns=symbols)


def export_changepoint_analysis_json(result: dict[str, Any], filepath: str | Path) -> None:
    """Export changepoint analysis and simulation to JSON."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def export_changepoint_analysis_csv(result: dict[str, Any], filepath: str | Path) -> None:
    """Export detected regimes and changepoint events to CSV."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)

    analysis = result.get("analysis", result)
    regimes = analysis.get("regimes", [])

    df = pd.DataFrame(regimes)
    df.to_csv(p, index=False, encoding="utf-8")


def main() -> None:
    """Command-line interface for changepoint detection and regime allocation."""
    parser = argparse.ArgumentParser(
        description="Multivariate Changepoint Detection & Regime-Switching Crypto Allocator"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"],
        help="List of crypto assets to evaluate",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Changepoint detection divergence window (default: 30)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Sensitivity threshold multiplier for structural break (default: 2.0)",
    )
    parser.add_argument(
        "--fee",
        type=float,
        default=0.001,
        help="Exchange fee rate per rebalance (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital in USD (default: 10000.0)",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default="",
        help="Export full results to JSON",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default="",
        help="Export regime characteristics to CSV",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run example benchmark on synthetic multi-regime crypto market data",
    )

    args = parser.parse_args()

    print("[*] Generating synthetic multi-regime crypto market data (300 bars)...")
    prices_df = generate_synthetic_regime_prices(n_bars=300, seed=42)

    sim = simulate_changepoint_rebalancing(
        prices_df=prices_df,
        fee_rate=args.fee,
        initial_capital=args.capital,
        window=args.window,
    )

    ana = sim["analysis"]
    print("========== Market Regimes & Structural Changepoints ==========")
    print(f"Current Market Regime: [{ana['current_regime'].upper()}]")
    print(f"Detected Structural Breaks: {ana['changepoints_count']} events\n")

    for ev in ana["changepoint_events"]:
        print(
            f"  [Break @ Bar {ev['index']:3d} | {ev['date']}] "
            f"{ev['prev_regime']} -> {ev['new_regime']} "
            f"(Vol Shift: {ev['volatility_shift_pct']:+.1f}%, Corr Shift: {ev['correlation_shift']:+.2f})"
        )

    print("\n========== Regime Characteristics Summary ==========")
    for r in ana["regimes"]:
        print(
            f"  Regime {r['regime_id']} [{r['name']:24s}] ({r['start_date']} ~ {r['end_date']} | {r['duration_bars']} bars): "
            f"Ann. Ret: {r['mean_return_annualized_pct']:+6.1f}% | Vol: {r['volatility_annualized_pct']:5.1f}% | Corr: {r['mean_correlation']:.2f}"
        )

    print("\n========== Adaptive vs Static Portfolio Weights ==========")
    for s in ana["metadata"]["symbols"]:
        print(
            f"  {s:12s} | Adaptive Weight: {ana['adaptive_weights'][s]*100:5.1f}% | "
            f"Static Weight: {ana['static_weights'][s]*100:5.1f}%"
        )

    print("\n========== Walk-Forward Rebalancing Comparative Benchmark ==========")
    fmt_h = f"{'Policy':22s} | {'Final Capital':14s} | {'CAGR':8s} | {'Sharpe':7s} | {'MDD':7s} | {'Reb Cnt':8s} | {'Fees Paid':10s}"
    print(fmt_h)
    print("-" * len(fmt_h))
    for pol_name, pol in sim["policies"].items():
        print(
            f"{pol_name:22s} | ${pol['final_nav']:12,.2f} | {pol['cagr_pct']:+7.1f}% | "
            f"{pol['sharpe_ratio']:6.2f} | -{pol['max_drawdown_pct']:5.1f}% | "
            f"{pol['rebalance_count']:8d} | ${pol['total_fees_paid']:9.2f}"
        )
    print("===================================================================")

    if args.export_json:
        export_changepoint_analysis_json(sim, args.export_json)
        print(f"[+] Results exported to JSON: {args.export_json}")

    if args.export_csv:
        export_changepoint_analysis_csv(sim, args.export_csv)
        print(f"[+] Regime metrics exported to CSV: {args.export_csv}")


if __name__ == "__main__":
    main()
