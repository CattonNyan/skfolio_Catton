"""Kelly Criterion and Fractional-Kelly Position Sizer for Crypto Portfolios.

Provides optimal capital growth position sizing:
- Discrete Kelly formula (win rate, payoff ratio)
- Continuous Kelly formula (mean excess return / variance)
- Multi-Asset Kelly allocator (unconstrained and constrained)
- Half-Kelly / Fractional Kelly scaling to curb crypto tail risk drawdown
- Risk of Ruin (Ralph Vince & Perry Kaufman formula)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    risk_of_ruin: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert Kelly calculation result to a serializable dictionary."""
        return {
            "full_kelly": round(float(self.full_kelly), 4),
            "fractional_kelly": round(float(self.fractional_kelly), 4),
            "fraction": round(float(self.fraction), 4),
            "expected_growth_rate": round(float(self.expected_growth_rate), 6),
            "half_kelly": round(float(self.half_kelly), 4),
            "is_positive_edge": self.is_positive_edge,
            "risk_of_ruin": round(float(self.risk_of_ruin), 4),
        }


@dataclass
class MultiAssetKellyResult:
    """Multi-asset portfolio Kelly optimization results container."""
    assets: list[str]
    full_kelly_weights: dict[str, float]
    fractional_kelly_weights: dict[str, float]
    half_kelly_weights: dict[str, float]
    quarter_kelly_weights: dict[str, float]
    fraction: float
    expected_growth_rate: float
    expected_annual_growth_rate: float
    portfolio_volatility: float
    portfolio_sharpe: float
    drawdown_50_prob: float
    capital: float | None = None
    position_dollars: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert multi-asset Kelly result to a serializable dictionary."""
        res: dict[str, Any] = {
            "assets": self.assets,
            "fraction": round(float(self.fraction), 4),
            "full_kelly_weights": {k: round(float(v), 4) for k, v in self.full_kelly_weights.items()},
            "fractional_kelly_weights": {k: round(float(v), 4) for k, v in self.fractional_kelly_weights.items()},
            "half_kelly_weights": {k: round(float(v), 4) for k, v in self.half_kelly_weights.items()},
            "quarter_kelly_weights": {k: round(float(v), 4) for k, v in self.quarter_kelly_weights.items()},
            "expected_growth_rate": round(float(self.expected_growth_rate), 6),
            "expected_annual_growth_rate": round(float(self.expected_annual_growth_rate), 4),
            "portfolio_volatility": round(float(self.portfolio_volatility), 4),
            "portfolio_sharpe": round(float(self.portfolio_sharpe), 4),
            "drawdown_50_prob": round(float(self.drawdown_50_prob), 4),
        }
        if self.capital is not None:
            res["capital"] = float(self.capital)
        if self.position_dollars is not None:
            res["position_dollars"] = {k: round(float(v), 2) for k, v in self.position_dollars.items()}
        return res


def calculate_risk_of_ruin(
    win_rate: float,
    payoff_ratio: float,
    bet_fraction: float = 0.02,
    loss_limit: float = 0.5,
) -> float:
    """Calculate theoretical Risk of Ruin (Ralph Vince & Perry Kaufman formula).

    Evaluates the probability of suffering an accumulated drawdown of loss_limit
    (e.g., 0.5 for a 50% account drawdown) before reaching capital growth targets.

    Parameters
    ----------
    win_rate : float
        Probability of winning trade (0.0 < p < 1.0).
    payoff_ratio : float
        Ratio of average win to average loss (b > 0).
    bet_fraction : float, default 0.02
        Fraction of account risked per trade (0.0 < bet_fraction <= 1.0).
    loss_limit : float, default 0.5
        Drawdown barrier considered ruin (0.0 < loss_limit <= 1.0).

    Returns
    -------
    float
        Probability of ruin between 0.0 and 1.0.
    """
    if not (0.0 < win_rate < 1.0):
        raise ValueError("Win rate must be strictly between 0 and 1.")
    if payoff_ratio <= 0.0:
        raise ValueError("Payoff ratio must be strictly positive.")
    if not (0.0 < bet_fraction <= 1.0):
        raise ValueError("Bet fraction must be strictly between 0 and 1.")
    if not (0.0 < loss_limit <= 1.0):
        raise ValueError("Loss limit must be strictly between 0 and 1.")

    loss_rate = 1.0 - win_rate
    expected_payoff = win_rate * payoff_ratio - loss_rate
    if expected_payoff <= 0.0:
        return 1.0

    denom = win_rate * payoff_ratio + loss_rate
    if denom <= 1e-9:
        return 1.0

    edge = expected_payoff / denom
    z = (1.0 - edge) / (1.0 + edge)

    # Number of units of bet risked before reaching loss limit
    units = loss_limit / bet_fraction
    ror = float(z ** units)
    return max(0.0, min(1.0, ror))


def calculate_discrete_kelly(
    win_rate: float,
    payoff_ratio: float,
    fraction: float = 0.5,
    max_allocation: float = 1.0,
    ruin_limit: float = 0.5,
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
    ruin_limit : float, default 0.5
        Capital drawdown barrier for risk-of-ruin assessment.
    """
    if not (0.0 < win_rate < 1.0):
        raise ValueError("Win rate must be strictly between 0 and 1.")
    if payoff_ratio <= 0.0:
        raise ValueError("Payoff ratio must be strictly positive.")
    if fraction <= 0.0:
        raise ValueError("Fraction must be strictly positive.")
    if max_allocation <= 0.0:
        raise ValueError("Max allocation must be strictly positive.")
    if not (0.0 < ruin_limit <= 1.0):
        raise ValueError("Ruin limit must be strictly between 0 and 1.")

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

    ror = calculate_risk_of_ruin(
        win_rate=win_rate,
        payoff_ratio=payoff_ratio,
        bet_fraction=frac_k if frac_k > 0 else 0.02,
        loss_limit=ruin_limit,
    )

    return KellyResult(
        full_kelly=full_k_clamped,
        fractional_kelly=frac_k,
        fraction=fraction,
        expected_growth_rate=float(growth),
        half_kelly=half_k,
        is_positive_edge=is_positive,
        risk_of_ruin=ror,
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


def calculate_multi_asset_kelly_allocation(
    returns_df: pd.DataFrame,
    fraction: float = 0.5,
    risk_free_rate: float = 0.0,
    covariance_shrinkage: float = 0.1,
    max_weight_per_asset: float = 0.5,
    capital: float | None = None,
    periods_per_year: int = 365,
    shrinkage: float | None = None,
) -> MultiAssetKellyResult:
    """Compute multi-asset continuous Kelly allocation with covariance shrinkage and position bounds.

    F* = fraction * Sigma_shrunk^-1 * (mu - r_f)
    """
    if shrinkage is not None:
        covariance_shrinkage = shrinkage
    if returns_df.empty or len(returns_df) < 2:
        raise ValueError("Returns DataFrame must have at least 2 observations.")
    if returns_df.shape[1] < 2:
        raise ValueError("Multi-asset Kelly requires at least 2 assets.")
    if not (0.0 < fraction <= 2.0):
        raise ValueError("Fraction must be strictly positive and <= 2.0.")
    if not (0.0 <= covariance_shrinkage <= 1.0):
        raise ValueError("Covariance shrinkage must be between 0.0 and 1.0.")
    if not (0.0 < max_weight_per_asset <= 1.0):
        raise ValueError("Max weight per asset must be strictly between 0 and 1.")

    assets = list(returns_df.columns)
    mu = returns_df.mean().to_numpy(dtype=float)
    sigma = returns_df.cov().to_numpy(dtype=float)
    n = len(assets)

    # Covariance shrinkage: (1 - alpha) * Sigma + alpha * diag(Sigma)
    diag_sigma = np.diag(np.diag(sigma))
    shrunk_sigma = (1.0 - covariance_shrinkage) * sigma + covariance_shrinkage * diag_sigma
    reg_sigma = shrunk_sigma + np.eye(n) * 1e-6

    excess_mu = mu - risk_free_rate
    try:
        inv_sigma = np.linalg.pinv(reg_sigma)
        raw_full = inv_sigma @ excess_mu
    except Exception:
        raw_full = np.ones(n) / n

    # Long-only non-negative constraint
    raw_full = np.maximum(0.0, raw_full)

    # Individual asset cap and normalization
    tot = float(np.sum(raw_full))
    if tot > 1e-9:
        w_norm = raw_full / max(1.0, tot)
    else:
        w_norm = np.zeros(n)

    # Apply asset cap and re-normalize if needed
    w_capped = np.minimum(w_norm, max_weight_per_asset)
    tot_capped = float(np.sum(w_capped))
    if tot_capped > 1.0:
        w_full = (w_capped / tot_capped)
    else:
        w_full = w_capped

    w_frac = w_full * fraction
    w_half = w_full * 0.5
    w_quarter = w_full * 0.25

    # Portfolio metrics under fractional weights
    port_var = float(w_frac.T @ sigma @ w_frac)
    port_vol = math.sqrt(max(0.0, port_var))
    port_excess = float(w_frac.T @ excess_mu)

    # Growth rate: g(w) = r_f + w^T (mu - r_f) - 0.5 * w^T Sigma w
    growth_rate = risk_free_rate + port_excess - 0.5 * port_var
    annual_growth = ((1.0 + growth_rate) ** periods_per_year) - 1.0 if (1.0 + growth_rate) > 0 else -1.0
    port_sharpe = (port_excess / port_vol) if port_vol > 1e-9 else 0.0

    # Theoretical probability of 50% drawdown under fractional Kelly
    f_val = min(1.0, fraction)
    dd_50_prob = max(0.0, min(1.0, (1.0 - f_val) / (1.0 + f_val))) if fraction < 1.0 else 1.0

    full_dict = {assets[i]: float(w_full[i]) for i in range(n)}
    frac_dict = {assets[i]: float(w_frac[i]) for i in range(n)}
    half_dict = {assets[i]: float(w_half[i]) for i in range(n)}
    quarter_dict = {assets[i]: float(w_quarter[i]) for i in range(n)}

    pos_dollars = None
    if capital is not None and capital > 0:
        pos_dollars = {assets[i]: float(w_frac[i] * capital) for i in range(n)}

    return MultiAssetKellyResult(
        assets=assets,
        full_kelly_weights=full_dict,
        fractional_kelly_weights=frac_dict,
        half_kelly_weights=half_dict,
        quarter_kelly_weights=quarter_dict,
        fraction=fraction,
        expected_growth_rate=float(growth_rate),
        expected_annual_growth_rate=float(annual_growth),
        portfolio_volatility=float(port_vol),
        portfolio_sharpe=float(port_sharpe),
        drawdown_50_prob=float(dd_50_prob),
        capital=capital,
        position_dollars=pos_dollars,
    )


def export_multi_asset_kelly_json(
    result: MultiAssetKellyResult,
    filepath: str | Path,
) -> Path:
    """Export multi-asset Kelly optimization results to a JSON file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    return out_path


def export_multi_asset_kelly_csv(
    result: MultiAssetKellyResult,
    filepath: str | Path,
) -> Path:
    """Export multi-asset Kelly allocation details to a CSV file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for asset in result.assets:
        row = {
            "asset": asset,
            "full_kelly_weight": round(result.full_kelly_weights.get(asset, 0.0), 4),
            "fractional_kelly_weight": round(result.fractional_kelly_weights.get(asset, 0.0), 4),
            "half_kelly_weight": round(result.half_kelly_weights.get(asset, 0.0), 4),
            "quarter_kelly_weight": round(result.quarter_kelly_weights.get(asset, 0.0), 4),
            "fraction": result.fraction,
            "expected_growth_rate": round(result.expected_growth_rate, 6),
            "expected_annual_growth_rate": round(result.expected_annual_growth_rate, 4),
            "portfolio_volatility": round(result.portfolio_volatility, 4),
            "portfolio_sharpe": round(result.portfolio_sharpe, 4),
            "drawdown_50_prob": round(result.drawdown_50_prob, 4),
        }
        if result.position_dollars:
            row["position_dollars"] = round(result.position_dollars.get(asset, 0.0), 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def to_dict_kelly_result(
    result: KellyResult,
    win_rate: float,
    payoff_ratio: float,
    fraction: float,
    capital: float | None = None,
) -> dict[str, Any]:
    """Convert Kelly sizing result to a serializable dictionary."""
    payload: dict[str, Any] = {
        "win_rate": float(win_rate),
        "payoff_ratio": float(payoff_ratio),
        "fraction": float(fraction),
        **result.to_dict(),
    }
    if capital is not None:
        payload["capital"] = float(capital)
        payload["full_kelly_dollars"] = round(float(result.full_kelly * capital), 2)
        payload["half_kelly_dollars"] = round(float(result.half_kelly * capital), 2)
        payload["fractional_kelly_dollars"] = round(float(result.fractional_kelly * capital), 2)
    return payload


def export_kelly_json(
    result: KellyResult,
    filepath: str | Path,
    win_rate: float,
    payoff_ratio: float,
    fraction: float,
    capital: float | None = None,
) -> None:
    """Export Kelly sizing results to a JSON file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_dict_kelly_result(
        result,
        win_rate=win_rate,
        payoff_ratio=payoff_ratio,
        fraction=fraction,
        capital=capital,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def export_kelly_csv(
    result: KellyResult,
    filepath: str | Path,
    win_rate: float,
    payoff_ratio: float,
    fraction: float,
    capital: float | None = None,
) -> None:
    """Export Kelly sizing results to a CSV file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_dict_kelly_result(
        result,
        win_rate=win_rate,
        payoff_ratio=payoff_ratio,
        fraction=fraction,
        capital=capital,
    )
    rows = [{"metric": k, "value": v} for k, v in payload.items()]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")


def export_portfolio_kelly_csv(
    weights: pd.Series,
    filepath: str | Path,
) -> Path:
    """Export multi-asset portfolio Kelly weights to a CSV file.

    Parameters
    ----------
    weights : pd.Series
        Asset weights Series with asset tickers/symbols as index.
    filepath : str | Path
        Target destination CSV path.

    Returns
    -------
    Path
        Resolved saved CSV filepath.
    """
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "asset": weights.index,
        "weight": weights.values,
        "allocation_pct": (weights.values * 100.0).round(2),
    })
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Crypto Kelly Criterion Position Sizer.")
    parser.add_argument("--win-rate", type=float, default=0.55, help="Strategy win rate (e.g. 0.55)")
    parser.add_argument("--payoff", type=float, default=1.8, help="Payoff ratio (win/loss ratio)")
    parser.add_argument("--fraction", type=float, default=0.5, help="Fractional Kelly multiplier (default 0.5)")
    parser.add_argument("--ruin-limit", type=float, default=0.5, help="Capital drawdown barrier for risk-of-ruin calculation (default 0.5 for 50%%).")
    parser.add_argument("--capital", type=float, default=None, help="Total account capital for position sizing in USD.")
    parser.add_argument("--multi-asset", action="store_true", help="Run multi-asset Kelly vector allocator.")
    parser.add_argument("--returns-csv", type=str, default=None, help="Path to asset returns CSV file for multi-asset Kelly.")
    parser.add_argument("--shrinkage", type=float, default=0.15, help="Covariance matrix shrinkage intensity (default 0.15).")
    parser.add_argument("--max-asset-weight", type=float, default=0.40, help="Maximum allocation weight cap per single asset (default 0.40).")
    parser.add_argument("--export-multi-json", type=str, default=None, help="Path to export multi-asset Kelly results to JSON file.")
    parser.add_argument("--export-multi-csv", type=str, default=None, help="Path to export multi-asset Kelly results to CSV file.")
    parser.add_argument("--export-json", type=str, default=None, help="Path to export Kelly sizing results to JSON file.")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to export Kelly sizing results to CSV file.")
    args = parser.parse_args()

    if args.multi_asset:
        if args.returns_csv and Path(args.returns_csv).exists():
            returns_df = pd.read_csv(args.returns_csv, index_col=0)
        else:
            rng = np.random.default_rng(42)
            n_obs = 180
            assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
            base_noise = rng.normal(0.001, 0.025, (n_obs, len(assets)))
            market_factor = rng.normal(0.0015, 0.02, (n_obs, 1))
            ret_data = base_noise + 0.6 * market_factor
            returns_df = pd.DataFrame(ret_data, columns=assets)

        multi_res = calculate_multi_asset_kelly_allocation(
            returns_df=returns_df,
            fraction=args.fraction,
            shrinkage=args.shrinkage,
            max_weight_per_asset=args.max_asset_weight,
            capital=args.capital,
        )
        print("=========== Crypto Multi-Asset Vector Kelly Allocation ===========")
        print(f"  Assets Evaluated     : {', '.join(multi_res.assets)}")
        print(f"  Shrinkage Intensity  : {args.shrinkage:.2f}")
        print(f"  Per-Asset Cap        : {args.max_asset_weight * 100:.1f}%")
        print(f"  Kelly Fraction       : {args.fraction:.2f}x")
        print(f"  Exp Growth Rate (d)  : {multi_res.expected_growth_rate * 100:.4f}%")
        print(f"  Exp Annual Growth    : {multi_res.expected_annual_growth_rate * 100:.2f}%")
        print(f"  Portfolio Volatility : {multi_res.portfolio_volatility * 100:.2f}%")
        print(f"  Portfolio Sharpe     : {multi_res.portfolio_sharpe:.2f}")
        print(f"  Prob 50% Drawdown    : {multi_res.drawdown_50_prob * 100:.2f}%")
        print("------------------------------------------------------------------")
        header_doll = f" {'Allocation ($)':<16}" if multi_res.capital else ""
        print(f"  {'Asset':<12} {'Full Kelly':<12} {'Frac Kelly':<12} {'Half Kelly':<12}{header_doll}")
        for a in multi_res.assets:
            fw = multi_res.full_kelly_weights[a] * 100
            rw = multi_res.fractional_kelly_weights[a] * 100
            hw = multi_res.half_kelly_weights[a] * 100
            doll_str = f"${multi_res.position_dollars[a]:,.2f}" if multi_res.position_dollars else ""
            print(f"  {a:<12} {fw:>9.2f}%  {rw:>9.2f}%  {hw:>9.2f}%   {doll_str}")
        print("==================================================================")

        if args.export_multi_json:
            export_multi_asset_kelly_json(multi_res, args.export_multi_json)
            print(f"[+] Multi-asset Kelly results exported to: {args.export_multi_json}")

        if args.export_multi_csv:
            export_multi_asset_kelly_csv(multi_res, args.export_multi_csv)
            print(f"[+] Multi-asset Kelly CSV exported to: {args.export_multi_csv}")
        return

    res = calculate_discrete_kelly(
        args.win_rate,
        args.payoff,
        fraction=args.fraction,
        ruin_limit=args.ruin_limit,
    )
    print("================ Crypto Kelly Criterion Sizing ================")
    print(f"  Win Rate           : {args.win_rate * 100:.1f}%")
    print(f"  Payoff Ratio       : {args.payoff:.2f}x")
    print(f"  Full Kelly (f*)    : {res.full_kelly * 100:.2f}%")
    print(f"  Half Kelly (0.5x)  : {res.half_kelly * 100:.2f}%")
    print(f"  Chosen Frac ({args.fraction}x): {res.fractional_kelly * 100:.2f}%")
    print(f"  Exp. Growth Rate   : {res.expected_growth_rate * 100:.3f}% per trade")
    print(f"  Risk of Ruin ({args.ruin_limit*100:.0f}% DD): {res.risk_of_ruin * 100:.2f}%")
    print(f"  Positive Edge      : {'YES' if res.is_positive_edge else 'NO'}")
    if args.capital:
        print(f"  Capital            : ${args.capital:,.2f}")
        print(f"  Full Kelly Stake   : ${res.full_kelly * args.capital:,.2f}")
        print(f"  Half Kelly Stake   : ${res.half_kelly * args.capital:,.2f}")
        print(f"  Fractional Stake   : ${res.fractional_kelly * args.capital:,.2f}")
    print("================================================================")

    if args.export_json:
        export_kelly_json(
            res,
            args.export_json,
            win_rate=args.win_rate,
            payoff_ratio=args.payoff,
            fraction=args.fraction,
            capital=args.capital,
        )
        print(f"[+] Kelly sizing results exported to: {args.export_json}")

    if args.export_csv:
        export_kelly_csv(
            res,
            args.export_csv,
            win_rate=args.win_rate,
            payoff_ratio=args.payoff,
            fraction=args.fraction,
            capital=args.capital,
        )
        print(f"[+] Kelly sizing results exported to: {args.export_csv}")


if __name__ == "__main__":
    main()
