"""Cryptocurrency Advanced Drawdown Risk & Performance Metrics Engine.

Provides institutional downside risk metrics:
- Ulcer Index (Peter Martin, 1987): Root-mean-square percentage drawdown
- Martin Ratio / Ulcer Performance Index (UPI): Excess return per unit of Ulcer Index
- Pain Index (Thomas Becker, 2001): Mean absolute percentage drawdown
- Pain Ratio: Excess return per unit of Pain Index
- Burke Ratio (Gibbon Burke, 1994): Excess return divided by root-sum-square drawdowns
- Maximum Drawdown (MDD) & Drawdown Duration
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


def _clean_series(data: pd.Series | np.ndarray) -> np.ndarray:
    """Validate and clean input series, stripping NaNs and Infs."""
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("Input data series must not be empty.")
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        raise ValueError("Input data series contains no finite values.")
    return valid


def compute_nav_series(
    data: pd.Series | np.ndarray,
    is_returns: bool = True,
    initial_value: float = 1.0,
) -> np.ndarray:
    """Convert returns or price levels into a normalized cumulative NAV series."""
    cleaned = _clean_series(data)
    if is_returns:
        # returns r_t -> (1 + r_t).cumprod()
        nav = initial_value * np.cumprod(1.0 + cleaned)
        # Prepend initial value for clean base
        return np.insert(nav, 0, initial_value)
    else:
        # price levels -> normalized to initial_value
        if np.any(cleaned <= 0):
            raise ValueError("Price series must contain strictly positive values.")
        return initial_value * (cleaned / cleaned[0])


def compute_drawdown_series(
    data: pd.Series | np.ndarray,
    is_returns: bool = True,
) -> np.ndarray:
    """Compute percentage drawdown series relative to high-water mark.

    Returns an array of non-positive percentage values (0.0 to -100.0 or lower).
    """
    nav = compute_nav_series(data, is_returns=is_returns)
    running_max = np.maximum.accumulate(nav)
    # Avoid zero division if max is zero or negative
    safe_max = np.where(running_max > 0, running_max, 1e-9)
    drawdowns = (nav - running_max) / safe_max * 100.0
    # Omit initial 0.0 base if from returns to keep length aligned with original returns
    return drawdowns[1:] if is_returns else drawdowns


def compute_max_drawdown(
    data: pd.Series | np.ndarray,
    is_returns: bool = True,
) -> float:
    """Calculate maximum peak-to-trough percentage drawdown (expressed as positive percentage)."""
    dd = compute_drawdown_series(data, is_returns=is_returns)
    min_dd = float(np.min(dd))
    return abs(min_dd) if min_dd < 0 else 0.0


def compute_ulcer_index(
    data: pd.Series | np.ndarray,
    is_returns: bool = True,
) -> float:
    """Calculate Peter G. Martin's Ulcer Index (1987).

    UI = sqrt( (1 / N) * sum(DD_t^2) )
    where DD_t is the percentage drawdown at time t.
    """
    dd = compute_drawdown_series(data, is_returns=is_returns)
    squared_dd = np.square(dd)
    return float(np.sqrt(np.mean(squared_dd)))


def compute_pain_index(
    data: pd.Series | np.ndarray,
    is_returns: bool = True,
) -> float:
    """Calculate Thomas Becker's Pain Index (2001).

    PI = (1 / N) * sum(|DD_t|)
    Mean absolute percentage drawdown over the sample period.
    """
    dd = compute_drawdown_series(data, is_returns=is_returns)
    return float(np.mean(np.abs(dd)))


def compute_annualized_cagr(
    data: pd.Series | np.ndarray,
    is_returns: bool = True,
    periods_per_year: int = 365,
) -> float:
    """Calculate Compound Annual Growth Rate (CAGR) in percent."""
    nav = compute_nav_series(data, is_returns=is_returns)
    n_periods = len(nav) - 1 if is_returns else len(nav) - 1
    if n_periods <= 0:
        return 0.0

    total_return = nav[-1] / nav[0]
    if total_return <= 0:
        return -100.0

    years = n_periods / float(periods_per_year)
    if years <= 0:
        return 0.0

    cagr = (total_return ** (1.0 / years) - 1.0) * 100.0
    return float(cagr)


def compute_martin_ratio(
    data: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    is_returns: bool = True,
    periods_per_year: int = 365,
) -> float:
    """Calculate Martin Ratio (Ulcer Performance Index - UPI).

    Martin Ratio = (CAGR - RiskFreeRate) / Ulcer Index
    If Ulcer Index is 0.0 (no drawdowns), returns high positive ratio if CAGR > Rf, else 0.0.
    """
    ui = compute_ulcer_index(data, is_returns=is_returns)
    cagr = compute_annualized_cagr(data, is_returns=is_returns, periods_per_year=periods_per_year)
    excess_return = cagr - risk_free_rate

    if ui < 1e-8:
        return 999.0 if excess_return > 0 else 0.0

    return float(excess_return / ui)


def compute_pain_ratio(
    data: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    is_returns: bool = True,
    periods_per_year: int = 365,
) -> float:
    """Calculate Pain Ratio.

    Pain Ratio = (CAGR - RiskFreeRate) / Pain Index
    """
    pi = compute_pain_index(data, is_returns=is_returns)
    cagr = compute_annualized_cagr(data, is_returns=is_returns, periods_per_year=periods_per_year)
    excess_return = cagr - risk_free_rate

    if pi < 1e-8:
        return 999.0 if excess_return > 0 else 0.0

    return float(excess_return / pi)


def compute_burke_ratio(
    data: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    is_returns: bool = True,
    periods_per_year: int = 365,
    modified: bool = False,
) -> float:
    """Calculate Burke Ratio (Gibbon Burke, 1994).

    Burke Ratio = (CAGR - RiskFreeRate) / sqrt(sum(DD_t^2))
    If modified=True:
    Modified Burke = (CAGR - RiskFreeRate) / sqrt((1/N) * sum(DD_t^2))  (which equals Martin Ratio)
    """
    if modified:
        return compute_martin_ratio(
            data,
            risk_free_rate=risk_free_rate,
            is_returns=is_returns,
            periods_per_year=periods_per_year,
        )

    dd = compute_drawdown_series(data, is_returns=is_returns)
    cagr = compute_annualized_cagr(data, is_returns=is_returns, periods_per_year=periods_per_year)
    excess_return = cagr - risk_free_rate

    sum_sq_dd = float(np.sum(np.square(dd)))
    denom = np.sqrt(sum_sq_dd)

    if denom < 1e-8:
        return 999.0 if excess_return > 0 else 0.0

    return float(excess_return / denom)


def compute_drawdown_duration_stats(
    data: pd.Series | np.ndarray,
    is_returns: bool = True,
) -> dict[str, float]:
    """Calculate underwater drawdown duration statistics.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        Returns or price series.
    is_returns : bool, default True
        Whether the input data represents returns (True) or price levels (False).

    Returns
    -------
    dict[str, float]
        - max_drawdown_duration: Longest period (bars) spent continuously underwater.
        - avg_drawdown_duration: Average duration of completed underwater drawdown episodes.
        - current_drawdown_duration: Bars underwater at the end of the series.
        - drawdown_episodes_count: Total count of drawdown episodes.
    """
    dd = compute_drawdown_series(data, is_returns=is_returns)
    if len(dd) == 0:
        return {
            "max_drawdown_duration": 0.0,
            "avg_drawdown_duration": 0.0,
            "current_drawdown_duration": 0.0,
            "drawdown_episodes_count": 0.0,
            "time_underwater_pct": 0.0,
        }

    durations: list[int] = []
    current_len = 0
    underwater_bars = 0

    for val in dd:
        if val < -1e-6:
            current_len += 1
            underwater_bars += 1
        else:
            if current_len > 0:
                durations.append(current_len)
                current_len = 0

    current_dd_dur = current_len
    all_episodes = list(durations)
    if current_len > 0:
        all_episodes.append(current_len)

    max_dur = max(all_episodes) if all_episodes else 0
    avg_dur = (sum(durations) / len(durations)) if durations else (float(current_len) if current_len > 0 else 0.0)
    time_underwater_pct = round((underwater_bars / len(dd)) * 100.0, 2) if len(dd) > 0 else 0.0

    return {
        "max_drawdown_duration": float(max_dur),
        "avg_drawdown_duration": round(float(avg_dur), 2),
        "current_drawdown_duration": float(current_dd_dur),
        "drawdown_episodes_count": float(len(all_episodes)),
        "time_underwater_pct": time_underwater_pct,
    }


def compute_drawdown_metrics_summary(
    data: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    is_returns: bool = True,
    periods_per_year: int = 365,
) -> dict[str, float]:
    """Compile comprehensive drawdown and downside performance metrics."""
    cagr = compute_annualized_cagr(data, is_returns=is_returns, periods_per_year=periods_per_year)
    mdd = compute_max_drawdown(data, is_returns=is_returns)
    ui = compute_ulcer_index(data, is_returns=is_returns)
    pi = compute_pain_index(data, is_returns=is_returns)
    martin = compute_martin_ratio(data, risk_free_rate=risk_free_rate, is_returns=is_returns, periods_per_year=periods_per_year)
    pain = compute_pain_ratio(data, risk_free_rate=risk_free_rate, is_returns=is_returns, periods_per_year=periods_per_year)
    burke = compute_burke_ratio(data, risk_free_rate=risk_free_rate, is_returns=is_returns, periods_per_year=periods_per_year)
    calmar = (cagr - risk_free_rate) / mdd if mdd > 1e-8 else (999.0 if cagr > risk_free_rate else 0.0)
    dur_stats = compute_drawdown_duration_stats(data, is_returns=is_returns)

    return {
        "cagr_pct": round(cagr, 4),
        "max_drawdown_pct": round(mdd, 4),
        "ulcer_index": round(ui, 4),
        "pain_index": round(pi, 4),
        "martin_ratio": round(martin, 4),
        "pain_ratio": round(pain, 4),
        "burke_ratio": round(burke, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown_duration": dur_stats["max_drawdown_duration"],
        "avg_drawdown_duration": dur_stats["avg_drawdown_duration"],
        "current_drawdown_duration": dur_stats["current_drawdown_duration"],
        "drawdown_episodes_count": dur_stats["drawdown_episodes_count"],
        "time_underwater_pct": dur_stats["time_underwater_pct"],
    }


def main():
    """CLI test runner demonstration."""
    parser = argparse.ArgumentParser(description="Cryptocurrency Drawdown & Ulcer Index Calculator")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic dataset for test")
    parser.add_argument("--rf", type=float, default=2.0, help="Annualized risk-free rate percentage (default: 2.0%)")
    args = parser.parse_args()

    # Generate test returns
    np.random.seed(42)
    sample_returns = np.random.normal(0.001, 0.02, 180)
    summary = compute_drawdown_metrics_summary(sample_returns, risk_free_rate=args.rf, is_returns=True)

    print("=== Advanced Drawdown & Downside Risk Summary ===")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
