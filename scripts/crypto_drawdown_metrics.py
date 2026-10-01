"""Cryptocurrency Advanced Drawdown Risk & Performance Metrics Engine.

Provides institutional downside risk metrics:
- Ulcer Index (Peter Martin, 1987): Root-mean-square percentage drawdown
- Martin Ratio / Ulcer Performance Index (UPI): Excess return per unit of Ulcer Index
- Pain Index (Thomas Becker, 2001): Mean absolute percentage drawdown
- Pain Ratio: Excess return per unit of Pain Index
- Burke Ratio (Gibbon Burke, 1994): Excess return divided by root-sum-square drawdowns
- Sterling Ratio: CAGR excess return per unit of average drawdown
- Omega Ratio (Con Keating & William Shadwick, 2002): Ratio of upside gains to downside losses
- Gain-to-Pain Ratio (Jack Schwager): Net return divided by absolute downside losses
- Tail Ratio: 95th percentile return divided by absolute 5th percentile return
- Common Sense Ratio (CSR, Jack Schwager): Tail Ratio multiplied by Gain-to-Pain Ratio
- K-Ratio (Lars Kestner): Cumulative equity curve slope normalized by standard error and sqrt(T)
- Maximum Drawdown (MDD) & Drawdown Duration
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


def compute_sterling_ratio(
    data: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    is_returns: bool = True,
    periods_per_year: int = 365,
) -> float:
    """Calculate Sterling Ratio.

    Sterling Ratio = (CAGR - RiskFreeRate) / Average Drawdown
    Evaluates return generation efficiency against typical downside drawdown depth.
    """
    dd = compute_drawdown_series(data, is_returns=is_returns)
    cagr = compute_annualized_cagr(data, is_returns=is_returns, periods_per_year=periods_per_year)
    excess_return = cagr - risk_free_rate

    underwater = [abs(val) for val in dd if val < -1e-6]
    avg_dd = float(np.mean(underwater)) if underwater else 0.0

    if avg_dd < 1e-8:
        return 999.0 if excess_return > 0 else 0.0

    return float(excess_return / avg_dd)


def compute_omega_ratio(
    data: pd.Series | np.ndarray,
    threshold: float = 0.0,
    is_returns: bool = True,
) -> float:
    """Calculate Keating-Shadwick Omega Ratio (Con Keating & William F. Shadwick, 2002).

    Omega(L) = Sum(max(r - L, 0)) / Sum(max(L - r, 0))
    Evaluates probability-weighted gains relative to probability-weighted losses
    against a return threshold L (default 0.0).
    """
    cleaned = _clean_series(data)
    if not is_returns:
        if np.any(cleaned <= 0):
            raise ValueError("Price series must contain strictly positive values.")
        returns = np.diff(cleaned) / cleaned[:-1]
    else:
        returns = cleaned

    upside = returns[returns > threshold] - threshold
    downside = threshold - returns[returns < threshold]

    sum_downside = float(np.sum(downside))
    sum_upside = float(np.sum(upside))

    if sum_downside < 1e-8:
        return 999.0 if sum_upside > 0 else 0.0

    return float(sum_upside / sum_downside)


def compute_gain_to_pain_ratio(
    data: pd.Series | np.ndarray,
    is_returns: bool = True,
) -> float:
    """Calculate Gain-to-Pain Ratio (Jack Schwager).

    Gain-to-Pain = Sum(All Returns) / Sum(|Negative Returns|)
    Measures net return generation efficiency against raw downside loss volume.
    """
    cleaned = _clean_series(data)
    if not is_returns:
        if np.any(cleaned <= 0):
            raise ValueError("Price series must contain strictly positive values.")
        returns = np.diff(cleaned) / cleaned[:-1]
    else:
        returns = cleaned

    sum_all = float(np.sum(returns))
    sum_losses = float(np.sum(np.abs(returns[returns < 0.0])))

    if sum_losses < 1e-8:
        return 999.0 if sum_all > 0 else 0.0

    return float(sum_all / sum_losses)


def compute_tail_ratio(
    data: pd.Series | np.ndarray,
    percentile: float = 95.0,
    is_returns: bool = True,
) -> float:
    """Calculate Tail Ratio (95th percentile return / abs(5th percentile return)).

    Measures right-tail upside vs left-tail downside asymmetry.
    A ratio > 1.0 indicates upside outliers outstrip downside tail risk.
    """
    cleaned = _clean_series(data)
    if not is_returns:
        if np.any(cleaned <= 0):
            raise ValueError("Price series must contain strictly positive values.")
        returns = np.diff(cleaned) / cleaned[:-1]
    else:
        returns = cleaned

    if len(returns) < 2:
        return 0.0

    p_upper = float(np.percentile(returns, percentile))
    p_lower = float(np.percentile(returns, 100.0 - percentile))
    abs_lower = abs(p_lower)

    if abs_lower < 1e-8:
        return 999.0 if p_upper > 0 else 0.0

    return float(p_upper / abs_lower)


def compute_common_sense_ratio(
    data: pd.Series | np.ndarray,
    percentile: float = 95.0,
    is_returns: bool = True,
) -> float:
    """Calculate Jack Schwager's Common Sense Ratio (CSR).

    CSR = Tail Ratio * Gain-to-Pain Ratio
    Combines right-tail skew asymmetry with net return efficiency against loss volume.
    """
    tail = compute_tail_ratio(data, percentile=percentile, is_returns=is_returns)
    gpr = compute_gain_to_pain_ratio(data, is_returns=is_returns)
    if tail <= 0.0 or gpr <= 0.0:
        return 0.0
    return float(tail * gpr)


def compute_k_ratio(
    data: pd.Series | np.ndarray,
    is_returns: bool = True,
) -> float:
    """Calculate Lars Kestner's K-Ratio.

    K-Ratio = Slope of cumulative NAV / (Standard Error of slope * sqrt(T))
    Evaluates consistency of equity curve upward trajectory against noise.
    """
    nav = compute_nav_series(data, is_returns=is_returns)
    T = len(nav)
    if T < 3:
        return 0.0

    x = np.arange(T, dtype=float)
    y = (nav - nav[0]) / (nav[0] if nav[0] > 0 else 1.0) * 100.0

    mean_x = (T - 1) / 2.0
    mean_y = float(np.mean(y))
    ss_xx = float(np.sum((x - mean_x) ** 2))
    ss_xy = float(np.sum((x - mean_x) * (y - mean_y)))

    if ss_xx < 1e-9:
        return 0.0

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    residuals = y - (intercept + slope * x)
    ss_res = float(np.sum(residuals ** 2))
    se_slope = np.sqrt(ss_res / ((T - 2) * ss_xx))

    if se_slope < 1e-8:
        return 999.0 if slope > 0 else (-999.0 if slope < 0 else 0.0)

    return float(slope / (se_slope * np.sqrt(T)))


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
    sterling = compute_sterling_ratio(data, risk_free_rate=risk_free_rate, is_returns=is_returns, periods_per_year=periods_per_year)
    omega = compute_omega_ratio(data, threshold=0.0, is_returns=is_returns)
    gain_to_pain = compute_gain_to_pain_ratio(data, is_returns=is_returns)
    tail_ratio = compute_tail_ratio(data, is_returns=is_returns)
    csr = compute_common_sense_ratio(data, is_returns=is_returns)
    k_ratio = compute_k_ratio(data, is_returns=is_returns)
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
        "sterling_ratio": round(sterling, 4),
        "omega_ratio": round(omega, 4),
        "gain_to_pain_ratio": round(gain_to_pain, 4),
        "tail_ratio": round(tail_ratio, 4),
        "common_sense_ratio": round(csr, 4),
        "k_ratio": round(k_ratio, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown_duration": dur_stats["max_drawdown_duration"],
        "avg_drawdown_duration": dur_stats["avg_drawdown_duration"],
        "current_drawdown_duration": dur_stats["current_drawdown_duration"],
        "drawdown_episodes_count": dur_stats["drawdown_episodes_count"],
        "time_underwater_pct": dur_stats["time_underwater_pct"],
    }


def export_drawdown_metrics_json(
    summary: dict[str, float],
    filepath: str | Path,
) -> None:
    """Export drawdown metrics summary dictionary to a JSON file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def export_drawdown_metrics_csv(
    summary: dict[str, float],
    filepath: str | Path,
) -> None:
    """Export drawdown metrics summary dictionary to a CSV file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"metric": k, "value": v} for k, v in summary.items()]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")


def main():
    """CLI test runner demonstration."""
    parser = argparse.ArgumentParser(description="Cryptocurrency Drawdown & Ulcer Index Calculator")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic dataset for test")
    parser.add_argument("--rf", type=float, default=2.0, help="Annualized risk-free rate percentage (default: 2.0%%)")
    parser.add_argument("--export-json", type=str, default=None, help="Path to export drawdown metrics summary to JSON file.")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to export drawdown metrics summary to CSV file.")
    args = parser.parse_args()

    # Generate test returns
    np.random.seed(42)
    sample_returns = np.random.normal(0.001, 0.02, 180)
    summary = compute_drawdown_metrics_summary(sample_returns, risk_free_rate=args.rf, is_returns=True)

    print("=== Advanced Drawdown & Downside Risk Summary ===")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")

    if args.export_json:
        export_drawdown_metrics_json(summary, args.export_json)
        print(f"[+] Drawdown metrics exported to: {args.export_json}")

    if args.export_csv:
        export_drawdown_metrics_csv(summary, args.export_csv)
        print(f"[+] Drawdown metrics exported to: {args.export_csv}")


if __name__ == "__main__":
    main()
