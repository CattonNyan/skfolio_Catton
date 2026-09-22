"""Periodic Portfolio Rebalancing Backtest Engine for Cryptocurrencies.

Simulates walk-forward rolling window portfolio rebalancing:
1. Trains skfolio optimization models (e.g. Risk Parity, Max Sharpe, Min Variance) on historical rolling window.
2. Holds weights for a fixed rebalancing period (e.g. every 7 days, 14 days, or 30 days).
3. Applies transaction costs (slippage + trading fees) during rebalancing.
4. Compares against Buy & Hold benchmark and Equal-Weight benchmark.
5. Computes key quant metrics: CAGR, Volatility, Sharpe Ratio, Sortino Ratio, Maximum Drawdown (MDD), Turnover.
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
    MarketDataUnavailableError,
    find_freqtrade_data_dirs,
    generate_synthetic_crypto_data,
    load_market_data,
    load_from_feather_dir,
)

try:
    from skfolio import RiskMeasure
    from skfolio.optimization import (
        HierarchicalRiskParity,
        MeanRisk,
        MeanVariance,
        ObjectiveFunction,
        RiskBudgeting,
        SchurComplementary,
    )
    from skfolio.preprocessing import prices_to_returns
    HAS_SKFOLIO = True
except ImportError:
    HAS_SKFOLIO = False


def calculate_drawdown(nav_series: pd.Series) -> tuple[float, pd.Series]:
    """Calculate Maximum Drawdown (MDD) and Drawdown series."""
    peak = nav_series.cummax()
    drawdown = (nav_series - peak) / peak
    mdd = float(drawdown.min())
    return mdd, drawdown


def simulate_rebalancing(
    prices: pd.DataFrame,
    train_bars: int = 400,
    rebalance_freq_bars: int = 50,
    fee_rate: float = 0.001,
    model_choice: str = "Risk Parity",
    tolerance_band: float | None = None,
    drawdown_guard: float | None = None,
) -> dict[str, object]:
    """
    Run rolling-window walk-forward rebalancing backtest.

    Parameters:
    - prices: DataFrame of asset prices
    - train_bars: Lookback window to fit the model
    - rebalance_freq_bars: How often weights are recalculated
    - fee_rate: Transaction fee (e.g., 0.001 = 0.1% per turnover)
    - model_choice: "Risk Parity", "Max Sharpe", "Min Variance", "Min Semi-Variance", "Min CVaR", "HRP", "Schur", or "Equal Weight"
    - tolerance_band: Minimum weight deviation threshold (0.0~1.0) to execute rebalancing; avoids needless turnover/fees
    - drawdown_guard: Maximum tolerable drawdown (e.g. 0.15 = 15%) from peak NAV before temporarily moving to cash/stablecoin
    """
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] < 2 or len(prices) < 2:
        raise ValueError("Rebalancing requires at least two assets and two price rows.")
    try:
        price_values = prices.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Rebalancing prices must contain only numeric values.") from error
    if not np.all(np.isfinite(price_values)) or np.any(price_values <= 0):
        raise ValueError("Rebalancing prices must contain only finite, strictly positive values.")
    if isinstance(train_bars, bool) or not isinstance(train_bars, int) or train_bars < 2:
        raise ValueError("Training window must be an integer of at least 2 bars.")
    if isinstance(rebalance_freq_bars, bool) or not isinstance(rebalance_freq_bars, int) or rebalance_freq_bars <= 0:
        raise ValueError("Rebalancing frequency must be a strictly positive integer.")
    if isinstance(fee_rate, bool) or not isinstance(fee_rate, (int, float, np.number)) or not np.isfinite(fee_rate) or not 0 <= fee_rate < 1:
        raise ValueError("Fee rate must be a finite number between 0 and 1.")
    if tolerance_band is not None:
        if (
            isinstance(tolerance_band, bool)
            or not isinstance(tolerance_band, (int, float, np.number))
            or not np.isfinite(tolerance_band)
            or tolerance_band < 0
            or tolerance_band > 1
        ):
            raise ValueError("Tolerance band must be a finite number between 0 and 1.")
    if drawdown_guard is not None:
        if (
            isinstance(drawdown_guard, bool)
            or not isinstance(drawdown_guard, (int, float, np.number))
            or not np.isfinite(drawdown_guard)
            or drawdown_guard <= 0
            or drawdown_guard >= 1
        ):
            raise ValueError("Drawdown guard must be a finite number strictly between 0 and 1.")
    supported_models = {
        "Risk Parity",
        "Max Sharpe",
        "Min Variance",
        "Min Semi-Variance",
        "Min CVaR",
        "HRP",
        "Schur",
        "Equal Weight",
    }
    if model_choice not in supported_models:
        raise ValueError(f"Unsupported rebalancing model: {model_choice}")

    returns = prices.pct_change().dropna()
    assets = list(returns.columns)
    n_bars = len(returns)

    if n_bars <= train_bars + rebalance_freq_bars:
        raise ValueError(f"Insufficient data: {n_bars} bars available, need at least {train_bars + rebalance_freq_bars}")

    # Portfolios Net Asset Value (NAV) tracking, starting at 1.0
    nav_portfolio = [1.0]
    nav_benchmark_equal = [1.0]
    nav_benchmark_bh = [1.0]

    # Equal weights for benchmark
    eq_weights = np.ones(len(assets)) / len(assets)

    current_weights = eq_weights.copy()
    turnover_history: list[float] = []
    rebalance_dates: list[pd.Timestamp] = []
    weight_history: list[dict[str, float]] = []
    skipped_rebalances = 0
    guard_triggers = 0
    rolling_peak_nav = 1.0

    # Simulation loop
    test_start = train_bars
    test_dates = returns.index[test_start:]

    for idx, current_t in enumerate(range(test_start, n_bars)):
        date = returns.index[current_t]
        bar_ret = returns.iloc[current_t].values

        # Check if rebalancing should happen
        if (current_t - test_start) % rebalance_freq_bars == 0:
            rebalance_dates.append(date)
            # Window slice for training
            window_returns = returns.iloc[current_t - train_bars : current_t]

            # Fit optimization model
            candidate_weights = eq_weights.copy()
            if HAS_SKFOLIO and model_choice != "Equal Weight":
                try:
                    if model_choice == "Max Sharpe":
                        m = MeanVariance(
                            objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
                            risk_measure=RiskMeasure.VARIANCE,
                        )
                    elif model_choice == "Min Variance":
                        m = MeanVariance(
                            objective_function=ObjectiveFunction.MINIMIZE_RISK,
                            risk_measure=RiskMeasure.VARIANCE,
                        )
                    elif model_choice == "Min Semi-Variance":
                        m = MeanVariance(
                            objective_function=ObjectiveFunction.MINIMIZE_RISK,
                            risk_measure=RiskMeasure.SEMI_VARIANCE,
                        )
                    elif model_choice == "Min CVaR":
                        m = MeanRisk(
                            objective_function=ObjectiveFunction.MINIMIZE_RISK,
                            risk_measure=RiskMeasure.CVAR,
                        )
                    elif model_choice == "HRP":
                        m = HierarchicalRiskParity(risk_measure=RiskMeasure.VARIANCE)
                    elif model_choice == "Schur":
                        m = SchurComplementary()
                    else:  # Default to Risk Parity
                        m = RiskBudgeting(risk_measure=RiskMeasure.VARIANCE)

                    m.fit(window_returns)
                    candidate_weights = np.array(m.weights_)
                except Exception:
                    candidate_weights = current_weights.copy()

            # Check tolerance band threshold
            max_drift = float(np.max(np.abs(candidate_weights - current_weights)))
            if tolerance_band is not None and max_drift < tolerance_band and len(turnover_history) > 0:
                skipped_rebalances += 1
                new_weights = current_weights.copy()
                turnover = 0.0
            else:
                new_weights = candidate_weights
                turnover = float(np.sum(np.abs(new_weights - current_weights)))

            turnover_history.append(turnover)
            cost = turnover * fee_rate

            current_weights = new_weights
            weight_history.append(dict(zip(assets, current_weights)))

            # Deduct cost from portfolio NAV at rebalancing
            nav_portfolio[-1] *= (1.0 - cost)

        # Check drawdown guard
        current_nav = nav_portfolio[-1]
        rolling_peak_nav = max(rolling_peak_nav, current_nav)
        dd = (current_nav - rolling_peak_nav) / rolling_peak_nav if rolling_peak_nav > 0 else 0.0

        if drawdown_guard is not None and dd <= -drawdown_guard:
            guard_triggers += 1
            port_ret = 0.0
        else:
            # Portfolio return on this bar
            port_ret = float(np.dot(current_weights, bar_ret))
        next_nav_port = nav_portfolio[-1] * (1.0 + port_ret)
        nav_portfolio.append(next_nav_port)

        # Benchmark 1: Equal Weight (rebalanced at same frequency)
        eq_ret = float(np.dot(eq_weights, bar_ret))
        next_nav_eq = nav_benchmark_equal[-1] * (1.0 + eq_ret)
        nav_benchmark_equal.append(next_nav_eq)

        # Benchmark 2: Simple Buy & Hold (first asset or un-rebalanced basket)
        bh_ret = float(bar_ret[0])  # Primary crypto (e.g. BTC)
        next_nav_bh = nav_benchmark_bh[-1] * (1.0 + bh_ret)
        nav_benchmark_bh.append(next_nav_bh)

        # Passive drift of asset weights between rebalance intervals
        current_weights = current_weights * (1.0 + bar_ret)
        denom = np.sum(current_weights)
        if denom > 0:
            current_weights = current_weights / denom

    # Strip initial seed value
    nav_port_series = pd.Series(nav_portfolio[1:], index=test_dates, name=f"Rebalanced ({model_choice})")
    nav_eq_series = pd.Series(nav_benchmark_equal[1:], index=test_dates, name="Benchmark (Equal Weight)")
    nav_bh_series = pd.Series(nav_benchmark_bh[1:], index=test_dates, name=f"Benchmark ({assets[0]} Buy&Hold)")

    # Metrics calculation
    port_mdd, _ = calculate_drawdown(nav_port_series)
    eq_mdd, _ = calculate_drawdown(nav_eq_series)
    bh_mdd, _ = calculate_drawdown(nav_bh_series)

    total_return_port = (nav_port_series.iloc[-1] - 1.0) * 100
    total_return_eq = (nav_eq_series.iloc[-1] - 1.0) * 100
    total_return_bh = (nav_bh_series.iloc[-1] - 1.0) * 100

    # Detect candle frequency for annualization factor
    annual_factor = 365.0 * 24.0 * 4.0  # Default 15m
    if isinstance(test_dates, pd.DatetimeIndex) and len(test_dates) > 1:
        try:
            diffs = test_dates.to_series().diff().dropna()
            median_sec = float(diffs.dt.total_seconds().median())
            if median_sec > 0:
                annual_factor = (365.0 * 86400.0) / median_sec
        except Exception:
            pass

    pct_changes = nav_port_series.pct_change().dropna()
    mean_ret = float(pct_changes.mean()) if len(pct_changes) > 0 else 0.0
    vol = float(pct_changes.std()) if len(pct_changes) > 0 else 0.0
    sharpe = (mean_ret / (vol + 1e-9)) * np.sqrt(annual_factor) if len(pct_changes) > 0 else 0.0

    # Downside semi-deviation & Sortino Ratio
    downside = pct_changes[pct_changes < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else vol
    sortino = (mean_ret / (downside_std + 1e-9)) * np.sqrt(annual_factor) if len(pct_changes) > 0 else 0.0

    # Calmar Ratio: Return / Max Drawdown
    calmar = (total_return_port / (abs(port_mdd * 100.0) + 1e-9)) if abs(port_mdd) > 0 else 0.0

    # Ulcer Index & Martin Ratio calculation
    peak_port = nav_port_series.cummax()
    dd_port_pct = (nav_port_series - peak_port) / np.where(peak_port > 0, peak_port, 1e-9) * 100.0
    ulcer_index = float(np.sqrt(np.mean(np.square(dd_port_pct))))
    ann_return = mean_ret * annual_factor * 100.0
    martin_ratio = (ann_return / ulcer_index) if ulcer_index > 1e-6 else (999.0 if ann_return > 0 else 0.0)

    avg_turnover = float(np.mean(turnover_history)) if turnover_history else 0.0

    summary = {
        "Model": model_choice,
        "Total Return (%)": round(total_return_port, 2),
        "Max Drawdown (%)": round(port_mdd * 100, 2),
        "Ulcer Index (%)": round(ulcer_index, 2),
        "Martin Ratio": round(martin_ratio, 3),
        "Sharpe Ratio (Ann.)": round(sharpe, 3),
        "Sortino Ratio (Ann.)": round(sortino, 3),
        "Calmar Ratio": round(calmar, 3),
        "Average Turnover (%)": round(avg_turnover * 100, 2),
        "Rebalancing Count": len(rebalance_dates),
        "Skipped Rebalances": skipped_rebalances,
        "Tolerance Band (%)": round(tolerance_band * 100, 2) if tolerance_band is not None else "None",
        "Drawdown Guard (%)": round(drawdown_guard * 100, 2) if drawdown_guard is not None else "None",
        "Guard Triggers": guard_triggers,
        "Equal Weight Return (%)": round(total_return_eq, 2),
        "Equal Weight MDD (%)": round(eq_mdd * 100, 2),
        "Buy & Hold Return (%)": round(total_return_bh, 2),
        "Buy & Hold MDD (%)": round(bh_mdd * 100, 2),
    }

    return {
        "summary": summary,
        "nav_port": nav_port_series,
        "nav_eq": nav_eq_series,
        "nav_bh": nav_bh_series,
        "rebalance_dates": rebalance_dates,
        "weight_history": weight_history,
        "skipped_rebalances": skipped_rebalances,
        "guard_triggers": guard_triggers,
    }


def calculate_weight_drift(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
) -> float:
    """Calculate maximum absolute weight drift across any constituent asset."""
    c = np.asarray(current_weights, dtype=float)
    t = np.asarray(target_weights, dtype=float)
    if c.shape != t.shape:
        raise ValueError("Current weights and target weights must have matching dimensions.")
    return float(np.max(np.abs(c - t)))


def simulate_drift_band_rebalancing(
    prices: pd.DataFrame,
    band: float = 0.05,
    train_bars: int = 300,
    max_holding_bars: int = 100,
    fee_rate: float = 0.001,
    model_choice: str = "Equal Weight",
    target_weights: dict[str, float] | None = None,
) -> dict[str, object]:
    """Corridor-based portfolio rebalancing engine.

    Rebalancing is only triggered when weight drift exceeds the corridor band
    or when elapsed time exceeds max_holding_bars.
    """
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] < 2 or len(prices) < 2:
        raise ValueError("Drift band rebalancing requires at least two assets and two price rows.")
    if band <= 0 or band >= 1:
        raise ValueError("Band threshold must be strictly between 0 and 1.")
    if train_bars < 2:
        raise ValueError("Training window must be at least 2 bars.")
    if max_holding_bars <= 0:
        raise ValueError("Max holding bars must be positive.")

    returns = prices.pct_change().dropna()
    assets = list(returns.columns)
    n_bars = len(returns)

    if n_bars <= train_bars + 1:
        raise ValueError(f"Insufficient data: {n_bars} bars available, need at least {train_bars + 1}")

    test_start = train_bars
    test_dates = returns.index[test_start:]

    if target_weights is not None:
        target_vec = np.array([target_weights.get(a, 1.0 / len(assets)) for a in assets], dtype=float)
        target_vec = target_vec / np.sum(target_vec)
    else:
        target_vec = np.ones(len(assets)) / len(assets)

    current_weights = target_vec.copy()
    nav_portfolio = [1.0]
    nav_eq = [1.0]
    eq_weights = np.ones(len(assets)) / len(assets)

    rebalance_dates: list[pd.Timestamp] = []
    turnover_history: list[float] = []
    max_drift_observed = 0.0
    bars_since_rebalance = 0

    for idx, current_t in enumerate(range(test_start, n_bars)):
        date = returns.index[current_t]
        bar_ret = returns.iloc[current_t].values

        drift = calculate_weight_drift(current_weights, target_vec)
        max_drift_observed = max(max_drift_observed, drift)

        should_rebalance = (drift >= band) or (bars_since_rebalance >= max_holding_bars) or (idx == 0)

        if should_rebalance:
            rebalance_dates.append(date)
            # Rebalance to target
            turnover = float(np.sum(np.abs(target_vec - current_weights)))
            turnover_history.append(turnover)
            cost = turnover * fee_rate
            nav_portfolio[-1] *= (1.0 - cost)
            current_weights = target_vec.copy()
            bars_since_rebalance = 0
        else:
            bars_since_rebalance += 1

        # Realize portfolio return
        port_ret = float(np.dot(current_weights, bar_ret))
        next_nav = nav_portfolio[-1] * (1.0 + port_ret)
        nav_portfolio.append(next_nav)

        # Passive intra-period drift of actual holdings
        current_weights = current_weights * (1.0 + bar_ret)
        s = np.sum(current_weights)
        if s > 0:
            current_weights = current_weights / s

        # Equal weight comparison
        eq_ret = float(np.dot(eq_weights, bar_ret))
        nav_eq.append(nav_eq[-1] * (1.0 + eq_ret))

    nav_port_series = pd.Series(nav_portfolio[1:], index=test_dates, name=f"Corridor Band ({band*100:.1f}%)")
    nav_eq_series = pd.Series(nav_eq[1:], index=test_dates, name="Benchmark (Equal Weight)")

    port_mdd, _ = calculate_drawdown(nav_port_series)
    total_ret = (nav_port_series.iloc[-1] - 1.0) * 100.0
    eq_ret = (nav_eq_series.iloc[-1] - 1.0) * 100.0

    peak = nav_port_series.cummax()
    dd_pct = (nav_port_series - peak) / np.where(peak > 0, peak, 1e-9) * 100.0
    ui = float(np.sqrt(np.mean(np.square(dd_pct))))

    avg_turnover = float(np.mean(turnover_history)) if turnover_history else 0.0

    summary = {
        "Band (%)": round(band * 100.0, 2),
        "Total Return (%)": round(total_ret, 2),
        "Max Drawdown (%)": round(port_mdd * 100.0, 2),
        "Ulcer Index (%)": round(ui, 2),
        "Rebalance Triggers": len(rebalance_dates),
        "Max Drift Observed (%)": round(max_drift_observed * 100.0, 2),
        "Average Turnover (%)": round(avg_turnover * 100.0, 2),
        "Equal Weight Return (%)": round(eq_ret, 2),
    }

    return {
        "summary": summary,
        "nav_port": nav_port_series,
        "nav_eq": nav_eq_series,
        "rebalance_dates": rebalance_dates,
        "max_drift_observed": max_drift_observed,
    }


def print_backtest_report(summary: dict[str, object]):
    """Print clean terminal comparison table."""
    print("================================================================================")
    print("           PORTFOLIO REBALANCING WALK-FORWARD BACKTEST RESULTS                  ")
    print("================================================================================")
    print(f"{'Strategy / Benchmark':<30} | {'Total Return':>14} | {'Max Drawdown':>14}")
    print("--------------------------------------------------------------------------------")
    print(f"[*] {summary['Model'] + ' (Rebalanced)':<26} | {summary['Total Return (%)']:>13.2f}% | {summary['Max Drawdown (%)']:>13.2f}%")
    print(f"    {'Equal Weight (Rebalanced)':<26} | {summary['Equal Weight Return (%)']:>13.2f}% | {summary['Equal Weight MDD (%)']:>13.2f}%")
    print(f"    {'Buy & Hold Benchmark':<26} | {summary['Buy & Hold Return (%)']:>13.2f}% | {summary['Buy & Hold MDD (%)']:>13.2f}%")
    print("--------------------------------------------------------------------------------")
    print(f" - Annualized Sharpe Ratio  : {summary['Sharpe Ratio (Ann.)']}")
    print(f" - Annualized Sortino Ratio : {summary['Sortino Ratio (Ann.)']}")
    print(f" - Calmar Ratio             : {summary['Calmar Ratio']}")
    print(f" - Average Turnover Rate    : {summary['Average Turnover (%)']}% per rebalance")
    print(f" - Total Rebalance Events   : {summary['Rebalancing Count']} times")
    if summary.get("Skipped Rebalances", 0) > 0 or summary.get("Tolerance Band (%)") != "None":
        print(f" - Tolerance Band Drift     : {summary['Tolerance Band (%)']}%")
        print(f" - Skipped Low-Drift Events : {summary['Skipped Rebalances']} times (fee saved)")
    print("================================================================================\n")


def to_dict_rebalancing_result(result: dict[str, object]) -> dict[str, object]:
    """Convert simulate_rebalancing results to a JSON-serializable dictionary."""
    summary = result.get("summary", {})
    rebalance_dates = [
        d.isoformat() if hasattr(d, "isoformat") else str(d)
        for d in result.get("rebalance_dates", [])
    ]
    turnover_history = [
        round(float(t), 4) for t in result.get("turnover_history", [])
    ]

    out: dict[str, object] = {
        "summary": summary,
        "rebalance_count": len(rebalance_dates),
        "rebalance_dates": rebalance_dates,
        "turnover_history": turnover_history,
    }

    nav_port = result.get("nav_port")
    if isinstance(nav_port, pd.Series) and len(nav_port) > 0:
        out["portfolio_nav"] = {
            "initial": round(float(nav_port.iloc[0]), 4),
            "final": round(float(nav_port.iloc[-1]), 4),
            "peak": round(float(nav_port.max()), 4),
            "trough": round(float(nav_port.min()), 4),
        }
    return out


def export_rebalancing_json(result: dict[str, object], output_path: Path | str, indent: int = 2) -> None:
    """Export rebalancing results to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_dict_rebalancing_result(result)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Crypto Portfolio Rebalancing Backtest")
    parser.add_argument("--data-dir", type=str, default="", help="Directory with Freqtrade feather files")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe")
    parser.add_argument(
        "--model",
        type=str,
        default="Risk Parity",
        choices=["Risk Parity", "Max Sharpe", "Min Variance", "Min Semi-Variance", "Min CVaR", "HRP", "Schur", "Equal Weight"],
        help="Model to rebalance",
    )
    parser.add_argument("--train-bars", type=int, default=300, help="Lookback training window in bars")
    parser.add_argument("--rebalance-bars", type=int, default=50, help="Rebalancing frequency in bars")
    parser.add_argument("--fee", type=float, default=0.001, help="Transaction fee rate (0.001 = 0.1%%)")
    parser.add_argument("--tolerance-band", type=float, default=None, help="Drift threshold to execute rebalancing (e.g. 0.05 for 5%%)")
    parser.add_argument("--use-synthetic", action="store_true", help="Force synthetic data")
    parser.add_argument("--export-json", type=str, default=None, help="Path to export rebalancing backtest summary to JSON file.")
    args = parser.parse_args()

    try:
        prices, data_source = load_market_data(
            data_dir=args.data_dir or None,
            timeframe=args.timeframe,
            use_synthetic=args.use_synthetic,
            synthetic_periods=1000,
        )
    except MarketDataUnavailableError as error:
        parser.error(str(error))

    print(
        "[!] DATA SOURCE: SYNTHETIC (--use-synthetic was explicitly enabled)"
        if data_source == "synthetic"
        else f"[+] DATA SOURCE: REAL ({data_source}, {len(prices)} bars)"
    )

    res = simulate_rebalancing(
        prices=prices,
        train_bars=args.train_bars,
        rebalance_freq_bars=args.rebalance_bars,
        fee_rate=args.fee,
        model_choice=args.model,
        tolerance_band=args.tolerance_band,
    )

    print_backtest_report(res["summary"])

    if args.export_json:
        export_rebalancing_json(res, args.export_json)
        print(f"[+] Rebalancing backtest results exported to: {args.export_json}")


if __name__ == "__main__":
    main()
