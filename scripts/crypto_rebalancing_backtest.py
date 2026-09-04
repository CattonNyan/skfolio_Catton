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
        MeanVariance,
        ObjectiveFunction,
        RiskBudgeting,
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
) -> dict[str, object]:
    """
    Run rolling-window walk-forward rebalancing backtest.

    Parameters:
    - prices: DataFrame of asset prices
    - train_bars: Lookback window to fit the model
    - rebalance_freq_bars: How often weights are recalculated
    - fee_rate: Transaction fee (e.g., 0.001 = 0.1% per turnover)
    - model_choice: "Risk Parity", "Max Sharpe", "Min Variance", or "Equal Weight"
    """
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
            new_weights = eq_weights.copy()
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
                    elif model_choice == "HRP":
                        m = HierarchicalRiskParity(risk_measure=RiskMeasure.VARIANCE)
                    else:  # Default to Risk Parity
                        m = RiskBudgeting(risk_measure=RiskMeasure.VARIANCE)

                    m.fit(window_returns)
                    new_weights = np.array(m.weights_)
                except Exception:
                    new_weights = current_weights.copy()

            # Calculate turnover and apply transaction fee
            turnover = float(np.sum(np.abs(new_weights - current_weights)))
            turnover_history.append(turnover)
            cost = turnover * fee_rate

            current_weights = new_weights
            weight_history.append(dict(zip(assets, current_weights)))

            # Deduct cost from portfolio NAV at rebalancing
            nav_portfolio[-1] *= (1.0 - cost)

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

    avg_turnover = float(np.mean(turnover_history)) if turnover_history else 0.0

    summary = {
        "Model": model_choice,
        "Total Return (%)": round(total_return_port, 2),
        "Max Drawdown (%)": round(port_mdd * 100, 2),
        "Sharpe Ratio (Ann.)": round(sharpe, 3),
        "Sortino Ratio (Ann.)": round(sortino, 3),
        "Calmar Ratio": round(calmar, 3),
        "Average Turnover (%)": round(avg_turnover * 100, 2),
        "Rebalancing Count": len(rebalance_dates),
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
    print(f" - Calmar Ratio (Ret / MDD) : {summary['Calmar Ratio']}")
    print(f" - Average Turnover Rate    : {summary['Average Turnover (%)']}% per rebalance")
    print(f" - Total Rebalance Events   : {summary['Rebalancing Count']} times")
    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Crypto Portfolio Rebalancing Backtest")
    parser.add_argument("--data-dir", type=str, default="", help="Directory with Freqtrade feather files")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe")
    parser.add_argument("--model", type=str, default="Risk Parity", choices=["Risk Parity", "Max Sharpe", "Min Variance", "HRP", "Equal Weight"], help="Model to rebalance")
    parser.add_argument("--train-bars", type=int, default=300, help="Lookback training window in bars")
    parser.add_argument("--rebalance-bars", type=int, default=50, help="Rebalancing frequency in bars")
    parser.add_argument("--fee", type=float, default=0.001, help="Transaction fee rate (0.001 = 0.1%%)")
    parser.add_argument("--use-synthetic", action="store_true", help="Force synthetic data")
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
    )

    print_backtest_report(res["summary"])


if __name__ == "__main__":
    main()
