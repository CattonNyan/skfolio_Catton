"""Cryptocurrency Correlation Breakdown & Decoupling Anomaly Detector.

Monitors rolling correlation dynamics between a benchmark (e.g. BTC) and altcoins:
- Identifies sudden decoupling events (when an altcoin breaks away from BTC trend).
- Detects correlation regime shifts (Surge vs Plummet vs Inverse decoupling).
- Evaluates diversification enhancement opportunities when correlation drops significantly.
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


def detect_correlation_breakdown(
    prices: pd.DataFrame,
    benchmark: str | None = None,
    rolling_window: int = 30,
    z_threshold: float = 1.8,
) -> dict[str, dict[str, object]]:
    """
    Analyze rolling correlation changes against a benchmark asset.

    Parameters:
    - prices: Price DataFrame indexed by timestamp
    - benchmark: Anchor asset (default: 'BTC/USDT' or first column)
    - rolling_window: Number of bars for rolling window correlation
    - z_threshold: Z-score threshold to flag anomalous correlation breakdowns
    """
    if len(prices) < rolling_window + 5:
        raise ValueError(
            f"Prices length ({len(prices)}) is too short for rolling window ({rolling_window})."
        )

    returns = prices.pct_change().dropna()
    columns = list(returns.columns)

    if benchmark is None or benchmark not in columns:
        # Default to BTC/USDT or first column
        candidate_btc = [c for c in columns if "BTC" in c.upper()]
        benchmark = candidate_btc[0] if candidate_btc else columns[0]

    bench_returns = returns[benchmark]
    results: dict[str, dict[str, object]] = {}

    for asset in columns:
        if asset == benchmark:
            continue

        asset_returns = returns[asset]
        rolling_corr = asset_returns.rolling(window=rolling_window).corr(bench_returns).dropna()

        if rolling_corr.empty:
            continue

        current_corr = float(rolling_corr.iloc[-1])
        hist_mean = float(rolling_corr.mean())
        hist_std = float(rolling_corr.std() + 1e-9)
        z_score = float((current_corr - hist_mean) / hist_std)
        delta_corr = float(current_corr - hist_mean)

        # Classify state
        if current_corr < 0.0:
            status = "INVERSE_DECOUPLING (역상관 디커플링)"
            is_anomaly = True
        elif z_score <= -z_threshold:
            status = "CORRELATION_PLUMMET (상관관계 급락)"
            is_anomaly = True
        elif z_score >= z_threshold:
            status = "CORRELATION_SURGE (동조화 급증)"
            is_anomaly = True
        else:
            status = "STABLE (정상 상관)"
            is_anomaly = False

        # Diversification Benefit: Lower current correlation -> higher diversification value
        div_score = round(max(0.0, min(100.0, (1.0 - current_corr) * 50.0)), 2)

        results[asset] = {
            "benchmark": benchmark,
            "current_correlation": round(current_corr, 4),
            "historical_mean_corr": round(hist_mean, 4),
            "delta_correlation": round(delta_corr, 4),
            "z_score": round(z_score, 2),
            "status": status,
            "is_anomaly": is_anomaly,
            "diversification_score": div_score,
            "recent_series": [round(x, 4) for x in rolling_corr.iloc[-10:].tolist()],
        }

    return results


def print_breakdown_report(results: dict[str, dict[str, object]], benchmark: str):
    """Print clean terminal report of correlation decoupling analysis."""
    print("================================================================================")
    print(f"      CORRELATION BREAKDOWN & DECOUPLING MONITOR (Benchmark: {benchmark})       ")
    print("================================================================================")
    print(f"{'Asset':<14} | {'Current':>8} | {'Hist Mean':>10} | {'Z-Score':>8} | {'Div Score':>9} | {'Status'}")
    print("--------------------------------------------------------------------------------")

    for asset, data in results.items():
        curr_str = f"{data['current_correlation']:+.3f}"
        mean_str = f"{data['historical_mean_corr']:+.3f}"
        z_str = f"{data['z_score']:+.2f}"
        div_str = f"{data['diversification_score']:.1f}/100"
        flag = "[!]" if data["is_anomaly"] else "[ ]"
        status_str = f"{flag} {data['status']}"
        print(f"{asset:<14} | {curr_str:>8} | {mean_str:>10} | {z_str:>8} | {div_str:>9} | {status_str}")

    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Crypto Correlation Breakdown Detector")
    parser.add_argument("--benchmark", type=str, default=None, help="Benchmark asset (e.g. BTC/USDT)")
    parser.add_argument("--window", type=int, default=30, help="Rolling correlation window")
    parser.add_argument("--threshold", type=float, default=1.8, help="Anomaly Z-score threshold")
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
        prices = generate_synthetic_crypto_data(periods=150)

    res = detect_correlation_breakdown(
        prices=prices,
        benchmark=args.benchmark,
        rolling_window=args.window,
        z_threshold=args.threshold,
    )

    bench = args.benchmark or (prices.columns[0] if not prices.empty else "BTC/USDT")
    print_breakdown_report(res, benchmark=bench)

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[+] Correlation breakdown results exported to: {out_path}")


if __name__ == "__main__":
    main()
