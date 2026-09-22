"""Hierarchical Risk Parity (HRP) and Correlation Clustering for Cryptocurrencies.

Implements Marcos Lopez de Prado's Hierarchical Risk Parity:
1. Tree clustering using correlation distance
2. Quasi-diagonalization of covariance matrix
3. Recursive bisection for robust risk budgeting without matrix inversion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure local skfolio source is discovered
src_dir = str(Path(__file__).resolve().parents[1] / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

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
        HierarchicalEqualRiskContribution,
        HierarchicalRiskParity,
        SchurComplementary,
    )
    from skfolio.preprocessing import prices_to_returns
    HAS_SKFOLIO_HRP = True
except ImportError:
    HAS_SKFOLIO_HRP = False


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute and format the pairwise correlation matrix."""
    return returns.corr()


def run_hrp_analysis(prices: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Run HRP and HERC models and print correlation & clustering analysis."""
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] < 2 or len(prices) < 3:
        raise ValueError("HRP analysis requires at least two assets and three price rows.")
    try:
        price_values = prices.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("HRP prices must contain only numeric values.") from error
    if not np.all(np.isfinite(price_values)) or np.any(price_values <= 0):
        raise ValueError("HRP prices must contain only finite, strictly positive values.")

    if not HAS_SKFOLIO_HRP:
        print("[!] Error: Hierarchical clustering modules not available in current environment.")
        return {}

    returns = prices_to_returns(prices)
    assets = list(returns.columns)

    print(f"[*] Assets: {assets}")
    print(f"[*] Data range: {returns.index[0]} ~ {returns.index[-1]} ({len(returns)} bars)\n")

    # 1. Pairwise Correlation Matrix
    corr = compute_correlation_matrix(returns)
    print("================ Pairwise Correlation Matrix ================")
    print(corr.round(3).to_string())
    print("=============================================================\n")

    # 2. Fit HRP (Hierarchical Risk Parity)
    model_hrp = HierarchicalRiskParity(
        risk_measure=RiskMeasure.VARIANCE,
    )
    model_hrp.fit(returns)

    # 3. Fit HERC (Hierarchical Equal Risk Contribution)
    model_herc = HierarchicalEqualRiskContribution(
        risk_measure=RiskMeasure.VARIANCE,
    )
    model_herc.fit(returns)

    # 4. Fit Schur (Peter Cotton Schur Complementary Allocation)
    model_schur = SchurComplementary()
    model_schur.fit(returns)

    models = {
        "HRP (Variance)": model_hrp,
        "HERC (Equal Risk)": model_herc,
        "Schur (Cotton)": model_schur,
    }

    results: dict[str, dict[str, float]] = {}

    print("================ Optimal Clustering Weights ================")
    print(f"{'Model':<20} | " + " | ".join([f"{a:>10}" for a in assets]))
    print("------------------------------------------------------------")

    for name, model in models.items():
        weights = model.weights_
        weight_str = " | ".join([f"{w*100:>9.2f}%" for w in weights])
        print(f"{name:<20} | {weight_str}")
        results[name] = dict(zip(assets, weights))

    print("============================================================\n")

    # 4. Out-of-sample / In-sample Performance
    for name, model in models.items():
        portfolio = model.predict(returns)
        mean_ret = portfolio.mean * 100
        vol = portfolio.variance ** 0.5 * 100
        sharpe = portfolio.mean / (portfolio.variance ** 0.5 + 1e-9)
        print(f"[{name} Portfolio]")
        print(f"  - Return per bar  : {mean_ret:.4f}%")
        print(f"  - Volatility      : {vol:.4f}%")
        print(f"  - Return/Risk     : {sharpe:.4f}")

    return results


def to_dict_hrp_result(
    results: dict[str, dict[str, float]],
    corr_matrix: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Serialize HRP allocation results and correlation matrix to a dictionary."""
    out: dict[str, object] = {
        "models": {
            model_name: {
                asset: round(float(weight), 6)
                for asset, weight in asset_weights.items()
            }
            for model_name, asset_weights in results.items()
        }
    }
    if corr_matrix is not None:
        out["correlation_matrix"] = {
            str(col): {
                str(row): round(float(corr_matrix.loc[row, col]), 4)
                for row in corr_matrix.index
            }
            for col in corr_matrix.columns
        }
    return out


def export_hrp_json(
    results: dict[str, dict[str, float]],
    output_path: Path | str,
    corr_matrix: pd.DataFrame | None = None,
    indent: int = 2,
) -> None:
    """Export HRP analysis results and correlation matrix to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_dict_hrp_result(results, corr_matrix=corr_matrix)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Crypto HRP & Clustering Optimizer")
    parser.add_argument("--data-dir", type=str, default="", help="Directory containing Freqtrade feather files")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe (e.g., 5m, 15m)")
    parser.add_argument("--use-synthetic", action="store_true", help="Force synthetic sample crypto data")
    parser.add_argument("--export-freqtrade", type=str, default="", help="Path to export Freqtrade config or allocation JSON")
    parser.add_argument("--export-json", type=str, default=None, help="Path to export HRP allocations and correlation matrix to JSON file.")
    args = parser.parse_args()

    print("==========================================================")
    print("  Hierarchical Risk Parity (HRP) Crypto Optimizer         ")
    print("==========================================================")

    try:
        prices, data_source = load_market_data(
            data_dir=args.data_dir or None,
            timeframe=args.timeframe,
            use_synthetic=args.use_synthetic,
        )
    except MarketDataUnavailableError as error:
        parser.error(str(error))

    print(
        "[!] DATA SOURCE: SYNTHETIC (--use-synthetic was explicitly enabled)"
        if data_source == "synthetic"
        else f"[+] DATA SOURCE: REAL ({data_source})"
    )

    if data_source == "synthetic" and args.export_freqtrade:
        parser.error("Freqtrade export is disabled for synthetic data.")

    results = run_hrp_analysis(prices)

    if args.export_freqtrade and results:
        from scripts.crypto_portfolio_optimizer import export_freqtrade_allocation
        export_freqtrade_allocation(
            results=results,
            target_path=Path(args.export_freqtrade),
            model_name="HRP (Variance)",
            data_source=data_source,
        )

    if args.export_json and results:
        returns = prices_to_returns(prices)
        corr = compute_correlation_matrix(returns)
        export_hrp_json(results, args.export_json, corr_matrix=corr)
        print(f"[+] HRP clustering results exported to: {args.export_json}")


if __name__ == "__main__":
    main()
