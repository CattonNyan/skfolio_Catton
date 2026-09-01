"""Crypto Portfolio Optimization Pipeline for skfolio_Catton.

Loads cryptocurrency historical data (e.g. from Freqtrade feather files or CCXT/sample data),
computes asset returns, and runs portfolio optimization models using skfolio:
1. Maximum Sharpe Ratio (Mean-Variance)
2. Minimum Variance (Conservative)
3. Risk Parity / Equal Risk Contribution (Risk Budgeting)
4. Minimum Semi-Variance (Downside Risk Minimization)
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

# skfolio optimization models
try:
    from skfolio import RiskMeasure
    from skfolio.optimization import (
        MeanVariance,
        ObjectiveFunction,
        RiskBudgeting,
    )
    from skfolio.preprocessing import prices_to_returns
    HAS_SKFOLIO = True
except ImportError:
    HAS_SKFOLIO = False


def find_freqtrade_data_dirs() -> list[Path]:
    """Search for known Freqtrade data directories in workspace."""
    base_dirs = [
        Path("D:/private project/freqtrade-vibe-strategies/user_data/data/binance"),
        Path("D:/private project/freqtrade/user_data/data/binance"),
        Path("../freqtrade-vibe-strategies/user_data/data/binance"),
        Path("../freqtrade/user_data/data/binance"),
    ]
    return [d for d in base_dirs if d.is_dir()]


def load_from_feather_dir(data_dir: Path, timeframe: str = "15m") -> pd.DataFrame:
    """Load feather files for a specific timeframe and build a combined Close price DataFrame."""
    feather_files = list(data_dir.glob(f"*-{timeframe}.feather"))
    if not feather_files:
        feather_files = list(data_dir.glob("*.feather"))

    prices_dict: dict[str, pd.Series] = {}
    for f in feather_files:
        try:
            pair_name = f.stem.split("-")[0].replace("_", "/")
            df = pd.read_feather(f)
            if "date" in df.columns and "close" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
                prices_dict[pair_name] = df["close"]
        except Exception as err:
            print(f"[!] Warning reading {f.name}: {err}")

    if not prices_dict:
        return pd.DataFrame()

    prices_df = pd.DataFrame(prices_dict).dropna()
    return prices_df


def generate_synthetic_crypto_data(periods: int = 1000) -> pd.DataFrame:
    """Generate realistic synthetic crypto price movements for standalone testing."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq="15min")
    
    # Drift and volatility representative of major cryptocurrencies
    params = {
        "BTC/USDT": {"drift": 0.0001, "vol": 0.008, "start": 60000.0},
        "ETH/USDT": {"drift": 0.00015, "vol": 0.012, "start": 3000.0},
        "SOL/USDT": {"drift": 0.0002, "vol": 0.018, "start": 150.0},
        "XRP/USDT": {"drift": 0.00008, "vol": 0.014, "start": 0.60},
    }

    price_series = {}
    for symbol, p in params.items():
        ret = np.random.normal(p["drift"], p["vol"], periods)
        price_series[symbol] = p["start"] * np.cumprod(1 + ret)

    return pd.DataFrame(price_series, index=dates)


def run_optimization(prices: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Run multiple skfolio optimization models and return comparative results."""
    if not HAS_SKFOLIO:
        print("[!] Error: skfolio is not installed in the current environment.")
        print("    Please run setup.ps1 or pip install -r requirements-local.txt")
        return {}

    # Convert prices to returns
    returns = prices_to_returns(prices)
    assets = list(returns.columns)

    print(f"[*] Loaded asset returns: {assets}")
    print(f"[*] Sample size: {len(returns)} bars from {returns.index[0]} to {returns.index[-1]}\n")

    # 1. Mean-Variance: Maximum Sharpe Ratio
    model_sharpe = MeanVariance(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=RiskMeasure.VARIANCE,
    )
    model_sharpe.fit(returns)

    # 2. Mean-Variance: Minimum Variance
    model_min_var = MeanVariance(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.VARIANCE,
    )
    model_min_var.fit(returns)

    # 3. Risk Budgeting: Equal Risk Contribution (Risk Parity)
    model_risk_parity = RiskBudgeting(
        risk_measure=RiskMeasure.VARIANCE,
    )
    model_risk_parity.fit(returns)

    # 4. Minimum Semi-Variance (Downside Deviation Minimization)
    model_semi_variance = MeanVariance(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.SEMI_VARIANCE,
    )
    model_semi_variance.fit(returns)

    models = {
        "Max Sharpe Ratio": model_sharpe,
        "Min Variance": model_min_var,
        "Risk Parity (ERC)": model_risk_parity,
        "Min Semi-Variance": model_semi_variance,
    }

    results: dict[str, dict[str, float]] = {}

    print("================================================================================")
    print(f"{'Optimization Model':<22} | " + " | ".join([f"{a:>10}" for a in assets]))
    print("--------------------------------------------------------------------------------")

    for name, model in models.items():
        weights = model.weights_
        weight_str = " | ".join([f"{w*100:>9.2f}%" for w in weights])
        print(f"{name:<22} | {weight_str}")
        results[name] = dict(zip(assets, weights))

    print("================================================================================\n")

    # Evaluate Portfolios
    print("================ Portfolio Performance Summary ================")
    for name, model in models.items():
        portfolio = model.predict(returns)
        # Annualized or periodic statistics
        mean_ret = portfolio.mean * 100
        vol = portfolio.variance ** 0.5 * 100
        sharpe = portfolio.mean / (portfolio.variance ** 0.5 + 1e-9)
        print(f"[{name}]")
        print(f"  - Mean Return (per bar): {mean_ret:.4f}%")
        print(f"  - Volatility  (per bar): {vol:.4f}%")
        print(f"  - Return/Risk Ratio   : {sharpe:.4f}")

    return results


def export_freqtrade_allocation(
    results: dict[str, dict[str, float]],
    target_path: Path,
    model_name: str = "Risk Parity (ERC)",
    total_wallet: float | None = None,
) -> bool:
    """Export optimal asset allocation to Freqtrade config or JSON file."""
    if not results:
        return False

    if model_name not in results:
        model_name = list(results.keys())[0]

    weights = results[model_name]
    pair_whitelist = list(weights.keys())

    export_data = {
        "source_model": model_name,
        "pair_whitelist": pair_whitelist,
        "pair_weights": {k: round(float(v), 4) for k, v in weights.items()},
    }

    if total_wallet is not None:
        export_data["stake_amounts"] = {
            k: round(float(v) * total_wallet, 2) for k, v in weights.items()
        }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.suffix.lower() == ".json":
        # If target file exists and is a Freqtrade config, selectively update pair_whitelist and pair_weights
        if target_path.is_file():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    existing_config = json.load(f)
                if "exchange" in existing_config and isinstance(existing_config["exchange"], dict):
                    existing_config["exchange"]["pair_whitelist"] = pair_whitelist
                existing_config["pair_weights"] = export_data["pair_weights"]
                if "stake_amounts" in export_data:
                    existing_config["pair_stake_amounts"] = export_data["stake_amounts"]
                with open(target_path, "w", encoding="utf-8") as f:
                    json.dump(existing_config, f, indent=2)
                print(f"\n[+] Successfully updated existing Freqtrade config: {target_path}")
                return True
            except Exception as e:
                print(f"[!] Warning updating {target_path}: {e}. Writing standalone allocation file.")

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        print(f"\n[+] Exported Freqtrade allocation: {target_path}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Crypto Portfolio Optimizer via skfolio")
    parser.add_argument("--data-dir", type=str, default="", help="Directory containing Freqtrade feather files")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe (e.g., 5m, 15m)")
    parser.add_argument("--use-synthetic", action="store_true", help="Force synthetic sample crypto data")
    parser.add_argument("--export-freqtrade", type=str, default="", help="Path to export Freqtrade config or allocation JSON")
    parser.add_argument("--export-model", type=str, default="Risk Parity (ERC)", help="Model to use for export")
    parser.add_argument("--wallet-size", type=float, default=None, help="Optional wallet size in USDT for stake allocation")
    parser.add_argument("--export-html", type=str, default="", help="Path to export standalone HTML report")
    args = parser.parse_args()

    print("==========================================================")
    print("       skfolio Crypto Portfolio Optimizer Pipeline        ")
    print("==========================================================")

    prices = pd.DataFrame()

    if not args.use_synthetic:
        candidate_dirs = [Path(args.data_dir)] if args.data_dir else find_freqtrade_data_dirs()
        for d in candidate_dirs:
            if d.is_dir():
                print(f"[*] Reading market data from: {d}")
                prices = load_from_feather_dir(d, timeframe=args.timeframe)
                if not prices.empty:
                    print(f"[+] Successfully loaded pairs: {list(prices.columns)}")
                    break

    if prices.empty or args.use_synthetic:
        if not args.use_synthetic:
            print("[*] No Freqtrade market feather files found. Using realistic crypto sample data.")
        else:
            print("[*] Generating synthetic multi-asset crypto dataset.")
        prices = generate_synthetic_crypto_data()

    results = run_optimization(prices)

    if args.export_freqtrade and results:
        export_freqtrade_allocation(
            results=results,
            target_path=Path(args.export_freqtrade),
            model_name=args.export_model,
            total_wallet=args.wallet_size,
        )

    if args.export_html and results:
        from scripts.export_html_report import generate_html_report
        weights = results.get(args.export_model, list(results.values())[0])
        generate_html_report(
            prices=prices,
            weights=weights,
            model_name=args.export_model,
            total_wallet=args.wallet_size or 10000.0,
            output_file=args.export_html,
        )


if __name__ == "__main__":
    main()
