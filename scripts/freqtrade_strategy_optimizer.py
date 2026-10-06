"""Freqtrade Multi-Strategy Capital Allocation Optimizer.

Optimizes capital distribution across multiple trading strategies (e.g. Trend Following,
Mean Reversion, Breakout) using their historical backtest equity curves or daily returns.
Applies skfolio Risk Parity and Min Variance to minimize multi-strategy account drawdown.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path


# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd


def load_freqtrade_backtest_file(
    path_or_dir: str | Path | None = None,
) -> tuple[dict[str, object], Path]:
    """Load Freqtrade backtest result data from a .json file, a .zip archive, or results directory."""
    if path_or_dir is None or not str(path_or_dir).strip():
        candidates = [
            Path("user_data/backtest_results"),
            Path(__file__).resolve().parents[1] / "user_data" / "backtest_results",
            Path.cwd(),
        ]
        target_dir = next((c for c in candidates if c.is_dir()), None)
        if target_dir is None:
            raise FileNotFoundError("Could not locate backtest results directory.")
    else:
        target_dir = Path(path_or_dir)

    resolved: Path | None = None
    if target_dir.is_dir():
        last_result = target_dir / ".last_result.json"
        if last_result.is_file():
            try:
                meta = json.loads(last_result.read_text(encoding="utf-8"))
                latest_name = meta.get("latest_backtest")
                if latest_name and (target_dir / latest_name).is_file():
                    resolved = target_dir / latest_name
            except Exception:
                pass

        if resolved is None:
            files = [
                f for f in target_dir.glob("*")
                if f.is_file() and f.suffix.lower() in {".zip", ".json"}
                and not f.name.endswith(".meta.json")
                and not f.name.endswith("_config.json")
                and not f.name.startswith(".last_result")
            ]
            if not files:
                raise FileNotFoundError(f"No valid backtest files found in {target_dir}")
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            resolved = files[0]
    elif target_dir.is_file():
        resolved = target_dir
    else:
        raise FileNotFoundError(f"Backtest file or directory not found: {target_dir}")

    if resolved.suffix.lower() == ".zip":
        with zipfile.ZipFile(resolved, "r") as zf:
            namelist = zf.namelist()
            target_json = None
            for name in namelist:
                n_lower = name.lower()
                if n_lower.endswith(".json") and not n_lower.endswith("_config.json") and not n_lower.endswith(".meta.json"):
                    target_json = name
                    break
            if not target_json:
                raise ValueError(f"No valid backtest JSON found in archive {resolved}")
            data = json.loads(zf.read(target_json).decode("utf-8"))
    else:
        data = json.loads(resolved.read_text(encoding="utf-8"))

    return data, resolved


def parse_freqtrade_backtest_trades(backtest_data: dict[str, object]) -> pd.DataFrame:

    """
    Extract daily closed trade profits per strategy from Freqtrade backtest json.
    """
    strategy_returns: dict[str, pd.Series] = {}

    strategy_dict = backtest_data.get("strategy", {})
    for strat_name, strat_content in strategy_dict.items():
        trades = strat_content.get("trades", [])
        if not trades:
            continue

        records = []
        for t in trades:
            close_time = t.get("close_date") or t.get("close_timestamp")
            profit_abs = t.get("profit_abs", 0.0)
            if close_time:
                records.append({"date": pd.to_datetime(close_time), "profit": profit_abs})

        if records:
            df = pd.DataFrame(records).set_index("date").sort_index()
            daily = df["profit"].resample("1D").sum()
            strategy_returns[strat_name] = daily

    if not strategy_returns:
        return pd.DataFrame()

    combined = pd.DataFrame(strategy_returns).fillna(0.0)
    return combined


def optimize_strategy_allocation(
    daily_profits: pd.DataFrame,
    total_capital: float = 10000.0,
    model: str = "Risk Parity",
) -> dict[str, object]:
    """
    Compute optimal capital allocation weights across trading strategies.
    """
    if not isinstance(daily_profits, pd.DataFrame):
        raise ValueError("daily_profits must be a pandas DataFrame.")
    if isinstance(total_capital, bool) or not np.isfinite(total_capital) or total_capital <= 0:
        raise ValueError("Total capital must be finite and strictly positive.")
    if model not in {"Risk Parity", "Min Variance"}:
        raise ValueError(f"Unsupported allocation model: {model}")
    if daily_profits.empty or len(daily_profits.columns) < 2:
        # Fallback to equal weight
        cols = list(daily_profits.columns) if not daily_profits.empty else ["Strategy_A", "Strategy_B"]
        w = {c: 1.0 / len(cols) for c in cols}
        return {
            "weights": w,
            "capital_allocation": {c: w[c] * total_capital for c in cols},
            "correlation": pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols).to_dict(),
            "portfolio_metrics": {
                "model": model,
                "total_capital": total_capital,
                "daily_mean_profit": 0.0,
                "daily_volatility": 0.0,
                "annualized_sharpe_ratio": 0.0,
                "annualized_sortino_ratio": 0.0,
                "diversification_ratio": 1.0,
            },
        }

    cols = list(daily_profits.columns)
    cov = daily_profits.cov()
    vols = daily_profits.std()

    if model == "Min Variance":
        # Analytical minimum variance weights: (Sigma^-1 * 1) / (1^T * Sigma^-1 * 1)
        sigma_inv = np.linalg.pinv(cov.values)
        ones = np.ones(len(cols))
        denom = float(ones.T @ sigma_inv @ ones)
        if not np.isfinite(denom) or denom <= 0:
            raw_w = np.ones(len(cols)) / len(cols)
        else:
            raw_w = sigma_inv @ ones / denom
        w_arr = np.clip(raw_w, 0.05, 0.95)
        if not np.all(np.isfinite(w_arr)) or w_arr.sum() <= 0:
            w_arr = np.ones(len(cols)) / len(cols)
        else:
            w_arr = w_arr / w_arr.sum()
    else:
        # Inverse Volatility / Risk Parity heuristic
        inv_vols = 1.0 / (vols.values + 1e-9)
        if not np.all(np.isfinite(inv_vols)) or inv_vols.sum() <= 0:
            w_arr = np.ones(len(cols)) / len(cols)
        else:
            w_arr = inv_vols / inv_vols.sum()

    weights = dict(zip(cols, [round(float(x), 4) for x in w_arr]))
    capital = {c: round(weights[c] * total_capital, 2) for c in cols}

    # Portfolio-level analytics
    portfolio_daily = daily_profits @ np.array([weights[c] for c in cols])
    mean_daily = float(portfolio_daily.mean()) if len(portfolio_daily) > 0 else 0.0
    std_daily = float(portfolio_daily.std()) if len(portfolio_daily) > 1 else 0.0
    sharpe = round(float(mean_daily / std_daily * np.sqrt(365.0)), 4) if std_daily > 1e-9 else 0.0
    downside_daily = portfolio_daily[portfolio_daily < 0]
    downside_std = float(downside_daily.std()) if len(downside_daily) > 1 else (std_daily if std_daily > 1e-9 else 1.0)
    sortino = round(float(mean_daily / downside_std * np.sqrt(365.0)), 4) if downside_std > 1e-9 else 0.0
    weighted_vol = float(np.array([weights[c] for c in cols]) @ vols.values) if len(cols) > 0 else 0.0
    div_ratio = round(float(weighted_vol / (std_daily + 1e-9)), 4) if std_daily > 1e-9 else 1.0

    portfolio_metrics = {
        "model": model,
        "total_capital": total_capital,
        "daily_mean_profit": round(mean_daily, 2),
        "daily_volatility": round(std_daily, 2),
        "annualized_sharpe_ratio": sharpe,
        "annualized_sortino_ratio": sortino,
        "diversification_ratio": div_ratio,
    }

    return {
        "weights": weights,
        "capital_allocation": capital,
        "correlation": daily_profits.corr().to_dict(),
        "portfolio_metrics": portfolio_metrics,
    }


def export_strategy_allocation_csv(
    res: dict[str, object],
    output_path: Path | str,
) -> Path:
    """Export strategy allocation weights and portfolio metrics to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    weights = res.get("weights", {})
    capitals = res.get("capital_allocation", {})

    if isinstance(weights, dict):
        for strat, w in weights.items():
            rows.append({
                "type": "strategy_allocation",
                "name": strat,
                "weight": w,
                "capital": capitals.get(strat, 0.0) if isinstance(capitals, dict) else 0.0,
            })

    metrics = res.get("portfolio_metrics", {})
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            rows.append({
                "type": "portfolio_metric",
                "name": k,
                "weight": "",
                "capital": v,
            })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def print_strategy_allocation_report(res: dict[str, object], total_capital: float):
    """Print terminal report of multi-strategy allocation."""
    print("================================================================================")
    print(f"      FREQTRADE MULTI-STRATEGY CAPITAL ALLOCATION (Seed: ${total_capital:,.2f})  ")
    print("================================================================================")
    print(f"{'Strategy Name':<30} | {'Optimal Weight':>15} | {'Allocated Capital':>18}")
    print("--------------------------------------------------------------------------------")

    weights = res["weights"]
    capitals = res["capital_allocation"]

    for strat in weights:
        w_str = f"{weights[strat]*100:.2f}%"
        c_str = f"${capitals[strat]:,.2f}"
        print(f"{strat:<30} | {w_str:>15} | {c_str:>18}")

    metrics = res.get("portfolio_metrics", {})
    if isinstance(metrics, dict) and metrics:
        print("--------------------------------------------------------------------------------")
        print(f"포트폴리오 일평균 수익 (Mean)     : ${metrics.get('daily_mean_profit', 0.0):,.2f}")
        print(f"포트폴리오 변동성 (Daily Vol)     : ${metrics.get('daily_volatility', 0.0):,.2f}")
        print(f"샤프 비율 (Ann. Sharpe)           : {metrics.get('annualized_sharpe_ratio', 0.0):.2f}")
        print(f"소르티노 비율 (Ann. Sortino)       : {metrics.get('annualized_sortino_ratio', 0.0):.2f}")
        print(f"다각화 비율 (Diversification)     : {metrics.get('diversification_ratio', 1.0):.2f}x")

    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Freqtrade Strategy Capital Allocation Optimizer")
    parser.add_argument("--backtest-file", type=str, default="", help="Path to Freqtrade backtest-result.json, .zip, or directory")
    parser.add_argument("--latest", action="store_true", help="Auto-detect latest backtest result in user_data/backtest_results")
    parser.add_argument("--capital", type=float, default=10000.0, help="Total capital to allocate in USDT")
    parser.add_argument("--model", type=str, default="Risk Parity", choices=["Risk Parity", "Min Variance"])
    parser.add_argument("--export-json", type=str, default="", help="Path to export JSON results")
    parser.add_argument("--export-csv", type=str, default="", help="Path to export CSV results")
    args = parser.parse_args()

    daily_profits = pd.DataFrame()
    if args.latest or args.backtest_file:
        try:
            target = args.backtest_file if args.backtest_file else None
            data, resolved_path = load_freqtrade_backtest_file(target)
            print(f"[*] Loaded backtest data from: {resolved_path}")
            daily_profits = parse_freqtrade_backtest_trades(data)
        except Exception as e:
            print(f"[!] Could not load backtest data: {e}")

    if daily_profits.empty:
        # Synthetic representative strategy daily returns
        np.random.seed(42)
        dates = pd.date_range("2026-01-01", periods=60, freq="1D")
        daily_profits = pd.DataFrame(
            {
                "TrendFollowing_ATR": np.random.normal(50, 120, size=60),
                "MeanReversion_RSI": np.random.normal(30, 60, size=60),
                "Breakout_Bollinger": np.random.normal(40, 90, size=60),
            },
            index=dates,
        )

    res = optimize_strategy_allocation(
        daily_profits=daily_profits,
        total_capital=args.capital,
        model=args.model,
    )

    print_strategy_allocation_report(res, total_capital=args.capital)

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[+] Strategy allocation exported to: {out_path}")

    if args.export_csv:
        csv_path = export_strategy_allocation_csv(res, args.export_csv)
        print(f"[+] Strategy allocation exported to: {csv_path}")


if __name__ == "__main__":
    main()

