"""Volatility-Targeting and Leverage Dynamic Allocator for Crypto Portfolios.

Implements institutional constant volatility targeting:
- Dynamically scales risky portfolio weights inversely proportional to rolling realized volatility
- Shifts allocation into cash/stablecoin during crypto flash crashes and high-volatility regimes
- Prevents deep drawdowns while compounding capital during low-volatility trend regimes
"""

from __future__ import annotations

import argparse
import json
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
class VolTargetResult:
    """Container for volatility targeting allocation result."""
    target_vol_ann: float
    realized_vol_ann: float
    vol_scalar: float
    scaled_weights: dict[str, float]
    cash_weight: float
    is_leveraged: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert volatility target result to serializable dictionary."""
        return {
            "target_vol_ann": self.target_vol_ann,
            "realized_vol_ann": self.realized_vol_ann,
            "vol_scalar": self.vol_scalar,
            "scaled_weights": dict(self.scaled_weights),
            "cash_weight": self.cash_weight,
            "is_leveraged": self.is_leveraged,
        }


def calculate_portfolio_realized_volatility(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    annual_factor: float = 365.0,
) -> float:
    """Calculate annualized realized volatility for a given portfolio weight mapping."""
    if returns_df.empty or len(returns_df) < 2:
        raise ValueError("Returns DataFrame must have at least 2 rows.")

    w = np.array([weights.get(c, 0.0) for c in returns_df.columns], dtype=float)
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()

    cov = returns_df.cov().values
    var = float(w @ cov @ w)
    if var <= 0:
        return 0.0
    return float(np.sqrt(var * annual_factor))


def apply_volatility_targeting(
    base_weights: dict[str, float],
    realized_vol_ann: float,
    target_vol_ann: float = 0.30,
    max_leverage: float = 1.0,
    min_scalar: float = 0.10,
) -> VolTargetResult:
    """Scale base portfolio weights by target volatility ratio.

    Scalar = clamp(target_vol / realized_vol, min_scalar, max_leverage)
    """
    if target_vol_ann <= 0.0 or max_leverage <= 0.0:
        raise ValueError("Target volatility and max leverage must be strictly positive.")
    if realized_vol_ann <= 0.0:
        # If realized volatility is near zero, clamp to max leverage
        scalar = max_leverage
    else:
        raw_scalar = target_vol_ann / realized_vol_ann
        scalar = max(min_scalar, min(float(raw_scalar), max_leverage))

    scaled = {}
    total_risky = 0.0
    for asset, w in base_weights.items():
        scaled_w = float(w * scalar)
        scaled[asset] = round(scaled_w, 4)
        total_risky += scaled_w

    cash_weight = max(0.0, round(1.0 - total_risky, 4)) if scalar <= 1.0 else 0.0

    return VolTargetResult(
        target_vol_ann=target_vol_ann,
        realized_vol_ann=round(realized_vol_ann, 4),
        vol_scalar=round(scalar, 4),
        scaled_weights=scaled,
        cash_weight=cash_weight,
        is_leveraged=scalar > 1.0,
    )


def simulate_vol_targeted_backtest(
    prices_df: pd.DataFrame,
    base_weights: dict[str, float],
    target_vol_ann: float = 0.35,
    lookback_bars: int = 30,
    max_leverage: float = 1.0,
    annual_factor: float = 365.0,
) -> dict[str, object]:
    """Simulate walk-forward volatility targeting portfolio vs unhedged static portfolio."""
    if len(prices_df) < lookback_bars + 2:
        raise ValueError("Prices must have more rows than lookback_bars.")

    returns = prices_df.pct_change().dropna()
    dates = returns.index[lookback_bars:]

    nav_static = [1.0]
    nav_targeted = [1.0]
    scalars = []

    for t in range(lookback_bars, len(returns)):
        window = returns.iloc[t - lookback_bars : t]
        realized_vol = calculate_portfolio_realized_volatility(window, base_weights, annual_factor)
        vt_res = apply_volatility_targeting(
            base_weights=base_weights,
            realized_vol_ann=realized_vol,
            target_vol_ann=target_vol_ann,
            max_leverage=max_leverage,
        )
        scalars.append(vt_res.vol_scalar)

        bar_ret = returns.iloc[t]
        static_ret = sum(base_weights.get(c, 0.0) * bar_ret[c] for c in returns.columns)
        targeted_ret = sum(vt_res.scaled_weights.get(c, 0.0) * bar_ret[c] for c in returns.columns)

        nav_static.append(nav_static[-1] * (1.0 + static_ret))
        nav_targeted.append(nav_targeted[-1] * (1.0 + targeted_ret))

    s_static = pd.Series(nav_static[1:], index=dates, name="Static Unhedged")
    s_targeted = pd.Series(nav_targeted[1:], index=dates, name="Vol-Targeted")

    # Max Drawdown
    def mdd(series: pd.Series) -> float:
        peak = series.cummax()
        return float(((series - peak) / peak).min() * 100.0)

    ret_static_daily = s_static.pct_change().dropna()
    ret_targeted_daily = s_targeted.pct_change().dropna()

    vol_static = float(ret_static_daily.std() * np.sqrt(annual_factor)) if len(ret_static_daily) > 1 else 0.0
    vol_targeted = float(ret_targeted_daily.std() * np.sqrt(annual_factor)) if len(ret_targeted_daily) > 1 else 0.0

    ret_static_pct = (s_static.iloc[-1] - 1.0) * 100.0
    ret_targeted_pct = (s_targeted.iloc[-1] - 1.0) * 100.0

    mdd_static = abs(mdd(s_static))
    mdd_targeted = abs(mdd(s_targeted))

    sharpe_static = (float(ret_static_daily.mean() * annual_factor) / vol_static) if vol_static > 1e-6 else 0.0
    sharpe_targeted = (float(ret_targeted_daily.mean() * annual_factor) / vol_targeted) if vol_targeted > 1e-6 else 0.0

    calmar_static = (ret_static_pct / mdd_static) if mdd_static > 1e-6 else 0.0
    calmar_targeted = (ret_targeted_pct / mdd_targeted) if mdd_targeted > 1e-6 else 0.0

    return {
        "nav_static": s_static,
        "nav_targeted": s_targeted,
        "mdd_static_pct": round(mdd(s_static), 2),
        "mdd_targeted_pct": round(mdd(s_targeted), 2),
        "return_static_pct": round(ret_static_pct, 2),
        "return_targeted_pct": round(ret_targeted_pct, 2),
        "sharpe_static": round(sharpe_static, 3),
        "sharpe_targeted": round(sharpe_targeted, 3),
        "calmar_static": round(calmar_static, 3),
        "calmar_targeted": round(calmar_targeted, 3),
        "mean_scalar": round(float(np.mean(scalars)), 3),
    }


def to_dict_vol_target_backtest(
    backtest_res: dict[str, object],
    include_nav: bool = False,
) -> dict[str, Any]:
    """Convert volatility target backtest results dictionary to serializable format."""
    out: dict[str, Any] = {
        "mdd_static_pct": float(backtest_res.get("mdd_static_pct", 0.0)),
        "mdd_targeted_pct": float(backtest_res.get("mdd_targeted_pct", 0.0)),
        "return_static_pct": float(backtest_res.get("return_static_pct", 0.0)),
        "return_targeted_pct": float(backtest_res.get("return_targeted_pct", 0.0)),
        "sharpe_static": float(backtest_res.get("sharpe_static", 0.0)),
        "sharpe_targeted": float(backtest_res.get("sharpe_targeted", 0.0)),
        "calmar_static": float(backtest_res.get("calmar_static", 0.0)),
        "calmar_targeted": float(backtest_res.get("calmar_targeted", 0.0)),
        "mean_scalar": float(backtest_res.get("mean_scalar", 1.0)),
    }
    if include_nav:
        s_static = backtest_res.get("nav_static")
        s_targeted = backtest_res.get("nav_targeted")
        if isinstance(s_static, pd.Series):
            out["nav_static"] = {str(k): round(float(v), 4) for k, v in s_static.items()}
        if isinstance(s_targeted, pd.Series):
            out["nav_targeted"] = {str(k): round(float(v), 4) for k, v in s_targeted.items()}
    return out


def export_vol_target_json(
    result: VolTargetResult | dict[str, Any],
    filepath: str | Path,
) -> None:
    """Export volatility targeting result or backtest dict to a JSON file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, VolTargetResult):
        payload = result.to_dict()
    elif isinstance(result, dict) and ("nav_static" in result or "nav_targeted" in result):
        payload = to_dict_vol_target_backtest(result)
    else:
        payload = result
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def export_vol_target_csv(
    result: VolTargetResult | dict[str, Any],
    filepath: str | Path,
) -> None:
    """Export volatility targeting result or backtest metrics to a CSV file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, VolTargetResult):
        d = result.to_dict()
        rows = []
        for k, v in d.items():
            if k == "scaled_weights" and isinstance(v, dict):
                for asset, w in v.items():
                    rows.append({"metric": f"weight_{asset}", "value": w})
            else:
                rows.append({"metric": k, "value": v})
    elif isinstance(result, dict) and ("nav_static" in result or "nav_targeted" in result):
        d = to_dict_vol_target_backtest(result, include_nav=False)
        rows = [{"metric": k, "value": v} for k, v in d.items()]
    elif isinstance(result, dict):
        rows = [{"metric": k, "value": v} for k, v in result.items() if not isinstance(v, (dict, list, pd.Series))]
    else:
        rows = []
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")


def export_vol_target_backtest_nav_csv(
    backtest_res: dict[str, object],
    filepath: str | Path,
) -> Path:
    """Export walk-forward simulation NAV time-series (Static vs Vol-Targeted) to a CSV file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s_static = backtest_res.get("nav_static")
    s_targeted = backtest_res.get("nav_targeted")
    if isinstance(s_static, pd.Series) and isinstance(s_targeted, pd.Series):
        df = pd.DataFrame({
            "nav_static": s_static,
            "nav_targeted": s_targeted,
        })
        df.index.name = "date"
        df.to_csv(out_path, index=True, encoding="utf-8")
    else:
        df = pd.DataFrame(columns=["date", "nav_static", "nav_targeted"])
        df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Volatility Targeting Allocator and Walk-Forward Backtester.")
    parser.add_argument("--target-vol", type=float, default=0.30, help="Target annualized volatility (e.g. 0.30 = 30%%).")
    parser.add_argument("--realized-vol", type=float, default=0.60, help="Current realized volatility (e.g. 0.60 = 60%%).")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run walk-forward volatility targeting simulation vs unhedged static portfolio.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Lookback window in bars/days for rolling realized volatility calculation (default: 30).",
    )
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=1.0,
        help="Maximum allowable leverage scalar (default: 1.0).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of simulation days for synthetic market walk (default: 180).",
    )
    parser.add_argument("--export-json", type=str, default=None, help="Export results to JSON file.")
    parser.add_argument("--export-csv", type=str, default=None, help="Export metrics summary to CSV file.")
    parser.add_argument("--export-nav-csv", type=str, default=None, help="Export walk-forward daily NAV series to CSV file.")
    args = parser.parse_args()

    base_w = {"BTC/USDT": 0.60, "ETH/USDT": 0.40}

    if args.simulate:
        dates = pd.date_range("2025-01-01", periods=args.days, freq="1D")
        np.random.seed(42)
        r_btc = np.random.normal(0.0005, 0.035, len(dates))
        r_eth = np.random.normal(0.0003, 0.045, len(dates))
        prices_df = pd.DataFrame({
            "BTC/USDT": 50000.0 * np.exp(np.cumsum(r_btc)),
            "ETH/USDT": 3000.0 * np.exp(np.cumsum(r_eth)),
        }, index=dates)

        sim_res = simulate_vol_targeted_backtest(
            prices_df,
            base_weights=base_w,
            target_vol_ann=args.target_vol,
            lookback_bars=args.lookback,
            max_leverage=args.max_leverage,
        )

        ret_s = float(sim_res["return_static_pct"])
        ret_t = float(sim_res["return_targeted_pct"])
        mdd_s = float(sim_res["mdd_static_pct"])
        mdd_t = float(sim_res["mdd_targeted_pct"])
        sh_s = float(sim_res["sharpe_static"])
        sh_t = float(sim_res["sharpe_targeted"])
        cal_s = float(sim_res["calmar_static"])
        cal_t = float(sim_res["calmar_targeted"])
        mean_scalar = float(sim_res["mean_scalar"])

        print("\n================== Walk-Forward Volatility Targeting Simulation ==================")
        print(f"Target Annual Vol: {args.target_vol * 100:.1f}% | Lookback: {args.lookback} bars | Max Leverage: {args.max_leverage:.2f}x")
        print("-" * 82)
        print(f"{'Metric':<28} {'Static (Unhedged)':>24} {'Vol-Targeted (Dynamic)':>26}")
        print("-" * 82)
        print(f"{'Total Return':<28} {ret_s:>+23.2f}% {ret_t:>+25.2f}%")
        print(f"{'Max Drawdown (MDD)':<28} {-abs(mdd_s):>23.2f}% {-abs(mdd_t):>25.2f}%")
        print(f"{'Annualized Sharpe':<28} {sh_s:>24.3f} {sh_t:>26.3f}")
        print(f"{'Calmar Ratio':<28} {cal_s:>24.3f} {cal_t:>26.3f}")
        print(f"{'Mean Leverage Scalar':<28} {'1.000x':>24} {f'{mean_scalar:.3f}x':>26}")
        print("=" * 82 + "\n")

        if args.export_json:
            export_vol_target_json(sim_res, args.export_json)
            print(f"[+] Simulation JSON exported to: {args.export_json}")

        if args.export_csv:
            export_vol_target_csv(sim_res, args.export_csv)
            print(f"[+] Simulation summary CSV exported to: {args.export_csv}")

        if args.export_nav_csv:
            export_vol_target_backtest_nav_csv(sim_res, args.export_nav_csv)
            print(f"[+] Simulation NAV time-series CSV exported to: {args.export_nav_csv}")
        return

    res = apply_volatility_targeting(base_w, realized_vol_ann=args.realized_vol, target_vol_ann=args.target_vol)

    print("================ Constant Volatility Targeting ================")
    print(f"  Target Volatility    : {res.target_vol_ann * 100:.1f}%")
    print(f"  Realized Volatility  : {res.realized_vol_ann * 100:.1f}%")
    print(f"  Volatility Scalar    : {res.vol_scalar:.3f}x")
    print(f"  Cash / USDT Buffer   : {res.cash_weight * 100:.1f}%")
    print(f"  Scaled Asset Weights : {res.scaled_weights}")
    print("================================================================")

    if args.export_json:
        export_vol_target_json(res, args.export_json)
        print(f"[+] Volatility targeting results exported to: {args.export_json}")

    if args.export_csv:
        export_vol_target_csv(res, args.export_csv)
        print(f"[+] Volatility targeting results exported to: {args.export_csv}")


if __name__ == "__main__":
    main()
