"""Crypto Statistical Arbitrage & Cointegration Pairs Trading Engine.

Implements Engle-Granger two-step cointegration, Ornstein-Uhlenbeck (OU) mean-reversion
half-life calculation, rolling Z-score spread tracking, and walk-forward execution simulation
with dynamic entry/exit thresholds, stop-loss protection, and exchange fee modeling.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

# Ensure local repository root is discoverable
root_dir = str(Path(__file__).resolve().parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd


@dataclass
class CointegrationResult:
    """Cointegration analysis results between two crypto assets."""

    asset_y: str
    asset_x: str
    hedge_ratio_beta: float
    intercept_alpha: float
    r_squared: float
    adf_statistic: float
    critical_values: dict[str, float]
    is_cointegrated: bool
    half_life_bars: float
    ou_theta: float
    mean_spread: float
    std_spread: float

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class PairsTradingSimulationResult:
    """Backtest simulation results for crypto pairs trading strategy."""

    asset_y: str
    asset_x: str
    hedge_ratio_beta: float
    half_life_bars: float
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    total_return_pct: float
    annualized_sharpe: float
    max_drawdown_pct: float
    avg_holding_bars: float
    entry_z: float
    exit_z: float
    stop_z: float
    fee_rate: float
    capital_initial: float
    capital_final: float

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to a JSON-serializable dictionary."""
        return asdict(self)


def calculate_cointegration(
    series_y: pd.Series | np.ndarray | list[float],
    series_x: pd.Series | np.ndarray | list[float],
    asset_y_name: str = "AssetY",
    asset_x_name: str = "AssetX",
    lags: int = 1,
) -> CointegrationResult:
    """Perform Engle-Granger cointegration test and calculate Ornstein-Uhlenbeck half-life.

    Step 1: OLS regression Y = alpha + beta * X + epsilon
    Step 2: Augmented Dickey-Fuller (ADF) test on residuals epsilon
    Step 3: Ornstein-Uhlenbeck process parameter estimation: dS = -theta * (S - mu) * dt + sigma * dW
            Half-life = ln(2) / theta
    """
    y = np.asarray(series_y, dtype=float)
    x = np.asarray(series_x, dtype=float)

    if len(y) != len(x):
        raise ValueError("Series X and Y must have identical length.")
    if len(y) < 15:
        raise ValueError("At least 15 observations required for cointegration estimation.")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        raise ValueError("Price series must contain only finite numeric values.")
    if np.any(y <= 0) or np.any(x <= 0):
        raise ValueError("Price series must contain strictly positive values.")

    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))
    if var_x < 1e-12 or var_y < 1e-12:
        raise ValueError("Price series must have non-zero variance.")

    # 1. OLS regression Y = alpha + beta * X
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    cov_xy = float(np.cov(x, y, ddof=1)[0, 1])

    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x

    # Residual spread
    spread = y - (alpha + beta * x)
    ss_res = float(np.sum((y - (alpha + beta * x)) ** 2))
    ss_tot = float(np.sum((y - mean_y) ** 2))
    r_squared = max(0.0, 1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0

    # 2. Ornstein-Uhlenbeck (OU) Mean-Reversion Half-Life
    # Delta S_t = a + b * S_{t-1} + e_t
    s_lag = spread[:-1]
    s_diff = np.diff(spread)

    var_s_lag = float(np.var(s_lag, ddof=1))
    if var_s_lag > 1e-12:
        cov_s = float(np.cov(s_lag, s_diff, ddof=1)[0, 1])
        b_ou = cov_s / var_s_lag
        if b_ou < -1e-6:
            theta = -b_ou
            half_life = float(np.log(2.0) / theta)
        else:
            theta = 0.0
            half_life = float("inf")
    else:
        theta = 0.0
        half_life = float("inf")

    # 3. Augmented Dickey-Fuller Unit Root Test on Residuals
    # Delta S_t = gamma * S_{t-1} + sum(delta_i * Delta S_{t-i}) + error
    n_obs = len(s_diff)
    p = max(1, min(lags, int(n_obs / 5)))

    # Construct design matrix for ADF
    # Target: s_diff[p:]
    y_adf = s_diff[p:]
    x_vars = [s_lag[p:]]
    for i in range(1, p + 1):
        x_vars.append(s_diff[p - i : n_obs - i])

    x_mat = np.column_stack(x_vars)
    # Add constant term
    x_mat = np.column_stack([np.ones(len(y_adf)), x_mat])

    try:
        # Solve OLS via least squares
        coeffs, residuals, rank, _ = np.linalg.lstsq(x_mat, y_adf, rcond=None)
        gamma = coeffs[1]  # coefficient on s_lag

        # Standard error of gamma
        fitted = x_mat @ coeffs
        resid = y_adf - fitted
        dof = len(y_adf) - x_mat.shape[1]
        sigma2 = float(np.sum(resid ** 2) / dof) if dof > 0 else 1.0
        xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
        se_gamma = math.sqrt(max(1e-12, sigma2 * xtx_inv[1, 1]))
        adf_stat = float(gamma / se_gamma)
    except Exception:
        adf_stat = 0.0

    # MacKinnon critical values for cointegration test with constant (2 variables)
    crit_vals = {
        "1%": -3.90,
        "5%": -3.34,
        "10%": -3.04,
    }
    is_coint = bool(adf_stat < crit_vals["5%"] and math.isfinite(half_life) and half_life > 0)

    mean_s = float(np.mean(spread))
    std_s = float(np.std(spread, ddof=1))

    return CointegrationResult(
        asset_y=asset_y_name,
        asset_x=asset_x_name,
        hedge_ratio_beta=round(beta, 5),
        intercept_alpha=round(alpha, 5),
        r_squared=round(r_squared, 4),
        adf_statistic=round(adf_stat, 3),
        critical_values=crit_vals,
        is_cointegrated=is_coint,
        half_life_bars=round(half_life, 2) if math.isfinite(half_life) else float("inf"),
        ou_theta=round(theta, 5),
        mean_spread=round(mean_s, 5),
        std_spread=round(std_s, 5),
    )


def calculate_rolling_zscore(
    spread: pd.Series | np.ndarray,
    window: int = 30,
) -> pd.Series:
    """Calculate rolling Z-score of spread: Z = (Spread - RollingMean) / RollingStd."""
    if isinstance(spread, np.ndarray):
        s = pd.Series(spread)
    else:
        s = spread.copy()

    if window < 2:
        raise ValueError("Rolling window must be at least 2.")

    r_mean = s.rolling(window=window, min_periods=max(2, window // 2)).mean()
    r_std = s.rolling(window=window, min_periods=max(2, window // 2)).std()

    # Avoid zero division
    zscore = (s - r_mean) / r_std.replace(0, np.nan)
    return zscore.fillna(0.0)


def simulate_pairs_trading(
    prices_y: pd.Series | np.ndarray | list[float],
    prices_x: pd.Series | np.ndarray | list[float],
    hedge_ratio: float | None = None,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    rolling_window: int = 30,
    fee_rate: float = 0.0006,
    slippage: float = 0.0002,
    initial_capital: float = 10000.0,
    asset_y_name: str = "AssetY",
    asset_x_name: str = "AssetX",
    bars_per_year: float = 365.25,
) -> dict[str, Any]:
    """Simulate walk-forward crypto pairs trading strategy.

    Positions:
      +1: LONG spread (Long AssetY, Short AssetX * beta) when Z <= -entry_z
      -1: SHORT spread (Short AssetY, Long AssetX * beta) when Z >= +entry_z
       0: FLAT / Cash when |Z| <= exit_z or |Z| >= stop_z
    """
    y = np.asarray(prices_y, dtype=float)
    x = np.asarray(prices_x, dtype=float)
    n = len(y)

    if n != len(x):
        raise ValueError("Price series must have identical length.")
    if n < rolling_window + 5:
        raise ValueError("Price length must exceed rolling_window + 5.")
    if entry_z <= 0 or exit_z < 0 or exit_z >= entry_z or stop_z <= entry_z:
        raise ValueError("Thresholds must satisfy: 0 <= exit_z < entry_z < stop_z.")
    if fee_rate < 0 or slippage < 0:
        raise ValueError("Fee and slippage rates must be non-negative.")

    # Calculate hedge ratio if omitted
    if hedge_ratio is None:
        coint_res = calculate_cointegration(y, x, asset_y_name, asset_x_name)
        beta = coint_res.hedge_ratio_beta
        half_life = coint_res.half_life_bars
    else:
        beta = float(hedge_ratio)
        coint_res = None
        half_life = float("nan")

    # Spread calculation
    spread = y - beta * x
    zscore = calculate_rolling_zscore(spread, window=rolling_window).to_numpy()

    position = 0  # 0: flat, +1: long spread, -1: short spread
    nav = initial_capital
    nav_history = [nav]
    trades: list[dict[str, Any]] = []

    pos_entry_idx = 0
    pos_entry_nav = nav

    # Calculate 1-period asset returns
    ret_y = np.diff(y) / y[:-1]
    ret_x = np.diff(x) / x[:-1]

    total_friction = (fee_rate + slippage) * 2.0  # 2 legs: AssetY and AssetX

    for i in range(rolling_window, n - 1):
        z = zscore[i]
        curr_ret_y = ret_y[i]
        curr_ret_x = ret_x[i]

        # Calculate spread return for holding position
        # Normalized by (1 + abs(beta)) to keep leverage bounded to 1.0x
        denom = 1.0 + abs(beta)
        spread_ret = (curr_ret_y - beta * curr_ret_x) / denom

        if position != 0:
            nav *= 1.0 + position * spread_ret

        new_pos = position

        # Check exit conditions first
        if position == 1:
            if z >= -exit_z:
                new_pos = 0  # Target profit exit
            elif z <= -stop_z:
                new_pos = 0  # Stoploss exit
        elif position == -1:
            if z <= exit_z:
                new_pos = 0  # Target profit exit
            elif z >= stop_z:
                new_pos = 0  # Stoploss exit

        # If flat, check entry conditions
        if position == 0:
            if z <= -entry_z:
                new_pos = 1  # Buy undervalued spread
            elif z >= entry_z:
                new_pos = -1  # Sell overvalued spread

        # Handle position state transition and fee deductions
        if new_pos != position:
            nav *= 1.0 - total_friction
            if position != 0:
                trade_profit_pct = (nav - pos_entry_nav) / pos_entry_nav * 100.0
                trades.append({
                    "entry_bar": pos_entry_idx,
                    "exit_bar": i,
                    "holding_bars": i - pos_entry_idx,
                    "direction": "LONG" if position == 1 else "SHORT",
                    "profit_pct": trade_profit_pct,
                })
            position = new_pos
            pos_entry_idx = i
            pos_entry_nav = nav

        nav_history.append(nav)

    # Close any open position at the end
    if position != 0:
        trade_profit_pct = (nav - pos_entry_nav) / pos_entry_nav * 100.0
        trades.append({
            "entry_bar": pos_entry_idx,
            "exit_bar": n - 1,
            "holding_bars": (n - 1) - pos_entry_idx,
            "direction": "LONG" if position == 1 else "SHORT",
            "profit_pct": trade_profit_pct,
        })

    # Summary statistics
    total_ret_pct = ((nav - initial_capital) / initial_capital) * 100.0
    wins = [t["profit_pct"] for t in trades if t["profit_pct"] > 0]
    losses = [abs(t["profit_pct"]) for t in trades if t["profit_pct"] < 0]
    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    pf = (sum(wins) / sum(losses)) if sum(losses) > 0 else (999.0 if sum(wins) > 0 else 0.0)
    avg_hold = (sum(t["holding_bars"] for t in trades) / len(trades)) if trades else 0.0

    nav_arr = np.array(nav_history)
    peak = np.maximum.accumulate(nav_arr)
    drawdowns = (nav_arr - peak) / peak
    mdd_pct = abs(float(np.min(drawdowns))) * 100.0

    # Annualized Sharpe
    nav_returns = np.diff(nav_arr) / nav_arr[:-1]
    if len(nav_returns) > 1 and np.std(nav_returns, ddof=1) > 1e-9:
        ann_sharpe = float((np.mean(nav_returns) / np.std(nav_returns, ddof=1)) * math.sqrt(bars_per_year))
    else:
        ann_sharpe = 0.0

    res = PairsTradingSimulationResult(
        asset_y=asset_y_name,
        asset_x=asset_x_name,
        hedge_ratio_beta=round(beta, 5),
        half_life_bars=round(half_life, 2) if math.isfinite(half_life) else 999.0,
        total_trades=len(trades),
        win_rate_pct=round(win_rate, 2),
        profit_factor=round(pf, 2),
        total_return_pct=round(total_ret_pct, 2),
        annualized_sharpe=round(ann_sharpe, 2),
        max_drawdown_pct=round(mdd_pct, 2),
        avg_holding_bars=round(avg_hold, 1),
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        fee_rate=fee_rate,
        capital_initial=round(initial_capital, 2),
        capital_final=round(nav, 2),
    )

    return {
        "metrics": res.to_dict(),
        "trades": trades,
        "nav_history": nav_history,
        "cointegration": coint_res.to_dict() if coint_res else None,
    }


def scan_crypto_pairs(
    prices_df: pd.DataFrame,
    min_r2: float = 0.30,
) -> list[dict[str, Any]]:
    """Scan all pairwise combinations in a multi-asset DataFrame for cointegration."""
    cols = list(prices_df.columns)
    results = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_y = cols[i]
            col_x = cols[j]
            y = prices_df[col_y].dropna()
            x = prices_df[col_x].dropna()

            # Align on common index
            common = y.index.intersection(x.index)
            if len(common) < 20:
                continue

            y_aligned = y.loc[common].to_numpy()
            x_aligned = x.loc[common].to_numpy()

            try:
                coint = calculate_cointegration(y_aligned, x_aligned, col_y, col_x)
                if coint.r_squared >= min_r2:
                    d = coint.to_dict()
                    results.append(d)
            except Exception:
                continue

    # Sort primarily by adf_statistic ascending (most negative = most statistically significant)
    results.sort(key=lambda item: item["adf_statistic"])
    return results


def export_pairs_trading_json(
    data: dict[str, Any] | list[dict[str, Any]],
    filepath: str | Path,
) -> Path:
    """Export pairs trading analysis or screener results to a JSON file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out_path


def export_pairs_trading_csv(
    data: dict[str, Any] | list[dict[str, Any]],
    filepath: str | Path,
) -> Path:
    """Export pairs trading analysis or screener results to a CSV file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif "metrics" in data:
        metrics = dict(data["metrics"])
        df = pd.DataFrame([metrics])
    else:
        df = pd.DataFrame([data])

    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def generate_synthetic_pairs_data(
    n_bars: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic cointegrated crypto pairs price series for simulation."""
    rng = np.random.default_rng(seed)
    # Underlying common stochastic driver (Geometric Brownian Motion)
    innovations = rng.normal(0.0005, 0.02, n_bars)
    log_x = np.cumsum(innovations) + np.log(50000.0)
    x = np.exp(log_x)

    # Stationary mean-reverting spread (Ornstein-Uhlenbeck)
    spread = np.zeros(n_bars)
    theta = 0.10  # fast mean-reversion
    mu = 0.0
    sigma = 0.015
    for t in range(1, n_bars):
        spread[t] = spread[t - 1] - theta * (spread[t - 1] - mu) + rng.normal(0, sigma)

    # Y is cointegrated with X: Y = 0.06 * X * exp(spread)
    y = 0.06 * x * np.exp(spread)
    return y, x


def main():
    parser = argparse.ArgumentParser(description="Crypto Statistical Arbitrage & Cointegration Pairs Trading Engine.")
    parser.add_argument("--symbol-a", type=str, default="ETH/USDT", help="Primary asset symbol (Y)")
    parser.add_argument("--symbol-b", type=str, default="BTC/USDT", help="Hedge asset symbol (X)")
    parser.add_argument("--entry-z", type=float, default=2.0, help="Entry threshold Z-score (default: 2.0)")
    parser.add_argument("--exit-z", type=float, default=0.5, help="Exit threshold Z-score (default: 0.5)")
    parser.add_argument("--stop-z", type=float, default=3.5, help="Stop-loss threshold Z-score (default: 3.5)")
    parser.add_argument("--window", type=int, default=30, help="Rolling Z-score lookback window (default: 30)")
    parser.add_argument("--fee", type=float, default=0.0006, help="Taker fee per leg (default: 0.06%%)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial allocation in quote currency")
    parser.add_argument("--export-json", type=str, default=None, help="Export backtest results to JSON")
    parser.add_argument("--export-csv", type=str, default=None, help="Export backtest results to CSV")
    parser.add_argument("--scan-example", action="store_true", help="Run multi-asset cointegration screening example")
    args = parser.parse_args()

    if args.scan_example:
        print("[*] Running multi-asset cointegration scanner on synthetic basket...")
        rng = np.random.default_rng(100)
        base = np.cumprod(1.0 + rng.normal(0.0005, 0.02, 300))
        df_basket = pd.DataFrame({
            "BTC/USDT": base * 60000.0,
            "ETH/USDT": base * 3200.0 * np.exp(np.sin(np.linspace(0, 10, 300)) * 0.05),
            "SOL/USDT": base * 150.0 * (1.0 + rng.normal(0, 0.03, 300)),
            "AVAX/USDT": base * 30.0 * (1.0 + rng.normal(0, 0.05, 300)),
        })
        scan_results = scan_crypto_pairs(df_basket)
        print("======== Cointegration Pairs Screener Results ========")
        for r in scan_results:
            coint_str = "[COINTEGRATED]" if r["is_cointegrated"] else "[WEAK]"
            print(f"{coint_str} {r['asset_y']} vs {r['asset_x']}:")
            print(f"  Hedge Ratio (Beta) : {r['hedge_ratio_beta']}")
            print(f"  ADF Statistic      : {r['adf_statistic']} (5% Crit: {r['critical_values']['5%']})")
            print(f"  OU Half-Life       : {r['half_life_bars']} bars | R²: {r['r_squared']:.3f}")
        print("======================================================")

        if args.export_json:
            export_pairs_trading_json(scan_results, args.export_json)
            print(f"[+] Screener JSON exported to {args.export_json}")
        if args.export_csv:
            export_pairs_trading_csv(scan_results, args.export_csv)
            print(f"[+] Screener CSV exported to {args.export_csv}")
        return

    print(f"[*] Simulating Pairs Trading: {args.symbol_a} vs {args.symbol_b}")
    y, x = generate_synthetic_pairs_data(n_bars=600, seed=42)

    sim = simulate_pairs_trading(
        prices_y=y,
        prices_x=x,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
        rolling_window=args.window,
        fee_rate=args.fee,
        initial_capital=args.capital,
        asset_y_name=args.symbol_a,
        asset_x_name=args.symbol_b,
    )

    m = sim["metrics"]
    c = sim["cointegration"]
    print("======== Pairs Trading Simulation Summary ========")
    print(f"Pair: {m['asset_y']} / {m['asset_x']}")
    print(f"  Hedge Ratio (Beta) : {m['hedge_ratio_beta']}")
    print(f"  OU Half-Life       : {m['half_life_bars']} bars")
    print(f"  Total Trades       : {m['total_trades']} (Win Rate: {m['win_rate_pct']:.1f}%)")
    print(f"  Profit Factor      : {m['profit_factor']:.2f}")
    print(f"  Total Net Return   : {m['total_return_pct']:+.2f}%")
    print(f"  Annualized Sharpe  : {m['annualized_sharpe']:.2f}")
    print(f"  Max Drawdown       : -{m['max_drawdown_pct']:.2f}%")
    print(f"  Avg Holding Period : {m['avg_holding_bars']:.1f} bars")
    print(f"  Final Capital      : ${m['capital_final']:,.2f} (from ${m['capital_initial']:,.2f})")
    if c:
        print(f"  ADF Statistic      : {c['adf_statistic']} (Cointegrated: {c['is_cointegrated']})")
    print("==================================================")

    if args.export_json:
        export_pairs_trading_json(sim, args.export_json)
        print(f"[+] Results exported to {args.export_json}")

    if args.export_csv:
        export_pairs_trading_csv(sim, args.export_csv)
        print(f"[+] Metrics CSV exported to {args.export_csv}")


if __name__ == "__main__":
    main()
