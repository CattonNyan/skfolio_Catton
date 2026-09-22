"""Cryptocurrency Liquidity Risk Filter and Estimator.

Provides empirical liquidity screening tools for crypto portfolio optimization:
- Amihud Illiquidity Ratio (price impact per unit of dollar volume)
- Corwin-Schultz (2012) High-Low Effective Spread estimator
- Rolling Dollar Volume Screener (filters out thin-book / illiquid altcoins)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass
class LiquidityMetrics:
    """Liquidity metrics container for a single cryptocurrency."""
    symbol: str
    mean_volume_usd: float
    median_volume_usd: float
    amihud_illiquidity: float
    estimated_spread_pct: float
    is_liquid: bool
    estimated_slippage_pct: float = 0.0
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert liquidity metrics to serializable dictionary."""
        return {
            "symbol": self.symbol,
            "mean_volume_usd": self.mean_volume_usd,
            "median_volume_usd": self.median_volume_usd,
            "amihud_illiquidity": None if (math.isinf(self.amihud_illiquidity) or math.isnan(self.amihud_illiquidity)) else self.amihud_illiquidity,
            "estimated_spread_pct": self.estimated_spread_pct,
            "estimated_slippage_pct": self.estimated_slippage_pct,
            "is_liquid": self.is_liquid,
            "rejection_reason": self.rejection_reason,
        }


def compute_amihud_illiquidity(
    returns: pd.Series | np.ndarray,
    dollar_volumes: pd.Series | np.ndarray,
    scale: float = 1e6,
) -> float:
    """Calculate Amihud (2002) Illiquidity Ratio.

    Illiquidity = (1/T) * sum(|R_t| / DollarVolume_t) * scale
    Higher value indicates higher price impact and lower liquidity.
    """
    ret_arr = np.asarray(returns, dtype=float)
    vol_arr = np.asarray(dollar_volumes, dtype=float)
    if len(ret_arr) != len(vol_arr) or len(ret_arr) == 0:
        raise ValueError("Returns and dollar volumes must be non-empty and have matching length.")

    # Filter out zero or negative volume bars to prevent division by zero
    valid_mask = (vol_arr > 0) & (~np.isnan(ret_arr)) & (~np.isnan(vol_arr))
    if not np.any(valid_mask):
        return float("inf")

    abs_ret = np.abs(ret_arr[valid_mask])
    vols = vol_arr[valid_mask]
    ratio = abs_ret / vols
    return float(np.mean(ratio) * scale)


def estimate_corwin_schultz_spread(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
) -> float:
    """Estimate bid-ask spread % using Corwin & Schultz (2012) High-Low estimator.

    Uses high and low prices over 2 consecutive periods.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    if len(h) != len(l) or len(h) < 2:
        return 0.0

    # Clean non-positive values
    valid = (h > 0) & (l > 0) & (h >= l)
    if np.sum(valid) < 2:
        return 0.0

    h = h[valid]
    l = l[valid]

    # Calculate beta over 2 consecutive periods: beta = sum_{i=0..1} (ln(H/L))^2
    log_hl = np.log(h / l)
    beta = log_hl[:-1] ** 2 + log_hl[1:] ** 2

    # Calculate gamma: (ln(max(H_t, H_{t+1}) / min(L_t, L_{t+1})))^2
    h2 = np.maximum(h[:-1], h[1:])
    l2 = np.minimum(l[:-1], l[1:])
    gamma = np.log(h2 / l2) ** 2

    k2 = 8.0 / np.pi
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / (3.0 - 2.0 * np.sqrt(2.0)) - np.sqrt(gamma / (3.0 - 2.0 * np.sqrt(2.0)))
    alpha = np.maximum(0.0, alpha)

    # Spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread_pct = float(np.nanmedian(spread) * 100.0)
    return max(0.0, spread_pct)


def filter_crypto_universe(
    ohlcv_data: dict[str, pd.DataFrame],
    min_mean_volume_usd: float = 10000.0,
    max_amihud: float = 5.0,
    max_spread_pct: float = 2.0,
    trade_size_usd: float = 10000.0,
    max_slippage_pct: float | None = None,
) -> tuple[list[str], dict[str, LiquidityMetrics]]:
    """Screen an OHLCV asset universe by liquidity thresholds.

    Parameters
    ----------
    ohlcv_data : dict[str, pd.DataFrame]
        Dictionary mapping symbols to DataFrames with columns: 'close', 'volume', optionally 'high', 'low'.
    min_mean_volume_usd : float
        Minimum average dollar volume required per bar.
    max_amihud : float
        Maximum allowed Amihud illiquidity ratio.
    max_spread_pct : float
        Maximum allowed estimated effective spread %.
    trade_size_usd : float
        Trade size in USD for estimating price impact and execution slippage.
    max_slippage_pct : float | None
        Optional maximum allowed estimated total slippage (half-spread + price impact).

    Returns
    -------
    liquid_symbols : list[str]
        Symbols passing all liquidity filters.
    metrics : dict[str, LiquidityMetrics]
        Detailed metrics for all evaluated assets.
    """
    if not ohlcv_data:
        return [], {}

    metrics_dict: dict[str, LiquidityMetrics] = {}
    liquid_symbols: list[str] = []

    for sym, df in ohlcv_data.items():
        if df.empty or len(df) < 5 or "close" not in df.columns or "volume" not in df.columns:
            metrics_dict[sym] = LiquidityMetrics(
                symbol=sym,
                mean_volume_usd=0.0,
                median_volume_usd=0.0,
                amihud_illiquidity=float("inf"),
                estimated_spread_pct=100.0,
                is_liquid=False,
                estimated_slippage_pct=100.0,
                rejection_reason="Insufficient data bars or missing required OHLCV columns.",
            )
            continue

        close = df["close"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)
        dollar_vol = close * volume
        ret = np.diff(close) / close[:-1]
        ret = np.concatenate([[0.0], ret])

        mean_vol = float(np.mean(dollar_vol))
        median_vol = float(np.median(dollar_vol))
        amihud = compute_amihud_illiquidity(ret, dollar_vol)

        if "high" in df.columns and "low" in df.columns:
            spread = estimate_corwin_schultz_spread(df["high"].to_numpy(), df["low"].to_numpy())
        else:
            spread = 0.0

        # Estimated one-way slippage = half spread + price impact (Amihud * trade_size / 1e6)
        estimated_slippage = (spread / 2.0) + (amihud * (trade_size_usd / 1e6)) if math.isfinite(amihud) else float("inf")

        # Evaluate filter criteria
        reasons = []
        if mean_vol < min_mean_volume_usd:
            reasons.append(f"Mean vol ${mean_vol:,.0f} < ${min_mean_volume_usd:,.0f}")
        if amihud > max_amihud:
            reasons.append(f"Amihud {amihud:.3f} > {max_amihud:.3f}")
        if spread > max_spread_pct:
            reasons.append(f"Spread {spread:.2f}% > {max_spread_pct:.2f}%")
        if max_slippage_pct is not None and estimated_slippage > max_slippage_pct:
            reasons.append(f"Est. slippage {estimated_slippage:.2f}% > {max_slippage_pct:.2f}% (for ${trade_size_usd:,.0f} trade)")

        is_liq = len(reasons) == 0
        metrics_dict[sym] = LiquidityMetrics(
            symbol=sym,
            mean_volume_usd=mean_vol,
            median_volume_usd=median_vol,
            amihud_illiquidity=amihud,
            estimated_spread_pct=spread,
            is_liquid=is_liq,
            estimated_slippage_pct=round(estimated_slippage, 4) if math.isfinite(estimated_slippage) else 999.0,
            rejection_reason="; ".join(reasons) if reasons else None,
        )
        if is_liq:
            liquid_symbols.append(sym)

    return liquid_symbols, metrics_dict


def main():
    parser = argparse.ArgumentParser(description="Cryptocurrency Liquidity Risk Filter and Screener.")
    parser.add_argument("--min-volume", type=float, default=50000.0, help="Minimum mean dollar volume per bar.")
    parser.add_argument("--max-amihud", type=float, default=2.0, help="Maximum allowed Amihud illiquidity.")
    parser.add_argument("--max-spread", type=float, default=1.5, help="Maximum allowed estimated spread percentage.")
    parser.add_argument("--trade-size", type=float, default=10000.0, help="Trade size in USD for slippage estimation.")
    parser.add_argument("--max-slippage", type=float, default=None, help="Maximum allowed estimated total slippage percentage.")
    parser.add_argument("--export-json", type=str, default=None, help="Export liquidity screening results to JSON file.")
    args = parser.parse_args()

    # Create dummy synthetic data for demonstration
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LOWCAP/USDT"]
    rng = np.random.default_rng(42)
    demo_data = {}
    for sym in symbols:
        base_price = 100.0 if "LOWCAP" not in sym else 0.05
        vol_mult = 10000.0 if "LOWCAP" not in sym else 100.0
        n_bars = 50
        ret = rng.normal(0.001, 0.03, n_bars)
        price = base_price * np.cumprod(1.0 + ret)
        high = price * (1.0 + rng.uniform(0.005, 0.02, n_bars))
        low = price * (1.0 - rng.uniform(0.005, 0.02, n_bars))
        volume = rng.uniform(vol_mult * 0.5, vol_mult * 2.0, n_bars)
        demo_data[sym] = pd.DataFrame({"close": price, "high": high, "low": low, "volume": volume})

    liquid_syms, report = filter_crypto_universe(
        demo_data,
        min_mean_volume_usd=args.min_volume,
        max_amihud=args.max_amihud,
        max_spread_pct=args.max_spread,
        trade_size_usd=args.trade_size,
        max_slippage_pct=args.max_slippage,
    )

    print(f"[*] Evaluated {len(demo_data)} assets. Liquid: {len(liquid_syms)}")
    for sym, m in report.items():
        status = "[PASS]" if m.is_liquid else "[FAIL]"
        print(f"  {status} {sym:<12}: Vol=${m.mean_volume_usd:,.0f} | Amihud={m.amihud_illiquidity:.4f} | Spread={m.estimated_spread_pct:.2f}% | Slippage={m.estimated_slippage_pct:.2f}% | Note: {m.rejection_reason or 'OK'}")

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_payload = {
            "liquid_symbols": liquid_syms,
            "metrics": {sym: m.to_dict() for sym, m in report.items()},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2, ensure_ascii=False)
        print(f"[+] Liquidity report exported to: {out_path}")


if __name__ == "__main__":
    main()
