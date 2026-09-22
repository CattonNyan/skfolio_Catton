"""Freqtrade Funding Rate and Basis Arbitrage Configuration Generator.

Generates production-grade Freqtrade configurations for delta-neutral and funding-carry strategies:
- Filters crypto futures pairs exceeding target annualized funding APR threshold
- Configures pair whitelists and per-pair stake budgets
- Injects futures-specific settings (margin_mode: cross/isolated, trading_mode: futures)
- Outputs standalone valid config JSON or injects into existing configuration
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


def filter_funding_rate_pairs(
    funding_rates_8h: dict[str, float],
    min_apr_pct: float = 10.0,
) -> dict[str, dict[str, float]]:
    """Filter perpetual pairs by annualized funding rate APR (8h rate * 3 * 365 * 100).

    Parameters
    ----------
    funding_rates_8h : dict[str, float]
        Mapping of pair symbol to 8-hour funding rate (e.g. 0.0003 = 0.03%).
    min_apr_pct : float
        Minimum annualized funding yield APR % to include.

    Returns
    -------
    dict of qualified pairs with 8h rate, daily rate %, and annualized APR %.
    """
    if not isinstance(funding_rates_8h, dict):
        raise ValueError("funding_rates_8h must be a dictionary.")
    if min_apr_pct < 0.0:
        raise ValueError("min_apr_pct must be non-negative.")

    qualified = {}
    for pair, rate_8h in funding_rates_8h.items():
        if not isinstance(rate_8h, (int, float)) or not isinstance(pair, str):
            continue
        daily_pct = rate_8h * 3.0 * 100.0
        apr_pct = daily_pct * 365.0
        if apr_pct >= min_apr_pct:
            qualified[pair] = {
                "rate_8h": float(rate_8h),
                "daily_rate_pct": round(daily_pct, 4),
                "annualized_apr_pct": round(apr_pct, 2),
            }

    # Sort descending by APR
    sorted_pairs = dict(sorted(qualified.items(), key=lambda item: item[1]["annualized_apr_pct"], reverse=True))
    return sorted_pairs


def generate_freqtrade_funding_config(
    qualified_pairs: list[str] | dict[str, dict[str, float]],
    stake_per_pair: float = 250.0,
    stake_currency: str = "USDT",
    exchange: str = "binance",
    dry_run: bool = True,
    margin_mode: str = "isolated",
) -> dict[str, object]:
    """Generate a clean Freqtrade futures config dictionary."""
    if isinstance(qualified_pairs, dict):
        pairs_list = list(qualified_pairs.keys())
    elif isinstance(qualified_pairs, list):
        pairs_list = qualified_pairs
    else:
        raise ValueError("qualified_pairs must be a list or dict.")

    if not pairs_list:
        raise ValueError("Cannot generate config with an empty pair list.")
    if stake_per_pair <= 0:
        raise ValueError("stake_per_pair must be strictly positive.")

    config = {
        "max_open_trades": len(pairs_list),
        "stake_currency": stake_currency.upper(),
        "stake_amount": stake_per_pair,
        "dry_run": dry_run,
        "trading_mode": "futures",
        "margin_mode": margin_mode,
        "exchange": {
            "name": exchange.lower(),
            "key": "",
            "secret": "",
            "pair_whitelist": pairs_list,
            "pair_blacklist": [],
        },
        "skfolio_funding_arbitrage": {
            "generated_by": "skfolio_Catton_v1.4.0",
            "pairs_count": len(pairs_list),
        },
    }
    return config


def export_funding_config(config: dict[str, object], output_path: Path | str) -> Path:
    """Save config dict to a formatted JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Freqtrade Funding Rate Config Generator.")
    parser.add_argument("--min-apr", type=float, default=12.0, help="Minimum funding APR %% threshold.")
    parser.add_argument("--stake", type=float, default=500.0, help="Stake per pair in USDT.")
    parser.add_argument("--export-json", type=str, default=None, help="Path to export generated Freqtrade funding config JSON.")
    args = parser.parse_args()

    sample_rates = {
        "BTC/USDT:USDT": 0.0001,   # ~10.95% APR
        "ETH/USDT:USDT": 0.00025,  # ~27.38% APR
        "SOL/USDT:USDT": 0.00035,  # ~38.33% APR
        "DOGE/USDT:USDT": 0.00045, # ~49.28% APR
        "LOW/USDT:USDT": 0.00005,  # ~5.48% APR (filtered out)
    }

    filtered = filter_funding_rate_pairs(sample_rates, min_apr_pct=args.min_apr)
    print(f"[*] Identified {len(filtered)} high-yield funding rate pairs (>= {args.min_apr}% APR):")
    for p, stats in filtered.items():
        print(f"  {p:<18}: 8h={stats['rate_8h']*100:.3f}% | APR={stats['annualized_apr_pct']:.2f}%")

    config = generate_freqtrade_funding_config(filtered, stake_per_pair=args.stake)
    print(f"[+] Successfully prepared Freqtrade futures config with {len(config['exchange']['pair_whitelist'])} pairs.")

    if args.export_json:
        export_funding_config(config, args.export_json)
        print(f"[+] Freqtrade funding config exported to: {args.export_json}")


if __name__ == "__main__":
    main()
