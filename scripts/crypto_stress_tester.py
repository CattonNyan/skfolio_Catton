"""Cryptocurrency Historical Black Swan Stress Testing Engine.

Simulates the impact of severe historical crypto market shocks:
- 2020 March Corona Shock (Black Thursday)
- 2022 May Luna / UST Collapse
- 2022 November FTX Insolvency Event
- 2021 May China Mining Ban Panic
Evaluates portfolio drawdown, dollar loss, and risk resilience grade.
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

import pandas as pd


HISTORICAL_SHOCKS: dict[str, dict[str, float]] = {
    "2020 March Covid Crash": {
        "BTC": -0.38,
        "ETH": -0.44,
        "SOL": -0.45,
        "XRP": -0.35,
        "DEFAULT": -0.42,
    },
    "2022 May Luna/UST Collapse": {
        "BTC": -0.22,
        "ETH": -0.29,
        "SOL": -0.38,
        "XRP": -0.25,
        "DEFAULT": -0.32,
    },
    "2022 Nov FTX Insolvency": {
        "BTC": -0.23,
        "ETH": -0.24,
        "SOL": -0.58,  # Solana ecosystem heavy exposure
        "XRP": -0.19,
        "DEFAULT": -0.28,
    },
    "2021 May China Mining Ban": {
        "BTC": -0.34,
        "ETH": -0.41,
        "SOL": -0.36,
        "XRP": -0.42,
        "DEFAULT": -0.37,
    },
}


def evaluate_stress_test(
    weights: dict[str, float],
    total_wallet: float = 10000.0,
    custom_shock: dict[str, float] | None = None,
) -> dict[str, dict[str, float | str]]:
    """
    Simulate portfolio loss across predefined and custom shock scenarios.

    Returns dict mapping scenario name to performance metrics.
    """
    shocks = dict(HISTORICAL_SHOCKS)
    if custom_shock:
        shocks["Custom User Shock"] = custom_shock

    results: dict[str, dict[str, float | str]] = {}

    for scenario_name, asset_shocks in shocks.items():
        portfolio_shock = 0.0
        default_drop = asset_shocks.get("DEFAULT", -0.30)

        for pair, weight in weights.items():
            # Robust normalization for "BTC/USDT", "BTC_USDT", "BTC:USDT", or plain "BTC"
            normalized = pair.replace("_", "/").replace(":", "/")
            base_coin = normalized.split("/")[0].strip().upper()
            drop_rate = asset_shocks.get(base_coin, default_drop)
            portfolio_shock += weight * drop_rate

        loss_amount = abs(portfolio_shock * total_wallet)
        remaining_balance = max(0.0, total_wallet + (portfolio_shock * total_wallet))

        # Assign resilience grade based on max drawdown
        abs_drop = abs(portfolio_shock)
        if abs_drop <= 0.20:
            grade = "A (High Resilience)"
        elif abs_drop <= 0.30:
            grade = "B (Moderate Resilience)"
        elif abs_drop <= 0.40:
            grade = "C (Significant Impact)"
        else:
            grade = "D (Severe Vulnerability)"

        results[scenario_name] = {
            "portfolio_loss_pct": round(portfolio_shock * 100, 2),
            "dollar_loss": round(loss_amount, 2),
            "remaining_balance": round(remaining_balance, 2),
            "resilience_grade": grade,
        }

    return results


def print_stress_test_report(
    results: dict[str, dict[str, float | str]],
    total_wallet: float,
    weights: dict[str, float],
):
    """Print clean terminal report of the stress test simulation."""
    print("================================================================================")
    print("            HISTORICAL BLACK SWAN CRYPTO STRESS TEST REPORT                     ")
    print("================================================================================")
    print(f"Total Portfolio Capital: ${total_wallet:,.2f}")
    w_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in weights.items()])
    print(f"Asset Allocation Weights: {w_str}\n")
    print(f"{'Historical Scenario':<30} | {'Impact':>9} | {'Loss ($)':>12} | {'Remaining':>12} | {'Grade'}")
    print("--------------------------------------------------------------------------------")

    for name, m in results.items():
        pct = f"{m['portfolio_loss_pct']:+.2f}%"
        loss = f"-${m['dollar_loss']:,.2f}"
        rem = f"${m['remaining_balance']:,.2f}"
        print(f"{name:<30} | {pct:>9} | {loss:>12} | {rem:>12} | {m['resilience_grade']}")

    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Crypto Historical Stress Testing Engine")
    parser.add_argument("--wallet-size", type=float, default=10000.0, help="Total wallet value in USDT")
    parser.add_argument("--config-file", type=str, default="", help="Path to allocation or config JSON")
    parser.add_argument("--export-json", type=str, default="", help="Path to export results JSON")
    args = parser.parse_args()

    weights = {"BTC/USDT": 0.50, "ETH/USDT": 0.30, "SOL/USDT": 0.20}
    if args.config_file and Path(args.config_file).is_file():
        try:
            data = json.loads(Path(args.config_file).read_text(encoding="utf-8"))
            weights = data.get("pair_weights", data.get("weights", weights))
        except Exception as e:
            print(f"[!] Warning: Could not read {args.config_file}: {e}")

    results = evaluate_stress_test(weights, total_wallet=args.wallet_size)
    print_stress_test_report(results, total_wallet=args.wallet_size, weights=weights)

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[+] Stress test results exported to: {out_path}")


if __name__ == "__main__":
    main()
