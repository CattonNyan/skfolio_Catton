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
import math
import sys
from pathlib import Path
from typing import Any

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
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
    "2024 August Yen Carry Trade Unwind": {
        "BTC": -0.18,
        "ETH": -0.25,
        "SOL": -0.28,
        "XRP": -0.20,
        "DEFAULT": -0.24,
    },
    "2023 March SVB USDC Depeg Crisis": {
        "BTC": -0.10,
        "ETH": -0.12,
        "SOL": -0.15,
        "XRP": -0.08,
        "DEFAULT": -0.12,
    },
    "2024 October Geopolitical De-escalation Shock": {
        "BTC": -0.08,
        "ETH": -0.12,
        "SOL": -0.15,
        "XRP": -0.10,
        "DEFAULT": -0.14,
    },
    "DeFi Liquidity Cascade & Flash Crash": {
        "BTC": -0.18,
        "ETH": -0.26,
        "SOL": -0.32,
        "XRP": -0.24,
        "DEFAULT": -0.35,
    },
    "2018 Jan Crypto Winter Freeze": {
        "BTC": -0.65,
        "ETH": -0.68,
        "SOL": -0.70,
        "XRP": -0.74,
        "DEFAULT": -0.70,
    },
    "2021 Nov ATH Liquidation Cascade": {
        "BTC": -0.32,
        "ETH": -0.35,
        "SOL": -0.42,
        "XRP": -0.30,
        "DEFAULT": -0.35,
    },
    "DeFi Staking & Bridge Exploit Shock": {
        "BTC": -0.15,
        "ETH": -0.30,
        "SOL": -0.45,
        "XRP": -0.18,
        "DEFAULT": -0.28,
    },
}


def evaluate_stress_test(
    weights: dict[str, float],
    total_wallet: float = 10000.0,
    custom_shock: dict[str, float] | None = None,
    cash_buffer: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """
    Simulate portfolio loss across predefined and custom shock scenarios.

    Parameters:
    - weights: Mapping of asset symbol / pair to portfolio weight.
    - total_wallet: Total portfolio capital in USD/USDT.
    - custom_shock: Optional custom scenario asset shock rate dictionary.
    - cash_buffer: Optional cash/stablecoin hedging buffer fraction [0.0, 1.0)
      that experiences 0.0% shock in stress scenarios.

    Returns dict mapping scenario name to performance metrics.
    """
    if isinstance(total_wallet, bool) or not math.isfinite(total_wallet) or total_wallet <= 0:
        raise ValueError("total_wallet must be finite and strictly positive.")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("weights must be a non-empty dictionary.")
    if (
        isinstance(cash_buffer, bool)
        or not isinstance(cash_buffer, (int, float))
        or not math.isfinite(cash_buffer)
        or not (0.0 <= cash_buffer < 1.0)
    ):
        raise ValueError("cash_buffer must be a finite number in the range [0.0, 1.0).")

    # Validate and normalize weights to guarantee sum == 1.0
    clean_weights = {}
    for k, v in weights.items():
        try:
            w_val = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(w_val):
            raise ValueError(f"Weight for {k} must be finite.")
        if w_val > 0:
            clean_weights[k] = w_val
    if not clean_weights:
        raise ValueError("No valid positive weights found.")
    total_w = sum(clean_weights.values())
    norm_weights = {k: v / total_w for k, v in clean_weights.items()}

    shocks = dict(HISTORICAL_SHOCKS)
    if custom_shock is not None:
        if not isinstance(custom_shock, dict) or not custom_shock:
            raise ValueError("custom_shock must be a non-empty dictionary.")
        clean_custom = {}
        for coin, drop in custom_shock.items():
            if not isinstance(coin, str) or not coin.strip():
                raise ValueError("custom_shock asset symbols must be non-empty strings.")
            if isinstance(drop, bool) or not isinstance(drop, (int, float)) or not math.isfinite(drop):
                raise ValueError(f"custom_shock value for {coin} must be a finite number.")
            if drop < -1.0:
                raise ValueError(f"custom_shock drop rate for {coin} cannot exceed -100% (-1.0).")
            clean_custom[coin.strip().upper()] = float(drop)
        shocks["Custom User Shock"] = clean_custom

    results: dict[str, dict[str, Any]] = {}
    risky_multiplier = 1.0 - cash_buffer

    for scenario_name, asset_shocks in shocks.items():
        unhedged_shock = 0.0
        default_drop = asset_shocks.get("DEFAULT", -0.30)

        for pair, weight in norm_weights.items():
            # Robust normalization for "BTC/USDT", "BTC_USDT", "BTC:USDT", or plain "BTC"
            normalized = pair.replace("_", "/").replace(":", "/")
            base_coin = normalized.split("/")[0].strip().upper()
            drop_rate = asset_shocks.get(base_coin, default_drop)
            unhedged_shock += weight * drop_rate

        portfolio_shock = unhedged_shock * risky_multiplier
        loss_amount = abs(portfolio_shock * total_wallet)
        remaining_balance = max(0.0, total_wallet + (portfolio_shock * total_wallet))
        unhedged_dollar_loss = abs(unhedged_shock * total_wallet)
        capital_saved = max(0.0, unhedged_dollar_loss - loss_amount)

        # Asset-level loss contributions
        asset_breakdown = {}
        primary_risk_driver = ""
        max_driver_impact = -1.0

        for pair, weight in norm_weights.items():
            normalized = pair.replace("_", "/").replace(":", "/")
            base_coin = normalized.split("/")[0].strip().upper()
            drop_rate = asset_shocks.get(base_coin, default_drop)
            eff_weight = weight * risky_multiplier
            contrib_pct = eff_weight * drop_rate
            dollar_impact = abs(contrib_pct * total_wallet)
            loss_share = (contrib_pct / portfolio_shock * 100) if portfolio_shock != 0 else 0.0

            asset_breakdown[pair] = {
                "weight": round(eff_weight, 4),
                "asset_drop_pct": round(drop_rate * 100, 2),
                "contribution_pct": round(contrib_pct * 100, 2),
                "dollar_impact": round(dollar_impact, 2),
                "loss_share_pct": round(loss_share, 2),
            }

            if abs(contrib_pct) > max_driver_impact:
                max_driver_impact = abs(contrib_pct)
                primary_risk_driver = pair

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

        # Break-even return required to recover to initial capital
        recovery_req = (abs_drop / (1.0 - abs_drop) * 100) if abs_drop < 1.0 else 9999.0

        results[scenario_name] = {
            "portfolio_loss_pct": round(portfolio_shock * 100, 2),
            "dollar_loss": round(loss_amount, 2),
            "remaining_balance": round(remaining_balance, 2),
            "recovery_required_pct": round(recovery_req, 2),
            "resilience_grade": grade,
            "primary_risk_driver": primary_risk_driver,
            "unhedged_loss_pct": round(unhedged_shock * 100, 2),
            "capital_saved_dollars": round(capital_saved, 2),
            "cash_buffer_pct": round(cash_buffer * 100, 2),
            "asset_breakdown": asset_breakdown,
        }

    return results


def summarize_stress_test_results(
    results: dict[str, dict[str, Any]],
) -> dict[str, float | str | int]:
    """Compute aggregate worst-case loss, average drawdown, sVaR, sES, and stress summary."""
    if not results:
        return {
            "worst_scenario": "",
            "worst_loss_pct": 0.0,
            "worst_dollar_loss": 0.0,
            "avg_loss_pct": 0.0,
            "median_loss_pct": 0.0,
            "svar_95_pct": 0.0,
            "ses_95_pct": 0.0,
            "scenarios_tested": 0,
            "overall_resilience": "A (High Resilience)",
            "primary_vulnerability": "",
            "total_capital_saved": 0.0,
        }

    scenarios = list(results.items())
    worst_item = min(scenarios, key=lambda s: float(s[1]["portfolio_loss_pct"]))
    worst_scenario_name, worst_metrics = worst_item
    worst_loss_pct = float(worst_metrics["portfolio_loss_pct"])
    worst_dollar = float(worst_metrics["dollar_loss"])

    losses = [float(s[1]["portfolio_loss_pct"]) for s in scenarios]
    avg_loss = sum(losses) / len(losses)
    median_loss = float(np.median(losses))

    # Stress VaR (95%) and Stress Expected Shortfall (sES 95%)
    # Losses are negative percentages; 5th percentile return = 95th percentile stress loss severity
    svar_95 = float(np.percentile(losses, 5.0))
    tail_losses = [l for l in losses if l <= svar_95]
    ses_95 = float(np.mean(tail_losses)) if tail_losses else svar_95

    abs_worst = abs(worst_loss_pct) / 100.0
    if abs_worst <= 0.20:
        overall_grade = "A (High Resilience)"
    elif abs_worst <= 0.30:
        overall_grade = "B (Moderate Resilience)"
    elif abs_worst <= 0.40:
        overall_grade = "C (Significant Impact)"
    else:
        overall_grade = "D (Severe Vulnerability)"

    # Count most frequent primary risk driver
    drivers = [str(s[1].get("primary_risk_driver", "")) for s in scenarios if s[1].get("primary_risk_driver")]
    primary_vulnerability = max(set(drivers), key=drivers.count) if drivers else ""

    total_saved = sum(float(s[1].get("capital_saved_dollars", 0.0)) for s in scenarios)

    return {
        "worst_scenario": worst_scenario_name,
        "worst_loss_pct": round(worst_loss_pct, 2),
        "worst_dollar_loss": round(worst_dollar, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "median_loss_pct": round(median_loss, 2),
        "svar_95_pct": round(svar_95, 2),
        "ses_95_pct": round(ses_95, 2),
        "scenarios_tested": len(results),
        "overall_resilience": overall_grade,
        "primary_vulnerability": primary_vulnerability,
        "total_capital_saved": round(total_saved, 2),
    }


def print_stress_test_report(
    results: dict[str, dict[str, Any]],
    total_wallet: float,
    weights: dict[str, float],
    cash_buffer: float = 0.0,
    show_asset_breakdown: bool = False,
):
    """Print terminal report of historical stress testing."""
    print("=================================================================================================================")
    print("                               HISTORICAL BLACK SWAN CRYPTO STRESS TEST REPORT                                   ")
    print("=================================================================================================================")
    print(f"Total Portfolio Capital: ${total_wallet:,.2f}")
    w_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in weights.items()])
    print(f"Asset Allocation Weights: {w_str}")
    if cash_buffer > 0.0:
        print(f"Hedging Cash / Stablecoin Buffer: {cash_buffer*100:.1f}% (Risky Allocation: {(1.0-cash_buffer)*100:.1f}%)")
    print()
    print(f"{'Historical Scenario':<36} | {'Impact':>8} | {'Loss ($)':>10} | {'Remaining':>10} | {'Recovery Req':>12} | {'Top Driver':<12} | {'Grade'}")
    print("-----------------------------------------------------------------------------------------------------------------")

    for name, m in results.items():
        pct = f"{m['portfolio_loss_pct']:+.2f}%"
        loss = f"-${m['dollar_loss']:,.2f}"
        rem = f"${m['remaining_balance']:,.2f}"
        rec = f"+{m['recovery_required_pct']:.1f}%"
        driver = str(m.get("primary_risk_driver", "-"))
        print(f"{name:<36} | {pct:>8} | {loss:>10} | {rem:>10} | {rec:>12} | {driver:<12} | {m['resilience_grade']}")

        if show_asset_breakdown and "asset_breakdown" in m:
            for asset, ab in m["asset_breakdown"].items():
                print(f"   * {asset:<12}: drop {ab['asset_drop_pct']:+.1f}% | contrib {ab['contribution_pct']:+.2f}% (-${ab['dollar_impact']:,.2f}) | share: {ab['loss_share_pct']:.1f}%")

    summary = summarize_stress_test_results(results)
    print("-----------------------------------------------------------------------------------------------------------------")
    print(f"Worst Scenario: {summary['worst_scenario']} ({summary['worst_loss_pct']:+.2f}% / -${summary['worst_dollar_loss']:,.2f})")
    print(f"Average Stress Loss: {summary['avg_loss_pct']:+.2f}% | Median Loss: {summary['median_loss_pct']:+.2f}%")
    print(f"Stress VaR (95%): {summary['svar_95_pct']:+.2f}% | Stress ES (95%): {summary['ses_95_pct']:+.2f}%")
    if summary["primary_vulnerability"]:
        print(f"Primary Vulnerability Driver: {summary['primary_vulnerability']}")
    if cash_buffer > 0.0 and summary["total_capital_saved"] > 0:
        print(f"Total Capital Preserved by Buffer: ${summary['total_capital_saved']:,.2f}")
    print(f"Overall Portfolio Resilience: {summary['overall_resilience']}")
    print("=================================================================================================================\n")


def to_dict_stress_test_result(
    results: dict[str, dict[str, Any]],
    total_wallet: float | None = None,
    weights: dict[str, float] | None = None,
    cash_buffer: float = 0.0,
) -> dict[str, Any]:
    """Convert stress test results to a serializable dictionary payload."""
    payload: dict[str, Any] = {
        "scenarios": results,
        "summary": summarize_stress_test_results(results),
    }
    if total_wallet is not None:
        payload["total_wallet"] = total_wallet
    if weights is not None:
        payload["weights"] = weights
    if cash_buffer > 0.0:
        payload["cash_buffer"] = cash_buffer
    return payload


def export_stress_test_json(
    results: dict[str, dict[str, Any]],
    filepath: str | Path,
    total_wallet: float | None = None,
    weights: dict[str, float] | None = None,
    cash_buffer: float = 0.0,
) -> None:
    """Export stress test results dictionary to a JSON file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_dict_stress_test_result(results, total_wallet=total_wallet, weights=weights, cash_buffer=cash_buffer)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_stress_test_csv(
    results: dict[str, dict[str, Any]],
    filepath: str | Path,
    include_asset_breakdown: bool = False,
) -> None:
    """Export stress test scenarios and metrics to a CSV file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if include_asset_breakdown:
        for scenario_name, m in results.items():
            breakdown = m.get("asset_breakdown", {})
            for asset, ab in breakdown.items():
                rows.append({
                    "scenario": scenario_name,
                    "asset": asset,
                    "effective_weight": ab.get("weight", 0.0),
                    "asset_drop_pct": ab.get("asset_drop_pct", 0.0),
                    "contribution_pct": ab.get("contribution_pct", 0.0),
                    "dollar_impact": ab.get("dollar_impact", 0.0),
                    "loss_share_pct": ab.get("loss_share_pct", 0.0),
                    "portfolio_loss_pct": m.get("portfolio_loss_pct", 0.0),
                    "dollar_loss": m.get("dollar_loss", 0.0),
                    "remaining_balance": m.get("remaining_balance", 0.0),
                    "resilience_grade": m.get("resilience_grade", ""),
                })
    else:
        for scenario_name, m in results.items():
            rows.append({
                "scenario": scenario_name,
                "portfolio_loss_pct": m.get("portfolio_loss_pct", 0.0),
                "dollar_loss": m.get("dollar_loss", 0.0),
                "remaining_balance": m.get("remaining_balance", 0.0),
                "recovery_required_pct": m.get("recovery_required_pct", 0.0),
                "resilience_grade": m.get("resilience_grade", ""),
                "primary_risk_driver": m.get("primary_risk_driver", ""),
                "capital_saved_dollars": m.get("capital_saved_dollars", 0.0),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Crypto Historical Stress Testing Engine")
    parser.add_argument("--wallet-size", "--wallet", dest="wallet_size", type=float, default=10000.0, help="Total wallet value in USDT (default: 10000.0)")
    parser.add_argument("--weights", nargs="+", default=None, help="Asset weights list (e.g. BTC/USDT:0.5 ETH/USDT:0.5)")
    parser.add_argument("--cash-buffer", "--hedge-buffer", dest="cash_buffer", type=float, default=0.0, help="Stablecoin/Cash buffer allocation fraction [0.0, 1.0) to simulate hedging")
    parser.add_argument("--custom-shock", nargs="+", default=None, help="Custom shock rates (e.g. BTC:-0.25 ETH:-0.35)")
    parser.add_argument("--config-file", type=str, default="", help="Path to allocation or config JSON to read weights from")
    parser.add_argument("--show-assets", action="store_true", help="Display asset-level loss contribution breakdown")
    parser.add_argument("--export-json", type=str, default="", help="Path to export results JSON file")
    parser.add_argument("--export-csv", type=str, default="", help="Path to export results CSV file")
    parser.add_argument("--export-asset-csv", type=str, default="", help="Path to export granular per-asset contribution CSV")
    args = parser.parse_args()

    weights = {"BTC/USDT": 0.50, "ETH/USDT": 0.30, "SOL/USDT": 0.20}
    if args.weights:
        weights = {}
        for item in args.weights:
            if ":" in item:
                pair, w_str = item.split(":", 1)
                try:
                    weights[pair.strip()] = float(w_str)
                except ValueError:
                    pass
    elif args.config_file and Path(args.config_file).is_file():
        try:
            data = json.loads(Path(args.config_file).read_text(encoding="utf-8"))
            weights = data.get("pair_weights", data.get("weights", weights))
        except Exception as e:
            print(f"[!] Warning: Could not read {args.config_file}: {e}")

    custom_shock = None
    if args.custom_shock:
        custom_shock = {}
        for item in args.custom_shock:
            if ":" in item:
                coin, drop_str = item.split(":", 1)
                try:
                    custom_shock[coin.strip().upper()] = float(drop_str)
                except ValueError:
                    pass

    results = evaluate_stress_test(
        weights,
        total_wallet=args.wallet_size,
        custom_shock=custom_shock,
        cash_buffer=args.cash_buffer,
    )
    print_stress_test_report(
        results,
        total_wallet=args.wallet_size,
        weights=weights,
        cash_buffer=args.cash_buffer,
        show_asset_breakdown=args.show_assets,
    )

    if args.export_json:
        export_stress_test_json(
            results,
            args.export_json,
            total_wallet=args.wallet_size,
            weights=weights,
            cash_buffer=args.cash_buffer,
        )
        print(f"[+] Stress test results exported to: {args.export_json}")

    if args.export_csv:
        export_stress_test_csv(results, args.export_csv)
        print(f"[+] Stress test results exported to: {args.export_csv}")

    if args.export_asset_csv:
        export_stress_test_csv(results, args.export_asset_csv, include_asset_breakdown=True)
        print(f"[+] Asset-level stress test results exported to: {args.export_asset_csv}")


if __name__ == "__main__":
    main()
