"""Kimchi Premium (KRW vs USDT) Real-Time & Historical Analyzer.

Calculates the price discrepancy between Upbit (KRW) and Binance (USDT):
Kimchi Premium (%) = ((Upbit_KRW / (Binance_USDT * FX_Rate)) - 1) * 100
Provides currency-adjusted arbitrage spread indicators and portfolio risk alerts.
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

import urllib.request
import pandas as pd


def fetch_live_usd_krw_rate(timeout: float = 3.0) -> tuple[float, str]:
    """
    Attempt to fetch real-time USD/KRW exchange rate from public open exchange API.

    Returns tuple (rate, source_description).
    Falls back safely to (1350.0, 'Default Fallback') on network error.
    """
    url = "https://open.er-api.com/v6/latest/USD"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rate = float(data["rates"].get("KRW", 1350.0))
            return rate, "Live Open Exchange API"
    except Exception:
        return 1350.0, "Default Fallback (Use --usdt-krw to override)"


def compute_kimchi_premium(
    upbit_prices: dict[str, float],
    binance_prices: dict[str, float],
    usdt_krw_rate: float = 1350.0,
) -> dict[str, dict[str, float | str]]:
    """
    Compute Kimchi Premium for intersecting assets between Upbit and Binance.

    Parameters:
    - upbit_prices: Mapping of coin symbol to KRW price, e.g. {"BTC": 135000000.0}
    - binance_prices: Mapping of coin symbol to USDT price, e.g. {"BTC": 97000.0}
    - usdt_krw_rate: USD/KRW exchange rate (default: 1350.0)
    """
    if usdt_krw_rate <= 0:
        raise ValueError("Exchange rate must be strictly positive.")

    results: dict[str, dict[str, float | str]] = {}

    common_symbols = sorted(set(upbit_prices.keys()) & set(binance_prices.keys()))

    for sym in common_symbols:
        p_upbit = upbit_prices[sym]
        p_binance = binance_prices[sym]

        if p_binance <= 0:
            continue

        fair_krw = p_binance * usdt_krw_rate
        premium_pct = ((p_upbit / fair_krw) - 1.0) * 100
        krw_diff = p_upbit - fair_krw

        # Qualitative risk status
        if premium_pct >= 6.0:
            status = "Extreme Overheated (High Dumping Risk)"
        elif premium_pct >= 3.0:
            status = "Moderate Premium (Domestic Buying Pressure)"
        elif premium_pct <= -1.0:
            status = "Negative Premium (Discount / Arbitrage Opportunity)"
        else:
            status = "Neutral / Fair Equilibrium"

        results[sym] = {
            "upbit_krw": round(p_upbit, 2),
            "binance_usdt": round(p_binance, 2),
            "fair_krw": round(fair_krw, 2),
            "krw_difference": round(krw_diff, 2),
            "premium_pct": round(premium_pct, 2),
            "status": status,
        }

    return results


def print_kimchi_premium_report(
    results: dict[str, dict[str, float | str]],
    usdt_krw_rate: float,
):
    """Print terminal report of Kimchi Premium spreads."""
    print("================================================================================")
    print(f"       KIMCHI PREMIUM SPREAD ANALYSIS (FX Rate: {usdt_krw_rate:,.2f} KRW/USDT)   ")
    print("================================================================================")
    print(f"{'Coin':<6} | {'Upbit (KRW)':>15} | {'Binance ($)':>13} | {'Fair KRW':>15} | {'Premium':>9} | {'Status'}")
    print("--------------------------------------------------------------------------------")

    for sym, m in results.items():
        up_str = f"{m['upbit_krw']:,.0f}"
        bi_str = f"${m['binance_usdt']:,.2f}"
        fair_str = f"{m['fair_krw']:,.0f}"
        prem_str = f"{m['premium_pct']:+.2f}%"
        print(f"{sym:<6} | {up_str:>15} | {bi_str:>13} | {fair_str:>15} | {prem_str:>9} | {m['status']}")

    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Kimchi Premium Arbitrage Analyzer")
    parser.add_argument("--usdt-krw", type=float, default=None, help="Explicit USD/KRW exchange rate (default: fetch live)")
    parser.add_argument("--export-json", type=str, default="", help="Path to export results JSON")
    args = parser.parse_args()

    if args.usdt_krw is not None:
        rate = args.usdt_krw
        rate_source = "User Specified (--usdt-krw)"
    else:
        rate, rate_source = fetch_live_usd_krw_rate()
    print(f"[*] Applied Exchange Rate: {rate:,.2f} KRW/USD (Source: {rate_source})")

    # Default representative crypto prices if live API is unavailable
    sample_upbit = {"BTC": 136500000.0, "ETH": 4750000.0, "SOL": 298000.0, "XRP": 1150.0}
    sample_binance = {"BTC": 98000.0, "ETH": 3450.0, "SOL": 215.0, "XRP": 0.83}

    try:
        import ccxt
        binance = ccxt.binance({"enableRateLimit": True})
        upbit = ccxt.upbit({"enableRateLimit": True})

        pairs = ["BTC", "ETH", "SOL", "XRP"]
        for p in pairs:
            try:
                b_ticker = binance.fetch_ticker(f"{p}/USDT")
                u_ticker = upbit.fetch_ticker(f"{p}/KRW")
                if b_ticker.get("last") and u_ticker.get("last"):
                    sample_binance[p] = float(b_ticker["last"])
                    sample_upbit[p] = float(u_ticker["last"])
            except Exception:
                pass
    except Exception:
        pass

    results = compute_kimchi_premium(
        upbit_prices=sample_upbit,
        binance_prices=sample_binance,
        usdt_krw_rate=rate,
    )

    print_kimchi_premium_report(results, usdt_krw_rate=rate)

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[+] Kimchi premium results exported to: {out_path}")


if __name__ == "__main__":
    main()
