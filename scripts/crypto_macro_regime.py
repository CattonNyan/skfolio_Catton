"""Crypto Macro Market Regime & Fear/Greed Dynamic Cash Allocator.

Fetches the Crypto Fear & Greed Index and dynamically adjusts portfolio cash (USDT)
vs risk-asset allocation to protect capital during market overheating:
- Extreme Greed (>= 75): 35%~50% Cash buffer (take-profit & risk-off)
- Greed (60 ~ 74): 20% Cash buffer
- Neutral (40 ~ 59): Normal model weight execution (5% cash buffer)
- Fear (25 ~ 39): 5% Cash buffer
- Extreme Fear (< 25): 0% Cash (Maximum deployment / oversold dip buying)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd


def fetch_fear_and_greed_index(limit: int = 1) -> tuple[int, str]:
    """
    Fetch the latest Crypto Fear & Greed Index from Alternative.me public API.

    Returns tuple of (index_value, sentiment_classification).
    Gracefully falls back to (50, 'Neutral') on network error.
    """
    url = f"https://api.alternative.me/fng/?limit={limit}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            latest = data["data"][0]
            val = int(latest["value"])
            label = latest["value_classification"]
            return val, label
    except Exception as e:
        # Fallback to neutral on network error or offline environment
        return 50, "Neutral"


def adjust_cash_allocation_by_regime(
    base_weights: dict[str, float],
    fng_value: int,
    total_wallet: float = 10000.0,
) -> dict[str, object]:
    """
    Adjust portfolio weights dynamically based on market sentiment.

    Allocates dynamic USDT cash buffer and rescales crypto weights.
    """
    if fng_value < 0 or fng_value > 100:
        raise ValueError("Fear and Greed index must be between 0 and 100.")

    # Determine required cash reserve
    if fng_value >= 80:
        regime = "Extreme Greed"
        cash_ratio = 0.40  # 40% cash
    elif fng_value >= 65:
        regime = "Greed"
        cash_ratio = 0.25  # 25% cash
    elif fng_value >= 40:
        regime = "Neutral"
        cash_ratio = 0.05  # 5% cash
    elif fng_value >= 25:
        regime = "Fear"
        cash_ratio = 0.05  # 5% cash
    else:
        regime = "Extreme Fear"
        cash_ratio = 0.00  # 0% cash (Full deploy)

    crypto_ratio = 1.0 - cash_ratio

    # Rescale crypto weights
    total_base_crypto = sum(base_weights.values())
    if total_base_crypto <= 0:
        adjusted_weights = {"USDT": 1.0}
    else:
        adjusted_weights = {
            coin: round((w / total_base_crypto) * crypto_ratio, 4)
            for coin, w in base_weights.items()
        }
        if cash_ratio > 0:
            adjusted_weights["USDT (Cash)"] = round(cash_ratio, 4)

    # Allocate wallet capital
    capital_allocation = {
        coin: round(w * total_wallet, 2)
        for coin, w in adjusted_weights.items()
    }

    # Separate pure tradable pairs for Freqtrade
    freqtrade_pairs = {
        k: v for k, v in adjusted_weights.items()
        if not ("(Cash)" in k or k.strip().upper() == "USDT")
    }

    return {
        "fng_value": fng_value,
        "market_regime": regime,
        "cash_ratio": cash_ratio,
        "crypto_ratio": crypto_ratio,
        "adjusted_weights": adjusted_weights,
        "freqtrade_pair_weights": freqtrade_pairs,
        "capital_allocation": capital_allocation,
        "total_wallet": total_wallet,
    }


def filter_crypto_weights_for_freqtrade(
    adjusted_weights: dict[str, float],
) -> tuple[dict[str, float], float]:
    """
    Separates tradable crypto pairs from virtual cash reserve ('USDT (Cash)').

    Returns (clean_pair_weights, cash_ratio).
    """
    clean_pairs = {}
    cash_ratio = 0.0
    for key, weight in adjusted_weights.items():
        if "(Cash)" in key or key.strip().upper() == "USDT":
            cash_ratio += weight
        else:
            clean_pairs[key] = weight
    return clean_pairs, cash_ratio


def print_macro_regime_report(res: dict[str, object]):
    """Print clean terminal report of regime cash allocation."""
    print("================================================================================")
    print(f"       CRYPTO MACRO REGIME & CASH ALLOCATION (F&G Index: {res['fng_value']}/100)        ")
    print("================================================================================")
    print(f"Market Sentiment Status  : {res['market_regime']}")
    print(f"Dynamic Cash (USDT) Ratio: {res['cash_ratio']*100:.1f}%")
    print(f"Crypto Investment Ratio  : {res['crypto_ratio']*100:.1f}%")
    print(f"Total Portfolio Capital  : ${res['total_wallet']:,.2f}")
    print("--------------------------------------------------------------------------------")
    print(f"{'Asset / Currency':<22} | {'Adjusted Weight':>16} | {'Allocated Amount':>18}")
    print("--------------------------------------------------------------------------------")

    weights = res["adjusted_weights"]
    capitals = res["capital_allocation"]

    for asset in weights:
        w_str = f"{weights[asset]*100:.2f}%"
        c_str = f"${capitals[asset]:,.2f}"
        print(f"{asset:<22} | {w_str:>16} | {c_str:>18}")

    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Crypto Macro Regime & Dynamic Cash Allocator")
    parser.add_argument("--fng", type=int, default=None, help="Explicit Fear & Greed index (0-100, default: fetch live)")
    parser.add_argument("--wallet-size", type=float, default=10000.0, help="Total wallet value in USDT")
    parser.add_argument("--export-json", type=str, default="", help="Path to export results JSON")
    args = parser.parse_args()

    if args.fng is not None:
        fng_val = args.fng
    else:
        fng_val, label = fetch_fear_and_greed_index()
        print(f"[*] Fetched Live Alternative.me Index: {fng_val}/100 ({label})")

    base_weights = {"BTC/USDT": 0.50, "ETH/USDT": 0.30, "SOL/USDT": 0.20}
    res = adjust_cash_allocation_by_regime(
        base_weights=base_weights,
        fng_value=fng_val,
        total_wallet=args.wallet_size,
    )

    print_macro_regime_report(res)

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[+] Macro regime cash allocation exported to: {out_path}")


if __name__ == "__main__":
    main()
