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
import math
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
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Limit must be a strictly positive integer.")
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
    except Exception:
        # Fallback to neutral on network error or offline environment
        return 50, "Neutral"


def fetch_fear_and_greed_history(limit: int = 30, timeout: float = 5.0) -> pd.DataFrame:
    """
    Fetch historical series of Crypto Fear & Greed Index from Alternative.me public API.

    Returns DataFrame indexed by date with columns: ['date', 'value', 'classification'].
    Falls back to a synthetic neutral/fear series on network error.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Limit must be a strictly positive integer.")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be a finite, strictly positive number.")

    url = f"https://api.alternative.me/fng/?limit={limit}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_data = json.loads(resp.read().decode("utf-8"))
            items = raw_data.get("data", [])
            if items:
                records = []
                for entry in items:
                    ts = pd.to_datetime(int(entry["timestamp"]), unit="s")
                    records.append({
                        "date": ts,
                        "value": int(entry["value"]),
                        "classification": str(entry["value_classification"]),
                    })
                df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
                return df
    except Exception:
        pass

    # Fallback synthetic historical series
    dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq="D")
    return pd.DataFrame({
        "date": dates,
        "value": [50] * limit,
        "classification": ["Neutral"] * limit,
    })


def adjust_cash_allocation_by_regime(
    base_weights: dict[str, float],
    fng_value: int,
    total_wallet: float = 10000.0,
) -> dict[str, object]:
    """
    Adjust portfolio weights dynamically based on market sentiment.

    Allocates dynamic USDT cash buffer and rescales crypto weights.
    """
    if isinstance(fng_value, bool) or not isinstance(fng_value, int) or fng_value < 0 or fng_value > 100:
        raise ValueError("Fear and Greed index must be an integer between 0 and 100.")
    if (
        isinstance(total_wallet, bool)
        or not isinstance(total_wallet, (int, float))
        or not math.isfinite(total_wallet)
        or total_wallet <= 0
    ):
        raise ValueError("Total wallet must be finite and strictly positive.")
    if not isinstance(base_weights, dict):
        raise ValueError("Base weights must be a dictionary.")
    for k, v in base_weights.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError("Asset names must be non-empty strings.")
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v < 0
        ):
            raise ValueError(f"Weight for {k} must be a finite, non-negative number.")

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


def calculate_macro_regime_weights(
    base_weights: dict[str, float],
    fear_and_greed_value: int = 50,
) -> dict[str, float]:
    """
    Convenience helper for UI/Dashboard to return adjusted weights dictionary directly.
    """
    res = adjust_cash_allocation_by_regime(
        base_weights=base_weights,
        fng_value=fear_and_greed_value,
    )
    return res["adjusted_weights"]


def calculate_sentiment_trend_regime(
    fng_value: int,
    btc_price: float,
    btc_sma200: float,
) -> dict[str, Any]:
    """Compute composite crypto macro regime by fusing Fear & Greed sentiment with 200D SMA trend.

    Combines contrarian sentiment signals with trend confirmation to prevent premature
    dip buying in structural bear markets or premature hedging in strong bull trends.

    Regime Matrix:
    1. Bull Overheating (F&G >= 75 & Price >= SMA): Bull mania phase -> 35% cash buffer (take-profit).
    2. Bull Healthy Trend (45 <= F&G < 75 & Price >= SMA): Strong trend continuation -> 10% cash buffer.
    3. Bull Dip (F&G < 45 & Price >= SMA): Pullback in bull market -> 0% cash buffer (aggressive dip buy).
    4. Bear Rally (F&G >= 60 & Price < SMA): Dead-cat bounce in downtrend -> 50% cash buffer (strong hedge).
    5. Bear Downtrend (25 <= F&G < 60 & Price < SMA): Structural bear downtrend -> 40% cash buffer (capital preservation).
    6. Bear Capitulation (F&G < 25 & Price < SMA): Deep despair bottoming -> 15% cash buffer (phased accumulation).

    Parameters
    ----------
    fng_value : int
        Fear & Greed Index (0 to 100).
    btc_price : float
        Current Bitcoin or benchmark price (> 0).
    btc_sma200 : float
        200-day Simple Moving Average price (> 0).

    Returns
    -------
    dict[str, Any]
        Dictionary with trend metrics, composite regime, recommended cash/crypto ratios, and action directive.
    """
    if isinstance(fng_value, bool) or not isinstance(fng_value, int) or fng_value < 0 or fng_value > 100:
        raise ValueError("Fear and Greed index must be an integer between 0 and 100.")
    if (
        isinstance(btc_price, bool)
        or not isinstance(btc_price, (int, float))
        or not math.isfinite(btc_price)
        or btc_price <= 0
    ):
        raise ValueError("BTC price must be a finite, strictly positive number.")
    if (
        isinstance(btc_sma200, bool)
        or not isinstance(btc_sma200, (int, float))
        or not math.isfinite(btc_sma200)
        or btc_sma200 <= 0
    ):
        raise ValueError("BTC 200D SMA must be a finite, strictly positive number.")

    trend_distance_pct = ((btc_price - btc_sma200) / btc_sma200) * 100.0
    is_bullish_trend = btc_price >= btc_sma200
    trend_status = "Bullish (Above 200D SMA)" if is_bullish_trend else "Bearish (Below 200D SMA)"

    if fng_value >= 80:
        sentiment_label = "Extreme Greed"
    elif fng_value >= 65:
        sentiment_label = "Greed"
    elif fng_value >= 40:
        sentiment_label = "Neutral"
    elif fng_value >= 25:
        sentiment_label = "Fear"
    else:
        sentiment_label = "Extreme Fear"

    if is_bullish_trend:
        if fng_value >= 75:
            composite_regime = "Bull Market Overheating"
            cash_ratio = 0.35
            action = "Take Profit & Raise Cash Buffer"
        elif fng_value >= 45:
            composite_regime = "Bull Market Healthy Trend"
            cash_ratio = 0.10
            action = "Trend Following (Normal Allocation)"
        else:
            composite_regime = "Bull Market Dip"
            cash_ratio = 0.00
            action = "Aggressive Dip Buying (Full Deployment)"
    else:
        if fng_value >= 60:
            composite_regime = "Bear Market Rally (Dead-Cat Bounce)"
            cash_ratio = 0.50
            action = "Defensive De-risking & Maximum Cash Buffer"
        elif fng_value >= 25:
            composite_regime = "Bear Market Downtrend"
            cash_ratio = 0.40
            action = "Capital Preservation & High Cash Hedge"
        else:
            composite_regime = "Bear Market Capitulation"
            cash_ratio = 0.15
            action = "Phased Bottom Accumulation"

    crypto_ratio = round(1.0 - cash_ratio, 4)

    return {
        "fng_value": fng_value,
        "sentiment_label": sentiment_label,
        "btc_price": round(float(btc_price), 2),
        "btc_sma200": round(float(btc_sma200), 2),
        "trend_distance_pct": round(float(trend_distance_pct), 2),
        "is_bullish_trend": is_bullish_trend,
        "trend_status": trend_status,
        "composite_regime": composite_regime,
        "market_regime": composite_regime,
        "recommended_cash_ratio": cash_ratio,
        "recommended_crypto_ratio": crypto_ratio,
        "cash_ratio": cash_ratio,
        "crypto_ratio": crypto_ratio,
        "hedging_action": action,
    }


def adjust_weights_by_sentiment_trend(
    base_weights: dict[str, float],
    fng_value: int,
    btc_price: float,
    btc_sma200: float,
    total_wallet: float = 10000.0,
) -> dict[str, Any]:
    """Adjust portfolio weights dynamically based on composite sentiment and 200D SMA trend regime."""
    if not isinstance(base_weights, dict):
        raise ValueError("Base weights must be a dictionary.")
    if (
        isinstance(total_wallet, bool)
        or not isinstance(total_wallet, (int, float))
        or not math.isfinite(total_wallet)
        or total_wallet <= 0
    ):
        raise ValueError("Total wallet must be finite and strictly positive.")

    regime_info = calculate_sentiment_trend_regime(
        fng_value=fng_value,
        btc_price=btc_price,
        btc_sma200=btc_sma200,
    )
    cash_ratio = regime_info["cash_ratio"]
    crypto_ratio = regime_info["crypto_ratio"]

    total_base = sum(base_weights.values())
    if total_base <= 0:
        adj_weights = {"USDT": 1.0}
    else:
        adj_weights = {
            coin: round((w / total_base) * crypto_ratio, 4)
            for coin, w in base_weights.items()
        }
        if cash_ratio > 0:
            adj_weights["USDT (Cash)"] = round(cash_ratio, 4)

    capitals = {
        coin: round(w * total_wallet, 2)
        for coin, w in adj_weights.items()
    }
    freqtrade_pairs = {
        k: v for k, v in adj_weights.items()
        if not ("(Cash)" in k or k.strip().upper() == "USDT")
    }

    return {
        **regime_info,
        "adjusted_weights": adj_weights,
        "freqtrade_pair_weights": freqtrade_pairs,
        "capital_allocation": capitals,
        "total_wallet": total_wallet,
    }


def print_macro_regime_report(res: dict[str, object]):
    """Print clean terminal report of regime cash allocation."""
    print("================================================================================")
    print(f"       CRYPTO MACRO REGIME & CASH ALLOCATION (F&G Index: {res['fng_value']}/100)        ")
    print("================================================================================")
    print(f"Market Sentiment Status  : {res.get('market_regime', '')}")
    if "trend_status" in res:
        print(f"Trend Status (200D SMA)  : {res.get('trend_status')} (Distance: {res.get('trend_distance_pct'):+.2f}%)")
    if "hedging_action" in res:
        print(f"Tactical Hedge Action    : {res.get('hedging_action')}")
    print(f"Dynamic Cash (USDT) Ratio: {float(res['cash_ratio'])*100:.1f}%")
    print(f"Crypto Investment Ratio  : {float(res['crypto_ratio'])*100:.1f}%")
    print(f"Total Portfolio Capital  : ${float(res['total_wallet']):,.2f}")
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


def export_macro_regime_csv(
    res: dict[str, object],
    filepath: str | Path,
) -> Path:
    """Export macro regime and tactical cash allocation results to a CSV file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    weights = res.get("adjusted_weights", {})
    capitals = res.get("capital_allocation", {})
    for asset in weights:
        row = {
            "section": "allocation",
            "asset": asset,
            "weight": round(weights[asset], 4),
            "allocated_usd": round(capitals.get(asset, 0.0), 2),
            "market_regime": res.get("market_regime", ""),
            "fng_value": res.get("fng_value", 0),
        }
        if "trend_status" in res:
            row["trend_status"] = res.get("trend_status", "")
            row["trend_distance_pct"] = res.get("trend_distance_pct", 0.0)
        rows.append(row)
    for key in ["fng_value", "market_regime", "cash_ratio", "crypto_ratio", "total_wallet", "trend_status", "trend_distance_pct", "hedging_action"]:
        if key in res:
            rows.append({
                "section": "summary",
                "asset": key,
                "weight": res[key] if isinstance(res[key], (int, float)) else 0.0,
                "allocated_usd": res[key] if isinstance(res[key], (int, float)) else 0.0,
                "market_regime": str(res[key]) if not isinstance(res[key], (int, float)) else res.get("market_regime", ""),
                "fng_value": res.get("fng_value", 0),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Crypto Macro Regime & Dynamic Cash Allocator")
    parser.add_argument("--fng", type=int, default=None, help="Explicit Fear & Greed index (0-100, default: fetch live)")
    parser.add_argument("--wallet-size", type=float, default=10000.0, help="Total wallet value in USDT")
    parser.add_argument("--config-file", type=str, default="", help="Path to allocation or config JSON to read weights from")
    parser.add_argument("--btc-price", type=float, default=None, help="Current Bitcoin benchmark price (USD)")
    parser.add_argument("--btc-sma200", type=float, default=None, help="Bitcoin 200-day Simple Moving Average price (USD)")
    parser.add_argument("--use-trend", action="store_true", help="Activate composite sentiment & 200D SMA trend regime allocator")
    parser.add_argument("--export-json", type=str, default="", help="Path to export results JSON")
    parser.add_argument("--export-csv", type=str, default="", help="Path to export results CSV")
    args = parser.parse_args()

    if args.fng is not None:
        fng_val = args.fng
    else:
        fng_val, label = fetch_fear_and_greed_index()
        print(f"[*] Fetched Live Alternative.me Index: {fng_val}/100 ({label})")

    base_weights = {"BTC/USDT": 0.50, "ETH/USDT": 0.30, "SOL/USDT": 0.20}
    if args.config_file and Path(args.config_file).is_file():
        try:
            data = json.loads(Path(args.config_file).read_text(encoding="utf-8"))
            base_weights = data.get("pair_weights", data.get("weights", base_weights))
        except Exception as e:
            print(f"[!] Warning: Could not load {args.config_file}: {e}")

    if args.use_trend or (args.btc_price is not None and args.btc_sma200 is not None):
        btc_p = args.btc_price if args.btc_price is not None else 65000.0
        btc_sma = args.btc_sma200 if args.btc_sma200 is not None else 58000.0
        res = adjust_weights_by_sentiment_trend(
            base_weights=base_weights,
            fng_value=fng_val,
            btc_price=btc_p,
            btc_sma200=btc_sma,
            total_wallet=args.wallet_size,
        )
    else:
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

    if args.export_csv:
        out_csv = export_macro_regime_csv(res, args.export_csv)
        print(f"[+] Macro regime cash allocation exported to: {out_csv}")


if __name__ == "__main__":
    main()
