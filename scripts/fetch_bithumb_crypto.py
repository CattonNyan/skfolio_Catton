"""Bithumb Korean Exchange Real-Time and Candlestick Data Fetcher.

Fetches OHLCV candlestick data and live tickers directly from Bithumb's public REST API:
- Candlestick endpoint: https://api.bithumb.com/public/candlestick/{order_currency}_{payment_currency}/{chart_intervals}
- Ticker endpoint: https://api.bithumb.com/public/ticker/{order_currency}_{payment_currency}
Formatted for direct consumption by skfolio portfolio optimizers.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

BITHUMB_API_BASE = "https://api.bithumb.com/public"

VALID_BITHUMB_INTERVALS = {"1m", "3m", "5m", "10m", "30m", "1h", "6h", "12h", "24h"}


def normalize_bithumb_symbol(symbol: str, payment_currency: str = "KRW") -> tuple[str, str]:
    """Normalize user symbol into (order_currency, payment_currency).

    Supports formats like 'BTC', 'BTC/KRW', 'KRW-BTC', 'BTC_KRW'.
    """
    if not isinstance(symbol, str):
        raise ValueError("Symbol must be a string.")
    cleaned = symbol.strip().upper().replace("/", "_").replace("-", "_")
    if not cleaned:
        raise ValueError("Symbol cannot be empty.")

    pay = payment_currency.strip().upper()
    if "_" in cleaned:
        parts = cleaned.split("_")
        if len(parts) == 2:
            if parts[0] == pay:
                return parts[1], pay
            elif parts[1] == pay:
                return parts[0], pay
            return parts[0], parts[1]

    return cleaned, pay


def fetch_bithumb_candlestick(
    symbol: str,
    chart_interval: str = "24h",
    payment_currency: str = "KRW",
    count: int = 100,
    timeout: float = 5.0,
) -> pd.DataFrame:
    """Fetch candlestick history from Bithumb public REST API.

    Returns DataFrame with columns: ['date', 'open', 'close', 'high', 'low', 'volume'].
    """
    if chart_interval not in VALID_BITHUMB_INTERVALS:
        raise ValueError(f"Invalid chart_interval '{chart_interval}'. Must be one of {VALID_BITHUMB_INTERVALS}")

    order_curr, pay_curr = normalize_bithumb_symbol(symbol, payment_currency)
    url = f"{BITHUMB_API_BASE}/candlestick/{order_curr}_{pay_curr}/{chart_interval}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) skfolio-catton/1.4.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if data.get("status") != "0000" or "data" not in data:
        raise RuntimeError(f"Bithumb API error for {order_curr}_{pay_curr}: {data.get('message', 'Unknown error')}")

    raw_bars = data["data"]
    if not raw_bars:
        return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])

    # Bithumb returns: [time (ms), open, close, high, low, volume]
    records = []
    for bar in raw_bars[-count:]:
        records.append({
            "date": pd.to_datetime(bar[0], unit="ms", utc=True),
            "open": float(bar[1]),
            "close": float(bar[2]),
            "high": float(bar[3]),
            "low": float(bar[4]),
            "volume": float(bar[5]),
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


def fetch_bithumb_multi_assets(
    symbols: list[str],
    chart_interval: str = "24h",
    payment_currency: str = "KRW",
    count: int = 100,
    timeout: float = 5.0,
) -> pd.DataFrame:
    """Fetch close prices for multiple symbols on Bithumb and merge into a single DataFrame."""
    close_series = {}
    for sym in symbols:
        try:
            df = fetch_bithumb_candlestick(sym, chart_interval, payment_currency, count, timeout)
            if not df.empty:
                order_curr, _ = normalize_bithumb_symbol(sym, payment_currency)
                close_series[f"{order_curr}/KRW"] = df.set_index("date")["close"]
        except Exception as e:
            print(f"[!] Warning: Failed to fetch {sym} from Bithumb: {e}")

    if not close_series:
        return pd.DataFrame()

    merged = pd.DataFrame(close_series).sort_index().ffill().dropna()
    return merged


def main():
    parser = argparse.ArgumentParser(description="Fetch candlestick data from Bithumb public API.")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL", "XRP"], help="List of crypto symbols.")
    parser.add_argument("--interval", default="24h", choices=list(VALID_BITHUMB_INTERVALS), help="Candle interval.")
    parser.add_argument("--count", type=int, default=30, help="Number of candles to fetch.")
    args = parser.parse_args()

    print(f"[*] Fetching Bithumb data for {args.symbols} ({args.interval})...")
    df = fetch_bithumb_multi_assets(args.symbols, chart_interval=args.interval, count=args.count)
    if df.empty:
        print("[!] No data returned.")
    else:
        print(f"[+] Successfully fetched {len(df)} rows across {len(df.columns)} assets:")
        print(df.tail())


if __name__ == "__main__":
    main()
