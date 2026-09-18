"""Coinbase Exchange Real-Time and Candlestick Data Fetcher.

Fetches OHLCV candlestick data and spot prices directly from Coinbase public REST API:
- Exchange candles: https://api.exchange.coinbase.com/products/{product_id}/candles
- Spot prices: https://api.coinbase.com/v2/prices/{currency_pair}/spot
Used as the global institutional USD pricing benchmark for crypto portfolio optimization.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.http_retry_helper import fetch_json_with_retry

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

COINBASE_API_BASE = "https://api.exchange.coinbase.com"
COINBASE_V2_BASE = "https://api.coinbase.com/v2"

GRANULARITY_MAP = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
}


def normalize_coinbase_symbol(symbol: str, quote: str = "USD") -> str:
    """Normalize user symbol to Coinbase product ID format (e.g. 'BTC' -> 'BTC-USD')."""
    if not isinstance(symbol, str):
        raise ValueError("Symbol must be a string.")
    cleaned = symbol.strip().upper().replace("/", "-").replace("_", "-")
    if not cleaned:
        raise ValueError("Symbol cannot be empty.")

    quote = quote.strip().upper()
    if "-" in cleaned:
        parts = cleaned.split("-")
        if len(parts) == 2:
            return f"{parts[0]}-{parts[1]}"
    return f"{cleaned}-{quote}"


def fetch_coinbase_spot_price(pair: str = "BTC-USD", timeout: float = 5.0) -> float:
    """Fetch current spot price from Coinbase v2 API."""
    product = normalize_coinbase_symbol(pair)
    url = f"{COINBASE_V2_BASE}/prices/{product}/spot"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) skfolio-catton/1.6.0"},
    )
    data = fetch_json_with_retry(req, timeout=timeout)

    if "data" in data and "amount" in data["data"]:
        return float(data["data"]["amount"])
    raise RuntimeError(f"Failed to fetch Coinbase spot price: {data}")


def fetch_coinbase_candles(
    symbol: str,
    timeframe: str = "1d",
    limit: int = 100,
    timeout: float = 5.0,
) -> pd.DataFrame:
    """Fetch candlestick historical bars from Coinbase Exchange API.

    Returns DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume'].
    """
    if timeframe not in GRANULARITY_MAP:
        raise ValueError(f"Invalid timeframe '{timeframe}'. Supported: {list(GRANULARITY_MAP.keys())}")
    if limit <= 0 or limit > 300:
        raise ValueError("limit must be between 1 and 300.")

    product = normalize_coinbase_symbol(symbol)
    granularity = GRANULARITY_MAP[timeframe]
    url = f"{COINBASE_API_BASE}/products/{product}/candles?granularity={granularity}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) skfolio-catton/1.6.0"},
    )
    raw_bars = fetch_json_with_retry(req, timeout=timeout)

    if not isinstance(raw_bars, list):
        raise RuntimeError(f"Coinbase API error for {product}: {raw_bars}")

    records = []
    # Coinbase returns: [ time, low, high, open, close, volume ]
    for bar in raw_bars[:limit]:
        records.append({
            "date": pd.to_datetime(bar[0], unit="s", utc=True),
            "low": float(bar[1]),
            "high": float(bar[2]),
            "open": float(bar[3]),
            "close": float(bar[4]),
            "volume": float(bar[5]),
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


def fetch_coinbase_multi_assets(
    symbols: list[str],
    timeframe: str = "1d",
    limit: int = 100,
    timeout: float = 5.0,
) -> pd.DataFrame:
    """Fetch Close prices for multiple Coinbase assets and merge into a DataFrame."""
    close_series = {}
    for sym in symbols:
        try:
            df = fetch_coinbase_candles(sym, timeframe=timeframe, limit=limit, timeout=timeout)
            if not df.empty:
                prod = normalize_coinbase_symbol(sym)
                close_series[prod] = df.set_index("date")["close"]
        except Exception as e:
            print(f"[!] Warning: Failed to fetch {sym} from Coinbase: {e}")

    if not close_series:
        return pd.DataFrame()

    merged = pd.DataFrame(close_series).sort_index().ffill().dropna()
    return merged


def main():
    parser = argparse.ArgumentParser(description="Coinbase Institutional Crypto Price Fetcher.")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"], help="Symbols to fetch.")
    parser.add_argument("--timeframe", default="1d", choices=list(GRANULARITY_MAP.keys()), help="Timeframe.")
    parser.add_argument("--limit", type=int, default=30, help="Number of candles.")
    args = parser.parse_args()

    print(f"[*] Fetching Coinbase data for {args.symbols} ({args.timeframe})...")
    df = fetch_coinbase_multi_assets(args.symbols, timeframe=args.timeframe, limit=args.limit)
    if df.empty:
        print("[!] No data returned.")
    else:
        print(f"[+] Successfully fetched {len(df)} bars across {len(df.columns)} assets:")
        print(df.tail())


if __name__ == "__main__":
    main()
