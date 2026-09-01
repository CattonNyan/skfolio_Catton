"""Live Cryptocurrency Market Data Fetcher.

Fetches OHLCV candlestick data from Binance or Upbit without requiring API keys:
- Binance: USDT pairs (e.g., BTC/USDT, ETH/USDT, SOL/USDT)
- Upbit: KRW pairs (e.g., BTC/KRW, ETH/KRW, SOL/KRW)
Saves output as clean CSV and/or Freqtrade feather format.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False


def fetch_ohlcv_ccxt(
    exchange_id: str = "binance",
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    limit: int = 500,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data using ccxt public endpoints."""
    if not HAS_CCXT:
        print("[!] ccxt is not installed. Please run: pip install ccxt")
        return {}

    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]

    exchange_class = getattr(ccxt, exchange_id.lower(), None)
    if not exchange_class:
        print(f"[!] Unsupported exchange: {exchange_id}")
        return {}

    exchange = exchange_class({"enableRateLimit": True})
    results: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            print(f"[*] Fetching {symbol} ({timeframe}, {limit} bars) from {exchange_id}...")
            raw_data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(
                raw_data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.drop(columns=["timestamp"])
            results[symbol] = df
            print(f"[+] Downloaded {len(df)} candles for {symbol}")
        except Exception as err:
            print(f"[!] Failed to fetch {symbol}: {err}")

    return results


def save_market_data(
    data_dict: dict[str, pd.DataFrame],
    output_dir: Path,
    timeframe: str = "1h",
) -> list[Path]:
    """Save fetched data into CSV and Feather files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for symbol, df in data_dict.items():
        clean_symbol = symbol.replace("/", "_")
        csv_path = output_dir / f"{clean_symbol}-{timeframe}.csv"
        df.to_csv(csv_path, index=False)
        saved_paths.append(csv_path)

        try:
            feather_path = output_dir / f"{clean_symbol}-{timeframe}.feather"
            df.to_feather(feather_path)
            saved_paths.append(feather_path)
        except Exception:
            pass  # pyarrow may not be present in basic environment

    print(f"[+] Successfully saved {len(data_dict)} assets to {output_dir}")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Live Crypto Market Data Fetcher")
    parser.add_argument("--exchange", type=str, default="binance", choices=["binance", "upbit"], help="Exchange ID")
    parser.add_argument("--pairs", nargs="+", default=None, help="Pairs to fetch (e.g. BTC/USDT ETH/USDT or BTC/KRW ETH/KRW)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (e.g. 15m, 1h, 1d)")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles (max usually 500~1000)")
    parser.add_argument("--output-dir", type=str, default="data/live", help="Output directory")
    args = parser.parse_args()

    default_pairs = (
        ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        if args.exchange == "binance"
        else ["BTC/KRW", "ETH/KRW", "SOL/KRW"]
    )
    pairs = args.pairs if args.pairs else default_pairs

    print("==========================================================")
    print(f"  Fetching Live Crypto Data from {args.exchange.upper()}  ")
    print("==========================================================")

    data = fetch_ohlcv_ccxt(
        exchange_id=args.exchange,
        symbols=pairs,
        timeframe=args.timeframe,
        limit=args.limit,
    )

    if data:
        save_market_data(data, Path(args.output_dir), timeframe=args.timeframe)


if __name__ == "__main__":
    main()
