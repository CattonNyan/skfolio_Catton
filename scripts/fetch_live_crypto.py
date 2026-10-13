"""Live Cryptocurrency Market Data Fetcher.

Fetches OHLCV candlestick data from Binance or Upbit without requiring API keys:
- Binance: USDT pairs (e.g., BTC/USDT, ETH/USDT, SOL/USDT)
- Upbit: KRW pairs (e.g., BTC/KRW, ETH/KRW, SOL/KRW)
Saves output as clean CSV and/or Freqtrade feather format.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
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

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False


def data_dict_to_prices(data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert ccxt OHLCV dict to a combined Close price DataFrame indexed by date."""
    if not isinstance(data_dict, dict):
        raise ValueError("data_dict must be a dictionary.")
    close_series: dict[str, pd.Series] = {}
    for symbol, df in data_dict.items():
        if isinstance(df, pd.DataFrame) and "date" in df.columns and "close" in df.columns:
            s = df.set_index("date")["close"]
            close_series[symbol] = s[~s.index.duplicated(keep="last")]
    if not close_series:
        return pd.DataFrame()
    prices = pd.DataFrame(close_series).sort_index().ffill().dropna()
    return prices


def fetch_ohlcv_ccxt(
    exchange_id: str = "binance",
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    limit: int = 500,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data using ccxt public endpoints."""
    if not isinstance(timeframe, str) or re.fullmatch(r"[1-9]\d*[smhdwM]", timeframe) is None:
        raise ValueError("Timeframe must use a positive number followed by s, m, h, d, w, or M.")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a strictly positive integer.")
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


def fetch_orderbook_ccxt(
    exchange_id: str = "binance",
    symbol: str = "BTC/USDT",
    limit: int = 50,
) -> dict[str, Any]:
    """Fetch real-time order book (market depth) using ccxt public endpoints."""
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("Symbol must be a non-empty string.")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a strictly positive integer.")
    if not HAS_CCXT:
        print("[!] ccxt is not installed. Please run: pip install ccxt")
        return {}

    exchange_class = getattr(ccxt, exchange_id.lower(), None)
    if not exchange_class:
        raise ValueError(f"Unsupported exchange: {exchange_id}")

    exchange = exchange_class({"enableRateLimit": True})
    try:
        ob = exchange.fetch_order_book(symbol, limit=limit)
        return ob
    except Exception as err:
        print(f"[!] Failed to fetch orderbook for {symbol} on {exchange_id}: {err}")
        return {}


def compute_live_market_impact(
    order_size: float,
    bids: list,
    asks: list,
    side: str = "buy",
    quote_currency: str = "USDT",
) -> dict[str, Any]:
    """Calculate institutional VWAP market impact and execution slippage across order book depth.

    Parameters
    ----------
    order_size : float
        Order notional size in quote currency (e.g., USDT or KRW). Must be > 0.
    bids : list
        Order book bids [[price, size], ...] sorted descending by price.
    asks : list
        Order book asks [[price, size], ...] sorted ascending by price.
    side : str, default "buy"
        "buy" executes against asks, "sell" executes against bids.
    quote_currency : str, default "USDT"
        Quote currency denominator symbol.

    Returns
    -------
    dict
        Execution metrics including vwap_price, mid_price, spread_bps,
        slippage_bps, price_impact_bps, fully_filled, and unfilled_size.
    """
    if isinstance(order_size, bool) or not isinstance(order_size, (int, float)) or order_size <= 0:
        raise ValueError("order_size must be a strictly positive number.")
    if side not in ("buy", "sell"):
        raise ValueError("side must be either 'buy' or 'sell'.")
    if not bids or not asks:
        raise ValueError("Order book bids and asks cannot be empty.")

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid_price = (best_bid + best_ask) / 2.0
    spread = max(0.0, best_ask - best_bid)
    spread_bps = (spread / mid_price * 10000.0) if mid_price > 0 else 0.0

    levels = asks if side == "buy" else bids
    benchmark_quote = best_ask if side == "buy" else best_bid

    remaining_quote = float(order_size)
    executed_quote = 0.0
    executed_base_qty = 0.0

    for lvl in levels:
        price = float(lvl[0])
        qty = float(lvl[1])
        if price <= 0.0 or qty <= 0.0:
            continue
        level_quote = price * qty

        if level_quote <= remaining_quote:
            fill_quote = level_quote
            fill_qty = qty
        else:
            fill_quote = remaining_quote
            fill_qty = remaining_quote / price

        executed_quote += fill_quote
        executed_base_qty += fill_qty
        remaining_quote -= fill_quote

        if remaining_quote <= 1e-9:
            break

    vwap_price = (executed_quote / executed_base_qty) if executed_base_qty > 0 else benchmark_quote

    if side == "buy":
        slippage_bps = ((vwap_price - mid_price) / mid_price * 10000.0) if mid_price > 0 else 0.0
        price_impact_bps = ((vwap_price - best_ask) / best_ask * 10000.0) if best_ask > 0 else 0.0
    else:
        slippage_bps = ((mid_price - vwap_price) / mid_price * 10000.0) if mid_price > 0 else 0.0
        price_impact_bps = ((best_bid - vwap_price) / best_bid * 10000.0) if best_bid > 0 else 0.0

    unfilled_quote = max(0.0, remaining_quote)
    fully_filled = unfilled_quote <= 1e-6
    fill_pct = (executed_quote / order_size * 100.0) if order_size > 0 else 100.0

    bid_depth = sum(float(b[0]) * float(b[1]) for b in bids if float(b[0]) > 0 and float(b[1]) > 0)
    ask_depth = sum(float(a[0]) * float(a[1]) for a in asks if float(a[0]) > 0 and float(a[1]) > 0)

    return {
        "side": side,
        "order_size": order_size,
        "quote_currency": quote_currency,
        "executed_order_size": executed_quote,
        "executed_base_qty": executed_base_qty,
        "vwap_price": vwap_price,
        "mid_price": mid_price,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "spread_bps": spread_bps,
        "slippage_bps": slippage_bps,
        "price_impact_bps": price_impact_bps,
        "fill_percentage": fill_pct,
        "fully_filled": fully_filled,
        "unfilled_size": unfilled_quote,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
    }


def print_live_orderbook_ladder(
    symbol: str,
    bids: list,
    asks: list,
    depth_levels: int = 5,
    quote_currency: str = "USDT",
) -> None:
    """Print a terminal-friendly order book depth ladder."""
    if depth_levels <= 0:
        return
    bids_sub = bids[:depth_levels]
    asks_sub = asks[:depth_levels]
    q_sym = "$" if quote_currency.upper() in ("USD", "USDT", "USDC") else f"{quote_currency} "

    print(f"\n    [+] Top {min(len(bids_sub), len(asks_sub))} Depth Ladder for {symbol}:")
    print(f"        {'Level':<6} {'Bid Price':>14} {'Bid Qty':>12} | {'Ask Price':>14} {'Ask Qty':>12}")
    print("        " + "-" * 66)
    for idx in range(max(len(bids_sub), len(asks_sub))):
        b_p = f"{q_sym}{float(bids_sub[idx][0]):,.2f}" if idx < len(bids_sub) else "-"
        b_q = f"{float(bids_sub[idx][1]):,.4f}" if idx < len(bids_sub) else "-"
        a_p = f"{q_sym}{float(asks_sub[idx][0]):,.2f}" if idx < len(asks_sub) else "-"
        a_q = f"{float(asks_sub[idx][1]):,.4f}" if idx < len(asks_sub) else "-"
        print(f"        #{idx+1:<5} {b_p:>14} {b_q:>12} | {a_p:>14} {a_q:>12}")
    print()


def export_live_orderbook_csv(
    records: list[dict[str, Any]] | dict[str, Any],
    output_path: Path | str,
) -> Path:
    """Export orderbook depth ladder breakdown or market impact simulation records to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(records, dict):
        if "bids" in records and "asks" in records:
            bids = records.get("bids", [])
            asks = records.get("asks", [])
            rows = []
            for i in range(max(len(bids), len(asks))):
                bid = bids[i] if i < len(bids) else []
                ask = asks[i] if i < len(asks) else []
                b_price = float(bid[0]) if len(bid) >= 2 else np.nan
                b_qty = float(bid[1]) if len(bid) >= 2 else np.nan
                a_price = float(ask[0]) if len(ask) >= 2 else np.nan
                a_qty = float(ask[1]) if len(ask) >= 2 else np.nan
                b_depth = (b_price * b_qty) if (np.isfinite(b_price) and np.isfinite(b_qty)) else np.nan
                a_depth = (a_price * a_qty) if (np.isfinite(a_price) and np.isfinite(a_qty)) else np.nan
                rows.append({
                    "level": i + 1,
                    "bid_price": b_price,
                    "bid_quantity": b_qty,
                    "bid_depth": b_depth,
                    "ask_price": a_price,
                    "ask_quantity": a_qty,
                    "ask_depth": a_depth,
                })
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame([records])
    elif isinstance(records, list):
        df = pd.DataFrame(records)
    else:
        raise ValueError("records must be a list or dictionary.")

    df.to_csv(path, index=False, encoding="utf-8")
    return path


def save_market_data(
    data_dict: dict[str, pd.DataFrame],
    output_dir: Path,
    timeframe: str = "1h",
) -> list[Path]:
    """Save fetched data into CSV and Feather files."""
    if not isinstance(data_dict, dict):
        raise ValueError("data_dict must be a dictionary.")
    if not isinstance(timeframe, str) or re.fullmatch(r"[1-9]\d*[smhdwM]", timeframe) is None:
        raise ValueError("Timeframe must use a positive number followed by s, m, h, d, w, or M.")
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


def export_live_prices_csv(
    prices: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Export pivoted live price series DataFrame to a CSV file."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index=True, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Live Crypto Market Data Fetcher")
    parser.add_argument("--exchange", type=str, default="binance", choices=["binance", "upbit"], help="Exchange ID")
    parser.add_argument("--pairs", nargs="+", default=None, help="Pairs to fetch (e.g. BTC/USDT ETH/USDT or BTC/KRW ETH/KRW)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (e.g. 15m, 1h, 1d)")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles (max usually 500~1000)")
    parser.add_argument("--output-dir", type=str, default="data/live", help="Output directory")
    parser.add_argument("--optimize", action="store_true", help="Run portfolio optimization immediately after fetching")
    parser.add_argument("--export-csv", type=str, default=None, help="Export combined close prices to CSV")
    parser.add_argument("--output-csv", type=str, default=None, help="Alias for --export-csv")
    parser.add_argument("--orderbook", action="store_true", help="Fetch real-time order book depth ladder")
    parser.add_argument("--order-size", type=float, default=None, help="Notional order size in quote currency (e.g. 50000.0) for execution impact simulation")
    parser.add_argument("--side", type=str, choices=["buy", "sell"], default="buy", help="Order execution side (buy/sell)")
    parser.add_argument("--depth-levels", type=int, default=5, help="Number of order book depth levels to display (default: 5)")
    parser.add_argument("--export-orderbook-csv", type=str, default=None, help="Export order book depth ladder or simulation metrics to CSV")
    args = parser.parse_args()

    default_pairs = (
        ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        if args.exchange == "binance"
        else ["BTC/KRW", "ETH/KRW", "SOL/KRW"]
    )
    pairs = args.pairs if args.pairs else default_pairs
    quote_curr = "KRW" if args.exchange == "upbit" else "USDT"

    print("==========================================================")
    print(f"  Fetching Live Crypto Data from {args.exchange.upper()}  ")
    print("==========================================================")

    if args.orderbook:
        print(f"\n[*] Fetching real-time order books from {args.exchange.upper()}...")
        sim_records = []
        for symbol in pairs:
            ob = fetch_orderbook_ccxt(exchange_id=args.exchange, symbol=symbol, limit=max(50, args.depth_levels))
            if ob and "bids" in ob and "asks" in ob:
                print_live_orderbook_ladder(symbol, ob["bids"], ob["asks"], depth_levels=args.depth_levels, quote_currency=quote_curr)

                if args.order_size and args.order_size > 0:
                    impact = compute_live_market_impact(
                        order_size=args.order_size,
                        bids=ob["bids"],
                        asks=ob["asks"],
                        side=args.side,
                        quote_currency=quote_curr,
                    )
                    impact["symbol"] = symbol
                    sim_records.append(impact)
                    q_prefix = "$" if quote_curr == "USDT" else f"{quote_curr} "
                    print(f"    [Execution Simulation - {symbol}]")
                    print(f"      Side: {args.side.upper()} | Size: {q_prefix}{args.order_size:,.2f}")
                    print(f"      VWAP Price: {q_prefix}{impact['vwap_price']:,.2f} | Mid: {q_prefix}{impact['mid_price']:,.2f}")
                    print(f"      Spread: {impact['spread_bps']:.1f} bps | Slippage: {impact['slippage_bps']:.1f} bps | Impact: {impact['price_impact_bps']:.1f} bps")
                    print(f"      Filled: {impact['fill_percentage']:.1f}% ({'FULLY FILLED' if impact['fully_filled'] else 'PARTIAL'})")

        if args.export_orderbook_csv:
            target_records = sim_records if sim_records else (ob if len(pairs) == 1 and 'ob' in locals() and ob else sim_records)
            if target_records:
                out = export_live_orderbook_csv(target_records, args.export_orderbook_csv)
                print(f"[+] Exported order book data to: {out}")

    data = fetch_ohlcv_ccxt(
        exchange_id=args.exchange,
        symbols=pairs,
        timeframe=args.timeframe,
        limit=args.limit,
    )

    if data:
        save_market_data(data, Path(args.output_dir), timeframe=args.timeframe)
        csv_target = args.export_csv or args.output_csv
        if csv_target:
            prices = data_dict_to_prices(data)
            if not prices.empty:
                out = export_live_prices_csv(prices, csv_target)
                print(f"[+] Exported live price matrix to: {out}")
        if args.optimize:
            prices = data_dict_to_prices(data)
            if not prices.empty:
                print("\n[*] Running live portfolio optimization on fetched market prices...")
                from scripts.crypto_portfolio_optimizer import run_optimization
                run_optimization(prices)


if __name__ == "__main__":
    main()
