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
from typing import Any

import numpy as np
import pandas as pd

from scripts.http_retry_helper import fetch_json_with_retry

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

    if count <= 0:
        raise ValueError("count must be strictly positive.")

    order_curr, pay_curr = normalize_bithumb_symbol(symbol, payment_currency)
    url = f"{BITHUMB_API_BASE}/candlestick/{order_curr}_{pay_curr}/{chart_interval}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) skfolio-catton/1.6.2"},
    )
    data = fetch_json_with_retry(req, timeout=timeout)

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

    df = pd.DataFrame(records).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_bithumb_spot_price(
    symbol: str,
    payment_currency: str = "KRW",
    timeout: float = 5.0,
) -> float:
    """Fetch real-time spot closing price from Bithumb public ticker API."""
    order_curr, pay_curr = normalize_bithumb_symbol(symbol, payment_currency)
    url = f"{BITHUMB_API_BASE}/ticker/{order_curr}_{pay_curr}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) skfolio-catton/1.6.2"},
    )
    data = fetch_json_with_retry(req, timeout=timeout)
    if data.get("status") != "0000" or "data" not in data:
        raise RuntimeError(f"Bithumb API error for {order_curr}_{pay_curr}: {data.get('message', 'Unknown error')}")

    closing_price = float(data["data"].get("closing_price", 0.0))
    if closing_price <= 0.0:
        raise ValueError(f"Invalid closing price {closing_price} for {order_curr}_{pay_curr}")
    return closing_price


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


def fetch_bithumb_orderbook(
    symbol: str,
    payment_currency: str = "KRW",
    count: int = 30,
    timeout: float = 5.0,
) -> dict:
    """Fetch order book from Bithumb public REST API.

    Returns dict with keys: 'timestamp', 'order_currency', 'payment_currency', 'bids', 'asks'.
    """
    if count <= 0 or count > 50:
        raise ValueError("count must be between 1 and 50.")

    order_curr, pay_curr = normalize_bithumb_symbol(symbol, payment_currency)
    url = f"{BITHUMB_API_BASE}/orderbook/{order_curr}_{pay_curr}?count={count}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) skfolio-catton/1.6.1"},
    )
    data = fetch_json_with_retry(req, timeout=timeout)

    if data.get("status") != "0000" or "data" not in data:
        raise RuntimeError(f"Bithumb API error for {order_curr}_{pay_curr}: {data.get('message', 'Unknown error')}")

    return data["data"]


def compute_bithumb_spread(orderbook_data: dict) -> dict[str, float]:
    """Calculate best bid, best ask, spread in KRW, spread in bps, and market depth from Bithumb order book."""
    if not isinstance(orderbook_data, dict):
        raise ValueError("orderbook_data must be a dictionary.")

    bids = orderbook_data.get("bids", [])
    asks = orderbook_data.get("asks", [])
    if not bids or not asks:
        raise ValueError("Order book bids and asks cannot be empty.")

    best_bid = float(bids[0]["price"])
    best_ask = float(asks[0]["price"])
    mid_price = (best_bid + best_ask) / 2.0
    spread_krw = max(0.0, best_ask - best_bid)
    spread_bps = (spread_krw / mid_price * 10000.0) if mid_price > 0 else 0.0

    bid_depth_krw = sum(float(b["price"]) * float(b["quantity"]) for b in bids)
    ask_depth_krw = sum(float(a["price"]) * float(a["quantity"]) for a in asks)

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid_price,
        "spread_krw": spread_krw,
        "spread_bps": round(spread_bps, 2),
        "bid_depth_krw": round(bid_depth_krw, 2),
        "ask_depth_krw": round(ask_depth_krw, 2),
    }


def compute_bithumb_market_impact(
    orderbook_data: dict[str, Any],
    order_size_krw: float,
    side: str = "buy",
) -> dict[str, Any]:
    """Calculate institutional VWAP, market impact, and execution slippage in bps for an order in Bithumb.

    Parameters
    ----------
    orderbook_data : dict[str, Any]
        Bithumb order book dictionary containing 'bids' and 'asks'.
    order_size_krw : float
        Target order execution size in KRW (> 0).
    side : str
        Execution side: 'buy' (consumes ask ladder) or 'sell' (consumes bid ladder).

    Returns
    -------
    dict[str, Any]
        Execution simulation metrics including VWAP price, mid price, spread bps,
        slippage bps, price impact bps, fill percentage, and fully filled flag.
    """
    if isinstance(order_size_krw, bool) or not isinstance(order_size_krw, (int, float)) or not np.isfinite(order_size_krw) or order_size_krw <= 0:
        raise ValueError("order_size_krw must be a finite, strictly positive number.")
    side_clean = str(side).strip().lower()
    if side_clean not in ("buy", "sell"):
        raise ValueError("side must be either 'buy' or 'sell'.")
    if not isinstance(orderbook_data, dict):
        raise ValueError("orderbook_data must be a dictionary.")

    bids = orderbook_data.get("bids", [])
    asks = orderbook_data.get("asks", [])
    if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
        raise ValueError("Order book bids and asks cannot be empty.")

    best_bid = float(bids[0]["price"])
    best_ask = float(asks[0]["price"])
    mid_price = (best_bid + best_ask) / 2.0
    spread_krw = max(0.0, best_ask - best_bid)
    spread_bps = (spread_krw / mid_price * 10000.0) if mid_price > 0 else 0.0

    remaining_krw = float(order_size_krw)
    executed_krw = 0.0
    executed_base_qty = 0.0

    ladder = asks if side_clean == "buy" else bids
    for level in ladder:
        price = float(level["price"])
        qty = float(level["quantity"])
        if price <= 0.0 or qty <= 0.0:
            continue
        level_krw = price * qty

        if level_krw <= remaining_krw:
            fill_krw = level_krw
            fill_qty = qty
        else:
            fill_krw = remaining_krw
            fill_qty = remaining_krw / price

        executed_krw += fill_krw
        executed_base_qty += fill_qty
        remaining_krw -= fill_krw

        if remaining_krw <= 1e-9:
            break

    benchmark_quote = best_ask if side_clean == "buy" else best_bid
    vwap_price = (executed_krw / executed_base_qty) if executed_base_qty > 0 else benchmark_quote

    if side_clean == "buy":
        slippage_bps = ((vwap_price - mid_price) / mid_price * 10000.0) if mid_price > 0 else 0.0
        price_impact_bps = ((vwap_price - best_ask) / best_ask * 10000.0) if best_ask > 0 else 0.0
    else:
        slippage_bps = ((mid_price - vwap_price) / mid_price * 10000.0) if mid_price > 0 else 0.0
        price_impact_bps = ((best_bid - vwap_price) / best_bid * 10000.0) if best_bid > 0 else 0.0

    unfilled_krw = max(0.0, remaining_krw)
    fully_filled = unfilled_krw <= 1e-6
    fill_pct = (executed_krw / order_size_krw * 100.0) if order_size_krw > 0 else 0.0

    symbol = f"{orderbook_data.get('order_currency', '')}/{orderbook_data.get('payment_currency', 'KRW')}"

    return {
        "symbol": symbol,
        "side": side_clean,
        "order_size_krw": round(float(order_size_krw), 2),
        "executed_krw": round(executed_krw, 2),
        "unfilled_krw": round(unfilled_krw, 2),
        "fill_percentage": round(fill_pct, 2),
        "fully_filled": fully_filled,
        "mid_price": round(mid_price, 2),
        "best_bid": round(best_bid, 2),
        "best_ask": round(best_ask, 2),
        "vwap_price": round(vwap_price, 2),
        "spread_bps": round(spread_bps, 2),
        "slippage_bps": round(slippage_bps, 2),
        "price_impact_bps": round(price_impact_bps, 2),
        "executed_base_qty": round(executed_base_qty, 6),
    }


def export_bithumb_prices_csv(
    df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Export Bithumb price series DataFrame to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True, encoding="utf-8")
    return path


def export_bithumb_orderbook_csv(
    records: list[dict[str, Any]] | dict[str, Any],
    output_path: Path | str,
) -> Path:
    """Export Bithumb orderbook spread, market depth, ladder, or market impact to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(records, dict):
        if "bids" in records and "asks" in records:
            bids = records.get("bids", [])
            asks = records.get("asks", [])
            rows = []
            for i in range(max(len(bids), len(asks))):
                bid = bids[i] if i < len(bids) else {}
                ask = asks[i] if i < len(asks) else {}
                rows.append({
                    "level": i + 1,
                    "bid_price": float(bid.get("price", 0.0)) if bid else np.nan,
                    "bid_quantity": float(bid.get("quantity", 0.0)) if bid else np.nan,
                    "ask_price": float(ask.get("price", 0.0)) if ask else np.nan,
                    "ask_quantity": float(ask.get("quantity", 0.0)) if ask else np.nan,
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


def main():
    parser = argparse.ArgumentParser(description="Fetch candlestick data and orderbook depth from Bithumb public API.")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL", "XRP"], help="List of crypto symbols.")
    parser.add_argument("--interval", default="24h", choices=list(VALID_BITHUMB_INTERVALS), help="Candle interval.")
    parser.add_argument("--count", type=int, default=30, help="Number of candles to fetch.")
    parser.add_argument(
        "--orderbook",
        action="store_true",
        help="Fetch and analyze real-time order book, bid/ask spread, and market depth instead of candle history.",
    )
    parser.add_argument(
        "--orderbook-count",
        type=int,
        default=30,
        help="Orderbook depth level count (1 to 50, default: 30).",
    )
    parser.add_argument(
        "--order-size",
        type=float,
        default=None,
        help="Target order execution size in KRW for institutional VWAP and slippage simulation (e.g. 50000000).",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="buy",
        choices=["buy", "sell"],
        help="Execution side for order simulation ('buy' or 'sell', default: 'buy').",
    )
    parser.add_argument(
        "--depth-levels",
        type=int,
        default=5,
        help="Number of top depth ladder levels to display in terminal view (default: 5).",
    )
    parser.add_argument("--export-csv", type=str, default="", help="Path to export fetched Bithumb prices or orderbook metrics to CSV.")
    args = parser.parse_args()

    if args.orderbook:
        print(f"[*] Fetching Bithumb order book & market depth for {args.symbols} (count={args.orderbook_count})...")
        records = []
        impact_records = []

        print("\n" + "=" * 115)
        print(f"{'Symbol':<12} {'Best Bid (KRW)':>16} {'Best Ask (KRW)':>16} {'Mid Price (KRW)':>16} {'Spread (bps)':>14} {'Bid Depth (KRW)':>18} {'Ask Depth (KRW)':>18}")
        print("-" * 115)

        for sym in args.symbols:
            try:
                order_curr, pay_curr = normalize_bithumb_symbol(sym)
                ob = fetch_bithumb_orderbook(sym, payment_currency=pay_curr, count=args.orderbook_count)
                spread = compute_bithumb_spread(ob)
                rec = {
                    "symbol": f"{order_curr}/{pay_curr}",
                    "best_bid": spread["best_bid"],
                    "best_ask": spread["best_ask"],
                    "mid_price": spread["mid_price"],
                    "spread_krw": spread["spread_krw"],
                    "spread_bps": spread["spread_bps"],
                    "bid_depth_krw": spread["bid_depth_krw"],
                    "ask_depth_krw": spread["ask_depth_krw"],
                }
                records.append(rec)
                print(
                    f"{rec['symbol']:<12} "
                    f"{rec['best_bid']:>16,.0f} "
                    f"{rec['best_ask']:>16,.0f} "
                    f"{rec['mid_price']:>16,.1f} "
                    f"{rec['spread_bps']:>13.2f}bp "
                    f"{rec['bid_depth_krw']:>18,.0f} "
                    f"{rec['ask_depth_krw']:>18,.0f}"
                )

                if args.order_size is not None and args.order_size > 0:
                    impact = compute_bithumb_market_impact(ob, order_size_krw=args.order_size, side=args.side)
                    impact_records.append(impact)

                # Print top depth ladder
                if args.depth_levels > 0:
                    bids = ob.get("bids", [])[:args.depth_levels]
                    asks = ob.get("asks", [])[:args.depth_levels]
                    print(f"\n    [+] Top {len(bids)} Depth Ladder for {rec['symbol']}:")
                    print(f"        {'Level':<6} {'Bid Price (KRW)':>16} {'Bid Qty':>12} | {'Ask Price (KRW)':>16} {'Ask Qty':>12}")
                    print("        " + "-" * 70)
                    for idx in range(max(len(bids), len(asks))):
                        b_p = f"{float(bids[idx]['price']):,.0f}" if idx < len(bids) else "-"
                        b_q = f"{float(bids[idx]['quantity']):,.4f}" if idx < len(bids) else "-"
                        a_p = f"{float(asks[idx]['price']):,.0f}" if idx < len(asks) else "-"
                        a_q = f"{float(asks[idx]['quantity']):,.4f}" if idx < len(asks) else "-"
                        print(f"        #{idx+1:<5} {b_p:>16} {b_q:>12} | {a_p:>16} {a_q:>12}")
                    print()
            except Exception as e:
                print(f"[!] Warning: Failed to fetch orderbook for {sym}: {e}")
        print("=" * 115 + "\n")

        if impact_records:
            print("=" * 115)
            print(f"[*] Bithumb Institutional Order Execution Simulation ({args.side.upper()} order size: KRW {args.order_size:,.0f})")
            print("=" * 115)
            print(f"{'Symbol':<12} {'Side':<6} {'VWAP Fill (KRW)':>18} {'Mid Price (KRW)':>16} {'Slippage (bps)':>16} {'Impact (bps)':>14} {'Fill (%)':>10}")
            print("-" * 115)
            for imp in impact_records:
                print(
                    f"{imp['symbol']:<12} "
                    f"{imp['side'].upper():<6} "
                    f"{imp['vwap_price']:>18,.1f} "
                    f"{imp['mid_price']:>16,.1f} "
                    f"{imp['slippage_bps']:>15.2f}bp "
                    f"{imp['price_impact_bps']:>13.2f}bp "
                    f"{imp['fill_percentage']:>9.1f}%"
                )
            print("=" * 115 + "\n")

        if args.export_csv:
            export_target = impact_records if impact_records else records
            if export_target:
                csv_path = export_bithumb_orderbook_csv(export_target, args.export_csv)
                print(f"[+] Saved Bithumb orderbook metrics to {csv_path}")
        return

    print(f"[*] Fetching Bithumb data for {args.symbols} ({args.interval})...")
    df = fetch_bithumb_multi_assets(args.symbols, chart_interval=args.interval, count=args.count)
    if df.empty:
        print("[!] No data returned.")
    else:
        print(f"[+] Successfully fetched {len(df)} rows across {len(df.columns)} assets:")
        print(df.tail())

    if args.export_csv and not df.empty:
        csv_path = export_bithumb_prices_csv(df, args.export_csv)
        print(f"[+] Saved Bithumb price data to {csv_path}")


if __name__ == "__main__":
    main()

