"""Korean Cryptocurrency Exchange Fee & Portfolio Drag Simulator.

Evaluates the real compounding drag of trading fees, maker/taker distribution,
and KRW withdrawal charges across major South Korean exchanges (Upbit, Bithumb, Coinone, Korbit)
on annual portfolio returns and periodic rebalancing turnovers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np


KOREAN_EXCHANGE_PRESETS: dict[str, dict[str, float | str]] = {
    "upbit": {
        "name": "Upbit (KRW Market)",
        "maker_fee": 0.0005,  # 0.05%
        "taker_fee": 0.0005,  # 0.05%
        "withdrawal_fee_krw": 1000.0,
    },
    "bithumb_coupon": {
        "name": "Bithumb (Fee Coupon Applied)",
        "maker_fee": 0.0004,  # 0.04%
        "taker_fee": 0.0004,  # 0.04%
        "withdrawal_fee_krw": 1000.0,
    },
    "bithumb_standard": {
        "name": "Bithumb (Standard Rate)",
        "maker_fee": 0.0025,  # 0.25%
        "taker_fee": 0.0025,  # 0.25%
        "withdrawal_fee_krw": 1000.0,
    },
    "coinone": {
        "name": "Coinone (KRW Market)",
        "maker_fee": 0.0020,  # 0.20%
        "taker_fee": 0.0020,  # 0.20%
        "withdrawal_fee_krw": 1000.0,
    },
    "korbit": {
        "name": "Korbit (KRW Market)",
        "maker_fee": 0.0005,  # 0.05%
        "taker_fee": 0.0005,  # 0.05%
        "withdrawal_fee_krw": 1000.0,
    },
}


def get_korean_exchange_preset(name: str) -> dict[str, float | str]:
    """Retrieve default fee structure for a Korean exchange."""
    key = name.strip().lower()
    if key not in KOREAN_EXCHANGE_PRESETS:
        raise ValueError(
            f"Unknown exchange '{name}'. Available presets: {sorted(KOREAN_EXCHANGE_PRESETS.keys())}"
        )
    return dict(KOREAN_EXCHANGE_PRESETS[key])


def compute_krw_fee_drag(
    portfolio_value_krw: float,
    annual_turnover: float = 4.0,
    maker_fee: float = 0.0005,
    taker_fee: float = 0.0005,
    maker_ratio: float = 0.5,
    annual_withdrawals: int = 12,
    withdrawal_fee_krw: float = 1000.0,
) -> dict[str, object]:
    """
    Calculate annual fee drag and break-even trading margin.

    Parameters:
    - portfolio_value_krw: Total capital in KRW
    - annual_turnover: Portfolio turnover ratio per year (e.g., 4.0 for 400% annual trading volume)
    - maker_fee: Maker fee rate (e.g. 0.0005 for 0.05%)
    - taker_fee: Taker fee rate (e.g. 0.0005 for 0.05%)
    - maker_ratio: Proportion of orders executed as maker (0.0 to 1.0)
    - annual_withdrawals: Number of KRW cash withdrawals per year
    - withdrawal_fee_krw: Bank wire withdrawal fee per transaction (typically 1,000 KRW)
    """
    if isinstance(portfolio_value_krw, bool) or not isinstance(portfolio_value_krw, (int, float)) or not math.isfinite(portfolio_value_krw) or portfolio_value_krw <= 0:
        raise ValueError("Portfolio value must be a strictly positive finite number.")
    if isinstance(annual_turnover, bool) or not isinstance(annual_turnover, (int, float)) or not math.isfinite(annual_turnover) or annual_turnover < 0:
        raise ValueError("Annual turnover must be a non-negative finite number.")
    if isinstance(maker_fee, bool) or not isinstance(maker_fee, (int, float)) or not math.isfinite(maker_fee) or maker_fee < 0 or maker_fee > 1:
        raise ValueError("Maker fee must be between 0.0 and 1.0.")
    if isinstance(taker_fee, bool) or not isinstance(taker_fee, (int, float)) or not math.isfinite(taker_fee) or taker_fee < 0 or taker_fee > 1:
        raise ValueError("Taker fee must be between 0.0 and 1.0.")
    if isinstance(maker_ratio, bool) or not isinstance(maker_ratio, (int, float)) or not math.isfinite(maker_ratio) or maker_ratio < 0 or maker_ratio > 1:
        raise ValueError("Maker ratio must be between 0.0 and 1.0.")
    if isinstance(annual_withdrawals, bool) or not isinstance(annual_withdrawals, int) or annual_withdrawals < 0:
        raise ValueError("Annual withdrawals must be a non-negative integer.")
    if isinstance(withdrawal_fee_krw, bool) or not isinstance(withdrawal_fee_krw, (int, float)) or not math.isfinite(withdrawal_fee_krw) or withdrawal_fee_krw < 0:
        raise ValueError("Withdrawal fee must be a non-negative finite number.")

    # Weighted trade fee rate
    weighted_fee_rate = (maker_fee * maker_ratio) + (taker_fee * (1.0 - maker_ratio))

    # Total trading volume per year
    annual_trade_volume_krw = portfolio_value_krw * annual_turnover

    # Annual trading fees
    annual_trading_fees_krw = annual_trade_volume_krw * weighted_fee_rate

    # Annual withdrawal fees
    annual_cash_out_fees_krw = annual_withdrawals * withdrawal_fee_krw

    # Total fees and drag
    total_annual_fees_krw = annual_trading_fees_krw + annual_cash_out_fees_krw
    fee_drag_pct = (total_annual_fees_krw / portfolio_value_krw) * 100.0
    effective_bps = (total_annual_fees_krw / annual_trade_volume_krw * 10000.0) if annual_trade_volume_krw > 0 else 0.0

    return {
        "portfolio_value_krw": round(portfolio_value_krw, 2),
        "annual_turnover": round(annual_turnover, 2),
        "annual_trade_volume_krw": round(annual_trade_volume_krw, 2),
        "weighted_fee_rate_pct": round(weighted_fee_rate * 100.0, 4),
        "annual_trading_fees_krw": round(annual_trading_fees_krw, 2),
        "annual_withdrawal_fees_krw": round(annual_cash_out_fees_krw, 2),
        "total_annual_fees_krw": round(total_annual_fees_krw, 2),
        "fee_drag_pct": round(fee_drag_pct, 4),
        "effective_cost_bps": round(effective_bps, 2),
        "breakeven_gross_hurdle_pct": round(fee_drag_pct, 4),
    }


def print_krw_fee_report(res: dict[str, object], exchange_name: str = "Custom"):
    """Print formatted terminal report of fee analysis."""
    print("================================================================================")
    print(f"        KOREAN CRYPTO EXCHANGE FEE DRAG REPORT ({exchange_name})              ")
    print("================================================================================")
    print(f"포트폴리오 평가액 (Capital)        : KRW {res['portfolio_value_krw']:,.0f}")
    print(f"연간 포트폴리오 회전율 (Turnover)  : {res['annual_turnover']:.1f}x")
    print(f"연간 총 거래대금 (Trading Volume)  : KRW {res['annual_trade_volume_krw']:,.0f}")
    print(f"가중 평균 거래 수수료율            : {res['weighted_fee_rate_pct']:.4f}%")
    print("--------------------------------------------------------------------------------")
    print(f"연간 누적 매매 수수료             : KRW {res['annual_trading_fees_krw']:,.0f}")
    print(f"연간 원화 출금 수수료             : KRW {res['annual_withdrawal_fees_krw']:,.0f}")
    print(f"연간 총 금융 비용 (Total Fees)     : KRW {res['total_annual_fees_krw']:,.0f}")
    print("--------------------------------------------------------------------------------")
    print(f"수익률 잠식률 (Annual Fee Drag)    : {res['fee_drag_pct']:.2f}% / year")
    print(f"거래대금 대비 실효 비용            : {res['effective_cost_bps']:.1f} bps")
    print(f"손익분기 최소 요구수익률           : +{res['breakeven_gross_hurdle_pct']:.2f}%")
    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Korean Crypto Exchange Fee Drag Calculator")
    parser.add_argument(
        "--exchange",
        choices=list(KOREAN_EXCHANGE_PRESETS.keys()),
        default="upbit",
        help="Korean exchange preset (default: upbit)",
    )
    parser.add_argument("--capital", type=float, default=10000000.0, help="Portfolio value in KRW (default: 10,000,000)")
    parser.add_argument("--turnover", type=float, default=4.0, help="Annual turnover multiplier (default: 4.0)")
    parser.add_argument("--maker-ratio", type=float, default=0.5, help="Proportion of maker orders (0.0 to 1.0)")
    parser.add_argument("--withdrawals", type=int, default=12, help="Annual KRW bank withdrawals (default: 12)")
    parser.add_argument("--export-json", type=str, default="", help="Path to export results JSON")
    args = parser.parse_args()

    preset = get_korean_exchange_preset(args.exchange)
    res = compute_krw_fee_drag(
        portfolio_value_krw=args.capital,
        annual_turnover=args.turnover,
        maker_fee=float(preset["maker_fee"]),
        taker_fee=float(preset["taker_fee"]),
        maker_ratio=args.maker_ratio,
        annual_withdrawals=args.withdrawals,
        withdrawal_fee_krw=float(preset["withdrawal_fee_krw"]),
    )

    print_krw_fee_report(res, exchange_name=str(preset["name"]))

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[+] Fee report exported to: {out_path}")


if __name__ == "__main__":
    main()
