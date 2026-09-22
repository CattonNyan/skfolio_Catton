"""Korea Travel Rule Compliance & Safe Transfer Batch Advisor.

Analyzes cryptocurrency withdrawal/transfer sizes against the statutory KRW 1,000,000
Travel Rule reporting threshold (특정 금융거래정보의 보고 및 이용 등에 관한 법률).
Calculates optimal safe batch sizes (e.g. KRW 950,000 safety buffer) to avoid
unexpected deposit lockups, account freezes, or complex identity verification on unsupported exchanges.
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


TRAVEL_RULE_STATUTORY_LIMIT_KRW = 1000000.0  # 1,000,000 KRW
DEFAULT_SAFE_BUFFER_KRW = 950000.0            # 950,000 KRW (5% safety buffer)

COMMON_REMITTANCE_FEE_PRESETS: dict[str, float] = {
    "XRP": 1.0,        # Ripple standard withdrawal fee (1 XRP)
    "TRX": 1.0,        # Tron standard fee (1 TRX)
    "SOL": 0.01,       # Solana network withdrawal fee (~0.01 SOL)
    "BTC": 0.0005,     # Bitcoin native withdrawal fee
    "ETH": 0.005,      # Ethereum ERC20 gas fee
    "USDT": 1.0,       # Tether TRC20 fee (1 USDT)
    "DOGE": 1.0,       # Dogecoin withdrawal fee
    "ADA": 1.0,        # Cardano withdrawal fee
}


def get_coin_transfer_preset(coin_symbol: str) -> float:
    """Retrieve standard Korean exchange withdrawal network fee for a remittance coin."""
    if not isinstance(coin_symbol, str):
        raise ValueError("Coin symbol must be a string.")
    sym = coin_symbol.strip().upper().replace("KRW-", "").replace("/USDT", "").replace("-USDT", "")
    return COMMON_REMITTANCE_FEE_PRESETS.get(sym, 0.0)


def calculate_travel_rule_plan(
    coin_symbol: str,
    target_amount: float,
    coin_price_krw: float,
    threshold_krw: float = TRAVEL_RULE_STATUTORY_LIMIT_KRW,
    safe_buffer_krw: float = DEFAULT_SAFE_BUFFER_KRW,
    network_fee_coins: float = 0.0,
    auto_preset_fee: bool = False,
    interval_minutes: int = 20,
    daily_warning_threshold_krw: float = 5000000.0,
) -> dict[str, object]:
    """
    Calculate Travel Rule applicability, batch transfer recommendations, and anti-structuring intervals.

    Parameters:
    - coin_symbol: Crypto asset symbol (e.g. 'XRP', 'TRX', 'SOL', 'BTC')
    - target_amount: Total quantity of coins to transfer
    - coin_price_krw: Current unit price of the coin in KRW
    - threshold_krw: Legal reporting threshold (default: 1,000,000 KRW)
    - safe_buffer_krw: Conservative batch threshold to avoid market volatility breach
    - network_fee_coins: Withdrawal fee charged per transaction by the originating exchange
    - auto_preset_fee: If True and network_fee_coins is 0, auto-fill from standard Korean presets
    - interval_minutes: Recommended interval between split transfers to minimize AML STR flag risk
    - daily_warning_threshold_krw: Threshold above which total volume triggers aggressive AML monitoring
    """
    if not isinstance(coin_symbol, str) or not coin_symbol.strip():
        raise ValueError("Coin symbol must be a non-empty string.")
    coin_symbol = coin_symbol.strip().upper()

    if isinstance(auto_preset_fee, bool) and auto_preset_fee and network_fee_coins == 0.0:
        network_fee_coins = get_coin_transfer_preset(coin_symbol)

    if isinstance(target_amount, bool) or not isinstance(target_amount, (int, float)) or not math.isfinite(target_amount) or target_amount <= 0:
        raise ValueError("Target transfer amount must be a strictly positive finite number.")
    if isinstance(coin_price_krw, bool) or not isinstance(coin_price_krw, (int, float)) or not math.isfinite(coin_price_krw) or coin_price_krw <= 0:
        raise ValueError("Coin price in KRW must be a strictly positive finite number.")
    if isinstance(threshold_krw, bool) or not isinstance(threshold_krw, (int, float)) or not math.isfinite(threshold_krw) or threshold_krw <= 0:
        raise ValueError("Threshold must be a strictly positive finite number.")
    if isinstance(safe_buffer_krw, bool) or not isinstance(safe_buffer_krw, (int, float)) or not math.isfinite(safe_buffer_krw) or safe_buffer_krw <= 0:
        raise ValueError("Safe buffer must be a strictly positive finite number.")
    if safe_buffer_krw >= threshold_krw:
        raise ValueError("Safe buffer must be strictly less than statutory threshold.")
    if isinstance(network_fee_coins, bool) or not isinstance(network_fee_coins, (int, float)) or not math.isfinite(network_fee_coins) or network_fee_coins < 0:
        raise ValueError("Network fee must be a non-negative finite number.")
    if isinstance(interval_minutes, bool) or not isinstance(interval_minutes, int) or interval_minutes <= 0:
        raise ValueError("Interval minutes must be a strictly positive integer.")
    if isinstance(daily_warning_threshold_krw, bool) or not isinstance(daily_warning_threshold_krw, (int, float)) or not math.isfinite(daily_warning_threshold_krw) or daily_warning_threshold_krw <= 0:
        raise ValueError("Daily warning threshold must be a strictly positive finite number.")

    total_value_krw = target_amount * coin_price_krw
    requires_travel_rule = total_value_krw >= threshold_krw

    # Maximum coins allowed under the safe buffer
    max_safe_coin_per_tx = safe_buffer_krw / coin_price_krw

    if requires_travel_rule:
        num_batches = math.ceil(total_value_krw / safe_buffer_krw)
        per_batch_coins = target_amount / num_batches
        per_batch_krw = per_batch_coins * coin_price_krw
        total_fee_coins = network_fee_coins * num_batches
        total_fee_krw = total_fee_coins * coin_price_krw

        total_duration_minutes = (num_batches - 1) * interval_minutes
        anti_structuring_alert = total_value_krw >= daily_warning_threshold_krw or num_batches >= 4

        if anti_structuring_alert:
            aml_note = (
                f"주의: 분할 횟수({num_batches}회) 또는 총액(KRW {total_value_krw:,.0f})이 높아 "
                f"거래소 FDS/STR(이상거래탐지/전신환 분할송금) 의심 계좌로 등록될 수 있습니다. "
                f"회당 최소 {interval_minutes}분 이상의 텀을 두고 송금하거나 합법적 제휴 거래소(VASP)를 이용하세요."
            )
        else:
            aml_note = f"안전: 회당 약 {interval_minutes}분의 시간차를 두고 분할 전송하는 것을 권장합니다."

        advice = (
            f"Transfer exceeds KRW {threshold_krw:,.0f} limit. Mandatory VASP-to-VASP Travel Rule applies. "
            f"If sending to an unregistered exchange or personal wallet, split into {num_batches} batches "
            f"of ~{per_batch_coins:.4f} {coin_symbol} (~KRW {per_batch_krw:,.0f}) with {interval_minutes}m spacing. "
            f"Estimated completion: ~{total_duration_minutes} minutes."
        )
    else:
        num_batches = 1
        per_batch_coins = target_amount
        per_batch_krw = total_value_krw
        total_fee_coins = network_fee_coins
        total_fee_krw = total_fee_coins * coin_price_krw
        total_duration_minutes = 0
        anti_structuring_alert = False
        aml_note = "단일 송금 가능 (100만원 미만 특례)."

        advice = (
            f"Transfer is below KRW {threshold_krw:,.0f} (KRW {total_value_krw:,.0f}). "
            "Simplified withdrawal applies without full Travel Rule KYC exchange verification."
        )

    return {
        "coin_symbol": coin_symbol,
        "coin_price_krw": round(coin_price_krw, 2),
        "target_amount": round(target_amount, 6),
        "total_value_krw": round(total_value_krw, 2),
        "statutory_threshold_krw": round(threshold_krw, 2),
        "safe_buffer_krw": round(safe_buffer_krw, 2),
        "requires_travel_rule": requires_travel_rule,
        "recommended_batches": num_batches,
        "batch_interval_minutes": interval_minutes,
        "total_duration_minutes": total_duration_minutes,
        "total_duration_hours": round(total_duration_minutes / 60.0, 2),
        "anti_structuring_alert": anti_structuring_alert,
        "anti_structuring_note": aml_note,
        "per_batch_coins": round(per_batch_coins, 6),
        "per_batch_krw": round(per_batch_krw, 2),
        "max_safe_single_amount": round(max_safe_coin_per_tx, 6),
        "total_network_fee_coins": round(total_fee_coins, 6),
        "total_network_fee_krw": round(total_fee_krw, 2),
        "fee_pct_of_transfer": round((total_fee_krw / total_value_krw) * 100.0, 4) if total_value_krw > 0 else 0.0,
        "compliance_advice": advice,
    }


def print_travel_rule_report(res: dict[str, object]):
    """Print terminal report of Travel Rule plan."""
    status_str = "적용 대상 (MANDATORY TRAVEL RULE)" if res["requires_travel_rule"] else "면제 대상 (SIMPLIFIED TRANSFER)"
    print("================================================================================")
    print("           KOREA TRAVEL RULE COMPLIANCE & SAFE TRANSFER PLAN                    ")
    print("================================================================================")
    print(f"전송 코인 심볼 (Asset)             : {res['coin_symbol']}")
    print(f"현재 코인 단가 (Price)             : KRW {res['coin_price_krw']:,.0f}")
    print(f"총 전송 수량 (Total Coins)         : {res['target_amount']:,.4f} {res['coin_symbol']}")
    print(f"총 원화 환산액 (Total Value)       : KRW {res['total_value_krw']:,.0f}")
    print(f"법정 보고 기준액 (Threshold)       : KRW {res['statutory_threshold_krw']:,.0f}")
    print("--------------------------------------------------------------------------------")
    print(f"트래블룰 적용 여부                 : {status_str}")
    print(f"안전 버퍼 기준 (Safe Buffer)       : KRW {res['safe_buffer_krw']:,.0f} (변동성 대비)")
    print(f"권장 분할 전송 횟수 (Batches)      : {res['recommended_batches']}회")
    print(f"1회당 분할 전송 수량               : {res['per_batch_coins']:,.4f} {res['coin_symbol']} (~KRW {res['per_batch_krw']:,.0f})")
    print(f"예상 누적 전송 수수료             : KRW {res['total_network_fee_krw']:,.0f} ({res['fee_pct_of_transfer']:.4f}%)")
    if res.get("total_duration_minutes", 0) > 0:
        print(f"예상 소요 시간 (Estimated Time)    : 약 {res['total_duration_minutes']}분 ({res['total_duration_hours']:.2f}시간)")
    print("--------------------------------------------------------------------------------")
    print(f"준수 가이드                        : {res['compliance_advice']}")
    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Korea Travel Rule Safe Transfer Advisor")
    parser.add_argument("--coin", type=str, default="XRP", help="Crypto symbol (e.g. XRP, TRX, SOL, BTC)")
    parser.add_argument("--amount", type=float, default=2000.0, help="Total amount to transfer")
    parser.add_argument("--price-krw", type=float, default=None, help="Coin price in KRW (default: fetch live from Upbit)")
    parser.add_argument("--network-fee", type=float, default=1.0, help="Withdrawal fee per tx in coins")
    parser.add_argument("--safe-buffer", type=float, default=DEFAULT_SAFE_BUFFER_KRW, help="Safe buffer in KRW")
    parser.add_argument("--export-json", type=str, default="", help="Path to export results JSON")
    args = parser.parse_args()

    price = args.price_krw
    if price is None:
        try:
            from scripts.fetch_upbit_crypto import fetch_upbit_ticker
            m_code = f"KRW-{args.coin.upper()}"
            ticker = fetch_upbit_ticker([m_code])
            price = float(ticker[m_code])
        except Exception:
            price = 1150.0 if args.coin.upper() == "XRP" else 100000.0

    res = calculate_travel_rule_plan(
        coin_symbol=args.coin,
        target_amount=args.amount,
        coin_price_krw=price,
        safe_buffer_krw=args.safe_buffer,
        network_fee_coins=args.network_fee,
    )

    print_travel_rule_report(res)

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[+] Travel rule plan exported to: {out_path}")


if __name__ == "__main__":
    main()
