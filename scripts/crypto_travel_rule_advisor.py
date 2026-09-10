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


def calculate_travel_rule_plan(
    coin_symbol: str,
    target_amount: float,
    coin_price_krw: float,
    threshold_krw: float = TRAVEL_RULE_STATUTORY_LIMIT_KRW,
    safe_buffer_krw: float = DEFAULT_SAFE_BUFFER_KRW,
    network_fee_coins: float = 0.0,
) -> dict[str, object]:
    """
    Calculate Travel Rule applicability and batch transfer recommendations.

    Parameters:
    - coin_symbol: Crypto asset symbol (e.g. 'XRP', 'TRX', 'SOL', 'BTC')
    - target_amount: Total quantity of coins to transfer
    - coin_price_krw: Current unit price of the coin in KRW
    - threshold_krw: Legal reporting threshold (default: 1,000,000 KRW)
    - safe_buffer_krw: Conservative batch threshold to avoid market volatility breach
    - network_fee_coins: Withdrawal fee charged per transaction by the originating exchange
    """
    if not isinstance(coin_symbol, str) or not coin_symbol.strip():
        raise ValueError("Coin symbol must be a non-empty string.")
    coin_symbol = coin_symbol.strip().upper()

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

        advice = (
            f"Transfer exceeds KRW {threshold_krw:,.0f} limit. Mandatory VASP-to-VASP Travel Rule applies. "
            f"If sending to an unregistered exchange or personal wallet, split into {num_batches} batches "
            f"of ~{per_batch_coins:.4f} {coin_symbol} (~KRW {per_batch_krw:,.0f}) with adequate time spacing."
        )
    else:
        num_batches = 1
        per_batch_coins = target_amount
        per_batch_krw = total_value_krw
        total_fee_coins = network_fee_coins
        total_fee_krw = total_fee_coins * coin_price_krw

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
        "per_batch_coins": round(per_batch_coins, 6),
        "per_batch_krw": round(per_batch_krw, 2),
        "max_safe_single_amount": round(max_safe_coin_per_tx, 6),
        "total_network_fee_coins": round(total_fee_coins, 6),
        "total_network_fee_krw": round(total_fee_krw, 2),
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
    print(f"예상 누적 전송 수수료             : KRW {res['total_network_fee_krw']:,.0f}")
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
