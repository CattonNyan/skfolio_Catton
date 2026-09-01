"""Standalone Interactive HTML Quant Report Exporter.

Generates a fully self-contained, offline-viewable HTML quant report containing:
1. Strategy Summary & Key Performance Metrics Cards
2. Optimal Asset Allocation Donut Chart (Plotly)
3. Asset Weight & Capital Allocation Table
4. Inter-Asset Correlation Heatmap (Plotly)
5. Cumulative Wealth Growth Simulation Chart (Plotly)
6. Freqtrade Pair Whitelist & Stake JSON snippet
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

from scripts.crypto_portfolio_optimizer import (
    find_freqtrade_data_dirs,
    generate_synthetic_crypto_data,
    load_from_feather_dir,
    run_optimization,
)

try:
    import plotly.graph_objects as go
    from app_dashboard import (
        create_correlation_heatmap,
        create_cumulative_return_chart,
        create_pie_chart,
    )
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>skfolio 퀀트 포트폴리오 분석 리포트</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f8f9fa;
            color: #212529;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            padding: 35px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        }}
        .header {{
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #1a73e8;
            font-size: 28px;
        }}
        .header p {{
            margin: 0;
            color: #6c757d;
            font-size: 14px;
        }}
        .metric-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 35px;
        }}
        .card {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 18px;
            text-align: center;
        }}
        .card-label {{
            font-size: 13px;
            color: #6c757d;
            margin-bottom: 8px;
            font-weight: 500;
        }}
        .card-value {{
            font-size: 22px;
            font-weight: 700;
            color: #212529;
        }}
        .card-value.green {{ color: #2e7d32; }}
        .card-value.blue {{ color: #1565c0; }}
        .section {{
            margin-bottom: 40px;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #343a40;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background-color: #f1f3f5;
            color: #495057;
            font-size: 13px;
        }}
        pre {{
            background: #282c34;
            color: #abb2bf;
            padding: 16px;
            border-radius: 8px;
            font-size: 13px;
            overflow-x: auto;
        }}
        .footer {{
            margin-top: 50px;
            text-align: center;
            font-size: 12px;
            color: #adb5bd;
            border-top: 1px solid #e9ecef;
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 skfolio 암호화폐 퀀트 포트폴리오 분석 리포트</h1>
            <p>생성 일시: {created_at} | 분석 모델: {model_name} | 데이터 표본: {sample_count}개 캔들</p>
        </div>

        <div class="metric-cards">
            <div class="card">
                <div class="card-label">최적화 모델</div>
                <div class="card-value blue">{model_name}</div>
            </div>
            <div class="card">
                <div class="card-label">캔들당 기대 수익률</div>
                <div class="card-value green">{mean_return:.4f}%</div>
            </div>
            <div class="card">
                <div class="card-label">캔들당 변동성(위험)</div>
                <div class="card-value">{volatility:.4f}%</div>
            </div>
            <div class="card">
                <div class="card-label">샤프 지수 (Return/Risk)</div>
                <div class="card-value green">{sharpe_ratio:.3f}</div>
            </div>
        </div>

        <div class="section grid-2">
            <div>
                <div class="section-title">📊 최적 자산 배분 비중 (Donut Chart)</div>
                {pie_html}
            </div>
            <div>
                <div class="section-title">📋 코인별 투자금 배분 가이드 (총 자산: {total_wallet:,.0f} USDT)</div>
                {table_html}
            </div>
        </div>

        <div class="section grid-2">
            <div>
                <div class="section-title">📈 누적 수익률(Cumulative Wealth) 시뮬레이션</div>
                {cum_chart_html}
            </div>
            <div>
                <div class="section-title">🔥 코인 간 상관관계 히트맵 (Correlation)</div>
                {heatmap_html}
            </div>
        </div>

        <div class="section">
            <div class="section-title">🚀 Freqtrade 연계 설정 Snippet</div>
            <pre><code>{json_snippet}</code></pre>
        </div>

        <div class="footer">
            Generated by skfolio_Catton | Based on skfolio (BSD 3-Clause License) | Educational & Research Purpose
        </div>
    </div>
</body>
</html>
"""


def generate_html_report(
    prices: pd.DataFrame,
    weights: dict[str, float],
    model_name: str = "Risk Parity (ERC)",
    total_wallet: float = 10000.0,
    output_file: Path | str = "reports/crypto_portfolio_report.html",
) -> Path:
    """Generate and write standalone HTML report."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    returns = prices.pct_change().dropna()
    common_assets = [a for a in returns.columns if a in weights]
    if common_assets:
        returns = returns[common_assets]
        weight_series = pd.Series({a: weights[a] for a in common_assets})
        if weight_series.sum() > 0:
            weight_series = weight_series / weight_series.sum()
        port_ret = returns.dot(weight_series)
    else:
        port_ret = pd.Series(0.0, index=returns.index)

    mean_ret = float(port_ret.mean() * 100)
    vol = float(port_ret.std() * 100)
    sharpe = float(port_ret.mean() / (port_ret.std() + 1e-9))

    # Table HTML
    table_rows = []
    for asset, w in weights.items():
        allocated = w * total_wallet
        table_rows.append(
            f"<tr><td><strong>{asset}</strong></td><td>{w*100:.2f}%</td><td>{allocated:,.2f} USDT</td></tr>"
        )
    table_html = f"""
    <table>
        <thead><tr><th>코인/페어</th><th>최적 비중</th><th>배분 금액</th></tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
    </table>
    """

    # Interactive Charts
    pie_html = ""
    cum_chart_html = ""
    heatmap_html = ""

    if HAS_PLOTLY:
        fig_pie = create_pie_chart(weights)
        fig_cum = create_cumulative_return_chart(returns, weights)
        fig_heat = create_correlation_heatmap(returns.corr())

        pie_html = fig_pie.to_html(full_html=False, include_plotlyjs="cdn")
        cum_chart_html = fig_cum.to_html(full_html=False, include_plotlyjs=False)
        heatmap_html = fig_heat.to_html(full_html=False, include_plotlyjs=False)
    else:
        pie_html = "<p>Plotly is not installed. Visual charts require plotly.</p>"

    # JSON Snippet
    snippet_dict = {
        "source_model": model_name,
        "pair_whitelist": list(weights.keys()),
        "pair_weights": {k: round(float(v), 4) for k, v in weights.items()},
        "stake_amounts": {k: round(float(v) * total_wallet, 2) for k, v in weights.items()},
    }
    import json
    json_snippet = json.dumps(snippet_dict, indent=2)

    full_html = HTML_TEMPLATE.format(
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_name=model_name,
        sample_count=len(prices),
        mean_return=mean_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        total_wallet=total_wallet,
        pie_html=pie_html,
        table_html=table_html,
        cum_chart_html=cum_chart_html,
        heatmap_html=heatmap_html,
        json_snippet=json_snippet,
    )

    output_path.write_text(full_html, encoding="utf-8")
    print(f"[+] Standalone HTML report successfully exported to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export Standalone HTML Quant Report")
    parser.add_argument("--output", type=str, default="reports/crypto_portfolio_report.html", help="Target HTML path")
    parser.add_argument("--model", type=str, default="Risk Parity (ERC)", help="Optimization model name")
    parser.add_argument("--wallet", type=float, default=10000.0, help="Total wallet in USDT")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe")
    parser.add_argument("--use-synthetic", action="store_true", help="Force synthetic data")
    args = parser.parse_args()

    prices = pd.DataFrame()
    if not args.use_synthetic:
        candidate_dirs = find_freqtrade_data_dirs()
        for d in candidate_dirs:
            if d.is_dir():
                prices = load_from_feather_dir(d, timeframe=args.timeframe)
                if not prices.empty:
                    print(f"[*] Loaded data from {d} ({len(prices)} bars)")
                    break

    if prices.empty or args.use_synthetic:
        prices = generate_synthetic_crypto_data(periods=1000)

    # Run quick optimization
    results = run_optimization(prices)
    if not results:
        weights = {col: 1.0 / len(prices.columns) for col in prices.columns}
    else:
        weights = results.get(args.model, list(results.values())[0])

    generate_html_report(
        prices=prices,
        weights=weights,
        model_name=args.model,
        total_wallet=args.wallet,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
