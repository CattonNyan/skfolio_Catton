"""Interactive Web Dashboard for skfolio_Catton (Streamlit).

Run with:
    streamlit run app_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure local skfolio source is discovered
src_dir = str(Path(__file__).resolve().parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scripts.crypto_portfolio_optimizer import (
    export_freqtrade_allocation,
    find_freqtrade_data_dirs,
    generate_synthetic_crypto_data,
    load_from_feather_dir,
)

# Optional skfolio optimization imports
try:
    from skfolio import RiskMeasure
    from skfolio.optimization import (
        HierarchicalRiskParity,
        MeanVariance,
        ObjectiveFunction,
        RiskBudgeting,
    )
    from skfolio.preprocessing import prices_to_returns
    HAS_SKFOLIO = True
except ImportError:
    HAS_SKFOLIO = False


def create_pie_chart(weights: dict[str, float], title: str = "최적 자산 배분 비중") -> go.Figure:
    """Create an interactive Donut chart of asset weights."""
    labels = list(weights.keys())
    values = [round(v * 100, 2) for v in weights.values()]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                textinfo="label+percent",
                hoverinfo="label+value",
                marker=dict(line=dict(color="#ffffff", width=2)),
            )
        ]
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


def create_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap figure."""
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="코인 간 상관관계 히트맵 (Correlation)",
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_cumulative_return_chart(returns: pd.DataFrame, weights: dict[str, float]) -> go.Figure:
    """Simulate and plot cumulative wealth curves."""
    cum_returns = (1 + returns).cumprod()
    weight_series = pd.Series(weights)
    portfolio_ret = returns.dot(weight_series)
    cum_portfolio = (1 + portfolio_ret).cumprod()

    fig = go.Figure()
    for col in returns.columns:
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns[col],
                mode="lines",
                name=col,
                opacity=0.5,
                line=dict(dash="dot"),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=cum_portfolio.index,
            y=cum_portfolio,
            mode="lines",
            name="⭐ 최적화 포트폴리오",
            line=dict(color="#00C853", width=3),
        )
    )

    fig.update_layout(
        title="누적 수익률(Cumulative Wealth) 비교 시뮬레이션",
        xaxis_title="일시",
        yaxis_title="누적 배수 (초기값 = 1.0)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
    )
    return fig


def main():
    st.set_page_config(
        page_title="skfolio Crypto Dashboard",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 skfolio 암호화폐 퀀트 포트폴리오 최적화 대시보드")
    st.caption("scikit-learn 기반 포트폴리오 최적화 및 Freqtrade 자동매매 연동 플랫폼")

    # 1. Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ 분석 설정")

        data_source = st.selectbox(
            "데이터 소스 선택",
            options=["Freqtrade 로컬 데이터", "합성 시뮬레이션 데이터"],
            index=0,
        )

        timeframe = st.selectbox("타임프레임 (캔들 주기)", options=["15m", "5m", "1h"], index=0)

        model_type = st.selectbox(
            "최적화 알고리즘",
            options=[
                "Risk Parity (ERC, 위험 균등)",
                "Max Sharpe Ratio (샤프 최대화)",
                "Min Variance (최소 분산)",
                "Min Semi-Variance (하방 위험 최소화)",
                "Hierarchical Risk Parity (HRP)",
            ],
            index=0,
        )

        wallet_size = st.number_input(
            "총 투자 자산 (USDT 또는 원화)",
            min_value=100.0,
            max_value=1000000000.0,
            value=10000.0,
            step=500.0,
        )

        st.markdown("---")
        freqtrade_config_path = st.text_input(
            "Freqtrade config.json 경로",
            value="../freqtrade/user_data/config.json",
        )

    # 2. Data Loading
    prices = pd.DataFrame()
    if data_source == "Freqtrade 로컬 데이터":
        candidate_dirs = find_freqtrade_data_dirs()
        if candidate_dirs:
            prices = load_from_feather_dir(candidate_dirs[0], timeframe=timeframe)
            if not prices.empty:
                st.success(f"Freqtrade 시세 데이터 로드 완료 ({len(prices)}개 캔들, 페어: {', '.join(prices.columns)})")
        if prices.empty:
            st.warning("로컬 Freqtrade 데이터를 찾지 못하여 합성 시뮬레이션 데이터를 불러옵니다.")
            prices = generate_synthetic_crypto_data(periods=1000)
    else:
        prices = generate_synthetic_crypto_data(periods=1000)

    if prices.empty:
        st.error("데이터를 로드할 수 없습니다.")
        return

    # Calculate returns
    returns = prices.pct_change().dropna()
    assets = list(prices.columns)

    # 3. Model Optimization Execution
    weights_dict: dict[str, float] = {}
    if HAS_SKFOLIO:
        try:
            if "Max Sharpe" in model_type:
                model = MeanVariance(
                    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
                    risk_measure=RiskMeasure.VARIANCE,
                )
            elif "Min Variance" in model_type:
                model = MeanVariance(
                    objective_function=ObjectiveFunction.MINIMIZE_RISK,
                    risk_measure=RiskMeasure.VARIANCE,
                )
            elif "Min Semi-Variance" in model_type:
                model = MeanVariance(
                    objective_function=ObjectiveFunction.MINIMIZE_RISK,
                    risk_measure=RiskMeasure.SEMI_VARIANCE,
                )
            elif "HRP" in model_type:
                model = HierarchicalRiskParity(risk_measure=RiskMeasure.VARIANCE)
            else:
                model = RiskBudgeting(risk_measure=RiskMeasure.VARIANCE)

            model.fit(returns)
            weights_dict = dict(zip(assets, model.weights_))
        except Exception as e:
            st.warning(f"skfolio 최적화 중 예외 발생({e}). 동일 가중(Equal Weight)으로 폴백합니다.")
            weights_dict = {a: 1.0 / len(assets) for a in assets}
    else:
        # Graceful fallback: Equal weight or inverse volatility
        vols = returns.std()
        inv_vols = 1.0 / (vols + 1e-9)
        weights_dict = (inv_vols / inv_vols.sum()).to_dict()

    # 4. Top KPI Cards
    port_ret = returns.dot(pd.Series(weights_dict))
    mean_ret = port_ret.mean() * 100
    vol = port_ret.std() * 100
    sharpe = port_ret.mean() / (port_ret.std() + 1e-9)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("선택된 최적화 모델", model_type.split("(")[0].strip())
    col2.metric("캔들당 기대 수익률", f"{mean_ret:.4f}%")
    col3.metric("캔들당 변동성(위험)", f"{vol:.4f}%")
    col4.metric("샤프 지수 (Return/Risk)", f"{sharpe:.3f}")

    st.markdown("---")

    # 5. Charts Row 1: Pie Chart & Weights Table
    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        fig_pie = create_pie_chart(weights_dict)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_table:
        st.subheader("📋 코인별 최적 투자금 배분")
        table_df = pd.DataFrame({
            "코인/페어": list(weights_dict.keys()),
            "최적 비중": [f"{w*100:.2f}%" for w in weights_dict.values()],
            "배분 금액": [f"{w * wallet_size:,.2f}" for w in weights_dict.values()],
        })
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        if st.button("🚀 Freqtrade config.json으로 최적 비중 내보내기"):
            cfg_path = Path(freqtrade_config_path)
            res = {model_type: weights_dict}
            success = export_freqtrade_allocation(
                results=res,
                target_path=cfg_path,
                model_name=model_type,
                total_wallet=wallet_size,
            )
            if success:
                st.success(f"Freqtrade 설정 파일({cfg_path})에 최적 비중이 성공적으로 업데이트되었습니다!")
            else:
                st.error("내보내기 실패. 파일 경로를 확인해주세요.")

    st.markdown("---")

    # 6. Charts Row 2: Correlation & Cumulative Returns
    col_cum, col_corr = st.columns([3, 2])

    with col_cum:
        fig_cum = create_cumulative_return_chart(returns, weights_dict)
        st.plotly_chart(fig_cum, use_container_width=True)

    with col_corr:
        corr_df = returns.corr()
        fig_corr = create_correlation_heatmap(corr_df)
        st.plotly_chart(fig_corr, use_container_width=True)


if __name__ == "__main__":
    main()
