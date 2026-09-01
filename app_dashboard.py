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
    MarketDataUnavailableError,
    export_freqtrade_allocation,
    load_market_data,
)
from scripts.crypto_rebalancing_backtest import simulate_rebalancing

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


def create_rebalancing_nav_chart(nav_port: pd.Series, nav_eq: pd.Series, nav_bh: pd.Series) -> go.Figure:
    """Create comparison line chart for rebalancing backtest."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav_port.index, y=nav_port.values, mode="lines", name=nav_port.name, line=dict(color="#00C853", width=2.5)))
    fig.add_trace(go.Scatter(x=nav_eq.index, y=nav_eq.values, mode="lines", name=nav_eq.name, line=dict(color="#2979FF", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=nav_bh.index, y=nav_bh.values, mode="lines", name=nav_bh.name, line=dict(color="#FF9100", width=1.5, dash="dash")))
    fig.update_layout(
        title="주기적 리밸런싱 포트폴리오 자산 가치(NAV) 추이",
        xaxis_title="일시",
        yaxis_title="순자산 가치 (NAV, 초기값 = 1.0)",
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
    is_synthetic = data_source == "합성 시뮬레이션 데이터"
    try:
        prices, provenance = load_market_data(
            timeframe=timeframe,
            use_synthetic=is_synthetic,
            synthetic_periods=1000,
        )
    except MarketDataUnavailableError as error:
        st.error(f"실데이터를 불러오지 못했습니다: {error}")
        st.info("합성데이터 분석이 필요하면 사이드바에서 '합성 시뮬레이션 데이터'를 명시적으로 선택하세요.")
        return

    if is_synthetic:
        st.warning("⚠️ 데이터 모드: SYNTHETIC — 시뮬레이션 전용이며 Freqtrade 설정 내보내기가 차단됩니다.")
    else:
        st.success(
            f"✅ 데이터 모드: REAL — {provenance} "
            f"({len(prices)}개 캔들, 페어: {', '.join(prices.columns)})"
        )

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

    # 4. Tab Interface
    tab_opt, tab_rebalance = st.tabs(["📊 포트폴리오 최적화 & 자산배분", "🔄 주기적 리밸런싱 롤링 백테스트"])

    with tab_opt:
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

        # Charts Row 1: Pie Chart & Weights Table
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

            if is_synthetic:
                st.caption("합성데이터 결과는 실제 Freqtrade 설정으로 내보낼 수 없습니다.")

            if st.button(
                "🚀 Freqtrade config.json으로 최적 비중 내보내기",
                disabled=is_synthetic,
            ):
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

        # Charts Row 2: Correlation & Cumulative Returns
        col_cum, col_corr = st.columns([3, 2])

        with col_cum:
            fig_cum = create_cumulative_return_chart(returns, weights_dict)
            st.plotly_chart(fig_cum, use_container_width=True)

        with col_corr:
            corr_df = returns.corr()
            fig_corr = create_correlation_heatmap(corr_df)
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab_rebalance:
        st.subheader("🔄 주기적 포트폴리오 리밸런싱(Rolling Window) 백테스트")
        st.caption("일정 주기마다 최적 비중을 재계산하고 자산을 재조정(Rebalancing)할 때의 실제 워크포워드 성과를 측정합니다.")

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            train_bars = st.slider("학습 윈도우 크기 (Lookback Bars)", min_value=50, max_value=500, value=200, step=25)
        with col_r2:
            rebal_bars = st.slider("리밸런싱 주기 (Rebalance Every N Bars)", min_value=10, max_value=100, value=30, step=5)
        with col_r3:
            fee_rate = st.number_input("거래 수수료율 (Fee Rate)", min_value=0.0, max_value=0.01, value=0.001, step=0.0005, format="%.4f")

        if st.button("🚀 롤링 리밸런싱 백테스트 실행", key="btn_run_rebalance"):
            with st.spinner("리밸런싱 워크포워드 시뮬레이션 계산 중..."):
                try:
                    clean_model = "Equal Weight"
                    if "Risk Parity" in model_type:
                        clean_model = "Risk Parity"
                    elif "Max Sharpe" in model_type:
                        clean_model = "Max Sharpe"
                    elif "Min Variance" in model_type:
                        clean_model = "Min Variance"
                    elif "HRP" in model_type:
                        clean_model = "HRP"

                    reb_res = simulate_rebalancing(
                        prices=prices,
                        train_bars=train_bars,
                        rebalance_freq_bars=rebal_bars,
                        fee_rate=fee_rate,
                        model_choice=clean_model,
                    )
                    s = reb_res["summary"]

                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("총 수익률 (전략)", f"{s['Total Return (%)']:.2f}%", f"{s['Total Return (%)'] - s['Buy & Hold Return (%)']:+.2f}% vs B&H")
                    rc2.metric("최대 낙폭 (MDD)", f"{s['Max Drawdown (%)']:.2f}%", f"{s['Max Drawdown (%)'] - s['Buy & Hold MDD (%)']:+.2f}% vs B&H")
                    rc3.metric("연환산 샤프 지수", f"{s['Sharpe Ratio (Ann.)']:.3f}")
                    rc4.metric("평균 회전율 (Turnover)", f"{s['Average Turnover (%)']:.2f}%")

                    fig_reb_nav = create_rebalancing_nav_chart(reb_res["nav_port"], reb_res["nav_eq"], reb_res["nav_bh"])
                    st.plotly_chart(fig_reb_nav, use_container_width=True)

                    comp_df = pd.DataFrame({
                        "포트폴리오 전략 / 벤치마크": [f"{clean_model} (정기 리밸런싱)", "동일 가중 (1/N 균등)", f"{assets[0]} (단순 보유 Buy&Hold)"],
                        "누적 수익률": [f"{s['Total Return (%)']:.2f}%", f"{s['Equal Weight Return (%)']:.2f}%", f"{s['Buy & Hold Return (%)']:.2f}%"],
                        "최대 낙폭(MDD)": [f"{s['Max Drawdown (%)']:.2f}%", f"{s['Equal Weight MDD (%)']:.2f}%", f"{s['Buy & Hold MDD (%)']:.2f}%"],
                    })
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                except Exception as ex:
                    st.error(f"시뮬레이션 실행 중 오류 발생: {ex}")


if __name__ == "__main__":
    main()
