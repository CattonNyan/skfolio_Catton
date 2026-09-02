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
    sanitize_weight_constraints,
)
from scripts.crypto_rebalancing_backtest import simulate_rebalancing
from scripts.crypto_monte_carlo import simulate_monte_carlo
from scripts.crypto_stress_tester import evaluate_stress_test
from scripts.crypto_macro_regime import calculate_macro_regime_weights, fetch_fear_and_greed_index
from scripts.crypto_kimchi_premium import compute_kimchi_premium, fetch_live_usd_krw_rate
from scripts.crypto_tax_calculator import compute_crypto_tax_impact

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


@st.cache_data(show_spinner=False)
def cached_load_market_data(timeframe: str, use_synthetic: bool):
    """Cached market data loader to avoid repeated disk reads."""
    return load_market_data(timeframe=timeframe, use_synthetic=use_synthetic)


@st.cache_data(show_spinner=False)
def cached_fit_model(returns_df: pd.DataFrame, model_type: str, min_w: float | None, max_w: float | None) -> dict[str, float]:
    """Cached model optimization fitting with sanitized constraints."""
    assets = list(returns_df.columns)
    min_w, max_w = sanitize_weight_constraints(len(assets), min_w, max_w)
    c_kwargs = {}
    if min_w is not None:
        c_kwargs["min_weights"] = min_w
    if max_w is not None:
        c_kwargs["max_weights"] = max_w

    if "Max Sharpe" in model_type:
        model = MeanVariance(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, risk_measure=RiskMeasure.VARIANCE, **c_kwargs)
    elif "Min Variance" in model_type:
        model = MeanVariance(objective_function=ObjectiveFunction.MINIMIZE_RISK, risk_measure=RiskMeasure.VARIANCE, **c_kwargs)
    elif "Min Semi-Variance" in model_type:
        model = MeanVariance(objective_function=ObjectiveFunction.MINIMIZE_RISK, risk_measure=RiskMeasure.SEMI_VARIANCE, **c_kwargs)
    elif "HRP" in model_type:
        model = HierarchicalRiskParity(risk_measure=RiskMeasure.VARIANCE)
    else:
        model = RiskBudgeting(risk_measure=RiskMeasure.VARIANCE, **c_kwargs)

    model.fit(returns_df)
    return dict(zip(assets, model.weights_))


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
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
    )
    return fig


def create_efficient_frontier_chart(
    returns: pd.DataFrame,
    optimal_weights: dict[str, float],
    num_simulations: int = 400,
) -> go.Figure:
    """Generate 2D interactive Efficient Frontier scatter plot."""
    n_assets = len(returns.columns)
    mean_rets = returns.mean()
    cov_matrix = returns.cov()

    np.random.seed(42)
    weights_matrix = np.random.dirichlet(np.ones(n_assets), size=num_simulations)
    port_returns = np.dot(weights_matrix, mean_rets) * 100
    port_vols = np.sqrt(np.diag(np.dot(weights_matrix, np.dot(cov_matrix, weights_matrix.T)))) * 100
    sharpe_ratios = port_returns / (port_vols + 1e-9)

    fig = go.Figure()

    # Simulated portfolio cloud
    fig.add_trace(
        go.Scatter(
            x=port_vols,
            y=port_returns,
            mode="markers",
            marker=dict(
                color=sharpe_ratios,
                colorscale="Viridis",
                size=5,
                opacity=0.4,
                showscale=True,
                colorbar=dict(title="Sharpe"),
            ),
            name="시뮬레이션 포트폴리오",
            hoverinfo="skip",
        )
    )

    # Individual asset dots
    for asset in returns.columns:
        a_ret = float(mean_rets[asset] * 100)
        a_vol = float(np.sqrt(cov_matrix.loc[asset, asset]) * 100)
        fig.add_trace(
            go.Scatter(
                x=[a_vol],
                y=[a_ret],
                mode="markers+text",
                marker=dict(size=11, symbol="circle", line=dict(width=1.5, color="white")),
                text=[asset],
                textposition="top center",
                name=asset,
            )
        )

    # Current optimal portfolio star
    opt_w = np.array([optimal_weights.get(c, 0.0) for c in returns.columns])
    opt_ret = float(np.dot(opt_w, mean_rets) * 100)
    opt_vol = float(np.sqrt(np.dot(opt_w, np.dot(cov_matrix, opt_w))) * 100)

    fig.add_trace(
        go.Scatter(
            x=[opt_vol],
            y=[opt_ret],
            mode="markers+text",
            marker=dict(size=18, symbol="star", color="#FF1744", line=dict(width=2, color="white")),
            text=["⭐ 최적 포트폴리오"],
            textposition="top center",
            name="⭐ 최적 포트폴리오",
        )
    )

    fig.update_layout(
        title="효율적 투자선(Efficient Frontier) & 리스크-수익률 분포",
        xaxis_title="변동성(리스크, %)",
        yaxis_title="기대 수익률(%)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_monte_carlo_cone_chart(mc_res: dict[str, object]) -> go.Figure:
    """Create fan/cone chart of Monte Carlo wealth paths."""
    days = list(range(len(mc_res["path_p50"])))
    fig = go.Figure()

    # 95% Confidence Band
    fig.add_trace(go.Scatter(
        x=days, y=mc_res["path_p95"],
        mode="lines",
        line=dict(width=0),
        name="상위 95% 경로 (낙관적)",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=days, y=mc_res["path_p05"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0, 200, 83, 0.15)",
        name="95% 신뢰구간 밴드",
    ))

    # Median Path
    fig.add_trace(go.Scatter(
        x=days, y=mc_res["path_p50"],
        mode="lines",
        name="중앙값 (Median)",
        line=dict(color="#00E5FF", width=2.5),
    ))

    fig.update_layout(
        title="몬테카를로 미래 자산 경로 시뮬레이션 (95% 신뢰구간)",
        xaxis_title="경과 일수 (Days)",
        yaxis_title="예상 자산 가치 ($)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
        st.subheader("⚖️ 비중 제약조건 (Constraints)")
        max_weight_pct = st.slider("단일 종목 최대 비중(%)", min_value=20, max_value=100, value=100, step=5)
        min_weight_pct = st.slider("단일 종목 최소 비중(%)", min_value=0, max_value=20, value=0, step=1)
        max_w = max_weight_pct / 100.0 if max_weight_pct < 100 else None
        min_w = min_weight_pct / 100.0 if min_weight_pct > 0 else None

        st.markdown("---")
        freqtrade_config_path = st.text_input(
            "Freqtrade config.json 경로",
            value="../freqtrade/user_data/config.json",
        )

    # 2. Data Loading
    is_synthetic = data_source == "합성 시뮬레이션 데이터"
    try:
        prices, provenance = cached_load_market_data(
            timeframe=timeframe,
            use_synthetic=is_synthetic,
        )
    except MarketDataUnavailableError as error:
        st.error(str(error))
        st.stop()

    returns = prices_to_returns(prices)
    assets = list(returns.columns)

    # 3. Model Optimization
    if HAS_SKFOLIO:
        try:
            weights_dict = cached_fit_model(returns, model_type, min_w, max_w)
        except Exception as e:
            st.warning(f"skfolio 최적화 중 예외 발생({e}). 동일 가중(Equal Weight)으로 폴백합니다.")
            weights_dict = {a: 1.0 / len(assets) for a in assets}
    else:
        # Graceful fallback: Equal weight or inverse volatility
        vols = returns.std()
        inv_vols = 1.0 / (vols + 1e-9)
        weights_dict = (inv_vols / inv_vols.sum()).to_dict()

    # 4. Tab Interface
    tab_opt, tab_rebalance, tab_mc, tab_stress, tab_macro, tab_kimchi, tab_tax = st.tabs([
        "📊 포트폴리오 최적화 & 자산배분",
        "🔄 주기적 리밸런싱 백테스트",
        "🎲 몬테카를로 미래 시뮬레이션",
        "💥 역사적 블랙스완 스트레스 테스트",
        "😨 공포·탐욕 매크로 현금 조절",
        "⚡ 김치 프리미엄 차익거래",
        "💰 세후 순수익률 & 세금 시뮬레이터",
    ])

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
                    data_source=provenance,
                )
                if success:
                    st.success(f"Freqtrade 설정 파일({cfg_path})에 최적 비중이 성공적으로 업데이트되었습니다!")
                else:
                    st.error("내보내기 실패. 파일 경로를 확인해주세요.")

            # CSV Download Button
            csv_bytes = table_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                label="📊 최적 배분 비중 CSV 다운로드",
                data=csv_bytes,
                file_name="crypto_portfolio_allocation.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # HTML Quant Report Download Button
            try:
                from scripts.export_html_report import generate_html_report
                import tempfile
                with tempfile.TemporaryDirectory() as tmp_dir:
                    temp_html_path = Path(tmp_dir) / "report.html"
                    generate_html_report(
                        prices=prices,
                        weights=weights_dict,
                        model_name=model_type,
                        total_wallet=wallet_size,
                        output_file=temp_html_path,
                        data_source=provenance,
                    )
                    html_content = temp_html_path.read_text(encoding="utf-8")
                st.download_button(
                    label="📥 인터랙티브 HTML 퀀트 리포트 다운로드",
                    data=html_content,
                    file_name="crypto_quant_report.html",
                    mime="text/html",
                    use_container_width=True,
                )
            except Exception as e:
                pass

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

        st.markdown("---")
        # Charts Row 3: Efficient Frontier
        st.subheader("🎯 효율적 투자선 (Efficient Frontier)")
        fig_frontier = create_efficient_frontier_chart(returns, weights_dict)
        st.plotly_chart(fig_frontier, use_container_width=True)

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

    with tab_mc:
        st.subheader("🎲 몬테카를로 미래 자산 경로 & VaR/CVaR 시뮬레이션")
        st.caption("기하 브라운 운동(GBM) 기반으로 1,000개 이상의 미래 자산 경로를 시뮬레이션하여 95% 신뢰구간과 최대 손실 위험(VaR)을 산출합니다.")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            mc_days = st.slider("미래 시뮬레이션 기간 (Days)", min_value=30, max_value=365, value=90, step=15)
        with col_m2:
            mc_sims = st.slider("시뮬레이션 경로 수 (Paths)", min_value=200, max_value=2000, value=1000, step=100)

        if st.button("🚀 몬테카를로 시뮬레이션 실행", key="btn_run_mc"):
            with st.spinner("몬테카를로 경로 시뮬레이션 계산 중..."):
                try:
                    mc_res = simulate_monte_carlo(prices, weights_dict, initial_capital=wallet_size, days=mc_days, num_simulations=mc_sims)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("기대 최종 자산 (평균)", f"${mc_res['expected_final_wealth']:,.2f}")
                    m2.metric("중앙값 최종 자산", f"${mc_res['median_final_wealth']:,.2f}")
                    m3.metric("95% VaR (최대 5% 손실)", f"-${mc_res['var_95_dollar']:,.2f}")
                    m4.metric("95% CVaR (극단 평균 손실)", f"-${mc_res['cvar_95_dollar']:,.2f}")

                    fig_cone = create_monte_carlo_cone_chart(mc_res)
                    st.plotly_chart(fig_cone, use_container_width=True)

                    p1, p2, p3 = st.columns(3)
                    p1.metric("원금 손실 확률 (Prob of Loss)", f"{mc_res['prob_loss_pct']:.2f}%")
                    p2.metric("원금 30% 폭락 확률", f"{mc_res['prob_severe_loss_pct']:.2f}%")
                    p3.metric("원금 2배 달성 확률", f"{mc_res['prob_doubling_pct']:.2f}%")
                except Exception as ex:
                    st.error(f"몬테카를로 시뮬레이션 오류: {ex}")

    with tab_stress:
        st.subheader("💥 역사적 크립토 블랙스완 스트레스 테스터")
        st.caption("2020년 코로나 쇼크, 2022년 루나 붕괴, FTX 파산, 2021년 중국 채굴 금지 등 실제 역사적 극단 위기 상황을 현재 포트폴리오에 주입하여 자본 방어력을 진단합니다.")
        try:
            stress_res = evaluate_stress_test(weights_dict, total_wallet=wallet_size)
            stress_df = pd.DataFrame([
                {
                    "역사적 블랙스완 시나리오": k,
                    "포트폴리오 손실률": f"{v['portfolio_loss_pct']:+.2f}%",
                    "예상 손실액": f"-${v['dollar_loss']:,.2f}",
                    "충격 후 잔여 자산": f"${v['remaining_balance']:,.2f}",
                    "리스크 방어 등급": v["resilience_grade"],
                }
                for k, v in stress_res.items()
            ])
            st.dataframe(stress_df, use_container_width=True, hide_index=True)
        except Exception as ex:
            st.error(f"스트레스 테스트 오류: {ex}")

    with tab_macro:
        st.subheader("😨 공포·탐욕 지수 기반 거시 국면 동적 현금(USDT) 조절기")
        st.caption("Alternative.me 암호화폐 공포·탐욕 지수를 실시간 수집하여, 극단적 탐욕 구간에서는 현금 버퍼를 최대 40%까지 자동으로 확보합니다.")
        try:
            fng_val, fng_class = fetch_fear_and_greed_index()
            g1, g2 = st.columns(2)
            g1.metric("현재 공포·탐욕 지수 (Fear & Greed)", f"{fng_val} / 100", fng_class)
            macro_weights = calculate_macro_regime_weights(weights_dict, fear_and_greed_value=fng_val)
            g2.metric("권장 안전자산(현금) 버퍼", f"{macro_weights.get('USDT (Cash)', 0.0)*100:.1f}%")

            macro_df = pd.DataFrame({
                "자산 / 현금 버퍼": list(macro_weights.keys()),
                "거시 조정 비중": [f"{w*100:.2f}%" for w in macro_weights.values()],
                "배분 금액": [f"${w * wallet_size:,.2f}" for w in macro_weights.values()],
            })
            st.dataframe(macro_df, use_container_width=True, hide_index=True)
        except Exception as ex:
            st.error(f"거시 국면 분석 오류: {ex}")

    with tab_kimchi:
        st.subheader("⚡ 김치 프리미엄 & 글로벌 거래소 차익거래 분석")
        st.caption("업비트(KRW)와 바이낸스(USDT)의 동일 코인 가격 차이와 실시간 원/달러 환율을 분석합니다.")
        try:
            live_rate, rate_source = fetch_live_usd_krw_rate()
            st.info(f"적용 환율: {live_rate:,.2f} KRW/USD (출처: {rate_source})")
            sample_upbit = {"BTC": 136500000.0, "ETH": 4750000.0, "SOL": 298000.0, "XRP": 1150.0}
            sample_binance = {"BTC": 98000.0, "ETH": 3450.0, "SOL": 215.0, "XRP": 0.83}
            kp_res = compute_kimchi_premium(sample_upbit, sample_binance, usdt_krw_rate=live_rate)
            kp_df = pd.DataFrame([
                {
                    "코인": k,
                    "업비트 원화가": f"₩{v['upbit_krw']:,.0f}",
                    "글로벌 적정 원화가": f"₩{v['fair_krw']:,.0f}",
                    "프리미엄(%)": f"{v['premium_pct']:+.2f}%",
                    "격차 금액": f"₩{v['krw_difference']:+,.0f}",
                    "상태": v["status"],
                }
                for k, v in kp_res.items()
            ])
            st.dataframe(kp_df, use_container_width=True, hide_index=True)
        except Exception as ex:
            st.error(f"김치 프리미엄 분석 오류: {ex}")

    with tab_tax:
        st.subheader("💰 가상자산 세후 순수익률 & 양도소득세 시뮬레이터")
        st.caption("대한민국 가상자산 소득세법(연간 기본공제 250만 원, 22% 분리과세) 규정에 따른 실현 손익 통산 및 세후 수익률을 계산합니다.")
        try:
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                annual_profit_krw = st.number_input("연간 실현 손익 합계 (KRW)", value=12000000.0, step=1000000.0, format="%.0f")
            with col_t2:
                tax_allowance_krw = st.number_input("연간 법정 기본공제액 (KRW)", value=2500000.0, step=500000.0, format="%.0f")

            sample_trades = [annual_profit_krw * 0.7, annual_profit_krw * 0.5, -annual_profit_krw * 0.2]
            tax_res = compute_crypto_tax_impact(sample_trades, annual_allowance_krw=tax_allowance_krw, initial_capital_krw=wallet_size * 1350.0)

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("손익 통산 실현순이익", f"₩{tax_res['net_realized_profit']:,.0f}")
            t2.metric("과세 표준 (공제 후)", f"₩{tax_res['taxable_base']:,.0f}")
            t3.metric("예상 납부 세액 (22%)", f"₩{tax_res['estimated_tax_krw']:,.0f}")
            t4.metric("최종 세후 순이익", f"₩{tax_res['after_tax_profit_krw']:,.0f}")

            st.markdown("---")
            tr1, tr2, tr3 = st.columns(3)
            tr1.metric("세전 순수익률", f"{tax_res['pre_tax_return_pct']:+.2f}%")
            tr2.metric("세후 순수익률", f"{tax_res['after_tax_return_pct']:+.2f}%")
            tr3.metric("세금 잠식률 (Tax Drag)", f"-{tax_res['tax_drag_pct']:.2f}%p")
        except Exception as ex:
            st.error(f"세금 시뮬레이터 오류: {ex}")


if __name__ == "__main__":
    main()
