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
from scripts.crypto_kimchi_regime import adjust_portfolio_weights_by_kimchi, classify_kimchi_regime
from scripts.crypto_krw_fee_calculator import (
    KOREAN_EXCHANGE_PRESETS,
    compute_krw_fee_drag,
    get_korean_exchange_preset,
)
from scripts.crypto_tax_calculator import compute_crypto_tax_impact
from scripts.crypto_travel_rule_advisor import calculate_travel_rule_plan
from scripts.fetch_upbit_crypto import fetch_upbit_historical_prices
from scripts.crypto_factor_analyzer import compute_crypto_factors, generate_factor_tilted_weights
from scripts.crypto_risk_budget_calculator import (
    calculate_effective_number_of_bets,
    calculate_effective_number_of_constituents,
)
from scripts.crypto_vol_target_allocator import (
    apply_volatility_targeting,
    calculate_portfolio_realized_volatility,
    simulate_vol_targeted_backtest,
)
from scripts.crypto_tail_dependence import (
    compute_bivariate_tail_dependence,
    compute_tail_dependence_matrix,
)
from scripts.crypto_kelly_sizer import (
    calculate_continuous_kelly,
    calculate_discrete_kelly,
    calculate_portfolio_kelly,
)
from scripts.crypto_drawdown_metrics import compute_drawdown_metrics_summary

# Optional skfolio optimization imports
try:
    from skfolio import RiskMeasure
    from skfolio.optimization import (
        HierarchicalRiskParity,
        MeanRisk,
        MeanVariance,
        ObjectiveFunction,
        RiskBudgeting,
        SchurComplementary,
    )
    from skfolio.preprocessing import prices_to_returns
    HAS_SKFOLIO = True
except ImportError:
    HAS_SKFOLIO = False


@st.cache_data(show_spinner=False)
def cached_load_market_data(timeframe: str, use_synthetic: bool):
    """Cached market data loader to avoid repeated disk reads."""
    return load_market_data(timeframe=timeframe, use_synthetic=use_synthetic)


@st.cache_data(show_spinner=False, ttl=180)
def cached_load_upbit_data(markets: tuple[str, ...], count: int = 180, timeframe: str = "days") -> pd.DataFrame:
    """Cached Upbit real-time price fetcher for live KRW market optimization."""
    return fetch_upbit_historical_prices(list(markets), count=count, timeframe=timeframe)


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
    elif "Min CVaR" in model_type:
        model = MeanRisk(objective_function=ObjectiveFunction.MINIMIZE_RISK, risk_measure=RiskMeasure.CVAR, **c_kwargs)
    elif "Schur" in model_type:
        model = SchurComplementary()
    elif "HRP" in model_type:
        model = HierarchicalRiskParity(risk_measure=RiskMeasure.VARIANCE)
    else:
        model = RiskBudgeting(risk_measure=RiskMeasure.VARIANCE, **c_kwargs)

    model.fit(returns_df)
    return dict(zip(assets, model.weights_))


def create_pie_chart(weights: dict[str, float], title: str = "최적 자산 배분 비중") -> go.Figure:
    """Create an interactive Donut chart of asset weights."""
    if not isinstance(weights, dict) or not weights:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_dark")
        return fig

    labels = list(weights.keys())
    values = [round(float(v) * 100, 2) for v in weights.values()]

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
    if not isinstance(corr_df, pd.DataFrame) or corr_df.empty:
        fig = go.Figure()
        fig.update_layout(title="코인 간 상관관계 히트맵 (Correlation)", template="plotly_dark")
        return fig
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
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        fig = go.Figure()
        fig.update_layout(title="누적 수익률(Cumulative Wealth) 비교 시뮬레이션", template="plotly_dark")
        return fig

    cum_returns = (1 + returns).cumprod()
    common_assets = [c for c in returns.columns if c in weights]
    if common_assets:
        sub_returns = returns[common_assets]
        w_series = pd.Series({c: weights[c] for c in common_assets}, dtype=float)
        if w_series.sum() > 0:
            w_series = w_series / w_series.sum()
        portfolio_ret = sub_returns.dot(w_series)
    else:
        portfolio_ret = pd.Series(0.0, index=returns.index)
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
    for s, color, dash in (
        (nav_port, "#00C853", None),
        (nav_eq, "#2979FF", "dot"),
        (nav_bh, "#FF9100", "dash"),
    ):
        if isinstance(s, pd.Series) and not s.empty:
            line_kwargs = dict(color=color, width=2.5 if dash is None else 1.5)
            if dash:
                line_kwargs["dash"] = dash
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=s.name or "NAV", line=line_kwargs))

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


def create_vol_target_chart(sim_res: dict[str, object]) -> go.Figure:
    """Create comparison chart of unscaled vs volatility-targeted cumulative returns."""
    fig = go.Figure()
    if not isinstance(sim_res, dict) or "nav_unscaled" not in sim_res:
        fig.update_layout(title="변동성 타겟팅 백테스트 시뮬레이션", template="plotly_dark")
        return fig

    nav_unscaled = sim_res["nav_unscaled"]
    nav_target = sim_res["nav_vol_targeted"]
    dates = nav_unscaled.index

    fig.add_trace(go.Scatter(
        x=dates,
        y=nav_unscaled.values,
        mode="lines",
        name="기존 미조정 포트폴리오",
        line=dict(color="#FF9100", width=2, dash="dot"),
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=nav_target.values,
        mode="lines",
        name="🛡️ 변동성 타겟팅 포트폴리오",
        line=dict(color="#00E676", width=2.5),
    ))

    fig.update_layout(
        title="변동성 타겟팅 적용 전/후 누적 자산 가치(NAV) 비교",
        xaxis_title="일시",
        yaxis_title="순자산 가치 (NAV, 초기값 = 1.0)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
    )
    return fig


def create_tail_dependence_heatmap(matrix: pd.DataFrame, title: str, colorscale: str = "Reds") -> go.Figure:
    """Create interactive heatmap for lower/upper tail dependence or crash asymmetry."""
    if not isinstance(matrix, pd.DataFrame) or matrix.empty:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_dark")
        return fig

    fig = px.imshow(
        matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=colorscale,
        title=title,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_kelly_growth_chart(
    win_rate: float,
    payoff_ratio: float,
    current_fraction: float = 0.5,
) -> go.Figure:
    """Create interactive capital growth curve g(f) vs leverage fraction f."""
    if not (0.0 < win_rate < 1.0) or payoff_ratio <= 0:
        fig = go.Figure()
        fig.update_layout(title="켈리 기하성장률 곡선", template="plotly_dark")
        return fig

    f_vals = np.linspace(0.01, 0.98, 100)
    loss_rate = 1.0 - win_rate
    growth_vals = [
        float(win_rate * np.log(1.0 + payoff_ratio * f) + loss_rate * np.log(max(1e-9, 1.0 - f)))
        for f in f_vals
    ]

    full_k = (win_rate * payoff_ratio - loss_rate) / payoff_ratio
    full_k_clamped = max(0.0, min(float(full_k), 0.98))
    full_growth = (
        float(win_rate * np.log(1.0 + payoff_ratio * full_k_clamped) + loss_rate * np.log(max(1e-9, 1.0 - full_k_clamped)))
        if full_k_clamped > 0
        else 0.0
    )

    cur_k = max(0.0, min(float(full_k * current_fraction), 0.98))
    cur_growth = (
        float(win_rate * np.log(1.0 + payoff_ratio * cur_k) + loss_rate * np.log(max(1e-9, 1.0 - cur_k)))
        if cur_k > 0
        else 0.0
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f_vals * 100,
        y=np.array(growth_vals) * 100,
        mode="lines",
        name="기하 성장률 g(f)",
        line=dict(color="#29B6F6", width=2.5),
    ))

    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="손익분기(Edge=0)")

    if full_k > 0:
        fig.add_trace(go.Scatter(
            x=[full_k_clamped * 100],
            y=[full_growth * 100],
            mode="markers+text",
            marker=dict(color="#FF5252", size=11, symbol="diamond"),
            name="Full Kelly (1.0x)",
            text=["Full Kelly (최대 성장)"],
            textposition="top center",
        ))

        fig.add_trace(go.Scatter(
            x=[cur_k * 100],
            y=[cur_growth * 100],
            mode="markers+text",
            marker=dict(color="#00E676", size=11, symbol="circle"),
            name=f"선택 분할({current_fraction:.2f}x)",
            text=[f"선택 배분 ({current_fraction:.2f}x)"],
            textposition="bottom center",
        ))

    fig.update_layout(
        title="켈리 기준(Kelly Criterion) 포지션 비율별 기하 자산 성장률 곡선",
        xaxis_title="투자 비중 / 레버리지 (%)",
        yaxis_title="거래당 기하 성장률 (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
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


def create_factor_bar_chart(factors_df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart of composite factor z-scores across assets."""
    if not isinstance(factors_df, pd.DataFrame) or factors_df.empty:
        fig = go.Figure()
        fig.update_layout(title="스마트 베타 팩터 스코어", template="plotly_dark")
        return fig
    sorted_df = factors_df.sort_values("composite_score", ascending=True)
    fig = px.bar(
        sorted_df,
        x="composite_score",
        y=sorted_df.index,
        orientation="h",
        color="composite_score",
        color_continuous_scale="Viridis",
        title="자산별 스마트 베타 종합 팩터 점수 (Composite Z-Score)",
        labels={"composite_score": "종합 팩터 점수 (Z-Score)", "asset": "가상자산"},
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
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
            options=["Freqtrade 로컬 데이터", "업비트(Upbit) KRW 실시간 마켓", "합성 시뮬레이션 데이터"],
            index=0,
        )

        if data_source == "업비트(Upbit) KRW 실시간 마켓":
            upbit_markets = st.multiselect(
                "분석 대상 업비트 KRW 마켓",
                options=["KRW-BTC", "KRW-ETH", "KRW-SOL", "KRW-XRP", "KRW-ADA", "KRW-DOGE", "KRW-AVAX", "KRW-DOT"],
                default=["KRW-BTC", "KRW-ETH", "KRW-SOL", "KRW-XRP"],
            )
            upbit_timeframe = st.selectbox(
                "업비트 캔들 주기",
                options=["일봉 (1일)", "4시간봉 (240분)", "1시간봉 (60분)"],
                index=0,
            )
            timeframe = "1d"
        else:
            timeframe = st.selectbox("타임프레임 (캔들 주기)", options=["15m", "5m", "1h"], index=0)

        model_type = st.selectbox(
            "최적화 알고리즘",
            options=[
                "Risk Parity (ERC, 위험 균등)",
                "Max Sharpe Ratio (샤프 최대화)",
                "Min Variance (최소 분산)",
                "Min Semi-Variance (하방 위험 최소화)",
                "Min CVaR (조건부 가치위험 최소화)",
                "Hierarchical Risk Parity (HRP)",
                "Schur Complementary (Cotton 보완 배분)",
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
    if data_source == "업비트(Upbit) KRW 실시간 마켓":
        if not upbit_markets or len(upbit_markets) < 2:
            st.warning("포트폴리오 최적화를 위해 최소 2개 이상의 업비트 마켓을 선택해주세요.")
            st.stop()
        tf_code = "days" if "일봉" in upbit_timeframe else ("minutes/240" if "4시간봉" in upbit_timeframe else "minutes/60")
        try:
            with st.spinner("업비트(Upbit) 공개 API로부터 실시간 가격 캔들을 수신 중..."):
                prices = cached_load_upbit_data(tuple(upbit_markets), count=180, timeframe=tf_code)
                provenance = f"업비트(Upbit) KRW 마켓 실시간 데이터 ({', '.join(upbit_markets)})"
        except Exception as ex:
            st.error(f"업비트 데이터 수신 실패: {ex}")
            st.stop()
    else:
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
    tab_opt, tab_rebalance, tab_mc, tab_stress, tab_macro, tab_kimchi, tab_tax, tab_krw_fee, tab_travel, tab_factor, tab_vol_target, tab_tail, tab_kelly = st.tabs([
        "📊 포트폴리오 최적화 & 자산배분",
        "🔄 주기적 리밸런싱 백테스트",
        "🎲 몬테카를로 미래 시뮬레이션",
        "💥 역사적 블랙스완 스트레스 테스트",
        "😨 공포·탐욕 매크로 현금 조절",
        "⚡ 김치 프리미엄 차익거래",
        "💰 세후 순수익률 & 세금 시뮬레이터",
        "💸 국내 거래소 수수료 & Fee Drag",
        "🛡️ 특금법 트래블룰 안전 분할 전송",
        "🎯 퀀트 멀티팩터 & 스마트 베타",
        "🛡️ 변동성 타겟팅 & 동적 현금 버퍼",
        "📉 꼬리 위험 & 극단 붕괴 의존성",
        "🎯 켈리 기준(Kelly) 최적 포지션 사이징",
    ])

    with tab_opt:
        port_ret = returns.dot(pd.Series(weights_dict))
        mean_ret = port_ret.mean() * 100
        vol = port_ret.std() * 100
        sharpe = port_ret.mean() / (port_ret.std() + 1e-9)

        w_vec = np.array([weights_dict.get(c, 0.0) for c in returns.columns])
        enc = calculate_effective_number_of_constituents(w_vec)
        enb_res = calculate_effective_number_of_bets(w_vec, returns.cov().values)
        enb = enb_res["enb_entropy"]

        dd_metrics = compute_drawdown_metrics_summary(port_ret)
        ui = dd_metrics["ulcer_index"]
        upi = dd_metrics["martin_ratio"]

        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        col1.metric("선택 모델", model_type.split("(")[0].strip())
        col2.metric("기대 수익률", f"{mean_ret:.4f}%")
        col3.metric("변동성(위험)", f"{vol:.4f}%")
        col4.metric("샤프 지수", f"{sharpe:.3f}")
        col5.metric("얼서 지수(UI)", f"{ui:.2f}%")
        col6.metric("마틴 비율(UPI)", f"{upi:.2f}")
        col7.metric("유효 자산(ENC)", f"{enc:.2f}개")
        col8.metric("유효 베팅(ENB)", f"{enb:.2f}개")

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

        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
        with col_r1:
            train_bars = st.slider("학습 윈도우 크기 (Lookback Bars)", min_value=50, max_value=500, value=200, step=25)
        with col_r2:
            rebal_bars = st.slider("리밸런싱 주기 (Rebalance Every N Bars)", min_value=10, max_value=100, value=30, step=5)
        with col_r3:
            fee_rate = st.number_input("거래 수수료율 (Fee Rate)", min_value=0.0, max_value=0.01, value=0.001, step=0.0005, format="%.4f")
        with col_r4:
            tol_band_pct = st.slider("드리프트 허용 밴드(%)", min_value=0, max_value=20, value=0, step=1, help="오차 미만의 드리프트 발생 시 리밸런싱을 건너뛰어 수수료 절감")
            tol_band = tol_band_pct / 100.0 if tol_band_pct > 0 else None
        with col_r5:
            dd_guard_pct = st.slider("최대 낙폭 가드 (MDD Guard %)", min_value=0, max_value=30, value=0, step=5, help="고점 대비 지정 낙폭 발생 시 긴급 현금 대피")
            dd_guard = dd_guard_pct / 100.0 if dd_guard_pct > 0 else None

        if st.button("🚀 롤링 리밸런싱 백테스트 실행", key="btn_run_rebalance"):
            with st.spinner("리밸런싱 워크포워드 시뮬레이션 계산 중..."):
                try:
                    clean_model = "Equal Weight"
                    if "Risk Parity" in model_type:
                        clean_model = "Risk Parity"
                    elif "Max Sharpe" in model_type:
                        clean_model = "Max Sharpe"
                    elif "Min Semi-Variance" in model_type:
                        clean_model = "Min Semi-Variance"
                    elif "Min CVaR" in model_type:
                        clean_model = "Min CVaR"
                    elif "Schur" in model_type:
                        clean_model = "Schur"
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
                        tolerance_band=tol_band,
                        drawdown_guard=dd_guard,
                    )
                    s = reb_res["summary"]

                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("총 수익률 (전략)", f"{s['Total Return (%)']:.2f}%", f"{s['Total Return (%)'] - s['Buy & Hold Return (%)']:+.2f}% vs B&H")
                    rc2.metric("최대 낙폭 (MDD)", f"{s['Max Drawdown (%)']:.2f}%", f"{s['Max Drawdown (%)'] - s['Buy & Hold MDD (%)']:+.2f}% vs B&H")
                    rc3.metric("연환산 샤프 지수", f"{s['Sharpe Ratio (Ann.)']:.3f}")
                    turnover_label = f"{s['Average Turnover (%)']:.2f}%"
                    if s.get("Skipped Rebalances", 0) > 0:
                        turnover_label += f" ({s['Skipped Rebalances']}회 스킵)"
                    if s.get("Guard Triggers", 0) > 0:
                        turnover_label += f" [가드 {s['Guard Triggers']}회]"
                    rc4.metric("평균 회전율 (Turnover)", turnover_label)

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
        st.caption("기하 브라운 운동(GBM) 및 Student-t 팻테일 기반으로 미래 자산 경로를 시뮬레이션하여 95% 신뢰구간과 극단 손실 위험(VaR/CVaR)을 산출합니다.")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            mc_days = st.slider("미래 시뮬레이션 기간 (Days)", min_value=30, max_value=365, value=90, step=15)
        with col_m2:
            mc_sims = st.slider("시뮬레이션 경로 수 (Paths)", min_value=200, max_value=2000, value=1000, step=100)
        with col_m3:
            dist_opt = st.selectbox("확률 분포 모델", options=["정규분포 (표준 GBM)", "Student-t (팻테일/극단 충격 반영)"], index=0)
            mc_dist = "student_t" if "Student-t" in dist_opt else "normal"

        if st.button("🚀 몬테카를로 시뮬레이션 실행", key="btn_run_mc"):
            with st.spinner("몬테카를로 경로 시뮬레이션 계산 중..."):
                try:
                    mc_res = simulate_monte_carlo(
                        prices,
                        weights_dict,
                        initial_capital=wallet_size,
                        days=mc_days,
                        num_simulations=mc_sims,
                        distribution=mc_dist,
                    )
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
        st.caption("2020년 코로나 쇼크, 2022년 루나 붕괴, FTX 파산, 2024년 지정학적 쇼크 및 디파이 유동성 캐스케이드 등 실제 역사적 극단 위기 상황을 현재 포트폴리오에 주입하여 자본 방어력과 원금 회복 필요 수익률을 진단합니다.")
        try:
            stress_res = evaluate_stress_test(weights_dict, total_wallet=wallet_size)
            stress_df = pd.DataFrame([
                {
                    "역사적 블랙스완 시나리오": k,
                    "포트폴리오 손실률": f"{v['portfolio_loss_pct']:+.2f}%",
                    "예상 손실액": f"-${v['dollar_loss']:,.2f}",
                    "충격 후 잔여 자산": f"${v['remaining_balance']:,.2f}",
                    "원금 회복 필요 수익률": f"+{v.get('recovery_required_pct', 0.0):.1f}%",
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

            # Kimchi Premium Regime Tactical Allocation
            st.markdown("---")
            st.subheader("🎯 김치 프리미엄 기반 동적 자산배분 레짐")
            btc_prem = float(kp_res.get("BTC", {}).get("premium_pct", 3.0))
            regime_res = classify_kimchi_regime(btc_prem)

            k1, k2, k3 = st.columns(3)
            k1.metric("김프 시장 레짐", str(regime_res["regime"]))
            k2.metric("권장 크립토 비중", f"{float(regime_res['target_crypto_ratio'])*100:.1f}%")
            k3.metric("권장 안전자산(현금) 비중", f"{float(regime_res['target_cash_ratio'])*100:.1f}%")
            st.info(f"💡 전술 행동 지침: {regime_res['tactical_action']}")

            kimchi_weights = adjust_portfolio_weights_by_kimchi(weights_dict, premium_pct=btc_prem, cash_asset="KRW (Cash)")
            kimchi_df = pd.DataFrame({
                "자산 / 현금 버퍼": list(kimchi_weights.keys()),
                "전술적 배분 비중": [f"{w*100:.2f}%" for w in kimchi_weights.values()],
                "배분 금액": [f"${w * wallet_size:,.2f}" for w in kimchi_weights.values()],
            })
            st.dataframe(kimchi_df, use_container_width=True, hide_index=True)
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

    with tab_krw_fee:
        st.subheader("💸 국내 가상자산 거래소 수수료 & 포트폴리오 잠식률(Fee Drag)")
        st.caption("업비트, 빗썸, 코인원, 코빗 등 국내 원화마켓 수수료율과 리밸런싱 회전율에 따른 연간 수익률 갉아먹힘(Fee Drag)을 정밀 산출합니다.")
        try:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                selected_exchange_key = st.selectbox(
                    "기준 국내 거래소",
                    options=list(KOREAN_EXCHANGE_PRESETS.keys()),
                    index=0,
                    format_func=lambda k: str(KOREAN_EXCHANGE_PRESETS[k]["name"]),
                )
                preset = get_korean_exchange_preset(selected_exchange_key)
            with fc2:
                annual_turnover = st.slider(
                    "연간 포트폴리오 회전율 (Turnover)",
                    min_value=0.5,
                    max_value=24.0,
                    value=4.0,
                    step=0.5,
                    help="연간 총 매매대금 / 포트폴리오 원금. (예: 분기별 100% 교체 시 4.0, 월 1회 리밸런싱 시 12.0)",
                )
            with fc3:
                maker_pct = st.slider(
                    "지정가(Maker) 체결 비율 (%)",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=5,
                )

            current_portfolio_krw = float(wallet_size * 1350.0)
            maker_fee_val = float(preset["maker_fee"])
            taker_fee_val = float(preset["taker_fee"])
            maker_ratio_val = float(maker_pct) / 100.0

            fee_res = compute_krw_fee_drag(
                portfolio_value_krw=current_portfolio_krw,
                annual_turnover=float(annual_turnover),
                maker_fee=maker_fee_val,
                taker_fee=taker_fee_val,
                maker_ratio=maker_ratio_val,
                annual_withdrawals=12,
                withdrawal_fee_krw=float(preset["withdrawal_fee_krw"]),
            )

            f1, f2, f3, f4 = st.columns(4)
            f1.metric("연간 총 거래대금", f"₩{fee_res['annual_trade_volume_krw']:,.0f}")
            f2.metric("연간 총 지출 수수료", f"₩{fee_res['total_annual_fees_krw']:,.0f}")
            f3.metric("포트폴리오 잠식률 (Fee Drag)", f"-{fee_res['fee_drag_pct']:.4f}%p")
            f4.metric("유효 거래비용 (Effective)", f"{fee_res['effective_cost_bps']:.2f} bps")

            st.markdown("---")
            st.subheader("📊 5대 국내 원화거래소 수수료 잠식률 비교 (동일 회전율 기준)")

            comparison_rows = []
            for ex_key, ex_info in KOREAN_EXCHANGE_PRESETS.items():
                res_comp = compute_krw_fee_drag(
                    portfolio_value_krw=current_portfolio_krw,
                    annual_turnover=float(annual_turnover),
                    maker_fee=float(ex_info["maker_fee"]),
                    taker_fee=float(ex_info["taker_fee"]),
                    maker_ratio=maker_ratio_val,
                    annual_withdrawals=12,
                    withdrawal_fee_krw=float(ex_info["withdrawal_fee_krw"]),
                )
                comparison_rows.append({
                    "거래소": ex_info["name"],
                    "기본 수수료율": f"{float(ex_info['taker_fee'])*100:.2f}%",
                    "연간 지출 수수료": f"₩{res_comp['total_annual_fees_krw']:,.0f}",
                    "Fee Drag": f"-{res_comp['fee_drag_pct']:.4f}%p",
                    "손익분기 초과수익률": f"{res_comp['breakeven_gross_hurdle_pct']:.4f}%",
                })

            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
        except Exception as ex:
            st.error(f"수수료 잠식률 시뮬레이터 오류: {ex}")

    with tab_travel:
        st.subheader("🛡️ 특금법 100만 원 트래블룰 안전 분할 전송 어드바이저")
        st.caption("대한민국 특정금융정보법(100만 원 이상 출금 시 VASP 간 트래블룰 정보 교환 의무)에 맞춰, 미연동 해외거래소 및 개인 콜드월렛 전송 시 입출금 정지나 동결 리스크를 방지하는 최적 분할 전송 계획을 수립합니다.")
        try:
            coin_presets = {
                "XRP (리플)": {"symbol": "XRP", "price": 1150.0, "amount": 2500.0, "fee": 1.0},
                "TRX (트론)": {"symbol": "TRX", "price": 280.0, "fee": 1.0, "amount": 10000.0},
                "SOL (솔라나)": {"symbol": "SOL", "price": 298000.0, "fee": 0.01, "amount": 10.0},
                "BTC (비트코인)": {"symbol": "BTC", "price": 136500000.0, "fee": 0.0005, "amount": 0.03},
                "ETH (이더리움)": {"symbol": "ETH", "price": 4750000.0, "fee": 0.005, "amount": 0.8},
                "USDT (테더)": {"symbol": "USDT", "price": 1380.0, "fee": 1.0, "amount": 2000.0},
            }

            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1:
                selected_coin_name = st.selectbox("전송 코인", list(coin_presets.keys()), index=0)
                preset_data = coin_presets[selected_coin_name]
            with tc2:
                target_amount = st.number_input(
                    f"총 전송 희망 수량 ({preset_data['symbol']})",
                    value=float(preset_data["amount"]),
                    step=1.0 if preset_data["price"] < 10000 else 0.01,
                    format="%.4f" if preset_data["price"] >= 10000 else "%.1f",
                )
            with tc3:
                coin_price = st.number_input(
                    "현재 코인 단가 (KRW)",
                    value=float(preset_data["price"]),
                    step=10.0 if preset_data["price"] < 10000 else 10000.0,
                    format="%.0f",
                )
            with tc4:
                net_fee = st.number_input(
                    f"건당 출금 수수료 ({preset_data['symbol']})",
                    value=float(preset_data["fee"]),
                    step=0.1 if preset_data["price"] < 10000 else 0.0001,
                    format="%.4f",
                )

            safe_buf = st.slider(
                "변동성 대비 회당 안전 한도 (KRW, 법정 기준 100만 원 미만)",
                min_value=800000,
                max_value=980000,
                value=950000,
                step=10000,
                help="코인 가격의 실시간 급등락으로 전송 도중 원화 환산 100만 원을 초과하여 입금이 묶이는 사고를 방지하기 위한 안전 마진입니다.",
            )

            plan_res = calculate_travel_rule_plan(
                coin_symbol=preset_data["symbol"],
                target_amount=float(target_amount),
                coin_price_krw=float(coin_price),
                safe_buffer_krw=float(safe_buf),
                network_fee_coins=float(net_fee),
            )

            if plan_res["requires_travel_rule"]:
                st.warning(f"⚠️ 총 전송액이 ₩{plan_res['total_value_krw']:,.0f}으로 법정 트래블룰 기준(100만 원)을 초과합니다. VASP 연동 거래소가 아니거나 개인 지갑 전송 시 분할 전송이 권장됩니다.")
            else:
                st.success(f"✅ 총 전송액이 ₩{plan_res['total_value_krw']:,.0f}으로 100만 원 미만입니다. 간이 출금이 적용되어 안전하게 즉시 전송 가능합니다.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("총 전송 원화 환산액", f"₩{plan_res['total_value_krw']:,.0f}")
            m2.metric("권장 분할 전송 횟수", f"{plan_res['recommended_batches']} 회")
            m3.metric("1회당 안전 전송 수량", f"{plan_res['per_batch_coins']:.4f} {preset_data['symbol']}")
            m4.metric("총 네트워크 수수료", f"₩{plan_res['total_network_fee_krw']:,.0f} ({plan_res['total_network_fee_coins']:.4f} {preset_data['symbol']})")

            st.markdown("---")
            st.subheader("📋 권장 단계별 안전 전송 스케줄")

            batches = plan_res["recommended_batches"]
            schedule_data = []
            for b in range(1, batches + 1):
                schedule_data.append({
                    "전송 회차": f"{b} / {batches} 회차",
                    "전송 수량": f"{plan_res['per_batch_coins']:.4f} {preset_data['symbol']}",
                    "회차별 원화가": f"₩{plan_res['per_batch_krw']:,.0f}",
                    "권장 대기 시간": "이전 회차 입금 완전 확인 후 최소 10~15분 대기" if b > 1 else "첫 회 소액 테스트 권장",
                    "상태": "안전 (Safety Buffer 이내)",
                })
            st.dataframe(pd.DataFrame(schedule_data), use_container_width=True, hide_index=True)
            st.info(f"💡 컴플라이언스 가이드: {plan_res['compliance_advice']}")
        except Exception as ex:
            st.error(f"트래블룰 어드바이저 오류: {ex}")

    with tab_factor:
        st.subheader("🎯 퀀트 멀티팩터 & 스마트 베타 (Smart Beta Screener)")
        st.caption("모멘텀, 저변동성, 추세 강도, 소르티노(하방 위험) 4대 퀀트 팩터를 결합하여 최적 자산군을 스크리닝합니다.")

        max_lookback = max(30, min(180, len(prices)))
        min_lookback = min(30, max_lookback)
        factor_lookback = st.slider(
            "팩터 산출 룩백 기간 (Lookback Bars)",
            min_value=min_lookback,
            max_value=max_lookback,
            value=min(60, max_lookback),
            step=10,
        )

        with st.expander("⚖️ 팩터 가중치 사용자 정의 (Custom Factor Weights)", expanded=False):
            fw_c1, fw_c2, fw_c3, fw_c4 = st.columns(4)
            with fw_c1:
                w_mom = st.slider("모멘텀 가중치(%)", 0, 100, 30, 5)
            with fw_c2:
                w_vol = st.slider("저변동성 가중치(%)", 0, 100, 25, 5)
            with fw_c3:
                w_trend = st.slider("추세강도 가중치(%)", 0, 100, 25, 5)
            with fw_c4:
                w_sortino = st.slider("소르티노(하방) 가중치(%)", 0, 100, 20, 5)

        custom_weights = {
            "momentum": float(w_mom),
            "low_volatility": float(w_vol),
            "trend_strength": float(w_trend),
            "sortino_ratio": float(w_sortino),
        }
        if sum(custom_weights.values()) == 0:
            custom_weights = None

        if st.button("🚀 멀티팩터 스마트 베타 분석 실행", key="btn_run_factors"):
            try:
                with st.spinner("멀티팩터 점수 산출 중..."):
                    factors_df = compute_crypto_factors(
                        prices=prices,
                        lookback_bars=factor_lookback,
                        factor_weights=custom_weights,
                    )

                    col_fb1, col_fb2 = st.columns([3, 2])
                    with col_fb1:
                        fig_factor = create_factor_bar_chart(factors_df)
                        st.plotly_chart(fig_factor, use_container_width=True)

                    with col_fb2:
                        st.subheader("🏆 스마트 베타 팩터 순위표")
                        display_df = pd.DataFrame({
                            "자산": list(factors_df.index),
                            "종합 점수": [f"{v:+.2f}" for v in factors_df["composite_score"]],
                            "모멘텀(Z)": [f"{v:+.2f}" for v in factors_df["z_momentum"]],
                            "저변동성(Z)": [f"{v:+.2f}" for v in factors_df["z_low_vol"]],
                            "추세(Z)": [f"{v:+.2f}" for v in factors_df["z_trend"]],
                            "소르티노(Z)": [f"{v:+.2f}" for v in factors_df["z_sortino"]],
                        })
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

                    st.markdown("---")
                    top_coin = factors_df.index[0]
                    top_score = factors_df["composite_score"].iloc[0]
                    st.success(f"🌟 현재 스마트 베타 최고 순위 자산: **{top_coin}** (종합 Z-Score: {top_score:+.2f})")

                    # Factor-tilted portfolio allocation
                    st.subheader("💼 Top-N 스마트 베타 팩터 틸트(Tilt) 포트폴리오 비중")
                    tilted_w = generate_factor_tilted_weights(factors_df, top_n=min(3, len(factors_df)), weighting="score_weighted")
                    tilt_table = pd.DataFrame({
                        "자산": list(tilted_w.keys()),
                        "팩터 틸트 비중": [f"{w*100:.2f}%" for w in tilted_w.values()],
                        "배분 금액": [f"${w * wallet_size:,.2f}" for w in tilted_w.values()],
                    })
                    st.dataframe(tilt_table[tilt_table["배분 금액"] != "$0.00"], use_container_width=True, hide_index=True)
            except Exception as ex:
                st.error(f"팩터 분석 오류: {ex}")

    with tab_vol_target:
        st.subheader("🛡️ 목표 변동성 타겟팅 & 동적 현금 버퍼 (Volatility Targeting)")
        st.caption("포트폴리오의 실현 변동성에 반비례하여 위험 자산 비중을 조절하고, 급락/고변동성 장세에서 현금(USDT) 버퍼로 안전하게 대피합니다.")

        realized_vol = calculate_portfolio_realized_volatility(returns, weights_dict) * 100.0

        vt_col1, vt_col2, vt_col3 = st.columns(3)
        with vt_col1:
            target_vol = st.slider("목표 연환산 변동성 (%)", min_value=10.0, max_value=120.0, value=min(40.0, max(20.0, realized_vol)), step=5.0)
        with vt_col2:
            min_cash = st.slider("최소 현금 버퍼 비율 (%)", min_value=0.0, max_value=50.0, value=10.0, step=5.0)
        with vt_col3:
            max_lev = st.slider("최대 허용 레버리지 (배)", min_value=1.0, max_value=2.0, value=1.0, step=0.1)

        vt_res = apply_volatility_targeting(
            base_weights=weights_dict,
            realized_vol_ann=realized_vol / 100.0,
            target_vol_ann=target_vol / 100.0,
            max_leverage=max_lev,
            min_cash_buffer=min_cash / 100.0,
        )

        m_v1, m_v2, m_v3, m_v4 = st.columns(4)
        m_v1.metric("현재 포트폴리오 실현 변동성", f"{realized_vol:.1f}%")
        m_v2.metric("목표 변동성 스케일러 (k)", f"{vt_res.vol_scalar:.2f}x")
        m_v3.metric("동적 현금/스테이블코인 비중", f"{vt_res.cash_weight * 100:.1f}%")
        m_v4.metric("위험 자산 총 비중", f"{(1.0 - vt_res.cash_weight) * 100:.1f}%")

        if vt_res.cash_weight > min_cash / 100.0:
            st.warning(f"⚠️ 시장 변동성이 목표({target_vol:.1f}%)보다 높아 자산 비중을 축소하고 현금 버퍼를 {vt_res.cash_weight*100:.1f}%로 확대했습니다. (De-Risking 발동)")
        else:
            st.success(f"✅ 시장 변동성이 안정적입니다. 기본 자산 배분을 유지합니다.")

        st.markdown("---")
        col_vt_chart, col_vt_tbl = st.columns([3, 2])

        with col_vt_chart:
            chart_weights = dict(vt_res.scaled_weights)
            if vt_res.cash_weight > 0:
                chart_weights["💵 Cash (USDT)"] = vt_res.cash_weight
            fig_vt_pie = create_pie_chart(chart_weights, title="변동성 타겟팅 적용 후 최종 자산 비중")
            st.plotly_chart(fig_vt_pie, use_container_width=True)

        with col_vt_tbl:
            st.subheader("📋 변동성 조절 후 자산별 배분 금액")
            vt_table_rows = []
            for a, w in vt_res.scaled_weights.items():
                vt_table_rows.append({
                    "자산": a,
                    "기존 비중": f"{weights_dict.get(a, 0.0)*100:.2f}%",
                    "조정 비중": f"{w*100:.2f}%",
                    "배분 금액": f"${w * wallet_size:,.2f}",
                })
            if vt_res.cash_weight > 0:
                vt_table_rows.append({
                    "자산": "💵 Cash (USDT)",
                    "기존 비중": "0.00%",
                    "조정 비중": f"{vt_res.cash_weight*100:.2f}%",
                    "배분 금액": f"${vt_res.cash_weight * wallet_size:,.2f}",
                })
            st.dataframe(pd.DataFrame(vt_table_rows), use_container_width=True, hide_index=True)

        if st.button("📈 변동성 타겟팅 백테스트 시뮬레이션 실행", key="btn_run_vt_backtest"):
            try:
                with st.spinner("시뮬레이션 실행 중..."):
                    sim_vt = simulate_vol_targeted_backtest(
                        returns_df=returns,
                        base_weights=weights_dict,
                        target_vol_ann=target_vol / 100.0,
                        rolling_window=min(60, max(10, len(returns) // 2)),
                        max_leverage=max_lev,
                        min_cash_buffer=min_cash / 100.0,
                    )
                    fig_vt_sim = create_vol_target_chart(sim_vt)
                    st.plotly_chart(fig_vt_sim, use_container_width=True)

                    s_un = sim_vt["summary_unscaled"]
                    s_vt = sim_vt["summary_vol_targeted"]
                    st.info(f"📊 백테스트 결과: 기존 수익률 {s_un['total_return_pct']:.2f}% (MDD {s_un['max_drawdown_pct']:.2f}%) ➔ 타겟팅 적용 후 수익률 {s_vt['total_return_pct']:.2f}% (MDD {s_vt['max_drawdown_pct']:.2f}%, 평균 현금 비중 {s_vt['avg_cash_buffer_pct']:.1f}%)")
            except Exception as ex:
                st.error(f"변동성 타겟팅 백테스트 오류: {ex}")

    with tab_tail:
        st.subheader("📉 극단 꼬리 위험 & 붕괴 비대칭 행렬 (Empirical Tail Dependence)")
        st.caption("비선형 극단 상황(Flash Crash)에서의 코인 간 동반 폭락 확률(하방 꼬리 의존성, LTDC)과 동반 급등 확률(상방 꼬리 의존성, UTDC)을 측정합니다.")

        q_val = st.slider("극단 꼬리 분위수 기준 (Tail Quantile Cutoff)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

        try:
            lower_m, upper_m, asym_m = compute_tail_dependence_matrix(returns, quantile=q_val)

            tail_tab1, tail_tab2, tail_tab3 = st.tabs([
                "🚨 하방 꼬리 폭락 의존성 (LTDC)",
                "🚀 상방 꼬리 급등 의존성 (UTDC)",
                "⚖️ 붕괴 비대칭 지수 (Crash Asymmetry)",
            ])

            with tail_tab1:
                st.caption("하방 꼬리 의존성(LTDC): 한 코인이 하위 분위수 이하로 폭락할 때 다른 코인도 동반 폭락할 조건부 결합 확률")
                fig_lower = create_tail_dependence_heatmap(lower_m, f"하방 극단 꼬리 폭락 의존성 행렬 (q = {q_val*100:.0f}%)", colorscale="Reds")
                st.plotly_chart(fig_lower, use_container_width=True)

            with tail_tab2:
                st.caption("상방 꼬리 의존성(UTDC): 한 코인이 상위 분위수 이상으로 급등할 때 다른 코인도 동반 급등할 조건부 결합 확률")
                fig_upper = create_tail_dependence_heatmap(upper_m, f"상방 극단 꼬리 급등 의존성 행렬 (q = {q_val*100:.0f}%)", colorscale="Greens")
                st.plotly_chart(fig_upper, use_container_width=True)

            with tail_tab3:
                st.caption("붕괴 비대칭 지수(Crash Asymmetry = LTDC - UTDC): 양수(+)일수록 상승장보다 폭락장에서 동반 하락 위험이 훨씬 높음을 의미합니다.")
                fig_asym = create_tail_dependence_heatmap(asym_m, "극단 붕괴 비대칭 지수 (Asymmetry = LTDC - UTDC)", colorscale="RdBu_r")
                st.plotly_chart(fig_asym, use_container_width=True)

            # Highlight pair with highest crash vulnerability
            upper_tri = np.triu(np.ones(lower_m.shape), k=1).astype(bool)
            lower_m_masked = lower_m.where(upper_tri)
            max_pair_idx = lower_m_masked.stack().idxmax()
            if isinstance(max_pair_idx, tuple) and len(max_pair_idx) == 2:
                max_val = lower_m.loc[max_pair_idx[0], max_pair_idx[1]]
                st.warning(f"⚠️ **최고 동반 폭락 위험 페어**: `{max_pair_idx[0]}` ↔ `{max_pair_idx[1]}` (동반 폭락 확률: {max_val*100:.1f}%). 두 자산을 동시에 과다 편입 시 분산투자 효과가 급락할 수 있습니다.")
        except Exception as ex:
            st.error(f"꼬리 의존성 분석 오류: {ex}")

    with tab_kelly:
        st.subheader("🎯 켈리 기준(Kelly Criterion) 최적 포지션 사이징")
        st.caption("기하 급수적 자산 증식을 위한 최적 베팅 비율을 산출합니다. 암호화폐의 팻 테일(Fat-tail) 위험을 고려하여 통상 Half-Kelly(0.5x) 또는 Fractional Kelly가 권장됩니다.")

        kelly_mode = st.radio(
            "사이징 모드 선택",
            options=["포트폴리오 다중 자산 연속 켈리(Multi-Asset Continuous Kelly)", "단일 전략 이산 켈리(Discrete Trade-level Kelly)"],
            horizontal=True,
        )

        if "다중 자산" in kelly_mode:
            k_frac = st.slider("켈리 분할 배율 (Fractional Kelly Multiplier)", min_value=0.1, max_value=1.0, value=0.5, step=0.05, help="1.0은 Full Kelly(최대 성장, 극심한 변동성), 0.5는 Half Kelly(성장률 75% 유지 및 변동성 50% 감축)")
            max_tot = st.slider("총 투자 비중 상한(Max Total Leverage)", min_value=0.2, max_value=1.0, value=1.0, step=0.05)

            try:
                k_weights = calculate_portfolio_kelly(returns, fraction=k_frac, max_total_weight=max_tot)

                col_k1, col_k2 = st.columns([3, 2])
                with col_k1:
                    fig_k = px.bar(
                        x=k_weights.index,
                        y=k_weights.values * 100,
                        labels={"x": "자산", "y": "켈리 권장 비중 (%)"},
                        title=f"다중 자산 켈리 최적 비중 ({k_frac:.2f}x Kelly)",
                        text_auto=".1f",
                    )
                    fig_k.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_k, use_container_width=True)

                with col_k2:
                    st.write("##### 📋 켈리 비중 vs 현재 모델 비중")
                    comp_df = pd.DataFrame({
                        "자산": k_weights.index,
                        "켈리 비중": [f"{w*100:.2f}%" for w in k_weights.values],
                        "현재 모델": [f"{weights_dict.get(a, 0)*100:.2f}%" for a in k_weights.index],
                        "켈리 배분액": [f"{w * wallet_size:,.2f}" for w in k_weights.values],
                    })
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                    st.info(f"💡 총 켈리 익스포저: **{k_weights.sum()*100:.1f}%** (현금 버퍼: **{max(0.0, 1.0 - k_weights.sum())*100:.1f}%**)")
            except Exception as ex:
                st.error(f"다중 자산 켈리 계산 오류: {ex}")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                win_rate = st.slider("전략 승률 (Win Rate %)", min_value=10.0, max_value=90.0, value=55.0, step=1.0) / 100.0
            with c2:
                payoff = st.number_input("손익비 (Payoff Ratio, 평균수익/평균손실)", min_value=0.1, max_value=10.0, value=1.8, step=0.1)
            with c3:
                frac = st.slider("적용 켈리 분할 배율", min_value=0.1, max_value=1.0, value=0.5, step=0.05)

            try:
                k_res = calculate_discrete_kelly(win_rate, payoff, fraction=frac)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("풀 켈리(f*)", f"{k_res.full_kelly*100:.1f}%")
                m2.metric("하프 켈리(0.5x)", f"{k_res.half_kelly*100:.1f}%")
                m3.metric(f"선택 비중({frac:.2f}x)", f"{k_res.fractional_kelly*100:.1f}%")
                m4.metric("거래당 기하 성장률", f"{k_res.expected_growth_rate*100:.3f}%")

                fig_growth = create_kelly_growth_chart(win_rate, payoff, current_fraction=frac)
                st.plotly_chart(fig_growth, use_container_width=True)

                if not k_res.is_positive_edge:
                    st.error("⚠️ 해당 승률 및 손익비 조건에서는 수학적 엣지(Edge)가 없어 베팅 시 원금 손실이 발생합니다. 진입 금지를 권장합니다.")
                else:
                    st.success(f"✅ 전략 엣지가 유효합니다. 권장 단일 진입 비중은 총 자산의 **{k_res.fractional_kelly*100:.1f}%** 입니다.")
            except Exception as ex:
                st.error(f"이산 켈리 계산 오류: {ex}")


if __name__ == "__main__":
    main()


