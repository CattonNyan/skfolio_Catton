"""Meucci (2008) Fully Flexible Views & Entropy Pooling Portfolio Optimizer for Crypto Assets.

Implements Attilio Meucci's Entropy Pooling framework ("Fully Flexible Views: Theory and Practice", 2008).
Allows quantitative crypto portfolio managers to condition an empirical or Monte Carlo prior distribution
with subjective, quantitative, or on-chain views (e.g. absolute returns, cross-asset rankings, volatility targets,
and tail-risk probabilities) via Kullback-Leibler (KL) relative entropy minimization:

    min_{p~} sum_{s=1}^S p~_s * ln(p~_s / p_s)
    s.t.  A_eq * p~ = b_eq
          A_ineq * p~ <= b_ineq
          sum_{s=1}^S p~_s = 1,  p~_s >= 0

Generates posterior expected returns, posterior covariance matrix, Effective Number of Scenarios (ENS),
and solves for optimal posterior allocations (Maximum Sharpe, Minimum Variance, Equal Weight), benchmarked
via walk-forward multi-period backtesting.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

# Ensure local repository root is discoverable
root_dir = str(Path(__file__).resolve().parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class EntropyView:
    """Quantitative or model-driven view for Entropy Pooling conditioning."""

    view_type: str  # "absolute_return", "relative_return", "volatility", "tail_risk"
    assets: list[str]  # e.g. ["BTC"] or ["SOL", "ETH"]
    operator: str  # "=", ">=", "<="
    target_value: float  # Annualized return/vol or probability
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


@dataclass
class EntropyPoolingResult:
    """Complete summary of prior vs conditioned posterior entropy pooling state."""

    kl_divergence: float
    effective_scenarios_prior: float
    effective_scenarios_posterior: float
    entropy_reduction_pct: float
    confidence: float
    prior_expected_returns_ann: dict[str, float]
    posterior_expected_returns_ann: dict[str, float]
    prior_volatilities_ann: dict[str, float]
    posterior_volatilities_ann: dict[str, float]
    weights_prior_max_sharpe: dict[str, float]
    weights_posterior_max_sharpe: dict[str, float]
    weights_prior_min_var: dict[str, float]
    weights_posterior_min_var: dict[str, float]
    weights_equal_weight: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


def build_prior_probabilities(n_scenarios: int, decay_lambda: float = 1.0) -> np.ndarray:
    """Build prior scenario probability vector (uniform or exponential decay).

    Parameters:
    -----------
    n_scenarios : int
        Total number of historical or simulated scenarios S.
    decay_lambda : float
        Exponential decay factor in (0.0, 1.0]. 1.0 yields uniform prior 1/S.
        Values < 1.0 put higher weights on recent scenarios.

    Returns:
    --------
    p_prior : np.ndarray
        Array of shape (S,) summing to 1.0.
    """
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be a positive integer.")
    if not (0.0 < decay_lambda <= 1.0):
        raise ValueError("decay_lambda must be in (0.0, 1.0].")

    if math.isclose(decay_lambda, 1.0):
        return np.ones(n_scenarios, dtype=float) / n_scenarios

    # Exponential recency weighting: lambda^(S - 1 - s)
    powers = np.arange(n_scenarios - 1, -1, -1, dtype=float)
    weights = decay_lambda ** powers
    return weights / np.sum(weights)


def calculate_effective_number_of_scenarios(p: np.ndarray) -> float:
    """Calculate Meucci's Effective Number of Scenarios (ENS) based on exponential entropy.

    ENS(p) = exp( - sum_{s=1}^S p_s * ln(p_s) )
    Ranges from 1 (single scenario concentration) to S (uniform distribution).
    """
    clean_p = np.clip(p, 1e-12, 1.0)
    clean_p = clean_p / np.sum(clean_p)
    entropy = -float(np.sum(clean_p * np.log(clean_p)))
    return float(np.exp(entropy))


def calculate_kl_divergence(p_post: np.ndarray, p_prior: np.ndarray) -> float:
    """Compute Kullback-Leibler relative entropy D_KL(p_post || p_prior).

    D_KL = sum_{s=1}^S p_post_s * ln(p_post_s / p_prior_s)
    """
    p_pst = np.clip(p_post, 1e-12, 1.0)
    p_pst = p_pst / np.sum(p_pst)
    p_pri = np.clip(p_prior, 1e-12, 1.0)
    p_pri = p_pri / np.sum(p_pri)
    return float(np.sum(p_pst * np.log(p_pst / p_pri)))


def build_view_linear_constraints(
    returns_df: pd.DataFrame,
    views: list[EntropyView],
    p_prior: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Convert quantitative views into linear constraint matrices A_eq, b_eq, A_ineq, b_ineq.

    Scenario return matrix X has shape (S, N).
    Views are expressed as:
        A_eq @ p_post = b_eq
        A_ineq @ p_post <= b_ineq
    """
    S, N = returns_df.shape
    symbols = list(returns_df.columns)
    sym_idx = {sym: i for i, sym in enumerate(symbols)}

    A_eq_list: list[np.ndarray] = []
    b_eq_list: list[float] = []
    A_ineq_list: list[np.ndarray] = []
    b_ineq_list: list[float] = []

    X = returns_df.values  # (S, N)

    for view in views:
        for sym in view.assets:
            if sym not in sym_idx:
                raise ValueError(f"Asset '{sym}' specified in view is not present in returns columns: {symbols}")

        a_row = np.zeros(S, dtype=float)
        target = view.target_value

        if view.view_type == "absolute_return":
            # Annualized expected return target -> daily expected return = target / 365.0
            idx = sym_idx[view.assets[0]]
            a_row = X[:, idx].copy()
            rhs = target / 365.0

        elif view.view_type == "relative_return":
            # Relative spread: E[R_asset0 - R_asset1]
            if len(view.assets) != 2:
                raise ValueError("relative_return view requires exactly 2 assets [asset_long, asset_short].")
            idx0 = sym_idx[view.assets[0]]
            idx1 = sym_idx[view.assets[1]]
            a_row = (X[:, idx0] - X[:, idx1]).copy()
            rhs = target / 365.0

        elif view.view_type == "volatility":
            # Second moment constraint: E[R^2] = (target_vol / sqrt(365))^2 + (prior_mu)^2
            idx = sym_idx[view.assets[0]]
            prior_mu = float(np.sum(p_prior * X[:, idx]))
            daily_var = (target / math.sqrt(365.0)) ** 2
            a_row = (X[:, idx] ** 2).copy()
            rhs = daily_var + (prior_mu ** 2)

        elif view.view_type == "tail_risk":
            # Tail risk: P(R_i <= -5%) <= alpha
            # target_value represents max probability of loss exceeding threshold (-0.05 default if not custom)
            idx = sym_idx[view.assets[0]]
            # 5% daily drop indicator
            a_row = (X[:, idx] <= -0.05).astype(float)
            rhs = float(target)

        else:
            raise ValueError(f"Unsupported view_type: {view.view_type}")

        if view.operator in ("=", "=="):
            A_eq_list.append(a_row)
            b_eq_list.append(rhs)
        elif view.operator in ("<=", "<"):
            A_ineq_list.append(a_row)
            b_ineq_list.append(rhs)
        elif view.operator in (">=", ">"):
            # a @ p >= rhs  <=>  -a @ p <= -rhs
            A_ineq_list.append(-a_row)
            b_ineq_list.append(-rhs)
        else:
            raise ValueError(f"Unsupported operator '{view.operator}' in view: {view}")

    A_eq = np.array(A_eq_list, dtype=float) if A_eq_list else None
    b_eq = np.array(b_eq_list, dtype=float) if b_eq_list else None
    A_ineq = np.array(A_ineq_list, dtype=float) if A_ineq_list else None
    b_ineq = np.array(b_ineq_list, dtype=float) if b_ineq_list else None

    return A_eq, b_eq, A_ineq, b_ineq


def compute_entropy_pooling(
    p_prior: np.ndarray,
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    A_ineq: np.ndarray | None = None,
    b_ineq: np.ndarray | None = None,
    confidence: float = 1.0,
    tol: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Solve the Kullback-Leibler relative entropy minimization problem under view constraints.

    min_{p} sum_{s=1}^S p_s * ln(p_s / p_prior_s)
    s.t. sum(p) == 1, p_s >= 1e-8
         A_eq @ p == b_eq
         A_ineq @ p <= b_ineq

    Blends with prior according to confidence in [0.0, 1.0]:
        p_blended = confidence * p_posterior + (1 - confidence) * p_prior

    Returns:
    --------
    p_blended : np.ndarray
        Posterior probability vector over S scenarios.
    kl_divergence : float
        Kullback-Leibler distance from prior.
    """
    S = len(p_prior)
    if S == 0:
        raise ValueError("p_prior cannot be empty.")
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if math.isclose(confidence, 0.0) or (A_eq is None and A_ineq is None):
        return p_prior.copy(), 0.0

    K_eq = len(b_eq) if (A_eq is not None and b_eq is not None) else 0
    K_ineq = len(b_ineq) if (A_ineq is not None and b_ineq is not None) else 0

    if K_eq == 0 and K_ineq == 0:
        return p_prior.copy(), 0.0

    # Ensure 2D working arrays
    mat_eq = np.array(A_eq, dtype=float).copy() if K_eq > 0 else np.empty((0, S))
    vec_eq = np.array(b_eq, dtype=float).copy() if K_eq > 0 else np.empty(0)
    mat_ineq = np.array(A_ineq, dtype=float).copy() if K_ineq > 0 else np.empty((0, S))
    vec_ineq = np.array(b_ineq, dtype=float).copy() if K_ineq > 0 else np.empty(0)

    # Scale constraints for optimal numerical conditioning
    if K_eq > 0:
        for k in range(K_eq):
            norm_k = float(np.linalg.norm(mat_eq[k]))
            if norm_k > 1e-8:
                mat_eq[k] /= norm_k
                vec_eq[k] /= norm_k

    if K_ineq > 0:
        for k in range(K_ineq):
            norm_k = float(np.linalg.norm(mat_ineq[k]))
            if norm_k > 1e-8:
                mat_ineq[k] /= norm_k
                vec_ineq[k] /= norm_k

    def dual_obj_and_grad(y: np.ndarray) -> tuple[float, np.ndarray]:
        lam = y[:K_eq]
        nu = y[K_eq:]
        exponent = np.zeros(S, dtype=float)
        if K_eq > 0:
            exponent -= mat_eq.T @ lam
        if K_ineq > 0:
            exponent -= mat_ineq.T @ nu

        max_e = np.max(exponent)
        w = p_prior * np.exp(exponent - max_e)
        sum_w = float(np.sum(w))
        if sum_w < 1e-12:
            sum_w = 1e-12
        p = w / sum_w

        val = np.log(sum_w) + max_e
        grad_parts = []
        if K_eq > 0:
            val += float(lam @ vec_eq)
            grad_lam = vec_eq - mat_eq @ p
            grad_parts.append(grad_lam)
        if K_ineq > 0:
            val += float(nu @ vec_ineq)
            grad_nu = vec_ineq - mat_ineq @ p
            grad_parts.append(grad_nu)

        grad = np.concatenate(grad_parts) if grad_parts else np.zeros(0)
        return float(val), grad

    bounds = [(None, None)] * K_eq + [(0.0, None)] * K_ineq
    y0 = np.zeros(K_eq + K_ineq)

    res = minimize(
        dual_obj_and_grad,
        y0,
        jac=True,
        bounds=bounds,
        method="L-BFGS-B",
        options={"ftol": tol, "maxiter": 500},
    )

    lam_opt = res.x[:K_eq]
    nu_opt = res.x[K_eq:]
    exponent = np.zeros(S, dtype=float)
    if K_eq > 0:
        exponent -= mat_eq.T @ lam_opt
    if K_ineq > 0:
        exponent -= mat_ineq.T @ nu_opt

    p_raw = p_prior * np.exp(exponent - np.max(exponent))
    p_opt = p_raw / np.sum(p_raw)

    # Blend with confidence
    p_blended = confidence * p_opt + (1.0 - confidence) * p_prior
    p_blended = p_blended / np.sum(p_blended)

    kl_div = calculate_kl_divergence(p_blended, p_prior)
    return p_blended, kl_div


def compute_scenario_moments(
    returns_df: pd.DataFrame,
    p: np.ndarray,
    shrinkage: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute expected returns and regularized covariance matrix under probability vector p.

    Parameters:
    -----------
    returns_df : pd.DataFrame
        (S, N) asset return matrix.
    p : np.ndarray
        (S,) scenario probability vector.
    shrinkage : float
        Ledoit-Wolf-style diagonal shrinkage intensity in [0.0, 1.0].

    Returns:
    --------
    mu : np.ndarray
        (N,) expected returns.
    cov : np.ndarray
        (N, N) positive-definite covariance matrix.
    """
    X = returns_df.values  # (S, N)
    p_col = p.reshape(-1, 1)  # (S, 1)

    # Expected returns: mu = X^T @ p
    mu = np.sum(X * p_col, axis=0)  # (N,)

    # Centered returns
    X_cent = X - mu  # (S, N)

    # Covariance: Sigma = sum_{s=1}^S p_s * (X_s - mu)(X_s - mu)^T
    cov = (X_cent.T * p) @ X_cent  # (N, N)

    # Shrinkage regularization to guarantee well-conditioned positive definiteness
    if shrinkage > 0.0:
        diag_target = np.diag(np.diag(cov))
        cov = (1.0 - shrinkage) * cov + shrinkage * diag_target

    # Add small numerical jitter to eigenvalues if needed
    min_eig = np.min(np.real(np.linalg.eigvals(cov)))
    if min_eig < 1e-7:
        cov += (1e-7 - min_eig + 1e-8) * np.eye(cov.shape[0])

    return mu, cov


def optimize_portfolio_weights(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    objective: str = "max_sharpe",
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """Solve long-only portfolio optimization under given distribution moments.

    Objectives:
    - "max_sharpe": Maximize Sharpe ratio (mu - rf) / sqrt(w^T Sigma w)
    - "min_var": Minimize variance w^T Sigma w
    - "equal_weight": 1 / N
    """
    N = len(expected_returns)
    if N == 0:
        return np.array([])
    if objective == "equal_weight":
        return np.ones(N) / N

    w0 = np.ones(N) / N
    bounds = [(0.0, 1.0) for _ in range(N)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if objective == "min_var":
        def min_var_obj(w: np.ndarray) -> float:
            return float(w.T @ cov_matrix @ w)

        res = minimize(min_var_obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            w = np.clip(res.x, 0.0, 1.0)
            return w / np.sum(w)
        return w0

    # Max Sharpe
    rf_daily = risk_free_rate / 365.0

    def neg_sharpe_obj(w: np.ndarray) -> float:
        port_ret = float(np.sum(w * expected_returns)) - rf_daily
        port_vol = float(np.sqrt(np.maximum(1e-12, w.T @ cov_matrix @ w)))
        return -port_ret / port_vol

    res = minimize(neg_sharpe_obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        w = np.clip(res.x, 0.0, 1.0)
        return w / np.sum(w)
    return w0


def run_entropy_pooling_analysis(
    returns_df: pd.DataFrame,
    views: list[EntropyView],
    confidence: float = 0.8,
    decay_lambda: float = 1.0,
    shrinkage: float = 0.05,
) -> EntropyPoolingResult:
    """Execute complete Meucci Entropy Pooling pipeline on asset return matrix."""
    symbols = list(returns_df.columns)
    S, N = returns_df.shape

    # 1. Prior distribution
    p_prior = build_prior_probabilities(S, decay_lambda=decay_lambda)
    ens_prior = calculate_effective_number_of_scenarios(p_prior)

    # 2. Linear view constraints
    A_eq, b_eq, A_ineq, b_ineq = build_view_linear_constraints(returns_df, views, p_prior)

    # 3. Kullback-Leibler Relative Entropy Minimization
    p_post, kl_div = compute_entropy_pooling(
        p_prior, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, confidence=confidence
    )
    ens_post = calculate_effective_number_of_scenarios(p_post)
    entropy_reduc_pct = ((ens_prior - ens_post) / ens_prior) * 100.0 if ens_prior > 0 else 0.0

    # 4. Moments: Prior vs Posterior
    mu_prior, cov_prior = compute_scenario_moments(returns_df, p_prior, shrinkage=shrinkage)
    mu_post, cov_post = compute_scenario_moments(returns_df, p_post, shrinkage=shrinkage)

    # Annualized metrics (365 days for crypto)
    mu_prior_ann = {sym: round(float(mu_prior[i] * 365.0), 4) for i, sym in enumerate(symbols)}
    mu_post_ann = {sym: round(float(mu_post[i] * 365.0), 4) for i, sym in enumerate(symbols)}

    vol_prior_ann = {
        sym: round(float(np.sqrt(np.maximum(1e-12, cov_prior[i, i])) * math.sqrt(365.0)), 4)
        for i, sym in enumerate(symbols)
    }
    vol_post_ann = {
        sym: round(float(np.sqrt(np.maximum(1e-12, cov_post[i, i])) * math.sqrt(365.0)), 4)
        for i, sym in enumerate(symbols)
    }

    # 5. Optimal portfolio weights
    w_prior_sharpe = optimize_portfolio_weights(mu_prior, cov_prior, objective="max_sharpe")
    w_post_sharpe = optimize_portfolio_weights(mu_post, cov_post, objective="max_sharpe")
    w_prior_minvar = optimize_portfolio_weights(mu_prior, cov_prior, objective="min_var")
    w_post_minvar = optimize_portfolio_weights(mu_post, cov_post, objective="min_var")
    w_ew = optimize_portfolio_weights(mu_prior, cov_prior, objective="equal_weight")

    return EntropyPoolingResult(
        kl_divergence=round(kl_div, 6),
        effective_scenarios_prior=round(ens_prior, 2),
        effective_scenarios_posterior=round(ens_post, 2),
        entropy_reduction_pct=round(entropy_reduc_pct, 2),
        confidence=round(confidence, 2),
        prior_expected_returns_ann=mu_prior_ann,
        posterior_expected_returns_ann=mu_post_ann,
        prior_volatilities_ann=vol_prior_ann,
        posterior_volatilities_ann=vol_post_ann,
        weights_prior_max_sharpe={sym: round(float(w_prior_sharpe[i]), 4) for i, sym in enumerate(symbols)},
        weights_posterior_max_sharpe={sym: round(float(w_post_sharpe[i]), 4) for i, sym in enumerate(symbols)},
        weights_prior_min_var={sym: round(float(w_prior_minvar[i]), 4) for i, sym in enumerate(symbols)},
        weights_posterior_min_var={sym: round(float(w_post_minvar[i]), 4) for i, sym in enumerate(symbols)},
        weights_equal_weight={sym: round(float(w_ew[i]), 4) for i, sym in enumerate(symbols)},
    )


def simulate_entropy_pooling_walk_forward(
    returns_df: pd.DataFrame,
    views: list[EntropyView],
    lookback: int = 120,
    rebalance_freq: int = 15,
    confidence: float = 0.8,
    fee_rate: float = 0.001,
    initial_capital: float = 10000.0,
) -> dict[str, Any]:
    """Walk-forward backtest benchmarking Entropy-Pooled vs Historical Prior vs Equal-Weight portfolios.

    Evaluates:
    - Net CAGR
    - Annualized Volatility
    - Sharpe Ratio
    - Sortino Ratio
    - Maximum Drawdown (MDD)
    - Total Turnover & Transaction Fee Drag
    """
    T, N = returns_df.shape
    symbols = list(returns_df.columns)

    if T <= lookback + rebalance_freq:
        raise ValueError(f"Insufficient history length ({T}) for lookback {lookback} + freq {rebalance_freq}.")

    eval_returns = returns_df.iloc[lookback:].copy()
    eval_dates = list(eval_returns.index)
    n_eval_bars = len(eval_dates)

    policies = ["Entropy_Pooled_Sharpe", "Prior_Max_Sharpe", "Equal_Weight"]
    equity_curves: dict[str, list[float]] = {p: [initial_capital] for p in policies}
    current_weights: dict[str, np.ndarray] = {p: np.ones(N) / N for p in policies}
    turnovers: dict[str, float] = {p: 0.0 for p in policies}
    fee_drags: dict[str, float] = {p: 0.0 for p in policies}

    for t in range(n_eval_bars):
        hist_idx = lookback + t

        # Rebalance check
        if t % rebalance_freq == 0:
            window_df = returns_df.iloc[hist_idx - lookback : hist_idx]
            ep_res = run_entropy_pooling_analysis(
                window_df, views=views, confidence=confidence, shrinkage=0.05
            )

            # Target weights
            target_weights = {
                "Entropy_Pooled_Sharpe": np.array([ep_res.weights_posterior_max_sharpe[s] for s in symbols]),
                "Prior_Max_Sharpe": np.array([ep_res.weights_prior_max_sharpe[s] for s in symbols]),
                "Equal_Weight": np.ones(N) / N,
            }

            for p in policies:
                w_target = target_weights[p]
                w_curr = current_weights[p]
                turnover = float(np.sum(np.abs(w_target - w_curr)))
                turnovers[p] += turnover
                fee = equity_curves[p][-1] * turnover * fee_rate
                fee_drags[p] += fee
                equity_curves[p][-1] -= fee
                current_weights[p] = w_target.copy()

        # Daily performance step
        bar_ret = returns_df.iloc[hist_idx].values  # (N,)
        for p in policies:
            w = current_weights[p]
            step_ret = float(np.sum(w * bar_ret))
            new_eq = equity_curves[p][-1] * (1.0 + step_ret)
            equity_curves[p].append(new_eq)

            # Drift weights by asset performance
            unnorm_w = w * (1.0 + bar_ret)
            sum_w = np.sum(unnorm_w)
            current_weights[p] = (unnorm_w / sum_w) if sum_w > 1e-8 else (np.ones(N) / N)

    # Calculate summary metrics per policy
    results: dict[str, Any] = {}
    ann_factor = 365.0
    days = n_eval_bars

    for p in policies:
        eq = np.array(equity_curves[p])
        daily_ret = np.diff(eq) / eq[:-1]
        tot_ret = (eq[-1] / eq[0]) - 1.0
        cagr = ((eq[-1] / eq[0]) ** (ann_factor / days) - 1.0) if days > 0 and eq[-1] > 0 else 0.0
        vol = float(np.std(daily_ret, ddof=1) * math.sqrt(ann_factor)) if len(daily_ret) > 1 else 0.0
        sharpe = (float(np.mean(daily_ret) * ann_factor) / vol) if vol > 1e-6 else 0.0

        downside = daily_ret[daily_ret < 0]
        downside_vol = float(np.std(downside, ddof=1) * math.sqrt(ann_factor)) if len(downside) > 1 else 0.0
        sortino = (float(np.mean(daily_ret) * ann_factor) / downside_vol) if downside_vol > 1e-6 else 0.0

        # Maximum Drawdown
        peaks = np.maximum.accumulate(eq)
        drawdowns = (eq - peaks) / peaks
        mdd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        results[p] = {
            "final_equity": round(float(eq[-1]), 2),
            "total_return_pct": round(tot_ret * 100.0, 2),
            "cagr_pct": round(cagr * 100.0, 2),
            "annualized_volatility_pct": round(vol * 100.0, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown_pct": round(mdd * 100.0, 2),
            "total_turnover": round(turnovers[p], 2),
            "fee_drag_usd": round(fee_drags[p], 2),
        }

    return results


def generate_synthetic_crypto_scenarios(
    n_bars: int = 365,
    seed: int = 42,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Generate multi-asset synthetic crypto prices exhibiting realistic correlations and fat-tailed shocks."""
    rng = np.random.default_rng(seed)
    if not symbols:
        symbols = ["BTC", "ETH", "SOL", "AVAX"]
    N = len(symbols)

    default_vols = [0.45, 0.60, 0.75, 0.70, 0.80, 0.65]
    base_vols = np.array([default_vols[i % len(default_vols)] for i in range(N)]) / math.sqrt(365.0)

    if N == 4:
        corr = np.array([
            [1.0, 0.75, 0.65, 0.60],
            [0.75, 1.0, 0.70, 0.65],
            [0.65, 0.70, 1.0, 0.75],
            [0.60, 0.65, 0.75, 1.0],
        ])
    else:
        corr = np.full((N, N), 0.65)
        np.fill_diagonal(corr, 1.0)

    chol = np.linalg.cholesky(corr)
    daily_shocks = rng.standard_t(df=4.0, size=(n_bars, N))
    correlated_shocks = daily_shocks @ chol.T

    # Scale with vol and drift
    default_drifts = [0.20, 0.15, 0.35, 0.10, 0.25, 0.18]
    daily_drift = np.array([default_drifts[i % len(default_drifts)] for i in range(N)]) / 365.0
    daily_returns = daily_drift + correlated_shocks * base_vols

    default_prices = [65000.0, 3200.0, 150.0, 28.0, 600.0, 1.2]
    start_prices = np.array([default_prices[i % len(default_prices)] for i in range(N)])
    price_paths = start_prices * np.cumprod(1.0 + daily_returns, axis=0)

    dates = pd.date_range("2026-01-01", periods=n_bars, freq="D")
    return pd.DataFrame(price_paths, index=dates, columns=symbols)


def export_entropy_pooling_json(
    result: EntropyPoolingResult,
    backtest_results: dict[str, Any] | None = None,
    filepath: str | Path | None = None,
) -> str:
    """Export entropy pooling analysis to JSON format (and optionally save to filepath)."""
    payload = {
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "entropy_pooling": result.to_dict(),
        "walk_forward_backtest": backtest_results or {},
    }
    json_str = json.dumps(payload, indent=2, ensure_ascii=False)
    if filepath:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json_str, encoding="utf-8")
    return json_str


def export_entropy_pooling_csv(
    result: EntropyPoolingResult,
    filepath: str | Path | None = None,
) -> str:
    """Export asset moment changes and portfolio weights to CSV format (and optionally save to filepath)."""
    symbols = list(result.prior_expected_returns_ann.keys())
    lines = [
        "symbol,prior_return_ann_pct,posterior_return_ann_pct,prior_vol_ann_pct,posterior_vol_ann_pct,prior_max_sharpe_wt,posterior_max_sharpe_wt,prior_min_var_wt,posterior_min_var_wt,equal_wt"
    ]
    for sym in symbols:
        lines.append(
            f"{sym},"
            f"{result.prior_expected_returns_ann[sym] * 100.0:.2f},"
            f"{result.posterior_expected_returns_ann[sym] * 100.0:.2f},"
            f"{result.prior_volatilities_ann[sym] * 100.0:.2f},"
            f"{result.posterior_volatilities_ann[sym] * 100.0:.2f},"
            f"{result.weights_prior_max_sharpe[sym]:.4f},"
            f"{result.weights_posterior_max_sharpe[sym]:.4f},"
            f"{result.weights_prior_min_var[sym]:.4f},"
            f"{result.weights_posterior_min_var[sym]:.4f},"
            f"{result.weights_equal_weight[sym]:.4f}"
        )
    csv_str = "\n".join(lines)
    if filepath:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(csv_str + "\n", encoding="utf-8")
    return csv_str


def main() -> None:
    """CLI execution entrypoint for Meucci Entropy Pooling optimizer."""
    parser = argparse.ArgumentParser(
        description="Meucci (2008) Fully Flexible Views & Entropy Pooling Portfolio Optimizer for Crypto Assets"
    )
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL", "AVAX"], help="Crypto asset symbols")
    parser.add_argument("--confidence", type=float, default=0.8, help="Confidence in views [0.0, 1.0]")
    parser.add_argument("--decay-lambda", type=float, default=1.0, help="Prior exponential decay lambda (1.0 = uniform)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial simulation capital USD")
    parser.add_argument("--example", action="store_true", help="Run demonstration with synthetic crypto market paths")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format to stdout")
    parser.add_argument("--csv", action="store_true", help="Output asset comparison table in CSV format to stdout")
    parser.add_argument("--export-json", type=str, default="", help="Export full results to JSON file")
    parser.add_argument("--export-csv", type=str, default="", help="Export asset comparison table to CSV file")

    args = parser.parse_args()
    symbols = list(args.symbols)

    # Generate synthetic price paths matching requested symbols
    prices_df = generate_synthetic_crypto_scenarios(n_bars=365, seed=42, symbols=symbols)
    returns_df = prices_df.pct_change().dropna()

    # Build views dynamically matching available symbols
    sample_views: list[EntropyView] = []
    if "SOL" in symbols and "ETH" in symbols:
        sample_views.append(
            EntropyView(
                view_type="relative_return",
                assets=["SOL", "ETH"],
                operator=">=",
                target_value=0.15,
                description="SOL outperformance over ETH >= 15% p.a.",
            )
        )
    elif len(symbols) >= 2:
        sample_views.append(
            EntropyView(
                view_type="relative_return",
                assets=[symbols[-1], symbols[1]],
                operator=">=",
                target_value=0.15,
                description=f"{symbols[-1]} outperformance over {symbols[1]} >= 15% p.a.",
            )
        )

    first_sym = "BTC" if "BTC" in symbols else symbols[0]
    sample_views.append(
        EntropyView(
            view_type="absolute_return",
            assets=[first_sym],
            operator=">=",
            target_value=0.30,
            description=f"{first_sym} annualized return >= 30%",
        )
    )

    last_sym = "AVAX" if "AVAX" in symbols else symbols[-1]
    sample_views.append(
        EntropyView(
            view_type="volatility",
            assets=[last_sym],
            operator="<=",
            target_value=0.50,
            description=f"{last_sym} annualized volatility <= 50%",
        )
    )

    ep_res = run_entropy_pooling_analysis(
        returns_df,
        views=sample_views,
        confidence=args.confidence,
        decay_lambda=args.decay_lambda,
    )

    bt_res = simulate_entropy_pooling_walk_forward(
        returns_df,
        views=sample_views,
        lookback=120,
        rebalance_freq=15,
        confidence=args.confidence,
        initial_capital=args.capital,
    )

    if args.export_json:
        export_entropy_pooling_json(ep_res, bt_res, filepath=args.export_json)
    if args.export_csv:
        export_entropy_pooling_csv(ep_res, filepath=args.export_csv)

    if args.json:
        print(export_entropy_pooling_json(ep_res, bt_res))
    elif args.csv:
        print(export_entropy_pooling_csv(ep_res))
    else:
        print("=" * 78)
        print("MEUCCI (2008) FULLY FLEXIBLE VIEWS & ENTROPY POOLING PORTFOLIO OPTIMIZER")
        print("=" * 78)
        print(f"Number of Scenarios:          {len(returns_df)} bars")
        print(f"Prior Effective Scenarios:    {ep_res.effective_scenarios_prior:.1f}")
        print(f"Posterior Effective Scenarios:{ep_res.effective_scenarios_posterior:.1f}")
        print(f"Entropy Reduction (Entropy %):{ep_res.entropy_reduction_pct:.2f}%")
        print(f"Kullback-Leibler Divergence:  {ep_res.kl_divergence:.6f}")
        print(f"View Confidence:              {ep_res.confidence * 100.0:.0f}%")
        print("-" * 78)
        print("ASSET MOMENT SHIFTS (PRIOR vs CONDITIONED POSTERIOR):")
        for sym in symbols:
            pri_ret = ep_res.prior_expected_returns_ann[sym] * 100.0
            pst_ret = ep_res.posterior_expected_returns_ann[sym] * 100.0
            pri_vol = ep_res.prior_volatilities_ann[sym] * 100.0
            pst_vol = ep_res.posterior_volatilities_ann[sym] * 100.0
            print(f"  {sym:10s} | Ret: {pri_ret:+6.1f}% -> {pst_ret:+6.1f}% | Vol: {pri_vol:5.1f}% -> {pst_vol:5.1f}%")

        print("-" * 78)
        print("MAX SHARPE WEIGHT REALLOCATION:")
        for sym in symbols:
            w_pri = ep_res.weights_prior_max_sharpe[sym] * 100.0
            w_pst = ep_res.weights_posterior_max_sharpe[sym] * 100.0
            print(f"  {sym:10s} | Prior: {w_pri:5.1f}% -> Posterior: {w_pst:5.1f}%")

        print("-" * 78)
        print("WALK-FORWARD BACKTEST BENCHMARK (120-Day Lookback, 15-Day Rebalance):")
        for policy, m in bt_res.items():
            print(
                f"  [{policy:22s}] Return: {m['total_return_pct']:+6.2f}% | "
                f"CAGR: {m['cagr_pct']:+5.1f}% | Sharpe: {m['sharpe_ratio']:4.2f} | "
                f"MDD: {m['max_drawdown_pct']:5.1f}% | Fee: ${m['fee_drag_usd']:.1f}"
            )
        print("=" * 78)
        if args.export_json:
            print(f"[+] Exported JSON report to: {args.export_json}")
        if args.export_csv:
            print(f"[+] Exported CSV report to: {args.export_csv}")


if __name__ == "__main__":
    main()
