"""Cryptocurrency Tail Dependence and Copula Crash Asymmetry Analyzer.

Measures nonlinear co-movement during market extremes:
- Lower Tail Dependence Coefficient (LTDC): probability of joint crashes
- Upper Tail Dependence Coefficient (UTDC): probability of joint rallies
- Tail Asymmetry: measures whether crypto assets crash together more severely than they rise together
- Copula Rank Concordance (Kendall's Tau, Spearman's Rho, Pearson's R, and Tail Dependence Gap)
- Tail Diversification Index (TDI): weights-weighted lower-tail crash diversification metric
- Extreme Tail Shock Portfolio Simulation: empirical copula joint crash loss vs Gaussian parametric VaR/ES
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure local skfolio source and scripts are discovered
root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass
class TailDependenceResult:
    """Bivariate tail dependence analysis result."""
    lower_tail: float
    upper_tail: float
    tail_asymmetry: float
    quantile: float

    def to_dict(self) -> dict[str, Any]:
        """Convert tail dependence result to serializable dictionary."""
        return {
            "lower_tail": self.lower_tail,
            "upper_tail": self.upper_tail,
            "tail_asymmetry": self.tail_asymmetry,
            "quantile": self.quantile,
        }


@dataclass
class CopulaConcordanceResult:
    """Pairwise copula rank concordance and linear-vs-nonlinear dependence metrics."""
    asset_a: str
    asset_b: str
    pearson_corr: float
    spearman_rho: float
    kendall_tau: float
    lower_tail: float
    upper_tail: float
    tail_asymmetry: float
    tail_dep_gap: float  # lower_tail - pearson_corr (excess co-crash risk over linear Gaussian model)

    def to_dict(self) -> dict[str, Any]:
        """Convert copula concordance result to serializable dictionary."""
        return {
            "asset_a": self.asset_a,
            "asset_b": self.asset_b,
            "pearson_corr": self.pearson_corr,
            "spearman_rho": self.spearman_rho,
            "kendall_tau": self.kendall_tau,
            "lower_tail": self.lower_tail,
            "upper_tail": self.upper_tail,
            "tail_asymmetry": self.tail_asymmetry,
            "tail_dep_gap": self.tail_dep_gap,
        }


def compute_bivariate_tail_dependence(
    returns_a: pd.Series | np.ndarray,
    returns_b: pd.Series | np.ndarray,
    quantile: float = 0.05,
) -> TailDependenceResult:
    """Calculate empirical lower and upper tail dependence coefficients.

    Parameters
    ----------
    returns_a : pd.Series or np.ndarray
        Returns of asset A.
    returns_b : pd.Series or np.ndarray
        Returns of asset B.
    quantile : float
        Tail probability cutoff (default 0.05 = 5% extreme tail).
    """
    r_a = np.asarray(returns_a, dtype=float)
    r_b = np.asarray(returns_b, dtype=float)
    if len(r_a) != len(r_b) or len(r_a) < 10:
        raise ValueError("Returns must have equal length and at least 10 observations.")
    if not (0.0 < quantile < 0.5):
        raise ValueError("Quantile must be strictly between 0 and 0.5.")

    valid = np.isfinite(r_a) & np.isfinite(r_b)
    r_a = r_a[valid]
    r_b = r_b[valid]
    n = len(r_a)
    if n < 10:
        raise ValueError("Insufficient valid return pairs.")

    # 1. Lower tail dependence
    q_a_low = np.percentile(r_a, quantile * 100.0)
    q_b_low = np.percentile(r_b, quantile * 100.0)
    joint_low = np.sum((r_a <= q_a_low) & (r_b <= q_b_low))
    # Empirical conditional probability: P(A <= q_a | B <= q_b)
    # Expected under independence = q * n
    denom = quantile * n
    lambda_l = float(min(1.0, max(0.0, joint_low / denom)))

    # 2. Upper tail dependence
    q_a_high = np.percentile(r_a, (1.0 - quantile) * 100.0)
    q_b_high = np.percentile(r_b, (1.0 - quantile) * 100.0)
    joint_high = np.sum((r_a > q_a_high) & (r_b > q_b_high))
    lambda_u = float(min(1.0, max(0.0, joint_high / denom)))

    # Tail asymmetry: > 0 means crash correlation exceeds boom correlation
    asymmetry = float(lambda_l - lambda_u)

    return TailDependenceResult(
        lower_tail=round(lambda_l, 4),
        upper_tail=round(lambda_u, 4),
        tail_asymmetry=round(asymmetry, 4),
        quantile=quantile,
    )


def compute_tail_dependence_matrix(
    returns_df: pd.DataFrame,
    quantile: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Compute pairwise Lower Tail and Upper Tail Dependence matrices and systemic crash scores."""
    if returns_df.empty or returns_df.shape[1] < 2:
        raise ValueError("Returns DataFrame must have at least 2 assets.")

    assets = list(returns_df.columns)
    n = len(assets)
    lower_mat = np.zeros((n, n))
    upper_mat = np.zeros((n, n))

    for i in range(n):
        lower_mat[i, i] = 1.0
        upper_mat[i, i] = 1.0
        for j in range(i + 1, n):
            res = compute_bivariate_tail_dependence(
                returns_df[assets[i]],
                returns_df[assets[j]],
                quantile=quantile,
            )
            lower_mat[i, j] = lower_mat[j, i] = res.lower_tail
            upper_mat[i, j] = upper_mat[j, i] = res.upper_tail

    df_lower = pd.DataFrame(lower_mat, index=assets, columns=assets)
    df_upper = pd.DataFrame(upper_mat, index=assets, columns=assets)

    # Average systemic crash score (mean lower tail dependence excluding self)
    systemic_scores = (df_lower.sum() - 1.0) / (n - 1)
    systemic_scores.name = "systemic_crash_vulnerability"

    return df_lower, df_upper, systemic_scores.round(4)


def compute_copula_rank_concordance(
    returns_df: pd.DataFrame,
    quantile: float = 0.05,
) -> list[CopulaConcordanceResult]:
    """Compute pairwise Pearson correlation, Spearman rank rho, Kendall tau, and Tail Gap metrics.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Historical asset returns.
    quantile : float
        Tail cutoff probability for lower and upper tail dependence.

    Returns
    -------
    list[CopulaConcordanceResult]
        Detailed pairwise concordance and crash gap metrics.
    """
    if returns_df.empty or returns_df.shape[1] < 2:
        raise ValueError("Returns DataFrame must have at least 2 assets.")
    if len(returns_df) < 10:
        raise ValueError("Insufficient return observations (must be >= 10).")

    assets = list(returns_df.columns)
    n = len(assets)
    results: list[CopulaConcordanceResult] = []

    for i in range(n):
        for j in range(i + 1, n):
            a, b = assets[i], assets[j]
            s_a = returns_df[a].astype(float)
            s_b = returns_df[b].astype(float)
            valid = s_a.notna() & s_b.notna() & np.isfinite(s_a) & np.isfinite(s_b)
            v_a = s_a[valid]
            v_b = s_b[valid]

            p_corr = float(v_a.corr(v_b, method="pearson"))
            s_rho = float(v_a.corr(v_b, method="spearman"))
            k_tau = float(v_a.corr(v_b, method="kendall"))

            td_res = compute_bivariate_tail_dependence(v_a, v_b, quantile=quantile)
            tail_gap = round(td_res.lower_tail - p_corr, 4)

            results.append(
                CopulaConcordanceResult(
                    asset_a=a,
                    asset_b=b,
                    pearson_corr=round(p_corr, 4),
                    spearman_rho=round(s_rho, 4),
                    kendall_tau=round(k_tau, 4),
                    lower_tail=td_res.lower_tail,
                    upper_tail=td_res.upper_tail,
                    tail_asymmetry=td_res.tail_asymmetry,
                    tail_dep_gap=tail_gap,
                )
            )

    return results


def compute_tail_diversification_index(
    weights: dict[str, float] | pd.Series | np.ndarray,
    lower_tail_df: pd.DataFrame,
) -> float:
    """Calculate the Tail Diversification Index (TDI) for a portfolio.

    Measures whether portfolio allocations effectively reduce pairwise lower-tail co-crash risk:
    TDI = 1.0 - (sum_{i < j} w_i * w_j * lambda_L(i, j)) / (sum_{i < j} w_i * w_j)

    A TDI close to 1.0 indicates strong tail diversification (minimal joint crash overlap).
    A TDI close to 0.0 indicates severe concentration in assets that crash simultaneously.

    Parameters
    ----------
    weights : dict, pd.Series, or np.ndarray
        Portfolio asset weights.
    lower_tail_df : pd.DataFrame
        Pairwise lower tail dependence matrix.

    Returns
    -------
    float
        Tail Diversification Index in [0.0, 1.0].
    """
    assets = list(lower_tail_df.columns)
    n = len(assets)
    if n < 2:
        return 0.0

    if isinstance(weights, dict):
        w_vec = np.array([float(weights.get(a, 0.0)) for a in assets], dtype=float)
    elif isinstance(weights, pd.Series):
        w_vec = np.array([float(weights.get(a, 0.0)) for a in assets], dtype=float)
    else:
        w_vec = np.asarray(weights, dtype=float)
        if len(w_vec) != n:
            raise ValueError(f"Weights length ({len(w_vec)}) does not match asset count ({n}).")

    w_sum = np.sum(np.abs(w_vec))
    if w_sum <= 1e-12:
        return 0.0
    w_norm = w_vec / w_sum

    total_cross_weight = 0.0
    weighted_tail = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            wij = w_norm[i] * w_norm[j]
            if wij > 0:
                total_cross_weight += wij
                weighted_tail += wij * float(lower_tail_df.iloc[i, j])

    if total_cross_weight <= 1e-12:
        return 0.0

    tdi = 1.0 - (weighted_tail / total_cross_weight)
    return float(round(np.clip(tdi, 0.0, 1.0), 4))


def simulate_tail_shock_portfolio_loss(
    weights: dict[str, float] | pd.Series,
    returns_df: pd.DataFrame,
    quantile: float = 0.05,
) -> dict[str, Any]:
    """Simulate empirical copula joint crash losses and compare against Gaussian parametric VaR/ES.

    Parameters
    ----------
    weights : dict or pd.Series
        Portfolio asset weights.
    returns_df : pd.DataFrame
        Asset returns DataFrame.
    quantile : float
        Tail probability cutoff (default 0.05 = 5%).

    Returns
    -------
    dict[str, Any]
        Empirical joint tail loss metrics, Gaussian VaR/ES benchmarks, and diversification delta.
    """
    if returns_df.empty or returns_df.shape[1] < 2:
        raise ValueError("Returns DataFrame must have at least 2 assets.")
    if not (0.0 < quantile < 0.5):
        raise ValueError("Quantile must be strictly between 0 and 0.5.")

    assets = list(returns_df.columns)
    if isinstance(weights, dict):
        w_series = pd.Series({a: float(weights.get(a, 0.0)) for a in assets})
    else:
        w_series = weights.copy()

    total_w = w_series.abs().sum()
    if total_w <= 1e-12:
        raise ValueError("Portfolio weights sum must be positive.")
    w_norm = w_series / total_w

    # Portfolio return series
    p_returns = returns_df[assets].dot(w_norm)
    n_obs = len(p_returns)

    # Asset-specific tail cutoffs
    active_assets = [a for a in assets if w_norm.get(a, 0.0) > 1e-5]
    if not active_assets:
        active_assets = assets

    # Joint tail crash mask: any active holding breaches its individual tail quantile
    joint_crash_mask = pd.Series(False, index=returns_df.index)
    for a in active_assets:
        q_val = np.percentile(returns_df[a], quantile * 100.0)
        joint_crash_mask |= (returns_df[a] <= q_val)

    joint_crash_count = int(joint_crash_mask.sum())
    joint_crash_freq_pct = round((joint_crash_count / n_obs) * 100.0, 2) if n_obs > 0 else 0.0

    # Empirical conditional tail portfolio returns
    if joint_crash_count > 0:
        tail_p_rets = p_returns[joint_crash_mask]
        cond_tail_loss_pct = round(float(-tail_p_rets.mean() * 100.0), 2)
        cond_tail_max_loss_pct = round(float(-tail_p_rets.min() * 100.0), 2)
    else:
        cond_tail_loss_pct = 0.0
        cond_tail_max_loss_pct = 0.0

    # Empirical unconditional VaR and ES
    emp_var_threshold = np.percentile(p_returns, quantile * 100.0)
    emp_var_pct = round(float(-emp_var_threshold * 100.0), 2)
    es_tail = p_returns[p_returns <= emp_var_threshold]
    emp_es_pct = round(float(-es_tail.mean() * 100.0), 2) if len(es_tail) > 0 else emp_var_pct

    # Gaussian parametric VaR and ES
    mu = float(p_returns.mean())
    sigma = float(p_returns.std(ddof=1)) if len(p_returns) > 1 else 0.0

    # Normal quantiles via approximation
    # For quantile=0.05, z_alpha ~ 1.64485
    # Rational approximation for inverse standard normal CDF
    def _inv_norm_cdf(p: float) -> float:
        # Abramowitz and Stegun approximation
        if p >= 0.5:
            return 0.0
        t = math.sqrt(-2.0 * math.log(p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return -(t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0))

    z_alpha = abs(_inv_norm_cdf(quantile))
    phi_z = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z_alpha * z_alpha)
    gauss_var_pct = round(float(-(mu - z_alpha * sigma) * 100.0), 2)
    gauss_es_pct = round(float(-(mu - (phi_z / quantile) * sigma) * 100.0), 2)

    # Diversification Delta: excess unexpected crash loss compared to Gaussian ES
    div_delta_pct = round(float(cond_tail_loss_pct - gauss_es_pct), 2)

    # Tail Diversification Index
    df_l, _, _ = compute_tail_dependence_matrix(returns_df[assets], quantile=quantile)
    tdi = compute_tail_diversification_index(w_norm, df_l)

    return {
        "quantile": quantile,
        "portfolio_weights": {a: round(float(w_norm.get(a, 0.0)), 4) for a in assets},
        "joint_crash_timesteps": joint_crash_count,
        "joint_crash_frequency_pct": joint_crash_freq_pct,
        "conditional_tail_loss_pct": cond_tail_loss_pct,
        "conditional_tail_max_loss_pct": cond_tail_max_loss_pct,
        "empirical_var_pct": emp_var_pct,
        "empirical_es_pct": emp_es_pct,
        "gaussian_var_pct": gauss_var_pct,
        "gaussian_es_pct": gauss_es_pct,
        "diversification_delta_pct": div_delta_pct,
        "tail_diversification_index": tdi,
    }


def to_dict_tail_dependence_matrix(
    df_lower: pd.DataFrame,
    df_upper: pd.DataFrame,
    scores: pd.Series,
    quantile: float = 0.05,
    concordance: list[CopulaConcordanceResult] | None = None,
    tail_stress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert tail dependence matrices, scores, concordance, and stress metrics to dictionary."""
    data: dict[str, Any] = {
        "quantile": quantile,
        "lower_tail_matrix": {str(k): {str(ik): round(float(iv), 4) for ik, iv in v.items()} for k, v in df_lower.to_dict().items()},
        "upper_tail_matrix": {str(k): {str(ik): round(float(iv), 4) for ik, iv in v.items()} for k, v in df_upper.to_dict().items()},
        "systemic_crash_vulnerability": {str(k): round(float(v), 4) for k, v in scores.to_dict().items()},
    }
    if concordance is not None:
        data["copula_concordance"] = [c.to_dict() for c in concordance]
    if tail_stress is not None:
        data["tail_stress_simulation"] = tail_stress
    return data


def export_tail_dependence_json(
    df_lower: pd.DataFrame,
    df_upper: pd.DataFrame,
    scores: pd.Series,
    filepath: str | Path,
    quantile: float = 0.05,
    concordance: list[CopulaConcordanceResult] | None = None,
    tail_stress: dict[str, Any] | None = None,
) -> None:
    """Export tail dependence results and copula concordance to a JSON file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_dict_tail_dependence_matrix(
        df_lower, df_upper, scores, quantile=quantile, concordance=concordance, tail_stress=tail_stress
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def export_tail_dependence_csv(
    df_lower: pd.DataFrame,
    df_upper: pd.DataFrame,
    scores: pd.Series,
    filepath: str | Path,
    quantile: float = 0.05,
    concordance: list[CopulaConcordanceResult] | None = None,
    tail_stress: dict[str, Any] | None = None,
) -> None:
    """Export pairwise tail dependence metrics, copula concordance, and stress results to CSV."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    assets = list(df_lower.columns)
    n = len(assets)

    # 1. Pairwise tail dependence
    for i in range(n):
        for j in range(i + 1, n):
            a, b = assets[i], assets[j]
            l_val = float(df_lower.loc[a, b])
            u_val = float(df_upper.loc[a, b])
            rows.append({
                "record_type": "pair_tail_dependence",
                "asset_a": a,
                "asset_b": b,
                "lower_tail": round(l_val, 4),
                "upper_tail": round(u_val, 4),
                "tail_asymmetry": round(l_val - u_val, 4),
                "pearson_corr": "",
                "spearman_rho": "",
                "kendall_tau": "",
                "tail_dep_gap": "",
                "systemic_vulnerability": "",
                "metric_name": "",
                "metric_value": "",
                "quantile": quantile,
            })

    # 2. Systemic vulnerability scores
    for asset, score in scores.items():
        rows.append({
            "record_type": "systemic_vulnerability",
            "asset_a": str(asset),
            "asset_b": "",
            "lower_tail": "",
            "upper_tail": "",
            "tail_asymmetry": "",
            "pearson_corr": "",
            "spearman_rho": "",
            "kendall_tau": "",
            "tail_dep_gap": "",
            "systemic_vulnerability": round(float(score), 4),
            "metric_name": "",
            "metric_value": "",
            "quantile": quantile,
        })

    # 3. Copula Concordance results
    if concordance:
        for c in concordance:
            rows.append({
                "record_type": "copula_concordance",
                "asset_a": c.asset_a,
                "asset_b": c.asset_b,
                "lower_tail": c.lower_tail,
                "upper_tail": c.upper_tail,
                "tail_asymmetry": c.tail_asymmetry,
                "pearson_corr": c.pearson_corr,
                "spearman_rho": c.spearman_rho,
                "kendall_tau": c.kendall_tau,
                "tail_dep_gap": c.tail_dep_gap,
                "systemic_vulnerability": "",
                "metric_name": "",
                "metric_value": "",
                "quantile": quantile,
            })

    # 4. Tail stress simulation summary
    if tail_stress:
        for k, v in tail_stress.items():
            if k not in ["portfolio_weights", "quantile"]:
                rows.append({
                    "record_type": "tail_stress_simulation",
                    "asset_a": "",
                    "asset_b": "",
                    "lower_tail": "",
                    "upper_tail": "",
                    "tail_asymmetry": "",
                    "pearson_corr": "",
                    "spearman_rho": "",
                    "kendall_tau": "",
                    "tail_dep_gap": "",
                    "systemic_vulnerability": "",
                    "metric_name": str(k),
                    "metric_value": str(v),
                    "quantile": quantile,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Crypto Tail Dependence and Copula Crash Analyzer.")
    parser.add_argument("--quantile", type=float, default=0.05, help="Tail quantile cutoff (default 0.05 = 5%%).")
    parser.add_argument("--weights", type=str, default=None, help="Portfolio weights in format 'BTC/USDT:0.5,ETH/USDT:0.3,SOL/USDT:0.2'.")
    parser.add_argument("--simulate-tail-shock", action="store_true", help="Simulate empirical copula joint crash portfolio loss vs Gaussian VaR/ES.")
    parser.add_argument("--export-json", type=str, default=None, help="Path to export tail dependence results to JSON file.")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to export tail dependence results to CSV file.")
    args = parser.parse_args()

    # Create synthetic demonstration
    rng = np.random.default_rng(42)
    common_shock = rng.standard_t(df=3, size=200) * 0.02
    r_btc = common_shock + rng.normal(0, 0.01, 200)
    r_eth = common_shock * 1.2 + rng.normal(0, 0.015, 200)
    r_sol = common_shock * 1.5 + rng.normal(0, 0.02, 200)
    demo_df = pd.DataFrame({"BTC/USDT": r_btc, "ETH/USDT": r_eth, "SOL/USDT": r_sol})

    df_l, df_u, scores = compute_tail_dependence_matrix(demo_df, quantile=args.quantile)
    concordance = compute_copula_rank_concordance(demo_df, quantile=args.quantile)

    print(f"[*] Lower Tail Dependence Matrix (Crash Co-movement @ {args.quantile*100:.1f}%):")
    print(df_l.round(3))
    print(f"\n[*] Upper Tail Dependence Matrix (Rally Co-movement):")
    print(df_u.round(3))
    print(f"\n[*] Systemic Crash Vulnerability Ranking:\n{scores}")

    print("\n[*] Copula Rank Concordance & Tail Gap (Linear vs Extreme Dependence):")
    for c in concordance:
        print(f"  - {c.asset_a} <-> {c.asset_b}: Pearson={c.pearson_corr:+.3f}, Kendall Tau={c.kendall_tau:+.3f}, Lower Tail={c.lower_tail:.3f}, Tail Gap={c.tail_dep_gap:+.3f}")

    # Parse portfolio weights if provided
    weights_dict: dict[str, float] = {}
    if args.weights:
        for item in args.weights.split(","):
            if ":" in item:
                k, v = item.split(":", 1)
                weights_dict[k.strip()] = float(v.strip())
    else:
        weights_dict = {"BTC/USDT": 0.4, "ETH/USDT": 0.4, "SOL/USDT": 0.2}

    tdi = compute_tail_diversification_index(weights_dict, df_l)
    print(f"\n[*] Tail Diversification Index (TDI): {tdi:.4f} (Higher is better, 1.0=Zero Co-Crash)")

    tail_stress_res = None
    if args.simulate_tail_shock or args.weights:
        tail_stress_res = simulate_tail_shock_portfolio_loss(weights_dict, demo_df, quantile=args.quantile)
        print("\n[*] Extreme Tail Shock Portfolio Simulation vs Gaussian Baseline:")
        print(f"  - Conditional Joint Tail Loss: -{tail_stress_res['conditional_tail_loss_pct']:.2f}% (Max: -{tail_stress_res['conditional_tail_max_loss_pct']:.2f}%)")
        print(f"  - Empirical ES @ {args.quantile*100:.0f}%:       -{tail_stress_res['empirical_es_pct']:.2f}% (VaR: -{tail_stress_res['empirical_var_pct']:.2f}%)")
        print(f"  - Gaussian  ES @ {args.quantile*100:.0f}%:       -{tail_stress_res['gaussian_es_pct']:.2f}% (VaR: -{tail_stress_res['gaussian_var_pct']:.2f}%)")
        print(f"  - Diversification Delta:        +{tail_stress_res['diversification_delta_pct']:.2f}% (Excess copula joint crash risk)")

    if args.export_json:
        export_tail_dependence_json(df_l, df_u, scores, args.export_json, quantile=args.quantile, concordance=concordance, tail_stress=tail_stress_res)
        print(f"[+] Tail dependence results exported to: {args.export_json}")

    if args.export_csv:
        export_tail_dependence_csv(df_l, df_u, scores, args.export_csv, quantile=args.quantile, concordance=concordance, tail_stress=tail_stress_res)
        print(f"[+] Tail dependence results exported to: {args.export_csv}")


if __name__ == "__main__":
    main()
