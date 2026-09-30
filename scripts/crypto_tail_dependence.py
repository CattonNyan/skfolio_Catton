"""Cryptocurrency Tail Dependence and Copula Crash Asymmetry Analyzer.

Measures nonlinear co-movement during market extremes:
- Lower Tail Dependence Coefficient (LTDC): probability of joint crashes
- Upper Tail Dependence Coefficient (UTDC): probability of joint rallies
- Tail Asymmetry: measures whether crypto assets crash together more severely than they rise together
"""

from __future__ import annotations

import argparse
import json
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


def to_dict_tail_dependence_matrix(
    df_lower: pd.DataFrame,
    df_upper: pd.DataFrame,
    scores: pd.Series,
    quantile: float = 0.05,
) -> dict[str, Any]:
    """Convert tail dependence matrices and systemic vulnerability scores to serializable dictionary."""
    return {
        "quantile": quantile,
        "lower_tail_matrix": {str(k): {str(ik): round(float(iv), 4) for ik, iv in v.items()} for k, v in df_lower.to_dict().items()},
        "upper_tail_matrix": {str(k): {str(ik): round(float(iv), 4) for ik, iv in v.items()} for k, v in df_upper.to_dict().items()},
        "systemic_crash_vulnerability": {str(k): round(float(v), 4) for k, v in scores.to_dict().items()},
    }


def export_tail_dependence_json(
    df_lower: pd.DataFrame,
    df_upper: pd.DataFrame,
    scores: pd.Series,
    filepath: str | Path,
    quantile: float = 0.05,
) -> None:
    """Export tail dependence results to a JSON file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = to_dict_tail_dependence_matrix(df_lower, df_upper, scores, quantile=quantile)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def export_tail_dependence_csv(
    df_lower: pd.DataFrame,
    df_upper: pd.DataFrame,
    scores: pd.Series,
    filepath: str | Path,
    quantile: float = 0.05,
) -> None:
    """Export pairwise tail dependence metrics and systemic vulnerability to a CSV file."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    assets = list(df_lower.columns)
    n = len(assets)
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
                "systemic_vulnerability": "",
                "quantile": quantile,
            })
    for asset, score in scores.items():
        rows.append({
            "record_type": "systemic_vulnerability",
            "asset_a": str(asset),
            "asset_b": "",
            "lower_tail": "",
            "upper_tail": "",
            "tail_asymmetry": "",
            "systemic_vulnerability": round(float(score), 4),
            "quantile": quantile,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Crypto Tail Dependence Analyzer.")
    parser.add_argument("--quantile", type=float, default=0.05, help="Tail quantile cutoff (default 0.05 = 5%%).")
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
    print(f"[*] Lower Tail Dependence Matrix (Crash Co-movement @ {args.quantile*100:.1f}%):")
    print(df_l.round(3))
    print(f"\n[*] Upper Tail Dependence Matrix (Rally Co-movement):")
    print(df_u.round(3))
    print(f"\n[*] Systemic Crash Vulnerability Ranking:\n{scores}")

    if args.export_json:
        export_tail_dependence_json(df_l, df_u, scores, args.export_json, quantile=args.quantile)
        print(f"[+] Tail dependence results exported to: {args.export_json}")

    if args.export_csv:
        export_tail_dependence_csv(df_l, df_u, scores, args.export_csv, quantile=args.quantile)
        print(f"[+] Tail dependence results exported to: {args.export_csv}")


if __name__ == "__main__":
    main()
