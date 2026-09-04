"""Convex Optimization module."""

from skfolio.optimization.convex._base import ConvexOptimization, ObjectiveFunction
from skfolio.optimization.convex._benchmark_tracker import BenchmarkTracker
from skfolio.optimization.convex._distributionally_robust import (
    DistributionallyRobustCVaR,
)
from skfolio.optimization.convex._maximum_diversification import MaximumDiversification
from skfolio.optimization.convex._mean_risk import MeanRisk
from skfolio.optimization.convex._risk_budgeting import RiskBudgeting

# Backwards compatibility alias: MeanVariance is MeanRisk with variance risk measure
MeanVariance = MeanRisk

__all__ = [
    "BenchmarkTracker",
    "ConvexOptimization",
    "DistributionallyRobustCVaR",
    "MaximumDiversification",
    "MeanRisk",
    "MeanVariance",
    "ObjectiveFunction",
    "RiskBudgeting",
]
