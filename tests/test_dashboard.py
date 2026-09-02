"""Tests for Streamlit web dashboard components."""

import unittest
import pandas as pd

try:
    import plotly.graph_objects as go
    import streamlit as st
    from app_dashboard import (
        create_pie_chart,
        create_correlation_heatmap,
        create_cumulative_return_chart,
    )
    HAS_DASHBOARD_DEPS = True
except ImportError:
    HAS_DASHBOARD_DEPS = False


class DashboardTests(unittest.TestCase):
    @unittest.skipUnless(HAS_DASHBOARD_DEPS, "plotly or streamlit not installed")
    def test_create_pie_chart(self):
        sample_weights = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}
        fig = create_pie_chart(sample_weights)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)

    @unittest.skipUnless(HAS_DASHBOARD_DEPS, "plotly or streamlit not installed")
    def test_create_correlation_heatmap(self):
        corr_df = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]],
            index=["BTC", "ETH"],
            columns=["BTC", "ETH"],
        )
        fig = create_correlation_heatmap(corr_df)
        self.assertIsInstance(fig, go.Figure)

    @unittest.skipUnless(HAS_DASHBOARD_DEPS, "plotly or streamlit not installed")
    def test_create_cumulative_return_chart(self):
        dates = pd.date_range("2026-01-01", periods=10, freq="15min")
        returns = pd.DataFrame(
            {"BTC": [0.01] * 10, "ETH": [0.02] * 10},
            index=dates,
        )
        weights = {"BTC": 0.5, "ETH": 0.5}
        fig = create_cumulative_return_chart(returns, weights)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 3)

    @unittest.skipUnless(HAS_DASHBOARD_DEPS, "plotly or streamlit not installed")
    def test_create_rebalancing_nav_chart(self):
        from app_dashboard import create_rebalancing_nav_chart
        dates = pd.date_range("2026-01-01", periods=5, freq="15min")
        s1 = pd.Series([1.0, 1.05, 1.03, 1.08, 1.10], index=dates, name="Portfolio")
        s2 = pd.Series([1.0, 1.02, 1.01, 1.04, 1.05], index=dates, name="Equal Weight")
        s3 = pd.Series([1.0, 0.98, 0.99, 1.02, 1.03], index=dates, name="Buy & Hold")
        fig = create_rebalancing_nav_chart(s1, s2, s3)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 3)


    @unittest.skipUnless(HAS_DASHBOARD_DEPS, "plotly or streamlit not installed")
    def test_create_efficient_frontier_chart(self):
        from app_dashboard import create_efficient_frontier_chart
        dates = pd.date_range("2026-01-01", periods=10, freq="15min")
        returns = pd.DataFrame(
            {"BTC": [0.01] * 10, "ETH": [0.02] * 10},
            index=dates,
        )
        weights = {"BTC": 0.6, "ETH": 0.4}
        fig = create_efficient_frontier_chart(returns, weights, num_simulations=50)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreaterEqual(len(fig.data), 3)


if __name__ == "__main__":
    unittest.main()
