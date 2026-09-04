"""Tests for Black-Litterman optimization engine."""

import unittest
import pandas as pd
import numpy as np

from scripts.crypto_portfolio_optimizer import generate_synthetic_crypto_data
from scripts.crypto_black_litterman import compute_black_litterman_weights


class BlackLittermanTests(unittest.TestCase):
    def test_black_litterman_without_views(self):
        prices = generate_synthetic_crypto_data(periods=100)
        res = compute_black_litterman_weights(prices, views=[])
        prior_w = res["prior_weights"]
        post_w = res["posterior_weights"]
        self.assertEqual(prior_w, post_w)
        self.assertAlmostEqual(sum(post_w.values()), 1.0, places=4)

    def test_relative_view_shifts_weights(self):
        prices = generate_synthetic_crypto_data(periods=100)
        res = compute_black_litterman_weights(
            prices=prices,
            views=["BTC/USDT>ETH/USDT:0.05"],
            tau=0.05,
        )
        post_w = res["posterior_weights"]
        self.assertIn("BTC/USDT", post_w)
        self.assertIn("ETH/USDT", post_w)
        self.assertAlmostEqual(sum(post_w.values()), 1.0, places=4)
        # BTC weight should be higher than ETH weight due to the bullish relative view
        self.assertGreater(post_w["BTC/USDT"], post_w["ETH/USDT"])

    def test_absolute_view_execution(self):
        prices = generate_synthetic_crypto_data(periods=100)
        res = compute_black_litterman_weights(
            prices=prices,
            views=["SOL/USDT:0.15"],
            tau=0.05,
        )
        post_w = res["posterior_weights"]
        self.assertIn("SOL/USDT", post_w)
        self.assertAlmostEqual(sum(post_w.values()), 1.0, places=4)
        self.assertFalse(res["fallback_to_prior"])

    def test_extreme_bearish_views_fallback_flag(self):
        prices = generate_synthetic_crypto_data(periods=50)
        # Apply extreme negative view on all assets
        views = [f"{col}:-0.99" for col in prices.columns]
        res = compute_black_litterman_weights(prices, views=views)
        self.assertIn("fallback_to_prior", res)
        # Weights should sum to 1.0 and be non-negative
        self.assertAlmostEqual(sum(res["posterior_weights"].values()), 1.0, places=4)

    def test_custom_prior_weights(self):
        prices = generate_synthetic_crypto_data(periods=100)
        custom_prior = {"BTC/USDT": 0.70, "ETH/USDT": 0.30}
        res = compute_black_litterman_weights(prices, views=[], prior_weights=custom_prior)
        prior_w = res["prior_weights"]
        self.assertAlmostEqual(prior_w["BTC/USDT"], 0.70, places=3)
        self.assertAlmostEqual(prior_w["ETH/USDT"], 0.30, places=3)
        self.assertEqual(res["prior_weights"], res["posterior_weights"])


if __name__ == "__main__":
    unittest.main()
