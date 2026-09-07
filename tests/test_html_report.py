"""Tests for HTML report exporter module."""

import unittest
import tempfile
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = str(Path(__file__).resolve().parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pandas as pd

from scripts.crypto_portfolio_optimizer import generate_synthetic_crypto_data
from scripts.export_html_report import generate_html_report


class HtmlReportTests(unittest.TestCase):
    def test_non_positive_wallet_is_rejected(self):
        prices = generate_synthetic_crypto_data(periods=50)
        with self.assertRaises(ValueError):
            generate_html_report(prices, {"BTC/USDT": 1.0}, total_wallet=0)

    def test_generate_html_report(self):
        prices = generate_synthetic_crypto_data(periods=50)
        weights = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_report.html"
            result_path = generate_html_report(
                prices=prices,
                weights=weights,
                model_name="Risk Parity (ERC)",
                total_wallet=10000.0,
                output_file=out_file,
            )
            self.assertTrue(result_path.is_file())
            content = result_path.read_text(encoding="utf-8")
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("BTC/USDT", content)
            self.assertIn("Risk Parity (ERC)", content)
            self.assertIn("Freqtrade", content)
            # Verify Dark Theme styling
            self.assertIn("#0E1117", content)
            self.assertIn("#161B22", content)

    def test_invalid_inputs_and_xss_escaping(self):
        prices = generate_synthetic_crypto_data(periods=50)
        for bad_wallet in (True, False, -100.0, float("nan")):
            with self.subTest(bad_wallet=bad_wallet), self.assertRaises(ValueError):
                generate_html_report(prices, {"BTC/USDT": 1.0}, total_wallet=bad_wallet)

        for bad_prices in ("not_a_df", pd.DataFrame(), prices.replace(prices.iloc[0, 0], -1.0)):
            with self.subTest(bad_prices=type(bad_prices)), self.assertRaises(ValueError):
                generate_html_report(bad_prices, {"BTC/USDT": 1.0})

        for bad_weights in ("not_a_dict", {}, {"BTC": -0.1}, {"BTC": True}):
            with self.subTest(bad_weights=bad_weights), self.assertRaises(ValueError):
                generate_html_report(prices, bad_weights)

        # No common assets rejected
        with self.assertRaises(ValueError):
            generate_html_report(prices, {"NON_EXISTENT_COIN": 1.0})

        # XSS escaping test
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "xss_test.html"
            xss_prices = pd.DataFrame({"<script>alert(1)</script>": [10.0, 11.0, 12.0]})
            res = generate_html_report(
                prices=xss_prices,
                weights={"<script>alert(1)</script>": 1.0},
                model_name="<b onmouseover=alert(1)>Model</b>",
                output_file=out_file,
            )
            html_text = res.read_text(encoding="utf-8")
            self.assertNotIn("<script>alert(1)</script>", html_text)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html_text)
            self.assertNotIn("<b onmouseover", html_text)
            self.assertIn("&lt;b onmouseover=alert(1)&gt;Model&lt;/b&gt;", html_text)


if __name__ == "__main__":
    unittest.main()
