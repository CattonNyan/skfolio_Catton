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


if __name__ == "__main__":
    unittest.main()
