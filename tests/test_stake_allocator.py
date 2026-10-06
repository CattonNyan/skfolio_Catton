"""Tests for Freqtrade stake allocator bridge module."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.freqtrade_stake_allocator import (
    SkfolioStakeAllocator,
    export_stake_allocation_csv,
    get_custom_stake_amount,
)


class StakeAllocatorTests(unittest.TestCase):
    def test_fallback_when_file_missing(self):
        allocator = SkfolioStakeAllocator(allocation_file="non_existent_file.json")
        stake = allocator.get_stake_amount("BTC/USDT", proposed_stake=100.0)
        self.assertEqual(stake, 100.0)

    def test_apply_explicit_stake_amounts(self):
        sample = {
            "pair_stake_amounts": {"BTC/USDT": 450.0, "ETH/USDT": 300.0}
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(sample), encoding="utf-8")

            allocator = SkfolioStakeAllocator(allocation_file=path)
            stake_btc = allocator.get_stake_amount("BTC/USDT", proposed_stake=100.0)
            stake_eth = allocator.get_stake_amount("ETH/USDT", proposed_stake=100.0)
            stake_sol = allocator.get_stake_amount("SOL/USDT", proposed_stake=100.0)

            self.assertEqual(stake_btc, 450.0)
            self.assertEqual(stake_eth, 300.0)
            self.assertEqual(stake_sol, 100.0)  # Fallback to proposed

    def test_apply_weights_with_total_wallet(self):
        sample = {
            "pair_weights": {"BTC/USDT": 0.45, "ETH/USDT": 0.35}
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(sample), encoding="utf-8")

            allocator = SkfolioStakeAllocator(allocation_file=path)
            stake = allocator.get_stake_amount(
                pair="BTC/USDT",
                proposed_stake=100.0,
                total_wallet=10000.0,
            )
            self.assertEqual(stake, 4500.0)

    def test_min_and_max_stake_enforcement(self):
        sample = {
            "pair_stake_amounts": {"BTC/USDT": 10.0, "ETH/USDT": 5000.0}
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(sample), encoding="utf-8")

            allocator = SkfolioStakeAllocator(allocation_file=path)
            stake_btc = allocator.get_stake_amount("BTC/USDT", proposed_stake=100.0, min_stake=20.0)
            stake_eth = allocator.get_stake_amount("ETH/USDT", proposed_stake=100.0, max_stake=2000.0)

            self.assertEqual(stake_btc, 20.0)  # Capped by min_stake
            self.assertEqual(stake_eth, 2000.0)  # Capped by max_stake

    def test_invalid_stake_boundaries_rejected(self):
        allocator = SkfolioStakeAllocator(allocation_file="non_existent_file.json")

        invalid_bounds = (
            {"min_stake": -1.0},
            {"max_stake": float("inf")},
            {"min_stake": 200.0, "max_stake": 100.0},
        )
        for bounds in invalid_bounds:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                allocator.get_stake_amount(
                    "BTC/USDT", proposed_stake=100.0, **bounds
                )

    def test_invalid_proposed_stake_rejected(self):
        allocator = SkfolioStakeAllocator(allocation_file="non_existent_file.json")

        for proposed_stake in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(proposed_stake=proposed_stake), self.assertRaises(ValueError):
                allocator.get_stake_amount(
                    "BTC/USDT", proposed_stake=proposed_stake
                )

    def test_invalid_total_wallet_rejected(self):
        allocator = SkfolioStakeAllocator(allocation_file="non_existent_file.json")

        for total_wallet in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(total_wallet=total_wallet), self.assertRaises(ValueError):
                allocator.get_stake_amount(
                    "BTC/USDT",
                    proposed_stake=100.0,
                    total_wallet=total_wallet,
                )

    def test_synthetic_data_config_is_rejected(self):
        sample = {
            "skfolio_allocation": {"data_source": "synthetic"},
            "pair_stake_amounts": {"BTC/USDT": 999.0},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(sample), encoding="utf-8")

            allocator = SkfolioStakeAllocator(allocation_file=path)
            stake = allocator.get_stake_amount("BTC/USDT", proposed_stake=100.0)
            self.assertEqual(stake, 100.0)  # Synthetic ignored, fallback to proposed

    def test_invalid_explicit_stake_falls_back_to_proposed(self):
        for configured_stake in (-10.0, 0.0, float("inf"), "invalid"):
            with self.subTest(configured_stake=configured_stake):
                sample = {"pair_stake_amounts": {"BTC/USDT": configured_stake}}
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / "config.json"
                    path.write_text(json.dumps(sample), encoding="utf-8")
                    allocator = SkfolioStakeAllocator(allocation_file=path)

                    stake = allocator.get_stake_amount(
                        "BTC/USDT", proposed_stake=100.0
                    )

                    self.assertEqual(stake, 100.0)

    def test_invalid_pair_and_bool_parameters_rejected(self):
        allocator = SkfolioStakeAllocator(allocation_file="non_existent_file.json")
        for bad_pair in ("", "   ", 123, None):
            with self.subTest(bad_pair=bad_pair), self.assertRaises(ValueError):
                allocator.get_stake_amount(bad_pair, proposed_stake=100.0)

        for bad_bool in (True, False):
            with self.subTest(bad_bool=bad_bool):
                with self.assertRaises(ValueError):
                    allocator.get_stake_amount("BTC/USDT", proposed_stake=bad_bool)
                with self.assertRaises(ValueError):
                    allocator.get_stake_amount("BTC/USDT", proposed_stake=100.0, total_wallet=bad_bool)
                with self.assertRaises(ValueError):
                    allocator.get_stake_amount("BTC/USDT", proposed_stake=100.0, min_stake=bad_bool)
                with self.assertRaises(ValueError):
                    allocator.get_stake_amount("BTC/USDT", proposed_stake=100.0, max_stake=bad_bool)

    def test_export_stake_allocation_csv(self):
        sample = {
            "pair_weights": {"BTC/USDT": 0.5, "ETH/USDT": 0.3, "SOL/USDT": 0.2}
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(sample), encoding="utf-8")
            allocator = SkfolioStakeAllocator(allocation_file=config_path)

            csv_out = Path(tmpdir) / "allocation.csv"
            result = export_stake_allocation_csv(
                allocator=allocator,
                pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                total_wallet=10000.0,
                output_path=csv_out,
                min_stake=100.0,
                max_stake=6000.0,
            )
            self.assertEqual(result, csv_out)
            self.assertTrue(csv_out.is_file())

            import pandas as pd
            df = pd.read_csv(csv_out)
            self.assertEqual(list(df.columns), ["pair", "allocated_stake", "weight_pct", "min_stake", "max_stake"])
            self.assertEqual(len(df), 3)

            btc_row = df[df["pair"] == "BTC/USDT"].iloc[0]
            self.assertEqual(btc_row["allocated_stake"], 5000.0)
            self.assertEqual(btc_row["weight_pct"], 50.0)

    def test_export_stake_allocation_csv_validation(self):
        allocator = SkfolioStakeAllocator(allocation_file="non_existent.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_out = Path(tmpdir) / "allocation.csv"

            # Invalid allocator
            with self.assertRaises(TypeError):
                export_stake_allocation_csv(None, ["BTC/USDT"], 1000.0, csv_out)

            # Invalid pairs
            for bad_pairs in ([], ["  "], [123], "BTC/USDT", None):
                with self.subTest(bad_pairs=bad_pairs), self.assertRaises(ValueError):
                    export_stake_allocation_csv(allocator, bad_pairs, 1000.0, csv_out)

            # Invalid wallet
            for bad_wallet in (0.0, -100.0, float("nan"), True, False):
                with self.subTest(bad_wallet=bad_wallet), self.assertRaises(ValueError):
                    export_stake_allocation_csv(allocator, ["BTC/USDT"], bad_wallet, csv_out)

            # Invalid bounds
            with self.assertRaises(ValueError):
                export_stake_allocation_csv(allocator, ["BTC/USDT"], 1000.0, csv_out, min_stake=500.0, max_stake=100.0)

    def test_cli_main_execution(self):
        from unittest.mock import patch
        from scripts.freqtrade_stake_allocator import main

        sample = {"pair_weights": {"BTC/USDT": 0.6, "ETH/USDT": 0.4}}
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            cfg_path.write_text(json.dumps(sample), encoding="utf-8")
            csv_path = Path(tmpdir) / "output.csv"

            test_args = [
                "freqtrade_stake_allocator.py",
                "--config", str(cfg_path),
                "--pairs", "BTC/USDT", "ETH/USDT",
                "--wallet", "5000.0",
                "--min-stake", "50.0",
                "--max-stake", "4000.0",
                "--export-csv", str(csv_path),
            ]
            with patch("sys.argv", test_args):
                main()

            self.assertTrue(csv_path.is_file())


if __name__ == "__main__":
    unittest.main()
