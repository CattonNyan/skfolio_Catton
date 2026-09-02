"""Tests for Freqtrade stake allocator bridge module."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.freqtrade_stake_allocator import SkfolioStakeAllocator, get_custom_stake_amount


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


if __name__ == "__main__":
    unittest.main()
