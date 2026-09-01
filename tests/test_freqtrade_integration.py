"""Tests for the Freqtrade allocation callback integration."""

import unittest

from freqtrade_integration import SkfolioAllocationMixin


class _Wallets:
    def __init__(self, total: float):
        self.total = total

    def get_total_stake_amount(self) -> float:
        return self.total


class _Strategy(SkfolioAllocationMixin):
    def __init__(self, config: dict, wallet: float = 1000.0):
        self.config = config
        self.wallets = _Wallets(wallet)


class FreqtradeAllocationTests(unittest.TestCase):
    def _stake(self, strategy: _Strategy, pair: str, **overrides) -> float:
        params = {
            "pair": pair,
            "current_time": None,
            "current_rate": 100.0,
            "proposed_stake": 100.0,
            "min_stake": 10.0,
            "max_stake": 1000.0,
            "leverage": 1.0,
            "entry_tag": None,
            "side": "long",
        }
        params.update(overrides)
        return strategy.custom_stake_amount(**params)

    def test_weight_is_applied_to_available_wallet(self):
        strategy = _Strategy({"pair_weights": {"BTC/USDT": 0.6, "ETH/USDT": 0.4}})
        self.assertEqual(self._stake(strategy, "BTC/USDT"), 600.0)

    def test_exported_absolute_amount_takes_precedence(self):
        strategy = _Strategy(
            {
                "pair_weights": {"BTC/USDT": 1.0},
                "pair_stake_amounts": {"BTC/USDT": 250.0},
            }
        )
        self.assertEqual(self._stake(strategy, "BTC/USDT"), 250.0)

    def test_unknown_pair_and_invalid_config_fail_closed(self):
        strategy = _Strategy({"pair_weights": {"BTC/USDT": float("nan")}})
        self.assertEqual(self._stake(strategy, "ETH/USDT"), 0.0)

    def test_exchange_limits_are_respected(self):
        strategy = _Strategy({"pair_weights": {"BTC/USDT": 1.0}}, wallet=5000.0)
        self.assertEqual(self._stake(strategy, "BTC/USDT", max_stake=750.0), 750.0)
