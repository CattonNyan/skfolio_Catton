"""Tests for the Freqtrade allocation callback integration."""

import json
import tempfile
import unittest
from pathlib import Path

from freqtrade_integration import SkfolioAllocationMixin, SkfolioFreqtradeMixin


class _Wallets:
    def __init__(self, total: float):
        self.total = total

    def get_total_stake_amount(self) -> float:
        return self.total


class _Strategy(SkfolioAllocationMixin):
    def __init__(self, config: dict, wallet: float = 1000.0):
        self.config = config
        self.wallets = _Wallets(wallet)


class _IntegratedStrategy(SkfolioFreqtradeMixin):
    def __init__(self, config: dict):
        self.config = config


class _Trade:
    def __init__(self, leverage: float = 1.0):
        self.leverage = leverage


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

    def test_dynamic_stoploss_and_roi_use_pair_limits(self):
        strategy = _IntegratedStrategy(
            {
                "pair_risk_limits": {
                    "BTC/USDT": {
                        "recommended_stoploss": -0.04,
                        "recommended_take_profit": 0.08,
                    }
                }
            }
        )
        trade = _Trade()
        stop = strategy.custom_stoploss("BTC/USDT", trade, None, 100.0, 0.0)
        roi = strategy.custom_roi("BTC/USDT", trade, None, 10, None, "long")
        self.assertEqual(stop, 0.04)
        self.assertEqual(roi, 0.08)

    def test_risk_callbacks_adjust_price_targets_for_leverage(self):
        strategy = _IntegratedStrategy(
            {
                "pair_risk_limits": {
                    "BTC/USDT": {
                        "recommended_stoploss": -0.04,
                        "recommended_take_profit": 0.08,
                    }
                }
            }
        )
        trade = _Trade(leverage=2.0)
        self.assertEqual(strategy.custom_stoploss("BTC/USDT", trade, None, 100.0, 0.0), 0.08)
        self.assertEqual(strategy.custom_roi("BTC/USDT", trade, None, 10, None, "long"), 0.16)

    def test_risk_file_is_loaded_and_synthetic_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            risk_file = Path(tmpdir) / "risk.json"
            risk_file.write_text(
                json.dumps(
                    {
                        "data_source": "real-test-data",
                        "assets": {
                            "ETH/USDT": {
                                "recommended_stoploss": -0.03,
                                "recommended_take_profit": 0.06,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            strategy = _IntegratedStrategy({"skfolio_risk_file": str(risk_file)})
            self.assertEqual(strategy.custom_roi("ETH/USDT", _Trade(), None, 10, None, "long"), 0.06)

            risk_file.write_text(
                json.dumps({"data_source": "synthetic", "assets": {"ETH/USDT": {}}}),
                encoding="utf-8",
            )
            synthetic_strategy = _IntegratedStrategy({"skfolio_risk_file": str(risk_file)})
            self.assertIsNone(
                synthetic_strategy.custom_stoploss("ETH/USDT", _Trade(), None, 100.0, 0.0)
            )

    def test_alternative_risk_key_names(self):
        strategy = _IntegratedStrategy(
            {
                "pair_risk_limits": {
                    "SOL/USDT": {
                        "stoploss": -0.05,
                        "recommended_takeprofit": 0.10,
                    }
                }
            }
        )
        trade = _Trade()
        self.assertEqual(strategy.custom_stoploss("SOL/USDT", trade, None, 100.0, 0.0), 0.05)
        self.assertEqual(strategy.custom_roi("SOL/USDT", trade, None, 10, None, "long"), 0.10)
