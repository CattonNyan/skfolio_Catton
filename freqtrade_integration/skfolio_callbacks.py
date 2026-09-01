"""Freqtrade strategy mixin that applies skfolio portfolio allocations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class SkfolioAllocationMixin:
    """Apply exported ``pair_weights`` through Freqtrade's stake callback.

    Put this mixin before ``IStrategy`` in the strategy's base classes. The
    optimizer writes ``pair_weights`` and, when a wallet size is supplied,
    ``pair_stake_amounts`` into the Freqtrade configuration.
    """

    allocation_fail_closed = True

    def _validated_pair_weights(self) -> dict[str, float]:
        raw = getattr(self, "config", {}).get("pair_weights", {})
        if not isinstance(raw, dict) or not raw:
            return {}

        weights: dict[str, float] = {}
        for pair, value in raw.items():
            try:
                weight = float(value)
            except (TypeError, ValueError):
                return {}
            if not isinstance(pair, str) or not math.isfinite(weight) or weight < 0:
                return {}
            weights[pair] = weight

        total = sum(weights.values())
        if total <= 0:
            return {}
        return {pair: weight / total for pair, weight in weights.items()}

    @staticmethod
    def _clamp_stake(value: float, min_stake: float | None, max_stake: float) -> float:
        value = min(value, max_stake)
        if min_stake is not None and value < min_stake:
            return 0.0
        return max(value, 0.0)

    def custom_stake_amount(
        self,
        pair: str,
        current_time: Any,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        """Return the configured amount for ``pair`` at entry time."""
        weights = self._validated_pair_weights()
        if not weights or pair not in weights:
            return 0.0 if self.allocation_fail_closed else proposed_stake

        configured_amounts = getattr(self, "config", {}).get("pair_stake_amounts", {})
        amount = None
        if isinstance(configured_amounts, dict) and pair in configured_amounts:
            try:
                candidate = float(configured_amounts[pair])
                if math.isfinite(candidate) and candidate > 0:
                    amount = candidate
            except (TypeError, ValueError):
                pass

        if amount is None:
            wallets = getattr(self, "wallets", None)
            get_total = getattr(wallets, "get_total_stake_amount", None)
            if callable(get_total):
                total_stake = float(get_total())
                amount = total_stake * weights[pair]
            else:
                amount = proposed_stake

        return self._clamp_stake(amount, min_stake, max_stake)


class SkfolioRiskMixin:
    """Apply per-pair volatility limits through Freqtrade risk callbacks."""

    use_custom_stoploss = True
    use_custom_roi = True

    def _validated_risk_limits(self) -> dict[str, dict[str, float]]:
        cached = getattr(self, "_skfolio_risk_limits_cache", None)
        if cached is not None:
            return cached

        config = getattr(self, "config", {})
        raw = config.get("pair_risk_limits", {})
        if not raw and config.get("skfolio_risk_file"):
            try:
                payload = json.loads(Path(config["skfolio_risk_file"]).read_text(encoding="utf-8"))
                if str(payload.get("data_source", "")).lower() == "synthetic":
                    raw = {}
                else:
                    raw = payload.get("assets", {})
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                raw = {}

        limits: dict[str, dict[str, float]] = {}
        if isinstance(raw, dict):
            for pair, values in raw.items():
                if not isinstance(pair, str) or not isinstance(values, dict):
                    continue
                try:
                    stoploss = abs(float(values["recommended_stoploss"]))
                    take_profit = float(values["recommended_take_profit"])
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    math.isfinite(stoploss)
                    and math.isfinite(take_profit)
                    and 0 < stoploss < 1
                    and 0 < take_profit < 10
                ):
                    limits[pair] = {
                        "stoploss": stoploss,
                        "take_profit": take_profit,
                    }

        self._skfolio_risk_limits_cache = limits
        return limits

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: Any,
        current_rate: float,
        current_profit: float,
        after_fill: bool = False,
        **kwargs: Any,
    ) -> float | None:
        """Return the pair-specific stop distance, adjusted for leverage."""
        risk = self._validated_risk_limits().get(pair)
        if risk is None:
            return None
        leverage = max(float(getattr(trade, "leverage", 1.0) or 1.0), 1.0)
        return risk["stoploss"] * leverage

    def custom_roi(
        self,
        pair: str,
        trade: Any,
        current_time: Any,
        trade_duration: int,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float | None:
        """Return the pair-specific take-profit threshold."""
        risk = self._validated_risk_limits().get(pair)
        if risk is None:
            return None
        leverage = max(float(getattr(trade, "leverage", 1.0) or 1.0), 1.0)
        return risk["take_profit"] * leverage


class SkfolioFreqtradeMixin(SkfolioRiskMixin, SkfolioAllocationMixin):
    """Combined allocation, stoploss, and take-profit integration."""
