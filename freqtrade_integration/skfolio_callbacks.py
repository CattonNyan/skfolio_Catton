"""Freqtrade strategy mixin that applies skfolio portfolio allocations."""

from __future__ import annotations

import math
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
