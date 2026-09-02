"""Freqtrade Dynamic Stake Amount Allocation Bridge.

Provides a clean integration helper for Freqtrade strategies:
1. Loads optimal pair weights or stake amounts calculated by skfolio (from config.json or allocation.json).
2. Calculates custom order stake amounts per asset inside Freqtrade's custom_stake_amount() callback.
3. Enforces exchange min_stake and max_stake boundaries safely.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


class SkfolioStakeAllocator:
    """Helper to allocate dynamic trade stake amounts based on skfolio optimization."""

    def __init__(self, allocation_file: str | Path = "user_data/portfolio_allocation.json"):
        self.allocation_file = Path(allocation_file)
        self._cached_weights: dict[str, float] = {}
        self._cached_stakes: dict[str, float] = {}
        self._mtime: float = 0.0
        self._load_allocation()

    def _load_allocation(self):
        """Load weights and absolute stakes from allocation or config JSON."""
        if not self.allocation_file.is_file():
            return

        try:
            mtime = self.allocation_file.stat().st_mtime
            if mtime <= self._mtime:
                return

            data = json.loads(self.allocation_file.read_text(encoding="utf-8"))
            self._mtime = mtime

            # Check if it's a Freqtrade config.json
            if "skfolio_allocation" in data:
                sk_info = data["skfolio_allocation"]
                if sk_info.get("data_source", "").lower() == "synthetic":
                    self._cached_weights = {}
                    self._cached_stakes = {}
                    return
                self._cached_weights = data.get("pair_weights", {})
                self._cached_stakes = data.get("pair_stake_amounts", {})
            else:
                self._cached_weights = data.get("pair_weights", data.get("weights", {}))
                self._cached_stakes = data.get("pair_stake_amounts", data.get("stake_amounts", {}))
        except Exception:
            # Keep previous valid cache if any error occurs
            pass

    def get_stake_amount(
        self,
        pair: str,
        proposed_stake: float,
        total_wallet: float | None = None,
        min_stake: float | None = None,
        max_stake: float | None = None,
    ) -> float:
        """
        Calculate the adjusted stake amount for a given pair.

        Parameters:
        - pair: Trading pair name (e.g. 'BTC/USDT')
        - proposed_stake: Freqtrade's default calculated stake
        - total_wallet: Total wallet balance in stake currency
        - min_stake: Exchange minimum allowed order size
        - max_stake: Exchange maximum allowed order size
        """
        self._load_allocation()

        target_stake = proposed_stake

        # 1. Prefer explicit absolute stake amount if defined
        if pair in self._cached_stakes:
            target_stake = float(self._cached_stakes[pair])
        # 2. Or apply weight to total available wallet balance
        elif pair in self._cached_weights and total_wallet is not None and total_wallet > 0:
            weight = float(self._cached_weights[pair])
            if math.isfinite(weight) and weight > 0:
                target_stake = total_wallet * weight

        # 3. Enforce boundary conditions
        if min_stake is not None and target_stake < min_stake:
            target_stake = min_stake
        if max_stake is not None and target_stake > max_stake:
            target_stake = max_stake

        return round(target_stake, 4)


# Convenience singleton for simple strategy usage
_default_allocator: SkfolioStakeAllocator | None = None


def get_custom_stake_amount(
    pair: str,
    proposed_stake: float,
    total_wallet: float | None = None,
    min_stake: float | None = None,
    max_stake: float | None = None,
    config_path: str | Path = "user_data/config.json",
) -> float:
    """Standalone function to be called directly from custom_stake_amount()."""
    global _default_allocator
    if _default_allocator is None or _default_allocator.allocation_file != Path(config_path):
        _default_allocator = SkfolioStakeAllocator(allocation_file=config_path)

    return _default_allocator.get_stake_amount(
        pair=pair,
        proposed_stake=proposed_stake,
        total_wallet=total_wallet,
        min_stake=min_stake,
        max_stake=max_stake,
    )
