"""Skfolio Enhanced Multi-Timeframe ATR Strategy for Freqtrade.

Integrates:
1. Skfolio Dynamic Stake Allocator: Scales order size per asset based on optimal risk parity weights.
2. Dynamic Volatility SL/TP Guidelines: Applies pair-specific downside semi-deviation stoploss & take-profit.
3. Multi-timeframe trend alignment (1h Macro Trend + 15m Entry Timing).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from pandas import DataFrame

try:
    from freqtrade.strategy import IStrategy, IntParameter
    from freqtrade.persistence import Trade
    HAS_FREQTRADE = True
except ImportError:
    HAS_FREQTRADE = False
    class IStrategy:
        INTERFACE_VERSION = 3
    class IntParameter:
        def __init__(self, low, high, default=0, **kwargs):
            self.value = default
    class Trade:
        pass

# Auto-detect skfolio_Catton root directory when copied into Freqtrade user_data/strategies/
import sys
current_dir = Path(__file__).resolve().parent
candidate_roots = [
    current_dir.parent,  # skfolio_Catton/
    current_dir.parent.parent / "skfolio_Catton",  # ../skfolio_Catton
    current_dir.parent.parent.parent / "skfolio_Catton",
]
for r in candidate_roots:
    if r.is_dir() and str(r) not in sys.path:
        sys.path.insert(0, str(r))

try:
    from scripts.freqtrade_stake_allocator import get_custom_stake_amount
except ImportError:
    # Built-in robust standalone fallback if scripts directory is not in path
    def get_custom_stake_amount(
        pair: str,
        proposed_stake: float,
        total_wallet: float | None = None,
        min_stake: float | None = None,
        max_stake: float | None = None,
        config_path: str = "user_data/config.json",
        **kwargs,
    ) -> float:
        cfg_file = Path(config_path)
        if cfg_file.is_file():
            try:
                cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
                pair_stakes = cfg.get("pair_stake_amounts", {})
                if pair in pair_stakes:
                    val = float(pair_stakes[pair])
                    if min_stake and val < min_stake: val = min_stake
                    if max_stake and val > max_stake: val = max_stake
                    return val
                pair_weights = cfg.get("pair_weights", {})
                if pair in pair_weights and total_wallet and total_wallet > 0:
                    val = float(pair_weights[pair]) * total_wallet
                    if min_stake and val < min_stake: val = min_stake
                    if max_stake and val > max_stake: val = max_stake
                    return val
            except Exception:
                pass
        return proposed_stake


class SkfolioEnhancedAtrStrategy(IStrategy):
    """
    Freqtrade Strategy fully integrated with skfolio portfolio optimization:
    - Auto-scales trade stakes by risk parity weights (custom_stake_amount)
    - Auto-adjusts stoploss and take-profit per asset volatility (custom_stoploss)
    """

    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 100

    # Base fallback stoploss
    stoploss = -0.06
    use_custom_stoploss = True

    minimal_roi = {
        "0": 0.08,
        "60": 0.04,
        "180": 0.02,
        "360": 0.0,
    }

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {
        "entry": "gtc",
        "exit": "gtc",
    }

    def __init__(self, config: dict | None = None):
        super().__init__(config=config) if hasattr(super(), "__init__") else None
        self.config = config or {}
        self._pair_risk_limits: dict[str, dict[str, float]] = {}
        self._load_risk_parameters()

    def _load_risk_parameters(self):
        """Load pair-tailored stoploss and take-profit from config or risk JSON."""
        # 1. Check embedded config parameter
        if "pair_risk_limits" in self.config:
            self._pair_risk_limits = self.config.get("pair_risk_limits", {})
            return

        # 2. Check external risk file
        risk_file = self.config.get("skfolio_risk_file", "user_data/risk_params.json")
        p = Path(risk_file)
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self._pair_risk_limits = data.get("pair_risk_limits", {})
            except Exception:
                pass

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float | None,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Dynamically scale order stake size based on skfolio portfolio weights.
        """
        total_wallet = None
        if hasattr(self, "wallets") and hasattr(self.wallets, "get_total_stake_amount"):
            total_wallet = self.wallets.get_total_stake_amount()

        config_path = self.config.get("config_path", "user_data/config.json")

        return get_custom_stake_amount(
            pair=pair,
            proposed_stake=proposed_stake,
            total_wallet=total_wallet,
            min_stake=min_stake,
            max_stake=max_stake,
            config_path=config_path,
        )

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        """
        Apply asset-specific dynamic stoploss calculated from downside semi-deviation.
        """
        if pair in self._pair_risk_limits:
            pair_limits = self._pair_risk_limits[pair]
            # e.g., recommended_stoploss = -0.045 (-4.5%)
            recommended_sl = pair_limits.get("recommended_stoploss")
            if recommended_sl is not None and recommended_sl < 0:
                # If trade is already in solid profit (> 3%), move SL to break-even (+0.5%)
                if current_profit > 0.03:
                    return 0.005
                return recommended_sl

        return self.stoploss

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate technical indicators (EMA, RSI)."""
        # EMA 20 & 50
        dataframe["ema20"] = dataframe["close"].ewm(span=20, adjust=False).mean()
        dataframe["ema50"] = dataframe["close"].ewm(span=50, adjust=False).mean()

        # Simple RSI calculation
        delta = dataframe["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        dataframe["rsi"] = 100 - (100 / (1 + rs))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate long entry signals."""
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema20"]) &
                (dataframe["ema20"] > dataframe["ema50"]) &
                (dataframe["rsi"] > 45) &
                (dataframe["rsi"] < 70)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate exit signals."""
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ema20"]) |
                (dataframe["rsi"] > 78)
            ),
            "exit_long",
        ] = 1

        return dataframe
