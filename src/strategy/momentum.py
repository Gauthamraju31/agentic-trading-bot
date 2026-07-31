"""Momentum trading strategy."""
from typing import List
from src.core.models import Candle, IndicatorValues, PortfolioState, Signal
from src.core.enums import SignalAction
from src.strategy.base import Strategy

class MomentumStrategy(Strategy):
    """Trend-following strategy based on EMA crossover, RSI, and ADX.

    Tunable parameters (used by walk-forward optimization) with defaults:
        rsi_buy_min / rsi_buy_max: RSI band required for a long entry (40-70).
        rsi_sell_max: RSI must be below this for a short entry (60).
        adx_min: minimum ADX (trend strength) to trade (25).
    """

    def __init__(
        self,
        rsi_buy_min: float = 40.0,
        rsi_buy_max: float = 70.0,
        rsi_sell_max: float = 60.0,
        adx_min: float = 25.0,
        **_ignored,
    ) -> None:
        self.rsi_buy_min = rsi_buy_min
        self.rsi_buy_max = rsi_buy_max
        self.rsi_sell_max = rsi_sell_max
        self.adx_min = adx_min

    @property
    def name(self) -> str:
        return "Momentum"

    @property
    def required_history(self) -> int:
        return 50

    @property
    def required_indicators(self) -> list[str]:
        return ["ema_9", "ema_21", "rsi_14", "adx", "volume_sma_20"]

    def generate_signals(
        self,
        candles: list[Candle],
        indicators: IndicatorValues,
        portfolio: PortfolioState,
    ) -> list[Signal]:
        if len(candles) < 2:
            return []

        current_ema9 = indicators.ema_9
        current_ema21 = indicators.ema_21
        current_rsi = indicators.rsi_14
        current_adx = indicators.adx
        current_vol_sma = indicators.volume_sma_20
        
        if current_ema9 is None or current_ema21 is None or current_rsi is None or current_adx is None or current_vol_sma is None:
            return []

        signals: list[Signal] = []
        current_candle = candles[-1]

        # Buy condition
        if current_ema9 > current_ema21 and self.rsi_buy_min <= current_rsi <= self.rsi_buy_max and current_adx > self.adx_min:
            if current_candle.volume > current_vol_sma:
                confidence = min(1.0, current_adx / 100.0)
                if current_rsi > 50:
                    confidence += 0.1
                confidence = min(1.0, confidence)
                
                signals.append(Signal(
                    strategy=self.name,
                    symbol=current_candle.symbol,
                    action=SignalAction.BUY,
                    confidence=confidence,
                    timestamp=current_candle.timestamp,
                    metadata={
                        "ema_9": current_ema9,
                        "ema_21": current_ema21,
                        "rsi_14": current_rsi,
                        "adx": current_adx,
                    }
                ))

        # Sell condition
        elif current_ema9 < current_ema21 and current_rsi < self.rsi_sell_max and current_adx > self.adx_min:
            confidence = min(1.0, current_adx / 100.0)
            if current_rsi < 50:
                confidence += 0.1
            confidence = min(1.0, confidence)
            
            signals.append(Signal(
                strategy=self.name,
                symbol=current_candle.symbol,
                action=SignalAction.SELL,
                confidence=confidence,
                timestamp=current_candle.timestamp,
                metadata={
                    "ema_9": current_ema9,
                    "ema_21": current_ema21,
                    "rsi_14": current_rsi,
                    "adx": current_adx,
                }
            ))

        return signals
