"""Mean reversion trading strategy."""
from typing import List
from src.core.models import Candle, IndicatorValues, PortfolioState, Signal
from src.core.enums import SignalAction
from src.strategy.base import Strategy

class MeanReversionStrategy(Strategy):
    """Mean reversion strategy using Bollinger Bands and RSI.

    Tunable parameters (used by walk-forward optimization) with defaults:
        rsi_buy_below: buy when RSI is under this and price <= lower band (30).
        rsi_sell_above: sell when RSI is over this and price >= upper band (70).
        adx_max: only trade when ADX is below this (ranging market) (30).
    """

    def __init__(
        self,
        rsi_buy_below: float = 30.0,
        rsi_sell_above: float = 70.0,
        adx_max: float = 30.0,
        **_ignored,
    ) -> None:
        self.rsi_buy_below = rsi_buy_below
        self.rsi_sell_above = rsi_sell_above
        self.adx_max = adx_max

    @property
    def name(self) -> str:
        return "MeanReversion"

    @property
    def required_history(self) -> int:
        return 30

    @property
    def required_indicators(self) -> list[str]:
        return ["bb_upper", "bb_lower", "rsi_14", "adx", "supertrend"]

    def generate_signals(
        self,
        candles: list[Candle],
        indicators: IndicatorValues,
        portfolio: PortfolioState,
    ) -> list[Signal]:
        if not candles:
            return []

        current_candle = candles[-1]
        
        current_rsi = indicators.rsi_14
        bb_upper = indicators.bb_upper
        bb_lower = indicators.bb_lower
        current_adx = indicators.adx

        # Any of these can be None until enough history has accumulated.
        if (
            current_rsi is None or bb_upper is None or bb_lower is None
            or current_adx is None
        ):
            return []

        supertrend_bullish = (indicators.supertrend or 0) > 0

        signals: List[Signal] = []

        # Buy condition
        if current_candle.close <= bb_lower and current_rsi < self.rsi_buy_below:
            if current_adx < self.adx_max or supertrend_bullish:
                confidence = 0.5 + min(0.5, (self.rsi_buy_below - current_rsi) / 30.0)
                
                signals.append(Signal(
                    strategy=self.name,
                    symbol=current_candle.symbol,
                    action=SignalAction.BUY,
                    confidence=confidence,
                    timestamp=current_candle.timestamp,
                    metadata={
                        "rsi_14": current_rsi,
                        "bb_lower": bb_lower,
                        "adx": current_adx,
                    }
                ))

        # Sell condition
        elif current_candle.close >= bb_upper and current_rsi > self.rsi_sell_above:
            if current_adx < self.adx_max or not supertrend_bullish:
                confidence = 0.5 + min(0.5, (current_rsi - self.rsi_sell_above) / 30.0)
                
                signals.append(Signal(
                    strategy=self.name,
                    symbol=current_candle.symbol,
                    action=SignalAction.SELL,
                    confidence=confidence,
                    timestamp=current_candle.timestamp,
                    metadata={
                        "rsi_14": current_rsi,
                        "bb_upper": bb_upper,
                        "adx": current_adx,
                    }
                ))

        return signals
