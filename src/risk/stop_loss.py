"""Stop loss and take profit manager."""
from loguru import logger
from src.core.enums import StopLossType, Side

class StopLossManager:
    """Manages stop loss and take profit calculations."""

    @staticmethod
    def calculate_stop_loss(
        stop_type: StopLossType,
        side: Side,
        entry_price: float,
        pct: float | None = None,
        atr: float | None = None,
        atr_multiplier: float | None = None,
    ) -> float:
        """Calculate initial stop loss price.

        Args:
            stop_type: The type of stop loss to use.
            side: The side of the trade (BUY or SELL).
            entry_price: The entry price.
            pct: Percentage for fixed percentage stop loss.
            atr: Current ATR value.
            atr_multiplier: Multiplier for ATR-based stop loss.

        Returns:
            The calculated stop loss price.
        """
        if stop_type == StopLossType.FIXED_PCT:
            if pct is None:
                raise ValueError("pct must be provided for FIXED_PCT stop loss")
            
            if side == Side.BUY:
                return entry_price * (1.0 - pct / 100.0)
            elif side == Side.SELL:
                return entry_price * (1.0 + pct / 100.0)
            else:
                raise ValueError(f"Invalid side: {side}")

        elif stop_type == StopLossType.ATR_BASED:
            if atr is None or atr_multiplier is None:
                raise ValueError("atr and atr_multiplier must be provided for ATR_BASED stop loss")
            
            if side == Side.BUY:
                return entry_price - (atr * atr_multiplier)
            elif side == Side.SELL:
                return entry_price + (atr * atr_multiplier)
            else:
                raise ValueError(f"Invalid side: {side}")
                
        else:
            raise ValueError(f"Unsupported stop loss type: {stop_type}")

    @staticmethod
    def calculate_take_profit(
        side: Side,
        entry_price: float,
        risk_reward_ratio: float,
        stop_loss: float,
    ) -> float:
        """Calculate take profit price based on risk-reward ratio.

        Args:
            side: The side of the trade.
            entry_price: The entry price.
            risk_reward_ratio: Desired reward / risk.
            stop_loss: The stop loss price.

        Returns:
            The calculated take profit price.
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if side == Side.BUY:
            return entry_price + reward
        elif side == Side.SELL:
            return entry_price - reward
        else:
            raise ValueError(f"Invalid side: {side}")

    @staticmethod
    def update_trailing_stop(
        side: Side,
        current_price: float,
        current_stop: float,
        trail_pct: float | None = None,
        atr: float | None = None,
        atr_multiplier: float | None = None,
    ) -> float:
        """Calculate updated trailing stop loss price.

        Args:
            side: The side of the trade.
            current_price: The current market price.
            current_stop: The current stop loss price.
            trail_pct: Percentage to trail by.
            atr: Current ATR value.
            atr_multiplier: Multiplier for ATR trailing stop.

        Returns:
            The new stop loss price.
        """
        new_stop = current_stop
        
        if trail_pct is not None:
            if side == Side.BUY:
                potential_stop = current_price * (1.0 - trail_pct / 100.0)
                new_stop = max(current_stop, potential_stop)
            elif side == Side.SELL:
                potential_stop = current_price * (1.0 + trail_pct / 100.0)
                new_stop = min(current_stop, potential_stop)
                
        elif atr is not None and atr_multiplier is not None:
            if side == Side.BUY:
                potential_stop = current_price - (atr * atr_multiplier)
                new_stop = max(current_stop, potential_stop)
            elif side == Side.SELL:
                potential_stop = current_price + (atr * atr_multiplier)
                new_stop = min(current_stop, potential_stop)
                
        return new_stop
