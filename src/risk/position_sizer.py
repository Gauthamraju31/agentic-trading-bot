"""Position sizing calculator."""
import math
from loguru import logger
from src.core.enums import PositionSizingMethod
from src.core.config import settings

class PositionSizer:
    """Calculates position size based on various methods."""

    @staticmethod
    def calculate_size(
        method: PositionSizingMethod,
        capital: float,
        risk_per_trade_pct: float,
        entry_price: float,
        stop_loss_price: float,
        win_rate: float | None = None,
        avg_win_loss_ratio: float | None = None,
    ) -> int:
        """Calculate the position size.

        Args:
            method: The position sizing method to use.
            capital: Total capital available.
            risk_per_trade_pct: Risk percentage per trade (e.g. 1.0 for 1%).
            entry_price: Expected entry price.
            stop_loss_price: Stop loss price.
            win_rate: Historical win rate (required for Kelly methods).
            avg_win_loss_ratio: Historical average win/loss ratio (required for Kelly methods).

        Returns:
            Number of shares to trade (minimum 1, rounded down).
        """
        if capital <= 0 or entry_price <= 0:
            return 0
            
        risk_pct_decimal = risk_per_trade_pct / 100.0
        quantity = 0.0

        if method == PositionSizingMethod.FIXED_FRACTIONAL:
            risk_amount = capital * risk_pct_decimal
            risk_per_share = abs(entry_price - stop_loss_price)
            if risk_per_share > 0:
                quantity = risk_amount / risk_per_share
            else:
                logger.warning("Stop loss price is equal to entry price. Cannot calculate fixed fractional size.")
                return 0
                
        elif method in (PositionSizingMethod.KELLY, PositionSizingMethod.HALF_KELLY):
            if win_rate is None or avg_win_loss_ratio is None or avg_win_loss_ratio <= 0:
                logger.warning("Win rate and win/loss ratio required for Kelly sizing. Falling back to FIXED_FRACTIONAL.")
                return PositionSizer.calculate_size(
                    PositionSizingMethod.FIXED_FRACTIONAL,
                    capital,
                    risk_per_trade_pct,
                    entry_price,
                    stop_loss_price
                )
                
            kelly_f = win_rate - ((1 - win_rate) / avg_win_loss_ratio)
            
            if kelly_f <= 0:
                logger.info("Kelly fraction is <= 0. Not taking the trade.")
                return 0
                
            if method == PositionSizingMethod.HALF_KELLY:
                kelly_f = kelly_f / 2.0
                
            risk_amount = capital * kelly_f
            quantity = risk_amount / entry_price
            
        else:
            logger.error(f"Unsupported position sizing method: {method}")
            return 0
            
        max_position_pct = settings.risk.max_position_pct / 100.0
        max_position_value = capital * max_position_pct
        max_quantity = int(max_position_value / entry_price)
        
        final_quantity = int(math.floor(quantity))
        final_quantity = min(final_quantity, max_quantity)
        final_quantity = max(1, final_quantity) if final_quantity > 0 else 0
        
        return final_quantity
