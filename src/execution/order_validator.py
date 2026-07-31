"""Deterministic order validation (NO LLM)."""

from datetime import datetime

from loguru import logger

from src.core.config import settings
from src.core.enums import OrderType, Side
from src.core.models import Order, PortfolioState


class OrderValidator:
    """Validates orders before they are sent to the broker."""

    def __init__(self) -> None:
        self.config = settings.risk
        self.market_config = settings.market
        self.is_backtest = False

    def set_backtest_mode(self, enabled: bool) -> None:
        """Disables market hours check for backtesting."""
        self.is_backtest = enabled

    def validate(self, order: Order, portfolio: PortfolioState) -> tuple[bool, str]:
        """Validate order against risk limits and market conditions."""
        
        # 1. Market hours
        if not self.is_backtest:
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            start_time = self.market_config.trading_hours.start
            end_time = self.market_config.trading_hours.end
            
            if not (start_time <= current_time <= end_time):
                return False, f"Outside market hours ({start_time} - {end_time})"

        # 2. Order parameters
        if order.quantity <= 0:
            return False, "Order quantity must be positive"
            
        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LOSS) and (order.price is None or order.price <= 0):
            return False, "Limit/Stop orders must have a positive price"

        # 3. Portfolio limits
        # Check if closing position
        is_closing = False
        for pos in portfolio.positions:
            if pos.symbol == order.symbol and pos.side != order.side:
                is_closing = True
                break

        if not is_closing:
            if portfolio.open_position_count >= self.config.max_open_positions:
                return False, f"Max open positions ({self.config.max_open_positions}) reached"

            if portfolio.daily_pnl < 0:
                daily_loss_pct = abs(portfolio.daily_pnl) / portfolio.initial_capital * 100
                if daily_loss_pct >= self.config.max_daily_loss_pct:
                    return False, f"Daily loss limit ({self.config.max_daily_loss_pct}%) exceeded"

            # Reference price for cost/size checks. MARKET orders may not carry a
            # limit price, so fall back to trigger/average price when available.
            ref_price = order.price or order.trigger_price or order.average_price
            if ref_price:
                est_cost = order.quantity * ref_price

                if est_cost > portfolio.current_capital:
                    return False, "Insufficient capital"

                pos_pct = (est_cost / portfolio.current_capital) * 100
                if pos_pct > self.config.max_position_pct:
                    return False, f"Position size ({pos_pct:.1f}%) exceeds max_position_pct ({self.config.max_position_pct}%)"
            else:
                logger.warning(
                    f"Order {order.id} has no reference price; skipping capital/size "
                    f"validation. Provide a price for MARKET orders in backtests."
                )

        return True, "Valid"
