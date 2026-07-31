"""Position lifecycle manager."""

from datetime import datetime
from typing import Optional

from loguru import logger

from src.core.enums import Side
from src.core.events import Events, event_bus
from src.core.models import Order, Position, Trade


class PositionManager:
    """Manages position lifecycle, stops, and take profits."""

    def __init__(self) -> None:
        self.positions: dict[str, Position] = {}

    async def open_position(self, order: Order) -> Position:
        """Create a new position from a filled order."""
        position = Position(
            symbol=order.symbol,
            exchange=order.exchange,
            side=order.side,
            quantity=order.filled_quantity,
            entry_price=order.average_price,
            current_price=order.average_price,
            position_type=order.position_type,
            entry_order_id=order.id,
            signal_id=order.signal_id,
            entry_time=order.filled_at or datetime.now(),
        )
        self.positions[position.id] = position
        logger.info(f"Position {position.id} opened for {position.symbol}")
        await event_bus.publish(Events.POSITION_OPENED, position=position)
        return position

    async def close_position(self, position_id: str, exit_order: Order) -> Trade:
        """Close a position and create a Trade record."""
        pos = self.positions.pop(position_id)
        
        multiplier = 1 if pos.side == Side.BUY else -1
        gross_pnl = multiplier * (exit_order.average_price - pos.entry_price) * pos.quantity
        
        trade = Trade(
            symbol=pos.symbol,
            exchange=pos.exchange,
            side=pos.side,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_price=exit_order.average_price,
            entry_time=pos.entry_time,
            exit_time=exit_order.filled_at or datetime.now(),
            gross_pnl=gross_pnl,
            total_fees=exit_order.fees.total,
            net_pnl=gross_pnl - exit_order.fees.total,
            entry_order_id=pos.entry_order_id,
            exit_order_id=exit_order.id,
            signal_id=pos.signal_id
        )
        
        logger.info(f"Position {position_id} closed for {pos.symbol}, PnL: {trade.net_pnl:.2f}")
        await event_bus.publish(Events.POSITION_CLOSED, trade=trade)
        return trade

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current_price on all open positions."""
        for pos in self.positions.values():
            if pos.symbol in prices:
                pos.current_price = prices[pos.symbol]

    def check_stop_losses(self) -> list[str]:
        """Returns position IDs that hit stop loss."""
        triggered = []
        for pos in self.positions.values():
            if pos.should_stop_loss():
                triggered.append(pos.id)
        return triggered

    def check_take_profits(self) -> list[str]:
        """Returns position IDs that hit take profit."""
        triggered = []
        for pos in self.positions.values():
            if pos.should_take_profit():
                triggered.append(pos.id)
        return triggered

    def update_trailing_stops(self, atr_values: dict[str, float], multiplier: float) -> None:
        """Adjust trailing stops based on ATR."""
        for pos in self.positions.values():
            if pos.symbol in atr_values:
                atr = atr_values[pos.symbol]
                distance = atr * multiplier
                
                if pos.side == Side.BUY:
                    new_stop = pos.current_price - distance
                    if pos.trailing_stop is None or new_stop > pos.trailing_stop:
                        pos.trailing_stop = new_stop
                        if pos.stop_loss is None or new_stop > pos.stop_loss:
                            pos.stop_loss = new_stop
                else:
                    new_stop = pos.current_price + distance
                    if pos.trailing_stop is None or new_stop < pos.trailing_stop:
                        pos.trailing_stop = new_stop
                        if pos.stop_loss is None or new_stop < pos.stop_loss:
                            pos.stop_loss = new_stop

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        return self.positions.get(position_id)

    def get_positions_by_symbol(self, symbol: str) -> list[Position]:
        """Get all open positions for a symbol."""
        return [p for p in self.positions.values() if p.symbol == symbol]
