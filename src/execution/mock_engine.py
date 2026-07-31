"""Paper trading engine."""

from datetime import datetime

from loguru import logger

from src.core.config import settings
from src.core.enums import OrderStatus, OrderType, Side
from src.core.events import Events, event_bus
from src.core.models import Candle, Order, PortfolioState, Position, Trade

from .base import BrokerInterface
from .fees import FeeCalculator


class MockBroker(BrokerInterface):
    """Paper trading engine for backtesting and paper trading."""

    def __init__(self, initial_capital: float | None = None) -> None:
        self.initial_capital = initial_capital if initial_capital is not None else settings.mock.initial_capital
        self.slippage_pct = settings.mock.slippage_pct
        self.fee_calculator = FeeCalculator()
        
        self.orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}
        self.completed_trades: list[Trade] = []
        self.current_capital = self.initial_capital
        
        self.current_time = datetime.now()
        self.last_prices: dict[str, float] = {}
        
    @property
    def is_mock(self) -> bool:
        return True

    async def place_order(self, order: Order) -> Order:
        """Place a mock order."""
        self.orders[order.id] = order
        
        if order.order_type == OrderType.MARKET:
            if order.symbol in self.last_prices:
                await self._fill_order(order, self.last_prices[order.symbol])
            else:
                logger.warning(
                    f"Market order {order.id} for {order.symbol} placed "
                    f"but no price available, will fill on next candle."
                )
        else:
            logger.info(
                f"Order {order.id} placed: {order.order_type.value} "
                f"{order.side.value} {order.quantity} {order.symbol}"
            )
            
        return order

    async def _fill_order(self, order: Order, base_price: float) -> None:
        """Fill an order, calculating slippage and fees."""
        slippage_amt = base_price * (self.slippage_pct / 100.0)
        if order.side == Side.BUY:
            fill_price = base_price + slippage_amt
        else:
            fill_price = base_price - slippage_amt
            
        order.filled_quantity = order.quantity
        order.average_price = fill_price
        order.status = OrderStatus.FILLED
        order.filled_at = self.current_time
        
        order.fees = self.fee_calculator.calculate_fees(
            side=order.side,
            quantity=order.filled_quantity,
            price=order.average_price,
            position_type=order.position_type
        )
        
        self.current_capital -= order.fees.total
        
        logger.info(f"Order {order.id} filled at {fill_price:.2f}")
        
        await event_bus.publish(Events.ORDER_FILLED, order=order)
        await self._update_position_from_fill(order)

    async def _update_position_from_fill(self, order: Order) -> None:
        """Update internal positions and capital based on fill."""
        pos = self.positions.get(order.symbol)
        
        if pos is None:
            new_pos = Position(
                symbol=order.symbol,
                exchange=order.exchange,
                side=order.side,
                quantity=order.filled_quantity,
                entry_price=order.average_price,
                current_price=order.average_price,
                position_type=order.position_type,
                entry_order_id=order.id,
                entry_time=order.filled_at or datetime.now()
            )
            self.positions[order.symbol] = new_pos
            
            cost = new_pos.quantity * new_pos.entry_price
            if new_pos.side == Side.BUY:
                self.current_capital -= cost
            else:
                self.current_capital += cost
                
            await event_bus.publish(Events.POSITION_OPENED, position=new_pos)
        else:
            if pos.side == order.side:
                # Add to position
                total_qty = pos.quantity + order.filled_quantity
                total_cost = (pos.quantity * pos.entry_price) + (order.filled_quantity * order.average_price)
                pos.entry_price = total_cost / total_qty
                pos.quantity = total_qty
                
                cost = order.filled_quantity * order.average_price
                if order.side == Side.BUY:
                    self.current_capital -= cost
                else:
                    self.current_capital += cost
                    
                await event_bus.publish(Events.POSITION_UPDATED, position=pos)
            else:
                # Close or partial close
                close_qty = min(pos.quantity, order.filled_quantity)
                
                multiplier = 1 if pos.side == Side.BUY else -1
                gross_pnl = multiplier * (order.average_price - pos.entry_price) * close_qty
                
                margin_released = close_qty * pos.entry_price
                if pos.side == Side.BUY:
                    self.current_capital += margin_released + gross_pnl
                else:
                    self.current_capital -= margin_released - gross_pnl
                
                trade = Trade(
                    symbol=order.symbol,
                    exchange=order.exchange,
                    side=pos.side,
                    quantity=close_qty,
                    entry_price=pos.entry_price,
                    exit_price=order.average_price,
                    entry_time=pos.entry_time,
                    exit_time=order.filled_at or datetime.now(),
                    gross_pnl=gross_pnl,
                    total_fees=order.fees.total,
                    net_pnl=gross_pnl - order.fees.total,
                    entry_order_id=pos.entry_order_id,
                    exit_order_id=order.id
                )
                self.completed_trades.append(trade)
                
                pos.quantity -= close_qty
                if pos.quantity == 0:
                    del self.positions[order.symbol]
                    await event_bus.publish(Events.POSITION_CLOSED, trade=trade)
                else:
                    await event_bus.publish(Events.POSITION_UPDATED, position=pos)
                    
                # Reversal if remaining qty > 0
                remaining_qty = order.filled_quantity - close_qty
                if remaining_qty > 0:
                    new_pos = Position(
                        symbol=order.symbol,
                        exchange=order.exchange,
                        side=order.side,
                        quantity=remaining_qty,
                        entry_price=order.average_price,
                        current_price=order.average_price,
                        position_type=order.position_type,
                        entry_order_id=order.id,
                        entry_time=order.filled_at or datetime.now()
                    )
                    self.positions[order.symbol] = new_pos
                    cost = remaining_qty * new_pos.entry_price
                    if new_pos.side == Side.BUY:
                        self.current_capital -= cost
                    else:
                        self.current_capital += cost
                    await event_bus.publish(Events.POSITION_OPENED, position=new_pos)

    async def cancel_order(self, order_id: str) -> Order:
        """Cancel a pending order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                order.cancelled_at = datetime.now()
                await event_bus.publish(Events.ORDER_CANCELLED, order=order)
            return order
        raise ValueError(f"Order {order_id} not found")

    async def get_order_status(self, order_id: str) -> Order:
        """Get the current status of an order."""
        if order_id in self.orders:
            return self.orders[order_id]
        raise ValueError(f"Order {order_id} not found")

    async def get_positions(self) -> list[Position]:
        """Fetch all currently open positions."""
        return list(self.positions.values())

    async def get_portfolio(self) -> PortfolioState:
        """Get the current portfolio state snapshot."""
        total_realized = sum(t.net_pnl for t in self.completed_trades)
        
        state = PortfolioState(
            initial_capital=self.initial_capital,
            current_capital=self.current_capital,
            positions=list(self.positions.values()),
            completed_trades=self.completed_trades,
            total_realized_pnl=total_realized,
            peak_equity=self.initial_capital,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
        )
        return state

    async def process_candle(self, candle: Candle) -> None:
        """Check pending orders against OHLC range."""
        self.last_prices[candle.symbol] = candle.close
        self.current_time = candle.timestamp
        
        if candle.symbol in self.positions:
            self.positions[candle.symbol].current_price = candle.close
            
        pending = [
            o for o in self.orders.values() 
            if o.status == OrderStatus.PENDING and o.symbol == candle.symbol
        ]
        
        for order in pending:
            if order.order_type == OrderType.LIMIT and order.price:
                if order.side == Side.BUY and candle.low <= order.price:
                    await self._fill_order(order, order.price)
                elif order.side == Side.SELL and candle.high >= order.price:
                    await self._fill_order(order, order.price)
                    
            elif order.order_type == OrderType.STOP_LOSS and order.trigger_price:
                if order.side == Side.BUY and candle.high >= order.trigger_price:
                    await self._fill_order(order, order.trigger_price)
                elif order.side == Side.SELL and candle.low <= order.trigger_price:
                    await self._fill_order(order, order.trigger_price)
                    
            elif order.order_type == OrderType.STOP_LOSS_MARKET and order.trigger_price:
                if order.side == Side.BUY and candle.high >= order.trigger_price:
                    await self._fill_order(order, order.trigger_price)
                elif order.side == Side.SELL and candle.low <= order.trigger_price:
                    await self._fill_order(order, order.trigger_price)
            elif order.order_type == OrderType.MARKET:
                await self._fill_order(order, candle.close)

    def reset(self) -> None:
        """Clear all state for new backtest run."""
        self.orders.clear()
        self.positions.clear()
        self.completed_trades.clear()
        self.last_prices.clear()
        self.current_capital = self.initial_capital
