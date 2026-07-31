from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from loguru import logger
from pydantic import BaseModel

from src.core.models import Trade, Candle, Signal, Order
from src.core.enums import Side, SignalAction, OrderType, PositionType
from src.core.config import settings

from src.execution import MockBroker, FeeCalculator, OrderValidator, PositionManager
from src.risk import PositionSizer, StopLossManager, CircuitBreaker
from src.strategy import Strategy
from src.data import IndicatorEngine
from src.backtest.metrics import PerformanceMetrics


class BacktestResult(BaseModel):
    """Result model containing all outputs of a backtest."""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_equity: float
    trades: List[Trade]
    metrics: Dict[str, Any]
    equity_curve: List[float]
    signals_generated: int
    signals_executed: int
    signals_rejected: int


class BacktestEngine:
    """
    Event-driven backtesting engine to simulate trading strategies on historical data.
    Ensures no lookahead bias by streaming candles one by one.
    """
    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float,
        fee_calculator: FeeCalculator,
        settings: Any
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.fee_calculator = fee_calculator
        self.settings = settings
        
        self.broker = MockBroker(initial_capital=initial_capital)
        self.position_manager = PositionManager()
        self.order_validator = OrderValidator()
        self.order_validator.set_backtest_mode(True)
        self.position_sizer = PositionSizer()
        self.stop_loss_manager = StopLossManager()
        self.circuit_breaker = CircuitBreaker()
        self.indicator_engine = IndicatorEngine()
        
        self.signals_generated = 0
        self.signals_executed = 0
        self.signals_rejected = 0

        # Active stop-loss / take-profit levels per open symbol:
        # symbol -> (stop_loss_price, take_profit_price, entry_side)
        self.active_stops: dict[str, tuple[float, float, Side]] = {}

    async def run(self, candles_df: pd.DataFrame, symbol: str) -> BacktestResult:
        """
        Runs the backtest loop over historical data.

        Execution model (no look-ahead): at bar ``i`` the strategy decides using
        indicators computed only through bar ``i-1``; entries then fill at bar
        ``i``'s OPEN. Exits are checked against bar ``i``'s High/Low range. The
        MockBroker is the single source of truth for positions and trades.

        Args:
            candles_df: DataFrame containing OHLCV data.
            symbol: Trading symbol.

        Returns:
            BacktestResult with trades, equity curve, and performance metrics.
        """
        logger.info(f"Starting backtest for {self.strategy.name} on {symbol}")

        if candles_df.empty:
            logger.warning("Empty dataframe provided to backtest engine.")
            return self._empty_result(symbol)

        start_date = candles_df.index.min()
        end_date = candles_df.index.max()

        # Internal state tracking
        candles_history: list[Candle] = []
        equity_curve: list[float] = [self.initial_capital]
        # Indicators computed through the PREVIOUS bar (None until enough history).
        prev_indicators = None

        for i in range(len(candles_df)):
            current_row = candles_df.iloc[i]

            # Construct Candle object for current row
            current_candle = Candle(
                symbol=symbol,
                timestamp=current_row.name if isinstance(current_row.name, datetime) else pd.to_datetime(current_row.get('timestamp', datetime.now())),
                open=current_row['open'],
                high=current_row['high'],
                low=current_row['low'],
                close=current_row['close'],
                volume=current_row['volume'],
            )
            self.broker.current_time = current_candle.timestamp

            # 1. Manage exits first: check if the open position's SL/TP was touched
            #    within this bar's High/Low range and close it if so.
            await self._manage_exits(symbol, current_candle)

            # 2. Generate signals from indicators through the PREVIOUS bar and,
            #    if valid, enter at THIS bar's open (no same-bar look-ahead).
            if prev_indicators is not None and not self.circuit_breaker.is_active:
                if symbol not in self.broker.positions:
                    portfolio = await self.broker.get_portfolio()
                    signals = self.strategy.generate_signals(candles_history, prev_indicators, portfolio)
                    for signal in signals:
                        if signal.action not in (SignalAction.BUY, SignalAction.SELL):
                            continue
                        self.signals_generated += 1
                        await self._process_signal(signal, current_candle)
            elif self.circuit_breaker.is_active:
                logger.debug(f"Circuit breaker active for {symbol}. Trading halted.")

            # 3. Mark open position to this bar's close and update the circuit breaker.
            if symbol in self.broker.positions:
                self.broker.positions[symbol].current_price = current_candle.close
            portfolio = await self.broker.get_portfolio()
            self.circuit_breaker.check(portfolio)
            equity_curve.append(portfolio.equity)

            # 4. Append current bar to history and recompute indicators THROUGH bar i,
            #    to be used as `prev_indicators` on the next iteration.
            candles_history.append(current_candle)
            indicators = self.indicator_engine.calculate(candles_df.iloc[:i + 1])
            if len(indicators) >= 15:
                prev_indicators = self.indicator_engine.get_latest_indicators(indicators)

        # Post-loop: Close out any remaining open position at the last close.
        open_pos = self.broker.positions.get(symbol)
        if open_pos and open_pos.quantity > 0:
            logger.info("Closing open position at the end of backtest.")
            last_candle = candles_history[-1]
            await self._close_position(symbol, last_candle.close, open_pos.side, open_pos.quantity)

        # Compute metrics
        trades = self.broker.completed_trades
        metrics_engine = PerformanceMetrics(trades, self.initial_capital)
        portfolio = await self.broker.get_portfolio()
        final_equity = portfolio.equity
        
        result = BacktestResult(
            strategy_name=self.strategy.name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            trades=trades,
            metrics=metrics_engine.summary(),
            equity_curve=equity_curve,
            signals_generated=self.signals_generated,
            signals_executed=self.signals_executed,
            signals_rejected=self.signals_rejected
        )
        
        logger.info(f"Backtest completed for {self.strategy.name}. Final Equity: ₹{final_equity:,.2f}")
        return result

    async def _process_signal(self, signal: Signal, current_candle: Candle) -> None:
        """Validates, sizes, and places an entry order, filling at this bar's open."""
        portfolio = await self.broker.get_portfolio()
        trade_side = Side.BUY if signal.action == SignalAction.BUY else Side.SELL

        # Entry reference = this bar's OPEN (decision was made on the prior close).
        entry_ref = current_candle.open
        stop_pct = self.settings.risk.default_stop_loss_pct / 100.0
        rr = 2.0  # take-profit at 2x the stop distance (2:1 reward:risk)
        if trade_side == Side.BUY:
            stop_loss_price = entry_ref * (1.0 - stop_pct)
            take_profit_price = entry_ref * (1.0 + stop_pct * rr)
        else:
            stop_loss_price = entry_ref * (1.0 + stop_pct)
            take_profit_price = entry_ref * (1.0 - stop_pct * rr)

        qty = self.position_sizer.calculate_size(
            method=self.settings.risk.position_sizing,
            capital=portfolio.current_capital,
            risk_per_trade_pct=self.settings.risk.risk_per_trade_pct,
            entry_price=entry_ref,
            stop_loss_price=stop_loss_price,
        )
        if qty <= 0:
            self.signals_rejected += 1
            return

        order = Order(
            symbol=signal.symbol,
            side=trade_side,
            order_type=OrderType.MARKET,
            quantity=qty,
            price=entry_ref,
            position_type=PositionType.INTRADAY,
        )

        is_valid, reason = self.order_validator.validate(order, portfolio)
        if not is_valid:
            logger.debug(f"Signal rejected by validator: {reason}")
            self.signals_rejected += 1
            return

        # Fill the market entry at this bar's open.
        self.broker.last_prices[signal.symbol] = entry_ref
        filled_order = await self.broker.place_order(order)
        if filled_order and filled_order.filled_quantity > 0:
            self.signals_executed += 1
            self.active_stops[signal.symbol] = (stop_loss_price, take_profit_price, trade_side)
        else:
            self.signals_rejected += 1

    async def _manage_exits(self, symbol: str, candle: Candle) -> None:
        """Close the open position if its stop-loss or take-profit was touched this bar."""
        pos = self.broker.positions.get(symbol)
        if pos is None or symbol not in self.active_stops:
            return

        stop_loss, take_profit, side = self.active_stops[symbol]
        exit_price = None
        # Pessimistic ordering: assume the stop is hit before the target if both
        # fall inside the bar's range.
        if side == Side.BUY:
            if candle.low <= stop_loss:
                exit_price = stop_loss
            elif candle.high >= take_profit:
                exit_price = take_profit
        else:  # short
            if candle.high >= stop_loss:
                exit_price = stop_loss
            elif candle.low <= take_profit:
                exit_price = take_profit

        if exit_price is not None:
            await self._close_position(symbol, exit_price, side, pos.quantity)
            self.active_stops.pop(symbol, None)

    async def _close_position(self, symbol: str, price: float, entry_side: Side, quantity: int) -> None:
        """Place a market order to flatten an open position at the given price."""
        close_side = Side.SELL if entry_side == Side.BUY else Side.BUY
        close_order = Order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            position_type=PositionType.INTRADAY,
        )
        self.broker.last_prices[symbol] = price
        await self.broker.place_order(close_order)

    def _empty_result(self, symbol: str) -> BacktestResult:
        """Returns an empty result when no data is provided."""
        return BacktestResult(
            strategy_name=self.strategy.name,
            symbol=symbol,
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow(),
            initial_capital=self.initial_capital,
            final_equity=self.initial_capital,
            trades=[],
            metrics={},
            equity_curve=[self.initial_capital],
            signals_generated=0,
            signals_executed=0,
            signals_rejected=0
        )
