"""Main execution script for running the agentic trading bot."""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import sys
import threading
import time

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
import pandas as pd

from src.agents.orchestrator import AgentOrchestrator
from src.core.config import settings
from src.core.enums import Exchange, Side, SignalAction, TimeFrame
from src.core.models import MarketContext
from src.dashboard.alerts import AlertManager
from src.dashboard.app import start_server
from src.data.feeds.csv_feed import CSVDataFeed
from src.data.indicators import IndicatorEngine
from src.execution.mock_engine import MockBroker
from src.execution.order_validator import OrderValidator
from src.execution.position_manager import PositionManager
from src.risk.circuit_breaker import CircuitBreaker


async def main_loop(symbol: str, interval: int, mode: str, max_ticks: int = 0, use_live_data: bool = False):
    """Main trading bot loop."""
    logger.info(f"Initializing Trading Bot in {mode} mode for {symbol}")

    alert_manager = AlertManager()
    await alert_manager.send_alert(
        f"Trading bot started in {mode} mode for {symbol}", level="INFO"
    )

    from src.data.feeds.live_feed import LiveDataFeed

    if use_live_data or mode in ("live", "mock_live"):
        logger.info(f"Connecting to LIVE NSE market data feed for {symbol}...")
        data_feed = LiveDataFeed()
        candles = await data_feed.get_historical_candles(symbol, Exchange.NSE, TimeFrame.M5)
    else:
        feed_path = Path("data/historical") / f"{symbol.replace(' ', '_')}_5m.csv"
        if not feed_path.exists():
            from scripts.download_data import generate_random_walk_candles
            df_gen = generate_random_walk_candles(symbol, datetime.now(), 30)
            feed_path.parent.mkdir(parents=True, exist_ok=True)
            df_gen.to_csv(feed_path, index=False)
        data_feed = CSVDataFeed(feed_path)
        candles = await data_feed.get_historical_candles(symbol, Exchange.NSE, TimeFrame.M5)

    broker = MockBroker(initial_capital=settings.mock.initial_capital)
    orchestrator = AgentOrchestrator()
    position_manager = PositionManager()
    order_validator = OrderValidator()
    circuit_breaker = CircuitBreaker()

    if mode in ("mock", "mock_live"):
        order_validator.set_backtest_mode(True)

    logger.info(f"Loaded {len(candles)} candles for {symbol}")

    tick_count = 0
    window_size = 50  # minimum history window for indicators

    logger.info(f"Starting main trading loop (tick interval: {interval}s)")

    for idx in range(window_size, len(candles)):
        tick_count += 1
        if max_ticks > 0 and tick_count > max_ticks:
            logger.info(f"Reached max ticks ({max_ticks}). Halting loop.")
            break

        current_candle = candles[idx]
        history_candles = candles[idx - window_size : idx + 1]

        # 1. Check circuit breaker
        portfolio = await broker.get_portfolio()
        is_circuit_broken, reason = circuit_breaker.check(portfolio)
        if is_circuit_broken:
            logger.error(f"CIRCUIT BREAKER ACTIVE: {reason}")
            await alert_manager.send_circuit_breaker_alert(reason)
            break

        # 2. Process mock broker bar update
        await broker.process_candle(current_candle)

        # 3. Calculate indicators
        df_history = pd.DataFrame([c.model_dump() for c in history_candles])
        df_with_ind = IndicatorEngine.calculate(df_history)
        latest_indicators = IndicatorEngine.get_latest_indicators(df_with_ind)

        # 4. Build Market Context
        context = MarketContext(
            symbol=symbol,
            exchange=Exchange.NSE,
            current_price=current_candle.close,
            candles=history_candles,
            indicators=latest_indicators,
            portfolio=portfolio,
            timestamp=current_candle.timestamp,
        )

        # 5. Run Multi-Agent Decision Pipeline
        decision = await orchestrator.run(context)
        logger.info(
            f"Tick #{tick_count} | Candle: {current_candle.timestamp} | Price: ₹{current_candle.close:.2f} | Action: {decision.action.value} (Conf: {decision.confidence:.2f})"
        )
        for op in decision.agent_opinions:
            logger.info(f"  └─ [{op.agent_role.value.upper()}] Action: {op.action.value} ({op.confidence*100:.0f}%) | {op.reasoning}")

        # 6. Execute Order if Decision is BUY/SELL/EXIT
        if decision.approved_by_risk and decision.action in (SignalAction.BUY, SignalAction.SELL):
            side = Side.BUY if decision.action == SignalAction.BUY else Side.SELL
            qty = decision.position_size or 10

            from src.core.models import Order
            from src.core.enums import OrderType, PositionType

            order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=qty,
                price=current_candle.close,
                position_type=PositionType.DELIVERY,
            )

            is_valid, reject_reason = order_validator.validate(order, portfolio)
            if is_valid:
                filled_order = await broker.place_order(order)
                if filled_order.is_terminal and filled_order.filled_quantity > 0:
                    position = position_manager.open_position(filled_order)
                    logger.info(f"✅ Order executed: {side.value} {qty} x {symbol} @ ₹{filled_order.average_price:.2f}")
            else:
                logger.warning(f"Order rejected by validator: {reject_reason}")

        await asyncio.sleep(interval)

    logger.info("Trading bot loop completed gracefully.")


def main():
    parser = argparse.ArgumentParser(description="Run the main trading bot.")
    parser.add_argument("--mode", type=str, choices=["mock", "live"], default="mock", help="Execution mode")
    parser.add_argument("--symbol", type=str, default="RELIANCE", help="Symbol to trade")
    parser.add_argument("--interval", type=int, default=1, help="Tick interval in seconds")
    parser.add_argument("--max-ticks", type=int, default=10, help="Max ticks to run (0 for infinite)")
    parser.add_argument("--live-data", action="store_true", help="Use live market data from NSE")
    parser.add_argument("--with-dashboard", action="store_true", help="Start the web dashboard")

    args = parser.parse_args()

    if args.with_dashboard:
        logger.info("Starting dashboard server in background...")
        dashboard_thread = threading.Thread(target=start_server, daemon=True)
        dashboard_thread.start()
        time.sleep(1)

    try:
        asyncio.run(main_loop(args.symbol, args.interval, args.mode, args.max_ticks, use_live_data=args.live_data))
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")


if __name__ == "__main__":
    main()
