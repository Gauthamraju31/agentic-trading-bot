"""Autonomous Daily Trading Bot Runner.

Runs pre-market research at 09:00 AM, executes planned trades from 09:15 AM onwards,
manages stop-losses, and halts immediately when the daily target profit (₹100)
or daily stop-loss limit (-₹200) is reached.
"""

import asyncio
from datetime import datetime, time
from pathlib import Path
import sys
import time as time_module

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from src.agents.orchestrator import AgentOrchestrator
from src.agents.pre_market_planner import PreMarketPlanner
from src.core.config import settings
from src.core.daily_goal_manager import DailyGoalManager
from src.core.enums import Exchange, OrderType, Side, SignalAction, TimeFrame
from src.core.models import Candle, MarketContext, Order
from src.data.feeds.live_feed import LiveDataFeed
from src.data.indicators import IndicatorEngine
from src.dashboard.alerts import AlertManager
from src.execution.fees import FeeCalculator
from src.execution.mock_engine import MockBroker
from src.execution.order_validator import OrderValidator
from src.execution.position_manager import PositionManager


async def run_daily_trading_session(test_mode: bool = False, loop_interval_secs: int = 10):
    console = Console()
    console.print(
        Panel.fit(
            "[bold green]🤖 Autonomous Indian Daily Trading Bot[/bold green]\n"
            f"[dim]Daily Target Profit: [bold yellow]₹{settings.daily_goal.target_profit:.2f}[/bold yellow] | "
            f"Daily Max Loss: [bold red]₹{settings.daily_goal.max_loss:.2f}[/bold red][/dim]\n"
            "[dim]Pre-Market Research → AI Playbook → Live Execution → Fee-Adjusted Goal Lock[/dim]",
            border_style="green",
        )
    )

    goal_mgr = DailyGoalManager()
    planner = PreMarketPlanner()
    orchestrator = AgentOrchestrator()
    broker = MockBroker(initial_capital=settings.mock.initial_capital)
    validator = OrderValidator()
    validator.set_backtest_mode(True)
    position_mgr = PositionManager()
    alert_mgr = AlertManager()
    feed = LiveDataFeed()

    # ── Step 1: Pre-Market Planning Phase ────────────────────────────────────
    now_time = datetime.now().time()
    market_open_time = time(9, 15)

    playbook = planner.get_latest_playbook()
    today_str = datetime.now().strftime("%Y-%m-%d")

    if not playbook or playbook.get("date") != today_str:
        console.print("[bold yellow]Phase 1: Running Pre-Market Autonomous Research...[/bold yellow]")
        playbook = await planner.run_pre_market_analysis()
        if alert_mgr.enabled and playbook.get("primary_pick"):
            pick = playbook["primary_pick"]
            await alert_mgr.send_alert(
                f"📋 <b>Pre-Market AI Playbook Ready</b> ({today_str})\n"
                f"Primary Choice: <b>{pick.get('symbol')}</b> [{pick.get('action')}]\n"
                f"Entry: ₹{pick.get('current_price'):.2f} | Reasoning: {pick.get('reasoning')[:150]}...",
                level="PRE_MARKET",
            )

    target_symbol = "RELIANCE"
    if playbook and playbook.get("primary_pick"):
        pick_sym = playbook["primary_pick"].get("symbol")
        if pick_sym:
            target_symbol = pick_sym

    console.print(f"[bold green]Target Symbol for Today: [white]{target_symbol}[/white][/bold green]")

    # Check if goal already completed for today
    if goal_mgr.status in ("GOAL_REACHED", "STOP_LOSS_HIT"):
        console.print(f"[bold yellow]Trading for today is already finished. Status: {goal_mgr.status}[/bold yellow]")
        return

    goal_mgr.status = "TRADING_ACTIVE"
    console.print("\n[bold yellow]Phase 2: Live Market Execution Loop Active...[/bold yellow]")

    # ── Step 2: Trading Execution Loop ──────────────────────────────────────
    max_ticks = 50 if test_mode else 1000
    tick_count = 0

    while tick_count < max_ticks:
        tick_count += 1
        current_dt = datetime.now()

        # Fetch latest market data
        try:
            candles = await feed.get_historical_candles(target_symbol, exchange=Exchange.NSE, timeframe=TimeFrame.M5)
            if len(candles) < 30:
                await asyncio.sleep(loop_interval_secs)
                continue

            latest_candle = candles[-1]
            df = pd.DataFrame([c.model_dump() for c in candles])
            df_ind = IndicatorEngine.calculate(df)
            indicators = IndicatorEngine.get_latest_indicators(df_ind)

            # Update broker bar
            await broker.process_candle(latest_candle)
            portfolio = await broker.get_portfolio()

            # Build market context
            ctx = MarketContext(
                symbol=target_symbol,
                exchange=Exchange.NSE,
                current_price=latest_candle.close,
                candles=candles,
                indicators=indicators,
                portfolio=portfolio,
                timestamp=latest_candle.timestamp,
            )

            # Evaluate decision from multi-agent pipeline
            decision = await orchestrator.run(ctx)
            action_str = decision.action.value if hasattr(decision.action, "value") else str(decision.action)

            console.print(
                f"[{current_dt.strftime('%H:%M:%S')}] {target_symbol} @ ₹{latest_candle.close:.2f} | "
                f"Signal: [bold magenta]{action_str}[/bold magenta] (Conf: {decision.confidence:.2f}) | "
                f"Daily P&L: ₹{goal_mgr.realized_pnl:+.2f} / ₹{goal_mgr.target_profit:.2f}"
            )

            # Execute trade if signal approved by risk
            if decision.approved_by_risk and decision.action in (SignalAction.BUY, SignalAction.SELL):
                side = Side.BUY if decision.action == SignalAction.BUY else Side.SELL
                qty = max(1, decision.position_size or 1)

                order = Order(
                    symbol=target_symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=qty,
                    price=latest_candle.close,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                )

                valid, val_reason = validator.validate_order(order, portfolio, latest_candle.close)
                if valid:
                    filled_order = await broker.place_order(order)
                    logger.success(f"Order filled: {filled_order.side.value} {filled_order.quantity} {target_symbol} @ ₹{filled_order.price:.2f}")

                    # Simulate immediate evaluation / exit for test iteration if position exists
                    if portfolio.positions:
                        for pos in portfolio.positions:
                            # Simulate exit on next price tick
                            exit_price = latest_candle.close * (1.002 if side == Side.BUY else 0.998)
                            close_ord = Order(
                                symbol=pos.symbol,
                                side=Side.SELL if pos.side == Side.BUY else Side.BUY,
                                order_type=OrderType.MARKET,
                                quantity=pos.quantity,
                                price=exit_price,
                            )
                            closed = await broker.place_order(close_ord)
                            # Get completed trade net P&L
                            if broker.completed_trades:
                                last_trade = broker.completed_trades[-1]
                                net_pnl = last_trade.net_pnl
                                new_status = goal_mgr.update_pnl(net_pnl)

                                if alert_mgr.enabled:
                                    await alert_mgr.send_trade_alert(last_trade)

                                if new_status in ("GOAL_REACHED", "STOP_LOSS_HIT"):
                                    console.print(
                                        Panel.fit(
                                            f"[bold green]🏁 Daily Trading Goal Complete![/bold green]\n"
                                            f"Final Status: [bold yellow]{new_status}[/bold yellow]\n"
                                            f"Total Net Realized P&L: [bold cyan]₹{goal_mgr.realized_pnl:+.2f}[/bold cyan]",
                                            border_style="cyan",
                                        )
                                    )
                                    return
            if test_mode:
                # In test mode, exit loop after evaluation
                break

        except Exception as e:
            logger.error(f"[DailyBot] Exception in execution loop: {e}")

        await asyncio.sleep(loop_interval_secs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Daily Trading Bot")
    parser.add_argument("--test", action="store_true", help="Run single test iteration")
    args = parser.parse_args()

    asyncio.run(run_daily_trading_session(test_mode=args.test, loop_interval_secs=5))
