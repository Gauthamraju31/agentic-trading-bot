"""Autonomous Daily Trading Bot Runner.

Architecture:
- Phase 1 (08:45 - 09:15 IST): Pre-market AI planner runs once to analyze macro cues, VIX, news, and select target stocks.
- Phase 2 (09:15 - 15:30 IST): Deterministic quantitative execution engine evaluates market bars, manages positions,
  trailing stops, and risk limits (sub-second execution, zero LLM quota burn).
- Phase 3 (15:30 - 15:45 IST): Auto-squareoff all open intraday positions and update daily goals.
- Idle Phase: Outside market hours and weekends, daemon stays alive, sleeping with quiet periodic heartbeats.
"""

import argparse
import asyncio
from datetime import datetime, time, timedelta
from pathlib import Path
import sys

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from src.agents.pre_market_planner import PreMarketPlanner
from src.core.config import settings
from src.core.daily_goal_manager import DailyGoalManager
from src.core.enums import Exchange, OrderType, PositionType, Side, SignalAction, TimeFrame
from src.core.market_hours import get_ist_now, get_market_status, is_trading_day
from src.core.models import Candle, Order
from src.dashboard.alerts import AlertManager
from src.data.feeds.live_feed import LiveDataFeed
from src.data.indicators import IndicatorEngine
from src.execution.fees import FeeCalculator
from src.execution.mock_engine import MockBroker
from src.execution.order_validator import OrderValidator
from src.execution.position_manager import PositionManager
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossManager
from src.strategy.momentum import MomentumStrategy


class AutonomousTradingBot:
    def __init__(self, test_mode: bool = False, loop_interval_secs: int = 15):
        self.test_mode = test_mode
        self.loop_interval = loop_interval_secs
        self.console = Console()

        self.goal_mgr = DailyGoalManager()
        self.planner = PreMarketPlanner()
        self.broker = MockBroker(initial_capital=settings.mock.initial_capital)
        self.validator = OrderValidator()
        self.validator.set_backtest_mode(True)  # allows simulated execution
        self.position_mgr = PositionManager()
        self.position_sizer = PositionSizer()
        self.stop_loss_mgr = StopLossManager()
        self.circuit_breaker = CircuitBreaker()
        self.indicator_engine = IndicatorEngine()
        self.alert_mgr = AlertManager()
        self.feed = LiveDataFeed()
        self.strategy = MomentumStrategy()

        # Active stops per symbol: symbol -> (stop_loss_price, take_profit_price, trailing_stop, side)
        self.active_stops: dict[str, dict] = {}
        self.target_symbol: str = "INFY"
        self.pre_market_done_for_date: str = ""

    def log_banner(self):
        now_ist = get_ist_now()
        mkt_status = get_market_status()
        self.console.print(
            Panel.fit(
                "[bold green]🤖 Autonomous Indian Daily Trading Bot[/bold green]\n"
                f"[dim]Timezone: [bold cyan]IST (Asia/Kolkata)[/bold cyan] | Current Time: [bold white]{now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}[/bold white][/dim]\n"
                f"[dim]Market Status: [bold yellow]{mkt_status['status']}[/bold yellow] ({mkt_status['reason']})[/dim]\n"
                f"[dim]Capital: [bold green]₹{self.broker.current_capital:,.2f}[/bold green] | "
                f"Daily Goal: [bold yellow]₹{self.goal_mgr.target_profit:.2f}[/bold yellow] | "
                f"Max Loss: [bold red]₹{self.goal_mgr.max_loss:.2f}[/bold red][/dim]",
                border_style="green",
            )
        )

    async def run_pre_market_phase(self, today_str: str):
        """Runs AI pre-market analysis once per day at ~08:45 - 09:15 IST."""
        if self.pre_market_done_for_date == today_str:
            return

        self.console.print("[bold yellow]Phase 1: Running Pre-Market Autonomous Research...[/bold yellow]")
        playbook = self.planner.get_latest_playbook()

        if not playbook or playbook.get("date") != today_str:
            try:
                playbook = await self.planner.run_pre_market_analysis()
            except Exception as e:
                logger.error(f"[PreMarket] Analysis error: {e}. Falling back to default list.")
                playbook = None

        if playbook and playbook.get("primary_pick"):
            pick = playbook["primary_pick"]
            pick_sym = pick.get("symbol")
            if pick_sym:
                self.target_symbol = pick_sym
            if self.alert_mgr.enabled:
                await self.alert_mgr.send_alert(
                    f"📋 <b>Pre-Market AI Playbook Ready</b> ({today_str})\n"
                    f"Primary Choice: <b>{pick.get('symbol')}</b> [{pick.get('action')}]\n"
                    f"Reasoning: {pick.get('reasoning', '')[:150]}...",
                    level="PRE_MARKET",
                )

        self.pre_market_done_for_date = today_str
        logger.info(f"[PreMarket] Target Symbol for Today: {self.target_symbol}")

    async def manage_open_positions(self, candle: Candle):
        """Check SL / TP and update trailing stops for open positions."""
        symbol = candle.symbol
        pos = self.broker.positions.get(symbol)
        if not pos or symbol not in self.active_stops:
            return

        stop_info = self.active_stops[symbol]
        stop_loss = stop_info["stop_loss"]
        take_profit = stop_info["take_profit"]
        side = stop_info["side"]

        exit_price = None
        exit_reason = None

        if side == Side.BUY:
            if candle.low <= stop_loss:
                exit_price = stop_loss
                exit_reason = "STOP_LOSS_HIT"
            elif candle.high >= take_profit:
                exit_price = take_profit
                exit_reason = "TAKE_PROFIT_HIT"
        else:  # SELL
            if candle.high >= stop_loss:
                exit_price = stop_loss
                exit_reason = "STOP_LOSS_HIT"
            elif candle.low <= take_profit:
                exit_price = take_profit
                exit_reason = "TAKE_PROFIT_HIT"

        if exit_price is not None:
            logger.info(f"[PositionManager] {exit_reason} for {symbol} at ₹{exit_price:.2f}")
            close_ord = Order(
                symbol=symbol,
                side=Side.SELL if side == Side.BUY else Side.BUY,
                order_type=OrderType.MARKET,
                quantity=pos.quantity,
                price=exit_price,
                position_type=PositionType.INTRADAY,
            )
            self.broker.last_prices[symbol] = exit_price
            await self.broker.place_order(close_ord)
            self.active_stops.pop(symbol, None)

            if self.broker.completed_trades:
                last_trade = self.broker.completed_trades[-1]
                new_status = self.goal_mgr.update_pnl(last_trade.net_pnl)
                if self.alert_mgr.enabled:
                    await self.alert_mgr.send_trade_alert(last_trade)
                logger.info(f"[DailyGoal] Trade closed: Net P&L = ₹{last_trade.net_pnl:+.2f} | Status = {new_status}")

    async def execute_trading_tick(self):
        """Fetches latest live data, computes technical signals, executes orders if valid."""
        target_symbol = self.target_symbol
        candles = await self.feed.get_historical_candles(target_symbol, Exchange.NSE, TimeFrame.M5)

        if not candles or len(candles) < 28:
            logger.warning(f"Insufficient candle data for {target_symbol} ({len(candles) if candles else 0} bars)")
            return

        latest_candle = candles[-1]
        await self.broker.process_candle(latest_candle)

        # Check existing position exit conditions first
        await self.manage_open_positions(latest_candle)

        # If goal reached or max loss hit, stop new orders
        if self.goal_mgr.status in ("GOAL_REACHED", "STOP_LOSS_HIT"):
            return

        # Check if already in a position for this symbol
        portfolio = await self.broker.get_portfolio()
        if target_symbol in self.broker.positions:
            pos = self.broker.positions[target_symbol]
            current_pnl = (latest_candle.close - pos.entry_price) * pos.quantity if pos.side == Side.BUY else (pos.entry_price - latest_candle.close) * pos.quantity
            self.console.print(
                f"[{get_ist_now().strftime('%H:%M:%S')}] HOLDING {pos.quantity}x {target_symbol} @ ₹{pos.entry_price:.2f} | "
                f"LTP: ₹{latest_candle.close:.2f} | Open P&L: ₹{current_pnl:+.2f} | Daily Realized: ₹{self.goal_mgr.realized_pnl:+.2f}"
            )
            return

        # Compute technical indicators
        df = pd.DataFrame([c.model_dump() for c in candles])
        df_ind = self.indicator_engine.calculate(df)
        indicators = self.indicator_engine.get_latest_indicators(df_ind)

        if not indicators:
            return

        signals = self.strategy.generate_signals(candles, indicators, portfolio)
        signal_action = "HOLD"
        confidence = 0.0

        if signals:
            active_sig = signals[-1]
            signal_action = active_sig.action.value
            confidence = active_sig.confidence

            # Size and validate order
            trade_side = Side.BUY if active_sig.action == SignalAction.BUY else Side.SELL
            entry_price = latest_candle.close
            stop_pct = settings.risk.default_stop_loss_pct / 100.0
            rr = 2.0  # 2:1 Reward to Risk

            stop_loss = entry_price * (1.0 - stop_pct) if trade_side == Side.BUY else entry_price * (1.0 + stop_pct)
            take_profit = entry_price * (1.0 + stop_pct * rr) if trade_side == Side.BUY else entry_price * (1.0 - stop_pct * rr)

            qty = self.position_sizer.calculate_size(
                method=settings.risk.position_sizing,
                capital=portfolio.current_capital,
                risk_per_trade_pct=settings.risk.risk_per_trade_pct,
                entry_price=entry_price,
                stop_loss_price=stop_loss,
            )

            if qty > 0:
                order = Order(
                    symbol=target_symbol,
                    side=trade_side,
                    order_type=OrderType.MARKET,
                    quantity=qty,
                    price=entry_price,
                    position_type=PositionType.INTRADAY,
                )

                # Correct validation call
                is_valid, reason = self.validator.validate(order, portfolio)
                if is_valid:
                    self.broker.last_prices[target_symbol] = entry_price
                    filled = await self.broker.place_order(order)
                    if filled and filled.status.value == "FILLED":
                        self.active_stops[target_symbol] = {
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "side": trade_side,
                        }
                        logger.success(f"ORDER FILLED: {trade_side.value} {qty} {target_symbol} @ ₹{entry_price:.2f} (SL: ₹{stop_loss:.2f}, TP: ₹{take_profit:.2f})")
                        if self.alert_mgr.enabled:
                            await self.alert_mgr.send_alert(f"🚀 Executed {trade_side.value} {qty} {target_symbol} @ ₹{entry_price:.2f}")
                else:
                    logger.debug(f"Order rejected by validator: {reason}")

        self.console.print(
            f"[{get_ist_now().strftime('%H:%M:%S')}] {target_symbol} @ ₹{latest_candle.close:.2f} | "
            f"Signal: [bold magenta]{signal_action}[/bold magenta] (Conf: {confidence:.2f}) | "
            f"Daily P&L: ₹{self.goal_mgr.realized_pnl:+.2f} / ₹{self.goal_mgr.target_profit:.2f}"
        )

    async def auto_squareoff(self):
        """Auto-squareoff remaining open intraday positions before market close."""
        for sym, pos in list(self.broker.positions.items()):
            ltp = self.broker.last_prices.get(sym, pos.entry_price)
            close_ord = Order(
                symbol=sym,
                side=Side.SELL if pos.side == Side.BUY else Side.BUY,
                order_type=OrderType.MARKET,
                quantity=pos.quantity,
                price=ltp,
                position_type=PositionType.INTRADAY,
            )
            await self.broker.place_order(close_ord)
            logger.info(f"[SquareOff] Flattened position {sym} at ₹{ltp:.2f}")
        self.active_stops.clear()

    async def run(self):
        """Main autonomous daemon loop."""
        self.log_banner()

        if self.test_mode:
            logger.info("Running single test mode iteration...")
            await self.execute_trading_tick()
            self.console.print("[bold green]✔ Test iteration completed successfully.[/bold green]")
            return

        while True:
            try:
                now_ist = get_ist_now()
                current_time = now_ist.time()
                today_str = now_ist.strftime("%Y-%m-%d")
                mkt_status = get_market_status()

                # Weekend / Holiday Check
                if not mkt_status["is_trading_day"]:
                    logger.info(f"[Daemon] Market closed ({mkt_status['reason']}). Sleeping for 30 minutes...")
                    await asyncio.sleep(1800)
                    continue

                # Pre-Market Window: 08:45 - 09:14 IST
                if time(8, 45) <= current_time < time(9, 15):
                    await self.run_pre_market_phase(today_str)
                    await asyncio.sleep(60)
                    continue

                # Live Market Hours: 09:15 - 15:20 IST
                elif time(9, 15) <= current_time < time(15, 20):
                    await self.execute_trading_tick()

                # Square-off Window: 15:20 - 15:30 IST
                elif time(15, 20) <= current_time < time(15, 30):
                    if self.broker.positions:
                        logger.warning("[Daemon] Approaching 15:30 IST market close. Squaring off open positions...")
                        await self.auto_squareoff()
                    await asyncio.sleep(60)
                    continue

                # Post-Market / Overnight: After 15:30 IST
                else:
                    if self.broker.positions:
                        await self.auto_squareoff()
                    logger.info(f"[Daemon] Market closed for today. Realized P&L: ₹{self.goal_mgr.realized_pnl:+.2f}. Resting...")
                    await asyncio.sleep(600)
                    continue

            except Exception as e:
                logger.error(f"[DailyBot] Unhandled error in execution cycle: {e}")

            await asyncio.sleep(self.loop_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Daily Trading Bot")
    parser.add_argument("--test", action="store_true", help="Run single test iteration and exit")
    parser.add_argument("--interval", type=int, default=15, help="Loop interval in seconds")
    args = parser.parse_args()

    bot = AutonomousTradingBot(test_mode=args.test, loop_interval_secs=args.interval)
    asyncio.run(bot.run())
