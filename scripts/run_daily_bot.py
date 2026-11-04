"""Autonomous Daily Trading Bot Runner.

Architecture:
- Phase 1 (08:45 - 09:15 IST): Pre-market AI planner & multi-stock scanner select prime watchlist.
- Phase 2 (09:15 - 15:20 IST): Real-time execution engine evaluates NIFTY market regime, scans watchlist,
  executes deterministic technical signals, and manages dynamic ATR trailing stops.
- Phase 3 (15:20 - 15:30 IST): Auto-squareoff all open intraday positions before market close.
- Phase 4 (15:45 - 16:00 IST): Post-market reflection review logs trade lessons into persistent memory.
- Idle Phase: Outside market hours and weekends, daemon stays alive, sleeping with quiet periodic heartbeats.
"""

import argparse
import asyncio
from datetime import datetime, time, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time as time_mod
from typing import Dict, List, Optional

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
import pandas as pd
from rich.console import Console
from rich.panel import Panel

# Configure persistent file logging for dashboard & audit
Path("data").mkdir(exist_ok=True)
logger.add("data/bot.log", rotation="10 MB", retention="5 days", level="INFO", enqueue=True)

from src.agents.pre_market_planner import PreMarketPlanner
from src.agents.trade_memory import TradeReflectionMemory
from src.core.config import settings
from src.core.daily_goal_manager import DailyGoalManager
from src.core.enums import Exchange, OrderType, PositionType, Side, SignalAction, TimeFrame
from src.core.market_hours import get_ist_now, get_market_status, is_trading_day
from src.core.models import Candle, Order
from src.dashboard.alerts import AlertManager
from src.data.feeds.live_feed import LiveDataFeed
from src.data.indicators import IndicatorEngine
from src.data.market_regime import MarketRegime, MarketRegimeDetector
from src.data.scanner import MarketScanner
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
        self.scanner = MarketScanner()
        self.regime_detector = MarketRegimeDetector(cache_ttl_seconds=60)
        self.reflection_memory = TradeReflectionMemory()

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

        # Multi-stock watchlist & execution state
        self.watchlist: List[str] = ["INFY", "TCS", "RELIANCE"]
        self.max_open_positions: int = 1
        self.active_stops: Dict[str, Dict] = {}
        self.pre_market_done_for_date: str = ""
        self.post_market_done_for_date: str = ""

    def update_heartbeat(self, status: str = "RUNNING", sleep_interval: int = 15):
        try:
            Path("data").mkdir(exist_ok=True)
            with open("data/bot_heartbeat.json", "w") as f:
                json.dump({
                    "status": status,
                    "pid": os.getpid(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "last_seen_epoch": time_mod.time(),
                    "sleep_interval": sleep_interval,
                    "realized_pnl": self.goal_mgr.realized_pnl,
                    "watchlist": self.watchlist,
                }, f)
        except Exception:
            pass

    def check_control_signal(self) -> str:
        try:
            ctrl_file = Path("data/bot_control.json")
            if ctrl_file.exists():
                with open(ctrl_file, "r") as f:
                    return json.load(f).get("action", "RESUME")
        except Exception:
            pass
        return "RESUME"

    def log_banner(self):
        now_ist = get_ist_now()
        mkt_status = get_market_status()
        self.console.print(
            Panel.fit(
                "[bold green]🤖 Autonomous Indian Daily Trading Bot (Phase 2)[/bold green]\n"
                f"[dim]Timezone: [bold cyan]IST (Asia/Kolkata)[/bold cyan] | Current Time: [bold white]{now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}[/bold white][/dim]\n"
                f"[dim]Market Status: [bold yellow]{mkt_status['status']}[/bold yellow] ({mkt_status['reason']})[/dim]\n"
                f"[dim]Capital: [bold green]₹{self.broker.current_capital:,.2f}[/bold green] | "
                f"Daily Goal: [bold yellow]₹{self.goal_mgr.target_profit:.2f}[/bold yellow] | "
                f"Max Loss: [bold red]₹{self.goal_mgr.max_loss:.2f}[/bold red][/dim]\n"
                f"[dim]Active Watchlist: [bold cyan]{', '.join(self.watchlist)}[/bold cyan][/dim]",
                border_style="green",
            )
        )

    async def run_pre_market_phase(self, today_str: str):
        """Runs AI pre-market analysis and multi-stock scan once per day at ~08:45 - 09:15 IST."""
        if self.pre_market_done_for_date == today_str:
            return

        self.console.print("[bold yellow]Phase 1: Running Pre-Market Autonomous Research & Multi-Stock Scanner...[/bold yellow]")

        # 1. Multi-Stock Opportunity Scan
        try:
            top_watchlist = await self.scanner.get_top_watchlist(n=3)
            if top_watchlist:
                self.watchlist = top_watchlist
                logger.info(f"[PreMarket] Multi-Stock Scanner selected watchlist: {self.watchlist}")
        except Exception as e:
            logger.warning(f"[PreMarket] Multi-stock scan error: {e}. Keeping current watchlist.")

        # 2. AI Pre-market Planner (Macro & News)
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
            if pick_sym and pick_sym not in self.watchlist:
                self.watchlist.insert(0, pick_sym)
                self.watchlist = self.watchlist[:3]

            if self.alert_mgr.enabled:
                await self.alert_mgr.send_alert(
                    f"📋 <b>Pre-Market Playbook Ready</b> ({today_str})\n"
                    f"Primary Pick: <b>{pick.get('symbol')}</b> [{pick.get('action')}]\n"
                    f"Watchlist: {', '.join(self.watchlist)}\n"
                    f"Reasoning: {pick.get('reasoning', '')[:120]}...",
                    level="PRE_MARKET",
                )

        self.pre_market_done_for_date = today_str
        logger.info(f"[PreMarket] Ready for market open with Watchlist: {self.watchlist}")

    async def manage_open_positions(self, candle: Candle):
        """Check SL / TP and dynamically advance ATR trailing stops for open positions."""
        symbol = candle.symbol
        pos = self.broker.positions.get(symbol)
        if not pos or symbol not in self.active_stops:
            return

        stop_info = self.active_stops[symbol]
        entry_price = stop_info["entry_price"]
        stop_loss = stop_info["stop_loss"]
        take_profit = stop_info["take_profit"]
        stop_dist = stop_info.get("stop_dist", abs(entry_price - stop_loss))
        side = stop_info["side"]

        exit_price = None
        exit_reason = None

        # 1. Update Trailing Stop Milestones
        if side == Side.BUY:
            highest = max(stop_info.get("highest_price", entry_price), candle.high)
            stop_info["highest_price"] = highest

            # Milestone A: If price reached +1.0R, move SL to Breakeven
            if highest >= (entry_price + stop_dist) and stop_loss < entry_price:
                stop_info["stop_loss"] = entry_price
                stop_loss = entry_price
                logger.info(f"[TrailingStop] {symbol}: +1.0R reached! Trailing SL moved to Breakeven (₹{entry_price:.2f})")

            # Milestone B: If price reached +1.5R, lock in +0.75R profit
            elif highest >= (entry_price + 1.5 * stop_dist) and stop_loss < (entry_price + 0.75 * stop_dist):
                stop_info["stop_loss"] = entry_price + (0.75 * stop_dist)
                stop_loss = stop_info["stop_loss"]
                logger.info(f"[TrailingStop] {symbol}: +1.5R reached! Trailing SL moved to lock profit (₹{stop_loss:.2f})")

            # Check exits
            if candle.low <= stop_loss:
                exit_price = stop_loss
                exit_reason = "TRAILING_STOP_HIT" if stop_loss >= entry_price else "STOP_LOSS_HIT"
            elif candle.high >= take_profit:
                exit_price = take_profit
                exit_reason = "TAKE_PROFIT_HIT"

        else:  # SELL (Short)
            lowest = min(stop_info.get("lowest_price", entry_price), candle.low)
            stop_info["lowest_price"] = lowest

            # Milestone A: If price dropped -1.0R, move SL to Breakeven
            if lowest <= (entry_price - stop_dist) and stop_loss > entry_price:
                stop_info["stop_loss"] = entry_price
                stop_loss = entry_price
                logger.info(f"[TrailingStop] {symbol}: +1.0R reached! Trailing SL moved to Breakeven (₹{entry_price:.2f})")

            # Milestone B: If price dropped -1.5R, lock in +0.75R profit
            elif lowest <= (entry_price - 1.5 * stop_dist) and stop_loss > (entry_price - 0.75 * stop_dist):
                stop_info["stop_loss"] = entry_price - (0.75 * stop_dist)
                stop_loss = stop_info["stop_loss"]
                logger.info(f"[TrailingStop] {symbol}: +1.5R reached! Trailing SL moved to lock profit (₹{stop_loss:.2f})")

            # Check exits
            if candle.high >= stop_loss:
                exit_price = stop_loss
                exit_reason = "TRAILING_STOP_HIT" if stop_loss <= entry_price else "STOP_LOSS_HIT"
            elif candle.low <= take_profit:
                exit_price = take_profit
                exit_reason = "TAKE_PROFIT_HIT"

        # Execute exit if triggered
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
        """Fetches live data for active positions & watchlist, applies NIFTY regime filter, executes valid setups."""
        # 1. Manage open positions first
        for sym in list(self.broker.positions.keys()):
            pos_candles = await self.feed.get_historical_candles(sym, Exchange.NSE, TimeFrame.M5)
            if pos_candles:
                latest_pos_candle = pos_candles[-1]
                await self.broker.process_candle(latest_pos_candle)
                await self.manage_open_positions(latest_pos_candle)

        # 2. Check daily goal or circuit breaker
        if self.goal_mgr.status in ("GOAL_REACHED", "STOP_LOSS_HIT"):
            return

        portfolio = await self.broker.get_portfolio()

        # 3. If currently holding max positions, display status and hold
        if len(self.broker.positions) >= self.max_open_positions:
            for sym, pos in self.broker.positions.items():
                ltp = self.broker.last_prices.get(sym, pos.entry_price)
                open_pnl = (ltp - pos.entry_price) * pos.quantity if pos.side == Side.BUY else (pos.entry_price - ltp) * pos.quantity
                self.console.print(
                    f"[{get_ist_now().strftime('%H:%M:%S')}] HOLDING {pos.quantity}x {sym} @ ₹{pos.entry_price:.2f} | "
                    f"LTP: ₹{ltp:.2f} | Open P&L: ₹{open_pnl:+.2f} | Daily Realized: ₹{self.goal_mgr.realized_pnl:+.2f}"
                )
            return

        # 4. Check NIFTY 50 Index Market Regime Filter
        regime, regime_details = await self.regime_detector.get_current_regime()

        # 5. Scan Watchlist candidates for new entry opportunities
        for candidate_symbol in self.watchlist:
            if candidate_symbol in self.broker.positions:
                continue

            candles = await self.feed.get_historical_candles(candidate_symbol, Exchange.NSE, TimeFrame.M5)
            if not candles or len(candles) < 28:
                continue

            latest_candle = candles[-1]
            await self.broker.process_candle(latest_candle)

            df = pd.DataFrame([c.model_dump() for c in candles])
            df_ind = self.indicator_engine.calculate(df)
            indicators = self.indicator_engine.get_latest_indicators(df_ind)
            if not indicators:
                continue

            signals = self.strategy.generate_signals(candles, indicators, portfolio)
            if not signals:
                continue

            active_sig = signals[-1]
            if active_sig.action not in (SignalAction.BUY, SignalAction.SELL):
                continue

            # Validate Signal against NIFTY Regime
            allowed, regime_reason = await self.regime_detector.validate_trade(candidate_symbol, active_sig.action)
            if not allowed:
                logger.info(f"[RegimeFilter] {candidate_symbol} {active_sig.action.value} blocked: {regime_reason}")
                continue

            # Sizing & Dynamic ATR Stop Loss / Take Profit
            trade_side = Side.BUY if active_sig.action == SignalAction.BUY else Side.SELL
            entry_price = latest_candle.close

            # ATR-based volatility stop (clamped between 0.8% and 2.5%)
            atr_val = indicators.atr_14 or (entry_price * 0.012)
            stop_dist = max(entry_price * 0.008, min(entry_price * 0.025, 1.5 * atr_val))
            take_profit_dist = 2.0 * stop_dist  # Strict 2:1 R:R

            stop_loss = entry_price - stop_dist if trade_side == Side.BUY else entry_price + stop_dist
            take_profit = entry_price + take_profit_dist if trade_side == Side.BUY else entry_price - take_profit_dist

            qty = self.position_sizer.calculate_size(
                method=settings.risk.position_sizing,
                capital=portfolio.current_capital,
                risk_per_trade_pct=settings.risk.risk_per_trade_pct,
                entry_price=entry_price,
                stop_loss_price=stop_loss,
            )

            if qty > 0:
                order = Order(
                    symbol=candidate_symbol,
                    side=trade_side,
                    order_type=OrderType.MARKET,
                    quantity=qty,
                    price=entry_price,
                    position_type=PositionType.INTRADAY,
                )

                is_valid, reason = self.validator.validate(order, portfolio)
                if is_valid:
                    self.broker.last_prices[candidate_symbol] = entry_price
                    filled = await self.broker.place_order(order)
                    if filled and filled.status.value == "FILLED":
                        self.active_stops[candidate_symbol] = {
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "stop_dist": stop_dist,
                            "side": trade_side,
                            "highest_price": entry_price,
                            "lowest_price": entry_price,
                        }
                        logger.success(
                            f"ORDER FILLED: {trade_side.value} {qty} {candidate_symbol} @ ₹{entry_price:.2f} "
                            f"(SL: ₹{stop_loss:.2f}, TP: ₹{take_profit:.2f}, ATR: ₹{atr_val:.2f}) | Regime: {regime.value}"
                        )
                        if self.alert_mgr.enabled:
                            await self.alert_mgr.send_alert(
                                f"🚀 Executed {trade_side.value} {qty} {candidate_symbol} @ ₹{entry_price:.2f}\n"
                                f"SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f} (2:1 R:R)"
                            )
                        # Entered position; stop evaluating further symbols this tick
                        break
                else:
                    logger.debug(f"Order rejected by validator for {candidate_symbol}: {reason}")

        # Tick Summary Heartbeat
        primary_sym = self.watchlist[0] if self.watchlist else "INFY"
        self.console.print(
            f"[{get_ist_now().strftime('%H:%M:%S')}] Watchlist: {', '.join(self.watchlist)} | "
            f"NIFTY: {regime.value} | "
            f"Daily P&L: ₹{self.goal_mgr.realized_pnl:+.2f} / ₹{self.goal_mgr.target_profit:.2f}"
        )

    async def run_post_market_phase(self, today_str: str):
        """Runs post-market reflection and persistent memory calibration at ~15:45 IST."""
        if self.post_market_done_for_date == today_str:
            return

        self.console.print("[bold cyan]Phase 4: Running Post-Market Autonomous Reflection...[/bold cyan]")
        logger.info("[PostMarket] Generating trade reflection memory entries for completed trades...")

        completed = self.broker.completed_trades
        if completed:
            for trade in completed:
                pnl_pct = (trade.net_pnl / (trade.entry_price * trade.quantity)) * 100.0 if trade.entry_price else 0.0
                outcome = "SUCCESS" if trade.net_pnl > 0 else "LOSS"
                self.reflection_memory.record_reflection(
                    symbol=trade.symbol,
                    action=trade.side.value if hasattr(trade.side, "value") else str(trade.side),
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    pnl_pct=pnl_pct,
                    alpha_pct=None,
                    reasoning_summary=f"Intraday 5m Momentum trade. Regime-aligned execution. Net P&L: ₹{trade.net_pnl:+.2f}",
                    outcome=outcome,
                )
            logger.success(f"[PostMarket] Recorded {len(completed)} trade reflections to memory.")
        else:
            logger.info("[PostMarket] No trades executed today. No new reflections needed.")

        self.post_market_done_for_date = today_str

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
        self.update_heartbeat("RUNNING", self.loop_interval)

        if self.test_mode:
            logger.info("Running single test mode iteration...")
            await self.execute_trading_tick()
            self.update_heartbeat("STOPPED", 0)
            self.console.print("[bold green]✔ Test iteration completed successfully.[/bold green]")
            return

        while True:
            try:
                ctrl_signal = self.check_control_signal()
                if ctrl_signal == "KILL":
                    logger.warning("[Daemon] KILL signal received from Control Center! Squaring off and halting...")
                    await self.auto_squareoff()
                    self.update_heartbeat("STOPPED", 30)
                    await asyncio.sleep(30)
                    continue
                elif ctrl_signal == "PAUSE":
                    logger.info("[Daemon] PAUSE signal active from Control Center. Resting...")
                    self.update_heartbeat("PAUSED", 15)
                    await asyncio.sleep(15)
                    continue

                now_ist = get_ist_now()
                current_time = now_ist.time()
                today_str = now_ist.strftime("%Y-%m-%d")
                mkt_status = get_market_status()

                # Weekend / Holiday Check
                if not mkt_status["is_trading_day"]:
                    logger.info(f"[Daemon] Market closed ({mkt_status['reason']}). Sleeping for 30 minutes...")
                    self.update_heartbeat("RUNNING", 1800)
                    await asyncio.sleep(1800)
                    continue

                # Pre-Market Window: 08:45 - 09:14 IST
                if time(8, 45) <= current_time < time(9, 15):
                    self.update_heartbeat("RUNNING", 60)
                    await self.run_pre_market_phase(today_str)
                    await asyncio.sleep(60)
                    continue

                # Live Market Hours: 09:15 - 15:20 IST
                elif time(9, 15) <= current_time < time(15, 20):
                    self.update_heartbeat("RUNNING", self.loop_interval)
                    await self.execute_trading_tick()

                # Square-off Window: 15:20 - 15:30 IST
                elif time(15, 20) <= current_time < time(15, 30):
                    if self.broker.positions:
                        logger.warning("[Daemon] Approaching 15:30 IST market close. Squaring off open positions...")
                        await self.auto_squareoff()
                    self.update_heartbeat("RUNNING", 60)
                    await asyncio.sleep(60)
                    continue

                # Post-Market Reflection Window: 15:30 - 16:00 IST
                elif time(15, 30) <= current_time < time(16, 0):
                    if self.broker.positions:
                        await self.auto_squareoff()
                    if current_time >= time(15, 45):
                        await self.run_post_market_phase(today_str)
                    logger.info(f"[Daemon] Post-Market Session. Realized P&L: ₹{self.goal_mgr.realized_pnl:+.2f}. Resting...")
                    self.update_heartbeat("RUNNING", 600)
                    await asyncio.sleep(600)
                    continue

                # Overnight Rest: After 16:00 or before 08:45 IST
                else:
                    if self.broker.positions:
                        await self.auto_squareoff()
                    logger.info(f"[Daemon] Market closed for today. Realized P&L: ₹{self.goal_mgr.realized_pnl:+.2f}. Resting...")
                    self.update_heartbeat("RUNNING", 600)
                    await asyncio.sleep(600)
                    continue

            except Exception as e:
                logger.error(f"[DailyBot] Unhandled error in execution cycle: {e}")

            self.update_heartbeat("RUNNING", self.loop_interval)
            await asyncio.sleep(self.loop_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Daily Trading Bot")
    parser.add_argument("--test", action="store_true", help="Run single test iteration and exit")
    parser.add_argument("--interval", type=int, default=15, help="Loop interval in seconds")
    args = parser.parse_args()

    bot = AutonomousTradingBot(test_mode=args.test, loop_interval_secs=args.interval)
    asyncio.run(bot.run())
