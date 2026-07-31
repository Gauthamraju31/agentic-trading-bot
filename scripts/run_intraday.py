"""Morning-open INTRADAY paper-trading runner.

Runs one trading session for a single symbol:
  * starts at market open, steps through 5-minute bars,
  * runs the multi-agent decision pipeline to ENTER (throttled to save
    latency/quota; the full debate takes ~1 min per call),
  * enforces a per-trade stop-loss / take-profit on EVERY bar
    (deterministically — a stop always fires, no LLM needed),
  * halts for the day when the DailyGoalController hits the profit target
    or the loss threshold,
  * squares off any open position before the close (true intraday / MIS).

Paper trading only (MockBroker). No real broker is wired.

Example:
    python scripts/run_intraday.py --symbol ITC --capital 5000 --target 100 --stop 100
    python scripts/run_intraday.py --symbol ITC --fast-mock          # quick, no LLM
"""

import argparse
import asyncio
from datetime import time as dtime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.core.config import settings
from src.core.enums import Exchange, Side, SignalAction, OrderType, PositionType, TimeFrame
from src.core.models import MarketContext, Order, Candle
from src.data.indicators import IndicatorEngine
from src.execution.mock_engine import MockBroker
from src.risk.daily_goal import DailyGoalController

# Default to an AFFORDABLE symbol: ₹5,000 must buy a meaningful number of shares
# for a ~₹100 (2%) target to be reachable. ITC (~₹450) → ~11 shares; a 2% move
# ≈ ₹100. A ₹3,000+ stock (RELIANCE/TCS) would only buy 1 share — don't use it here.
DEFAULT_SYMBOL = "ITC"
HISTORY_WINDOW = 50  # bars of context for indicators


def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


def _size_for_capital(capital: float, price: float, deploy_fraction: float = 0.95) -> int:
    """Shares affordable with a small account (bypasses the 5%-of-portfolio
    diversification cap, which is meaningless for a single tiny intraday bet —
    risk is bounded instead by the per-trade stop and the daily loss halt)."""
    if price <= 0:
        return 0
    return int((capital * deploy_fraction) // price)


async def _load_session(symbol: str, use_csv: bool) -> tuple[list[Candle], int]:
    """Fetch candles and keep only the latest calendar day's session."""
    if use_csv:
        from src.data.feeds.csv_feed import CSVDataFeed
        feed = CSVDataFeed(Path("data/historical") / f"{symbol.replace(' ', '_')}_5m.csv")
    else:
        from src.data.feeds.live_feed import LiveDataFeed
        feed = LiveDataFeed()
    candles = await feed.get_historical_candles(symbol, Exchange.NSE, TimeFrame.M5)
    if not candles:
        return [], 0
    last_day = candles[-1].timestamp.date()
    session = [c for c in candles if c.timestamp.date() == last_day]
    # Need some prior-day history for indicator warm-up; prepend the tail.
    warmup = [c for c in candles if c.timestamp.date() != last_day][-HISTORY_WINDOW:]
    return warmup + session, len(warmup)


async def run_intraday(symbol: str, capital: float, target: float, stop: float,
                       throttle: int, use_csv: bool, max_bars: int) -> None:
    console = Console()

    # Single tiny intraday position: relax the portfolio-diversification cap and
    # cap concurrent positions to 1. Risk is governed by the stop + daily halt.
    settings.risk.max_position_pct = 100.0
    settings.risk.max_open_positions = 1

    # Import AFTER the settings tweak so agents/validator read the updated values.
    from src.agents.orchestrator import AgentOrchestrator

    start_t = _parse_hhmm(settings.market.trading_hours.start)   # 09:15
    squareoff_t = dtime(15, 20)                                  # flatten 10 min pre-close
    entry_cutoff_t = dtime(15, 0)                                # no new entries after 15:00

    loaded, warmup_n = await _load_session(symbol, use_csv)
    if not loaded:
        console.print(f"[bold red]No data for {symbol}. Is the market feed reachable?[/bold red]")
        return

    broker = MockBroker(initial_capital=capital)
    orch = AgentOrchestrator()
    goal = DailyGoalController(daily_budget=capital, target_profit=target, max_loss_threshold=stop)
    stop_pct = settings.risk.default_stop_loss_pct / 100.0

    console.print(f"[bold green]Intraday paper session[/bold green]: {symbol} | capital ₹{capital:,.0f} | "
                  f"target +₹{target:,.0f} | stop -₹{stop:,.0f} | bars={len(loaded) - warmup_n}")

    active_stop = None  # (stop_loss, take_profit, side)
    decisions = 0

    session_bars = loaded[warmup_n:]
    if max_bars > 0:
        session_bars = session_bars[:max_bars]

    async def close_position(price: float, why: str):
        nonlocal active_stop
        pos = broker.positions.get(symbol)
        if not pos:
            return
        close_side = Side.SELL if pos.side == Side.BUY else Side.BUY
        broker.last_prices[symbol] = price
        await broker.place_order(Order(symbol=symbol, side=close_side, order_type=OrderType.MARKET,
                                       quantity=pos.quantity, price=price, position_type=PositionType.INTRADAY))
        console.print(f"  [yellow]EXIT[/yellow] {symbol} @ ₹{price:.2f} ({why})")
        active_stop = None

    for i, candle in enumerate(session_bars):
        idx = warmup_n + i
        broker.current_time = candle.timestamp
        bar_time = candle.timestamp.time()

        # 1. Manage an open position's stop/target against this bar's range.
        if broker.positions.get(symbol) and active_stop:
            sl, tp, side = active_stop
            exit_price = None
            if side == Side.BUY:
                if candle.low <= sl:
                    exit_price = sl
                elif candle.high >= tp:
                    exit_price = tp
            else:
                if candle.high >= sl:
                    exit_price = sl
                elif candle.low <= tp:
                    exit_price = tp
            if exit_price is not None:
                await close_position(exit_price, "stop/target")

        # 2. End-of-day square-off.
        if bar_time >= squareoff_t and broker.positions.get(symbol):
            await close_position(candle.close, "EOD square-off")

        # 3. Mark open position to close & evaluate the daily goal on realized+unrealized P&L.
        if symbol in broker.positions:
            broker.positions[symbol].current_price = candle.close
        portfolio = await broker.get_portfolio()
        portfolio.daily_pnl = portfolio.total_realized_pnl + portfolio.total_unrealized_pnl
        halted, reason = goal.evaluate_portfolio(portfolio)
        if halted:
            if broker.positions.get(symbol):
                await close_position(candle.close, "daily goal reached")
            console.print(f"[bold]{reason}[/bold]")
            break

        # 4. Entry: only when flat, inside the entry window, throttled.
        in_entry_window = start_t <= bar_time < entry_cutoff_t
        if (symbol not in broker.positions and in_entry_window and (i % max(1, throttle) == 0)):
            history = loaded[max(0, idx - HISTORY_WINDOW):idx + 1]
            df = pd.DataFrame([c.model_dump() for c in history])
            latest_ind = IndicatorEngine.get_latest_indicators(IndicatorEngine.calculate(df))
            ctx = MarketContext(symbol=symbol, exchange=Exchange.NSE, current_price=candle.close,
                                candles=history, indicators=latest_ind, portfolio=portfolio,
                                timestamp=candle.timestamp)
            decisions += 1
            decision = await orch.run(ctx)
            console.print(f"  [{candle.timestamp:%H:%M}] decision: {decision.action.value} "
                          f"(conf {decision.confidence:.2f}, approved={decision.approved_by_risk})")

            if decision.approved_by_risk and decision.action in (SignalAction.BUY, SignalAction.SELL):
                side = Side.BUY if decision.action == SignalAction.BUY else Side.SELL
                qty = _size_for_capital(broker.current_capital, candle.close)
                if qty >= 1:
                    entry = candle.close
                    broker.last_prices[symbol] = entry
                    await broker.place_order(Order(symbol=symbol, side=side, order_type=OrderType.MARKET,
                                                   quantity=qty, price=entry, position_type=PositionType.INTRADAY))
                    if side == Side.BUY:
                        sl = decision.stop_loss or entry * (1 - stop_pct)
                        tp = decision.take_profit or entry * (1 + 2 * stop_pct)
                    else:
                        sl = decision.stop_loss or entry * (1 + stop_pct)
                        tp = decision.take_profit or entry * (1 - 2 * stop_pct)
                    active_stop = (sl, tp, side)
                    console.print(f"  [green]ENTER[/green] {side.value} {qty} {symbol} @ ₹{entry:.2f} "
                                  f"| SL ₹{sl:.2f} TP ₹{tp:.2f}")
                else:
                    console.print(f"  [dim]Signal {decision.action.value} but ₹{broker.current_capital:.0f} "
                                  f"can't afford 1 share @ ₹{candle.close:.2f}.[/dim]")

    # Session summary.
    portfolio = await broker.get_portfolio()
    trades = broker.completed_trades
    realized = portfolio.total_realized_pnl
    table = Table(title=f"Intraday Session Summary — {symbol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")
    table.add_row("Agent decisions run", str(decisions))
    table.add_row("Trades taken", str(len(trades)))
    table.add_row("Wins", str(sum(1 for t in trades if t.net_pnl > 0)))
    table.add_row("Realized P&L", f"₹{realized:,.2f}")
    table.add_row("Ending cash", f"₹{portfolio.current_capital:,.2f}")
    table.add_row("Target hit", "yes" if goal.target_achieved else "no")
    table.add_row("Daily-loss halt", "yes" if goal.is_halted and not goal.target_achieved else "no")
    console.print(table)


def main():
    p = argparse.ArgumentParser(description="Morning-open intraday paper-trading runner.")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Symbol (use an affordable one for small capital)")
    p.add_argument("--capital", type=float, default=5000.0, help="Daily paper capital (₹)")
    p.add_argument("--target", type=float, default=100.0, help="Daily profit target to halt at (₹)")
    p.add_argument("--stop", type=float, default=100.0, help="Daily loss threshold to halt at (₹)")
    p.add_argument("--throttle", type=int, default=3, help="Run the agent debate every N bars while flat")
    p.add_argument("--max-bars", type=int, default=0, help="Limit bars processed (0 = full session; for testing)")
    p.add_argument("--csv", action="store_true", help="Use local CSV data instead of the live (yfinance) feed")
    p.add_argument("--fast-mock", action="store_true", help="Force deterministic agents (no LLM) — fast mechanics test")
    args = p.parse_args()

    if args.fast_mock:
        settings.agents.llm_provider = "mock"

    asyncio.run(run_intraday(args.symbol, args.capital, args.target, args.stop,
                             args.throttle, args.csv, args.max_bars))


if __name__ == "__main__":
    main()
