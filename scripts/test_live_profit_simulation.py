"""End-to-End Live Data Profit Simulation & Self-Improvement Test Script."""

import asyncio
from datetime import datetime
from pathlib import Path
import sys

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agents.orchestrator import AgentOrchestrator
from src.agents.self_learning import SelfLearningEngine
from src.core.config import settings
from src.core.enums import Exchange, OrderType, PositionType, Side, SignalAction, TimeFrame
from src.core.models import Candle, MarketContext, Order
from src.data.feeds.live_feed import LiveDataFeed
from src.data.indicators import IndicatorEngine
from src.execution.fees import FeeCalculator
from src.execution.mock_engine import MockBroker
from src.execution.order_validator import OrderValidator
from src.execution.position_manager import PositionManager


async def run_live_profit_simulation():
    console = Console()
    console.print(
        Panel.fit(
            "[bold green]End-to-End Pipeline Demo (single paper trade on historical data)[/bold green]\n"
            "[dim]Delayed Yahoo Finance quotes → News Sentiment → Multi-Agent Decision → "
            "ONE paper trade evaluated against REAL subsequent price movement.[/dim]\n"
            "[bold yellow]NOTE:[/bold yellow] [dim]This is a plumbing demo, not proof of profitability. "
            "It executes a single trade and measures its outcome on actual past prices — "
            "a positive result here does NOT mean the strategy is profitable.[/dim]",
            border_style="green",
        )
    )

    feed = LiveDataFeed()
    orchestrator = AgentOrchestrator()
    broker = MockBroker(initial_capital=settings.mock.initial_capital)
    validator = OrderValidator()
    validator.set_backtest_mode(True)
    position_manager = PositionManager()
    learning_engine = SelfLearningEngine()

    candidate_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "NIFTY 50"]
    candidate_contexts = []

    console.print("\n[bold yellow]Step 1: Fetching Real-Time Market Data from NSE...[/bold yellow]")

    for symbol in candidate_symbols:
        candles = await feed.get_historical_candles(symbol, exchange=Exchange.NSE, timeframe=TimeFrame.M5)
        if len(candles) >= 30:
            df = pd.DataFrame([c.model_dump() for c in candles])
            df_ind = IndicatorEngine.calculate(df)
            latest_ind = IndicatorEngine.get_latest_indicators(df_ind)

            ctx = MarketContext(
                symbol=symbol,
                exchange=Exchange.NSE,
                current_price=candles[-1].close,
                candles=candles,
                indicators=latest_ind,
                timestamp=candles[-1].timestamp,
            )
            candidate_contexts.append(ctx)
            console.print(
                f"  • [green]Fetched NSE Live[/green] | [bold white]{symbol}[/bold white] | Price: [bold cyan]₹{ctx.current_price:.2f}[/bold cyan] | RSI: {latest_ind.rsi_14:.1f} | ADX: {latest_ind.adx:.1f} | ATR: ₹{latest_ind.atr_14:.2f}"
            )

    if not candidate_contexts:
        console.print("[bold red]No candidate market contexts loaded. Exiting.[/bold red]")
        return

    # Step 2: AI Market Selection
    console.print("\n[bold yellow]Step 2: AI Market Opportunity Scanner...[/bold yellow]")
    selected_ctx, decision, rationale = await orchestrator.select_and_run(candidate_contexts)

    console.print(
        Panel(
            f"[bold green]Selected Market Target:[/bold green] [bold white]{selected_ctx.symbol}[/bold white]\n"
            f"[dim]{rationale}[/dim]",
            title="🎯 Market Selector Decision",
            border_style="green",
        )
    )

    # Step 3: Multi-Agent Opinions & Decision
    console.print("\n[bold yellow]Step 3: Multi-Agent Pipeline & Live News Sentiment...[/bold yellow]")

    tbl_opinions = Table(title=f"AI Agent Opinions & Sentiment Analysis for {selected_ctx.symbol}")
    tbl_opinions.add_column("Agent Role", style="cyan")
    tbl_opinions.add_column("Action", style="magenta")
    tbl_opinions.add_column("Confidence", style="yellow")
    tbl_opinions.add_column("Calibrated Weight", style="green")
    tbl_opinions.add_column("Reasoning & News", style="white")

    for op in decision.agent_opinions:
        weight = learning_engine.get_agent_weight(op.agent_role)
        tbl_opinions.add_row(
            op.agent_role.value.upper(),
            op.action.value,
            f"{op.confidence * 100:.0f}%",
            f"{weight:.2f}x",
            op.reasoning[:90] + "..." if len(op.reasoning) > 90 else op.reasoning,
        )

    console.print(tbl_opinions)

    console.print(
        f"\n[bold green]Final Decision:[/bold green] [bold yellow]{decision.action.value}[/bold yellow] | Confidence: [bold cyan]{decision.confidence * 100:.0f}%[/bold cyan] | Size: [bold white]{decision.position_size or 10}[/bold white] shares"
    )

    # Step 4: Paper Trade Fill & Profit Tracking
    console.print("\n[bold yellow]Step 4: Order Execution & Fee Calculation (NSE MIS)...[/bold yellow]")

    if decision.action not in (SignalAction.BUY, SignalAction.SELL):
        console.print(
            f"[bold yellow]Decision was {decision.action.value} — no trade taken.[/bold yellow] "
            f"(HOLD/EXIT are not tradable entries in this demo.)"
        )
        return

    # Honest evaluation: enter on a PAST candle and exit using REAL subsequent
    # price action, applying a 2% stop / 4% target over the held window. No
    # assumed favorable move — the outcome is whatever the real prices did.
    candles = selected_ctx.candles
    hold_bars = min(12, max(1, len(candles) // 4))
    entry_idx = len(candles) - hold_bars - 1
    if entry_idx < 0:
        console.print("[bold red]Not enough candles to evaluate a realistic exit. Exiting.[/bold red]")
        return

    entry_price = candles[entry_idx].close
    trade_side = Side.BUY if decision.action == SignalAction.BUY else Side.SELL

    from src.risk.position_sizer import PositionSizer
    sizer = PositionSizer()
    stop_pct = settings.risk.default_stop_loss_pct / 100.0
    if trade_side == Side.BUY:
        stop_loss_price = entry_price * (1 - stop_pct)
        take_profit_price = entry_price * (1 + 2 * stop_pct)
    else:
        stop_loss_price = entry_price * (1 + stop_pct)
        take_profit_price = entry_price * (1 - 2 * stop_pct)

    # Seed the broker's last price at the entry candle, then fill the entry there.
    broker.last_prices[selected_ctx.symbol] = entry_price
    portfolio = await broker.get_portfolio()
    calculated_qty = sizer.calculate_size(
        method=settings.risk.position_sizing,
        capital=portfolio.current_capital,
        risk_per_trade_pct=settings.risk.risk_per_trade_pct,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
    )
    final_qty = max(1, calculated_qty)

    order = Order(
        symbol=selected_ctx.symbol,
        side=trade_side,
        order_type=OrderType.MARKET,
        quantity=final_qty,
        price=entry_price,
        position_type=PositionType.INTRADAY,
    )

    is_valid, reject_reason = validator.validate(order, portfolio)
    if is_valid:
        filled_order = await broker.place_order(order)
        position = await position_manager.open_position(filled_order)

        # Walk forward through the REAL held candles; exit at whichever of
        # stop-loss / take-profit is touched first, else at the last real close.
        exit_price = candles[-1].close
        exit_reason = "held to last candle"
        for c in candles[entry_idx + 1:]:
            if trade_side == Side.BUY:
                if c.low <= stop_loss_price:
                    exit_price, exit_reason = stop_loss_price, "stop-loss hit"
                    break
                if c.high >= take_profit_price:
                    exit_price, exit_reason = take_profit_price, "take-profit hit"
                    break
            else:
                if c.high >= stop_loss_price:
                    exit_price, exit_reason = stop_loss_price, "stop-loss hit"
                    break
                if c.low <= take_profit_price:
                    exit_price, exit_reason = take_profit_price, "take-profit hit"
                    break

        fee_calc = FeeCalculator()
        fees_exit = fee_calc.calculate_fees(
            Side.SELL if trade_side == Side.BUY else Side.BUY,
            filled_order.filled_quantity,
            exit_price,
            PositionType.INTRADAY,
        )

        gross_pnl = (exit_price - filled_order.average_price) * filled_order.filled_quantity if trade_side == Side.BUY else (filled_order.average_price - exit_price) * filled_order.filled_quantity
        net_pnl = gross_pnl - (filled_order.fees.total + fees_exit.total)

        tbl_profit = Table(title="Single Paper Trade — Outcome on Real Subsequent Prices")
        tbl_profit.add_column("Metric", style="cyan")
        tbl_profit.add_column("Value", style="bold white")

        tbl_profit.add_row("Instrument Symbol", filled_order.symbol)
        tbl_profit.add_row("Execution Side", filled_order.side.value)
        tbl_profit.add_row("Quantity", str(filled_order.filled_quantity))
        tbl_profit.add_row("Entry Price", f"₹{filled_order.average_price:.2f}")
        tbl_profit.add_row(f"Exit Price ({exit_reason})", f"₹{exit_price:.2f}")
        tbl_profit.add_row("Gross Intraday P&L", f"₹{gross_pnl:.2f}")
        tbl_profit.add_row("Round-Trip Brokerage + STT + GST", f"₹{filled_order.fees.total + fees_exit.total:.2f}")
        tbl_profit.add_row("Net Realized P&L", f"[bold green]₹{net_pnl:.2f}[/bold green]" if net_pnl >= 0 else f"[bold red]₹{net_pnl:.2f}[/bold red]")

        console.print(tbl_profit)

        # Step 5: Self-Learning Loop
        console.print("\n[bold yellow]Step 5: Self-Improvement & Agent Calibration Loop...[/bold yellow]")
        decision_id = learning_engine.record_decision(selected_ctx.symbol, filled_order.average_price, decision)
        current_prices = {selected_ctx.symbol: exit_price}
        evaluated = learning_engine.evaluate_outcomes(current_prices)

        tbl_learning = Table(title="Self-Learning Engine Agent Accuracy & Calibrated Weights")
        tbl_learning.add_column("Agent Role", style="cyan")
        tbl_learning.add_column("Evaluated Trades", style="yellow")
        tbl_learning.add_column("Historical Accuracy", style="magenta")
        tbl_learning.add_column("Dynamic Confidence Weight", style="green")

        for role_str, stats in learning_engine.agent_accuracy.items():
            tbl_learning.add_row(
                role_str.upper(),
                str(stats["total"]),
                f"{stats['accuracy'] * 100:.1f}%",
                f"{stats['weight']:.2f}x",
            )

        console.print(tbl_learning)
        console.print("\n[bold green]🎉 End-to-End Live Profit & Self-Learning Simulation Complete![/bold green]")

    else:
        console.print(f"[bold red]Order Rejected:[/bold red] {reject_reason}")


if __name__ == "__main__":
    asyncio.run(run_live_profit_simulation())
