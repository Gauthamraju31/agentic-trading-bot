"""Day Trading & Market Selection Test Script for the Multi-Agent Trading System."""

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
from src.core.config import settings
from src.core.enums import Exchange, OrderType, PositionType, Side, SignalAction, TimeFrame
from src.core.models import Candle, MarketContext, Order
from src.data.feeds.csv_feed import CSVDataFeed
from src.data.indicators import IndicatorEngine
from src.execution.mock_engine import MockBroker
from src.execution.order_validator import OrderValidator
from src.execution.position_manager import PositionManager


async def run_day_trading_test():
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]Multi-Agent Day Trading & Market Selection Test[/bold cyan]\n"
            "[dim]Scanning NSE Equities & Indices → Selecting Prime Intraday Target → Executing Multi-Agent Trade[/dim]",
            border_style="cyan",
        )
    )

    candidate_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "NIFTY_50"]
    candidate_contexts = []

    console.print("\n[bold yellow]Step 1: Loading Candidate Market Contexts...[/bold yellow]")

    for symbol in candidate_symbols:
        feed_path = Path("data/historical") / f"{symbol}_5m.csv"
        if not feed_path.exists():
            from scripts.download_data import generate_random_walk_candles

            df_gen = generate_random_walk_candles(symbol.replace("_", " "), datetime.now(), 30)
            feed_path.parent.mkdir(parents=True, exist_ok=True)
            df_gen.to_csv(feed_path, index=False)

        feed = CSVDataFeed(feed_path)
        candles = await feed.get_historical_candles(symbol)

        if len(candles) > 50:
            df = pd.DataFrame([c.model_dump() for c in candles[-60:]])
            df_ind = IndicatorEngine.calculate(df)
            latest_ind = IndicatorEngine.get_latest_indicators(df_ind)

            ctx = MarketContext(
                symbol=symbol.replace("_", " "),
                exchange=Exchange.NSE,
                current_price=candles[-1].close,
                candles=candles[-60:],
                indicators=latest_ind,
                timestamp=candles[-1].timestamp,
            )
            candidate_contexts.append(ctx)
            console.print(
                f"  • [green]Loaded {symbol.replace('_', ' ')}[/green] | Price: ₹{ctx.current_price:.2f} | RSI: {latest_ind.rsi_14:.1f} | ADX: {latest_ind.adx:.1f} | ATR: ₹{latest_ind.atr_14:.2f}"
            )

    # Step 2: Ask Market Selector Agent to choose the best intraday target
    console.print("\n[bold yellow]Step 2: AI Market Selection Scanner...[/bold yellow]")
    orchestrator = AgentOrchestrator()
    selected_ctx, decision, selection_reasoning = await orchestrator.select_and_run(candidate_contexts)

    console.print(
        Panel(
            f"[bold green]Selected Market Target:[/bold green] [bold white]{selected_ctx.symbol}[/bold white]\n"
            f"[dim]{selection_reasoning}[/dim]",
            title="🎯 Market Selector Decision",
            border_style="green",
        )
    )

    # Step 3: Multi-Agent Analysis & Trade Decision
    console.print("\n[bold yellow]Step 3: Multi-Agent Debate & Final Trade Signal...[/bold yellow]")

    tbl_opinions = Table(title=f"AI Agent Pipeline Opinions for {selected_ctx.symbol}")
    tbl_opinions.add_column("Agent Role", style="cyan")
    tbl_opinions.add_column("Action", style="magenta")
    tbl_opinions.add_column("Confidence", style="yellow")
    tbl_opinions.add_column("Reasoning", style="white")

    for op in decision.agent_opinions:
        tbl_opinions.add_row(
            op.agent_role.value.upper(),
            op.action.value,
            f"{op.confidence * 100:.0f}%",
            op.reasoning[:80] + "..." if len(op.reasoning) > 80 else op.reasoning,
        )

    console.print(tbl_opinions)

    console.print(
        f"\n[bold green]Final Portfolio Decision:[/bold green] [bold yellow]{decision.action.value}[/bold yellow] | Confidence: [bold cyan]{decision.confidence * 100:.0f}%[/bold cyan] | Recommended Size: [bold white]{decision.position_size}[/bold white] shares"
    )

    # Step 4: Intraday Execution Simulation (MIS Format)
    console.print("\n[bold yellow]Step 4: Intraday Execution & Risk Check (MIS)...[/bold yellow]")
    broker = MockBroker(initial_capital=settings.mock.initial_capital)
    validator = OrderValidator()
    validator.set_backtest_mode(True)
    position_manager = PositionManager()

    # Feed current candle into broker
    await broker.process_candle(selected_ctx.candles[-1])
    portfolio = await broker.get_portfolio()

    # Force a sample intraday trade to demonstrate full round-trip MIS execution if HOLD
    trade_action = decision.action if decision.action in (SignalAction.BUY, SignalAction.SELL) else Side.BUY

    order = Order(
        symbol=selected_ctx.symbol,
        side=Side.BUY if trade_action == SignalAction.BUY or trade_action == Side.BUY else Side.SELL,
        order_type=OrderType.MARKET,
        quantity=decision.position_size or 20,
        price=selected_ctx.current_price,
        position_type=PositionType.INTRADAY,  # Intraday MIS
    )

    is_valid, reject_reason = validator.validate(order, portfolio)
    if is_valid:
        filled_order = await broker.place_order(order)
        position = await position_manager.open_position(filled_order)

        tbl_exec = Table(title="Execution Fill & Intraday Margin (MIS)")
        tbl_exec.add_column("Order ID", style="dim")
        tbl_exec.add_column("Symbol", style="cyan")
        tbl_exec.add_column("Side", style="green")
        tbl_exec.add_column("Qty", style="yellow")
        tbl_exec.add_column("Fill Price", style="bold white")
        tbl_exec.add_column("Brokerage + Taxes", style="magenta")

        tbl_exec.add_row(
            filled_order.id[:8],
            filled_order.symbol,
            filled_order.side.value,
            str(filled_order.filled_quantity),
            f"₹{filled_order.average_price:.2f}",
            f"₹{filled_order.fees.total:.2f}",
        )
        console.print(tbl_exec)
        console.print("[bold green]✅ Intraday day-trading test completed successfully![/bold green]")
    else:
        console.print(f"[bold red]Order Rejected:[/bold red] {reject_reason}")


if __name__ == "__main__":
    asyncio.run(run_day_trading_test())
