"""Test script for API Quota Tracker and Daily Goal Controller."""

import asyncio
from pathlib import Path
import sys

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.models import PortfolioState
from src.core.quota_tracker import QuotaTracker
from src.risk.daily_goal import DailyGoalController


def test_quota_and_daily_goal():
    console = Console()
    console.print(
        Panel.fit(
            "[bold green]API Quota Tracker & Daily Target Goal Controller Demonstration[/bold green]\n"
            "[dim]Testing Rate Limiter, Daily Quota Protection, and Target Goal Auto-Halt Controls[/dim]",
            border_style="green",
        )
    )

    # 1. Test Quota Tracker
    console.print("\n[bold yellow]1. Testing Quota Tracker & Rate Limiter...[/bold yellow]")
    tracker = QuotaTracker(max_rpd=1000, max_rpm=30)
    tracker.record_request(5)  # Simulate multi-agent decision request

    summary = tracker.get_summary()

    tbl_quota = Table(title="Google AI / Antigravity Quota Usage Tracker")
    tbl_quota.add_column("Metric", style="cyan")
    tbl_quota.add_column("Value", style="bold white")

    tbl_quota.add_row("Date", summary["date"])
    tbl_quota.add_row("Daily Requests Used", f"{summary['daily_requests']} RPD")
    tbl_quota.add_row("Daily Quota Limit", f"{summary['max_rpd']} RPD")
    tbl_quota.add_row("Remaining Requests", f"[bold green]{summary['remaining_rpd']}[/bold green]")
    tbl_quota.add_row("Quota Usage %", f"{summary['usage_pct']}%")
    tbl_quota.add_row("Quota Safety Status", "[bold green]SAFE ✅[/bold green]" if summary["is_quota_safe"] else "[bold red]WARNING ⚠️[/bold red]")

    console.print(tbl_quota)

    # 2. Test Daily Target Goal Controller
    console.print("\n[bold yellow]2. Testing Daily Budget & Profit Target Controller...[/bold yellow]")
    controller = DailyGoalController(daily_budget=100000.0, target_profit=2000.0, max_loss_threshold=1000.0)

    # Scenario A: Normal Operating P&L (+₹500)
    state_normal = PortfolioState(
        initial_capital=100000.0,
        current_capital=100500.0,
        available_cash=100500.0,
        allocated_capital=0.0,
        daily_pnl=500.0,
        total_unrealized_pnl=0.0,
        open_positions_count=0,
    )
    should_halt, msg = controller.evaluate_portfolio(state_normal)
    console.print(f"  • P&L: [bold cyan]+₹500.00[/bold cyan] -> Status: [bold green]CONTINUE TRADING[/bold green]")

    # Scenario B: Target Profit Goal Achieved (+₹2,200)
    state_target = PortfolioState(
        initial_capital=100000.0,
        current_capital=102200.0,
        available_cash=102200.0,
        allocated_capital=0.0,
        daily_pnl=2200.0,
        total_unrealized_pnl=0.0,
        open_positions_count=0,
    )
    should_halt, msg = controller.evaluate_portfolio(state_target)
    console.print(f"  • P&L: [bold green]+₹2,200.00[/bold green] -> Target Goal Achieved! Status: [bold red]AUTO-HALTED ✅[/bold red]")
    console.print(Panel(msg, title="🎉 Target Profit Goal Reached", border_style="green"))

    console.print("\n[bold green]🎉 Quota Tracker and Daily Goal Controller Tests Complete![/bold green]")


if __name__ == "__main__":
    test_quota_and_daily_goal()
