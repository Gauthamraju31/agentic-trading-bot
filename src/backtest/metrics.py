from datetime import timedelta
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.core.models import Trade

class PerformanceMetrics:
    """
    Calculates and stores performance metrics for a backtest or live trading session.
    """
    def __init__(self, trades: List[Trade], initial_capital: float):
        self._trades = trades
        self._initial_capital = initial_capital
        
        # Pre-compute basic stats to be used by properties
        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        """Internal method to calculate and set all metrics based on the provided trades."""
        self._total_trades = len(self._trades)
        
        winning_trades = [t for t in self._trades if t.net_pnl > 0]
        losing_trades = [t for t in self._trades if t.net_pnl <= 0]
        
        self._winning_trades = len(winning_trades)
        self._losing_trades = len(losing_trades)
        
        self._win_rate = (self._winning_trades / self._total_trades * 100) if self._total_trades > 0 else 0.0
        
        self._total_pnl = sum(t.net_pnl for t in self._trades)
        self._total_return_pct = (self._total_pnl / self._initial_capital * 100) if self._initial_capital > 0 else 0.0
        
        gross_profit = sum(t.net_pnl for t in winning_trades)
        gross_loss = abs(sum(t.net_pnl for t in losing_trades))
        
        self._profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        self._avg_win = (gross_profit / self._winning_trades) if self._winning_trades > 0 else 0.0
        self._avg_loss = (gross_loss / self._losing_trades) if self._losing_trades > 0 else 0.0
        
        self._avg_win_loss_ratio = (self._avg_win / self._avg_loss) if self._avg_loss > 0 else float('inf') if self._avg_win > 0 else 0.0
        
        self._largest_win = max([t.net_pnl for t in winning_trades] + [0.0])
        self._largest_loss = min([t.net_pnl for t in losing_trades] + [0.0])
        
        # Calculate consecutive wins/losses
        max_cons_wins = 0
        max_cons_losses = 0
        curr_cons_wins = 0
        curr_cons_losses = 0
        
        for trade in self._trades:
            if trade.net_pnl > 0:
                curr_cons_wins += 1
                curr_cons_losses = 0
                if curr_cons_wins > max_cons_wins:
                    max_cons_wins = curr_cons_wins
            else:
                curr_cons_losses += 1
                curr_cons_wins = 0
                if curr_cons_losses > max_cons_losses:
                    max_cons_losses = curr_cons_losses
                    
        self._max_consecutive_wins = max_cons_wins
        self._max_consecutive_losses = max_cons_losses
        
        # Equity curve and Drawdown
        equity = self._initial_capital
        self._equity_curve = [equity]
        self._drawdowns = []
        self._drawdown_pcts = []
        
        running_max_equity = equity
        
        for trade in self._trades:
            equity += trade.net_pnl
            self._equity_curve.append(equity)
            
            if equity > running_max_equity:
                running_max_equity = equity
                
            dd = running_max_equity - equity
            dd_pct = (dd / running_max_equity) * 100 if running_max_equity > 0 else 0
            
            self._drawdowns.append(dd)
            self._drawdown_pcts.append(dd_pct)
            
        self._max_drawdown = max(self._drawdowns + [0.0])
        self._max_drawdown_pct = max(self._drawdown_pcts + [0.0])
        
        # Average holding period
        holding_periods = [t.exit_time - t.entry_time for t in self._trades if t.exit_time and t.entry_time]
        if holding_periods:
            avg_ticks = sum(td.total_seconds() for td in holding_periods) / len(holding_periods)
            self._avg_holding_period = timedelta(seconds=avg_ticks)
        else:
            self._avg_holding_period = timedelta(0)

    @property
    def total_trades(self) -> int:
        return self._total_trades

    @property
    def winning_trades(self) -> int:
        return self._winning_trades

    @property
    def losing_trades(self) -> int:
        return self._losing_trades

    @property
    def win_rate(self) -> float:
        return self._win_rate

    @property
    def total_pnl(self) -> float:
        return self._total_pnl

    @property
    def total_return_pct(self) -> float:
        return self._total_return_pct

    @property
    def profit_factor(self) -> float:
        return self._profit_factor

    @property
    def avg_win(self) -> float:
        return self._avg_win

    @property
    def avg_loss(self) -> float:
        return self._avg_loss

    @property
    def avg_win_loss_ratio(self) -> float:
        return self._avg_win_loss_ratio

    @property
    def largest_win(self) -> float:
        return self._largest_win

    @property
    def largest_loss(self) -> float:
        return self._largest_loss

    @property
    def max_consecutive_wins(self) -> int:
        return self._max_consecutive_wins

    @property
    def max_consecutive_losses(self) -> int:
        return self._max_consecutive_losses

    @property
    def max_drawdown(self) -> float:
        return self._max_drawdown

    @property
    def max_drawdown_pct(self) -> float:
        return self._max_drawdown_pct

    @property
    def avg_holding_period(self) -> timedelta:
        return self._avg_holding_period

    @property
    def equity_curve(self) -> List[float]:
        return self._equity_curve

    def _get_trade_returns(self) -> np.ndarray:
        """Helper to get percentage returns per trade."""
        if not self._trades:
            return np.array([])
            
        returns = []
        equity = self._initial_capital
        for trade in self._trades:
            ret = trade.net_pnl / equity
            returns.append(ret)
            equity += trade.net_pnl
        return np.array(returns)

    def sharpe_ratio(self, risk_free_rate: float = 0.065) -> float:
        """Calculates the annualized Sharpe ratio."""
        returns = self._get_trade_returns()
        if len(returns) < 2:
            return 0.0
            
        # Assuming ~252 trading days and average trades per day
        # For simplicity in a trade-based metric without a strict timeline, 
        # we treat trade returns relative to a per-trade risk free rate.
        # Estimate total days from first to last trade:
        if not self._trades or not self._trades[0].entry_time or not self._trades[-1].exit_time:
            return 0.0
            
        total_days = (self._trades[-1].exit_time - self._trades[0].entry_time).days
        total_days = max(1, total_days)
        trades_per_year = len(self._trades) * (252 / total_days)
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
            
        rf_per_trade = risk_free_rate / trades_per_year if trades_per_year > 0 else 0
        sharpe = (mean_return - rf_per_trade) / std_return * np.sqrt(trades_per_year)
        return float(sharpe)

    def sortino_ratio(self, risk_free_rate: float = 0.065) -> float:
        """Calculates the annualized Sortino ratio."""
        returns = self._get_trade_returns()
        if len(returns) < 2:
            return 0.0
            
        if not self._trades or not self._trades[0].entry_time or not self._trades[-1].exit_time:
            return 0.0
            
        total_days = (self._trades[-1].exit_time - self._trades[0].entry_time).days
        total_days = max(1, total_days)
        trades_per_year = len(self._trades) * (252 / total_days)
        
        mean_return = np.mean(returns)
        rf_per_trade = risk_free_rate / trades_per_year if trades_per_year > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        
        if downside_std == 0:
            return float('inf') if mean_return > 0 else 0.0
            
        sortino = (mean_return - rf_per_trade) / downside_std * np.sqrt(trades_per_year)
        return float(sortino)

    def calmar_ratio(self) -> float:
        """Calculates the Calmar ratio (Annualized Return / Max Drawdown)."""
        if self._max_drawdown_pct == 0 or not self._trades:
            return 0.0
            
        if not self._trades[0].entry_time or not self._trades[-1].exit_time:
            return 0.0
            
        total_days = (self._trades[-1].exit_time - self._trades[0].entry_time).days
        total_years = total_days / 365.25 if total_days > 0 else 1.0
        
        annualized_return = ((self._equity_curve[-1] / self._initial_capital) ** (1 / total_years)) - 1
        max_dd_decimal = self._max_drawdown_pct / 100
        
        if max_dd_decimal == 0:
            return float('inf') if annualized_return > 0 else 0.0
            
        return float(annualized_return / max_dd_decimal)

    def summary(self) -> Dict[str, Any]:
        """Returns all metrics as a dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_win_loss_ratio": self.avg_win_loss_ratio,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "avg_holding_period": self.avg_holding_period
        }

    def print_report(self) -> None:
        """Pretty-prints the performance metrics using rich."""
        console = Console()
        table = Table(title="Backtest Performance Report", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        metrics = self.summary()
        
        # Formatting helpers
        def fmt_curr(val): return f"₹{val:,.2f}"
        def fmt_pct(val): return f"{val:.2f}%"
        def fmt_num(val): return f"{val:.2f}"
        
        table.add_row("Total Trades", str(metrics["total_trades"]))
        table.add_row("Win Rate", fmt_pct(metrics["win_rate"]))
        table.add_row("Total P&L", fmt_curr(metrics["total_pnl"]))
        table.add_row("Total Return", fmt_pct(metrics["total_return_pct"]))
        table.add_row("Profit Factor", fmt_num(metrics["profit_factor"]))
        table.add_row("Max Drawdown", fmt_curr(metrics["max_drawdown"]))
        table.add_row("Max Drawdown (%)", fmt_pct(metrics["max_drawdown_pct"]))
        table.add_row("Sharpe Ratio", fmt_num(metrics["sharpe_ratio"]))
        table.add_row("Sortino Ratio", fmt_num(metrics["sortino_ratio"]))
        table.add_row("Calmar Ratio", fmt_num(metrics["calmar_ratio"]))
        table.add_row("Average Win", fmt_curr(metrics["avg_win"]))
        table.add_row("Average Loss", fmt_curr(metrics["avg_loss"]))
        table.add_row("Largest Win", fmt_curr(metrics["largest_win"]))
        table.add_row("Largest Loss", fmt_curr(metrics["largest_loss"]))
        table.add_row("Max Cons. Wins", str(metrics["max_consecutive_wins"]))
        table.add_row("Max Cons. Losses", str(metrics["max_consecutive_losses"]))
        table.add_row("Avg Holding Period", str(metrics["avg_holding_period"]))

        console.print(table)
