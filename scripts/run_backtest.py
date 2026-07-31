import argparse
import asyncio
from pathlib import Path
import sys

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.backtest.engine import BacktestEngine
from src.backtest.walk_forward import WalkForwardAnalyzer
from src.core.config import settings
from src.execution.fees import FeeCalculator
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.momentum import MomentumStrategy

DATA_DIR = Path("data/historical")

# NOTE: This project ships only SYNTHETIC random-walk data (see
# scripts/download_data.py). A backtest over synthetic data validates that the
# engine runs end-to-end; it says NOTHING about real-world profitability. Replace
# the CSVs under data/historical/ with real OHLCV data before trusting any metric.


def _load_dataframe(symbol: str) -> tuple[pd.DataFrame, bool]:
    """Load OHLCV for a symbol from data/historical, generating synthetic data
    only as a fallback. Returns (dataframe_indexed_by_timestamp, is_synthetic)."""
    file_path = DATA_DIR / f"{symbol.replace(' ', '_')}_5m.csv"
    is_synthetic = False

    if not file_path.exists():
        from datetime import datetime, timedelta
        from scripts.download_data import generate_random_walk_candles

        logger.warning(
            f"No data file at {file_path}; generating SYNTHETIC random-walk data. "
            f"Results will NOT reflect real performance."
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_gen = generate_random_walk_candles(symbol, datetime.now() - timedelta(days=180), 180)
        df_gen.to_csv(file_path, index=False)
        is_synthetic = True

    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df, is_synthetic


def _make_strategy(strategy: str):
    if strategy == "mean_reversion":
        return MeanReversionStrategy()
    return MomentumStrategy()


def format_results(symbol: str, strategy: str, metrics: dict, extra: dict) -> None:
    console = Console()
    table = Table(title=f"Backtest Results — {symbol} ({strategy})")
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    rows = {
        "Signals Generated": str(extra.get("signals_generated", 0)),
        "Signals Executed": str(extra.get("signals_executed", 0)),
        "Signals Rejected": str(extra.get("signals_rejected", 0)),
        "Total Trades": str(metrics.get("total_trades", 0)),
        "Win Rate": f"{metrics.get('win_rate', 0):.2f}%",
        "Total P&L": f"₹{metrics.get('total_pnl', 0):,.2f}",
        "Total Return": f"{metrics.get('total_return_pct', 0):.2f}%",
        "Profit Factor": f"{metrics.get('profit_factor', 0):.2f}",
        "Max Drawdown": f"{metrics.get('max_drawdown_pct', 0):.2f}%",
        "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
        "Final Equity": f"₹{extra.get('final_equity', 0):,.2f}",
    }
    for key, value in rows.items():
        table.add_row(key, value)
    console.print(table)


async def run_backtest(symbol: str, strategy: str, days: int, walk_forward: bool):
    logger.info(f"Starting backtest for {symbol} using {strategy} strategy")
    df, is_synthetic = _load_dataframe(symbol)
    if df.empty:
        logger.error("No data available; aborting.")
        return

    if is_synthetic:
        logger.warning(
            "⚠️  DATA IS SYNTHETIC (random walk). These numbers do NOT indicate "
            "that the strategy is profitable on real markets."
        )

    fee_calculator = FeeCalculator()

    if walk_forward:
        logger.info("Running walk-forward analysis")
        param_grid = {
            "adx_min": [20.0, 25.0, 30.0],
            "rsi_sell_max": [55.0, 60.0],
        } if strategy == "momentum" else {
            "rsi_buy_below": [25.0, 30.0, 35.0],
            "adx_max": [25.0, 30.0],
        }
        analyzer = WalkForwardAnalyzer(
            strategy_class=MomentumStrategy if strategy == "momentum" else MeanReversionStrategy,
            param_grid=param_grid,
            initial_capital=settings.backtest.initial_capital,
            fee_calculator=fee_calculator,
            settings=settings,
        )
        wf = await analyzer.run(
            df, symbol,
            in_sample_days=settings.backtest.walk_forward.in_sample_days,
            out_sample_days=settings.backtest.walk_forward.out_sample_days,
        )
        logger.info(f"Walk-forward robust: {wf.is_robust}")
        format_results(symbol, strategy, wf.aggregate_oos_metrics, {})
        return

    engine = BacktestEngine(
        strategy=_make_strategy(strategy),
        initial_capital=settings.backtest.initial_capital,
        fee_calculator=fee_calculator,
        settings=settings,
    )
    result = await engine.run(df, symbol)

    logger.info("Backtest complete!")
    format_results(
        symbol, strategy, result.metrics,
        {
            "signals_generated": result.signals_generated,
            "signals_executed": result.signals_executed,
            "signals_rejected": result.signals_rejected,
            "final_equity": result.final_equity,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Run backtests for trading strategies.")
    parser.add_argument("--symbol", type=str, default="RELIANCE", help="Symbol to backtest")
    parser.add_argument("--strategy", type=str, choices=["momentum", "mean_reversion"], default="momentum", help="Strategy to use")
    parser.add_argument("--days", type=int, default=365, help="Number of days to backtest")
    parser.add_argument("--walk-forward", action="store_true", help="Enable walk-forward analysis")

    args = parser.parse_args()
    asyncio.run(run_backtest(args.symbol, args.strategy, args.days, args.walk_forward))


if __name__ == "__main__":
    main()
