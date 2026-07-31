import itertools
from datetime import timedelta
from typing import List, Dict, Any, Type
import pandas as pd
from loguru import logger
from pydantic import BaseModel

from src.execution import FeeCalculator
from src.strategy import Strategy
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import PerformanceMetrics


class WalkForwardResult(BaseModel):
    """Result model containing the output of a Walk-Forward Analysis."""
    strategy_name: str
    symbol: str
    windows: List[Dict[str, Any]]
    aggregate_oos_metrics: Dict[str, Any]
    best_params_per_window: List[Dict[str, Any]]
    is_robust: bool


class WalkForwardAnalyzer:
    """
    Implements Walk-Forward Analysis (WFA) to optimize strategy parameters 
    and validate robustness while mitigating overfitting.
    """
    def __init__(
        self,
        strategy_class: Type[Strategy],
        param_grid: Dict[str, List[Any]],
        initial_capital: float,
        fee_calculator: FeeCalculator,
        settings: Any
    ):
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.initial_capital = initial_capital
        self.fee_calculator = fee_calculator
        self.settings = settings

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generates all combinations of parameters from the parameter grid."""
        if not self.param_grid:
            # No grid → a single run with the strategy's default parameters.
            return [{}]
        keys, values = zip(*self.param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return combinations

    async def run(
        self,
        candles_df: pd.DataFrame,
        symbol: str,
        in_sample_days: int,
        out_sample_days: int
    ) -> WalkForwardResult:
        """
        Runs the walk-forward analysis on historical data.
        
        Args:
            candles_df: DataFrame containing OHLCV data.
            symbol: Trading symbol.
            in_sample_days: Size of the optimization window in days.
            out_sample_days: Size of the testing window in days.
            
        Returns:
            WalkForwardResult with detailed metrics and robustness checks.
        """
        logger.info(f"Starting Walk-Forward Analysis for {self.strategy_class.__name__} on {symbol}")
        
        if candles_df.empty:
            raise ValueError("Empty dataframe provided for Walk-Forward Analysis.")
            
        # Ensure index is datetime for time-based slicing
        if not isinstance(candles_df.index, pd.DatetimeIndex):
            candles_df.index = pd.to_datetime(candles_df.index)
            
        start_date = candles_df.index.min()
        end_date = candles_df.index.max()
        
        total_days = (end_date - start_date).days
        if total_days < (in_sample_days + out_sample_days):
            raise ValueError(
                f"Dataset length ({total_days} days) is insufficient for "
                f"IS ({in_sample_days}) + OOS ({out_sample_days}) windows."
            )
            
        param_combinations = self._generate_param_combinations()
        logger.info(f"Generated {len(param_combinations)} parameter combinations to test.")
        
        windows = []
        best_params_per_window = []
        all_oos_trades = []
        
        current_start = start_date
        window_idx = 1
        
        while True:
            is_end = current_start + timedelta(days=in_sample_days)
            oos_end = is_end + timedelta(days=out_sample_days)
            
            if oos_end > end_date:
                # If the remaining data is not enough for a full OOS period, stop
                break
                
            logger.info(f"Processing Window {window_idx}: IS[{current_start.date()} to {is_end.date()}], OOS[{is_end.date()} to {oos_end.date()}]")
            
            is_data = candles_df[(candles_df.index >= current_start) & (candles_df.index < is_end)]
            oos_data = candles_df[(candles_df.index >= is_end) & (candles_df.index < oos_end)]
            
            if is_data.empty or oos_data.empty:
                logger.warning(f"Window {window_idx} skipped due to empty data slices.")
                current_start += timedelta(days=out_sample_days)
                continue
                
            # 1. Optimize on IS data
            best_sharpe = -float('inf')
            best_params = None
            is_metrics = None
            
            for params in param_combinations:
                strategy = self.strategy_class(**params)
                engine = BacktestEngine(
                    strategy=strategy,
                    initial_capital=self.initial_capital,
                    fee_calculator=self.fee_calculator,
                    settings=self.settings
                )
                
                result = await engine.run(is_data, symbol)
                sharpe = result.metrics.get("sharpe_ratio", 0.0)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    is_metrics = result.metrics
                    
            if not best_params:
                # Fallback if no trades or no positive sharpe was found
                best_params = param_combinations[0]
                logger.warning(f"No optimal params found for Window {window_idx}. Falling back to default.")
                
            best_params_per_window.append({
                "window": window_idx,
                "params": best_params,
                "is_sharpe": best_sharpe
            })
            
            # 2. Test best params on OOS data
            strategy_oos = self.strategy_class(**best_params)
            engine_oos = BacktestEngine(
                strategy=strategy_oos,
                initial_capital=self.initial_capital,
                fee_calculator=self.fee_calculator,
                settings=self.settings
            )
            
            oos_result = await engine_oos.run(oos_data, symbol)
            oos_metrics = oos_result.metrics
            
            # Record trades for aggregate OOS evaluation
            all_oos_trades.extend(oos_result.trades)
            
            windows.append({
                "window_idx": window_idx,
                "is_start": current_start,
                "is_end": is_end,
                "oos_start": is_end,
                "oos_end": oos_end,
                "best_params": best_params,
                "is_metrics": is_metrics,
                "oos_metrics": oos_metrics
            })
            
            # Roll forward by out_sample_days
            current_start += timedelta(days=out_sample_days)
            window_idx += 1
            
        # 3. Compute aggregate OOS metrics
        if all_oos_trades:
            aggregate_metrics_engine = PerformanceMetrics(all_oos_trades, self.initial_capital)
            aggregate_oos_metrics = aggregate_metrics_engine.summary()
        else:
            aggregate_oos_metrics = {}
            
        # 4. Assess robustness
        # Criteria: Aggregate OOS Sharpe > 1.0 and consistent profitability
        agg_sharpe = aggregate_oos_metrics.get("sharpe_ratio", 0.0)
        positive_oos_windows = sum(1 for w in windows if w["oos_metrics"].get("total_pnl", 0) > 0)
        consistency_ratio = positive_oos_windows / len(windows) if windows else 0
        
        is_robust = (agg_sharpe > 1.0) and (consistency_ratio >= 0.5)
        
        result = WalkForwardResult(
            strategy_name=self.strategy_class.__name__,
            symbol=symbol,
            windows=windows,
            aggregate_oos_metrics=aggregate_oos_metrics,
            best_params_per_window=best_params_per_window,
            is_robust=is_robust
        )
        
        logger.info(f"Walk-Forward Analysis completed. Robust: {is_robust} (Sharpe: {agg_sharpe:.2f})")
        return result
