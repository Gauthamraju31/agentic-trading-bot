import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Callable, Any, Dict
from loguru import logger

from src.core.models import Candle
from src.core.enums import TimeFrame, Exchange
from src.data.feeds.base import DataFeed

class CSVDataFeed(DataFeed):
    """
    Concrete DataFeed implementation that reads historical candle data from CSV files.
    Primarily used for backtesting.
    """
    def __init__(self, data_dir: str | Path = "data/historical"):
        path = Path(data_dir)
        if path.is_file():
            self.data_file = path
            self.data_dir = path.parent
        else:
            self.data_file = None
            self.data_dir = path
            self.data_dir.mkdir(parents=True, exist_ok=True)
        self._subscriptions: Dict[str, Callable[[Candle], Any]] = {}

    def _get_file_path(self, symbol: str, timeframe: TimeFrame) -> Path:
        """Helper to resolve CSV file path based on symbol and timeframe."""
        if self.data_file and self.data_file.exists():
            return self.data_file
        return self.data_dir / f"{symbol.replace(' ', '_')}_{timeframe.value}.csv"

    async def get_historical_candles(
        self,
        symbol: str,
        exchange: Exchange = Exchange.NSE,
        timeframe: TimeFrame = TimeFrame.M5,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Candle]:
        file_path = self._get_file_path(symbol, timeframe)
        if not file_path.exists():
            logger.error(f"Data file not found: {file_path}")
            return []

        try:
            df = pd.read_csv(file_path, parse_dates=["timestamp"])
            
            # Ensure required columns exist
            required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
            if not required_cols.issubset(df.columns):
                logger.error(f"CSV {file_path.name} missing required columns. Expected: {required_cols}")
                return []
                
            filtered_df = df
            if start is not None:
                filtered_df = filtered_df[filtered_df["timestamp"] >= pd.to_datetime(start)]
            if end is not None:
                filtered_df = filtered_df[filtered_df["timestamp"] <= pd.to_datetime(end)]

            candles = []
            for _, row in filtered_df.iterrows():
                candles.append(
                    Candle(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        timestamp=row["timestamp"].to_pydatetime(),
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"]
                    )
                )
            return candles
        except Exception as e:
            logger.error(f"Error reading CSV feed for {symbol}: {e}")
            return []

    async def get_latest_candle(
        self,
        symbol: str,
        exchange: Exchange,
        timeframe: TimeFrame
    ) -> Candle:
        file_path = self._get_file_path(symbol, timeframe)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            df = pd.read_csv(file_path, parse_dates=["timestamp"])
            if df.empty:
                raise ValueError(f"No data found in {file_path}")
                
            row = df.iloc[-1]
            return Candle(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                timestamp=row["timestamp"].to_pydatetime(),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"]
            )
        except Exception as e:
            logger.error(f"Error reading latest candle from {file_path}: {e}")
            raise

    async def subscribe(
        self,
        symbol: str,
        exchange: Exchange,
        callback: Callable[[Candle], Any]
    ) -> None:
        """
        Subscribe to data (Mock implementation for CSV feed).
        """
        sub_key = f"{exchange.value}:{symbol}"
        self._subscriptions[sub_key] = callback
        logger.info(f"Subscribed to CSV feed for {sub_key} (Note: CSV feed is static)")

    async def unsubscribe(self, symbol: str, exchange: Exchange) -> None:
        """
        Unsubscribe from data.
        """
        sub_key = f"{exchange.value}:{symbol}"
        if sub_key in self._subscriptions:
            del self._subscriptions[sub_key]
            logger.info(f"Unsubscribed from CSV feed for {sub_key}")
