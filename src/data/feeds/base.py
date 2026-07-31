from abc import ABC, abstractmethod
from typing import Callable, Any
from datetime import datetime

from src.core.models import Candle
from src.core.enums import TimeFrame, Exchange

class DataFeed(ABC):
    """
    Abstract base class for all data feeds.
    Provides standard interface for fetching historical and real-time market data.
    """

    @abstractmethod
    async def get_historical_candles(
        self,
        symbol: str,
        exchange: Exchange = Exchange.NSE,
        timeframe: TimeFrame = TimeFrame.M5,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Candle]:
        """
        Fetch historical candle data for a given symbol and time range.
        
        Args:
            symbol: Trading symbol
            exchange: Target exchange
            timeframe: Candle timeframe
            start: Start datetime
            end: End datetime
            
        Returns:
            List of Candle objects
        """
        pass

    @abstractmethod
    async def get_latest_candle(
        self,
        symbol: str,
        exchange: Exchange,
        timeframe: TimeFrame
    ) -> Candle:
        """
        Fetch the most recent candle for a given symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Target exchange
            timeframe: Candle timeframe
            
        Returns:
            The latest Candle object
        """
        pass

    @abstractmethod
    async def subscribe(
        self,
        symbol: str,
        exchange: Exchange,
        callback: Callable[[Candle], Any]
    ) -> None:
        """
        Subscribe to real-time candle updates for a symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Target exchange
            callback: Async function to call when a new candle arrives
        """
        pass

    @abstractmethod
    async def unsubscribe(self, symbol: str, exchange: Exchange) -> None:
        """
        Unsubscribe from real-time candle updates.
        
        Args:
            symbol: Trading symbol
            exchange: Target exchange
        """
        pass
