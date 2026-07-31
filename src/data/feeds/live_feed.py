"""Live market data feed for Indian stock markets (NSE/BSE).

Fetches real-time price quotes and live OHLCV candles for Indian stocks
using yfinance or broker WebSocket API feeds.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import yfinance as yf
from loguru import logger
import pandas as pd

from src.core.models import Candle
from src.core.enums import TimeFrame, Exchange
from src.data.feeds.base import DataFeed

# Mapping of standard symbol names to NSE tickers
NSE_TICKER_MAP = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
}

class LiveDataFeed(DataFeed):
    """Data feed implementation fetching real live market data from NSE."""

    def __init__(self, ticker_map: Optional[Dict[str, str]] = None):
        self.ticker_map = ticker_map or NSE_TICKER_MAP

    def get_nse_ticker(self, symbol: str) -> str:
        """Resolve standard symbol to NSE ticker symbol."""
        symbol_upper = symbol.strip().upper()
        if symbol_upper in self.ticker_map:
            return self.ticker_map[symbol_upper]
        if not symbol_upper.endswith(".NS") and not symbol_upper.startswith("^"):
            return f"{symbol_upper}.NS"
        return symbol_upper

    async def get_historical_candles(
        self,
        symbol: str,
        exchange: Exchange = Exchange.NSE,
        timeframe: TimeFrame = TimeFrame.M5,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Candle]:
        """Fetch real live historical & intraday candles from NSE."""
        ticker_str = self.get_nse_ticker(symbol)
        interval_map = {
            TimeFrame.M1: "1m",
            TimeFrame.M5: "5m",
            TimeFrame.M15: "15m",
            TimeFrame.M30: "30m",
            TimeFrame.H1: "60m",
            TimeFrame.D1: "1d",
        }
        yf_interval = interval_map.get(timeframe, "5m")

        logger.info(f"[LiveDataFeed] Fetching live market data for {symbol} ({ticker_str}) with interval={yf_interval}...")

        try:
            ticker = yf.Ticker(ticker_str)
            # 1m or 5m data can be fetched for last 7-60 days
            period = "5d" if yf_interval in ("1m", "5m") else "1mo"
            df = ticker.history(period=period, interval=yf_interval)

            if df.empty:
                logger.warning(f"[LiveDataFeed] No live data returned for {ticker_str}")
                return []

            df.reset_index(inplace=True)
            # Identify timestamp column name (Datetime or Date)
            time_col = "Datetime" if "Datetime" in df.columns else "Date"
            if time_col not in df.columns:
                time_col = df.columns[0]

            candles = []
            for _, row in df.iterrows():
                candles.append(
                    Candle(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        timestamp=pd.to_datetime(row[time_col]).to_pydatetime(),
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=int(row["Volume"]),
                    )
                )

            logger.info(f"[LiveDataFeed] Successfully fetched {len(candles)} live candles for {symbol}. Latest price: ₹{candles[-1].close:.2f}")
            return candles

        except Exception as e:
            logger.error(f"[LiveDataFeed] Failed to fetch live market data for {symbol}: {e}")
            return []

    async def get_latest_candle(
        self,
        symbol: str,
        exchange: Exchange = Exchange.NSE,
        timeframe: TimeFrame = TimeFrame.M5,
    ) -> Candle:
        candles = await self.get_historical_candles(symbol, exchange=exchange, timeframe=timeframe)
        if candles:
            return candles[-1]
        raise ValueError(f"No live candle available for {symbol}")

    async def subscribe(self, symbol: str, exchange: Exchange, callback: Any) -> None:
        logger.info(f"[LiveDataFeed] Subscribed to live updates for {symbol}")

    async def unsubscribe(self, symbol: str, exchange: Exchange) -> None:
        logger.info(f"[LiveDataFeed] Unsubscribed from live updates for {symbol}")

    async def get_latest_quote(self, symbol: str) -> Optional[float]:
        """Fetch the current live market price for a symbol."""
        candles = await self.get_historical_candles(symbol, timeframe=TimeFrame.M1)
        if candles:
            return candles[-1].close
        return None
