"""Multi-Stock Opportunity Scanner for NIFTY Liquid Equities.

Scans a universe of high-liquidity large-cap NSE stocks to identify
the highest-momentum, cleanest-trending setups for intraday trading.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
import pandas as pd

from src.core.enums import Exchange, TimeFrame
from src.core.models import Candle
from src.data.feeds.live_feed import LiveDataFeed
from src.data.indicators import IndicatorEngine

DEFAULT_SCANNER_UNIVERSE = [
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "BHARTIARTL",
    "LT",
    "AXISBANK",
    "TATAMOTORS",
]


class MarketScanner:
    """Scans liquid NSE universe and ranks candidate stocks based on momentum and volume."""

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        feed: Optional[LiveDataFeed] = None,
    ):
        self.universe = universe or DEFAULT_SCANNER_UNIVERSE
        self.feed = feed or LiveDataFeed()

    async def scan_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Compute momentum, volatility, and volume indicators for a single symbol."""
        try:
            candles = await self.feed.get_historical_candles(
                symbol=symbol,
                exchange=Exchange.NSE,
                timeframe=TimeFrame.M5,
            )

            if not candles or len(candles) < 28:
                return None

            df = pd.DataFrame([c.model_dump() for c in candles])
            df_ind = IndicatorEngine.calculate(df)
            latest = df_ind.iloc[-1]

            close = float(latest["close"])
            ema_9 = float(latest.get("EMA_9", close))
            ema_21 = float(latest.get("EMA_21", close))
            rsi = float(latest.get("RSI_14", 50.0))
            adx = float(latest.get("ADX_14", 20.0))
            atr = float(latest.get("ATR_14", close * 0.01))
            vol = float(latest["volume"])
            vol_sma = float(latest.get("VOL_SMA_20", latest.get("VOLUME_SMA_20", vol or 1.0)))

            rvol = round(vol / max(1.0, vol_sma), 2)
            atr_pct = round((atr / close) * 100.0, 2)

            # Trend direction
            trend = "NEUTRAL"
            if ema_9 > ema_21 and rsi > 50:
                trend = "BULLISH"
            elif ema_9 < ema_21 and rsi < 50:
                trend = "BEARISH"

            # Composite Score (0.0 to 1.0+)
            # Rewards strong ADX trend, healthy RSI deviation, and high relative volume
            adx_component = min(1.0, adx / 50.0) * 0.35
            rsi_component = min(1.0, abs(rsi - 50.0) / 25.0) * 0.30
            rvol_component = min(1.5, rvol / 1.5) * 0.35
            composite_score = round(adx_component + rsi_component + rvol_component, 3)

            return {
                "symbol": symbol,
                "close": close,
                "trend": trend,
                "composite_score": composite_score,
                "adx": round(adx, 1),
                "rsi": round(rsi, 1),
                "rvol": rvol,
                "atr": round(atr, 2),
                "atr_pct": atr_pct,
                "timestamp": latest.get("timestamp", datetime.now()).isoformat() if hasattr(latest.get("timestamp"), "isoformat") else str(latest.get("timestamp")),
            }
        except Exception as e:
            logger.warning(f"[MarketScanner] Error scanning {symbol}: {e}")
            return None

    async def scan_universe(self) -> List[Dict[str, Any]]:
        """Scan all universe symbols concurrently and return ranked results."""
        logger.info(f"[MarketScanner] Scanning {len(self.universe)} universe stocks...")
        tasks = [self.scan_symbol(sym) for sym in self.universe]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = [r for r in results if isinstance(r, dict) and r is not None]
        # Sort by composite_score descending
        ranked = sorted(valid_results, key=lambda x: x["composite_score"], reverse=True)

        logger.info(f"[MarketScanner] Completed scan. Found {len(ranked)} active candidates.")
        return ranked

    async def get_top_watchlist(self, n: int = 3) -> List[str]:
        """Return the top `n` symbols with the strongest setups."""
        ranked = await self.scan_universe()
        if not ranked:
            return self.universe[:n]
        return [c["symbol"] for c in ranked[:n]]
