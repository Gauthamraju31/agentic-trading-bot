"""NIFTY 50 Index Market Regime Detector.

Filters individual equity trades based on broader market direction and trend strength.
Prevents counter-trend equity buys during market sell-offs and chops.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
import pandas as pd

from src.core.enums import Exchange, SignalAction, TimeFrame
from src.core.models import Candle
from src.data.feeds.live_feed import LiveDataFeed
from src.data.indicators import IndicatorEngine


class MarketRegime(str, Enum):
    BULLISH_TREND = "BULLISH_TREND"
    BEARISH_TREND = "BEARISH_TREND"
    SIDEWAYS_CHOP = "SIDEWAYS_CHOP"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNKNOWN = "UNKNOWN"


class MarketRegimeDetector:
    """Monitors NIFTY 50 index to determine overall market regime and validate trade signals."""

    def __init__(self, cache_ttl_seconds: int = 60):
        self.feed = LiveDataFeed()
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._last_checked: Optional[datetime] = None
        self._cached_regime: MarketRegime = MarketRegime.UNKNOWN
        self._cached_details: Dict[str, Any] = {}

    async def get_current_regime(self, force_refresh: bool = False) -> Tuple[MarketRegime, Dict[str, Any]]:
        """Fetch latest NIFTY 50 candles and calculate current market regime."""
        now = datetime.now()
        if (
            not force_refresh
            and self._last_checked is not None
            and (now - self._last_checked) < self.cache_ttl
            and self._cached_regime != MarketRegime.UNKNOWN
        ):
            return self._cached_regime, self._cached_details

        try:
            candles = await self.feed.get_historical_candles(
                symbol="NIFTY 50",
                exchange=Exchange.NSE,
                timeframe=TimeFrame.M5,
            )

            if not candles or len(candles) < 30:
                logger.warning(f"[MarketRegime] Insufficient NIFTY candles ({len(candles) if candles else 0})")
                return self._cached_regime or MarketRegime.UNKNOWN, self._cached_details

            df = pd.DataFrame([c.model_dump() for c in candles])
            df_ind = IndicatorEngine.calculate(df)
            latest = df_ind.iloc[-1]

            def _safe_float(val, default_val):
                if val is None or pd.isna(val):
                    return default_val
                return float(val)

            close = float(latest["close"])
            ema_20 = _safe_float(latest.get("EMA_20"), close)
            ema_50 = _safe_float(latest.get("EMA_50"), ema_20)
            rsi = _safe_float(latest.get("RSI_14"), 50.0)
            adx = _safe_float(latest.get("ADX_14"), 20.0)
            supertrend = _safe_float(latest.get("SUPERTREND"), 0.0)

            # Calculate day range / trend strength
            day_open = float(df["open"].iloc[0])
            day_return_pct = ((close - day_open) / day_open) * 100.0

            # Classification logic
            regime = MarketRegime.SIDEWAYS_CHOP
            reason = "Market is consolidating in a range"

            # Trend identification
            is_bullish = close > ema_20 and ema_20 >= (ema_50 * 0.998) and rsi >= 48.0
            is_bearish = close < ema_20 and ema_20 <= (ema_50 * 1.002) and rsi <= 52.0

            if abs(day_return_pct) > 2.5 or (adx > 55 and rsi not in range(25, 75) and abs(day_return_pct) > 2.0):
                regime = MarketRegime.HIGH_VOLATILITY
                reason = f"High volatility event: NIFTY intraday move = {day_return_pct:+.2f}%, ADX = {adx:.1f}"
            elif is_bullish and (adx > 20 or supertrend > 0):
                regime = MarketRegime.BULLISH_TREND
                reason = f"NIFTY in Bullish Trend: Close (₹{close:.1f}) > 20 EMA (₹{ema_20:.1f}), RSI = {rsi:.1f}, ADX = {adx:.1f}"
            elif is_bearish and (adx > 20 or supertrend < 0):
                regime = MarketRegime.BEARISH_TREND
                reason = f"NIFTY in Bearish Trend: Close (₹{close:.1f}) < 20 EMA (₹{ema_20:.1f}), RSI = {rsi:.1f}, ADX = {adx:.1f}"
            else:
                regime = MarketRegime.SIDEWAYS_CHOP
                reason = f"NIFTY Rangebound / Chop: ADX = {adx:.1f}, RSI = {rsi:.1f}, Close near EMAs"

            details = {
                "regime": regime.value,
                "reason": reason,
                "nifty_close": close,
                "ema_20": ema_20,
                "ema_50": ema_50,
                "rsi_14": rsi,
                "adx_14": adx,
                "day_return_pct": day_return_pct,
                "supertrend_bullish": supertrend > 0,
                "timestamp": now.isoformat(),
            }

            self._last_checked = now
            self._cached_regime = regime
            self._cached_details = details

            logger.info(f"[MarketRegime] NIFTY 50: {regime.value} | {reason}")
            return regime, details

        except Exception as e:
            logger.error(f"[MarketRegime] Error detecting regime: {e}")
            return self._cached_regime, self._cached_details

    async def validate_trade(self, symbol: str, action: SignalAction) -> Tuple[bool, str]:
        """Validate if an equity trade signal is permitted under current NIFTY regime.

        Rules:
        - If NIFTY is BEARISH_TREND, reject BUY orders (don't catch falling knives).
        - If NIFTY is BULLISH_TREND, reject SELL/SHORT orders (don't short strong market).
        - If NIFTY is SIDEWAYS_CHOP, permit trades only if stock ADX is strongly independent.
        - If NIFTY is HIGH_VOLATILITY, permit with caution or reject extreme moves.
        """
        regime, details = await self.get_current_regime()

        if regime == MarketRegime.UNKNOWN:
            # If regime cannot be determined, allow trade with warning
            return True, "Regime unknown; allowing trade with baseline risk"

        if action == SignalAction.BUY:
            if regime == MarketRegime.BEARISH_TREND:
                return False, f"Trade Rejected: Cannot BUY {symbol} while NIFTY 50 is in a BEARISH_TREND (RSI: {details.get('rsi_14', 0):.1f})"
            return True, f"Trade Permitted: NIFTY regime is {regime.value}"

        elif action == SignalAction.SELL:
            if regime == MarketRegime.BULLISH_TREND:
                return False, f"Trade Rejected: Cannot SHORT {symbol} while NIFTY 50 is in a BULLISH_TREND (RSI: {details.get('rsi_14', 0):.1f})"
            return True, f"Trade Permitted: NIFTY regime is {regime.value}"

        return False, f"Action {action} requires no entry validation"
