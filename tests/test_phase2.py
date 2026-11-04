"""Test suite for Phase 2: Market Regime Filter, Scanner, and Dynamic Trailing Stops."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta

from src.core.enums import SignalAction, Side, Exchange, TimeFrame
from src.core.models import Candle
from src.data.market_regime import MarketRegime, MarketRegimeDetector
from src.data.scanner import MarketScanner


def generate_mock_candles(trend: str = "bullish", n: int = 40) -> list[Candle]:
    """Helper to generate realistic candles for testing."""
    candles = []
    base_price = 24000.0 if trend != "bearish" else 24300.0
    start_time = datetime(2026, 11, 4, 9, 15)

    for i in range(n):
        if trend == "bullish":
            price = base_price + (i * 6.0)
            high = price + 8.0
            low = price - 4.0
        elif trend == "bearish":
            price = base_price - (i * 6.0)
            high = price + 4.0
            low = price - 8.0
        else:  # sideways
            price = base_price + (5.0 if i % 2 == 0 else -5.0)
            high = price + 6.0
            low = price - 6.0

        candles.append(
            Candle(
                symbol="NIFTY 50",
                exchange=Exchange.NSE,
                timeframe=TimeFrame.M5,
                timestamp=start_time + timedelta(minutes=5 * i),
                open=price,
                high=high,
                low=low,
                close=price,
                volume=100000 + (i * 1000),
            )
        )
    return candles


@pytest.mark.asyncio
async def test_regime_detector_bullish():
    detector = MarketRegimeDetector()
    detector.feed.get_historical_candles = AsyncMock(return_value=generate_mock_candles("bullish", 40))

    regime, details = await detector.get_current_regime(force_refresh=True)
    assert regime == MarketRegime.BULLISH_TREND

    # Check trade validation
    allowed_buy, msg_buy = await detector.validate_trade("INFY", SignalAction.BUY)
    assert allowed_buy is True
    assert "Permitted" in msg_buy

    allowed_sell, msg_sell = await detector.validate_trade("INFY", SignalAction.SELL)
    assert allowed_sell is False
    assert "Rejected" in msg_sell


@pytest.mark.asyncio
async def test_regime_detector_bearish():
    detector = MarketRegimeDetector()
    detector.feed.get_historical_candles = AsyncMock(return_value=generate_mock_candles("bearish", 40))

    regime, details = await detector.get_current_regime(force_refresh=True)
    assert regime == MarketRegime.BEARISH_TREND

    # In bearish regime: BUY must be blocked, SHORT permitted
    allowed_buy, msg_buy = await detector.validate_trade("TCS", SignalAction.BUY)
    assert allowed_buy is False
    assert "Rejected" in msg_buy

    allowed_sell, msg_sell = await detector.validate_trade("TCS", SignalAction.SELL)
    assert allowed_sell is True
    assert "Permitted" in msg_sell


@pytest.mark.asyncio
async def test_scanner_symbol_and_ranking():
    scanner = MarketScanner(universe=["RELIANCE", "INFY"])

    # RELIANCE has high momentum
    scanner.feed.get_historical_candles = AsyncMock(side_effect=lambda symbol, **kwargs: (
        generate_mock_candles("bullish", 35) if symbol == "RELIANCE" else generate_mock_candles("sideways", 35)
    ))

    ranked = await scanner.scan_universe()
    assert len(ranked) == 2
    # Strong bullish stock should have higher composite score than sideways stock
    assert ranked[0]["symbol"] == "RELIANCE"
    assert ranked[0]["composite_score"] > ranked[1]["composite_score"]

    top = await scanner.get_top_watchlist(n=1)
    assert top == ["RELIANCE"]
