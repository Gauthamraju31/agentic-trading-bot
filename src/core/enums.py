"""Core enumerations for the trading bot."""

from enum import Enum


class TimeFrame(str, Enum):
    """Candle timeframe / resolution."""
    TICK = "tick"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN = "1M"


class Side(str, Enum):
    """Order / position side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type (aligned with Indian broker conventions)."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "SL"            # stop-loss limit
    STOP_LOSS_MARKET = "SL-M"   # stop-loss market


class OrderStatus(str, Enum):
    """Lifecycle status of an order."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class SignalAction(str, Enum):
    """Trading signal action."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"       # close existing position


class Exchange(str, Enum):
    """Indian exchanges and segments."""
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"     # NSE Futures & Options
    BFO = "BFO"     # BSE Futures & Options
    MCX = "MCX"     # Multi Commodity Exchange


class Segment(str, Enum):
    """Instrument segment."""
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"


class PositionType(str, Enum):
    """Position product type."""
    INTRADAY = "INTRADAY"   # MIS — squared off by EOD
    DELIVERY = "DELIVERY"   # CNC — held overnight


class AgentRole(str, Enum):
    """Roles in the multi-agent trading pipeline."""
    TECHNICAL_ANALYST = "technical_analyst"
    SENTIMENT_ANALYST = "sentiment_analyst"
    BULL = "bull"
    BEAR = "bear"
    RISK_MANAGER = "risk_manager"
    PORTFOLIO_MANAGER = "portfolio_manager"


class PositionSizingMethod(str, Enum):
    """Position sizing algorithm."""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY = "kelly"
    HALF_KELLY = "half_kelly"


class StopLossType(str, Enum):
    """Stop-loss strategy type."""
    FIXED_PCT = "fixed_pct"
    TRAILING_PCT = "trailing_pct"
    ATR_BASED = "atr_based"
    TRAILING_ATR = "trailing_atr"
