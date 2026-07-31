"""Core package — enums, models, events, and configuration."""

from .config import Settings, load_settings, settings
from .enums import (
    AgentRole,
    Exchange,
    OrderStatus,
    OrderType,
    PositionSizingMethod,
    PositionType,
    Segment,
    Side,
    SignalAction,
    StopLossType,
    TimeFrame,
)
from .events import EventBus, Events, event_bus
from .models import (
    AgentOpinion,
    Candle,
    IndicatorValues,
    MarketContext,
    Order,
    OrderFees,
    PortfolioState,
    Position,
    Signal,
    Trade,
    TradingDecision,
)

__all__ = [
    # Config
    "Settings",
    "load_settings",
    "settings",
    # Enums
    "AgentRole",
    "Exchange",
    "OrderStatus",
    "OrderType",
    "PositionSizingMethod",
    "PositionType",
    "Segment",
    "Side",
    "SignalAction",
    "StopLossType",
    "TimeFrame",
    # Events
    "EventBus",
    "Events",
    "event_bus",
    # Models
    "AgentOpinion",
    "Candle",
    "IndicatorValues",
    "MarketContext",
    "Order",
    "OrderFees",
    "PortfolioState",
    "Position",
    "Signal",
    "Trade",
    "TradingDecision",
]
