"""Async event bus for decoupled communication between trading bot components.

Components publish events (e.g., ORDER_FILLED, STOP_LOSS_TRIGGERED) and
other components subscribe to react — without tight coupling.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine

from loguru import logger


# Type alias for async event handlers
EventHandler = Callable[..., Coroutine[Any, Any, None]]


class EventBus:
    """Lightweight async pub/sub event bus.

    Usage:
        bus = EventBus()

        async def on_fill(order=None):
            print(f"Order filled: {order}")

        bus.subscribe(Events.ORDER_FILLED, on_fill)
        await bus.publish(Events.ORDER_FILLED, order=my_order)
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._event_log: list[dict[str, Any]] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register an async handler for an event type."""
        self._handlers[event_type].append(handler)
        logger.debug(f"EventBus: subscribed {handler.__name__} → '{event_type}'")

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a handler from an event type."""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)
            logger.debug(f"EventBus: unsubscribed {handler.__name__} from '{event_type}'")

    async def publish(self, event_type: str, **data: Any) -> None:
        """Publish an event to all subscribed handlers.

        Handlers are called concurrently via asyncio.gather. Exceptions
        in individual handlers are caught and logged, not propagated.
        """
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            return

        logger.debug(f"EventBus: publishing '{event_type}' → {len(handlers)} handler(s)")
        self._event_log.append({"event": event_type, "data": data})

        results = await asyncio.gather(
            *(handler(**data) for handler in handlers),
            return_exceptions=True,
        )
        for handler, result in zip(handlers, results):
            if isinstance(result, Exception):
                logger.error(
                    f"EventBus: handler {handler.__name__} raised {type(result).__name__}: {result}"
                )

    @property
    def event_count(self) -> int:
        """Total number of events published since creation."""
        return len(self._event_log)

    def clear(self) -> None:
        """Remove all subscriptions and clear event log."""
        self._handlers.clear()
        self._event_log.clear()


class Events:
    """Event type constants — prevents typos in string-based event names."""

    # Market data
    CANDLE_RECEIVED = "candle.received"
    TICK_RECEIVED = "tick.received"

    # Signals
    SIGNAL_GENERATED = "signal.generated"
    DECISION_MADE = "decision.made"

    # Orders
    ORDER_PLACED = "order.placed"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"

    # Positions
    POSITION_OPENED = "position.opened"
    POSITION_UPDATED = "position.updated"
    POSITION_CLOSED = "position.closed"

    # Risk
    STOP_LOSS_TRIGGERED = "stop_loss.triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit.triggered"
    TRAILING_STOP_UPDATED = "trailing_stop.updated"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker.triggered"

    # System
    TRADING_STARTED = "trading.started"
    TRADING_HALTED = "trading.halted"
    DAILY_REPORT = "daily.report"
    ERROR_OCCURRED = "error.occurred"


# ── Global singleton ─────────────────────────────────────────────────────────
# Import and use this instance across the application.
event_bus = EventBus()
