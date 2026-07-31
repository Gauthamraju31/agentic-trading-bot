"""Circuit breaker for emergency halts."""
import asyncio
from loguru import logger
from src.core.models import PortfolioState
from src.core.events import event_bus, Events
from src.core.config import settings

class CircuitBreaker:
    """Circuit breaker that halts trading on severe losses."""

    def __init__(self, max_daily_loss_pct: float | None = None):
        """Initialize the circuit breaker.
        
        Args:
            max_daily_loss_pct: Maximum daily loss percentage before halting.
                                Reads from config if not provided.
        """
        self._max_daily_loss_pct = (
            max_daily_loss_pct if max_daily_loss_pct is not None
            else settings.risk.max_daily_loss_pct
        )
        self._is_active = False
        self._trigger_reason = ""
        self._critical_drawdown_pct = 10.0

    @property
    def is_active(self) -> bool:
        """Returns True if the circuit breaker has been triggered."""
        return self._is_active

    def check(self, portfolio: PortfolioState) -> tuple[bool, str]:
        """Check if circuit breaker should trigger based on portfolio state.

        Args:
            portfolio: Current portfolio state.

        Returns:
            Tuple of (is_triggered, reason).
        """
        if self._is_active:
            return True, self._trigger_reason

        if portfolio.initial_capital <= 0:
            return False, ""

        daily_pnl = portfolio.daily_pnl + portfolio.total_unrealized_pnl
        daily_loss_pct = 0.0
        if daily_pnl < 0:
            daily_loss_pct = abs(daily_pnl) / portfolio.initial_capital * 100.0

        if daily_loss_pct >= self._max_daily_loss_pct:
            reason = f"Daily loss ({daily_loss_pct:.2f}%) exceeds max allowed ({self._max_daily_loss_pct:.2f}%)"
            self.force_trigger(reason)
            return True, reason

        if portfolio.max_drawdown_pct >= self._critical_drawdown_pct:
            reason = f"Drawdown ({portfolio.max_drawdown_pct:.2f}%) exceeds critical threshold ({self._critical_drawdown_pct:.2f}%)"
            self.force_trigger(reason)
            return True, reason

        return False, ""

    def force_trigger(self, reason: str) -> None:
        """Manually trigger the circuit breaker.

        Args:
            reason: Reason for triggering.
        """
        if not self._is_active:
            self._is_active = True
            self._trigger_reason = reason
            logger.error(f"CIRCUIT BREAKER TRIGGERED: {reason}")
            # Ensure event bus publish doesn't block sync execution, fire and forget if called sync
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(event_bus.publish(Events.CIRCUIT_BREAKER_TRIGGERED, {"reason": reason}))
            except RuntimeError:
                # No running event loop
                pass

    def reset(self) -> None:
        """Reset the circuit breaker for a new trading day."""
        if self._is_active:
            logger.info("Circuit breaker reset")
            self._is_active = False
            self._trigger_reason = ""
