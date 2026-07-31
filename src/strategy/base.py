"""Base strategy definition."""
from abc import ABC, abstractmethod
from src.core.models import Candle, IndicatorValues, PortfolioState, Signal

class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the strategy."""
        pass

    @property
    @abstractmethod
    def required_history(self) -> int:
        """Minimum number of candles needed."""
        pass

    @property
    @abstractmethod
    def required_indicators(self) -> list[str]:
        """List of indicator names this strategy uses."""
        pass

    @abstractmethod
    def generate_signals(
        self,
        candles: list[Candle],
        indicators: IndicatorValues,
        portfolio: PortfolioState,
    ) -> list[Signal]:
        """Generate trading signals based on current data.

        Args:
            candles: List of historical candles.
            indicators: Current indicator values.
            portfolio: Current portfolio state.

        Returns:
            List of generated signals.
        """
        pass
