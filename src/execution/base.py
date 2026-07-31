"""Base interfaces for the execution layer."""

from abc import ABC, abstractmethod

from src.core.models import Order, PortfolioState, Position


class BrokerInterface(ABC):
    """Abstract base class for all broker integrations (live and mock)."""

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Place an order with the broker.

        Args:
            order: The requested order.
            
        Returns:
            The order with an updated status and ID.
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> Order:
        """Cancel a pending order.

        Args:
            order_id: The broker's order ID.
            
        Returns:
            The cancelled order.
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get the current status of an order.

        Args:
            order_id: The broker's order ID.
            
        Returns:
            The current order state.
        """
        pass

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Fetch all currently open positions.

        Returns:
            A list of open positions.
        """
        pass

    @abstractmethod
    async def get_portfolio(self) -> PortfolioState:
        """Get the current portfolio state.

        Returns:
            The current snapshot of capital and positions.
        """
        pass

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """Indicates if this is a paper-trading engine."""
        pass
