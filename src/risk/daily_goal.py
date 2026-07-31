"""Daily Target Profit Goal and Loss Threshold Controller."""

from typing import Tuple
from loguru import logger
from src.core.models import PortfolioState

class DailyGoalController:
    """Monitors daily P&L against daily profit targets and maximum loss thresholds."""

    def __init__(
        self,
        daily_budget: float = 100000.0,
        target_profit: float = 2000.0,
        max_loss_threshold: float = 1000.0
    ):
        self.daily_budget = daily_budget
        self.target_profit = target_profit
        self.max_loss_threshold = abs(max_loss_threshold)
        self.is_halted = False
        self.halt_reason = ""
        self.target_achieved = False

    def update_targets(self, budget: float, target: float, max_loss: float) -> None:
        """Dynamically update daily budget and target goals."""
        self.daily_budget = budget
        self.target_profit = target
        self.max_loss_threshold = abs(max_loss)
        self.is_halted = False
        self.halt_reason = ""
        self.target_achieved = False
        logger.info(
            f"[DailyGoalController] Goals set: Budget=₹{self.daily_budget:,.2f} | "
            f"Target=+[₹{self.target_profit:,.2f}] | Max Loss=-[₹{self.max_loss_threshold:,.2f}]"
        )

    def evaluate_portfolio(self, portfolio: PortfolioState) -> Tuple[bool, str]:
        """Evaluate daily P&L against target goal and loss threshold.

        Returns:
            Tuple of (should_halt: bool, reason_message: str)
        """
        if self.is_halted:
            return True, self.halt_reason

        daily_pnl = portfolio.daily_pnl

        # Check Target Profit Goal
        if daily_pnl >= self.target_profit:
            self.is_halted = True
            self.target_achieved = True
            self.halt_reason = (
                f"🎉 Daily Profit Target Goal Achieved! Net Daily P&L (+₹{daily_pnl:,.2f}) "
                f"reached target (+₹{self.target_profit:,.2f}). Locking in profits and halting bot for today."
            )
            logger.info(f"[DailyGoalController] {self.halt_reason}")
            return True, self.halt_reason

        # Check Maximum Loss Threshold
        if daily_pnl <= -self.max_loss_threshold:
            self.is_halted = True
            self.target_achieved = False
            self.halt_reason = (
                f"⚠️ Daily Stop-Loss Threshold Hit! Net Daily Loss (-₹{abs(daily_pnl):,.2f}) "
                f"exceeded maximum loss threshold (-₹{self.max_loss_threshold:,.2f}). "
                f"Halting bot to protect remaining daily budget."
            )
            logger.warning(f"[DailyGoalController] {self.halt_reason}")
            return True, self.halt_reason

        return False, "Operating within daily goal parameters."

    def reset_day(self) -> None:
        """Reset goal tracker for a new trading day."""
        self.is_halted = False
        self.halt_reason = ""
        self.target_achieved = False
