"""Daily Profit Goal & Risk Limit Manager.

Tracks cumulative daily net P&L (fee-adjusted), checks whether daily target (e.g. ₹100)
or daily stop-loss limit (e.g. -₹200) has been reached, and controls daily trading state.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from src.core.config import settings


class DailyGoalManager:
    """Manages daily profit targets, risk stops, and daily session state."""

    def __init__(self, target_file: Optional[Path] = None):
        self.target_file = target_file or Path("data/daily_goal_state.json")
        self.target_file.parent.mkdir(parents=True, exist_ok=True)

        cfg = getattr(settings, "daily_goal", None)
        self.target_profit: float = float(getattr(cfg, "target_profit", 100.0))
        self.max_loss: float = float(getattr(cfg, "max_loss", 200.0))
        self.auto_stop: bool = bool(getattr(cfg, "auto_stop_on_target", True))

        self.date_str = datetime.now().strftime("%Y-%m-%d")
        self.realized_pnl: float = 0.0
        self.trade_count: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.status: str = "PRE_MARKET"  # PRE_MARKET | TRADING_ACTIVE | GOAL_REACHED | STOP_LOSS_HIT | HALTED

        self._load_today_state()

    def _load_today_state(self):
        """Load state for today or reset if new date."""
        if self.target_file.exists():
            try:
                with open(self.target_file, "r") as f:
                    data = json.load(f)
                if data.get("date") == self.date_str:
                    self.realized_pnl = float(data.get("realized_pnl", 0.0))
                    self.trade_count = int(data.get("trade_count", 0))
                    self.winning_trades = int(data.get("winning_trades", 0))
                    self.losing_trades = int(data.get("losing_trades", 0))
                    self.status = str(data.get("status", "TRADING_ACTIVE"))
                    logger.info(
                        f"[DailyGoalManager] Resumed today's state ({self.date_str}): "
                        f"P&L = ₹{self.realized_pnl:.2f} / ₹{self.target_profit:.2f} | Status = {self.status}"
                    )
                    return
            except Exception as e:
                logger.warning(f"[DailyGoalManager] Could not parse state file: {e}")

        # Default init for new trading day
        self._save_state()

    def _save_state(self):
        """Persist today's state to disk."""
        data = {
            "date": self.date_str,
            "realized_pnl": round(self.realized_pnl, 2),
            "target_profit": self.target_profit,
            "max_loss": self.max_loss,
            "progress_pct": round(self.progress_pct, 1),
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "status": self.status,
            "last_updated": datetime.now().isoformat(),
        }
        try:
            with open(self.target_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[DailyGoalManager] Failed to save state: {e}")

    @property
    def progress_pct(self) -> float:
        """Percentage progress towards daily target profit."""
        if self.target_profit <= 0:
            return 100.0
        pct = (self.realized_pnl / self.target_profit) * 100.0
        return max(-100.0, min(200.0, pct))

    def update_pnl(self, trade_pnl: float) -> str:
        """Record a completed trade's net fee-adjusted P&L and evaluate daily goal."""
        self.trade_count += 1
        self.realized_pnl += trade_pnl

        if trade_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        logger.info(
            f"[DailyGoalManager] Recorded trade P&L: ₹{trade_pnl:+.2f} | "
            f"Daily Cumulative: ₹{self.realized_pnl:+.2f} / ₹{self.target_profit:.2f}"
        )

        # Check target profit reached
        if self.realized_pnl >= self.target_profit and self.auto_stop:
            self.status = "GOAL_REACHED"
            logger.success(
                f"🎉 DAILY PROFIT TARGET REACHED! Net P&L: ₹{self.realized_pnl:.2f} >= ₹{self.target_profit:.2f}. "
                "Halting trading for today to lock in gains!"
            )

        # Check max loss limit hit
        elif self.realized_pnl <= -abs(self.max_loss):
            self.status = "STOP_LOSS_HIT"
            logger.warning(
                f"🚨 DAILY MAX LOSS LIMIT HIT! Net P&L: ₹{self.realized_pnl:.2f} <= -₹{self.max_loss:.2f}. "
                "Halting trading for today to preserve capital!"
            )

        self._save_state()
        return self.status

    def set_config(self, target_profit: float, max_loss: float):
        """Update daily targets dynamically from dashboard UI."""
        self.target_profit = abs(target_profit)
        self.max_loss = abs(max_loss)
        logger.info(f"[DailyGoalManager] Updated targets: Target = ₹{self.target_profit:.2f}, Max Loss = ₹{self.max_loss:.2f}")
        self._save_state()

    def get_summary(self) -> Dict[str, Any]:
        """Return structured summary dict for dashboard UI and notifications."""
        return {
            "date": self.date_str,
            "realized_pnl": round(self.realized_pnl, 2),
            "target_profit": self.target_profit,
            "max_loss": self.max_loss,
            "progress_pct": round(self.progress_pct, 1),
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "status": self.status,
            "is_active": self.status in ("PRE_MARKET", "TRADING_ACTIVE"),
        }
