"""API Quota Tracker & Rate Limiter for Google Gemini / Antigravity LLM calls."""

from datetime import datetime, date
import json
from pathlib import Path
from typing import Dict, Any
from loguru import logger

QUOTA_FILE = Path("data/quota_tracker.json")

class QuotaTracker:
    """Tracks LLM request quotas (RPM/RPD), prevents quota exhaustion, and saves daily usage stats."""

    def __init__(
        self,
        max_rpd: int = 1000,
        max_rpm: int = 30,
        quota_file: Path = QUOTA_FILE
    ):
        self.max_rpd = max_rpd
        self.max_rpm = max_rpm
        self.quota_file = quota_file
        self.today_str = date.today().isoformat()
        self.daily_requests = 0
        self.minute_requests = 0
        self.last_minute_timestamp = datetime.now().minute
        self.load_quota()

    def load_quota(self) -> None:
        """Load today's quota usage from disk."""
        if self.quota_file.exists():
            try:
                with open(self.quota_file, "r") as f:
                    data = json.load(f)
                    if data.get("date") == self.today_str:
                        self.daily_requests = data.get("daily_requests", 0)
                    else:
                        # Reset for new trading day
                        self.daily_requests = 0
                logger.info(f"[QuotaTracker] Loaded today's LLM usage: {self.daily_requests}/{self.max_rpd} RPD.")
            except Exception as e:
                logger.warning(f"[QuotaTracker] Could not load quota file: {e}")

    def save_quota(self) -> None:
        """Save quota usage state to disk."""
        try:
            self.quota_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.quota_file, "w") as f:
                json.dump({
                    "date": self.today_str,
                    "daily_requests": self.daily_requests,
                    "max_rpd": self.max_rpd,
                    "remaining_rpd": max(0, self.max_rpd - self.daily_requests),
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"[QuotaTracker] Could not save quota file: {e}")

    def can_make_request(self) -> bool:
        """Check if request is within safe RPM and RPD quota limits."""
        curr_date = date.today().isoformat()
        if curr_date != self.today_str:
            self.today_str = curr_date
            self.daily_requests = 0

        curr_minute = datetime.now().minute
        if curr_minute != self.last_minute_timestamp:
            self.last_minute_timestamp = curr_minute
            self.minute_requests = 0

        if self.daily_requests >= self.max_rpd:
            logger.warning(f"[QuotaTracker] Daily Quota Limit Reached ({self.daily_requests}/{self.max_rpd} RPD)! Pausing LLM requests.")
            return False

        if self.minute_requests >= self.max_rpm:
            logger.warning(f"[QuotaTracker] Minute Rate Limit Reached ({self.minute_requests}/{self.max_rpm} RPM)! Pausing brief moment.")
            return False

        return True

    def record_request(self, count: int = 1) -> None:
        """Record an LLM request."""
        self.daily_requests += count
        self.minute_requests += count
        self.save_quota()

    def get_summary(self) -> Dict[str, Any]:
        """Return quota statistics dictionary."""
        remaining = max(0, self.max_rpd - self.daily_requests)
        used_pct = round((self.daily_requests / max(1, self.max_rpd)) * 100.0, 1)
        return {
            "date": self.today_str,
            "daily_requests": self.daily_requests,
            "max_rpd": self.max_rpd,
            "remaining_rpd": remaining,
            "usage_pct": used_pct,
            "is_quota_safe": self.daily_requests < (self.max_rpd * 0.95)
        }
