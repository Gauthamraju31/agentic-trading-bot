"""Indian Stock Market (NSE/BSE) Hours, Timezone & Holiday Utility.

Ensures the bot strictly adheres to IST timezone (Asia/Kolkata), official NSE trading hours
(09:00 - 15:30 IST), excludes weekends (Saturday & Sunday), and respects official NSE trading holidays.
"""

from datetime import datetime, time, date
import zoneinfo
from typing import Tuple, Dict, Any, List
from loguru import logger

# Official IST Timezone
IST_TZ = zoneinfo.ZoneInfo("Asia/Kolkata")

# Official NSE Market Schedules (IST)
PRE_MARKET_START = time(9, 0)
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# Official NSE Trading Holidays 2026 (YYYY-MM-DD)
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26): "Republic Day",
    date(2026, 3, 6): "Holi",
    date(2026, 3, 30): "Id-Ul-Fitr (Ramzan Id)",
    date(2026, 4, 3): "Good Friday",
    date(2026, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
    date(2026, 5, 1): "Maharashtra Day",
    date(2026, 5, 27): "Bakri Id",
    date(2026, 6, 26): "Muharram",
    date(2026, 8, 15): "Independence Day",
    date(2026, 9, 4): "Ganesh Chaturthi",
    date(2026, 10, 2): "Mahatma Gandhi Jayanti",
    date(2026, 10, 20): "Dussehra",
    date(2026, 11, 9): "Diwali-Laxmi Pujan (Samvat New Year)",
    date(2026, 11, 10): "Diwali Balipratipada",
    date(2026, 11, 24): "Guru Nanak Jayanti",
    date(2026, 12, 25): "Christmas",
}


def get_ist_now() -> datetime:
    """Returns current datetime in Indian Standard Time (Asia/Kolkata)."""
    return datetime.now(IST_TZ)


def is_trading_day(target_date: date = None) -> Tuple[bool, str]:
    """Checks if a given date is an official NSE trading day (Monday - Friday, not a holiday).

    Returns:
        (is_trading_day: bool, reason: str)
    """
    if target_date is None:
        target_date = get_ist_now().date()

    # Check weekend (5 = Saturday, 6 = Sunday)
    weekday = target_date.weekday()
    if weekday == 5:
        return False, "Saturday (Weekend - Market Closed)"
    if weekday == 6:
        return False, "Sunday (Weekend - Market Closed)"

    # Check NSE Holidays
    if target_date in NSE_HOLIDAYS_2026:
        holiday_name = NSE_HOLIDAYS_2026[target_date]
        return False, f"NSE Holiday: {holiday_name}"

    return True, "Trading Day"


def get_market_status() -> Dict[str, Any]:
    """Evaluates current IST market session status.

    Returns:
        Dict with status, is_open, is_pre_market, time_until_open, time_until_close, reason
    """
    now_ist = get_ist_now()
    today = now_ist.date()
    current_time = now_ist.time()

    is_trade_day, day_reason = is_trading_day(today)

    if not is_trade_day:
        return {
            "status": "MARKET_CLOSED",
            "is_trading_day": False,
            "is_open": False,
            "is_pre_market": False,
            "reason": day_reason,
            "ist_time": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        }

    # Market open evaluation on trading days
    if current_time < PRE_MARKET_START:
        return {
            "status": "BEFORE_PRE_MARKET",
            "is_trading_day": True,
            "is_open": False,
            "is_pre_market": False,
            "reason": f"Before 09:00 AM IST (Current IST: {now_ist.strftime('%H:%M:%S')})",
            "ist_time": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        }

    if PRE_MARKET_START <= current_time < MARKET_OPEN:
        return {
            "status": "PRE_MARKET",
            "is_trading_day": True,
            "is_open": False,
            "is_pre_market": True,
            "reason": "Pre-Market Research Window (09:00 - 09:15 AM IST)",
            "ist_time": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        }

    if MARKET_OPEN <= current_time <= MARKET_CLOSE:
        return {
            "status": "MARKET_OPEN",
            "is_trading_day": True,
            "is_open": True,
            "is_pre_market": False,
            "reason": "Regular Trading Session (09:15 AM - 03:30 PM IST)",
            "ist_time": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        }

    # After 03:30 PM IST
    return {
        "status": "MARKET_CLOSED",
        "is_trading_day": True,
        "is_open": False,
        "is_pre_market": False,
        "reason": f"After Market Close (03:30 PM IST) (Current IST: {now_ist.strftime('%H:%M:%S')})",
        "ist_time": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
    }
