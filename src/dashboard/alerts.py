import httpx
from loguru import logger
from src.core.models import Trade
from src.core.config import settings

class AlertManager:
    """Manager for sending Telegram alerts."""
    def __init__(self):
        self.bot_token = settings.alerts.telegram_bot_token if hasattr(settings, "alerts") and hasattr(settings.alerts, "telegram_bot_token") else None
        self.chat_id = settings.alerts.telegram_chat_id if hasattr(settings, "alerts") and hasattr(settings.alerts, "telegram_chat_id") else None
        self.enabled = settings.alerts.telegram_enabled if hasattr(settings, "alerts") and hasattr(settings.alerts, "telegram_enabled") else False
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage" if self.bot_token else ""

    async def send_alert(self, message: str, level: str = "INFO"):
        """Send a general alert."""
        if not self.enabled or not self.bot_token or not self.chat_id:
            logger.info(f"Alert not sent (disabled or not configured) [{level}]: {message}")
            return
            
        formatted_message = f"[{level}] {message}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    json={
                        "chat_id": self.chat_id,
                        "text": formatted_message,
                        "parse_mode": "HTML"
                    }
                )
                if response.status_code != 200:
                    logger.error(f"Failed to send Telegram alert: {response.text}")
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")

    async def send_trade_alert(self, trade: Trade):
        """Send an alert for a completed trade."""
        message = (
            f"🔔 <b>Trade Completed</b>\n"
            f"Symbol: {trade.symbol}\n"
            f"Side: {trade.side.value if hasattr(trade.side, 'value') else trade.side}\n"
            f"Quantity: {trade.quantity}\n"
            f"Entry Price: ₹{trade.entry_price:.2f}\n"
            f"Exit Price: ₹{trade.exit_price:.2f}\n"
            f"P&L: ₹{trade.net_pnl:.2f}\n"
        )
        await self.send_alert(message, level="TRADE")

    async def send_circuit_breaker_alert(self, reason: str):
        """Send an alert when the circuit breaker is triggered."""
        message = (
            f"🚨 <b>CIRCUIT BREAKER TRIGGERED</b> 🚨\n"
            f"Reason: {reason}\n"
            f"Trading has been halted."
        )
        await self.send_alert(message, level="CRITICAL")
