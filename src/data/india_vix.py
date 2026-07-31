"""India VIX (Volatility Index) fetcher for regime-aware risk management.

Fetches India VIX from Yahoo Finance (^INDIAVIX) and provides:
- Current VIX value
- Regime classification (LOW / MODERATE / HIGH / EXTREME)
- Position sizing recommendations based on VIX level
"""

from typing import Any, Dict, Optional

import yfinance as yf
from loguru import logger

from src.core.config import settings


class IndiaVIXFetcher:
    """Fetches and interprets India VIX for risk-aware trading decisions."""

    def __init__(self):
        vix_cfg = getattr(settings, "vix", None)
        self.enabled = getattr(vix_cfg, "enabled", True) if vix_cfg else True
        self.ticker = getattr(vix_cfg, "ticker", "^INDIAVIX") if vix_cfg else "^INDIAVIX"
        self.high_threshold = getattr(vix_cfg, "high_vix_threshold", 18.0) if vix_cfg else 18.0
        self.halt_threshold = getattr(vix_cfg, "halt_vix_threshold", 25.0) if vix_cfg else 25.0

    async def fetch_current_vix(self) -> Dict[str, Any]:
        """Fetch current India VIX value and classify the regime.

        Returns:
            dict with keys: vix_value, regime, position_size_multiplier, should_halt, summary
        """
        if not self.enabled:
            return self._default_response("VIX monitoring disabled.")

        try:
            ticker = yf.Ticker(self.ticker)
            hist = ticker.history(period="5d", interval="1d")

            if hist.empty:
                logger.warning("[IndiaVIX] No VIX data returned from Yahoo Finance.")
                return self._default_response("VIX data unavailable.")

            current_vix = float(hist["Close"].iloc[-1])
            prev_vix = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current_vix
            vix_change = current_vix - prev_vix

            regime = self._classify_regime(current_vix)
            multiplier = self._position_size_multiplier(current_vix)
            should_halt = current_vix >= self.halt_threshold

            summary = (
                f"India VIX: {current_vix:.2f} ({vix_change:+.2f} from prev) | "
                f"Regime: {regime} | Position multiplier: {multiplier:.2f}x"
            )
            if should_halt:
                summary += " | ⚠️ TRADING HALT RECOMMENDED (extreme volatility)"

            logger.info(f"[IndiaVIX] {summary}")

            return {
                "vix_value": round(current_vix, 2),
                "vix_change": round(vix_change, 2),
                "regime": regime,
                "position_size_multiplier": multiplier,
                "should_halt": should_halt,
                "summary": summary,
            }

        except Exception as e:
            logger.warning(f"[IndiaVIX] Failed to fetch VIX: {e}")
            return self._default_response(f"VIX fetch error: {e}")

    def _classify_regime(self, vix: float) -> str:
        """Classify market volatility regime based on India VIX level."""
        if vix < 12:
            return "LOW_VOLATILITY"
        elif vix < self.high_threshold:
            return "MODERATE"
        elif vix < self.halt_threshold:
            return "HIGH_VOLATILITY"
        else:
            return "EXTREME"

    def _position_size_multiplier(self, vix: float) -> float:
        """Scale position sizes inversely with volatility.

        - VIX < 12: full size (1.0x)
        - VIX 12-18: slightly reduced (0.8x)
        - VIX 18-25: significantly reduced (0.5x)
        - VIX > 25: minimal positions (0.25x)
        """
        if vix < 12:
            return 1.0
        elif vix < self.high_threshold:
            return 0.8
        elif vix < self.halt_threshold:
            return 0.5
        else:
            return 0.25

    @staticmethod
    def _default_response(message: str) -> Dict[str, Any]:
        return {
            "vix_value": None,
            "vix_change": None,
            "regime": "UNKNOWN",
            "position_size_multiplier": 1.0,
            "should_halt": False,
            "summary": message,
        }
