"""Pre-Market Autonomous Research & Planning Agent.

Scans candidate stock universe, fetches overnight news, India VIX, macro sentiment,
and technical chart structures BEFORE market opens (09:00 - 09:15 AM IST).
Generates an autonomous Daily Playbook with selected stocks, directional bias,
entry triggers, stop-loss, and take-profit targets.
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
import pandas as pd

from src.agents.orchestrator import AgentOrchestrator
from src.core.config import settings
from src.core.enums import Exchange, SignalAction, TimeFrame
from src.core.models import Candle, MarketContext
from src.data.feeds.live_feed import LiveDataFeed
from src.data.india_vix import IndiaVIXFetcher
from src.data.indicators import IndicatorEngine
from src.data.sentiment import SentimentFetcher


class PreMarketPlanner:
    """Pre-market research engine that plans the day's trades before market open."""

    def __init__(self, playbook_file: Optional[Path] = None):
        self.playbook_file = playbook_file or Path("data/daily_playbook.json")
        self.playbook_file.parent.mkdir(parents=True, exist_ok=True)
        self.feed = LiveDataFeed()
        self.vix_fetcher = IndiaVIXFetcher()
        self.sentiment_fetcher = SentimentFetcher()
        self.orchestrator = AgentOrchestrator()

        self.candidate_symbols: List[str] = [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
            "SBIN", "BAJFINANCE", "ITC", "LT", "BHARTIARTL"
        ]

    async def run_pre_market_analysis(self) -> Dict[str, Any]:
        """Execute autonomous pre-market research and generate daily playbook."""
        logger.info("[PreMarketPlanner] Starting autonomous pre-market research & planning...")
        date_str = datetime.now().strftime("%Y-%m-%d")

        # 1. Fetch India VIX Market Regime
        vix_data = await self.vix_fetcher.fetch_current_vix()
        regime = vix_data.get("regime", "MODERATE")
        vix_val = vix_data.get("vix_value", 15.0)
        should_halt = vix_data.get("should_halt", False)

        logger.info(f"[PreMarketPlanner] India VIX: {vix_val:.2f} | Regime: {regime}")

        if should_halt:
            playbook = {
                "date": date_str,
                "timestamp": datetime.now().isoformat(),
                "vix_data": vix_data,
                "status": "HALTED",
                "reason": f"India VIX at {vix_val:.2f} is extreme. Pre-market planner recommends NO trading today.",
                "target_symbols": [],
                "primary_pick": None,
            }
            self._save_playbook(playbook)
            return playbook

        # 2. Fetch market candles & compute indicators for candidate universe
        candidate_contexts: List[MarketContext] = []

        for symbol in self.candidate_symbols:
            try:
                candles = await self.feed.get_historical_candles(symbol, exchange=Exchange.NSE, timeframe=TimeFrame.M5)
                if len(candles) >= 30:
                    df = pd.DataFrame([c.model_dump() for c in candles])
                    df_ind = IndicatorEngine.calculate(df)
                    latest_ind = IndicatorEngine.get_latest_indicators(df_ind)

                    ctx = MarketContext(
                        symbol=symbol,
                        exchange=Exchange.NSE,
                        current_price=candles[-1].close,
                        candles=candles,
                        indicators=latest_ind,
                        timestamp=candles[-1].timestamp,
                        vix_data=vix_data,
                    )
                    candidate_contexts.append(ctx)
            except Exception as e:
                logger.warning(f"[PreMarketPlanner] Could not load candles for {symbol}: {e}")

        if not candidate_contexts:
            logger.error("[PreMarketPlanner] No market contexts loaded. Aborting pre-market plan.")
            return {"date": date_str, "status": "ERROR", "reason": "No market data available"}

        # 3. Market Selector Agent ranks candidates and picks prime target
        selected_ctx, decision, rationale = await self.orchestrator.select_and_run(candidate_contexts)

        # 4. Fetch sentiment summary for the chosen target
        sentiment_info = await self.sentiment_fetcher.compute_headline_sentiment(selected_ctx.symbol)

        # 5. Construct Daily Playbook
        playbook = {
            "date": date_str,
            "timestamp": datetime.now().isoformat(),
            "status": "READY",
            "vix_data": vix_data,
            "sentiment_summary": sentiment_info.get("fii_dii_summary", "Neutral"),
            "primary_pick": {
                "symbol": selected_ctx.symbol,
                "current_price": selected_ctx.current_price,
                "action": decision.action.value if hasattr(decision.action, "value") else str(decision.action),
                "confidence": round(decision.confidence, 2),
                "entry_price": decision.entry_price or selected_ctx.current_price,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
                "position_size": decision.position_size,
                "reasoning": decision.reasoning,
                "selector_rationale": rationale,
            },
            "candidate_count": len(candidate_contexts),
            "agents_consulted": ["TECHNICAL_ANALYST", "SENTIMENT_ANALYST", "BULL", "BEAR", "RISK_MANAGER", "PORTFOLIO_MANAGER"],
        }

        self._save_playbook(playbook)
        logger.success(
            f"[PreMarketPlanner] Pre-market research complete! Selected target: {selected_ctx.symbol} "
            f"[{decision.action.value if hasattr(decision.action, 'value') else decision.action}] @ ₹{selected_ctx.current_price:.2f}"
        )
        return playbook

    def _save_playbook(self, playbook: Dict[str, Any]):
        """Persist playbook to JSON."""
        try:
            with open(self.playbook_file, "w") as f:
                json.dump(playbook, f, indent=2)
        except Exception as e:
            logger.error(f"[PreMarketPlanner] Failed to save playbook: {e}")

    def get_latest_playbook(self) -> Optional[Dict[str, Any]]:
        """Read existing playbook from disk."""
        if self.playbook_file.exists():
            try:
                with open(self.playbook_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[PreMarketPlanner] Failed to read playbook: {e}")
        return None
