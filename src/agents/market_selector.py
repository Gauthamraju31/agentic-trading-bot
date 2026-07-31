"""Market Selector Agent for scanning across multiple stocks/indices and choosing the best intraday opportunity."""

from typing import List, Tuple, Optional
from loguru import logger

from src.core.models import MarketContext, AgentOpinion
from src.core.enums import AgentRole, SignalAction
from .base import BaseAgent
from .prompts import TECHNICAL_ANALYST_PROMPT

MARKET_SELECTOR_PROMPT = """
You are an expert Market Selector and Opportunity Scanner for intraday day trading on Indian stock markets (NSE/BSE).
Your task is to analyze multiple candidate stock symbols, indices, and market segments.
Rank them based on:
1. Volatility & Liquidity (ATR, Volume)
2. Trend Clarity & Momentum (ADX, RSI, EMA alignment)
3. Risk-Reward Potential for Intraday Trading (MIS)

Identify the single BEST stock/market to trade right now and state your detailed rationale.
"""

class MarketSelectorAgent(BaseAgent):
    """Agent that scans multiple symbols/markets and picks the best intraday trading candidate."""

    def __init__(self):
        super().__init__(role=AgentRole.TECHNICAL_ANALYST, system_prompt=MARKET_SELECTOR_PROMPT)

    async def select_best_market(self, candidate_contexts: List[MarketContext]) -> Tuple[MarketContext, str]:
        """Scans candidate market contexts and selects the prime opportunity.

        Args:
            candidate_contexts: List of MarketContext objects for different symbols.

        Returns:
            Tuple of (Selected MarketContext, Rationale string).
        """
        if not candidate_contexts:
            raise ValueError("No candidate market contexts provided to MarketSelectorAgent.")

        if len(candidate_contexts) == 1:
            return candidate_contexts[0], "Single symbol provided."

        logger.info(f"[{self.role.name}] Scanning {len(candidate_contexts)} market candidates for intraday opportunities...")

        candidates_summary = []
        best_candidate = candidate_contexts[0]
        max_score = -1.0

        for ctx in candidate_contexts:
            indicators = ctx.indicators
            if not indicators:
                continue

            rsi = indicators.rsi_14 or 50.0
            adx = indicators.adx or 15.0
            atr = indicators.atr_14 or 5.0
            price = ctx.current_price or 1.0
            volatility_pct = (atr / price) * 100.0

            # Score for intraday opportunity (high trend strength ADX + high volatility ATR% + non-neutral RSI)
            rsi_extremity = abs(rsi - 50.0)
            score = (adx * 1.5) + (volatility_pct * 20.0) + rsi_extremity

            candidates_summary.append({
                "symbol": ctx.symbol,
                "price": price,
                "rsi": round(rsi, 2),
                "adx": round(adx, 2),
                "atr_pct": round(volatility_pct, 2),
                "score": round(score, 2)
            })

            if score > max_score:
                max_score = score
                best_candidate = ctx

        # Prompt LLM with candidates summary
        prompt_data = {
            "candidates": candidates_summary,
            "selected_symbol": best_candidate.symbol
        }

        reasoning = await self._generate_response(prompt_data)
        logger.info(f"[{self.role.name}] Selected prime candidate: {best_candidate.symbol} (Score: {max_score:.2f})")

        return best_candidate, f"Selected {best_candidate.symbol} (Intraday Score: {max_score:.2f}). Reason: {reasoning}"

    async def analyze(self, market_context: MarketContext, **kwargs) -> AgentOpinion:
        return self._create_opinion(reasoning=f"Market selection scanner active for {market_context.symbol}", confidence=0.7, action=SignalAction.HOLD)
