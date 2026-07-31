from loguru import logger
from src.core.models import MarketContext, AgentOpinion, BearOpinion
from src.core.enums import AgentRole, SignalAction
from .base import BaseAgent
from .prompts import BEAR_AGENT_PROMPT

class BearAgent(BaseAgent):
    def __init__(self):
        super().__init__(role=AgentRole.BEAR, system_prompt=BEAR_AGENT_PROMPT)

    async def analyze(self, market_context: MarketContext, tech_op: AgentOpinion = None,
                      sent_op: AgentOpinion = None, opponent_thesis: str = "", **kwargs) -> AgentOpinion:
        logger.info(f"[{self.role.name}] Building bearish thesis for {market_context.symbol}...")

        curr_price = market_context.current_price
        context_data = {
            "symbol": market_context.symbol,
            "current_price": curr_price,
            "technical_analysis": tech_op.reasoning if tech_op else "N/A",
            "technical_action": tech_op.action.value if tech_op else "HOLD",
            "sentiment_analysis": sent_op.reasoning if sent_op else "N/A",
            "sentiment_action": sent_op.action.value if sent_op else "HOLD",
            "bull_counter_argument": opponent_thesis or "(none yet)",
            "constraint": f"Current price ₹{curr_price:.2f}. Do NOT invent unprovided fundamentals; keep any "
                          f"invalidation_price within 2-5% of ₹{curr_price:.2f}.",
        }

        # LLM-driven structured bear thesis; bear_score in [0,1] drives weighting.
        op = await self._generate_structured(
            task="Argue the strongest realistic BEARISH case, rebutting the bull's counter-argument. "
                 "Set bear_score in [0,1] (conviction the price falls) and an optional invalidation_price.",
            context=context_data,
            schema=BearOpinion,
        )
        if op is not None:
            return self._create_opinion(reasoning=op.thesis_summary, confidence=op.bear_score, action=SignalAction.SELL)

        # Deterministic fallback: conviction from technical + sentiment confirmation.
        conf = 0.65
        if tech_op and tech_op.action == SignalAction.SELL:
            conf += 0.15
        if sent_op and sent_op.action == SignalAction.SELL:
            conf += 0.10
        reasoning = f"[fallback] Bear thesis: tech={context_data['technical_action']}, sent={context_data['sentiment_action']}."
        return self._create_opinion(reasoning=reasoning, confidence=min(0.90, conf), action=SignalAction.SELL)
