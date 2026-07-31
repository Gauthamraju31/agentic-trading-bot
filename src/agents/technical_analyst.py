from loguru import logger
from src.core.models import MarketContext, AgentOpinion, TechnicalOpinion
from src.core.enums import AgentRole, SignalAction
from .base import BaseAgent
from .prompts import TECHNICAL_ANALYST_PROMPT

class TechnicalAnalystAgent(BaseAgent):
    def __init__(self):
        super().__init__(role=AgentRole.TECHNICAL_ANALYST, system_prompt=TECHNICAL_ANALYST_PROMPT)

    async def analyze(self, market_context: MarketContext, **kwargs) -> AgentOpinion:
        logger.info(f"[{self.role.name}] Analyzing market context for {market_context.symbol}...")

        curr_price = market_context.current_price
        candles = market_context.candles or []
        high = max([c.high for c in candles[-20:]], default=curr_price * 1.01)
        low = min([c.low for c in candles[-20:]], default=curr_price * 0.99)
        pivot = (high + low + curr_price) / 3.0
        support1 = round((2 * pivot) - high, 2)
        resistance1 = round((2 * pivot) - low, 2)

        ind = market_context.indicators
        rsi = getattr(ind, 'rsi_14', None) if ind else None
        ema9 = getattr(ind, 'ema_9', None) if ind else None
        ema21 = getattr(ind, 'ema_21', None) if ind else None
        macd = getattr(ind, 'macd', None) if ind else None
        adx = getattr(ind, 'adx', None) if ind else None
        atr = getattr(ind, 'atr_14', None) if ind else None

        context_data = {
            "symbol": market_context.symbol,
            "current_price": curr_price,
            "support_1": support1,
            "resistance_1": resistance1,
            "rsi_14": rsi,
            "ema_9": ema9,
            "ema_21": ema21,
            "macd": macd,
            "adx": adx,
            "atr_14": atr,
            "constraint": f"Use ONLY these levels: price ₹{curr_price:.2f}, support ₹{support1:.2f}, resistance ₹{resistance1:.2f}.",
        }

        # LLM-driven structured opinion (drives the decision when available).
        op = await self._generate_structured(
            task="Judge the technical trend and momentum. Decide action (BUY/SELL/HOLD) and a confidence in [0,1]. "
                 "Set support_level and resistance_level from the provided values.",
            context=context_data,
            schema=TechnicalOpinion,
        )
        if op is not None:
            reasoning = op.reasoning or f"Technical trend {op.trend_strength}; RSI={rsi}, EMA9/21={ema9}/{ema21}."
            return self._create_opinion(reasoning=reasoning, confidence=op.confidence, action=op.action)

        # Deterministic fallback (no LLM): RSI + EMA crossover rules.
        action = SignalAction.HOLD
        confidence = 0.5
        if rsi is not None:
            if rsi < 35 or (ema9 and ema21 and ema9 > ema21):
                action = SignalAction.BUY
                confidence = 0.70
            elif rsi > 65 or (ema9 and ema21 and ema9 < ema21):
                action = SignalAction.SELL
                confidence = 0.70
        reasoning = f"[fallback] RSI={rsi}, EMA9={ema9}, EMA21={ema21} → {action.value}."
        return self._create_opinion(reasoning=reasoning, confidence=confidence, action=action)
