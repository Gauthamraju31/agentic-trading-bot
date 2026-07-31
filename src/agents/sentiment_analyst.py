from loguru import logger
from src.core.models import MarketContext, AgentOpinion, SentimentOpinion
from src.core.enums import AgentRole, SignalAction
from src.data.sentiment import SentimentFetcher
from .base import BaseAgent
from .prompts import SENTIMENT_ANALYST_PROMPT

class SentimentAnalystAgent(BaseAgent):
    """Sentiment Analyst Agent evaluating live news feeds, headlines, and market tone."""

    def __init__(self):
        super().__init__(role=AgentRole.SENTIMENT_ANALYST, system_prompt=SENTIMENT_ANALYST_PROMPT)
        self.fetcher = SentimentFetcher()

    async def analyze(self, market_context: MarketContext, **kwargs) -> AgentOpinion:
        logger.info(f"[{self.role.name}] Analyzing live news & sentiment for {market_context.symbol}...")
        
        sentiment_data = await self.fetcher.compute_headline_sentiment(market_context.symbol)
        
        context_data = {
            "symbol": market_context.symbol,
            "company_sentiment": sentiment_data["company_sentiment"],
            "sector_sentiment": sentiment_data["sector_sentiment"],
            "macro_sentiment": sentiment_data["macro_sentiment"],
            "overall_score": sentiment_data["overall_score"],
            "tone": sentiment_data["tone"],
            "fii_dii_summary": sentiment_data.get("fii_dii_summary", "N/A"),
            "company_headlines": sentiment_data["company_headlines"],
            "sector_headlines": sentiment_data["sector_headlines"],
            "macro_headlines": sentiment_data["macro_headlines"],
            "constraint": "Judge sentiment from company, sector, and FII/DII institutional news; do not confuse broad macro headlines with stock-specific earnings."
        }

        op = await self._generate_structured(
            task="Assess sentiment for the stock. Set overall_action (BUY/SELL/HOLD) and overall_confidence in [0,1], "
                 "plus company/sector/macro_sentiment in [-1,1] grounded in the given headlines.",
            context=context_data,
            schema=SentimentOpinion,
        )
        if op is not None:
            reasoning = op.reasoning or (
                f"Sentiment company={op.company_sentiment:.2f}, sector={op.sector_sentiment:.2f}, "
                f"macro={op.macro_sentiment:.2f}."
            )
            return self._create_opinion(reasoning=reasoning, confidence=op.overall_confidence, action=op.overall_action)

        score = sentiment_data["overall_score"]
        if score > 0.2:
            action = SignalAction.BUY
            confidence = min(0.9, 0.6 + abs(score) * 0.3)
            reasoning = f"Bullish sentiment (Company: {sentiment_data['company_sentiment']:.2f}, Sector: {sentiment_data['sector_sentiment']:.2f}, Overall: {score:.2f})."
        elif score < -0.2:
            action = SignalAction.SELL
            confidence = min(0.9, 0.6 + abs(score) * 0.3)
            reasoning = f"Bearish sentiment (Company: {sentiment_data['company_sentiment']:.2f}, Sector: {sentiment_data['sector_sentiment']:.2f}, Overall: {score:.2f})."
        else:
            action = SignalAction.HOLD
            confidence = 0.5
            reasoning = f"Neutral market sentiment (Overall: {score:.2f})."
            
        return self._create_opinion(reasoning=reasoning, confidence=confidence, action=action)
