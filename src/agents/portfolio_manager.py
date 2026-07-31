from typing import Optional, Dict
from loguru import logger
from src.core.models import MarketContext, AgentOpinion, TradingDecision, PortfolioProposal
from src.core.enums import AgentRole, SignalAction
from src.core.config import settings
from src.risk.position_sizer import PositionSizer
from src.agents.trade_memory import TradeReflectionMemory
from .base import BaseAgent
from .prompts import PORTFOLIO_MANAGER_PROMPT


class PortfolioManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__(role=AgentRole.PORTFOLIO_MANAGER, system_prompt=PORTFOLIO_MANAGER_PROMPT)
        from pathlib import Path
        learning_cfg = getattr(settings, "learning", None)
        mem_file = Path(getattr(learning_cfg, "reflection_memory_file", "data/trade_reflection_memory.md")) if learning_cfg else Path("data/trade_reflection_memory.md")
        max_ref = int(getattr(learning_cfg, "max_reflections_in_prompt", 5)) if learning_cfg else 5
        self._memory = TradeReflectionMemory(memory_file=mem_file, max_in_prompt=max_ref)

    async def analyze(self, market_context: MarketContext, **kwargs) -> AgentOpinion:
        raise NotImplementedError("PortfolioManagerAgent uses make_decision(), not analyze().")

    async def make_decision(
        self,
        market_context: MarketContext,
        bull_opinion: AgentOpinion,
        bear_opinion: AgentOpinion,
        tech_opinion: Optional[AgentOpinion] = None,
        sent_opinion: Optional[AgentOpinion] = None,
        agent_weights: Optional[Dict[str, float]] = None,
    ) -> TradingDecision:
        """Synthesize the analysts' opinions into a final action.

        When the LLM backend is available it *drives* the decision (returns a
        structured PortfolioProposal). Otherwise a deterministic, self-learning-
        weighted score decides. Either way the decision passes a min-confidence
        gate and a real position-sizing step; the Risk Manager veto (called
        separately) has the final say.
        """
        logger.info(f"[{self.role.name}] Making final portfolio decision for {market_context.symbol}...")
        curr_price = market_context.current_price
        weights = agent_weights or {}

        context_data = {
            "symbol": market_context.symbol,
            "current_price": curr_price,
            "technical": {"action": tech_opinion.action.value if tech_opinion else "HOLD",
                          "confidence": tech_opinion.confidence if tech_opinion else 0.0,
                          "notes": tech_opinion.reasoning if tech_opinion else ""},
            "sentiment": {"action": sent_opinion.action.value if sent_opinion else "HOLD",
                          "confidence": sent_opinion.confidence if sent_opinion else 0.0,
                          "notes": sent_opinion.reasoning if sent_opinion else ""},
            "bull": {"score": bull_opinion.confidence if bull_opinion else 0.0,
                     "thesis": bull_opinion.reasoning if bull_opinion else ""},
            "bear": {"score": bear_opinion.confidence if bear_opinion else 0.0,
                     "thesis": bear_opinion.reasoning if bear_opinion else ""},
            "agent_reliability_weights": weights,
            "constraint": f"Entry near ₹{curr_price:.2f}. stop_loss/take_profit within a few percent of it. "
                          f"position_size_pct in [0,100] but keep <= {settings.risk.max_position_pct}.",
        }

        # Inject past trade reflections for this symbol
        same_ticker_reflections = self._memory.get_recent_reflections(symbol=market_context.symbol)
        cross_ticker_lessons = self._memory.get_cross_ticker_lessons(exclude_symbol=market_context.symbol)
        if same_ticker_reflections:
            context_data["past_reflections_same_ticker"] = same_ticker_reflections
        if cross_ticker_lessons:
            context_data["cross_ticker_lessons"] = cross_ticker_lessons

        # Inject VIX context if available in market_context
        if hasattr(market_context, 'vix_data') and market_context.vix_data:
            context_data["india_vix"] = market_context.vix_data

        # ── LLM-driven path ───────────────────────────────────────────────────
        proposal = await self._generate_structured(
            task="Weigh the bull vs bear theses and the technical/sentiment reads (scaled by the reliability "
                 "weights) and decide the final action. Provide confidence in [0,1], stop_loss, take_profit, "
                 "and position_size_pct.",
            context=context_data,
            schema=PortfolioProposal,
        )

        if proposal is not None:
            action = proposal.proposed_action
            confidence = proposal.confidence
            reasoning = f"[LLM] {proposal.rationale}".strip()
            sl_p = proposal.stop_loss
            tp_p = proposal.take_profit
            size_pct = proposal.position_size_pct or settings.risk.risk_per_trade_pct
        else:
            action, confidence, reasoning, size_pct = self._deterministic_decision(
                bull_opinion, bear_opinion, tech_opinion, sent_opinion, weights
            )
            sl_p = tp_p = None

        # ── Min-confidence gate ───────────────────────────────────────────────
        if action in (SignalAction.BUY, SignalAction.SELL) and confidence < settings.agents.min_confidence:
            reasoning += (f" | Confidence {confidence:.2f} below min_confidence "
                          f"{settings.agents.min_confidence:.2f} → downgraded to HOLD.")
            action = SignalAction.HOLD

        # ── Stop / target defaults (if the LLM didn't set sane ones) ──────────
        stop_pct = settings.risk.default_stop_loss_pct / 100.0
        if action == SignalAction.BUY:
            sl_p = round(sl_p if sl_p and sl_p < curr_price else curr_price * (1 - stop_pct), 2)
            tp_p = round(tp_p if tp_p and tp_p > curr_price else curr_price * (1 + 2 * stop_pct), 2)
        elif action == SignalAction.SELL:
            sl_p = round(sl_p if sl_p and sl_p > curr_price else curr_price * (1 + stop_pct), 2)
            tp_p = round(tp_p if tp_p and tp_p < curr_price else curr_price * (1 - 2 * stop_pct), 2)
        else:
            sl_p = tp_p = None

        # ── Real position sizing (not a hardcoded constant) ───────────────────
        position_size = 0
        if action in (SignalAction.BUY, SignalAction.SELL):
            portfolio = market_context.portfolio
            capital = portfolio.current_capital if portfolio else settings.mock.initial_capital
            risk_pct = min(size_pct, settings.risk.max_position_pct) if size_pct else settings.risk.risk_per_trade_pct
            position_size = PositionSizer.calculate_size(
                method=settings.risk.position_sizing,
                capital=capital,
                risk_per_trade_pct=settings.risk.risk_per_trade_pct,
                entry_price=curr_price,
                stop_loss_price=sl_p or curr_price * (1 - stop_pct),
            )
            position_size = max(1, position_size)

        full_reasoning = f"Final Action: {action.value} | Confidence: {confidence:.2f} | {reasoning}"

        return TradingDecision(
            symbol=market_context.symbol,
            action=action,
            confidence=confidence,
            entry_price=curr_price,
            stop_loss=sl_p,
            take_profit=tp_p,
            reasoning=full_reasoning,
            position_size=position_size,
            approved_by_risk=False,  # set by the Risk Manager veto gate
            agent_opinions=[op for op in [tech_opinion, sent_opinion, bull_opinion, bear_opinion] if op is not None],
            timestamp=market_context.timestamp,
        )

    def _deterministic_decision(self, bull_opinion, bear_opinion, tech_opinion, sent_opinion, weights):
        """Self-learning-weighted score used when no LLM backend is available."""
        def w(role_value: str, default: float = 1.0) -> float:
            return weights.get(role_value, default)

        bull_score = (bull_opinion.confidence if bull_opinion else 0.5) * w("bull")
        bear_score = (bear_opinion.confidence if bear_opinion else 0.5) * w("bear")
        tech_score = 0.0
        if tech_opinion and tech_opinion.action in (SignalAction.BUY, SignalAction.SELL):
            signed = tech_opinion.confidence if tech_opinion.action == SignalAction.BUY else -tech_opinion.confidence
            tech_score = signed * w("technical_analyst")
        sent_score = 0.0
        if sent_opinion and sent_opinion.action in (SignalAction.BUY, SignalAction.SELL):
            signed = sent_opinion.confidence if sent_opinion.action == SignalAction.BUY else -sent_opinion.confidence
            sent_score = signed * w("sentiment_analyst")

        net_score = (0.35 * (bull_score - bear_score)) + (0.40 * tech_score) + (0.25 * sent_score)

        if net_score > 0.25 and bull_score > bear_score:
            action = SignalAction.BUY
            confidence = min(0.90, 0.50 + net_score)
        elif net_score < -0.25 and bear_score > bull_score:
            action = SignalAction.SELL
            confidence = min(0.90, 0.50 + abs(net_score))
        else:
            action = SignalAction.HOLD
            confidence = 0.50

        reasoning = f"[deterministic] net_score={net_score:.2f} (weighted bull/bear/tech/sent)."
        return action, confidence, reasoning, settings.risk.risk_per_trade_pct
