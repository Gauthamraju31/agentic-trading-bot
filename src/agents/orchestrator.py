import asyncio
import sqlite3
from typing import TypedDict, Optional, List, Tuple, Dict

from loguru import logger

from src.core.models import MarketContext, TradingDecision, AgentOpinion
from src.core.config import settings

from .technical_analyst import TechnicalAnalystAgent
from .sentiment_analyst import SentimentAnalystAgent
from .bull_agent import BullAgent
from .bear_agent import BearAgent
from .risk_manager import RiskManagerAgent
from .portfolio_manager import PortfolioManagerAgent
from .market_selector import MarketSelectorAgent
from .self_learning import SelfLearningEngine


class TradingState(TypedDict):
    market_context: MarketContext
    technical_opinion: Optional[AgentOpinion]
    sentiment_opinion: Optional[AgentOpinion]
    bull_opinion: Optional[AgentOpinion]
    bear_opinion: Optional[AgentOpinion]
    decision: Optional[TradingDecision]


try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not available. Orchestrator will use the sequential pipeline.")

try:
    from langgraph.checkpoint.memory import MemorySaver
    CHECKPOINT_AVAILABLE = True
except ImportError:
    MemorySaver = None
    CHECKPOINT_AVAILABLE = False


class AgentOrchestrator:
    """Multi-agent orchestrator: parallel analysis → bull/bear debate → LLM
    portfolio decision → deterministic risk veto. The graph and sequential
    paths delegate to the SAME helper methods so their logic can never diverge.
    """

    def __init__(self, enable_checkpoint: bool = True):
        self.market_selector = MarketSelectorAgent()
        self.tech_agent = TechnicalAnalystAgent()
        self.sentiment_agent = SentimentAnalystAgent()
        self.bull_agent = BullAgent()
        self.bear_agent = BearAgent()
        self.risk_agent = RiskManagerAgent()
        self.portfolio_agent = PortfolioManagerAgent()
        self.learning = SelfLearningEngine()

        # India VIX fetcher for regime-aware decisions
        self._vix_fetcher = None
        self._vix_data = None

        # LangGraph checkpoint for crash-safe state management
        self._checkpointer = None
        if enable_checkpoint and CHECKPOINT_AVAILABLE:
            try:
                self._checkpointer = MemorySaver()
                logger.info("[Orchestrator] LangGraph Memory checkpointer enabled for state persistence.")
            except Exception as e:
                logger.warning(f"[Orchestrator] Failed to init Memory checkpointer: {e}")

        self.graph = self._build_graph() if LANGGRAPH_AVAILABLE else None

    # ── Shared pipeline steps ────────────────────────────────────────────────

    def _agent_weights(self) -> Dict[str, float]:
        """Reliability weights per agent role, calibrated by the learning engine."""
        return {role: stats.get("weight", 1.0) for role, stats in self.learning.agent_accuracy.items()}

    async def _analyze(self, ctx: MarketContext) -> Tuple[AgentOpinion, AgentOpinion]:
        """Technical + sentiment analysis, run concurrently."""
        tech_op, sent_op = await asyncio.gather(
            self.tech_agent.analyze(ctx),
            self.sentiment_agent.analyze(ctx),
        )
        return tech_op, sent_op

    async def _debate(self, ctx: MarketContext, tech_op: AgentOpinion,
                      sent_op: AgentOpinion) -> Tuple[AgentOpinion, AgentOpinion]:
        """Bull vs bear debate over `debate_rounds` rounds; each round each side
        sees the other's latest thesis and rebuts it."""
        rounds = max(1, int(settings.agents.debate_rounds))
        bull_op = await self.bull_agent.analyze(ctx, tech_op=tech_op, sent_op=sent_op)
        bear_op = await self.bear_agent.analyze(ctx, tech_op=tech_op, sent_op=sent_op)
        for r in range(rounds - 1):
            logger.debug(f"[Orchestrator] Debate round {r + 2}/{rounds} for {ctx.symbol}")
            bull_next, bear_next = await asyncio.gather(
                self.bull_agent.analyze(ctx, tech_op=tech_op, sent_op=sent_op, opponent_thesis=bear_op.reasoning),
                self.bear_agent.analyze(ctx, tech_op=tech_op, sent_op=sent_op, opponent_thesis=bull_op.reasoning),
            )
            bull_op, bear_op = bull_next, bear_next
        return bull_op, bear_op

    async def _finalize(self, ctx: MarketContext, tech_op, sent_op, bull_op, bear_op) -> TradingDecision:
        """Portfolio decision → risk veto → attach opinions → record for learning."""
        decision = await self.portfolio_agent.make_decision(
            ctx, bull_op, bear_op, tech_op, sent_op, agent_weights=self._agent_weights()
        )
        _, decision = await self.risk_agent.evaluate_and_veto(decision, ctx)

        decision.agent_opinions = [op for op in [tech_op, sent_op, bull_op, bear_op] if op is not None]
        # Record actionable, risk-approved decisions so outcomes can recalibrate weights later.
        if decision.approved_by_risk and decision.action.value in ("BUY", "SELL"):
            try:
                self.learning.record_decision(ctx.symbol, ctx.current_price, decision)
            except Exception as e:
                logger.warning(f"[Orchestrator] Could not record decision for learning: {e}")
        return decision

    # ── LangGraph wiring (delegates to the shared steps above) ────────────────

    def _build_graph(self):
        workflow = StateGraph(TradingState)

        async def analysis_node(state: TradingState):
            tech_op, sent_op = await self._analyze(state["market_context"])
            return {"technical_opinion": tech_op, "sentiment_opinion": sent_op}

        async def debate_node(state: TradingState):
            bull_op, bear_op = await self._debate(
                state["market_context"], state["technical_opinion"], state["sentiment_opinion"]
            )
            return {"bull_opinion": bull_op, "bear_opinion": bear_op}

        async def decision_node(state: TradingState):
            decision = await self._finalize(
                state["market_context"], state["technical_opinion"], state["sentiment_opinion"],
                state["bull_opinion"], state["bear_opinion"],
            )
            return {"decision": decision}

        workflow.add_node("analysis", analysis_node)
        workflow.add_node("debate", debate_node)
        workflow.add_node("decision", decision_node)

        workflow.set_entry_point("analysis")
        workflow.add_edge("analysis", "debate")
        workflow.add_edge("debate", "decision")
        workflow.add_edge("decision", END)

        if self._checkpointer:
            return workflow.compile(checkpointer=self._checkpointer)
        return workflow.compile()

    # ── Public API ────────────────────────────────────────────────────────────

    async def select_and_run(self, candidate_contexts: List[MarketContext]) -> Tuple[MarketContext, TradingDecision, str]:
        """Scan candidate markets, pick the best, and run the decision pipeline."""
        selected_context, rationale = await self.market_selector.select_best_market(candidate_contexts)
        decision = await self.run(selected_context)
        return selected_context, decision, rationale

    async def _fetch_vix(self) -> Optional[Dict]:
        """Fetch India VIX data for regime-aware position sizing."""
        if self._vix_fetcher is None:
            from src.data.india_vix import IndiaVIXFetcher
            self._vix_fetcher = IndiaVIXFetcher()
        try:
            return await self._vix_fetcher.fetch_current_vix()
        except Exception as e:
            logger.warning(f"[Orchestrator] VIX fetch failed: {e}")
            return None

    async def run(self, market_context: MarketContext) -> TradingDecision:
        logger.info(f"Starting agent orchestrator for {market_context.symbol}")

        # Fetch India VIX and attach to context
        vix_data = await self._fetch_vix()
        if vix_data:
            self._vix_data = vix_data
            market_context.vix_data = vix_data
            if vix_data.get("should_halt"):
                logger.warning(f"[Orchestrator] India VIX at {vix_data.get('vix_value')} — EXTREME volatility, recommending HOLD.")

        if self.graph:
            initial_state = TradingState(
                market_context=market_context,
                technical_opinion=None,
                sentiment_opinion=None,
                bull_opinion=None,
                bear_opinion=None,
                decision=None,
            )
            config = {}
            if self._checkpointer:
                config = {"configurable": {"thread_id": f"{market_context.symbol}_{market_context.timestamp.strftime('%Y%m%d_%H%M') if market_context.timestamp else 'now'}"}}
            result = await self.graph.ainvoke(initial_state, config=config)
            return result["decision"]

        # Sequential fallback (identical logic, no langgraph).
        tech_op, sent_op = await self._analyze(market_context)
        bull_op, bear_op = await self._debate(market_context, tech_op, sent_op)
        return await self._finalize(market_context, tech_op, sent_op, bull_op, bear_op)
