from loguru import logger
from src.core.models import MarketContext, AgentOpinion, TradingDecision, RiskVeto
from src.core.enums import AgentRole, SignalAction
from src.core.config import settings
from .base import BaseAgent
from .prompts import RISK_MANAGER_PROMPT

class RiskManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__(role=AgentRole.RISK_MANAGER, system_prompt=RISK_MANAGER_PROMPT)

    async def analyze(self, market_context: MarketContext, bull_op: AgentOpinion = None, bear_op: AgentOpinion = None, **kwargs) -> AgentOpinion:
        logger.info(f"[{self.role.name}] Assessing general risk parameters for {market_context.symbol}...")
        return self._create_opinion(reasoning="Risk parameters baseline active.", confidence=0.9, action=SignalAction.HOLD)

    async def evaluate_and_veto(self, decision: TradingDecision, market_context: MarketContext) -> tuple[RiskVeto, TradingDecision]:
        """Final Circuit Breaker & Hard Veto Gate over Portfolio Manager proposed decision."""
        logger.info(f"[{self.role.name}] Evaluating proposed {decision.action.value} trade for {market_context.symbol}...")

        if decision.action == SignalAction.HOLD:
            veto = RiskVeto(approve=False, allowed_position_size=0, veto_reason="Portfolio decision is HOLD.")
            decision.approved_by_risk = False
            decision.veto_reason = veto.veto_reason
            return veto, decision

        portfolio = market_context.portfolio
        curr_price = market_context.current_price
        qty = decision.position_size or 1
        est_cost = curr_price * qty

        # Check 1: Portfolio available capital must cover the WHOLE order (not 1 share).
        if portfolio and portfolio.current_capital < est_cost:
            veto = RiskVeto(approve=False, allowed_position_size=0,
                            veto_reason=f"Insufficient capital: need ₹{est_cost:,.2f} for {qty} shares, "
                                        f"have ₹{portfolio.current_capital:,.2f}.")
            decision.approved_by_risk = False
            decision.veto_reason = veto.veto_reason
            return veto, decision

        # Check 2: Maximum concurrent open positions (from config).
        if portfolio and len(portfolio.positions) >= settings.risk.max_open_positions:
            veto = RiskVeto(approve=False, allowed_position_size=0,
                            veto_reason=f"Maximum open position limit ({settings.risk.max_open_positions}) reached.")
            decision.approved_by_risk = False
            decision.veto_reason = veto.veto_reason
            return veto, decision

        # Check 3: Per-position exposure cap (from config).
        if portfolio and portfolio.current_capital > 0:
            exposure_pct = (est_cost / portfolio.current_capital) * 100.0
            if exposure_pct > settings.risk.max_position_pct:
                veto = RiskVeto(approve=False, allowed_position_size=0,
                                veto_reason=f"Position exposure {exposure_pct:.1f}% exceeds cap "
                                            f"{settings.risk.max_position_pct}%.")
                decision.approved_by_risk = False
                decision.veto_reason = veto.veto_reason
                return veto, decision

        # Check 4: Hard drawdown circuit breaker.
        if portfolio and portfolio.max_drawdown_pct > (settings.risk.max_daily_loss_pct * 3):
            veto = RiskVeto(approve=False, allowed_position_size=0,
                            veto_reason=f"Max drawdown threshold exceeded ({portfolio.max_drawdown_pct:.1f}%).")
            decision.approved_by_risk = False
            decision.veto_reason = veto.veto_reason
            return veto, decision

        # Approved!
        veto = RiskVeto(approve=True, allowed_position_size=qty, veto_reason="Trade parameters within risk tolerance.")
        decision.approved_by_risk = True
        decision.veto_reason = None
        
        # Log Risk Manager approval
        risk_op = self._create_opinion(
            reasoning=f"Approved proposed {decision.action.value} trade for {market_context.symbol}. Risk checks passed.",
            confidence=0.95,
            action=decision.action
        )
        decision.agent_opinions.append(risk_op)
        
        return veto, decision
