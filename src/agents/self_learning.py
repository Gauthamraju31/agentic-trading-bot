"""Self-Improvement & Reinforcement Feedback Loop Engine for AI Trading Agents."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from src.core.enums import AgentRole, SignalAction
from src.core.models import TradingDecision

LEARNING_STATS_FILE = Path("data/learning_stats.json")

class SelfLearningEngine:
    """Tracks past agent decisions, evaluates price outcomes against realistic transaction cost buffers,
    calculates Alpha vs NIFTY 50 benchmark, records post-trade reflections,
    and dynamically calibrates agent confidence weights once statistically significant sample sizes are met.
    """

    MIN_EVAL_SAMPLES = 5  # minimum evaluations required before calibrating weights
    COST_BUFFER_PCT = 0.30  # minimum % move required to account for round-trip fees (STT, brokerage, GST)

    def __init__(self, stats_file: Path = LEARNING_STATS_FILE):
        self.stats_file = stats_file
        self.decision_history: List[Dict[str, Any]] = []
        self.agent_accuracy: Dict[str, Dict[str, float]] = {
            "technical_analyst": {"total": 0, "correct": 0, "accuracy": 0.5, "weight": 1.0},
            "sentiment_analyst": {"total": 0, "correct": 0, "accuracy": 0.5, "weight": 1.0},
            "bull": {"total": 0, "correct": 0, "accuracy": 0.5, "weight": 1.0},
            "bear": {"total": 0, "correct": 0, "accuracy": 0.5, "weight": 1.0},
            "risk_manager": {"total": 0, "correct": 0, "accuracy": 0.5, "weight": 1.0},
            "portfolio_manager": {"total": 0, "correct": 0, "accuracy": 0.5, "weight": 1.0},
        }

        # Lazy-init reflection memory to avoid circular imports
        self._reflection_memory = None
        self.load_stats()

    @property
    def reflection_memory(self):
        if self._reflection_memory is None:
            from src.agents.trade_memory import TradeReflectionMemory
            from src.core.config import settings
            learning_cfg = getattr(settings, "learning", None)
            mem_file = Path(getattr(learning_cfg, "reflection_memory_file", "data/trade_reflection_memory.md")) if learning_cfg else Path("data/trade_reflection_memory.md")
            max_ref = int(getattr(learning_cfg, "max_reflections_in_prompt", 5)) if learning_cfg else 5
            self._reflection_memory = TradeReflectionMemory(memory_file=mem_file, max_in_prompt=max_ref)
        return self._reflection_memory

    def load_stats(self) -> None:
        """Load historical learning stats from disk."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, "r") as f:
                    data = json.load(f)
                    self.agent_accuracy.update(data.get("agent_accuracy", {}))
                    self.decision_history = data.get("decision_history", [])[-100:]
                logger.info("[SelfLearningEngine] Loaded agent accuracy calibration stats.")
            except Exception as e:
                logger.warning(f"[SelfLearningEngine] Failed to load stats file: {e}")

    def save_stats(self) -> None:
        """Save calibrated agent stats and history to disk."""
        try:
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_file, "w") as f:
                json.dump({
                    "agent_accuracy": self.agent_accuracy,
                    "decision_history": self.decision_history[-100:]
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"[SelfLearningEngine] Failed to save stats: {e}")

    def record_decision(self, symbol: str, entry_price: float, decision: TradingDecision) -> str:
        """Record a new decision to be evaluated against future market price movement."""
        decision_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        record = {
            "id": decision_id,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "entry_price": entry_price,
            "action": decision.action.value,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning[:300] if decision.reasoning else "",
            "opinions": [
                {
                    "role": op.agent_role.value,
                    "action": op.action.value,
                    "confidence": op.confidence
                }
                for op in decision.agent_opinions
            ],
            "evaluated": False,
            "outcome": None,
            "pnl_pct": 0.0,
            "alpha_pct": None,
        }
        self.decision_history.append(record)
        self.save_stats()
        logger.info(f"[SelfLearningEngine] Recorded decision {decision_id} for evaluation.")
        return decision_id

    def evaluate_outcomes(
        self,
        current_prices: Dict[str, float],
        benchmark_prices: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate pending past decisions against current market prices.

        Args:
            current_prices: {symbol: current_price} for traded symbols.
            benchmark_prices: Optional {symbol: (entry_price, current_price)} for the
                              NIFTY 50 benchmark to calculate Alpha.
        """
        evaluated = []
        for rec in self.decision_history:
            if rec["evaluated"]:
                continue

            symbol = rec["symbol"]
            if symbol not in current_prices:
                continue

            entry = rec["entry_price"]
            curr = current_prices[symbol]
            action = rec["action"]

            if entry <= 0:
                continue

            # Calculate net price movement percentage
            change_pct = ((curr - entry) / entry) * 100.0
            rec["pnl_pct"] = round(change_pct, 2)

            # Calculate Alpha vs NIFTY 50 benchmark
            alpha_pct = None
            if benchmark_prices:
                for bench_sym, bench_data in benchmark_prices.items():
                    if isinstance(bench_data, (list, tuple)) and len(bench_data) == 2:
                        bench_entry, bench_curr = bench_data
                        if bench_entry > 0:
                            bench_return = ((bench_curr - bench_entry) / bench_entry) * 100.0
                            alpha_pct = round(change_pct - bench_return, 2)
                            rec["alpha_pct"] = alpha_pct
                            break

            # Determine success with transaction cost buffer (COST_BUFFER_PCT = 0.30%):
            is_success = False
            if action == "BUY" and change_pct > self.COST_BUFFER_PCT:
                is_success = True
            elif action == "SELL" and change_pct < -self.COST_BUFFER_PCT:
                is_success = True
            elif action == "HOLD" and abs(change_pct) <= self.COST_BUFFER_PCT:
                is_success = True

            rec["evaluated"] = True
            rec["outcome"] = "SUCCESS" if is_success else "FAILURE"
            evaluated.append(rec)

            # Record post-trade reflection
            try:
                self.reflection_memory.record_reflection(
                    symbol=symbol,
                    action=action,
                    entry_price=entry,
                    exit_price=curr,
                    pnl_pct=change_pct,
                    alpha_pct=alpha_pct,
                    reasoning_summary=rec.get("reasoning", "No reasoning recorded."),
                    outcome=rec["outcome"],
                )
            except Exception as e:
                logger.warning(f"[SelfLearningEngine] Could not record reflection: {e}")

            # Update accuracy and weights for each individual agent opinion
            for op in rec.get("opinions", []):
                role = op["role"]
                if role in self.agent_accuracy:
                    stats = self.agent_accuracy[role]
                    stats["total"] += 1
                    agent_action = op["action"]

                    # Check if agent prediction matched cost-buffered price movement
                    agent_success = False
                    if agent_action == "BUY" and change_pct > self.COST_BUFFER_PCT:
                        agent_success = True
                    elif agent_action == "SELL" and change_pct < -self.COST_BUFFER_PCT:
                        agent_success = True
                    elif agent_action == "HOLD" and abs(change_pct) <= self.COST_BUFFER_PCT:
                        agent_success = True

                    if agent_success:
                        stats["correct"] += 1

                    stats["accuracy"] = round(stats["correct"] / max(1, stats["total"]), 3)
                    
                    # Calibrate weight safely between 0.75 and 1.25 ONLY after MIN_EVAL_SAMPLES reached
                    if stats["total"] >= self.MIN_EVAL_SAMPLES:
                        stats["weight"] = round(0.75 + (stats["accuracy"] * 0.50), 2)
                    else:
                        stats["weight"] = 1.0

        if evaluated:
            self.save_stats()
            logger.info(f"[SelfLearningEngine] Evaluated {len(evaluated)} decisions with fee-adjusted buffer + alpha tracking.")

        return evaluated

    def get_agent_weight(self, role: AgentRole) -> float:
        """Get calibrated confidence weight for an agent role."""
        role_str = role.value
        return self.agent_accuracy.get(role_str, {}).get("weight", 1.0)
