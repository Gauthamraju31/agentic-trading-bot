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
        self.load_stats()

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
            "pnl_pct": 0.0
        }
        self.decision_history.append(record)
        self.save_stats()
        logger.info(f"[SelfLearningEngine] Recorded decision {decision_id} for evaluation.")
        return decision_id

    def evaluate_outcomes(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Evaluate pending past decisions against current market prices and update agent accuracy weights."""
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

            # Determine success with transaction cost buffer (COST_BUFFER_PCT = 0.30%):
            # BUY is correct if price rose beyond costs (> +0.30%)
            # SELL is correct if price fell beyond costs (< -0.30%)
            # HOLD is correct if price stayed within cost buffer range (abs(change_pct) <= 0.30%)
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
            logger.info(f"[SelfLearningEngine] Evaluated {len(evaluated)} decisions with fee-adjusted buffer.")

        return evaluated

    def get_agent_weight(self, role: AgentRole) -> float:
        """Get calibrated confidence weight for an agent role."""
        role_str = role.value
        return self.agent_accuracy.get(role_str, {}).get("weight", 1.0)
