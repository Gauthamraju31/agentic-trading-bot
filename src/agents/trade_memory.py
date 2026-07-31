"""Post-Trade Reflection Memory Engine.

Inspired by TauricResearch/TradingAgents' persistent decision log.
Each completed trade gets a one-paragraph LLM-generated (or deterministic)
reflection capturing WHY it succeeded or failed. These reflections are
injected into the Portfolio Manager prompt on subsequent runs for the same
ticker, so each decision carries forward lessons learned.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

REFLECTION_FILE = Path("data/trade_reflection_memory.md")


class TradeReflectionMemory:
    """Manages a persistent markdown log of trade reflections and lessons."""

    def __init__(self, memory_file: Path = REFLECTION_FILE, max_in_prompt: int = 5):
        self.memory_file = memory_file
        self.max_in_prompt = max_in_prompt
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_file.exists():
            self.memory_file.write_text(
                "# Trade Reflection Memory Log\n\n"
                "_Persistent lessons from past trades. Each entry records the decision, "
                "outcome, and a reflection on why it succeeded or failed._\n\n---\n\n"
            )

    def record_reflection(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        alpha_pct: Optional[float],
        reasoning_summary: str,
        outcome: str,
    ) -> None:
        """Append a structured reflection entry to the memory log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        alpha_str = f" | Alpha vs NIFTY: {alpha_pct:+.2f}%" if alpha_pct is not None else ""

        # Generate deterministic reflection
        if outcome == "SUCCESS":
            if alpha_pct is not None and alpha_pct > 0:
                lesson = (
                    f"The {action} on {symbol} was correct and beat the benchmark. "
                    f"Key factors: {reasoning_summary[:200]}. "
                    f"Continue monitoring similar setups when indicators align."
                )
            else:
                lesson = (
                    f"The {action} on {symbol} was profitable but underperformed NIFTY. "
                    f"Consider whether the risk-reward justified the trade vs holding index."
                )
        else:
            lesson = (
                f"The {action} on {symbol} resulted in a loss. "
                f"Review: {reasoning_summary[:200]}. "
                f"Potential causes: wrong timing, insufficient conviction threshold, "
                f"or adverse macro conditions. Consider tighter stops next time."
            )

        entry = (
            f"## [{timestamp}] {symbol} — {action} → {outcome}\n"
            f"- Entry: ₹{entry_price:.2f} → Exit: ₹{exit_price:.2f} | P&L: {pnl_pct:+.2f}%{alpha_str}\n"
            f"- **Reflection:** {lesson}\n\n---\n\n"
        )

        try:
            with open(self.memory_file, "a") as f:
                f.write(entry)
            logger.info(f"[TradeReflectionMemory] Recorded reflection for {symbol} {action} ({outcome})")
        except Exception as e:
            logger.error(f"[TradeReflectionMemory] Failed to write reflection: {e}")

    def get_recent_reflections(self, symbol: Optional[str] = None, limit: Optional[int] = None) -> str:
        """Read recent reflections from memory, optionally filtered by symbol.

        Returns a formatted string suitable for injection into an LLM prompt.
        """
        limit = limit or self.max_in_prompt
        if not self.memory_file.exists():
            return ""

        try:
            content = self.memory_file.read_text()
        except Exception:
            return ""

        # Parse entries (split by the --- separator)
        entries = [e.strip() for e in content.split("---") if e.strip() and e.strip().startswith("##")]

        # Filter by symbol if specified
        if symbol:
            symbol_upper = symbol.upper()
            entries = [e for e in entries if symbol_upper in e.upper()]

        # Take the most recent N entries
        recent = entries[-limit:] if len(entries) > limit else entries

        if not recent:
            return ""

        header = f"## Past Trade Reflections{f' for {symbol}' if symbol else ''} (most recent {len(recent)}):\n\n"
        return header + "\n---\n".join(recent) + "\n"

    def get_cross_ticker_lessons(self, exclude_symbol: str, limit: int = 3) -> str:
        """Get lessons from OTHER tickers to provide cross-market awareness."""
        if not self.memory_file.exists():
            return ""

        try:
            content = self.memory_file.read_text()
        except Exception:
            return ""

        entries = [e.strip() for e in content.split("---") if e.strip() and e.strip().startswith("##")]
        exclude_upper = exclude_symbol.upper()
        other_entries = [e for e in entries if exclude_upper not in e.upper()]

        recent = other_entries[-limit:] if len(other_entries) > limit else other_entries
        if not recent:
            return ""

        return f"## Recent cross-ticker lessons:\n\n" + "\n---\n".join(recent) + "\n"
