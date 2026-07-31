import os
import json
import re
import shutil
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Type

from pydantic import BaseModel, ValidationError

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    ChatGoogleGenerativeAI = None
    ChatOpenAI = None
    SystemMessage = None
    HumanMessage = None

try:
    from google.antigravity import Agent as AntigravityAgent, LocalAgentConfig
except ImportError:
    AntigravityAgent = None
    LocalAgentConfig = None

from loguru import logger

from src.core.models import MarketContext, AgentOpinion
from src.core.enums import AgentRole, SignalAction
from src.core.config import settings
from src.core.quota_tracker import QuotaTracker

# ── Shared LLM quota tracker (one per process, shared across all agents) ───────
_quota_tracker = QuotaTracker(
    max_rpd=getattr(settings.agents, "max_rpd", 1000),
    max_rpm=getattr(settings.agents, "max_rpm", 30),
)


def _resolve_agy_binary() -> Optional[str]:
    """Locate the Antigravity `agy` CLI. It commonly lives in ~/.local/bin which
    is NOT on a non-interactive shell's PATH, so check explicit locations too."""
    found = shutil.which("agy")
    if found:
        return found
    candidates = [
        Path.home() / ".local" / "bin" / "agy",
        Path("/usr/local/bin/agy"),
        Path("/usr/bin/agy"),
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return str(c)
    return None


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from an LLM response.
    Handles ```json fences and surrounding prose."""
    if not text:
        return None
    # Strip markdown code fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        # Fall back to the first '{' ... last '}' span.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


class BaseAgent(ABC):
    """Abstract base class for all AI agents in the trading bot.

    LLM backends supported (via `agents.llm_provider`):
      - "agy" / "antigravity-cli": the Antigravity `agy` CLI, invoked headlessly
        with `-p` (print mode). Requires a one-time interactive sign-in
        (`agy` with no args) on the host; until then calls fail and the agent
        falls back to deterministic logic.
      - "gemini" / "openai": LangChain chat models (need an API key in env).
      - "antigravity": the google.antigravity Python SDK (if installed).
    Any provider that is unavailable degrades gracefully to deterministic rules,
    logged loudly at WARNING so a silent facade is impossible.
    """

    def __init__(self, role: AgentRole, system_prompt: str):
        self.role = role
        self.system_prompt = system_prompt
        self._agy_path: Optional[str] = None
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Resolve the configured LLM backend, or return None for fallback mode."""
        provider = getattr(getattr(settings, "agents", None), "llm_provider", "mock").lower()
        model_name = getattr(getattr(settings, "agents", None), "model_name", "mock")

        if provider in ("agy", "antigravity-cli", "cli"):
            self._agy_path = _resolve_agy_binary()
            if self._agy_path:
                logger.info(f"[{self.role.name}] Antigravity CLI provider at {self._agy_path}.")
                return "agy"
            logger.warning(
                f"[{self.role.name}] agy CLI not found on host; falling back to deterministic logic. "
                f"Install it or add ~/.local/bin to PATH."
            )
            return None

        if provider in ("antigravity", "google-antigravity", "sdk"):
            if AntigravityAgent:
                logger.info(f"[{self.role.name}] Antigravity SDK provider initialized.")
                return "antigravity"
            logger.warning(f"[{self.role.name}] Antigravity SDK not installed; using deterministic fallback.")
            return None

        if provider in ("google", "gemini"):
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key and ChatGoogleGenerativeAI:
                return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=settings.agents.temperature)
            logger.warning(f"[{self.role.name}] GEMINI_API_KEY missing or langchain unavailable; deterministic fallback.")
            return None

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and ChatOpenAI:
                return ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=settings.agents.temperature)
            logger.warning(f"[{self.role.name}] OPENAI_API_KEY missing or langchain unavailable; deterministic fallback.")
            return None

        logger.info(f"[{self.role.name}] Provider '{provider}' → deterministic fallback mode.")
        return None

    @abstractmethod
    async def analyze(self, market_context: MarketContext, **kwargs) -> AgentOpinion:
        """Analyze the market context and return an opinion."""
        raise NotImplementedError

    # ── LLM invocation ────────────────────────────────────────────────────────

    async def _run_agy(self, prompt: str) -> Optional[str]:
        """Invoke the agy CLI in headless print mode. Returns text, or None if
        unavailable/unauthenticated/errored (caller then falls back).

        Uses dual-speed LLM allocation: each agent role can have its own
        reasoning effort level (low/medium/high) configured in
        settings.agents.agy_effort_per_role.
        """
        if not self._agy_path:
            return None
        if not _quota_tracker.can_make_request():
            logger.warning(f"[{self.role.name}] LLM quota/rate limit hit; skipping call, using fallback.")
            return None

        timeout = int(getattr(settings.agents, "llm_timeout_secs", 120))

        # Dual-speed LLM: resolve per-role effort, fall back to global default
        default_effort = getattr(settings.agents, "agy_effort", "low")
        effort_map = getattr(settings.agents, "agy_effort_per_role", None)
        role_key = self.role.value  # e.g. "technical_analyst", "bull", "portfolio_manager"
        if effort_map and hasattr(effort_map, role_key):
            effort = getattr(effort_map, role_key, default_effort)
        elif isinstance(effort_map, dict) and role_key in effort_map:
            effort = effort_map[role_key]
        else:
            effort = default_effort

        cmd = [self._agy_path, "-p", prompt, "--dangerously-skip-permissions",
               "--print-timeout", f"{timeout}s", "--effort", effort]
        agy_model = getattr(settings.agents, "agy_model", "")
        if agy_model:
            cmd += ["--model", agy_model]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout + 20)
        except asyncio.TimeoutError:
            logger.warning(f"[{self.role.name}] agy CLI timed out after {timeout}s; using fallback.")
            return None
        except Exception as e:
            logger.warning(f"[{self.role.name}] agy CLI invocation failed ({e}); using fallback.")
            return None

        _quota_tracker.record_request()
        out = (stdout or b"").decode(errors="ignore").strip()
        err = (stderr or b"").decode(errors="ignore").strip()

        combined = (out + " " + err).lower()
        if any(s in combined for s in ("please sign in", "authentication required", "authentication failed", "log in", "not signed in")):
            logger.warning(
                f"[{self.role.name}] agy is not authenticated — run `agy` interactively on the host to "
                f"log in, then retry. Using deterministic fallback until then."
            )
            return None
        if proc.returncode != 0 or not out:
            logger.warning(f"[{self.role.name}] agy returned rc={proc.returncode}, no usable output "
                           f"({(err or out)[:180]}); using fallback.")
            return None
        return out

    async def _run_llm_raw(self, prompt: str) -> Optional[str]:
        """Dispatch a raw prompt to the active backend; None if unavailable."""
        if self.llm is None:
            return None
        if self.llm == "agy":
            return await self._run_agy(prompt)

        if self.llm == "antigravity" and AntigravityAgent:
            try:
                api_key = os.getenv("GEMINI_API_KEY")
                config = (LocalAgentConfig(system_instructions=self.system_prompt, api_key=api_key)
                          if api_key else LocalAgentConfig(system_instructions=self.system_prompt, vertex=True))
                async with AntigravityAgent(config) as agent:
                    response = await agent.chat(prompt)
                    text = ""
                    async for token in response:
                        text += token
                    _quota_tracker.record_request()
                    return text
            except Exception as e:
                logger.warning(f"[{self.role.name}] Antigravity SDK failed ({e}); using fallback.")
                return None

        # LangChain chat models (gemini/openai)
        try:
            messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            _quota_tracker.record_request()
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning(f"[{self.role.name}] LLM call failed ({e}); using fallback.")
            return None

    @staticmethod
    def _schema_hint(model: Type[BaseModel]) -> str:
        """Render a compact JSON shape hint from a Pydantic model's fields."""
        parts = []
        for name, field in model.model_fields.items():
            ann = field.annotation
            type_name = getattr(ann, "__name__", str(ann))
            parts.append(f'"{name}": <{type_name}>')
        return "{ " + ", ".join(parts) + " }"

    async def _generate_structured(
        self, task: str, context: Dict[str, Any], schema: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """Ask the LLM for a structured decision and validate it into `schema`.

        Returns a validated model instance, or None if the LLM is unavailable or
        never produced valid output (the caller then uses deterministic logic).
        This is the mechanism that lets the LLM actually DRIVE decisions rather
        than narrate a pre-computed one.
        """
        if self.llm is None:
            return None

        hint = self._schema_hint(schema)
        base_prompt = (
            f"{self.system_prompt}\n\n"
            f"VERIFIED CONTEXT (use only these numbers; do not invent data):\n"
            f"{json.dumps(context, default=str, indent=2)}\n\n"
            f"TASK: {task}\n\n"
            f"Enum values: action/proposed_action/overall_action ∈ [\"BUY\", \"SELL\", \"HOLD\", \"EXIT\"]. "
            f"confidence/score fields are decimals in [0,1].\n"
            f"Respond with ONLY one JSON object (no markdown, no commentary) matching exactly:\n{hint}"
        )

        prompt = base_prompt
        attempts = max(1, int(getattr(settings.agents, "max_retries", 3)))
        for attempt in range(attempts):
            raw = await self._run_llm_raw(prompt)
            if raw is None:
                return None  # backend unavailable — fall back deterministically
            parsed = _extract_json(raw)
            if parsed is not None:
                try:
                    return schema(**parsed)
                except (ValidationError, TypeError) as e:
                    logger.debug(f"[{self.role.name}] structured parse invalid (attempt {attempt + 1}/{attempts}): {e}")
            prompt = base_prompt + "\n\nYour previous reply was not valid JSON for the schema. Return ONLY the JSON object."

        logger.warning(f"[{self.role.name}] No valid structured output after {attempts} attempts; using fallback.")
        return None

    async def _generate_response(self, prompt_kwargs: Dict[str, Any]) -> str:
        """Free-form prose response (kept for narration). Falls back to a
        deterministic string if the backend is unavailable."""
        prompt = (
            f"Context data:\n{json.dumps(prompt_kwargs, default=str)}\n\n"
            f"Provide a concise analysis and a clear recommendation (BUY, SELL, or HOLD)."
        )
        raw = await self._run_llm_raw(prompt)
        if raw:
            return raw
        return self._fallback_logic(prompt_kwargs)

    def _fallback_logic(self, context: Dict[str, Any]) -> str:
        """Heuristic prose used when no LLM backend is available."""
        return f"[deterministic fallback] {self.role.name} assessment from technical indicators and market structure."

    def _create_opinion(self, reasoning: str, confidence: float = 0.5, action: SignalAction = SignalAction.HOLD) -> AgentOpinion:
        return AgentOpinion(
            agent_role=self.role,
            reasoning=reasoning,
            confidence=max(0.0, min(1.0, confidence)),
            action=action,
        )
