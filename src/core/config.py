"""Configuration loader for the trading bot.

Reads config/settings.yaml and provides typed access to all settings
via a global `config` singleton.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


# ── Configuration Models ─────────────────────────────────────────────────────


class TradingHoursConfig(BaseModel):
    start: str = "09:15"
    end: str = "15:30"
    pre_open_start: str = "09:00"
    timezone: str = "Asia/Kolkata"


class MarketConfig(BaseModel):
    exchange: str = "NSE"
    symbols: list[str] = Field(default_factory=lambda: ["RELIANCE", "TCS", "INFY"])
    indices: list[str] = Field(default_factory=lambda: ["NIFTY 50", "NIFTY BANK"])
    timeframe: str = "5m"
    trading_hours: TradingHoursConfig = Field(default_factory=TradingHoursConfig)


class MockConfig(BaseModel):
    initial_capital: float = 1_000_000
    slippage_pct: float = 0.05
    brokerage_per_order: float = 20.0


class RiskConfig(BaseModel):
    max_position_pct: float = 5.0
    max_daily_loss_pct: float = 3.0
    max_open_positions: int = 5
    default_stop_loss_pct: float = 2.0
    risk_per_trade_pct: float = 1.0
    position_sizing: str = "fixed_fractional"
    trailing_stop_atr_multiplier: float = 2.0


class FeesConfig(BaseModel):
    """Indian equity market fee structure (rates in percent of turnover).

    Statutory rates as of 2024/2025 (NSE equity):
    - STT: delivery 0.1% on BOTH buy & sell; intraday 0.025% on sell only.
    - Stamp duty (buy only): delivery 0.015%, intraday 0.003%.
    - Exchange txn charge (NSE equity): ~0.00297%.
    - GST: 18% on (brokerage + exchange charges + SEBI charges).
    - SEBI turnover fee: 0.0001% (₹10 per crore).
    """
    brokerage_per_order: float = 20.0
    stt_delivery_pct: float = 0.1        # both buy & sell for delivery (CNC)
    stt_intraday_sell_pct: float = 0.025  # sell side only for intraday (MIS)
    exchange_charges_pct: float = 0.00297
    gst_pct: float = 18.0
    sebi_charges_pct: float = 0.0001
    stamp_duty_delivery_buy_pct: float = 0.015
    stamp_duty_intraday_buy_pct: float = 0.003


class AgyEffortPerRoleConfig(BaseModel):
    """Per-agent reasoning effort levels for dual-speed LLM allocation."""
    technical_analyst: str = "low"
    sentiment_analyst: str = "low"
    bull: str = "medium"
    bear: str = "medium"
    risk_manager: str = "low"
    portfolio_manager: str = "high"
    market_selector: str = "low"


class AgentsConfig(BaseModel):
    llm_provider: str = "agy"          # agy | gemini | openai | antigravity | mock
    model_name: str = "gemini-2.5-flash"  # used by the gemini/openai langchain backends
    temperature: float = 0.3
    debate_rounds: int = 2
    min_confidence: float = 0.6
    max_retries: int = 3
    # Antigravity CLI (agy) backend settings
    agy_model: str = ""                # optional agy model id; empty → agy default
    agy_effort: str = "low"            # default reasoning effort per call
    agy_effort_per_role: AgyEffortPerRoleConfig = Field(default_factory=AgyEffortPerRoleConfig)
    llm_timeout_secs: int = 120        # per-call timeout for the agy CLI
    # LLM quota / rate limiting
    max_rpd: int = 1000                # max requests per day
    max_rpm: int = 30                  # max requests per minute


class WalkForwardConfig(BaseModel):
    in_sample_days: int = 180
    out_sample_days: int = 30


class BacktestConfig(BaseModel):
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    benchmark: str = "NIFTY 50"
    initial_capital: float = 1_000_000


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    auto_refresh_seconds: int = 5


class AlertsConfig(BaseModel):
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    email_enabled: bool = False
    email_smtp_host: str = ""
    email_to: str = ""


class AppConfig(BaseModel):
    name: str = "TradingBot"
    mode: str = "mock"  # mock | live
    log_level: str = "INFO"
    data_dir: str = "data"


class LearningConfig(BaseModel):
    """Self-learning and post-trade reflection memory configuration."""
    reflection_memory_file: str = "data/trade_reflection_memory.md"
    max_reflections_in_prompt: int = 5
    alpha_benchmark: str = "NIFTY 50"


class VIXConfig(BaseModel):
    """India VIX integration for regime-aware risk management."""
    enabled: bool = True
    high_vix_threshold: float = 18.0
    halt_vix_threshold: float = 25.0
    ticker: str = "^INDIAVIX"


class DailyGoalConfig(BaseModel):
    """Daily profit goal and risk limit settings."""
    target_profit: float = 100.0    # Target profit per day in INR
    max_loss: float = 200.0         # Max allowable daily loss in INR
    auto_stop_on_target: bool = True
    pre_market_start: str = "09:00"
    market_open: str = "09:15"
    market_close: str = "15:30"


class Settings(BaseModel):
    """Root configuration model — maps to config/settings.yaml."""

    app: AppConfig = Field(default_factory=AppConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    mock: MockConfig = Field(default_factory=MockConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    fees: FeesConfig = Field(default_factory=FeesConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    vix: VIXConfig = Field(default_factory=VIXConfig)
    daily_goal: DailyGoalConfig = Field(default_factory=DailyGoalConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)


# ── Loader ───────────────────────────────────────────────────────────────────


def load_settings(config_path: Optional[str | Path] = None) -> Settings:
    """Load settings from a YAML file.

    Falls back to sensible defaults if the file doesn't exist.
    """
    if config_path is None:
        # Look relative to project root
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        return Settings(**raw)

    return Settings()


# ── Global singleton ─────────────────────────────────────────────────────────
settings = load_settings()
