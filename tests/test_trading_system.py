"""Comprehensive test suite for the Agentic Trading Bot system."""

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.agents.orchestrator import AgentOrchestrator
from src.core.config import settings
from src.core.enums import Exchange, OrderStatus, OrderType, PositionType, Side, SignalAction, TimeFrame
from src.core.models import Candle, MarketContext, Order, PortfolioState, Position, Signal
from src.data.indicators import IndicatorEngine
from src.execution.fees import FeeCalculator
from src.execution.mock_engine import MockBroker
from src.execution.order_validator import OrderValidator
from src.execution.position_manager import PositionManager
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import StopLossManager
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.momentum import MomentumStrategy


def test_core_enums():
    assert Side.BUY.value == "BUY"
    assert Side.SELL.value == "SELL"
    assert TimeFrame.M5.value == "5m"
    assert Exchange.NSE.value == "NSE"


def test_fee_calculator():
    calc = FeeCalculator()
    fees_buy = calc.calculate_fees(Side.BUY, 10, 2500.0, PositionType.DELIVERY)
    assert fees_buy.brokerage > 0
    assert fees_buy.total > 0

    fees_sell = calc.calculate_fees(Side.SELL, 10, 2550.0, PositionType.DELIVERY)
    assert fees_sell.stt > 0  # STT charged on sell side


def test_position_sizer():
    sizer = PositionSizer()
    qty = sizer.calculate_size(
        method="fixed_fractional",
        capital=1000000,
        risk_per_trade_pct=1.0,
        entry_price=2500.0,
        stop_loss_price=2450.0,
    )
    assert qty > 0
    assert qty * 2500.0 <= 1000000 * (settings.risk.max_position_pct / 100.0)


def test_stop_loss_manager():
    slm = StopLossManager()
    atr_sl = slm.calculate_stop_loss("atr_based", Side.BUY, 2500.0, atr=10.0, atr_multiplier=2.0)
    assert atr_sl == 2480.0

    fixed_sl = slm.calculate_stop_loss("fixed_pct", Side.BUY, 2500.0, pct=2.0)
    assert fixed_sl == 2450.0

    tp = slm.calculate_take_profit(Side.BUY, 2500.0, risk_reward_ratio=2.0, stop_loss=fixed_sl)
    assert tp == 2600.0


def test_circuit_breaker():
    cb = CircuitBreaker()
    portfolio = PortfolioState(initial_capital=1000000, current_capital=1000000)
    triggered, reason = cb.check(portfolio)
    assert not triggered

    portfolio.daily_pnl = -35000  # -3.5% daily loss
    triggered, reason = cb.check(portfolio)
    assert triggered
    assert "Daily loss" in reason


def test_indicators_engine():
    np.random.seed(42)
    prices = [2500.0 + i for i in range(100)]
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2025-01-01", periods=100, freq="5min"),
            "open": prices,
            "high": [p + 5 for p in prices],
            "low": [p - 5 for p in prices],
            "close": prices,
            "volume": [100000] * 100,
        }
    )
    df_ind = IndicatorEngine.calculate(df)
    assert "RSI_14" in df_ind.columns
    assert "MACD" in df_ind.columns
    assert "ATR_14" in df_ind.columns

    ind_values = IndicatorEngine.get_latest_indicators(df_ind)
    assert ind_values is not None
    assert ind_values.rsi_14 is not None


@pytest.mark.asyncio
async def test_mock_broker():
    broker = MockBroker(initial_capital=1000000)
    candle = Candle(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        open=2500.0,
        high=2510.0,
        low=2490.0,
        close=2505.0,
        volume=100000,
    )
    await broker.process_candle(candle)

    order = Order(
        symbol="RELIANCE",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=10,
        price=2500.0,
    )
    filled_order = await broker.place_order(order)
    assert filled_order.status == OrderStatus.FILLED
    assert filled_order.filled_quantity == 10

    portfolio = await broker.get_portfolio()
    assert portfolio.open_position_count == 1


@pytest.mark.asyncio
async def test_agent_orchestrator():
    orchestrator = AgentOrchestrator()
    candle = Candle(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        open=2500.0,
        high=2510.0,
        low=2490.0,
        close=2505.0,
        volume=100000,
    )
    context = MarketContext(
        symbol="RELIANCE",
        current_price=2505.0,
        candles=[candle],
    )
    decision = await orchestrator.run(context)
    assert decision is not None
    assert decision.action in (SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD, SignalAction.EXIT)


@pytest.mark.asyncio
async def test_sentiment_fetcher():
    from src.data.sentiment import SentimentFetcher
    fetcher = SentimentFetcher()
    data = await fetcher.compute_headline_sentiment("RELIANCE")
    assert "overall_score" in data
    assert "fii_dii_summary" in data
    assert -1.0 <= data["overall_score"] <= 1.0
    await fetcher.close()


def test_self_learning_engine_cost_buffer_and_limits(tmp_path):
    from src.agents.self_learning import SelfLearningEngine
    from src.core.models import TradingDecision, AgentOpinion
    from src.core.enums import AgentRole

    stats_file = tmp_path / "learning_stats.json"
    engine = SelfLearningEngine(stats_file=stats_file)

    decision = TradingDecision(
        symbol="RELIANCE",
        action=SignalAction.BUY,
        confidence=0.8,
        reasoning="Test decision",
        agent_opinions=[
            AgentOpinion(agent_role=AgentRole.BULL, reasoning="Bullish", confidence=0.8, action=SignalAction.BUY)
        ]
    )

    # 1. Record decision
    rec_id = engine.record_decision("RELIANCE", 2500.0, decision)
    assert rec_id is not None

    # 2. Evaluate with a tiny move (+0.1%) -> Should NOT count as success because of 0.3% fee buffer
    eval_res = engine.evaluate_outcomes({"RELIANCE": 2502.5}) # +0.1% move
    assert len(eval_res) == 1
    assert eval_res[0]["outcome"] == "FAILURE" # below 0.3% fee buffer

    # 3. Check weight remains 1.0 because total evaluations < MIN_EVAL_SAMPLES (5)
    weight = engine.get_agent_weight(AgentRole.BULL)
    assert weight == 1.0


# ── Tests for TradingAgents-inspired features ──────────────────────────────────


def test_trade_reflection_memory(tmp_path):
    """Test that the reflection memory system records and retrieves lessons."""
    from src.agents.trade_memory import TradeReflectionMemory

    mem_file = tmp_path / "test_reflections.md"
    memory = TradeReflectionMemory(memory_file=mem_file, max_in_prompt=3)

    # Record a successful trade
    memory.record_reflection(
        symbol="RELIANCE",
        action="BUY",
        entry_price=1300.0,
        exit_price=1320.0,
        pnl_pct=1.54,
        alpha_pct=0.72,
        reasoning_summary="Strong RSI momentum above 60 with bullish MACD crossover",
        outcome="SUCCESS",
    )

    # Record a failed trade
    memory.record_reflection(
        symbol="TCS",
        action="BUY",
        entry_price=2400.0,
        exit_price=2380.0,
        pnl_pct=-0.83,
        alpha_pct=-1.1,
        reasoning_summary="Bought on weak sentiment during FII selling",
        outcome="FAILURE",
    )

    # Retrieve same-ticker reflections
    reliance_reflections = memory.get_recent_reflections(symbol="RELIANCE")
    assert "RELIANCE" in reliance_reflections
    assert "SUCCESS" in reliance_reflections
    assert "+1.54%" in reliance_reflections

    # Retrieve cross-ticker lessons (excluding RELIANCE)
    cross_lessons = memory.get_cross_ticker_lessons(exclude_symbol="RELIANCE")
    assert "TCS" in cross_lessons
    assert "FAILURE" in cross_lessons
    assert "RELIANCE" not in cross_lessons.split("##")[1] if "##" in cross_lessons else True

    # Empty filter - use a symbol that does NOT appear in any reflection text
    empty = memory.get_recent_reflections(symbol="ZOMATO")
    assert empty == ""


def test_alpha_vs_nifty_tracking(tmp_path):
    """Test that evaluate_outcomes correctly computes alpha vs NIFTY 50."""
    from src.agents.self_learning import SelfLearningEngine
    from src.core.models import TradingDecision, AgentOpinion
    from src.core.enums import AgentRole

    stats_file = tmp_path / "alpha_test_stats.json"
    engine = SelfLearningEngine(stats_file=stats_file)

    decision = TradingDecision(
        symbol="RELIANCE",
        action=SignalAction.BUY,
        confidence=0.8,
        reasoning="Alpha test trade",
        agent_opinions=[
            AgentOpinion(agent_role=AgentRole.TECHNICAL_ANALYST, reasoning="TA", confidence=0.7, action=SignalAction.BUY)
        ]
    )
    engine.record_decision("RELIANCE", 1300.0, decision)

    # Stock gained +1.5%, NIFTY gained +1.0% → alpha should be +0.5%
    evaluated = engine.evaluate_outcomes(
        current_prices={"RELIANCE": 1319.5},
        benchmark_prices={"NIFTY 50": (24000.0, 24240.0)},  # +1.0%
    )

    assert len(evaluated) == 1
    assert evaluated[0]["alpha_pct"] is not None
    assert evaluated[0]["alpha_pct"] == pytest.approx(0.5, abs=0.1)
    assert evaluated[0]["outcome"] == "SUCCESS"  # +1.5% > 0.3% cost buffer


def test_india_vix_regime_classification():
    """Test VIX regime classification and position multiplier logic."""
    from src.data.india_vix import IndiaVIXFetcher

    fetcher = IndiaVIXFetcher()

    # Low volatility
    assert fetcher._classify_regime(10.0) == "LOW_VOLATILITY"
    assert fetcher._position_size_multiplier(10.0) == 1.0

    # Moderate
    assert fetcher._classify_regime(15.0) == "MODERATE"
    assert fetcher._position_size_multiplier(15.0) == 0.8

    # High volatility
    assert fetcher._classify_regime(20.0) == "HIGH_VOLATILITY"
    assert fetcher._position_size_multiplier(20.0) == 0.5

    # Extreme
    assert fetcher._classify_regime(28.0) == "EXTREME"
    assert fetcher._position_size_multiplier(28.0) == 0.25


def test_dual_speed_llm_config():
    """Test that per-agent effort levels are properly configured."""
    agents_cfg = settings.agents
    effort_map = getattr(agents_cfg, "agy_effort_per_role", None)
    assert effort_map is not None

    # Quick agents should use low effort
    assert getattr(effort_map, "technical_analyst", "low") == "low"
    assert getattr(effort_map, "sentiment_analyst", "low") == "low"

    # Debate agents should use medium effort
    assert getattr(effort_map, "bull", "low") == "medium"
    assert getattr(effort_map, "bear", "low") == "medium"

    # Portfolio manager should use high effort for deep reasoning
    assert getattr(effort_map, "portfolio_manager", "low") == "high"


def test_market_context_has_vix_and_benchmark_fields():
    """Test that MarketContext model accepts VIX and benchmark data."""
    candle = Candle(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        open=1300.0, high=1310.0, low=1295.0, close=1305.0,
        volume=50000,
    )
    ctx = MarketContext(
        symbol="RELIANCE",
        current_price=1305.0,
        candles=[candle],
        vix_data={"vix_value": 14.5, "regime": "MODERATE", "position_size_multiplier": 0.8},
        benchmark_data={"entry_price": 24000.0, "current_price": 24050.0},
    )
    assert ctx.vix_data["regime"] == "MODERATE"
    assert ctx.benchmark_data["entry_price"] == 24000.0
