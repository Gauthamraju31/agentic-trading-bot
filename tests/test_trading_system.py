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

