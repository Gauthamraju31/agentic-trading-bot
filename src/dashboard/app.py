"""FastAPI Web Dashboard & Interactive Control Center for the Trading Bot."""

import asyncio
from datetime import datetime
from pathlib import Path
import sys
import threading
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel
import uvicorn

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.config import settings
from src.core.enums import Exchange, OrderType, Side, SignalAction, TimeFrame
from src.core.models import MarketContext, Order, Position, Trade

app = FastAPI(title="Trading Bot Control Center")

# Global state for bot process control & UI state
bot_state = {
    "status": "STOPPED",  # STOPPED | RUNNING | PAUSED | CIRCUIT_BROKEN
    "mode": "mock",  # mock | live
    "symbol": "RELIANCE",
    "strategy": "multi_agent",  # multi_agent | momentum | mean_reversion
    "interval": 2,
    "circuit_breaker_active": False,
    "equity": settings.mock.initial_capital,
    "initial_capital": settings.mock.initial_capital,
    "cash": settings.mock.initial_capital,
    "realized_pnl": 0.0,
    "unrealized_pnl": 0.0,
    "max_drawdown_pct": 0.0,
    "win_rate": 0.0,
    "positions": [],
    "trades": [],
    "logs": [],
    "backtest_result": None,
}

bot_task: Optional[asyncio.Task] = None
bot_loop_stop_event = asyncio.Event()


def log_to_dashboard(msg: str, level: str = "INFO"):
    """Append a log entry to the dashboard log stream."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {"timestamp": timestamp, "message": msg, "level": level}
    bot_state["logs"].append(entry)
    if len(bot_state["logs"]) > 200:
        bot_state["logs"].pop(0)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agentic Trading Bot — Control Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --card-border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --accent-amber: #f59e0b;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--card-border);
            margin-bottom: 24px;
        }

        .title-area h1 {
            margin: 0;
            font-size: 24px;
            font-weight: 700;
            color: #ffffff;
        }

        .title-area p {
            margin: 4px 0 0 0;
            color: var(--text-muted);
            font-size: 13px;
        }

        .status-badge {
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 13px;
            letter-spacing: 0.5px;
        }

        .status-STOPPED { background: #475569; color: #f8fafc; }
        .status-RUNNING { background: #065f46; color: #34d399; }
        .status-PAUSED { background: #78350f; color: #fbbf24; }
        .status-CIRCUIT_BROKEN { background: #7f1d1d; color: #f87171; }

        .control-panel {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: center;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .form-group label {
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
        }

        select, input {
            background: #0f172a;
            border: 1px solid var(--card-border);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
            outline: none;
        }

        select:focus, input:focus {
            border-color: var(--accent-blue);
        }

        .btn-group {
            display: flex;
            gap: 10px;
            margin-left: auto;
        }

        button {
            border: none;
            padding: 10px 18px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn-start { background: var(--accent-green); color: #022c22; }
        .btn-start:hover { background: #059669; }
        .btn-stop { background: var(--accent-amber); color: #451a03; }
        .btn-stop:hover { background: #d97706; }
        .btn-kill { background: var(--accent-red); color: #ffffff; }
        .btn-kill:hover { background: #dc2626; }
        .btn-action { background: #334155; color: #f8fafc; }
        .btn-action:hover { background: #475569; }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 16px;
        }

        .metric-card .title {
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
            text-transform: uppercase;
        }

        .metric-card .value {
            font-size: 22px;
            font-weight: 700;
            margin-top: 8px;
        }

        .positive { color: var(--accent-green); }
        .negative { color: var(--accent-red); }

        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }

        @media (max-width: 1024px) {
            .main-grid { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 20px;
        }

        .card h2 {
            margin: 0 0 16px 0;
            font-size: 16px;
            font-weight: 600;
            color: #ffffff;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        th, td {
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid #2d3748;
        }

        th {
            color: var(--text-muted);
            font-weight: 500;
            background: #111827;
        }

        .log-console {
            background: #090d16;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 12px;
            height: 320px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
            line-height: 1.6;
        }

        .log-entry { margin-bottom: 4px; }
        .log-INFO { color: #94a3b8; }
        .log-SUCCESS { color: #34d399; }
        .log-WARNING { color: #fbbf24; }
        .log-ERROR { color: #f87171; }

        .modal-bg {
            display: none;
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.7);
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .modal {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 24px;
            max-width: 500px;
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="title-area">
            <h1>Agentic Trading Bot Control Center</h1>
            <p>NSE/BSE Multi-Agent Trading System — Real-Time Monitor & Execution Control</p>
        </div>
        <div id="status-badge" class="status-badge status-STOPPED">STOPPED</div>
    </div>

    <!-- CONTROL PANEL -->
    <div class="control-panel">
        <div class="form-group">
            <label>Mode</label>
            <select id="select-mode">
                <option value="mock" selected>Mock / Paper Trading</option>
                <option value="live">Live Trading (Broker)</option>
            </select>
        </div>

        <div class="form-group">
            <label>Symbol</label>
            <select id="select-symbol">
                <option value="RELIANCE" selected>RELIANCE</option>
                <option value="TCS">TCS</option>
                <option value="INFY">INFY</option>
                <option value="HDFCBANK">HDFCBANK</option>
                <option value="NIFTY 50">NIFTY 50</option>
            </select>
        </div>

        <div class="form-group">
            <label>Strategy / Pipeline</label>
            <select id="select-strategy">
                <option value="multi_agent" selected>Multi-Agent AI Pipeline (LangGraph)</option>
                <option value="momentum">Momentum (EMA + RSI + ADX)</option>
                <option value="mean_reversion">Mean Reversion (Bollinger + RSI)</option>
            </select>
        </div>

        <div class="form-group">
            <label>Tick Interval (s)</label>
            <input type="number" id="input-interval" value="2" min="1" max="60" style="width: 70px;">
        </div>

        <div class="btn-group">
            <button class="btn-start" onclick="startBot()">▶ Start Bot</button>
            <button class="btn-stop" onclick="stopBot()">⏸ Stop Bot</button>
            <button class="btn-action" onclick="runBacktest()">📊 Run Backtest</button>
            <button class="btn-action" onclick="downloadData()">📥 Download Data</button>
            <button class="btn-kill" onclick="resetCircuitBreaker()">⚡ Reset Circuit</button>
        </div>
    </div>

    <!-- METRICS GRID -->
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="title">Total Equity</div>
            <div class="value" id="val-equity">₹10,00,000.00</div>
        </div>
        <div class="metric-card">
            <div class="title">Cash Available</div>
            <div class="value" id="val-cash">₹10,00,000.00</div>
        </div>
        <div class="metric-card">
            <div class="title">Realized P&L</div>
            <div class="value" id="val-realized">₹0.00</div>
        </div>
        <div class="metric-card">
            <div class="title">Unrealized P&L</div>
            <div class="value" id="val-unrealized">₹0.00</div>
        </div>
        <div class="metric-card">
            <div class="title">Max Drawdown</div>
            <div class="value negative" id="val-drawdown">0.00%</div>
        </div>
    </div>

    <!-- MAIN GRID (POSITIONS & LOG CONSOLE) -->
    <div class="main-grid">
        <div class="card">
            <h2>Open Positions <span id="pos-count" style="font-size: 13px; color: var(--text-muted);">(0)</span></h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Qty</th>
                        <th>Entry Price</th>
                        <th>Current Price</th>
                        <th>Unrealized P&L</th>
                    </tr>
                </thead>
                <tbody id="tbl-positions">
                    <tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No open positions</td></tr>
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>Live Log Console</h2>
            <div class="log-console" id="log-console">
                <div class="log-entry log-INFO">[00:00:00] Dashboard connected. Ready to start bot.</div>
            </div>
        </div>
    </div>

    <!-- COMPLETED TRADES TABLE -->
    <div class="card">
        <h2>Completed Trades History</h2>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Entry Price</th>
                    <th>Exit Price</th>
                    <th>Net P&L</th>
                    <th>Holding Period</th>
                </tr>
            </thead>
            <tbody id="tbl-trades">
                <tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No completed trades yet</td></tr>
            </tbody>
        </table>
    </div>

    <!-- BACKTEST MODAL -->
    <div class="modal-bg" id="modal-backtest">
        <div class="modal">
            <h2 style="margin-top:0;">Backtest Results</h2>
            <div id="modal-content">Running backtest...</div>
            <button class="btn-action" style="margin-top:16px; float:right;" onclick="closeModal()">Close</button>
        </div>
    </div>

    <script>
        async function updateDashboard() {
            try {
                const res = await fetch('/api/state');
                const state = await res.json();

                // Update Status Badge
                const badge = document.getElementById('status-badge');
                badge.innerText = state.status;
                badge.className = 'status-badge status-' + state.status;

                // Update Metrics
                document.getElementById('val-equity').innerText = '₹' + state.equity.toLocaleString('en-IN', {minimumFractionDigits: 2});
                document.getElementById('val-cash').innerText = '₹' + state.cash.toLocaleString('en-IN', {minimumFractionDigits: 2});
                
                const realElem = document.getElementById('val-realized');
                realElem.innerText = '₹' + state.realized_pnl.toFixed(2);
                realElem.className = 'value ' + (state.realized_pnl >= 0 ? 'positive' : 'negative');

                const unrealElem = document.getElementById('val-unrealized');
                unrealElem.innerText = '₹' + state.unrealized_pnl.toFixed(2);
                unrealElem.className = 'value ' + (state.unrealized_pnl >= 0 ? 'positive' : 'negative');

                document.getElementById('val-drawdown').innerText = state.max_drawdown_pct.toFixed(2) + '%';

                // Update Positions
                const posTbody = document.getElementById('tbl-positions');
                document.getElementById('pos-count').innerText = `(${state.positions.length})`;
                if (state.positions.length === 0) {
                    posTbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No open positions</td></tr>';
                } else {
                    posTbody.innerHTML = state.positions.map(p => `
                        <tr>
                            <td><b>${p.symbol}</b></td>
                            <td><span style="color: ${p.side === 'BUY' ? 'var(--accent-green)' : 'var(--accent-red)'}">${p.side}</span></td>
                            <td>${p.quantity}</td>
                            <td>₹${p.entry_price.toFixed(2)}</td>
                            <td>₹${p.current_price.toFixed(2)}</td>
                            <td class="${p.unrealized_pnl >= 0 ? 'positive' : 'negative'}">₹${p.unrealized_pnl.toFixed(2)}</td>
                        </tr>
                    `).join('');
                }

                // Update Trades
                const tradeTbody = document.getElementById('tbl-trades');
                if (state.trades.length === 0) {
                    tradeTbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No completed trades yet</td></tr>';
                } else {
                    tradeTbody.innerHTML = state.trades.map(t => `
                        <tr>
                            <td><b>${t.symbol}</b></td>
                            <td>${t.side}</td>
                            <td>${t.quantity}</td>
                            <td>₹${t.entry_price.toFixed(2)}</td>
                            <td>₹${t.exit_price.toFixed(2)}</td>
                            <td class="${t.net_pnl >= 0 ? 'positive' : 'negative'}">₹${t.net_pnl.toFixed(2)}</td>
                            <td>${t.holding_period || 'N/A'}</td>
                        </tr>
                    `).join('');
                }

                // Update Logs
                const logConsole = document.getElementById('log-console');
                logConsole.innerHTML = state.logs.map(l => `
                    <div class="log-entry log-${l.level}">[${l.timestamp}] ${l.message}</div>
                `).join('');
                logConsole.scrollTop = logConsole.scrollHeight;

            } catch (e) {
                console.error("Failed to update dashboard", e);
            }
        }

        async function startBot() {
            const mode = document.getElementById('select-mode').value;
            const symbol = document.getElementById('select-symbol').value;
            const strategy = document.getElementById('select-strategy').value;
            const interval = parseInt(document.getElementById('input-interval').value);

            await fetch('/api/control/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ mode, symbol, strategy, interval })
            });
            updateDashboard();
        }

        async function stopBot() {
            await fetch('/api/control/stop', { method: 'POST' });
            updateDashboard();
        }

        async function resetCircuitBreaker() {
            await fetch('/api/circuit-breaker/reset', { method: 'POST' });
            updateDashboard();
        }

        async function downloadData() {
            const symbol = document.getElementById('select-symbol').value;
            const res = await fetch('/api/control/download-data?symbol=' + encodeURIComponent(symbol), { method: 'POST' });
            const data = await res.json();
            alert(data.message);
        }

        async function runBacktest() {
            const modal = document.getElementById('modal-backtest');
            const content = document.getElementById('modal-content');
            modal.style.display = 'flex';
            content.innerText = 'Running backtest simulation...';

            const symbol = document.getElementById('select-symbol').value;
            const strategy = document.getElementById('select-strategy').value;

            const res = await fetch(`/api/control/backtest?symbol=${encodeURIComponent(symbol)}&strategy=${encodeURIComponent(strategy)}`, { method: 'POST' });
            const result = await res.json();

            content.innerHTML = `
                <table style="margin-top:12px;">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Strategy</td><td><b>${result.strategy}</b></td></tr>
                    <tr><td>Total Trades</td><td>${result.total_trades}</td></tr>
                    <tr><td>Win Rate</td><td><b>${result.win_rate}%</b></td></tr>
                    <tr><td>Total P&L</td><td class="${result.total_pnl >= 0 ? 'positive' : 'negative'}"><b>₹${result.total_pnl.toFixed(2)}</b></td></tr>
                    <tr><td>Sharpe Ratio</td><td>${result.sharpe_ratio}</td></tr>
                    <tr><td>Max Drawdown</td><td class="negative">${result.max_drawdown_pct}%</td></tr>
                </table>
            `;
        }

        function closeModal() {
            document.getElementById('modal-backtest').style.display = 'none';
        }

        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>
"""


class StartBotRequest(BaseModel):
    mode: str = "mock"
    symbol: str = "RELIANCE"
    strategy: str = "multi_agent"
    interval: int = 2


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return HTMLResponse(content=HTML_TEMPLATE)


@app.get("/api/status")
@app.get("/api/state")
async def get_state():
    return bot_state

@app.get("/api/positions")
async def get_positions():
    return bot_state.get("positions", [])

@app.get("/api/trades")
async def get_trades():
    return bot_state.get("trades", [])


@app.post("/api/control/start")
async def start_bot_control(req: StartBotRequest, background_tasks: BackgroundTasks):
    global bot_task
    if bot_state["status"] == "RUNNING":
        return {"status": "already_running"}

    bot_state["mode"] = req.mode
    bot_state["symbol"] = req.symbol
    bot_state["strategy"] = req.strategy
    bot_state["interval"] = req.interval
    bot_state["status"] = "RUNNING"
    bot_loop_stop_event.clear()

    log_to_dashboard(
        f"Bot started via Dashboard in {req.mode.upper()} mode for {req.symbol} ({req.strategy})"
    )

    background_tasks.add_task(run_bot_loop, req.symbol, req.interval, req.mode, req.strategy)
    return {"status": "started"}


@app.post("/api/control/stop")
async def stop_bot_control():
    bot_state["status"] = "STOPPED"
    bot_loop_stop_event.set()
    log_to_dashboard("Bot stopped via Dashboard.")
    return {"status": "stopped"}


@app.post("/api/circuit-breaker/reset")
async def reset_circuit_breaker():
    bot_state["circuit_breaker_active"] = False
    if bot_state["status"] == "CIRCUIT_BROKEN":
        bot_state["status"] = "STOPPED"
    log_to_dashboard("Circuit breaker reset via Dashboard.", level="SUCCESS")
    return {"status": "reset"}


@app.post("/api/control/download-data")
async def trigger_download_data(symbol: str = "RELIANCE"):
    from scripts.download_data import generate_random_walk_candles

    df = generate_random_walk_candles(symbol, datetime.now(), 30)
    save_path = Path("data/historical") / f"{symbol.replace(' ', '_')}_5m.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    log_to_dashboard(f"Generated 30 days of 5m data for {symbol}", level="SUCCESS")
    return {"status": "success", "message": f"Generated data for {symbol}"}


@app.post("/api/control/backtest")
async def trigger_backtest(symbol: str = "RELIANCE", strategy: str = "momentum"):
    from src.backtest.engine import BacktestEngine
    from src.data.feeds.csv_feed import CSVDataFeed
    from src.execution.fees import FeeCalculator
    from src.strategy.mean_reversion import MeanReversionStrategy
    from src.strategy.momentum import MomentumStrategy

    feed_path = Path("data/historical") / f"{symbol.replace(' ', '_')}_5m.csv"
    if not feed_path.exists():
        from scripts.download_data import generate_random_walk_candles

        df_gen = generate_random_walk_candles(symbol, datetime.now(), 30)
        feed_path.parent.mkdir(parents=True, exist_ok=True)
        df_gen.to_csv(feed_path, index=False)

    feed = CSVDataFeed(feed_path)
    candles = await feed.get_historical_candles(symbol)
    df = feed.to_dataframe(candles) if hasattr(feed, "to_dataframe") else None
    if df is None or df.empty:
        import pandas as pd

        df = pd.DataFrame([c.model_dump() for c in candles])

    strat_obj = (
        MeanReversionStrategy() if strategy == "mean_reversion" else MomentumStrategy()
    )
    engine = BacktestEngine(
        strategy=strat_obj,
        initial_capital=settings.mock.initial_capital,
        fee_calculator=FeeCalculator(),
        settings=settings,
    )
    res = await engine.run(df, symbol)

    summary = res.metrics.summary() if hasattr(res.metrics, "summary") else res.metrics
    summary["strategy"] = strategy

    log_to_dashboard(
        f"Backtest completed for {symbol} ({strategy}): Win rate {summary.get('win_rate', 0):.1f}%, P&L ₹{summary.get('total_pnl', 0):.2f}",
        level="SUCCESS",
    )
    return summary


async def run_bot_loop(symbol: str, interval: int, mode: str, strategy: str):
    """Background async bot execution loop triggered by dashboard."""
    from src.agents.orchestrator import AgentOrchestrator
    from src.data.feeds.csv_feed import CSVDataFeed
    from src.data.indicators import IndicatorEngine
    from src.execution.mock_engine import MockBroker
    from src.execution.order_validator import OrderValidator
    from src.execution.position_manager import PositionManager
    from src.risk.circuit_breaker import CircuitBreaker

    feed_path = Path("data/historical") / f"{symbol.replace(' ', '_')}_5m.csv"
    if not feed_path.exists():
        from scripts.download_data import generate_random_walk_candles

        df_gen = generate_random_walk_candles(symbol, datetime.now(), 30)
        feed_path.parent.mkdir(parents=True, exist_ok=True)
        df_gen.to_csv(feed_path, index=False)

    data_feed = CSVDataFeed(feed_path)
    broker = MockBroker(initial_capital=settings.mock.initial_capital)
    orchestrator = AgentOrchestrator()
    position_manager = PositionManager()
    order_validator = OrderValidator()
    circuit_breaker = CircuitBreaker()

    if mode == "mock":
        order_validator.set_backtest_mode(True)

    candles = await data_feed.get_historical_candles(symbol)
    window_size = 50

    log_to_dashboard(f"Loaded {len(candles)} candles for {symbol}. Starting execution loop...")

    for idx in range(window_size, len(candles)):
        if bot_loop_stop_event.is_set() or bot_state["status"] != "RUNNING":
            log_to_dashboard("Bot loop stopped.")
            break

        current_candle = candles[idx]
        history_candles = candles[idx - window_size : idx + 1]

        # 1. Update Portfolio & Check Circuit Breaker
        portfolio = await broker.get_portfolio()
        is_broken, reason = circuit_breaker.check(portfolio)
        if is_broken:
            bot_state["status"] = "CIRCUIT_BROKEN"
            bot_state["circuit_breaker_active"] = True
            log_to_dashboard(f"CIRCUIT BREAKER TRIGGERED: {reason}", level="ERROR")
            break

        # 2. Update Dashboard State
        bot_state["equity"] = portfolio.equity
        bot_state["cash"] = portfolio.current_capital
        bot_state["realized_pnl"] = portfolio.total_realized_pnl
        bot_state["unrealized_pnl"] = portfolio.total_unrealized_pnl
        bot_state["max_drawdown_pct"] = portfolio.max_drawdown_pct
        bot_state["positions"] = [p.model_dump() for p in portfolio.positions]
        bot_state["trades"] = [t.model_dump() for t in portfolio.completed_trades]

        # 3. Process mock broker bar
        await broker.process_candle(current_candle)

        # 4. Calculate indicators
        import pandas as pd

        df_history = pd.DataFrame([c.model_dump() for c in history_candles])
        df_with_ind = IndicatorEngine.calculate(df_history)
        latest_indicators = IndicatorEngine.get_latest_indicators(df_with_ind)

        # 5. Build Context & Get Decision
        context = MarketContext(
            symbol=symbol,
            exchange=Exchange.NSE,
            current_price=current_candle.close,
            candles=history_candles,
            indicators=latest_indicators,
            portfolio=portfolio,
            timestamp=current_candle.timestamp,
        )

        decision = await orchestrator.run(context)

        # Stream multi-agent opinions & full reasoning directly to UI log stream
        if hasattr(decision, "agent_opinions") and decision.agent_opinions:
            for opinion in decision.agent_opinions:
                role_name = opinion.agent_role.value if hasattr(opinion.agent_role, "value") else str(opinion.agent_role)
                act_name = opinion.action.value if hasattr(opinion.action, "value") else str(opinion.action)
                log_to_dashboard(
                    f"🧠 [{role_name.upper()}] Recommendation: {act_name} (Conf: {opinion.confidence:.2f})\n   Reasoning: {opinion.reasoning}",
                    level="INFO",
                )

        act_val = decision.action.value if hasattr(decision.action, "value") else str(decision.action)
        log_to_dashboard(
            f"🎯 PORTFOLIO DECISION: {act_val} | Target: {symbol} @ ₹{current_candle.close:.2f} | Approved: {decision.approved_by_risk} | Rationale: {decision.reasoning}",
            level="SUCCESS" if decision.approved_by_risk and decision.action in (SignalAction.BUY, SignalAction.SELL) else "INFO",
        )

        # 6. Execute Order if Signal Approved
        if decision.approved_by_risk and decision.action in (SignalAction.BUY, SignalAction.SELL):
            side = Side.BUY if decision.action == SignalAction.BUY else Side.SELL
            qty = decision.position_size or 10

            order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=qty,
                price=current_candle.close,
            )

            is_valid, reject_reason = order_validator.validate(order, portfolio)
            if is_valid:
                filled_order = await broker.place_order(order)
                if filled_order.is_terminal and filled_order.filled_quantity > 0:
                    await position_manager.open_position(filled_order)
                    log_to_dashboard(
                        f"⚡ Order Filled: {side.value} {qty} {symbol} @ ₹{filled_order.average_price:.2f}",
                        level="SUCCESS",
                    )
            else:
                log_to_dashboard(f"⚠️ Order Rejected by Risk Validator: {reject_reason}", level="WARNING")

        await asyncio.sleep(interval)


def start_server(host: str = "0.0.0.0", port: int = 8080):
    """Start the dashboard server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
