"""FastAPI Web Dashboard & Interactive Control Center for the Trading Bot."""

import asyncio
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel
import uvicorn

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.config import settings
from src.core.daily_goal_manager import DailyGoalManager
from src.core.market_hours import get_ist_now, get_market_status
from src.agents.pre_market_planner import PreMarketPlanner

app = FastAPI(title="Agentic Trading Bot — Control Center")


def get_service_status() -> tuple[bool, str]:
    """Check actual systemd status of tradingbot.service."""
    try:
        res = subprocess.run(
            ["systemctl", "is-active", "tradingbot.service"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        is_active = res.stdout.strip() == "active"
        return is_active, "RUNNING" if is_active else "STOPPED"
    except Exception:
        return False, "STOPPED"


def get_daemon_logs(limit: int = 50) -> list[dict]:
    """Extract real-time execution logs from journalctl."""
    try:
        res = subprocess.run(
            ["journalctl", "-u", "tradingbot.service", "-n", str(limit), "--no-pager"],
            capture_output=True,
            text=True,
            timeout=4,
        )
        lines = res.stdout.strip().split("\n")
        parsed = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split("openclaw-server")
            if len(parts) > 1:
                ts = parts[0].strip()
                msg = parts[1].split("]:", 1)[-1].strip() if "]:" in parts[1] else parts[1].strip()
            else:
                ts = line[:15]
                msg = line[15:].strip()

            lvl = "INFO"
            if "ERROR" in msg or "Exception" in msg:
                lvl = "ERROR"
            elif "WARNING" in msg:
                lvl = "WARNING"
            elif "ORDER FILLED" in msg or "SUCCESS" in msg or "completed successfully" in msg:
                lvl = "SUCCESS"
            elif "Signal:" in msg:
                lvl = "SIGNAL"

            parsed.append({"timestamp": ts, "message": msg, "level": lvl})
        return parsed
    except Exception as e:
        return [{"timestamp": "now", "message": f"Log fetch error: {e}", "level": "WARNING"}]


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agentic Trading Bot — Control Center</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-page: #0a0b10;
            --bg-card: #12141c;
            --bg-card-subtle: #181b26;
            --border-card: #232738;
            --border-hover: #333a52;
            --text-primary: #ffffff;
            --text-secondary: #9aa3b8;
            --text-muted: #5e667d;
            --emerald: #10b981;
            --emerald-bg: rgba(16, 185, 129, 0.12);
            --rose: #f43f5e;
            --rose-bg: rgba(244, 63, 94, 0.12);
            --amber: #f59e0b;
            --amber-bg: rgba(245, 158, 11, 0.12);
            --cyan: #06b6d4;
            --cyan-bg: rgba(6, 182, 212, 0.12);
            --accent-blue: #3b82f6;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-page);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 24px 32px;
            line-height: 1.5;
        }

        .container {
            max-width: 1440px;
            margin: 0 auto;
        }

        /* Top Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border-card);
            margin-bottom: 28px;
            gap: 16px;
            flex-wrap: wrap;
        }

        .logo-area {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-icon {
            width: 42px;
            height: 42px;
            border-radius: 10px;
            background: linear-gradient(135deg, #2563eb, #06b6d4);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        }

        .brand-title h1 {
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.3px;
        }

        .brand-title p {
            font-size: 13px;
            color: var(--text-secondary);
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }

        .status-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            border: 1px solid transparent;
        }

        .pill-running {
            background: var(--emerald-bg);
            color: var(--emerald);
            border-color: rgba(16, 185, 129, 0.3);
        }

        .pill-stopped {
            background: var(--rose-bg);
            color: var(--rose);
            border-color: rgba(244, 63, 94, 0.3);
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
            box-shadow: 0 0 10px currentColor;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.85); }
        }

        .market-pill {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            color: var(--text-secondary);
            font-size: 12px;
            padding: 6px 12px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Buttons */
        .btn {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn:hover {
            border-color: var(--border-hover);
            background: var(--bg-card-subtle);
        }

        .btn-primary {
            background: #2563eb;
            border-color: #2563eb;
            color: white;
        }
        .btn-primary:hover {
            background: #1d4ed8;
            border-color: #1d4ed8;
        }

        .btn-danger {
            background: transparent;
            border-color: rgba(244, 63, 94, 0.4);
            color: var(--rose);
        }
        .btn-danger:hover {
            background: var(--rose-bg);
            border-color: var(--rose);
        }

        /* Metrics Row */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 18px;
            margin-bottom: 28px;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            padding: 20px;
            transition: border-color 0.2s ease;
        }

        .card:hover {
            border-color: var(--border-hover);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .card-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: var(--text-muted);
            font-weight: 600;
        }

        .card-value {
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.5px;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 6px;
        }

        .card-sub {
            font-size: 12px;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .progress-bar-bg {
            height: 6px;
            background: var(--border-card);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 8px;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--emerald), var(--cyan));
            border-radius: 3px;
            transition: width 0.4s ease;
        }

        /* Main Workspace Split */
        .workspace-grid {
            display: grid;
            grid-template-columns: 1.8fr 1.2fr;
            gap: 22px;
        }

        @media (max-width: 1024px) {
            .workspace-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Terminal Window */
        .terminal-box {
            background: #07080c;
            border: 1px solid var(--border-card);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            height: 520px;
            overflow: hidden;
        }

        .terminal-topbar {
            background: #0f1118;
            border-bottom: 1px solid var(--border-card);
            padding: 10px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 600;
        }

        .terminal-controls {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .terminal-body {
            padding: 14px;
            overflow-y: auto;
            flex: 1;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            line-height: 1.6;
        }

        .log-row {
            display: flex;
            gap: 12px;
            padding: 2px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.02);
        }

        .log-ts {
            color: var(--text-muted);
            flex-shrink: 0;
            font-size: 11px;
        }

        .log-msg {
            color: #d1d5db;
            word-break: break-all;
        }

        .log-SUCCESS { color: #34d399; }
        .log-SIGNAL { color: #38bdf8; font-weight: 600; }
        .log-WARNING { color: #fbbf24; }
        .log-ERROR { color: #f87171; font-weight: 600; }

        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        th {
            text-align: left;
            font-size: 11px;
            text-transform: uppercase;
            color: var(--text-muted);
            padding: 8px 12px;
            border-bottom: 1px solid var(--border-card);
        }

        td {
            padding: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-family: 'JetBrains Mono', monospace;
        }

        .empty-placeholder {
            text-align: center;
            padding: 36px 16px;
            color: var(--text-muted);
            font-size: 13px;
        }

        /* Config Form */
        .form-row {
            display: flex;
            gap: 12px;
            margin-top: 14px;
        }

        .form-input-group {
            flex: 1;
        }

        .form-input-group label {
            display: block;
            font-size: 11px;
            color: var(--text-muted);
            margin-bottom: 6px;
            text-transform: uppercase;
            font-weight: 600;
        }

        .form-input {
            width: 100%;
            background: #08090d;
            border: 1px solid var(--border-card);
            color: white;
            padding: 9px 12px;
            border-radius: 8px;
            font-size: 14px;
            font-family: 'JetBrains Mono', monospace;
            outline: none;
        }

        .form-input:focus {
            border-color: var(--accent-blue);
        }

        .toast {
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: #1e293b;
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            border: 1px solid var(--border-hover);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
            font-size: 13px;
            opacity: 0;
            transform: translateY(10px);
            transition: all 0.3s ease;
            pointer-events: none;
        }

        .toast.show {
            opacity: 1;
            transform: translateY(0);
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- HEADER -->
        <header>
            <div class="logo-area">
                <div class="brand-icon">⚡</div>
                <div class="brand-title">
                    <h1>Autonomous Indian Trading Bot</h1>
                    <p>NSE Intraday Engine • Real-Time Market Feeds • Mock Broker</p>
                </div>
            </div>
            <div class="header-actions">
                <div id="daemon-status-pill" class="status-pill pill-stopped">
                    <span class="pulse-dot"></span>
                    <span id="daemon-status-text">DAEMON STOPPED</span>
                </div>
                <div class="market-pill">
                    <span id="market-status-dot" style="color: var(--emerald);">●</span>
                    <span id="market-status-text">NSE OPEN</span>
                </div>
                <div class="market-pill">
                    <span id="ist-clock">--:--:-- IST</span>
                </div>
                <button id="btn-toggle-bot" class="btn btn-primary" onclick="toggleBot()">▶ Resume Bot</button>
                <button class="btn" onclick="restartBot()">🔄 Restart</button>
                <button class="btn btn-danger" onclick="triggerKillSwitch()">⚡ Kill Switch</button>
            </div>
        </header>

        <!-- TOP METRICS GRID -->
        <div class="metrics-grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Portfolio Equity</span>
                    <span style="font-size: 11px; color: var(--emerald); font-weight: 600;">ACTIVE MOCK</span>
                </div>
                <div id="metric-equity" class="card-value">₹100,000.00</div>
                <div class="card-sub">Starting Capital: ₹100,000.00</div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-title">Today's Realized P&L</span>
                    <span id="metric-trades-badge" style="font-size: 11px; color: var(--text-muted); font-weight: 600;">0 Trades</span>
                </div>
                <div id="metric-pnl" class="card-value" style="color: var(--text-primary);">₹+0.00</div>
                <div class="card-sub" id="metric-win-ratio">Win Rate: 0.0% • 0W / 0L</div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-title">Daily Goal Progress</span>
                    <span id="metric-progress-pct" style="font-size: 11px; color: var(--cyan); font-weight: 600;">0.0%</span>
                </div>
                <div id="metric-goal-txt" class="card-value" style="font-size: 20px;">₹0.00 / ₹1,500.00</div>
                <div class="progress-bar-bg">
                    <div id="metric-goal-bar" class="progress-bar-fill" style="width: 0%;"></div>
                </div>
                <div class="card-sub" style="margin-top: 6px;">Daily Stop-Loss Limit: <b id="lbl-max-loss" style="color: var(--rose);">-₹1,000.00</b></div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span class="card-title">Focus Target Symbol</span>
                    <span id="metric-signal-badge" style="font-size: 11px; padding: 2px 8px; border-radius: 12px; background: rgba(255,255,255,0.06); font-weight: 600;">HOLD</span>
                </div>
                <div id="metric-symbol" class="card-value" style="color: var(--cyan);">INFY</div>
                <div class="card-sub">LTP: <b id="metric-ltp" style="color: white;">₹1,121.80</b> • Conf: <span id="metric-conf">0.00</span></div>
            </div>
        </div>

        <!-- MAIN SPLIT WORKSPACE -->
        <div class="workspace-grid">
            <!-- LEFT: Positions & Terminal Logs -->
            <div style="display: flex; flex-direction: column; gap: 22px;">
                <!-- Positions Card -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Active Intraday Positions</span>
                        <span style="font-size: 11px; color: var(--text-muted);">Auto-Squareoff at 15:20 IST</span>
                    </div>
                    <div id="positions-container">
                        <div class="empty-placeholder">
                            No open positions right now. Bot is monitoring 5-minute candle structure for momentum breakouts.
                        </div>
                    </div>
                </div>

                <!-- Live Terminal Logs -->
                <div class="terminal-box">
                    <div class="terminal-topbar">
                        <span>LIVE AUTONOMOUS EXECUTION FEED</span>
                        <div class="terminal-controls">
                            <span style="color: var(--emerald);">● LIVE STREAMING</span>
                        </div>
                    </div>
                    <div id="terminal-body" class="terminal-body">
                        <!-- Streamed log rows will insert here -->
                    </div>
                </div>
            </div>

            <!-- RIGHT: Pre-Market AI Playbook & Settings -->
            <div style="display: flex; flex-direction: column; gap: 22px;">
                <!-- Pre-Market Playbook Card -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Pre-Market AI Strategy</span>
                        <button class="btn" style="padding: 4px 10px; font-size: 11px;" onclick="triggerPreMarket()">⚡ Run Scan</button>
                    </div>
                    <div id="playbook-box" style="font-size: 13px; color: var(--text-secondary); line-height: 1.6;">
                        <div style="margin-bottom: 10px; display: flex; justify-content: space-between;">
                            <span>Regime: <b id="pb-vix-regime" style="color: var(--amber);">MODERATE (VIX 15.7)</b></span>
                            <span>Multiplier: <b id="pb-vix-mult" style="color: var(--emerald);">0.80x</b></span>
                        </div>
                        <div style="background: var(--bg-card-subtle); padding: 12px; border-radius: 8px; border: 1px solid var(--border-card); margin-bottom: 12px;">
                            <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; margin-bottom: 4px;">AI Intraday Pick</div>
                            <div style="font-size: 16px; font-weight: 700; color: white;" id="pb-pick-title">INFY — HOLD</div>
                            <div id="pb-reasoning" style="font-size: 12px; color: var(--text-secondary); margin-top: 6px;">Pre-market analysis calibrated against sector momentum and technical structures.</div>
                        </div>
                    </div>
                </div>

                <!-- Daily Target Settings Form -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Risk & Target Configuration</span>
                    </div>
                    <div style="font-size: 12px; color: var(--text-secondary);">
                        Lock in profits and preserve capital automatically when daily limits are triggered.
                    </div>
                    <div class="form-row">
                        <div class="form-input-group">
                            <label>Daily Profit Target (₹)</label>
                            <input id="input-target-profit" class="form-input" type="number" value="1500" step="100">
                        </div>
                        <div class="form-input-group">
                            <label>Daily Max Loss Limit (₹)</label>
                            <input id="input-max-loss" class="form-input" type="number" value="1000" step="100">
                        </div>
                    </div>
                    <div style="margin-top: 14px;">
                        <button class="btn btn-primary" style="width: 100%; justify-content: center;" onclick="saveGoalConfig()">💾 Update Daily Targets</button>
                    </div>
                </div>

                <!-- System Info Card -->
                <div class="card" style="font-size: 12px; color: var(--text-muted); line-height: 1.8;">
                    <div style="font-weight: 600; color: var(--text-secondary); margin-bottom: 6px; text-transform: uppercase;">Engine Diagnostics</div>
                    <div>Host: <b style="color: white;">rick (68.233.98.15)</b></div>
                    <div>Process Service: <b style="color: white;">tradingbot.service</b></div>
                    <div>Execution Engine: <b style="color: white;">Momentum + 2x ATR Trailing Stop</b></div>
                    <div>Data Feed: <b style="color: white;">NSE Live Candles via yfinance</b></div>
                    <div>Fee Structure: <b style="color: white;">₹20 flat + STT + GST + Exchange (2026)</b></div>
                </div>
            </div>
        </div>
    </div>

    <div id="toast" class="toast">Action completed</div>

    <script>
        let isDaemonRunning = false;

        function showToast(msg) {
            const toast = document.getElementById('toast');
            toast.innerText = msg;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        function updateClock() {
            const now = new Date();
            const options = { timeZone: 'Asia/Kolkata', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' };
            const timeStr = now.toLocaleTimeString('en-GB', options);
            document.getElementById('ist-clock').innerText = timeStr + ' IST';
        }
        setInterval(updateClock, 1000);
        updateClock();

        async function fetchDashboardState() {
            try {
                const res = await fetch('/api/status');
                if (!res.ok) return;
                const data = await res.json();

                isDaemonRunning = data.daemon_active;

                // Daemon status pill
                const pill = document.getElementById('daemon-status-pill');
                const pillTxt = document.getElementById('daemon-status-text');
                const btnToggle = document.getElementById('btn-toggle-bot');

                if (data.daemon_active) {
                    pill.className = 'status-pill pill-running';
                    pillTxt.innerText = 'DAEMON ACTIVE';
                    btnToggle.innerText = '⏸ Pause Bot';
                    btnToggle.className = 'btn';
                } else {
                    pill.className = 'status-pill pill-stopped';
                    pillTxt.innerText = 'DAEMON STOPPED';
                    btnToggle.innerText = '▶ Start Bot';
                    btnToggle.className = 'btn btn-primary';
                }

                // Market Status
                const mktText = document.getElementById('market-status-text');
                const mktDot = document.getElementById('market-status-dot');
                if (data.market_status === 'MARKET_OPEN') {
                    mktText.innerText = 'NSE OPEN (09:15 - 15:30)';
                    mktDot.style.color = 'var(--emerald)';
                } else {
                    mktText.innerText = 'NSE CLOSED';
                    mktDot.style.color = 'var(--rose)';
                }

                // Equity & P&L
                document.getElementById('metric-equity').innerText = '₹' + Number(data.equity || 100000).toLocaleString('en-IN', { minimumFractionDigits: 2 });
                const pnlEl = document.getElementById('metric-pnl');
                const pnl = Number(data.realized_pnl || 0);
                pnlEl.innerText = (pnl >= 0 ? '₹+' : '₹') + pnl.toFixed(2);
                pnlEl.style.color = pnl > 0 ? 'var(--emerald)' : (pnl < 0 ? 'var(--rose)' : 'white');

                document.getElementById('metric-trades-badge').innerText = `${data.trade_count || 0} Trades`;
                document.getElementById('metric-win-ratio').innerText = `Win Rate: ${(data.win_rate || 0).toFixed(1)}% • ${data.winning_trades || 0}W / ${data.losing_trades || 0}L`;

                // Daily Goal Progress
                const target = Number(data.target_profit || 1500);
                const maxLoss = Number(data.max_loss || 1000);
                const pct = Math.max(0, Math.min(100, (pnl / target) * 100));
                document.getElementById('metric-progress-pct').innerText = pct.toFixed(1) + '%';
                document.getElementById('metric-goal-txt').innerText = `₹${pnl.toFixed(2)} / ₹${target.toFixed(2)}`;
                document.getElementById('metric-goal-bar').style.width = pct + '%';
                document.getElementById('lbl-max-loss').innerText = `-₹${maxLoss.toFixed(2)}`;

                // Active Symbol & LTP
                document.getElementById('metric-symbol').innerText = data.symbol || 'INFY';
                if (data.latest_price) {
                    document.getElementById('metric-ltp').innerText = '₹' + Number(data.latest_price).toFixed(2);
                }
                const sigBadge = document.getElementById('metric-signal-badge');
                sigBadge.innerText = data.latest_signal || 'HOLD';
                if (data.latest_signal === 'BUY') {
                    sigBadge.style.background = 'var(--emerald-bg)';
                    sigBadge.style.color = 'var(--emerald)';
                } else if (data.latest_signal === 'SELL') {
                    sigBadge.style.background = 'var(--rose-bg)';
                    sigBadge.style.color = 'var(--rose)';
                } else {
                    sigBadge.style.background = 'rgba(255,255,255,0.06)';
                    sigBadge.style.color = 'var(--text-secondary)';
                }
                document.getElementById('metric-conf').innerText = Number(data.latest_conf || 0).toFixed(2);

                // Open Positions
                const posContainer = document.getElementById('positions-container');
                if (data.positions && data.positions.length > 0) {
                    let html = '<table><thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>LTP</th><th>P&L</th></tr></thead><tbody>';
                    data.positions.forEach(p => {
                        const pnl = p.unrealized_pnl || 0;
                        const pnlColor = pnl >= 0 ? 'var(--emerald)' : 'var(--rose)';
                        html += `<tr>
                            <td><b>${p.symbol}</b></td>
                            <td style="color: ${p.side === 'BUY' ? 'var(--emerald)' : 'var(--rose)'}">${p.side}</td>
                            <td>${p.quantity}</td>
                            <td>₹${Number(p.entry_price).toFixed(2)}</td>
                            <td>₹${Number(p.current_price || p.entry_price).toFixed(2)}</td>
                            <td style="color: ${pnlColor}">₹${pnl.toFixed(2)}</td>
                        </tr>`;
                    });
                    html += '</tbody></table>';
                    posContainer.innerHTML = html;
                }

                // Render Logs
                if (data.logs && data.logs.length > 0) {
                    const term = document.getElementById('terminal-body');
                    const wasAtBottom = term.scrollHeight - term.clientHeight <= term.scrollTop + 30;
                    let logsHtml = '';
                    data.logs.forEach(l => {
                        logsHtml += `<div class="log-row">
                            <span class="log-ts">${l.timestamp}</span>
                            <span class="log-msg log-${l.level}">${escapeHtml(l.message)}</span>
                        </div>`;
                    });
                    term.innerHTML = logsHtml;
                    if (wasAtBottom) {
                        term.scrollTop = term.scrollHeight;
                    }
                }

            } catch (e) {
                console.error('State poll error:', e);
            }
        }

        async function fetchPlaybook() {
            try {
                const res = await fetch('/api/pre-market-playbook');
                if (!res.ok) return;
                const pb = await res.json();
                if (pb && pb.vix_data) {
                    document.getElementById('pb-vix-regime').innerText = `${pb.vix_data.regime} (VIX ${pb.vix_data.vix_value})`;
                    document.getElementById('pb-vix-mult').innerText = `${pb.vix_data.position_size_multiplier}x`;
                }
                if (pb && pb.primary_pick) {
                    document.getElementById('pb-pick-title').innerText = `${pb.primary_pick.symbol} — ${pb.primary_pick.action}`;
                    document.getElementById('pb-reasoning').innerText = pb.primary_pick.reasoning || pb.primary_pick.selector_rationale || '';
                }
            } catch (e) {}
        }

        function escapeHtml(text) {
            return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        async function toggleBot() {
            const endpoint = isDaemonRunning ? '/api/control/stop' : '/api/control/start';
            showToast(isDaemonRunning ? 'Stopping daemon...' : 'Starting daemon...');
            try {
                const res = await fetch(endpoint, { method: 'POST' });
                const d = await res.json();
                showToast(d.message || 'Updated');
                setTimeout(fetchDashboardState, 1000);
            } catch (e) {
                showToast('Failed to toggle bot');
            }
        }

        async function restartBot() {
            showToast('Restarting trading bot service...');
            try {
                const res = await fetch('/api/control/restart', { method: 'POST' });
                const d = await res.json();
                showToast(d.message || 'Restarted');
                setTimeout(fetchDashboardState, 1500);
            } catch (e) {
                showToast('Failed to restart bot');
            }
        }

        async function triggerKillSwitch() {
            if (!confirm('🚨 EMERGENCY: Are you sure you want to stop the bot and flatten all positions?')) return;
            showToast('Activating Emergency Kill Switch...');
            try {
                await fetch('/api/control/kill-switch', { method: 'POST' });
                showToast('Kill switch activated. Bot stopped.');
                setTimeout(fetchDashboardState, 1000);
            } catch (e) {
                showToast('Kill switch error');
            }
        }

        async function triggerPreMarket() {
            showToast('Running pre-market research...');
            try {
                const res = await fetch('/api/control/run-pre-market', { method: 'POST' });
                const d = await res.json();
                showToast(d.message || 'Started');
                setTimeout(fetchPlaybook, 3000);
            } catch (e) {
                showToast('Failed to trigger pre-market');
            }
        }

        async function saveGoalConfig() {
            const target = parseFloat(document.getElementById('input-target-profit').value);
            const loss = parseFloat(document.getElementById('input-max-loss').value);
            if (isNaN(target) || isNaN(loss)) {
                showToast('Invalid numbers');
                return;
            }
            try {
                const res = await fetch('/api/daily-goal/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target_profit: target, max_loss: loss })
                });
                if (res.ok) {
                    showToast('Daily targets updated successfully!');
                    fetchDashboardState();
                }
            } catch (e) {
                showToast('Failed to update targets');
            }
        }

        // Poll every 2.5 seconds
        setInterval(fetchDashboardState, 2500);
        fetchDashboardState();
        fetchPlaybook();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return HTMLResponse(content=HTML_TEMPLATE)


@app.get("/api/status")
@app.get("/api/state")
async def get_state():
    is_active, status_str = get_service_status()
    goal_summary = DailyGoalManager().get_summary()
    pb = PreMarketPlanner().get_latest_playbook() or {}

    primary_pick = pb.get("primary_pick", {}) if pb else {}
    target_symbol = primary_pick.get("symbol", "INFY")

    logs = get_daemon_logs(40)

    # Extract latest LTP from latest log if present
    latest_price = primary_pick.get("current_price", 0.0)
    latest_signal = primary_pick.get("action", "HOLD")
    latest_conf = primary_pick.get("confidence", 0.0)

    for item in reversed(logs):
        msg = item.get("message", "")
        if "Latest price: ₹" in msg:
            try:
                latest_price = float(msg.split("Latest price: ₹")[-1].split()[0])
            except Exception:
                pass
        if "Signal:" in msg:
            try:
                sig_part = msg.split("Signal:")[-1].strip()
                latest_signal = sig_part.split()[0]
                if "(Conf:" in sig_part:
                    latest_conf = float(sig_part.split("(Conf:")[-1].split(")")[0])
            except Exception:
                pass
        if latest_price and latest_signal != "HOLD":
            break

    mkt = get_market_status()

    return {
        "status": status_str,
        "daemon_active": is_active,
        "market_status": mkt.get("status", "UNKNOWN"),
        "market_reason": mkt.get("reason", ""),
        "mode": settings.app.mode,
        "symbol": target_symbol,
        "strategy": "momentum",
        "interval": 15,
        "circuit_breaker_active": False,
        "equity": settings.mock.initial_capital + goal_summary.get("realized_pnl", 0.0),
        "initial_capital": settings.mock.initial_capital,
        "cash": settings.mock.initial_capital + goal_summary.get("realized_pnl", 0.0),
        "realized_pnl": goal_summary.get("realized_pnl", 0.0),
        "target_profit": goal_summary.get("target_profit", 1500.0),
        "max_loss": goal_summary.get("max_loss", 1000.0),
        "progress_pct": goal_summary.get("progress_pct", 0.0),
        "trade_count": goal_summary.get("trade_count", 0),
        "winning_trades": goal_summary.get("winning_trades", 0),
        "losing_trades": goal_summary.get("losing_trades", 0),
        "win_rate": (goal_summary.get("winning_trades", 0) / goal_summary.get("trade_count", 1) * 100) if goal_summary.get("trade_count", 0) > 0 else 0.0,
        "positions": [],
        "trades": [],
        "logs": logs,
        "latest_price": latest_price,
        "latest_signal": latest_signal,
        "latest_conf": latest_conf,
    }


@app.get("/api/positions")
async def get_positions():
    return []


@app.get("/api/trades")
async def get_trades():
    return []


@app.get("/api/daemon/logs")
async def get_logs_endpoint():
    return get_daemon_logs(50)


@app.post("/api/control/start")
async def start_bot_control():
    try:
        subprocess.run(["sudo", "systemctl", "start", "tradingbot.service"], check=True, timeout=5)
        return {"status": "started", "message": "Autonomous trading bot service started successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/control/stop")
async def stop_bot_control():
    try:
        subprocess.run(["sudo", "systemctl", "stop", "tradingbot.service"], check=True, timeout=5)
        return {"status": "stopped", "message": "Autonomous trading bot service stopped."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/control/restart")
async def restart_bot_control():
    try:
        subprocess.run(["sudo", "systemctl", "restart", "tradingbot.service"], check=True, timeout=5)
        return {"status": "restarted", "message": "Autonomous trading bot service restarted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/control/kill-switch")
async def trigger_kill_switch():
    try:
        subprocess.run(["sudo", "systemctl", "stop", "tradingbot.service"], check=True, timeout=5)
        return {"status": "killed", "message": "Emergency Kill Switch activated: Service halted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/daily-goal")
async def get_daily_goal():
    return DailyGoalManager().get_summary()


class UpdateGoalRequest(BaseModel):
    target_profit: float = 1500.0
    max_loss: float = 1000.0


@app.post("/api/daily-goal/config")
async def update_daily_goal_config(req: UpdateGoalRequest):
    mgr = DailyGoalManager()
    mgr.set_config(req.target_profit, req.max_loss)
    return mgr.get_summary()


@app.get("/api/pre-market-playbook")
async def get_playbook():
    pb = PreMarketPlanner().get_latest_playbook()
    return pb or {"status": "NO_PLAYBOOK", "message": "Pre-market research has not run today."}


@app.post("/api/control/run-pre-market")
async def trigger_pre_market(background_tasks: BackgroundTasks):
    async def _run():
        planner = PreMarketPlanner()
        await planner.run_pre_market_analysis()

    background_tasks.add_task(_run)
    return {"status": "started", "message": "Pre-market research started in background"}


@app.post("/api/control/backtest")
async def trigger_backtest(symbol: str = "INFY", strategy: str = "momentum"):
    from src.backtest.engine import BacktestEngine
    from src.execution.fees import FeeCalculator
    from src.strategy.momentum import MomentumStrategy
    import pandas as pd

    feed_path = Path("data/historical") / f"{symbol.replace(' ', '_')}_5m.csv"
    if not feed_path.exists():
        raise HTTPException(status_code=404, detail="Historical data not found")

    df = pd.read_csv(feed_path, parse_dates=["timestamp"]).set_index("timestamp")
    engine = BacktestEngine(
        strategy=MomentumStrategy(),
        initial_capital=settings.backtest.initial_capital,
        fee_calculator=FeeCalculator(),
        settings=settings,
    )
    res = await engine.run(df, symbol)
    return {"status": "success", "metrics": res.metrics}


def start_server(host: str = "0.0.0.0", port: int = 8080):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
