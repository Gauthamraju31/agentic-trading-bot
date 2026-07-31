#!/bin/bash
# Autonomous Trading Bot — VM Deployment Script for rick.drunkcoder.dev

set -e

SERVER="ubuntu@rick.drunkcoder.dev"
REMOTE_DIR="/home/ubuntu/trading-bot"

echo "🚀 [Deploy] Pushing latest code to GitHub repository..."
git add .
git commit -m "feat: pre-market autonomous planner, daily profit goal manager, and systemd service" || true
git push origin main || true

echo "🌐 [Deploy] Connecting to $SERVER to pull code and restart services..."
ssh $SERVER bash -c "'
  set -e
  if [ ! -d \"$REMOTE_DIR\" ]; then
    echo \"Cloning repository on remote server...\"
    git clone https://github.com/Gauthamraju31/agentic-trading-bot.dir \"$REMOTE_DIR\" || git clone git@github.com:Gauthamraju31/agentic-trading-bot.git \"$REMOTE_DIR\"
  fi

  cd \"$REMOTE_DIR\"
  echo \"Pulling latest code...\"
  git pull origin main

  if [ ! -d \".venv\" ]; then
    echo \"Creating Python virtual environment...\"
    python3 -m venv .venv
  fi

  echo \"Installing requirements...\"
  .venv/bin/pip install -e .

  echo \"Installing systemd services & timers...\"
  sudo cp systemd/tradingbot.service /etc/systemd/system/
  sudo cp systemd/tradingbot.timer /etc/systemd/system/
  sudo cp systemd/tradingdashboard.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable tradingbot.timer tradingdashboard.service
  sudo systemctl start tradingbot.timer
  sudo systemctl restart tradingdashboard.service

  echo \"✅ Deployment complete!\"
  echo \"Dashboard running at: http://rick.drunkcoder.dev:8080\"
'"
