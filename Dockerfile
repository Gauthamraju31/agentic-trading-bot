# Multi-Stage / Optimized Python 3.12 Dockerfile for Autonomous Trading Bot
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for caching layers
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .[dev,alerts]

# Copy project source and configuration
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY data/ /app/data/
COPY tests/ /app/tests/

# Create non-root trader user and set permissions
RUN useradd -m -u 1000 trader \
    && chown -R trader:trader /app

USER trader

# Default to running the trading bot daemon
CMD ["python", "scripts/run_daily_bot.py"]
