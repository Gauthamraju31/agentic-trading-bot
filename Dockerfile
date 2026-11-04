# Multi-Stage / Optimized Python 3.12 Dockerfile for Autonomous Trading Bot
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOME=/home/trader \
    USER=trader

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

# Create non-root trader user matching host UID/GID (default 1001:1001 for ubuntu)
ARG USER_ID=1001
ARG GROUP_ID=1001
RUN (getent group ${GROUP_ID} || groupadd -g ${GROUP_ID} trader) \
    && (id -u ${USER_ID} >/dev/null 2>&1 || useradd -m -u ${USER_ID} -g ${GROUP_ID} trader) \
    && mkdir -p /home/trader/.gemini \
    && chown -R ${USER_ID}:${GROUP_ID} /app /home/trader

USER ${USER_ID}

# Default to running the trading bot daemon
CMD ["python", "scripts/run_daily_bot.py"]
