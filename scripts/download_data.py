import argparse
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
import numpy as np
import pandas as pd
import yfinance as yf

from src.core.config import settings

NSE_TICKER_MAP = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
}


def download_real_candles(symbol: str, interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    """Download real historical candles for Indian equities via yfinance."""
    ticker_str = NSE_TICKER_MAP.get(symbol, f"{symbol}.NS" if not symbol.endswith(".NS") and not symbol.startswith("^") else symbol)
    logger.info(f"Fetching real {interval} market data for {symbol} ({ticker_str}) for period={period}...")

    try:
        ticker = yf.Ticker(ticker_str)
        df = ticker.history(period=period, interval=interval)

        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient data returned for {ticker_str} ({len(df)} bars)")
            return pd.DataFrame()

        df.reset_index(inplace=True)
        time_col = "Datetime" if "Datetime" in df.columns else "Date"
        if time_col not in df.columns:
            time_col = df.columns[0]

        df = df.rename(columns={
            time_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.dropna().sort_values("timestamp").reset_index(drop=True)
        logger.success(f"Successfully downloaded {len(df)} real bars for {symbol} ({df['timestamp'].min()} to {df['timestamp'].max()})")
        return df

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def generate_random_walk_candles(symbol: str, start_date: datetime, days: int) -> pd.DataFrame:
    """Fallback generator: synthetic 5-minute candles using a random walk."""
    dates = pd.date_range(start=start_date, periods=days * 75, freq="5min")
    market_hours = [
        d for d in dates
        if d.weekday() < 5 and ((d.hour == 9 and d.minute >= 15) or (9 < d.hour < 15) or (d.hour == 15 and d.minute <= 30))
    ]
    dates = pd.DatetimeIndex(market_hours)
    n_periods = len(dates)

    start_prices = {"RELIANCE": 2500, "TCS": 3500, "INFY": 1500, "HDFCBANK": 1600, "NIFTY 50": 21000}
    initial_price = start_prices.get(symbol, 1000)

    np.random.seed(hash(symbol) % 2**32)
    returns = np.random.normal(loc=0.0001, scale=0.002, size=n_periods)
    price_path = initial_price * np.exp(np.cumsum(returns))

    high_noise = np.random.uniform(1.0, 1.002, n_periods)
    low_noise = np.random.uniform(0.998, 1.0, n_periods)
    open_noise = np.random.uniform(0.999, 1.001, n_periods)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": price_path * open_noise,
        "high": price_path * high_noise,
        "low": price_path * low_noise,
        "close": price_path,
        "volume": np.random.randint(1000, 100000, n_periods)
    })
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    return df


def main():
    parser = argparse.ArgumentParser(description="Download real historical market data from NSE via yfinance.")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to download (default: all configured in settings)")
    parser.add_argument("--interval", type=str, default="5m", choices=["1m", "5m", "15m", "1d"], help="Candle timeframe")
    parser.add_argument("--period", type=str, default="60d", help="Lookback period (e.g. 60d, 1mo, 1y)")
    parser.add_argument("--output-dir", type=str, default="data/historical", help="Output directory")
    args = parser.parse_args()

    symbols = args.symbols or settings.market.symbols
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        file_path = output_dir / f"{symbol.replace(' ', '_')}_{args.interval}.csv"
        df = download_real_candles(symbol, interval=args.interval, period=args.period)

        if df.empty:
            logger.warning(f"Falling back to synthetic data generation for {symbol}...")
            start_date = datetime.now() - timedelta(days=60)
            df = generate_random_walk_candles(symbol, start_date, 60)

        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} candles to {file_path}")

    logger.success("Historical market data download complete!")


if __name__ == "__main__":
    main()
