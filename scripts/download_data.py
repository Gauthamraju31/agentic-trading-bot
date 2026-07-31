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

def generate_random_walk_candles(symbol: str, start_date: datetime, days: int) -> pd.DataFrame:
    """Generate 5-minute candles using a random walk for testing."""
    dates = pd.date_range(start=start_date, periods=days * 75, freq="5min") # roughly 75 5-min candles per trading day
    
    # Filter to market hours (9:15 to 15:30)
    market_hours = []
    for d in dates:
        # Ignore weekends
        if d.weekday() >= 5:
            continue
        t = d.time()
        if (t.hour == 9 and t.minute >= 15) or (9 < t.hour < 15) or (t.hour == 15 and t.minute <= 30):
            market_hours.append(d)
            
    dates = pd.DatetimeIndex(market_hours)
    
    n_periods = len(dates)
    
    # Starting price based on symbol roughly
    start_prices = {
        "RELIANCE": 2500,
        "TCS": 3500,
        "INFY": 1500,
        "HDFCBANK": 1600,
        "NIFTY 50": 21000
    }
    
    initial_price = start_prices.get(symbol, 1000)
    
    # Generate random walk
    np.random.seed(hash(symbol) % 2**32)
    returns = np.random.normal(loc=0.0001, scale=0.002, size=n_periods)
    price_path = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
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
    
    # Ensure high is max and low is min
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Download/Generate historical market data.")
    parser.add_argument("--days", type=int, default=365, help="Number of days of data to generate")
    parser.add_argument("--output-dir", type=str, default="data/historical", help="Output directory")
    args = parser.parse_args()

    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "NIFTY 50"]
    start_date = datetime.now() - timedelta(days=args.days)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for symbol in symbols:
        logger.info(f"Generating {args.days} days of data for {symbol}...")
        df = generate_random_walk_candles(symbol, start_date, args.days)
        file_path = os.path.join(args.output_dir, f"{symbol.replace(' ', '_')}_5m.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"Saved to {file_path}")
        
    logger.info("Data generation complete!")

if __name__ == "__main__":
    main()
