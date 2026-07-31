"""Technical indicator calculator engine using the `ta` library.

Computes a comprehensive set of indicators — moving averages, momentum,
volatility, volume, and trend — and maps them to the IndicatorValues model.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import (
    ADXIndicator,
    CCIIndicator,
    EMAIndicator,
    MACD,
    SMAIndicator,
)
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

from src.core.models import IndicatorValues


class IndicatorEngine:
    """Technical indicator calculator using the `ta` library.

    All indicators are computed on historical data only — the caller is
    responsible for ensuring no future data is passed in.
    """

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate and append all indicator columns to the DataFrame.

        Expects columns: open, high, low, close, volume.
        Optionally: timestamp (used as index for VWAP).

        Args:
            df: OHLCV DataFrame.

        Returns:
            Copy of the DataFrame with indicator columns appended.
        """
        if df.empty or len(df) < 5:
            return df

        try:
            res = df.copy()

            close = res["close"]
            high = res["high"]
            low = res["low"]
            volume = res["volume"].astype(float)

            # ── Moving Averages ──────────────────────────────────────────
            res["SMA_20"] = SMAIndicator(close, window=20).sma_indicator()
            res["SMA_50"] = SMAIndicator(close, window=50).sma_indicator()
            res["SMA_200"] = SMAIndicator(close, window=200).sma_indicator()
            res["EMA_9"] = EMAIndicator(close, window=9).ema_indicator()
            res["EMA_21"] = EMAIndicator(close, window=21).ema_indicator()

            # ── Momentum ─────────────────────────────────────────────────
            res["RSI_14"] = RSIIndicator(close, window=14).rsi()

            macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
            res["MACD"] = macd.macd()
            res["MACD_SIGNAL"] = macd.macd_signal()
            res["MACD_HIST"] = macd.macd_diff()

            res["ADX_14"] = ADXIndicator(high, low, close, window=14).adx()
            res["CCI_20"] = CCIIndicator(high, low, close, window=20).cci()

            # ── Volatility ───────────────────────────────────────────────
            bb = BollingerBands(close, window=20, window_dev=2)
            res["BB_UPPER"] = bb.bollinger_hband()
            res["BB_MIDDLE"] = bb.bollinger_mavg()
            res["BB_LOWER"] = bb.bollinger_lband()

            res["ATR_14"] = AverageTrueRange(high, low, close, window=14).average_true_range()

            # ── Volume ───────────────────────────────────────────────────
            res["OBV"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            res["VOL_SMA_20"] = SMAIndicator(volume, window=20).sma_indicator()

            # VWAP (requires high, low, close, volume)
            try:
                vwap = VolumeWeightedAveragePrice(high, low, close, volume)
                res["VWAP"] = vwap.volume_weighted_average_price()
            except Exception:
                # VWAP may fail if data is insufficient or not intraday
                res["VWAP"] = float("nan")

            # ── SuperTrend (custom) ──────────────────────────────────────
            res = IndicatorEngine._supertrend(res, period=10, multiplier=3)

            return res

        except Exception as e:
            logger.debug(f"Indicator calculation warm-up ({len(df)} candles): {e}")
            return df

    @staticmethod
    def _supertrend(
        df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
    ) -> pd.DataFrame:
        """Calculate SuperTrend indicator.

        The `ta` library doesn't include SuperTrend natively, so we
        compute it using ATR and basic trend logic.
        """
        hl2 = (df["high"] + df["low"]) / 2
        atr = AverageTrueRange(
            df["high"], df["low"], df["close"], window=period
        ).average_true_range()

        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        supertrend = pd.Series(0.0, index=df.index)
        direction = pd.Series(1, index=df.index)  # 1 = bullish, -1 = bearish

        for i in range(1, len(df)):
            if pd.isna(atr.iloc[i]):
                supertrend.iloc[i] = float("nan")
                direction.iloc[i] = 1
                continue

            # Adjust bands based on previous values
            if lower_band.iloc[i] > lower_band.iloc[i - 1] or df["close"].iloc[i - 1] < lower_band.iloc[i - 1]:
                pass  # keep current lower_band
            else:
                lower_band.iloc[i] = lower_band.iloc[i - 1]

            if upper_band.iloc[i] < upper_band.iloc[i - 1] or df["close"].iloc[i - 1] > upper_band.iloc[i - 1]:
                pass  # keep current upper_band
            else:
                upper_band.iloc[i] = upper_band.iloc[i - 1]

            # Determine direction
            if supertrend.iloc[i - 1] == upper_band.iloc[i - 1]:
                if df["close"].iloc[i] > upper_band.iloc[i]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
            else:
                if df["close"].iloc[i] < lower_band.iloc[i]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1

        df["SUPERTREND"] = supertrend
        df["SUPERTREND_DIR"] = direction
        return df

    @staticmethod
    def get_latest_indicators(df: pd.DataFrame) -> Optional[IndicatorValues]:
        """Extract the most recent indicator values as an IndicatorValues model.

        Gracefully maps NaN → None for indicators lacking enough history.

        Args:
            df: DataFrame with indicator columns (from `calculate()`).

        Returns:
            IndicatorValues or None if extraction fails.
        """
        if df.empty:
            return None

        latest = df.iloc[-1]

        def safe_get(key: str) -> Optional[float]:
            val = latest.get(key)
            if val is None or pd.isna(val):
                return None
            return float(val)

        try:
            return IndicatorValues(
                sma_20=safe_get("SMA_20"),
                sma_50=safe_get("SMA_50"),
                sma_200=safe_get("SMA_200"),
                ema_9=safe_get("EMA_9"),
                ema_21=safe_get("EMA_21"),
                rsi_14=safe_get("RSI_14"),
                macd=safe_get("MACD"),
                macd_signal=safe_get("MACD_SIGNAL"),
                macd_histogram=safe_get("MACD_HIST"),
                bb_upper=safe_get("BB_UPPER"),
                bb_middle=safe_get("BB_MIDDLE"),
                bb_lower=safe_get("BB_LOWER"),
                atr_14=safe_get("ATR_14"),
                adx=safe_get("ADX_14"),
                cci_20=safe_get("CCI_20"),
                vwap=safe_get("VWAP"),
                obv=safe_get("OBV"),
                supertrend=safe_get("SUPERTREND"),
                supertrend_direction=(
                    int(latest["SUPERTREND_DIR"])
                    if "SUPERTREND_DIR" in latest and not pd.isna(latest.get("SUPERTREND_DIR"))
                    else None
                ),
                volume_sma_20=safe_get("VOL_SMA_20"),
            )
        except Exception as e:
            logger.error(f"Error extracting latest indicators: {e}")
            return None
