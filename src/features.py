"""
features.py – Scale-free, strictly as-of-time feature engineering.

All features use only information available at or before bar t.  Rolling
windows that touch more than one trading day are reset per day for
intraday-only features (VWAP, day-range, day-open distance).  The
no-lookahead property is verified by tests/test_features_no_lookahead.py.

Feature groups
--------------
1. Returns           : log returns at lags 1, 2, 3, 6, 12
2. Volatility        : rolling std of 1-bar returns at 12 and 36 bars
3. RSI(14)
4. MACD histogram    : (12, 26, 9), normalised by close
5. EMA ratios        : close/EMA(9), EMA(9)/EMA(21)
6. VWAP deviation    : (close - intraday VWAP) / VWAP  [resets per day]
7. Volume ratio      : volume / rolling mean volume (78 bars)
8. Range             : (high - low) / close
9. Day-open distance : close / first_open_today - 1
10. Position in day range : (close - day_low) / (day_high - day_low)
11. Overnight gap    : today_open / prev_day_close - 1
12. Time features    : minutes_since_open, sin/cos, day-of-week
13. Market context   : Nifty lag returns, Nifty vol, relative strength
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    EMA_LONG,
    EMA_SHORT,
    HORIZON_BARS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MARKET_OPEN,
    RETURN_LAGS,
    RSI_PERIOD,
    TZ,
    USE_SENTIMENT_FEATURES,
    VOL_WINDOWS,
    VOLUME_MA_BARS,
)

logger = logging.getLogger(__name__)

MARKET_OPEN_MINUTES = int(MARKET_OPEN.split(":")[0]) * 60 + int(MARKET_OPEN.split(":")[1])
SESSION_TOTAL_MINUTES = (15 * 60 + 30) - MARKET_OPEN_MINUTES  # 09:15 to 15:30


# ── Individual feature functions ──────────────────────────────────────────────

def _log_return(close: pd.Series) -> pd.Series:
    """Single-bar log return."""
    return np.log(close / close.shift(1))


def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log return lags."""
    ret1 = _log_return(df["close"])
    for lag in RETURN_LAGS:
        df[f"ret_lag{lag}"] = ret1.shift(lag - 1) if lag > 1 else ret1
    return df


def _add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling std of 1-bar log returns."""
    ret1 = _log_return(df["close"])
    for w in VOL_WINDOWS:
        df[f"vol_{w}"] = ret1.rolling(w, min_periods=max(2, w // 4)).std()
    return df


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI, shifted so only uses data up to and including t."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi"] = _rsi(df["close"])
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD histogram normalised by close (scale-free)."""
    close = df["close"]
    ema_fast = _ema(close, MACD_FAST)
    ema_slow = _ema(close, MACD_SLOW)
    macd_line = ema_fast - ema_slow
    signal = _ema(macd_line, MACD_SIGNAL)
    hist = macd_line - signal
    # Normalise — avoid division by zero
    df["macd_hist_norm"] = hist / close.replace(0, np.nan)
    return df


def _add_ema_ratios(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    ema9 = _ema(close, EMA_SHORT)
    ema21 = _ema(close, EMA_LONG)
    df["ema_ratio_close_9"] = close / ema9.replace(0, np.nan)
    df["ema_ratio_9_21"] = ema9 / ema21.replace(0, np.nan)
    return df


def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intraday VWAP deviation (cumulative within each calendar day).

    VWAP = cumsum(typical_price * volume) / cumsum(volume), reset per day.
    Deviation = (close - vwap) / vwap.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]

    # Group by calendar date and compute cumulative sums
    date_key = df.index.date if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["ts"]).dt.date
    pv_cum = pv.groupby(date_key).cumsum()
    vol_cum = df["volume"].groupby(date_key).cumsum()

    vwap = pv_cum / vol_cum.replace(0, np.nan)
    df["vwap_dev"] = (df["close"] - vwap) / vwap.replace(0, np.nan)
    return df


def _add_volume_ratio(df: pd.DataFrame) -> pd.DataFrame:
    vol_ma = df["volume"].rolling(VOLUME_MA_BARS, min_periods=5).mean()
    df["vol_ratio"] = df["volume"] / vol_ma.replace(0, np.nan)
    return df


def _add_range(df: pd.DataFrame) -> pd.DataFrame:
    df["bar_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    return df


def _add_day_open_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    close / today's first open bar - 1.
    The first open of each day is the open of the first bar on that date.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
    else:
        dates = pd.to_datetime(df["ts"]).dt.date.values

    date_series = pd.Series(dates, index=df.index)
    first_open = df.groupby(date_series)["open"].transform("first")
    df["day_open_dist"] = df["close"] / first_open.replace(0, np.nan) - 1
    return df


def _add_day_range_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    (close - cumulative_day_low) / (cumulative_day_high - cumulative_day_low).
    Uses only past bars of the same day.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
    else:
        dates = pd.to_datetime(df["ts"]).dt.date.values

    date_series = pd.Series(dates, index=df.index)
    day_high = df.groupby(date_series)["high"].transform(lambda x: x.cummax())
    day_low = df.groupby(date_series)["low"].transform(lambda x: x.cummin())
    rng = (day_high - day_low).replace(0, np.nan)
    df["day_range_pos"] = (df["close"] - day_low) / rng
    return df


def _add_overnight_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    today_open / prev_day_close - 1.
    Same value for all bars of a given day; computed from open-of-first-bar.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
    else:
        dates = pd.to_datetime(df["ts"]).dt.date.values

    date_series = pd.Series(dates, index=df.index)
    # First bar open of each day
    first_open = df.groupby(date_series)["open"].transform("first")
    # Last close of previous day: shift the daily last close
    daily_last_close = df.groupby(date_series)["close"].transform("last")
    # Build a date-indexed series of daily last close
    daily_lc = df.groupby(date_series)["close"].last()
    daily_lc_shifted = daily_lc.shift(1)  # previous day's last close

    # Map back to per-bar using date index
    prev_close_map = daily_lc_shifted.to_dict()
    prev_close = date_series.map(prev_close_map)
    prev_close.index = df.index

    gap = first_open / prev_close.replace(0, np.nan) - 1
    df["overnight_gap"] = gap.fillna(0.0)
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Minutes since open, its sin/cos encoding, and day of week."""
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        idx = pd.to_datetime(df["ts"])

    # Use numpy arrays to avoid Index.clip() compatibility issues in Python 3.14
    minutes_arr = (idx.hour * 60 + idx.minute - MARKET_OPEN_MINUTES).to_numpy()
    minutes_arr = np.clip(minutes_arr, 0, None).astype(float)
    phase = minutes_arr / SESSION_TOTAL_MINUTES * 2 * np.pi
    df["min_since_open"] = minutes_arr
    df["min_sin"] = np.sin(phase)
    df["min_cos"] = np.cos(phase)
    df["day_of_week"] = idx.dayofweek.to_numpy().astype(int)
    return df


def _add_market_context(
    df: pd.DataFrame, nifty_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge Nifty index context features onto the stock DataFrame.

    Features:
      - nifty_ret_lag1, nifty_ret_lag3, nifty_ret_lag6
      - nifty_vol_12  (rolling std of nifty 1-bar returns, 12 bars)
      - rel_strength_1 (stock ret_lag1 - nifty_ret_lag1)
    """
    if nifty_df is None or nifty_df.empty:
        for col in ["nifty_ret_lag1", "nifty_ret_lag3", "nifty_ret_lag6",
                    "nifty_vol_12"]:
            df[col] = 0.0
        df["rel_strength_1"] = df["ret_lag1"] if "ret_lag1" in df.columns else 0.0
        return df

    nret = _log_return(nifty_df["close"])
    nifty_feats = pd.DataFrame(index=nifty_df.index)
    nifty_feats["nifty_ret_lag1"] = nret
    nifty_feats["nifty_ret_lag3"] = nret.shift(2)
    nifty_feats["nifty_ret_lag6"] = nret.shift(5)
    nifty_feats["nifty_vol_12"] = nret.rolling(12, min_periods=3).std()

    # Align by timestamp
    df = df.join(nifty_feats[["nifty_ret_lag1", "nifty_ret_lag3",
                               "nifty_ret_lag6", "nifty_vol_12"]],
                 how="left")

    # Relative strength = stock 1-bar return - nifty 1-bar return
    if "ret_lag1" in df.columns:
        df["rel_strength_1"] = df["ret_lag1"] - df["nifty_ret_lag1"]
    else:
        df["rel_strength_1"] = np.nan

    return df


# ── Optional sentiment features ───────────────────────────────────────────────

def _add_sentiment_features(
    df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge pre-aggregated sentiment features.

    ``sentiment_df`` must have index = datetime and columns
    [sent_24h_mean, sent_count_24h].  Only added when
    USE_SENTIMENT_FEATURES is True.
    """
    if not USE_SENTIMENT_FEATURES or sentiment_df is None or sentiment_df.empty:
        df["sent_24h_mean"] = np.nan
        df["sent_count_24h"] = np.nan
        return df
    df = df.join(sentiment_df[["sent_24h_mean", "sent_count_24h"]], how="left")
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    nifty_df: Optional[pd.DataFrame] = None,
    sentiment_df: Optional[pd.DataFrame] = None,
    drop_nans: bool = True,
) -> pd.DataFrame:
    """
    Compute all features for a single ticker's OHLCV DataFrame.

    Parameters
    ----------
    df          : OHLCV DataFrame with DatetimeIndex in IST.  Must have columns
                  [open, high, low, close, volume].  Sorted ascending by time.
    nifty_df    : OHLCV DataFrame for ^NSEI, same index convention.
    sentiment_df: Optional per-bar sentiment (only used if USE_SENTIMENT_FEATURES).
    drop_nans   : If True, drop rows with any NaN in feature columns.

    Returns
    -------
    DataFrame with feature columns appended (original OHLCV columns kept).
    """
    df = df.copy().sort_index()

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=False)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize(TZ)

    df = _add_returns(df)
    df = _add_volatility(df)
    df = _add_rsi(df)
    df = _add_macd(df)
    df = _add_ema_ratios(df)
    df = _add_vwap(df)
    df = _add_volume_ratio(df)
    df = _add_range(df)
    df = _add_day_open_distance(df)
    df = _add_day_range_position(df)
    df = _add_overnight_gap(df)
    df = _add_time_features(df)
    df = _add_market_context(df, nifty_df)
    df = _add_sentiment_features(df, sentiment_df)

    if drop_nans:
        feature_cols = get_feature_columns()
        # Only drop where feature cols exist
        present = [c for c in feature_cols if c in df.columns]
        df = df.dropna(subset=present)

    return df


def get_feature_columns() -> list[str]:
    """
    Return the canonical ordered list of feature column names.

    This list must stay in sync with build_features() output and is saved
    to metadata.json at training time.
    """
    cols = []
    # 1. Returns
    for lag in RETURN_LAGS:
        cols.append(f"ret_lag{lag}")
    # 2. Volatility
    for w in VOL_WINDOWS:
        cols.append(f"vol_{w}")
    # 3. RSI
    cols.append("rsi")
    # 4. MACD
    cols.append("macd_hist_norm")
    # 5. EMA ratios
    cols += ["ema_ratio_close_9", "ema_ratio_9_21"]
    # 6. VWAP
    cols.append("vwap_dev")
    # 7. Volume ratio
    cols.append("vol_ratio")
    # 8. Range
    cols.append("bar_range")
    # 9. Day-open distance
    cols.append("day_open_dist")
    # 10. Day range position
    cols.append("day_range_pos")
    # 11. Overnight gap
    cols.append("overnight_gap")
    # 12. Time
    cols += ["min_since_open", "min_sin", "min_cos", "day_of_week"]
    # 13. Market context
    cols += ["nifty_ret_lag1", "nifty_ret_lag3", "nifty_ret_lag6",
             "nifty_vol_12", "rel_strength_1"]
    # Optional sentiment
    if USE_SENTIMENT_FEATURES:
        cols += ["sent_24h_mean", "sent_count_24h"]
    return cols
