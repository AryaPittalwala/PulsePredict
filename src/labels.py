"""
labels.py – Forward-return labels with a volatility-adaptive neutral band.

Label rules (from spec §7)
--------------------------
- fwd_ret[t] = close[t + HORIZON_BARS] / close[t] - 1
- Rows where t + HORIZON_BARS falls on a different trading day are dropped
  (no cross-day labels).
- thr[t] = NEUTRAL_K * vol[t] * sqrt(HORIZON_BARS)
  where vol[t] is the rolling std of 1-bar log returns over VOL_WINDOW bars.
- UP   if fwd_ret > thr
  DOWN if fwd_ret < -thr
  else NEUTRAL
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import (
    CLASS_DOWN,
    CLASS_MAP,
    CLASS_NAMES,
    CLASS_NEUTRAL,
    CLASS_UP,
    HORIZON_BARS,
    NEUTRAL_K,
    TZ,
    VOL_WINDOW,
)

logger = logging.getLogger(__name__)


def _same_day(ts_a: pd.Timestamp, ts_b: pd.Timestamp) -> bool:
    """Return True if two IST timestamps fall on the same calendar date."""
    return ts_a.date() == ts_b.date()


def make_labels(df: pd.DataFrame, log_distribution: bool = True) -> pd.DataFrame:
    """
    Compute forward-return labels for a single-ticker OHLCV DataFrame.

    Parameters
    ----------
    df                 : DataFrame with DatetimeIndex (IST) and a 'close' column.
                         Must be sorted ascending by time.
    log_distribution   : If True, log the resulting class distribution.

    Returns
    -------
    df with extra columns:
      - 'vol_for_label' : rolling std of 1-bar log returns (VOL_WINDOW bars)
      - 'fwd_ret'       : forward return over HORIZON_BARS
      - 'threshold'     : adaptive neutral band threshold
      - 'label'         : UP / DOWN / NEUTRAL
      - 'label_int'     : integer encoding (DOWN=0, NEUTRAL=1, UP=2)

    Rows that cross day boundaries or have NaN forward returns are dropped.
    """
    df = df.copy().sort_index()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize(TZ)

    close = df["close"]

    # ── Volatility estimate (past data only) ──────────────────────────────────
    ret1 = np.log(close / close.shift(1))
    df["vol_for_label"] = ret1.rolling(VOL_WINDOW, min_periods=max(2, VOL_WINDOW // 4)).std()

    # ── Forward return ─────────────────────────────────────────────────────────
    fwd_close = close.shift(-HORIZON_BARS)
    df["fwd_ret"] = fwd_close / close - 1

    # ── Cross-day mask ─────────────────────────────────────────────────────────
    # Build an array of target timestamps (t + HORIZON_BARS bars ahead)
    n = len(df)
    dates = df.index.date
    target_dates = np.full(n, None, dtype=object)
    for i in range(n - HORIZON_BARS):
        target_dates[i] = df.index[i + HORIZON_BARS].date()

    same_day_mask = pd.array(
        [
            (target_dates[i] is not None and target_dates[i] == dates[i])
            for i in range(n)
        ],
        dtype=bool,
    )
    df["_same_day"] = same_day_mask

    # ── Threshold ─────────────────────────────────────────────────────────────
    df["threshold"] = NEUTRAL_K * df["vol_for_label"] * np.sqrt(HORIZON_BARS)

    # ── Classification ────────────────────────────────────────────────────────
    def _classify(row: pd.Series) -> str:
        if pd.isna(row["fwd_ret"]) or pd.isna(row["threshold"]):
            return np.nan
        if row["fwd_ret"] > row["threshold"]:
            return CLASS_UP
        if row["fwd_ret"] < -row["threshold"]:
            return CLASS_DOWN
        return CLASS_NEUTRAL

    df["label"] = df.apply(_classify, axis=1)
    df["label_int"] = df["label"].map(CLASS_MAP)

    # ── Drop cross-day and NaN rows ───────────────────────────────────────────
    before = len(df)
    df = df[df["_same_day"] & df["label"].notna()].drop(columns=["_same_day"])
    after = len(df)
    logger.debug("Labels: dropped %d rows (cross-day or NaN), kept %d.", before - after, after)

    # ── Log distribution ──────────────────────────────────────────────────────
    if log_distribution and len(df) > 0:
        dist = df["label"].value_counts(normalize=True)
        logger.info(
            "Label distribution (n=%d): UP=%.1f%% DOWN=%.1f%% NEUTRAL=%.1f%%",
            len(df),
            dist.get(CLASS_UP, 0) * 100,
            dist.get(CLASS_DOWN, 0) * 100,
            dist.get(CLASS_NEUTRAL, 0) * 100,
        )

    return df


def add_labels_to_features(
    feat_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge labels (computed on the OHLCV data) into the feature matrix.

    Parameters
    ----------
    feat_df  : Feature DataFrame (output of features.build_features).
    ohlcv_df : Raw OHLCV DataFrame for the same ticker (must include 'close').

    Returns
    -------
    Feature DataFrame with 'label', 'label_int', 'fwd_ret', 'threshold' columns
    added.  Rows without a valid label are dropped.
    """
    labelled = make_labels(ohlcv_df, log_distribution=False)
    label_cols = ["label", "label_int", "fwd_ret", "threshold", "vol_for_label"]
    label_df = labelled[label_cols]

    merged = feat_df.join(label_df, how="inner")
    merged = merged.dropna(subset=["label", "label_int"])
    return merged


def compute_class_weights(labels: pd.Series) -> dict[int, float]:
    """
    Compute inverse-frequency class weights for LightGBM.

    Returns a dict {class_int: weight}.
    """
    counts = labels.value_counts()
    total = len(labels)
    weights = {}
    for name, idx in CLASS_MAP.items():
        count = counts.get(name, 1)
        weights[idx] = total / (len(CLASS_MAP) * count)
    return weights
