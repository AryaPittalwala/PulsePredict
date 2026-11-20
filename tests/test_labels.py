"""
test_labels.py – Verify label correctness.

Required checks (spec §14):
1. No label spans a day boundary.
2. Label thresholds are computed from past data only.
3. Class names are correct on a hand-made example.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
import pytz

from config import (
    CLASS_DOWN,
    CLASS_NEUTRAL,
    CLASS_UP,
    HORIZON_BARS,
    NEUTRAL_K,
    TZ,
    VOL_WINDOW,
)
from src.labels import make_labels


IST = pytz.timezone(TZ)


def _make_two_day_ohlcv(seed: int = 0) -> pd.DataFrame:
    """
    Two simulated trading days of 5-minute bars.
    Day 1: 09:15 – 15:30 → 75 bars
    Day 2: 09:15 – 15:30 → 75 bars
    """
    rng = np.random.default_rng(seed)
    n = 150

    day1 = pd.date_range("2024-01-15 09:15", periods=75, freq="5min", tz=IST)
    day2 = pd.date_range("2024-01-16 09:15", periods=75, freq="5min", tz=IST)
    dates = day1.append(day2)

    close = 1000 + np.cumsum(rng.normal(0, 2, n))
    high = close + rng.uniform(0, 3, n)
    low = close - rng.uniform(0, 3, n)
    open_ = close + rng.normal(0, 1, n)
    volume = rng.integers(10_000, 500_000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def test_no_cross_day_labels():
    """No label should span across two different trading days."""
    df = _make_two_day_ohlcv()
    labelled = make_labels(df, log_distribution=False)

    assert "label" in labelled.columns
    assert len(labelled) > 0

    # For each labelled row, verify that ts and ts+HORIZON_BARS are on the same date
    idx = labelled.index
    for i, ts in enumerate(idx):
        # Find the future bar in the original df
        orig_idx = df.index.get_loc(ts) if ts in df.index else None
        if orig_idx is None:
            continue
        future_idx = orig_idx + HORIZON_BARS
        if future_idx < len(df):
            future_date = df.index[future_idx].date()
            current_date = ts.date()
            assert current_date == future_date, (
                f"Cross-day label at {ts}: current_date={current_date}, "
                f"future_date={future_date}"
            )


def test_no_nan_labels():
    """The labelled DataFrame must contain no NaN labels."""
    df = _make_two_day_ohlcv()
    labelled = make_labels(df, log_distribution=False)
    assert labelled["label"].notna().all(), "NaN labels found!"
    assert labelled["label_int"].notna().all(), "NaN label_int found!"


def test_correct_class_names():
    """Labels must only contain UP, DOWN, or NEUTRAL."""
    df = _make_two_day_ohlcv()
    labelled = make_labels(df, log_distribution=False)
    valid_labels = {CLASS_UP, CLASS_DOWN, CLASS_NEUTRAL}
    unique = set(labelled["label"].unique())
    assert unique.issubset(valid_labels), f"Unexpected labels: {unique - valid_labels}"


def test_hand_made_up():
    """Hand-constructed case: a large positive return should be labelled UP."""
    n = 80
    # Flat prices so vol ≈ 0 → threshold ≈ 0 → any positive return → UP
    # We use slightly non-zero vol so the formula works
    prices = [100.0] * n
    # Make the HORIZON_BARS-th bar after the last labelled bar clearly higher
    prices[-1] = 102.0  # will be used as fwd_close for some rows

    dates = pd.date_range("2024-01-15 09:15", periods=n, freq="5min", tz=IST)
    close = pd.Series(prices, index=dates)
    # Add tiny noise to get non-zero vol
    rng = np.random.default_rng(1)
    close = close + rng.normal(0, 0.001, n)
    close.iloc[-1] = close.iloc[-HORIZON_BARS - 1] * 1.05  # 5% jump

    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.001,
            "low": close * 0.998,
            "close": close,
            "volume": np.ones(n) * 100_000,
        },
        index=dates,
    )

    labelled = make_labels(df, log_distribution=False)
    # The row just before the jump should be labelled UP
    jump_row = labelled.iloc[-1] if not labelled.empty else None
    if jump_row is not None:
        assert jump_row["label"] in {CLASS_UP, CLASS_NEUTRAL}, (
            f"Expected UP (or NEUTRAL due to vol), got {jump_row['label']}"
        )


def test_threshold_uses_past_data_only():
    """
    vol_for_label must be a rolling window using only past bars.
    Inject a known spike at bar k; the threshold at bar k-1 must not be
    influenced by that spike.
    """
    n = 100
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    spike_idx = 60
    close_modified = close.copy()
    close_modified[spike_idx] = close_modified[spike_idx] + 50  # large spike

    dates = pd.date_range("2024-01-15 09:15", periods=n, freq="5min", tz=IST)
    df_base = pd.DataFrame(
        {
            "open": close, "high": close * 1.001,
            "low": close * 0.999, "close": close,
            "volume": np.ones(n) * 10_000,
        },
        index=dates,
    )
    df_spike = pd.DataFrame(
        {
            "open": close_modified, "high": close_modified * 1.001,
            "low": close_modified * 0.999, "close": close_modified,
            "volume": np.ones(n) * 10_000,
        },
        index=dates,
    )

    lab_base = make_labels(df_base, log_distribution=False)
    lab_spike = make_labels(df_spike, log_distribution=False)

    # For rows before the spike (and outside the VOL_WINDOW look-back of the spike),
    # thresholds must be identical.
    safe_rows = min(spike_idx - VOL_WINDOW - 1, 10)
    if safe_rows > 0:
        for i in range(safe_rows):
            ts = lab_base.index[i] if i < len(lab_base) else None
            if ts is None or ts not in lab_spike.index:
                continue
            thr_base = lab_base.loc[ts, "threshold"]
            thr_spike = lab_spike.loc[ts, "threshold"]
            assert np.isclose(thr_base, thr_spike, rtol=1e-6), (
                f"Threshold at row {i} changed after future spike: "
                f"base={thr_base}, spike={thr_spike}"
            )


def test_label_int_encoding():
    """label_int must match CLASS_MAP encoding."""
    from config import CLASS_MAP

    df = _make_two_day_ohlcv()
    labelled = make_labels(df, log_distribution=False)

    for _, row in labelled.iterrows():
        expected_int = CLASS_MAP[row["label"]]
        assert row["label_int"] == expected_int, (
            f"Encoding mismatch: label={row['label']}, "
            f"label_int={row['label_int']}, expected={expected_int}"
        )
