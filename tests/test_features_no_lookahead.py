"""
test_features_no_lookahead.py – Verify that every feature is strictly as-of-time.

For each feature, compute it on the full DataFrame, then recompute on the same
data truncated at row t.  The feature value at row t must be identical.
This test must pass for every feature column.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
import pytz

from config import TZ
from src.features import build_features, get_feature_columns


IST = pytz.timezone(TZ)


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic OHLCV DataFrame with DatetimeIndex in IST.
    Two simulated trading days of 5-minute bars are generated.
    """
    rng = np.random.default_rng(seed)
    # Timestamps: two trading days, ~75 bars each (09:15 – 15:30)
    dates = pd.date_range("2024-01-15 09:15", periods=n, freq="5min", tz=IST)
    # Simple random walk
    close = 500 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 5, n)
    low = close - rng.uniform(0, 5, n)
    open_ = close + rng.normal(0, 1, n)
    volume = rng.integers(10_000, 1_000_000, n).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    return df


@pytest.fixture(scope="module")
def full_ohlcv():
    return _make_ohlcv(n=200)


@pytest.fixture(scope="module")
def full_features(full_ohlcv):
    nifty = _make_ohlcv(n=200, seed=99)  # synthetic Nifty
    return build_features(full_ohlcv.copy(), nifty_df=nifty, drop_nans=False)


def test_feature_columns_present(full_features):
    """All canonical feature columns must be present in the output."""
    expected = get_feature_columns()
    missing = [c for c in expected if c not in full_features.columns]
    assert not missing, f"Missing feature columns: {missing}"


@pytest.mark.parametrize("check_row", [60, 80, 100, 120, 150])
def test_no_lookahead_all_features(full_ohlcv, check_row):
    """
    Feature values at row `check_row` on the full DataFrame must equal
    feature values at the last row of the truncated DataFrame ([:check_row+1]).

    This is the core no-lookahead test required by the spec.
    """
    nifty_full = _make_ohlcv(n=200, seed=99)

    # Features on full dataset
    feat_full = build_features(full_ohlcv.copy(), nifty_df=nifty_full.copy(), drop_nans=False)
    if check_row >= len(feat_full):
        pytest.skip("check_row exceeds dataset length")

    full_row = feat_full.iloc[check_row]
    full_ts = feat_full.index[check_row]

    # Features on truncated dataset (up to and including check_row)
    ohlcv_trunc = full_ohlcv.iloc[: check_row + 1].copy()
    nifty_trunc = nifty_full.iloc[: check_row + 1].copy()
    feat_trunc = build_features(ohlcv_trunc, nifty_df=nifty_trunc, drop_nans=False)

    if feat_trunc.empty:
        pytest.skip("Truncated feature set is empty (warm-up period)")

    trunc_last = feat_trunc.iloc[-1]

    feature_cols = get_feature_columns()
    present = [c for c in feature_cols if c in feat_full.columns and c in feat_trunc.columns]

    discrepancies = []
    for col in present:
        v_full = full_row[col]
        v_trunc = trunc_last[col]

        if pd.isna(v_full) and pd.isna(v_trunc):
            continue
        if pd.isna(v_full) or pd.isna(v_trunc):
            # One is NaN and the other isn't — flag only if the full value is non-NaN
            # (truncated can be NaN due to warm-up, that's fine)
            if not pd.isna(v_full) and pd.isna(v_trunc):
                discrepancies.append(
                    f"{col}: full={v_full}, trunc=NaN (warm-up is acceptable)"
                )
            continue
        if not np.isclose(v_full, v_trunc, rtol=1e-5, atol=1e-10, equal_nan=True):
            discrepancies.append(f"{col}: full={v_full:.8f}, trunc={v_trunc:.8f}")

    assert not discrepancies, (
        f"Lookahead detected at row {check_row}:\n" + "\n".join(discrepancies)
    )
