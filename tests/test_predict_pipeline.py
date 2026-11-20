"""
test_predict_pipeline.py – Verify the prediction pipeline and replay safety.

Required checks (spec §14):
3. predict_for returns expected keys, probs sum to ~1, feature column order
   matches metadata.json.
4. Data accessed at cursor t never includes timestamps after t (replay safety).
"""

import sys
import os
import json
import pickle
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
import pytz

from config import TZ, CLASS_NAMES, CLASS_MAP, HORIZON_BARS

IST = pytz.timezone(TZ)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv_for_predict(n: int = 150, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with IST DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-15 09:15", periods=n, freq="5min", tz=IST)
    close = 500 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close + rng.uniform(0, 3, n),
            "low": close - rng.uniform(0, 3, n),
            "close": close,
            "volume": rng.integers(50_000, 500_000, n).astype(float),
        },
        index=dates,
    )
    return df


@pytest.fixture(scope="module")
def tmp_model_dir(tmp_path_factory):
    """Create a temporary model + metadata for the pipeline test."""
    from src.features import build_features, get_feature_columns
    from lightgbm import LGBMClassifier

    d = tmp_path_factory.mktemp("models")
    ohlcv = _make_ohlcv_for_predict(n=200)
    nifty = _make_ohlcv_for_predict(n=200, seed=99)
    feat = build_features(ohlcv.copy(), nifty_df=nifty, drop_nans=True)
    feature_cols = get_feature_columns()

    # Minimal synthetic labels (random, just for testing the pipeline)
    rng = np.random.default_rng(0)
    n = len(feat)
    y = rng.integers(0, 3, n)
    if n >= 3:
        y[:3] = [0, 1, 2]

    # Train a tiny model
    clf = LGBMClassifier(n_estimators=5, random_state=0, verbose=-1, n_jobs=1)
    present_cols = [c for c in feature_cols if c in feat.columns]
    X = feat[present_cols].values
    clf.fit(X, y)

    model_path = str(d / "lgbm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    metadata = {
        "feature_columns": feature_cols,
        "class_names": CLASS_NAMES,
        "class_map": CLASS_MAP,
        "best_iteration": 5,
        "config": {"HORIZON_BARS": HORIZON_BARS},
    }
    meta_path = str(d / "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    return str(d), model_path, meta_path, feature_cols


# ── Test: predict_for return structure ───────────────────────────────────────

def test_predict_for_keys_and_probs(tmp_model_dir):
    """
    predict_for must return the expected keys and probabilities summing to ~1.
    """
    model_dir, model_path, meta_path, feature_cols = tmp_model_dir

    ohlcv = _make_ohlcv_for_predict(n=200)
    nifty = _make_ohlcv_for_predict(n=200, seed=99)

    # Patch the model/metadata paths and the data fetch functions
    import src.predict as predict_module

    predict_module._MODEL_CACHE.clear()

    with (
        patch("src.predict.MODEL_PATH", model_path),
        patch("src.predict.METADATA_PATH", meta_path),
        patch("src.predict._fetch_live", side_effect=lambda ticker, **kw: ohlcv if ticker != "^NSEI" else nifty),
        patch("src.predict._fetch_replay", side_effect=lambda ticker, as_of, **kw: ohlcv if ticker != "^NSEI" else nifty),
    ):
        result = predict_module.predict_for("ITC.NS", mode="live", include_news=False)

    # Required keys
    required_keys = {
        "ticker", "as_of", "horizon_minutes", "market_open",
        "probs", "label", "confidence", "features_used",
    }
    missing = required_keys - set(result.keys())
    assert not missing, f"Missing keys: {missing}"

    # Probs must sum to ~1
    probs = result["probs"]
    assert set(probs.keys()) == {"UP", "DOWN", "NEUTRAL"}
    total = sum(probs.values())
    assert abs(total - 1.0) < 0.01, f"Probs sum to {total}, expected ~1"

    # Label must be one of the valid classes
    assert result["label"] in {"UP", "DOWN", "NEUTRAL"}

    # Confidence must be in [0, 1]
    assert 0.0 <= result["confidence"] <= 1.0

    # Feature columns must match metadata
    assert result["features_used"] == feature_cols


def test_predict_for_feature_column_order(tmp_model_dir):
    """
    The feature column order used for prediction must exactly match
    the order saved in metadata.json.
    """
    model_dir, model_path, meta_path, feature_cols = tmp_model_dir

    with open(meta_path) as f:
        meta = json.load(f)
    expected_cols = meta["feature_columns"]

    ohlcv = _make_ohlcv_for_predict(n=200)
    nifty = _make_ohlcv_for_predict(n=200, seed=99)

    import src.predict as predict_module
    predict_module._MODEL_CACHE.clear()

    with (
        patch("src.predict.MODEL_PATH", model_path),
        patch("src.predict.METADATA_PATH", meta_path),
        patch("src.predict._fetch_live", side_effect=lambda ticker, **kw: ohlcv if ticker != "^NSEI" else nifty),
    ):
        result = predict_module.predict_for("RELIANCE.NS", mode="live", include_news=False)

    assert result.get("features_used") == expected_cols, (
        "Feature column order mismatch between prediction and metadata.json"
    )


# ── Test: Replay no-lookahead ─────────────────────────────────────────────────

def _make_replay_db(tmp_path, n: int = 150):
    """Create a temporary SQLite DB with synthetic price data."""
    from src.db import init_db, upsert_prices

    db_path = str(tmp_path / "test_market.db")
    init_db(db_path=db_path)

    dates = pd.date_range("2024-01-15 09:15", periods=n, freq="5min", tz=IST)
    rng = np.random.default_rng(1)
    close = 400 + np.cumsum(rng.normal(0, 1, n))

    df = pd.DataFrame(
        {
            "ticker": "TESTSTOCK.NS",
            "ts": dates.strftime("%Y-%m-%d %H:%M:%S%z"),
            "open": close * 0.999,
            "high": close * 1.001,
            "low": close * 0.998,
            "close": close,
            "volume": np.ones(n) * 10_000,
        }
    )
    upsert_prices(df, db_path=db_path)
    return db_path, dates


@pytest.mark.parametrize("cursor_bar", [30, 60, 90, 120])
def test_replay_no_future_data(tmp_path, cursor_bar):
    """
    Data accessed at cursor position t must never include timestamps after t.
    This is the replay no-lookahead test from the spec.
    """
    db_path, dates = _make_replay_db(tmp_path)
    cursor_ts = dates[cursor_bar]

    from src.replay import ReplayCursor

    # Build cursor using the test DB
    with patch("src.replay.query_prices") as mock_qp:
        # Return real data from our test DB
        from src.db import query_prices as real_qp

        def _patched_qp(ticker, start=None, end=None, db_path=db_path):
            return real_qp(ticker, start=start, end=end, db_path=db_path)

        mock_qp.side_effect = _patched_qp

        cursor = ReplayCursor(["TESTSTOCK.NS"], n_days=5)
        cursor.jump_to(cursor_ts)

    # Now get prices up to cursor
    with patch("src.replay.query_prices") as mock_qp:
        from src.db import query_prices as real_qp

        def _patched_qp2(ticker, start=None, end=None, db_path=db_path):
            return real_qp(ticker, start=start, end=end, db_path=db_path)

        mock_qp.side_effect = _patched_qp2
        df = cursor.get_prices_up_to_cursor("TESTSTOCK.NS", as_of=cursor_ts)

    if df.empty:
        pytest.skip("No data at this cursor position")

    # All timestamps must be ≤ cursor_ts
    future_rows = df[df.index > cursor_ts]
    assert future_rows.empty, (
        f"Replay lookahead! Found {len(future_rows)} rows after cursor {cursor_ts}:\n"
        f"{future_rows.index.tolist()}"
    )


def test_replay_resolution_timing():
    """
    A prediction made at cursor t should resolve at cursor t + HORIZON_BARS bars.
    Verify the horizon_resolve_ts calculation.
    """
    n = 50
    dates = pd.date_range("2024-01-15 09:15", periods=n, freq="5min", tz=IST)

    with patch("src.replay.query_prices") as mock_qp:
        rng = np.random.default_rng(2)
        close = 300 + np.cumsum(rng.normal(0, 1, n))
        df = pd.DataFrame(
            {
                "ticker": "X.NS",
                "ts": dates.strftime("%Y-%m-%d %H:%M:%S%z"),
                "open": close, "high": close, "low": close,
                "close": close, "volume": np.ones(n),
            }
        )
        mock_qp.return_value = df

        from src.replay import ReplayCursor
        cursor = ReplayCursor(["X.NS"])

    cursor._cursor_idx = 10
    resolve_ts = cursor.horizon_resolve_ts()
    expected_idx = 10 + HORIZON_BARS

    if expected_idx < n:
        assert resolve_ts == cursor._timeline[expected_idx], (
            f"resolve_ts {resolve_ts} != expected {cursor._timeline[expected_idx]}"
        )


def test_news_sentiment_modifier_negative():
    """
    If technical model predicts UP, but strong negative news breaks (e.g. CEO fired,
    sentiment = -0.80), the sentiment modifier should suppress UP and boost DOWN.
    """
    from src.predict import _apply_news_sentiment_modifier

    raw_probs = {"UP": 0.55, "NEUTRAL": 0.25, "DOWN": 0.20}
    adj_probs, summary, applied = _apply_news_sentiment_modifier(
        raw_probs, mean_sentiment=-0.80, headline_count=4, news_weight=0.40
    )
    assert applied is True
    assert adj_probs["DOWN"] > raw_probs["DOWN"], "DOWN prob must increase on negative news"
    assert adj_probs["UP"] < raw_probs["UP"], "UP prob must decrease on negative news"
    assert abs(sum(adj_probs.values()) - 1.0) < 1e-5


def test_news_sentiment_modifier_positive():
    """
    Positive news should increase UP probability and decrease DOWN probability.
    """
    from src.predict import _apply_news_sentiment_modifier

    raw_probs = {"UP": 0.25, "NEUTRAL": 0.25, "DOWN": 0.50}
    adj_probs, summary, applied = _apply_news_sentiment_modifier(
        raw_probs, mean_sentiment=0.75, headline_count=3, news_weight=0.35
    )
    assert applied is True
    assert adj_probs["UP"] > raw_probs["UP"], "UP prob must increase on positive news"
    assert adj_probs["DOWN"] < raw_probs["DOWN"], "DOWN prob must decrease on positive news"
    assert abs(sum(adj_probs.values()) - 1.0) < 1e-5


def test_news_sentiment_modifier_neutral_or_none():
    """
    Neutral or absent news must leave probabilities unchanged.
    """
    from src.predict import _apply_news_sentiment_modifier

    raw_probs = {"UP": 0.40, "NEUTRAL": 0.35, "DOWN": 0.25}
    adj_none, _, applied_none = _apply_news_sentiment_modifier(
        raw_probs, mean_sentiment=None, headline_count=0
    )
    assert applied_none is False
    assert adj_none == raw_probs

    adj_neu, _, applied_neu = _apply_news_sentiment_modifier(
        raw_probs, mean_sentiment=0.01, headline_count=2
    )
    assert applied_neu is False
    assert adj_neu == raw_probs
