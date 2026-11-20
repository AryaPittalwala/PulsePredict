"""
predict.py – Single prediction function used by both live and replay modes.

Both modes call ``predict_for(ticker, as_of=...)`` — the only difference is
how the price data is sourced (yfinance live vs. SQLite truncated at as_of).
Replay mode **cannot** access future data because it reads from a snapshot
of the DB at as_of.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, ".")

from config import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    HORIZON_BARS,
    INDEX_TICKER,
    INTERVAL,
    LIVE_NEWS_SENTIMENT_ENABLED,
    MARKET_CLOSE,
    MARKET_OPEN,
    METADATA_PATH,
    MODEL_PATH,
    NEWS_SENTIMENT_WEIGHT,
    NEWS_SENTIMENT_WINDOW_HOURS,
    TZ,
)

logger = logging.getLogger(__name__)

IST = pytz.timezone(TZ)

# ── Lazy model cache ──────────────────────────────────────────────────────────
_MODEL_CACHE: dict = {}


def _load_model():
    """Load model and metadata (cached)."""
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["metadata"]

    import pickle

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["metadata"] = metadata
    logger.info("Model loaded (best_iter=%s).", metadata.get("best_iteration"))
    return model, metadata


# ── Market hours check ────────────────────────────────────────────────────────

def _is_market_open(as_of: Optional[datetime] = None) -> bool:
    """Return True if the market is currently open (IST, weekdays only)."""
    now = as_of or datetime.now(IST)
    if now.tzinfo is None:
        now = IST.localize(now)
    elif now.tzinfo != IST:
        now = now.astimezone(IST)

    if now.weekday() >= 5:  # Saturday / Sunday
        return False

    open_t = datetime.strptime(MARKET_OPEN, "%H:%M").time()
    close_t = datetime.strptime(MARKET_CLOSE, "%H:%M").time()
    return open_t <= now.time() <= close_t


# ── Live data fetch ───────────────────────────────────────────────────────────

def _fetch_live(ticker: str, nifty: bool = False) -> pd.DataFrame:
    """
    Fetch the last 3 trading days of 5-minute bars from yfinance.

    Returns a DatetimeIndex (IST) OHLCV DataFrame.
    """
    import yfinance as yf

    sym = ticker
    try:
        raw = yf.download(sym, period="5d", interval=INTERVAL,
                          auto_adjust=True, progress=False)
    except Exception as e:
        logger.warning("yfinance download error for %s: %s", sym, e)
        raw = pd.DataFrame()

    if raw.empty:
        from src.db import query_prices
        db_df = query_prices(ticker)
        if not db_df.empty:
            db_df["ts"] = pd.to_datetime(db_df["ts"], utc=False)
            db_df = db_df.set_index("ts").sort_index()
            if db_df.index.tzinfo is None:
                db_df.index = db_df.index.tz_localize(TZ)
            return db_df.tail(200)[["open", "high", "low", "close", "volume"]]
        return pd.DataFrame()

    # Normalise timezone
    if raw.index.tzinfo is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index = raw.index.tz_convert(TZ)

    # Flatten MultiIndex columns from modern yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        if "Price" in raw.columns.names:
            raw.columns = raw.columns.get_level_values("Price")
        elif ticker in raw.columns.get_level_values(0):
            raw = raw[ticker]
        elif ticker in raw.columns.get_level_values(1):
            raw = raw.xs(ticker, level=1, axis=1)
        else:
            raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]

    # Filter market hours
    from config import MARKET_OPEN as MO, MARKET_CLOSE as MC
    open_t = datetime.strptime(MO, "%H:%M").time()
    close_t = datetime.strptime(MC, "%H:%M").time()
    mask = (raw.index.time >= open_t) & (raw.index.time <= close_t)
    raw = raw[mask]

    return raw[["open", "high", "low", "close", "volume"]]


# ── Replay data fetch ─────────────────────────────────────────────────────────

def _fetch_replay(
    ticker: str, as_of: datetime, lookback_days: int = 3
) -> pd.DataFrame:
    """
    Fetch price bars from SQLite strictly before ``as_of``.

    Uses only completed bars (bar whose open time ≤ as_of - 5 min).
    """
    from src.db import query_prices

    # as_of must be IST-aware
    if as_of.tzinfo is None:
        as_of = IST.localize(as_of)
    else:
        as_of = as_of.astimezone(IST)

    start_dt = as_of - timedelta(days=lookback_days + 2)
    # Strictly less than as_of to avoid the forming bar
    end_dt = as_of - timedelta(minutes=5)

    df = query_prices(
        ticker,
        start=start_dt.strftime("%Y-%m-%d %H:%M:%S%z"),
        end=end_dt.strftime("%Y-%m-%d %H:%M:%S%z"),
    )

    if df.empty:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"], utc=False)
    df = df.set_index("ts").sort_index()
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize(TZ)

    # Filter market hours
    from config import MARKET_OPEN as MO, MARKET_CLOSE as MC
    open_t = datetime.strptime(MO, "%H:%M").time()
    close_t = datetime.strptime(MC, "%H:%M").time()
    mask = (df.index.time >= open_t) & (df.index.time <= close_t)
    return df[mask][["open", "high", "low", "close", "volume"]]


def _drop_forming_bar(df: pd.DataFrame, as_of: Optional[datetime]) -> pd.DataFrame:
    """
    Drop the last (forming) bar if the current time is within that bar's window.

    In live mode, as_of = now.  In replay mode, as_of = cursor.
    """
    if df.empty or as_of is None:
        return df

    if as_of.tzinfo is None:
        as_of = IST.localize(as_of)
    else:
        as_of = as_of.astimezone(IST)

    last_bar_open = df.index[-1]
    # 5-minute bar: if current time < bar_open + 5 min, bar is still forming
    bar_close_time = last_bar_open + timedelta(minutes=5)
    if as_of < bar_close_time:
        df = df.iloc[:-1]
    return df


def _apply_news_sentiment_modifier(
    raw_probs: dict[str, float],
    mean_sentiment: Optional[float],
    headline_count: int,
    news_weight: float = NEWS_SENTIMENT_WEIGHT,
) -> tuple[dict[str, float], str, bool]:
    """
    Adjust model probabilities using live news sentiment.

    Returns: (adjusted_probs, shift_summary, sentiment_applied)
    """
    if mean_sentiment is None or headline_count <= 0 or news_weight <= 0:
        return raw_probs.copy(), "No active news signal; using pure technical prediction.", False

    s = float(mean_sentiment)
    if abs(s) < 0.05:
        return raw_probs.copy(), "News sentiment is neutral; no probability shift applied.", False

    w_scale = min(1.0, headline_count / 2.0) * news_weight
    p_up = raw_probs["UP"]
    p_down = raw_probs["DOWN"]
    p_neu = raw_probs["NEUTRAL"]

    if s < -0.05:
        # Negative news (e.g. CEO fired, regulatory sanction, profit warning)
        m = abs(s) * w_scale
        shift_from_up = m * p_up * 0.85
        shift_from_neu = m * p_neu * 0.40
        adj_up = max(0.01, p_up - shift_from_up)
        adj_neu = max(0.01, p_neu - shift_from_neu)
        adj_down = p_down + shift_from_up + shift_from_neu
        shift_summary = (
            f"Negative live news sentiment ({s:+.2f}, {headline_count} headlines) "
            "adjusted the prediction downward."
        )
    else:
        # Positive news (e.g. strong earnings, new deal, upgrade)
        m = s * w_scale
        shift_from_down = m * p_down * 0.85
        shift_from_neu = m * p_neu * 0.40
        adj_down = max(0.01, p_down - shift_from_down)
        adj_neu = max(0.01, p_neu - shift_from_neu)
        adj_up = p_up + shift_from_down + shift_from_neu
        shift_summary = (
            f"Positive live news sentiment ({s:+.2f}, {headline_count} headlines) "
            "adjusted the prediction upward."
        )

    total = adj_up + adj_down + adj_neu
    adj_probs = {
        "UP": float(adj_up / total),
        "DOWN": float(adj_down / total),
        "NEUTRAL": float(adj_neu / total),
    }
    return adj_probs, shift_summary, True


# ── Core prediction function ──────────────────────────────────────────────────

def predict_for(
    ticker: str,
    as_of: Optional[datetime] = None,
    mode: str = "live",
    include_news: bool = True,
    news_weight: float = NEWS_SENTIMENT_WEIGHT,
) -> dict:
    """
    Predict the direction of ``ticker`` over the next 15 minutes (3 × 5 min).

    Parameters
    ----------
    ticker : NSE ticker string (e.g. "ITC.NS")
    as_of  : Datetime for the prediction; None = now (live mode).
             In replay mode, pass the cursor timestamp.
    mode   : "live" or "replay"

    Returns
    -------
    dict with keys:
      ticker, as_of, horizon_minutes, market_open,
      probs (UP/DOWN/NEUTRAL), label, confidence, features_used
    """
    model, metadata = _load_model()
    feature_cols: list[str] = metadata["feature_columns"]

    is_open = _is_market_open(as_of)

    # ── Fetch price data ──────────────────────────────────────────────────────
    if mode == "replay" and as_of is not None:
        price_df = _fetch_replay(ticker, as_of)
        nifty_df = _fetch_replay(INDEX_TICKER, as_of)
    else:
        price_df = _fetch_live(ticker)
        nifty_df = _fetch_live(INDEX_TICKER, nifty=True)

    if price_df.empty:
        return {
            "ticker": ticker,
            "as_of": str(as_of or datetime.now(IST)),
            "horizon_minutes": HORIZON_BARS * 5,
            "market_open": is_open,
            "error": "No price data available.",
            "probs": {c: 1 / 3 for c in ["UP", "DOWN", "NEUTRAL"]},
            "label": "NEUTRAL",
            "confidence": 1 / 3,
            "features_used": [],
        }

    # Drop the forming bar
    effective_as_of = as_of or datetime.now(IST)
    price_df = _drop_forming_bar(price_df, effective_as_of)
    nifty_df = _drop_forming_bar(nifty_df, effective_as_of) if not nifty_df.empty else nifty_df

    if price_df.empty:
        return {
            "ticker": ticker,
            "as_of": str(effective_as_of),
            "horizon_minutes": HORIZON_BARS * 5,
            "market_open": is_open,
            "error": "Only a forming bar available — cannot predict yet.",
            "probs": {c: 1 / 3 for c in ["UP", "DOWN", "NEUTRAL"]},
            "label": "NEUTRAL",
            "confidence": 1 / 3,
            "features_used": [],
        }

    # ── Feature engineering ───────────────────────────────────────────────────
    from src.features import build_features

    feat_df = build_features(price_df, nifty_df=nifty_df if not nifty_df.empty else None, drop_nans=True)

    if feat_df.empty:
        return {
            "ticker": ticker,
            "as_of": str(effective_as_of),
            "horizon_minutes": HORIZON_BARS * 5,
            "market_open": is_open,
            "error": "Feature engineering returned no rows (warm-up period).",
            "probs": {c: 1 / 3 for c in ["UP", "DOWN", "NEUTRAL"]},
            "label": "NEUTRAL",
            "confidence": 1 / 3,
            "features_used": feature_cols,
        }

    # Ensure all required feature columns are present (fill missing with NaN)
    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = np.nan

    last_row = feat_df.iloc[[-1]][feature_cols]

    # ── Predict ───────────────────────────────────────────────────────────────
    proba = model.predict_proba(last_row.values)[0]
    pred_int = int(np.argmax(proba))
    label = CLASS_NAMES[pred_int]  # CLASS_NAMES = [DOWN, NEUTRAL, UP]
    confidence = float(proba[pred_int])

    raw_probs = {
        "UP": float(proba[2]),
        "DOWN": float(proba[0]),
        "NEUTRAL": float(proba[1]),
    }

    # ── Live news sentiment integration ───────────────────────────────────────
    news_agg = {"mean_sentiment": None, "headline_count": 0, "latest_headlines": []}
    if include_news and LIVE_NEWS_SENTIMENT_ENABLED:
        try:
            from src.news import fetch_headlines
            from src.sentiment import aggregate_sentiment, score_headlines

            if mode == "live":
                fetch_headlines(ticker, sleep_sec=0.2)
                score_headlines(batch_size=16)

            news_agg = aggregate_sentiment(
                ticker,
                window_hours=NEWS_SENTIMENT_WINDOW_HOURS,
                as_of=effective_as_of if mode == "replay" else None,
            )
        except Exception as e:
            logger.debug("News sentiment not available for %s: %s", ticker, e)

    adj_probs, shift_summary, sentiment_applied = _apply_news_sentiment_modifier(
        raw_probs,
        news_agg.get("mean_sentiment"),
        news_agg.get("headline_count", 0),
        news_weight=news_weight,
    )

    if sentiment_applied:
        # Determine final label and confidence from adjusted probabilities
        # CLASS_NAMES = [DOWN, NEUTRAL, UP]
        prob_vec = [adj_probs["DOWN"], adj_probs["NEUTRAL"], adj_probs["UP"]]
        final_idx = int(np.argmax(prob_vec))
        final_label = CLASS_NAMES[final_idx]
        final_conf = float(prob_vec[final_idx])
    else:
        final_label = label
        final_conf = confidence

    return {
        "ticker": ticker,
        "as_of": str(effective_as_of),
        "horizon_minutes": HORIZON_BARS * 5,
        "market_open": is_open,
        "probs": adj_probs,
        "raw_probs": raw_probs,
        "label": final_label,
        "raw_label": label,
        "confidence": final_conf,
        "raw_confidence": confidence,
        "features_used": feature_cols,
        "mode": mode,
        "news_sentiment": {
            "mean_sentiment": news_agg.get("mean_sentiment"),
            "headline_count": news_agg.get("headline_count", 0),
            "sentiment_applied": sentiment_applied,
            "shift_summary": shift_summary,
            "latest_headlines": news_agg.get("latest_headlines", [])[:5],
        },
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", default="ITC.NS", nargs="?")
    parser.add_argument("--mode", default="live", choices=["live", "replay"])
    args = parser.parse_args()

    result = predict_for(args.ticker, mode=args.mode)
    import pprint
    pprint.pprint(result)
