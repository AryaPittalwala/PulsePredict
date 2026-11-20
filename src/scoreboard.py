"""
scoreboard.py – Log predictions, resolve outcomes, and compute accuracy.

Predictions are logged to the ``predictions`` SQLite table.  A resolver job
(called on each app refresh) fills in actual_ret, actual_label, and correct
once the horizon has elapsed and data exists.

Live and replay modes are stored separately and scored separately.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from config import (
    CLASS_DOWN,
    CLASS_MAP,
    CLASS_NEUTRAL,
    CLASS_UP,
    HORIZON_BARS,
    NEUTRAL_K,
    TZ,
    VOL_WINDOW,
)
from src.db import (
    get_unresolved_predictions,
    insert_prediction,
    query_predictions,
    query_prices,
    resolve_prediction,
)

logger = logging.getLogger(__name__)

IST = pytz.timezone(TZ)


def log_prediction(result: dict) -> int:
    """
    Persist a predict_for() result to the predictions table.

    Returns the new row id.
    """
    row = {
        "ticker": result["ticker"],
        "made_at": result.get("as_of", str(datetime.now(IST))),
        "horizon_bars": HORIZON_BARS,
        "pred_label": result["label"],
        "p_up": result["probs"]["UP"],
        "p_down": result["probs"]["DOWN"],
        "p_neutral": result["probs"]["NEUTRAL"],
        "confidence": result["confidence"],
        "mode": result.get("mode", "live"),
    }
    return insert_prediction(row)


def _classify_return(fwd_ret: float, vol: float) -> str:
    """Apply the same volatility-adaptive threshold used in training."""
    thr = NEUTRAL_K * vol * np.sqrt(HORIZON_BARS)
    if fwd_ret > thr:
        return CLASS_UP
    if fwd_ret < -thr:
        return CLASS_DOWN
    return CLASS_NEUTRAL


def _get_actual_return(
    ticker: str, made_at: str, horizon_bars: int
) -> Optional[tuple[float, str]]:
    """
    Look up the actual forward return for a prediction.

    Returns (fwd_ret, label_str) or None if data is not yet available.
    """
    try:
        made_dt = pd.to_datetime(made_at, utc=False)
        if made_dt.tzinfo is None:
            made_dt = IST.localize(made_dt)
        else:
            made_dt = made_dt.astimezone(IST)
    except Exception as e:
        logger.warning("Cannot parse made_at=%s: %s", made_at, e)
        return None

    resolve_dt = made_dt + timedelta(minutes=5 * horizon_bars)
    if resolve_dt > datetime.now(IST):
        return None  # horizon not yet elapsed

    # Fetch price bars around the prediction time
    start = (made_dt - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S%z")
    end = (resolve_dt + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S%z")
    df = query_prices(ticker, start=start, end=end)

    if df.empty:
        return None

    df["ts"] = pd.to_datetime(df["ts"], utc=False)
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(TZ)
    df = df.set_index("ts").sort_index()

    # Find entry close (closest bar at or just before made_at)
    entry_bars = df[df.index <= made_dt]
    if entry_bars.empty:
        return None
    entry_close = float(entry_bars.iloc[-1]["close"])

    # Find exit close (closest bar at or just after resolve_dt)
    exit_bars = df[df.index >= resolve_dt - timedelta(minutes=5)]
    if exit_bars.empty:
        return None
    exit_close = float(exit_bars.iloc[0]["close"])

    fwd_ret = exit_close / entry_close - 1

    # Estimate vol from recent bars (VOL_WINDOW bars before made_at)
    vol_bars = entry_bars.iloc[-VOL_WINDOW:]
    if len(vol_bars) >= 2:
        log_rets = np.log(vol_bars["close"] / vol_bars["close"].shift(1)).dropna()
        vol = float(log_rets.std()) if len(log_rets) > 0 else 0.01
    else:
        vol = 0.01

    actual_label = _classify_return(fwd_ret, vol)
    return fwd_ret, actual_label


def resolve_pending_predictions(mode: Optional[str] = None) -> int:
    """
    Try to resolve all unresolved predictions.

    Returns the number of predictions successfully resolved.
    """
    df = get_unresolved_predictions(mode=mode)
    if df.empty:
        return 0

    resolved_count = 0
    for _, row in df.iterrows():
        result = _get_actual_return(
            row["ticker"], row["made_at"], int(row["horizon_bars"])
        )
        if result is None:
            continue

        fwd_ret, actual_label = result
        correct = int(actual_label == row["pred_label"])
        resolve_prediction(
            pred_id=int(row["id"]),
            actual_ret=fwd_ret,
            actual_label=actual_label,
            correct=correct,
        )
        resolved_count += 1

    if resolved_count:
        logger.info("Resolved %d predictions.", resolved_count)
    return resolved_count


def get_scoreboard(mode: Optional[str] = None, limit: int = 50) -> dict:
    """
    Compute scoreboard statistics.

    Returns
    -------
    dict with keys:
      - overall_accuracy     : float or None
      - confident_accuracy   : float or None
      - total_predictions    : int
      - resolved_predictions : int
      - recent              : list of recent resolved predictions
    """
    from config import CONFIDENCE_THRESHOLD

    df = query_predictions(mode=mode, limit=10000)
    if df.empty:
        return {
            "overall_accuracy": None,
            "confident_accuracy": None,
            "total_predictions": 0,
            "resolved_predictions": 0,
            "recent": [],
        }

    resolved = df[df["resolved"] == 1].copy()

    overall_acc = None
    if not resolved.empty:
        overall_acc = float((resolved["correct"] == 1).mean())

    confident = resolved[resolved["confidence"] >= CONFIDENCE_THRESHOLD]
    confident_acc = None
    if not confident.empty:
        confident_acc = float((confident["correct"] == 1).mean())

    recent = (
        resolved.sort_values("made_at", ascending=False)
        .head(limit)[
            ["ticker", "made_at", "pred_label", "confidence",
             "actual_label", "correct", "actual_ret"]
        ]
        .to_dict("records")
    )

    return {
        "overall_accuracy": overall_acc,
        "confident_accuracy": confident_acc,
        "total_predictions": len(df),
        "resolved_predictions": len(resolved),
        "recent": recent,
        "mode": mode or "all",
    }
